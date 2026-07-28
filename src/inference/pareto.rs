//! Multi-objective optimization as Bayesian inference: the Pareto posterior
//!
//! Classic multi-objective EC (NSGA-II) has no scalar target, so it cannot be
//! a posterior sampler. The Bayesian counterpart puts the scalarization
//! weight **inside the model**: with objectives `f_1..f_k` (minimized, per
//! the [`MultiObjectiveFitness`] convention) and a uniform prior over the
//! weight simplex,
//!
//! ```text
//!     w ~ Uniform(simplex)          (stick-breaking Beta sites)
//!     π_β(x, w) ∝ p(x) · exp(−β · ⟨w, f(x)⟩)
//! ```
//!
//! the joint posterior's marginal over genomes traces out the **Pareto
//! front**: each weight vector `w` selects a scalarized optimum on the front,
//! and integrating over `w` spreads the population across it. Each particle's
//! trace carries its own `w` (at `pareto#v{i}` stick-breaking sites), telling
//! you *where on the front* that particle lives — a posterior over the front,
//! with the usual inference dividends (uncertainty, evidence), which NSGA-II
//! cannot express.
//!
//! Weighted-sum scalarization recovers the convex part of the front; for
//! non-convex fronts a Chebyshev scalarization would be needed (future work,
//! same architecture).

use fugue::{addr, factor, Beta, Model, ModelExt, Trace};

use super::likelihood::GenomeLikelihood;
use crate::fitness::multi_objective::MultiObjectiveFitness;

/// A scalarization likelihood with a latent weight vector: the Bayesian
/// multi-objective target. See the [module docs](self).
#[derive(Clone)]
pub struct ParetoScalarization<M> {
    /// The multi-objective fitness (objectives **minimized**).
    pub objectives: M,
    /// Sharpness of the scalarized likelihood, `exp(−sharpness·⟨w, f⟩)`.
    /// Larger values concentrate particles closer to the front. (This is a
    /// fixed model parameter; the SMC tempering β multiplies it on top.)
    pub sharpness: f64,
}

impl<M> ParetoScalarization<M> {
    /// Create a Pareto-posterior likelihood with the given sharpness.
    pub fn new(objectives: M, sharpness: f64) -> Self {
        Self {
            objectives,
            sharpness,
        }
    }
}

/// Build the uniform-simplex weight model via stick-breaking:
/// `v_i ~ Beta(1, k−1−i)` for `i = 0..k−1` (the last stick is deterministic
/// but sampled as `Beta(1, 1)`-degenerate skip), yielding `w` uniform on the
/// `k`-simplex. For `k = 2` this is a single `Beta(1,1)` site.
fn weight_model(k: usize) -> Model<Vec<f64>> {
    fn stick(i: usize, k: usize, acc: Vec<f64>, remaining: f64) -> Model<Vec<f64>> {
        if i == k - 1 {
            let mut acc = acc;
            acc.push(remaining);
            return fugue::pure(acc);
        }
        let b = (k - 1 - i) as f64;
        fugue::sample(
            addr!("pareto", format!("v{i}")),
            Beta::new(1.0, b).expect("valid stick-breaking Beta"),
        )
        .bind(move |v| {
            let mut acc = acc;
            let w = v * remaining;
            acc.push(w);
            stick(i + 1, k, acc, remaining - w)
        })
    }
    stick(0, k, Vec::with_capacity(k), 1.0)
}

impl<G, M> GenomeLikelihood<G> for ParetoScalarization<M>
where
    G: 'static,
    M: MultiObjectiveFitness<G> + Clone + Send + Sync + 'static,
{
    fn model(&self, genome: &G, beta: f64) -> Model<()> {
        let objs = self.objectives.evaluate(genome);
        let k = objs.len();
        let sharpness = self.sharpness;
        if k == 0 {
            return fugue::pure(());
        }
        weight_model(k).bind(move |w| {
            let scalarized: f64 = w.iter().zip(&objs).map(|(wi, fi)| wi * fi).sum();
            if scalarized.is_finite() {
                factor(-beta * sharpness * scalarized)
            } else {
                factor(f64::NEG_INFINITY)
            }
        })
    }
}

/// Read a particle's weight vector back off its trace (the stick-breaking
/// sites), i.e. *where on the front* the particle lives. Returns `None` when
/// the sites are absent (e.g. a prior-only trace).
pub fn particle_weights(trace: &Trace, num_objectives: usize) -> Option<Vec<f64>> {
    let k = num_objectives;
    let mut w = Vec::with_capacity(k);
    let mut remaining = 1.0;
    for i in 0..k - 1 {
        let v = trace.get_f64(&addr!("pareto", format!("v{i}")))?;
        let wi = v * remaining;
        w.push(wi);
        remaining -= wi;
    }
    w.push(remaining);
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::bounds::{Bounds, MultiBounds};
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use crate::inference::model::EvolutionModel;
    use crate::inference::prior::UniformBoxPrior;
    use crate::inference::smc::{CrossoverConfig, EvoSmcConfig, EvolutionSMC};
    use fugue::ResamplingMethod;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Analytic validation: 1-D biobjective `f1 = x²`, `f2 = (x−2)²`
    /// (minimized). The Pareto set is exactly `[0, 2]`, and the weighted-sum
    /// optimum for weight `w` on `f1` is `x*(w) = 2(1−w)`. The Pareto
    /// posterior must (a) concentrate on the Pareto set, (b) cover both ends
    /// of the front, and (c) place each particle near its own weight's
    /// scalarized optimum.
    #[test]
    fn test_pareto_posterior_traces_the_front() {
        #[derive(Clone)]
        struct BiObjective;
        impl MultiObjectiveFitness<RealVector> for BiObjective {
            fn num_objectives(&self) -> usize {
                2
            }
            fn evaluate(&self, g: &RealVector) -> Vec<f64> {
                let x = g.genes()[0];
                vec![x * x, (x - 2.0) * (x - 2.0)]
            }
        }
        let objectives = BiObjective;
        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-1.0, 3.0)]));
        let model =
            EvolutionModel::from_likelihood(prior, ParetoScalarization::new(objectives, 8.0));

        let mut rng = StdRng::seed_from_u64(1618);
        // Posterior, then anneal a little for front sharpness.
        let result = EvolutionSMC::anneal(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 500,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 5,
                crossover: Some(CrossoverConfig::default()),
            },
            8.0,
            8,
        );

        let model_fn = model.smc_model();
        let decoded = fugue::decode_particles(&result.particles, &model_fn);

        let mut on_set_mass = 0.0;
        let mut low_end = 0.0;
        let mut high_end = 0.0;
        let mut w_err_sum = 0.0;
        let mut w_err_n = 0.0;
        for (p, (g, w)) in result.particles.iter().zip(&decoded) {
            let x = g.genes()[0];
            if (-0.25..=2.25).contains(&x) {
                on_set_mass += w;
            }
            if x < 0.5 {
                low_end += w;
            }
            if x > 1.5 {
                high_end += w;
            }
            if let Some(wv) = particle_weights(&p.trace, 2) {
                let x_star = 2.0 * (1.0 - wv[0]);
                w_err_sum += w * (x - x_star).abs();
                w_err_n += w;
            }
        }
        assert!(
            on_set_mass > 0.9,
            "only {on_set_mass:.2} of posterior mass on the Pareto set [0,2]"
        );
        assert!(
            low_end > 0.08 && high_end > 0.08,
            "front ends not covered: low {low_end:.2}, high {high_end:.2}"
        );
        let mean_w_err = w_err_sum / w_err_n.max(1e-12);
        assert!(
            mean_w_err < 0.45,
            "particles should sit near their weight's scalarized optimum; mean |x − 2(1−w)| = {mean_w_err:.3}"
        );
    }

    /// Stick-breaking weights are a valid distribution over the simplex for
    /// k = 3: components positive, summing to 1, with symmetric means.
    #[test]
    fn test_stick_breaking_weights_uniform_simplex() {
        use fugue::runtime::handler::run;
        use fugue::runtime::interpreters::PriorHandler;
        let mut rng = StdRng::seed_from_u64(9);
        let mut sums = [0.0f64; 3];
        let n = 4000;
        for _ in 0..n {
            let (w, _) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                weight_model(3),
            );
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(w.iter().all(|&x| (0.0..=1.0).contains(&x)));
            for (s, wi) in sums.iter_mut().zip(&w) {
                *s += wi;
            }
        }
        for s in sums {
            let mean = s / n as f64;
            assert!(
                (mean - 1.0 / 3.0).abs() < 0.02,
                "uniform-simplex component mean {mean} should be 1/3"
            );
        }
    }
}
