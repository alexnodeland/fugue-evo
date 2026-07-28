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
//! the joint posterior spreads over front-adjacent configurations: each
//! weight vector `w` selects a scalarized optimum on the front, and each
//! particle's trace carries its own `w` (at `pareto#v{i}` stick-breaking
//! sites), telling you *where on the front* that particle lives — a posterior
//! over front positions, with the usual inference dividends (uncertainty,
//! evidence), which NSGA-II cannot express.
//!
//! **Marginal-tilt caveat (read this)**: in the latent-`w` model the
//! `w`-marginal is *not* uniform — it is tilted by `exp(−s·m(w))`, where
//! `m(w)` is the scalarized optimum's value at `w`, so weights whose optima
//! score better attract more mass, and high sharpness or heavy annealing
//! concentrates the population near the best-scoring front regions (often the
//! endpoints). The *conditional* `x | w` is what tracks the front. For
//! uniform front coverage, sweep **fixed** weights
//! ([`ChebyshevScalarization::with_weight`]) across a grid, or keep sharpness
//! moderate and read positions off `particle_weights`.
//!
//! [`ParetoScalarization`] uses weighted-sum scalarization, which recovers
//! the convex part of the front; [`ChebyshevScalarization`] uses the weighted
//! Chebyshev (weighted-max) norm, which reaches every (weakly)
//! Pareto-optimal point — including non-convex front regions where every
//! weighted-sum optimum collapses to the front's endpoints.

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

/// The Chebyshev (weighted-max) scalarization likelihood with a latent
/// weight vector:
///
/// ```text
///     w ~ Uniform(simplex)
///     π_β(x, w) ∝ p(x) · exp(−β · s · max_i  w_i · (f_i(x) − z_i))
/// ```
///
/// where `z` is the **ideal point** (a reference component-wise ≤ the
/// objective values of interest, e.g. per-objective minima or a slightly
/// optimistic estimate). Minimizing the weighted Chebyshev norm over `x`
/// reaches every weakly Pareto-optimal point as `w` varies over the simplex
/// (Miettinen 1999) — in particular the **non-convex** front regions where a
/// weighted sum's interior stationary point is a maximum and all its mass
/// collapses onto the front's endpoints. Use this when the front may be
/// non-convex; use [`ParetoScalarization`] when it is known convex (the
/// weighted sum is smoother).
#[derive(Clone)]
pub struct ChebyshevScalarization<M> {
    /// The multi-objective fitness (objectives **minimized**).
    pub objectives: M,
    /// Sharpness of the scalarized likelihood (see [`ParetoScalarization`]).
    pub sharpness: f64,
    /// The ideal/reference point `z` (one entry per objective).
    pub ideal: Vec<f64>,
    /// `None`: the weight is a latent site (subject to the marginal-tilt
    /// caveat in the [module docs](self)). `Some(w)`: a fixed weight — the
    /// posterior concentrates on that weight's own front point, which is the
    /// mode to use for sweeping the front uniformly.
    pub weight: Option<Vec<f64>>,
}

impl<M> ChebyshevScalarization<M> {
    /// Create a Chebyshev-scalarization likelihood with a **latent** weight.
    pub fn new(objectives: M, sharpness: f64, ideal: Vec<f64>) -> Self {
        Self {
            objectives,
            sharpness,
            ideal,
            weight: None,
        }
    }

    /// Fix the scalarization weight (front-sweeping mode): the posterior
    /// targets this weight's own scalarized optimum — reaching interior
    /// points of non-convex fronts that no weighted sum can select.
    pub fn with_weight(mut self, weight: Vec<f64>) -> Self {
        self.weight = Some(weight);
        self
    }
}

impl<G, M> GenomeLikelihood<G> for ChebyshevScalarization<M>
where
    G: 'static,
    M: MultiObjectiveFitness<G> + Clone + Send + Sync + 'static,
{
    fn model(&self, genome: &G, beta: f64) -> Model<()> {
        let objs = self.objectives.evaluate(genome);
        let k = objs.len();
        let sharpness = self.sharpness;
        let ideal = self.ideal.clone();
        if k == 0 {
            return fugue::pure(());
        }
        debug_assert_eq!(ideal.len(), k, "ideal point must match objective count");
        let cheby_factor = move |w: &[f64], objs: &[f64], ideal: &[f64]| -> Model<()> {
            let cheby = w
                .iter()
                .zip(objs.iter().zip(ideal))
                .map(|(wi, (fi, zi))| wi * (fi - zi))
                .fold(f64::NEG_INFINITY, f64::max);
            if cheby.is_finite() {
                factor(-beta * sharpness * cheby)
            } else {
                factor(f64::NEG_INFINITY)
            }
        };
        match self.weight.clone() {
            Some(w) => cheby_factor(&w, &objs, &ideal),
            None => weight_model(k).bind(move |w| cheby_factor(&w, &objs, &ideal)),
        }
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

    /// The non-convex-front contrast, at the theorem level. Objectives
    /// (minimized) on x ∈ [0,1]: `f1 = x`, `f2 = 1 − x²`. Every x in [0,1]
    /// is Pareto-optimal and the front `f2 = 1 − f1²` is CONCAVE, so for any
    /// FIXED weight the weighted-sum scalarization `w·x + (1−w)(1−x²)` has
    /// its interior stationary point as a MAXIMUM (second derivative
    /// −2(1−w) < 0): its minimizers are always the endpoints, and interior
    /// front points are unreachable. The Chebyshev scalarization's fixed-w
    /// optimum is the interior crossing point `w·x = (1−w)(1−x²)` — for
    /// w = 1/2, x* = (√5−1)/2 ≈ 0.618. We pin both facts.
    #[test]
    fn test_chebyshev_reaches_nonconvex_front_where_weighted_sum_cannot() {
        #[derive(Clone)]
        struct ConcaveFront;
        impl MultiObjectiveFitness<RealVector> for ConcaveFront {
            fn num_objectives(&self) -> usize {
                2
            }
            fn evaluate(&self, g: &RealVector) -> Vec<f64> {
                let x = g.genes()[0];
                vec![x, 1.0 - x * x]
            }
        }

        /// Test-local fixed-weight weighted-sum likelihood (the published
        /// ParetoScalarization is latent-w only).
        #[derive(Clone)]
        struct FixedWeightSum {
            w: f64,
            sharpness: f64,
        }
        impl GenomeLikelihood<RealVector> for FixedWeightSum {
            fn model(&self, g: &RealVector, beta: f64) -> Model<()> {
                let objs = ConcaveFront.evaluate(g);
                let s = self.w * objs[0] + (1.0 - self.w) * objs[1];
                factor(-beta * self.sharpness * s)
            }
        }

        let prior = || UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(0.0, 1.0)]));
        let cfg = || EvoSmcConfig {
            num_particles: 500,
            ess_threshold: 0.5,
            resampling: ResamplingMethod::Systematic,
            rejuvenation_steps: 5,
            crossover: Some(CrossoverConfig::default()),
        };

        let mut rng = StdRng::seed_from_u64(271828);

        // (a) Fixed w = 1/2, weighted sum: bimodal at the endpoints; the
        // interior is a scalarization MAXIMUM and must be avoided.
        let ws_model = EvolutionModel::from_likelihood(
            prior(),
            FixedWeightSum {
                w: 0.5,
                sharpness: 25.0,
            },
        );
        let ws = EvolutionSMC::anneal(&mut rng, &ws_model, cfg(), 8.0, 8);
        let ws_fn = ws_model.smc_model();
        let ws_interior: f64 = fugue::decode_particles(&ws.particles, &ws_fn)
            .iter()
            .filter(|(g, _)| (0.25..0.75).contains(&g.genes()[0]))
            .map(|(_, w)| w)
            .sum();
        assert!(
            ws_interior < 0.1,
            "fixed-w weighted sum put {ws_interior:.3} mass in the interior — impossible for a concave front"
        );

        // (b) Fixed w = 1/2, Chebyshev: concentrates on the interior front
        // point x* = (√5 − 1)/2 ≈ 0.618 — the point weighted-sum cannot reach.
        let x_star = (5.0f64.sqrt() - 1.0) / 2.0;
        let ch_model = EvolutionModel::from_likelihood(
            prior(),
            ChebyshevScalarization::new(ConcaveFront, 25.0, vec![0.0, 0.0])
                .with_weight(vec![0.5, 0.5]),
        );
        let ch = EvolutionSMC::anneal(&mut rng, &ch_model, cfg(), 8.0, 8);
        let mean = ch.weighted_mean(0);
        assert!(
            (mean - x_star).abs() < 0.08,
            "fixed-w Chebyshev posterior mean {mean:.3} should sit at the interior front point {x_star:.3}"
        );
        let ch_fn = ch_model.smc_model();
        let ch_interior: f64 = fugue::decode_particles(&ch.particles, &ch_fn)
            .iter()
            .filter(|(g, _)| (0.25..0.75).contains(&g.genes()[0]))
            .map(|(_, w)| w)
            .sum();
        assert!(
            ch_interior > 0.8,
            "fixed-w Chebyshev interior mass {ch_interior:.3} — must reach the non-convex front interior"
        );

        // (c) Sweeping fixed weights traces the whole front, ends included.
        for (w, lo, hi) in [(0.15, 0.75, 1.0), (0.5, 0.5, 0.75), (0.85, 0.1, 0.45)] {
            let m = EvolutionModel::from_likelihood(
                prior(),
                ChebyshevScalarization::new(ConcaveFront, 25.0, vec![0.0, 0.0])
                    .with_weight(vec![w, 1.0 - w]),
            );
            let r = EvolutionSMC::anneal(&mut rng, &m, cfg(), 8.0, 8);
            let mean = r.weighted_mean(0);
            assert!(
                (lo..=hi).contains(&mean),
                "weight {w}: front point {mean:.3} outside expected band [{lo}, {hi}]"
            );
        }
    }

    /// Latent-weight Chebyshev: the CONDITIONAL x | w tracks the front even
    /// though the w-marginal is tilted (module-docs caveat). Among particles
    /// whose latent weight is interior (w₀ ∈ [0.35, 0.65]), most mass must
    /// sit in the interior of the front — the region a weighted sum's
    /// conditional never occupies.
    #[test]
    fn test_chebyshev_latent_weight_conditional_tracks_front() {
        #[derive(Clone)]
        struct ConcaveFront;
        impl MultiObjectiveFitness<RealVector> for ConcaveFront {
            fn num_objectives(&self) -> usize {
                2
            }
            fn evaluate(&self, g: &RealVector) -> Vec<f64> {
                let x = g.genes()[0];
                vec![x, 1.0 - x * x]
            }
        }

        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(0.0, 1.0)]));
        let model = EvolutionModel::from_likelihood(
            prior,
            ChebyshevScalarization::new(ConcaveFront, 8.0, vec![0.0, 0.0]),
        );
        let mut rng = StdRng::seed_from_u64(314159);
        // β = 1 posterior only — annealing would concentrate the tilted
        // w-marginal onto the endpoints (see module docs).
        let result = EvolutionSMC::run(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 800,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 6,
                crossover: Some(CrossoverConfig::default()),
            },
        );
        let model_fn = model.smc_model();
        let decoded = fugue::decode_particles(&result.particles, &model_fn);

        let mut stratum_mass = 0.0;
        let mut stratum_interior = 0.0;
        for (p, (g, w)) in result.particles.iter().zip(&decoded) {
            if let Some(wv) = particle_weights(&p.trace, 2) {
                if (0.35..=0.65).contains(&wv[0]) {
                    stratum_mass += w;
                    let x = g.genes()[0];
                    if (0.25..0.75).contains(&x) {
                        stratum_interior += w;
                    }
                }
            }
        }
        assert!(
            stratum_mass > 0.02,
            "interior-weight stratum carries only {stratum_mass:.4} mass — too depleted to test"
        );
        let frac = stratum_interior / stratum_mass;
        assert!(
            frac > 0.5,
            "interior-weight particles put only {frac:.2} of their mass on the front interior"
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
