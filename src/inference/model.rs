//! The Boltzmann posterior over genomes, as a fugue program
//!
//! Fix a prior program `p(x)` (a [`GenomePrior`]) and an observation program
//! `p(data | x)` (a [`GenomeLikelihood`] — `observe` statements, latent
//! nuisance parameters, and/or black-box `factor`s). The target
//!
//! ```text
//!     π_β(x) ∝ p(x) · p(data | x)^β
//! ```
//!
//! *is literally a fugue model*: `prior.model().bind(|g|
//! likelihood.model(&g, β).map(|_| g))`. Every density this layer needs —
//! prior mass, tempered joint, MH acceptance — is obtained by running or
//! replaying that program; there is no hand-written density code.
//!
//! For the classical black-box case, `EvolutionModel::new(prior, fitness)`
//! wraps a scalar [`Fitness`] in [`FactorFitness`] (`factor(β·f(x))` — the
//! Gibbs / generalized-Bayes posterior); `EvolutionModel::from_likelihood`
//! accepts any observation program.
//!
//! Two builders exist because MH wants a fixed-β target while tempered SMC
//! must receive the β = 1 program (fugue's `adaptive_smc` supplies β by
//! tempering `log_likelihood + log_factors`; baking β in as well would
//! double-count it).

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use fugue::{factor, Model, ModelExt, Trace};
use rand::Rng;

use super::likelihood::{FactorFitness, GenomeLikelihood};
use super::prior::GenomePrior;
use crate::fitness::traits::Fitness;

/// A probabilistic model of an evolutionary population: a genome prior
/// program, an observation program (likelihood), and an inverse temperature
/// `β`.
#[derive(Clone)]
pub struct EvolutionModel<P, L>
where
    P: GenomePrior,
    L: GenomeLikelihood<P::Genome>,
{
    prior: P,
    likelihood: L,
    beta: f64,
}

impl<P, F> EvolutionModel<P, FactorFitness<F>>
where
    P: GenomePrior,
    F: Fitness<Genome = P::Genome, Value = f64> + Clone + Send + Sync + 'static,
{
    /// Create a model with a black-box scalar fitness entering as
    /// `factor(β·f(x))` (the classical Gibbs-posterior mode), at `β = 1`.
    pub fn new(prior: P, fitness: F) -> Self {
        Self {
            prior,
            likelihood: FactorFitness::new(fitness),
            beta: 1.0,
        }
    }

    /// Raw fitness `f(x)` (higher is better).
    pub fn fitness_value(&self, genome: &P::Genome) -> f64 {
        self.likelihood.fitness.evaluate(genome)
    }

    /// The tempered fitness log-factor `β · f(x)`.
    pub fn log_weight(&self, genome: &P::Genome) -> f64 {
        self.beta * self.fitness_value(genome)
    }

    /// Create a fugue trace whose choices equal the genome's encoding under
    /// this model's prior and whose `total_log_weight()` equals `β · f(x)` —
    /// the "fitness as likelihood" weighted-trace contract (EV-52): a genuine
    /// `factor(β·f)` model run through
    /// [`TraceScoringHandler`](super::effect_handlers::TraceScoringHandler),
    /// so the mass lands in `log_factors`.
    pub fn to_weighted_trace(&self, genome: &P::Genome) -> Trace {
        let logw = self.log_weight(genome);
        let base = self.prior.trace_of(genome);
        let (_r, trace) = run(
            super::effect_handlers::TraceScoringHandler::new(base),
            factor(logw),
        );
        trace
    }
}

impl<P, L> EvolutionModel<P, L>
where
    P: GenomePrior,
    L: GenomeLikelihood<P::Genome>,
{
    /// Create a model from an arbitrary observation program (`observe`
    /// statements, latent nuisance parameters, factors), at `β = 1`.
    pub fn from_likelihood(prior: P, likelihood: L) -> Self {
        Self {
            prior,
            likelihood,
            beta: 1.0,
        }
    }

    /// Set the inverse temperature `β` directly (`β ≥ 0`).
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.max(0.0);
        self
    }

    /// Set the temperature `T`; equivalent to `β = 1/T`.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };
        self
    }

    /// Current inverse temperature `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Current temperature `T = 1/β`.
    pub fn temperature(&self) -> f64 {
        1.0 / self.beta
    }

    /// The prior program.
    pub fn prior(&self) -> &P {
        &self.prior
    }

    /// The observation program.
    pub fn likelihood(&self) -> &L {
        &self.likelihood
    }

    /// The fixed-β target as a program:
    /// `log π_β(x) = log p(x) + β·log p(data|x)`. This is the model MH runs
    /// against.
    pub fn target_model(&self) -> impl Fn() -> Model<P::Genome> + Clone + '_ {
        let prior = self.prior.clone();
        let likelihood = self.likelihood.clone();
        let beta = self.beta;
        move || {
            let likelihood = likelihood.clone();
            prior
                .model()
                .bind(move |g| likelihood.model(&g, beta).map(move |_| g))
        }
    }

    /// The **untempered** (`β = 1`) joint program for tempered SMC: fugue's
    /// `adaptive_smc` supplies β by tempering
    /// `log_likelihood + log_factors`, applying it exactly once.
    pub fn smc_model(&self) -> impl Fn() -> Model<P::Genome> + Clone + '_ {
        let prior = self.prior.clone();
        let likelihood = self.likelihood.clone();
        move || {
            let likelihood = likelihood.clone();
            prior
                .model()
                .bind(move |g| likelihood.model(&g, 1.0).map(move |_| g))
        }
    }

    /// Draw a genome from the prior `p(x)` by running the prior program.
    pub fn sample_prior<R: Rng>(&self, rng: &mut R) -> P::Genome {
        let (g, _) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            self.prior.model(),
        );
        g
    }

    /// Score a genome under the fixed-β target by replaying its encoding
    /// **under this model's prior** ([`GenomePrior::trace_of`]) through the
    /// target program — so this works for every prior, including generative
    /// grammars over trees.
    ///
    /// The returned trace satisfies `log π_β(g) = trace.total_log_weight()`
    /// and `log p(g) = trace.log_prior`. A genome outside the prior's support
    /// scores `log_prior = −∞`.
    ///
    /// Note: if the likelihood contains latent nuisance sites, they are not
    /// part of `trace_of(g)` and would abort a strict replay — score via the
    /// SMC/MH drivers in that case (which sample them), or marginalize them
    /// externally.
    pub fn score(&self, genome: &P::Genome) -> (P::Genome, Trace) {
        run(
            ScoreGivenTrace {
                base: self.prior.trace_of(genome),
                trace: Trace::default(),
            },
            (self.target_model())(),
        )
    }

    /// Unnormalised log target `log π_β(x) = log p(x) + β·log p(data|x)`.
    pub fn log_boltzmann_target(&self, genome: &P::Genome) -> f64 {
        self.score(genome).1.total_log_weight()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::genome::bounds::MultiBounds;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use crate::inference::prior::{GaussianPrior, UniformBoxPrior};

    /// A `Clone`-able fitness wrapping a function pointer.
    #[derive(Clone, Copy)]
    pub(crate) struct PtrFitness(pub(crate) fn(&RealVector) -> f64);

    impl Fitness for PtrFitness {
        type Genome = RealVector;
        type Value = f64;
        fn evaluate(&self, genome: &RealVector) -> f64 {
            (self.0)(genome)
        }
    }

    pub(crate) fn quad_origin(g: &RealVector) -> f64 {
        -0.5 * g.genes().iter().map(|x| x * x).sum::<f64>()
    }

    #[test]
    fn test_to_weighted_trace_carries_fitness_mass() {
        // regression: EV-52 — total_log_weight() must equal β·f(x), not 0.
        let prior = UniformBoxPrior::new(MultiBounds::symmetric(5.0, 2));
        let model = EvolutionModel::new(prior, PtrFitness(quad_origin)).with_beta(2.0);
        let genome = RealVector::new(vec![1.0, 2.0]);
        let f = model.fitness_value(&genome); // -0.5*(1+4) = -2.5
        let trace = model.to_weighted_trace(&genome);
        assert!((trace.total_log_weight() - 2.0 * f).abs() < 1e-9);
        assert!((trace.log_factors - 2.0 * f).abs() < 1e-9);
        assert!(trace.total_log_weight().abs() > 1e-6);
    }

    #[test]
    fn test_score_composes_prior_and_factor() {
        // log π_β = log p + β·f, with each part in its own accumulator.
        let prior = GaussianPrior::new(0.0, 2.0, 2);
        let model = EvolutionModel::new(prior, PtrFitness(quad_origin)).with_beta(1.5);
        let g = RealVector::new(vec![0.5, -1.0]);
        let (decoded, scored) = model.score(&g);
        assert_eq!(decoded.genes(), g.genes());
        assert!((scored.log_factors - 1.5 * quad_origin(&g)).abs() < 1e-12);
        assert!(scored.log_prior.is_finite());
        assert!(
            (scored.total_log_weight() - (scored.log_prior + scored.log_factors)).abs() < 1e-12
        );
    }

    #[test]
    fn test_out_of_bounds_scores_neg_inf() {
        let prior = UniformBoxPrior::new(MultiBounds::symmetric(1.0, 1));
        let model = EvolutionModel::new(prior, PtrFitness(quad_origin));
        let g = RealVector::new(vec![3.0]);
        assert_eq!(model.log_boltzmann_target(&g), f64::NEG_INFINITY);
    }

    #[test]
    fn test_sample_prior_returns_decoded_genome() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let prior = GaussianPrior::new(0.0, 1.0, 4);
        let model = EvolutionModel::new(prior, PtrFitness(quad_origin));
        let mut rng = StdRng::seed_from_u64(3);
        let g = model.sample_prior(&mut rng);
        assert_eq!(g.genes().len(), 4);
    }

    /// An observation-program likelihood: per-datum `observe` statements land
    /// in `log_likelihood`, decomposed from the prior — structure the scalar
    /// factor could never expose.
    #[test]
    fn test_observation_likelihood_scores_in_log_likelihood() {
        use crate::inference::likelihood::{tempered_observe, GenomeLikelihood};
        use fugue::{addr, Normal};

        #[derive(Clone)]
        struct GaussianData {
            ys: Vec<f64>,
            sigma: f64,
        }
        impl GenomeLikelihood<RealVector> for GaussianData {
            fn model(&self, g: &RealVector, beta: f64) -> Model<()> {
                let mu = g.genes()[0];
                let sigma = self.sigma;
                let mut m = fugue::pure(());
                for (k, &y) in self.ys.iter().enumerate() {
                    m = m.and_then(move |_| {
                        tempered_observe(addr!("y", k), Normal::new(mu, sigma).unwrap(), y, beta)
                    });
                }
                m
            }
        }

        let prior = GaussianPrior::new(0.0, 2.0, 1);
        let data = GaussianData {
            ys: vec![0.4, 0.6, 0.5],
            sigma: 0.5,
        };
        let model = EvolutionModel::from_likelihood(prior, data.clone());
        let g = RealVector::new(vec![0.5]);
        let (_, scored) = model.score(&g);
        let normal = Normal::new(0.5, 0.5).unwrap();
        let analytic: f64 = data
            .ys
            .iter()
            .map(|y| fugue::Distribution::log_prob(&normal, y))
            .sum();
        assert!((scored.log_likelihood - analytic).abs() < 1e-12);
        assert_eq!(scored.log_factors, 0.0);
        assert!(scored.log_prior.is_finite());
    }
}
