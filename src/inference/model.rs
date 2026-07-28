//! The Boltzmann posterior over genomes, as a fugue program
//!
//! Fix a fitness `f: G → ℝ` (higher is better) and a prior program `p(x)`
//! (a [`GenomePrior`]). For inverse temperature `β ≥ 0` the **Boltzmann /
//! Gibbs posterior** is
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x)).
//! ```
//!
//! In this rewrite the target *is literally a fugue model*:
//! `prior.model().bind(|g| factor(β·f(g)).map(|_| g))`. Every density this
//! layer needs — prior mass, tempered joint, MH acceptance — is obtained by
//! running or replaying that program; there is no hand-written density code.
//!
//! Two builders exist because MH wants a fixed-β target while tempered SMC
//! must receive the *untempered* factor (fugue's `adaptive_smc` supplies β by
//! likelihood-tempering `log_likelihood + log_factors`; baking β in as well
//! would double-count it — the bug the old hand-rolled SMC had).

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use fugue::{factor, Model, ModelExt, Trace};
use rand::Rng;

use super::prior::GenomePrior;
use crate::fitness::traits::Fitness;

/// A probabilistic model of an evolutionary population: a genome prior
/// program, a fitness function entering as a `factor`, and an inverse
/// temperature `β`.
#[derive(Clone)]
pub struct EvolutionModel<P, F>
where
    P: GenomePrior,
    F: Fitness<Genome = P::Genome, Value = f64> + Clone + Send + Sync + 'static,
{
    prior: P,
    fitness: F,
    beta: f64,
}

impl<P, F> EvolutionModel<P, F>
where
    P: GenomePrior,
    F: Fitness<Genome = P::Genome, Value = f64> + Clone + Send + Sync + 'static,
{
    /// Create a new evolution model at `β = 1`.
    pub fn new(prior: P, fitness: F) -> Self {
        Self {
            prior,
            fitness,
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

    /// Raw fitness `f(x)` (higher is better).
    pub fn fitness_value(&self, genome: &P::Genome) -> f64 {
        self.fitness.evaluate(genome)
    }

    /// The tempered fitness log-factor `β · f(x)`.
    pub fn log_weight(&self, genome: &P::Genome) -> f64 {
        self.beta * self.fitness_value(genome)
    }

    /// The fixed-β Boltzmann target as a program:
    /// `log π_β(x) = log p(x) + β·f(x)`. This is the model MH runs against.
    pub fn target_model(&self) -> impl Fn() -> Model<P::Genome> + Clone + '_ {
        let prior = self.prior.clone();
        let fitness = self.fitness.clone();
        let beta = self.beta;
        move || {
            let fitness = fitness.clone();
            prior.model().bind(move |g| {
                let fit = fitness.evaluate(&g);
                factor(beta * fit).map(move |_| g)
            })
        }
    }

    /// The **untempered** target (`factor(f)`, β absent) for tempered SMC:
    /// fugue's `adaptive_smc` supplies β by tempering `log_factors`, applying
    /// it exactly once.
    pub fn smc_model(&self) -> impl Fn() -> Model<P::Genome> + Clone + '_ {
        let prior = self.prior.clone();
        let fitness = self.fitness.clone();
        move || {
            let fitness = fitness.clone();
            prior.model().bind(move |g| {
                let fit = fitness.evaluate(&g);
                factor(fit).map(move |_| g)
            })
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

    /// Score a genome under the fixed-β target by replaying its canonical
    /// trace through the target program.
    ///
    /// The returned trace satisfies `log π_β(g) = trace.total_log_weight()`,
    /// `log p(g) = trace.log_prior`, and `β·f(g) = trace.log_factors`. A
    /// genome outside the prior's support scores `log_prior = −∞` — the
    /// behavior the old hand-written `log_prior_density` match implemented by
    /// hand.
    pub fn score(&self, genome: &P::Genome) -> (P::Genome, Trace) {
        use crate::genome::trace_genome::TraceGenome;
        run(
            ScoreGivenTrace {
                base: genome.to_trace(),
                trace: Trace::default(),
            },
            (self.target_model())(),
        )
    }

    /// Unnormalised log target `log π_β(x) = log p(x) + β · f(x)`.
    pub fn log_boltzmann_target(&self, genome: &P::Genome) -> f64 {
        self.score(genome).1.total_log_weight()
    }

    /// Create a fugue trace whose choices equal `genome.to_trace()` and whose
    /// `total_log_weight()` equals `β · f(x)` — the "fitness as likelihood"
    /// weighted-trace contract (EV-52), preserved verbatim from the previous
    /// implementation: a genuine `factor(β·f)` model run through
    /// [`TraceScoringHandler`](super::effect_handlers::TraceScoringHandler),
    /// so the mass lands in `log_factors`.
    pub fn to_weighted_trace(&self, genome: &P::Genome) -> Trace {
        use crate::genome::trace_genome::TraceGenome;
        let logw = self.log_weight(genome);
        let base = genome.to_trace();
        let (_r, trace) = run(
            super::effect_handlers::TraceScoringHandler::new(base),
            factor(logw),
        );
        trace
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
}
