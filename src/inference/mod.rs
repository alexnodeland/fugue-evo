//! Evolution as inference: the PPL-native layer (requires the `ppl` feature)
//!
//! This module makes "evolutionary algorithms as probabilistic programs"
//! literal. Given a fitness `f` and a prior *program* `p(x)` over genomes
//! (a [`prior::GenomePrior`] — any fugue `Model<G>`), the Boltzmann/Gibbs
//! posterior
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x))
//! ```
//!
//! is itself a fugue program (`prior.model().bind(|g| factor(β·f(g)))`), and
//! every sampler here is fugue's own inference machinery run against it:
//!
//! - [`model::EvolutionModel`] assembles the target program; all densities are
//!   obtained by running/replaying it (no hand-written density code).
//! - [`mh::EvolutionChain`] delegates to `fugue::adaptive_single_site_mh` —
//!   typed proposals move every site kind (Bool/U64/Usize/I64/F64), fixing the
//!   historical F64-only dead-chain bug.
//! - [`smc::EvolutionSMC`] delegates to `fugue::adaptive_smc_with_kernel`:
//!   adaptive likelihood-tempering (β applied exactly once), ESS-driven
//!   resampling, per-particle MH rejuvenation, an optional population-coupled
//!   crossover kernel, and an unbiased log-evidence estimate.
//! - [`bayesian_ga::BayesianAdaptiveGA`] keeps the Thompson-sampling operator
//!   selection (conjugate Beta/Gamma posteriors, now backed by `rand_distr`).
//!
//! [`effect_handlers`] retains the genuine `fugue::Handler` implementations
//! (`TraceScoringHandler`, `RecordingHandler`) and the operator observation
//! hooks; [`trace_operators`] retains the value-level trace operators.
//!
//! See `examples/bayesian_evolution.rs` for an end-to-end pipeline.

pub mod bayesian_ga;
pub mod effect_handlers;
pub mod mh;
pub mod model;
pub mod prior;
pub mod smc;
pub mod trace_operators;

pub mod prelude {
    pub use super::bayesian_ga::*;
    pub use super::effect_handlers::*;
    pub use super::mh::*;
    pub use super::model::*;
    pub use super::prior::*;
    pub use super::smc::*;
    pub use super::trace_operators::*;
}
