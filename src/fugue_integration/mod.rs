//! Fugue PPL integration: evolution as inference over the Boltzmann posterior
//!
//! This module makes the "evolution as Bayesian inference" story load-bearing
//! rather than ornamental. Given a fitness function `f` and a prior `p` over
//! genomes, it targets the **Boltzmann / Gibbs posterior**
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x))
//! ```
//!
//! with real Fugue machinery:
//!
//! - [`evolution_model::EvolutionModel`] samples the prior by running a genuine
//!   `fugue::Model` under a `PriorHandler`, and injects fitness as a likelihood
//!   by running `factor(β·f(x))` through the [`effect_handlers::TraceScoringHandler`]
//!   so the trace's `total_log_weight()` equals `β·f(x)`.
//! - [`evolution_model::EvolutionStep`] is a Metropolis–Hastings kernel invariant
//!   for `π_β` (the prior's support — e.g. bounds — is honoured in the
//!   acceptance ratio).
//! - [`evolution_model::EvolutionarySMC`] is a valid tempered Sequential Monte
//!   Carlo sampler over a `β` ladder from `0` (prior) to `1` (posterior), using
//!   incremental importance weights, ESS-triggered resampling, and MH
//!   mutation/crossover rejuvenation.
//! - [`bayesian_ga::BayesianAdaptiveGA`] adapts its mutation operators with
//!   genuine conjugate `Beta`/`Gamma` posteriors and Thompson sampling.
//!
//! # Effect handlers
//!
//! [`effect_handlers`] provides genuine `fugue::Handler` implementations
//! ([`effect_handlers::TraceScoringHandler`], [`effect_handlers::RecordingHandler`])
//! plus plain operation hooks for the trace-based operators. See that module for
//! the distinction.
//!
//! See `examples/bayesian_evolution.rs` for an end-to-end pipeline.

pub mod bayesian_ga;
pub mod effect_handlers;
pub mod evolution_model;
pub mod trace_operators;

pub mod prelude {
    pub use super::bayesian_ga::*;
    pub use super::effect_handlers::*;
    pub use super::evolution_model::*;
    pub use super::trace_operators::*;
}
