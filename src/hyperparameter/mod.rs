//! Hyperparameter adaptation mechanisms
//!
//! This module provides various approaches to hyperparameter control in evolutionary algorithms,
//! following Eiben et al.'s classification:
//!
//! 1. **Deterministic Control (Schedules)**: Parameters change according to a predetermined schedule
//! 2. **Adaptive Control**: Parameters adapt based on feedback from the search process
//! 3. **Self-Adaptive Control**: Parameters are encoded in the genome and evolve
//! 4. **Bayesian / Bandit Learning**: Parameters are learned online from observed
//!    improvement events. The [`bayesian::ThompsonSamplingTuner`] is wired into
//!    [`SimpleGA`](crate::algorithms::simple_ga::SimpleGA) via
//!    [`SimpleGABuilder::adaptive_operators`](crate::algorithms::simple_ga::SimpleGABuilder::adaptive_operators)
//!    and [`SimpleGA::run_adaptive`](crate::algorithms::simple_ga::SimpleGA::run_adaptive);
//!    the conjugate posteriors ([`bayesian::BetaPosterior`], [`bayesian::GammaPosterior`])
//!    and [`bayesian::RunningLogMoments`] are honest, self-describing building blocks.

pub mod adaptive;
pub mod bayesian;
pub mod schedules;
pub mod self_adaptive;

pub mod prelude {
    pub use super::adaptive::*;
    pub use super::bayesian::*;
    pub use super::schedules::*;
    pub use super::self_adaptive::*;
}
