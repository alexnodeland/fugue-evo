//! Hyperparameter adaptation mechanisms
//!
//! This module provides various approaches to hyperparameter control in evolutionary algorithms,
//! following Eiben et al.'s classification.
//!
//! **Integration status (EV-21):** only two of these are actually wired into a
//! built-in algorithm today — self-adaptive control (Evolution Strategy) and the
//! Thompson bandit ([`SimpleGA::run_adaptive`](crate::algorithms::simple_ga::SimpleGA::run_adaptive)).
//! The deterministic **schedules** ([`schedules`]) and the feedback-driven
//! **adaptive-control** primitives ([`adaptive`]) are, as shipped, *unintegrated
//! building blocks*: no default algorithm's run loop consumes them, so you drive
//! them yourself (e.g. query [`ParameterSchedule::value_at`](schedules::ParameterSchedule::value_at)
//! each generation and apply the result). They are exercised only by unit tests.
//! This is the same honest-building-block framing used for the conjugate
//! posteriors below.
//!
//! 1. **Deterministic Control (Schedules)** — [`schedules`]: predetermined
//!    parameter values by generation ([`LinearAnnealing`](schedules::LinearAnnealing),
//!    [`CosineAnnealing`](schedules::CosineAnnealing),
//!    [`ExponentialDecay`](schedules::ExponentialDecay),
//!    [`PolynomialDecay`](schedules::PolynomialDecay),
//!    [`CyclicalSchedule`](schedules::CyclicalSchedule),
//!    [`CompositeSchedule`](schedules::CompositeSchedule),
//!    [`DynamicSchedule`](schedules::DynamicSchedule)). **Unintegrated building
//!    blocks** — not wired into any algorithm's run loop.
//! 2. **Adaptive Control** — [`adaptive`]: feedback-driven parameter adaptation
//!    ([`OneFifthRule`](adaptive::OneFifthRule),
//!    [`AdaptiveOperatorSelection`](adaptive::AdaptiveOperatorSelection),
//!    [`AdaptiveMutationRate`](adaptive::AdaptiveMutationRate),
//!    [`DiversityBasedAdaptation`](adaptive::DiversityBasedAdaptation)). **Also
//!    unintegrated building blocks** — no built-in algorithm consumes them; the
//!    wired online-adaptation path today is the Thompson bandit in item 4.
//! 3. **Self-Adaptive Control** — [`self_adaptive`]: parameters encoded in the
//!    genome and evolved; **integrated** into the Evolution Strategy path.
//! 4. **Bayesian / Bandit Learning**: Parameters are learned online from observed
//!    improvement events. The [`bayesian::ThompsonSamplingTuner`] is **wired into**
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
