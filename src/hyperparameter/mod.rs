//! Hyperparameter adaptation mechanisms
//!
//! This module provides various approaches to hyperparameter control in evolutionary algorithms,
//! following Eiben et al.'s classification:
//!
//! 1. **Deterministic Control (Schedules)**: Parameters change according to a predetermined schedule
//! 2. **Adaptive Control**: Parameters adapt based on feedback from the search process
//! 3. **Self-Adaptive Control**: Parameters are encoded in the genome and evolve
//! 4. **Bayesian Learning**: Parameters are inferred using probabilistic methods

pub mod schedules;
pub mod adaptive;
pub mod self_adaptive;
pub mod bayesian;

pub mod prelude {
    pub use super::schedules::*;
    pub use super::adaptive::*;
    pub use super::self_adaptive::*;
    pub use super::bayesian::*;
}
