//! Estimation of Distribution Algorithms (EDAs)
//!
//! This module provides algorithms that use probabilistic models to guide search:
//! - **UMDA**: Univariate Marginal Distribution Algorithm
//!
//! EDAs work by:
//! 1. Selecting promising individuals from the population
//! 2. Estimating a probability distribution from the selected individuals
//! 3. Sampling new individuals from the learned distribution
//! 4. Repeating until termination

pub mod umda;

pub use umda::*;
