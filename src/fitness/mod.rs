//! Fitness evaluation and benchmarks
//!
//! This module provides the fitness abstraction and benchmark functions.

pub mod benchmarks;
pub mod multi_objective;
pub mod traits;

pub mod prelude {
    pub use super::benchmarks::*;
    pub use super::multi_objective::{ClosureMultiObjective, MultiObjectiveFitness};
    pub use super::traits::*;
}
