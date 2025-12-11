//! Fitness evaluation and benchmarks
//!
//! This module provides the fitness abstraction and benchmark functions.

pub mod traits;
pub mod benchmarks;

pub mod prelude {
    pub use super::traits::*;
    pub use super::benchmarks::*;
}
