//! Fitness evaluation and benchmarks
//!
//! This module provides the fitness abstraction and benchmark functions.

pub mod benchmarks;
pub mod traits;

pub mod prelude {
    pub use super::benchmarks::*;
    pub use super::traits::*;
}
