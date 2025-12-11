//! Genetic operators
//!
//! This module provides selection, crossover, and mutation operators.

pub mod traits;
pub mod selection;
pub mod crossover;
pub mod mutation;

pub mod prelude {
    pub use super::traits::*;
    pub use super::selection::*;
    pub use super::crossover::*;
    pub use super::mutation::*;
}
