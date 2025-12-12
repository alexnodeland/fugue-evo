//! Genetic operators
//!
//! This module provides selection, crossover, and mutation operators.

pub mod crossover;
pub mod mutation;
pub mod selection;
pub mod traits;

pub mod prelude {
    pub use super::crossover::*;
    pub use super::mutation::*;
    pub use super::selection::*;
    pub use super::traits::*;
}
