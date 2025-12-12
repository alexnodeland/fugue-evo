//! Genome abstractions and implementations
//!
//! This module provides the core `EvolutionaryGenome` trait and built-in genome types.

pub mod bit_string;
pub mod bounds;
pub mod permutation;
pub mod real_vector;
pub mod traits;

pub mod prelude {
    pub use super::bit_string::*;
    pub use super::bounds::*;
    pub use super::permutation::*;
    pub use super::real_vector::*;
    pub use super::traits::*;
}
