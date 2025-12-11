//! Genome abstractions and implementations
//!
//! This module provides the core `EvolutionaryGenome` trait and built-in genome types.

pub mod traits;
pub mod real_vector;
pub mod bit_string;
pub mod bounds;

pub mod prelude {
    pub use super::traits::*;
    pub use super::real_vector::*;
    pub use super::bit_string::*;
    pub use super::bounds::*;
}
