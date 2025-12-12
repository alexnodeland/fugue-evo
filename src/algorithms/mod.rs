//! Evolutionary algorithms
//!
//! This module provides various evolutionary algorithm implementations.

pub mod cmaes;
pub mod island;
pub mod nsga2;
pub mod simple_ga;

pub mod prelude {
    pub use super::cmaes::*;
    pub use super::island::*;
    pub use super::nsga2::*;
    pub use super::simple_ga::*;
}
