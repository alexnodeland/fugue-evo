//! Evolutionary algorithms
//!
//! This module provides various evolutionary algorithm implementations.

pub mod cmaes;
pub mod eda;
pub mod evolution_strategy;
pub mod island;
pub mod nsga2;
pub mod simple_ga;
pub mod steady_state;

pub mod prelude {
    pub use super::cmaes::*;
    pub use super::eda::*;
    pub use super::evolution_strategy::*;
    pub use super::island::*;
    pub use super::nsga2::*;
    pub use super::simple_ga::*;
    pub use super::steady_state::*;
}
