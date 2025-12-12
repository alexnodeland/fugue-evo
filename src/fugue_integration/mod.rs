//! Fugue PPL Integration
//!
//! This module provides deep integration with Fugue's probabilistic programming
//! primitives, enabling trace-based evolutionary operators and effect handlers.
//!
//! # Core Concepts
//!
//! - **Traces as Genomes**: Fugue traces (address→value maps) naturally represent genetic material
//! - **Mutation as Resampling**: Trace-based mutation selectively resamples addresses
//! - **Crossover as Trace Merging**: Crossover merges parent traces with constraints
//! - **Effect Handlers**: Poutine-style handlers for evolutionary operations

pub mod trace_operators;
pub mod effect_handlers;
pub mod evolution_model;

pub mod prelude {
    pub use super::trace_operators::*;
    pub use super::effect_handlers::*;
    pub use super::evolution_model::*;
}
