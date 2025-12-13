// Clippy allows for intentional patterns in this library
#![allow(clippy::needless_range_loop)] // Matrix operations are clearer with explicit indices
#![allow(clippy::derivable_impls)] // Some Default impls have doc comments
#![allow(clippy::redundant_closure)] // Closure style consistency
#![allow(clippy::should_implement_trait)] // Custom add methods for domain types
#![allow(clippy::get_first)] // Explicit .get(0) is clearer in some contexts
#![allow(clippy::useless_conversion)] // into_iter() for clarity
#![allow(clippy::unnecessary_unwrap)] // Pattern clarity
#![allow(clippy::wrong_self_convention)] // from_* methods for domain types
#![allow(clippy::only_used_in_recursion)] // Tree traversal parameters
#![allow(clippy::if_same_then_else)] // Sometimes intentional for clarity
#![allow(clippy::manual_clamp)] // Explicit clamp logic for clarity
#![allow(clippy::manual_memcpy)] // Matrix operations clarity

//! # fugue-evo
//!
//! A Probabilistic Genetic Algorithm Library for Rust.
//!
//! This library implements genetic algorithms through the lens of probabilistic programming,
//! treating evolution as Bayesian inference over solution spaces. It integrates with
//! [fugue-ppl](https://github.com/fugue-ppl/fugue) for trace-based evolutionary operators.
//!
//! ## Features
//!
//! - **Multiple Algorithms**: SimpleGA, CMA-ES, NSGA-II, Island Model, EDA, Interactive GA
//! - **Flexible Genomes**: RealVector, BitString, Permutation, TreeGenome
//! - **Modular Operators**: Pluggable selection, crossover, and mutation operators
//! - **Adaptive Hyperparameters**: Bayesian learning of mutation rates and other parameters
//! - **Production Ready**: Checkpointing, parallel evaluation, WASM support
//!
//! ## Core Concepts
//!
//! - **Fitness as Likelihood**: Selection pressure maps directly to Bayesian conditioning
//! - **Learnable Operators**: Automatic inference of optimal hyperparameters using conjugate priors
//! - **Trace-Based Evolution**: Deep Fugue integration enables novel probabilistic operators
//! - **Type Safety**: Compile-time guarantees via Rust's type system
//!
//! ## Quick Start
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! fugue-evo = "0.1"
//! rand = "0.8"
//! ```
//!
//! Basic optimization example:
//!
//! ```rust,ignore
//! use fugue_evo::prelude::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!
//!     // Define search bounds: 10 dimensions in [-5.12, 5.12]
//!     let bounds = MultiBounds::symmetric(5.12, 10);
//!
//!     // Run optimization
//!     let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
//!         .population_size(100)
//!         .bounds(bounds)
//!         .selection(TournamentSelection::new(3))
//!         .crossover(SbxCrossover::new(20.0))
//!         .mutation(PolynomialMutation::new(20.0))
//!         .fitness(Sphere::new(10))
//!         .max_generations(200)
//!         .build()?
//!         .run(&mut rng)?;
//!
//!     println!("Best fitness: {:.6}", result.best_fitness);
//!     Ok(())
//! }
//! ```
//!
//! ## Module Overview
//!
//! - [`algorithms`]: Optimization algorithms (SimpleGA, CMA-ES, NSGA-II, Island Model)
//! - [`genome`]: Genome types and the [`EvolutionaryGenome`](genome::traits::EvolutionaryGenome) trait
//! - [`operators`]: Selection, crossover, and mutation operators
//! - [`fitness`]: Fitness traits and benchmark functions
//! - [`population`]: Population management and individual types
//! - [`termination`]: Stopping criteria (max generations, target fitness, stagnation)
//! - [`hyperparameter`]: Adaptive and Bayesian hyperparameter tuning
//! - [`interactive`]: Human-in-the-loop evolutionary optimization
//! - [`checkpoint`]: State serialization for pause/resume
//! - [`fugue_integration`]: Trace operators and effect handlers
//!
//! ## Examples
//!
//! See the `examples/` directory for complete examples:
//!
//! - `sphere_optimization.rs`: Basic continuous optimization
//! - `rastrigin_benchmark.rs`: Multimodal function optimization
//! - `cma_es_example.rs`: CMA-ES algorithm usage
//! - `island_model.rs`: Parallel island evolution
//! - `hyperparameter_learning.rs`: Bayesian parameter adaptation
//! - `symbolic_regression.rs`: Genetic programming
//! - `checkpointing.rs`: Save/restore evolution state
//! - `interactive_evolution.rs`: Human-in-the-loop optimization

pub mod algorithms;
pub mod checkpoint;
pub mod diagnostics;
pub mod error;
pub mod fitness;
pub mod fugue_integration;
pub mod genome;
pub mod hyperparameter;
pub mod interactive;
pub mod operators;
pub mod population;
pub mod termination;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::prelude::*;
    pub use crate::checkpoint::prelude::*;
    pub use crate::diagnostics::prelude::*;
    pub use crate::error::*;
    pub use crate::fitness::prelude::*;
    pub use crate::fugue_integration::prelude::*;
    pub use crate::genome::prelude::*;
    pub use crate::hyperparameter::prelude::*;
    pub use crate::interactive::prelude::*;
    pub use crate::operators::prelude::*;
    pub use crate::population::prelude::*;
    pub use crate::termination::prelude::*;
}
