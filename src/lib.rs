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
//! This library is primarily a broad, standalone **evolutionary computation**
//! toolkit that *optionally* interoperates with the
//! [fugue-ppl](https://github.com/fugue-ppl/fugue) probabilistic-programming
//! library. Being precise about the coupling (EV-17): the default flagship
//! algorithms — [`SimpleGA`](algorithms::simple_ga::SimpleGA), CMA-ES, NSGA-II,
//! Island Model, Evolution Strategy, EDA/UMDA, SteadyState — are ordinary EC.
//! They use fugue's [`Trace`](fugue::Trace) only as an address→value **data
//! container** for the optional `to_trace`/`from_trace` round-trip; they never
//! construct a fugue `Model` or call its inference engines. The genuine
//! "evolution as Bayesian inference over solution spaces" machinery —
//! `EvolutionarySMC` (tempered SMC over a Boltzmann posterior), `EvolutionStep`,
//! and `BayesianAdaptiveGA` — lives entirely in the [`fugue_integration`]
//! module (exercised by `examples/bayesian_evolution.rs`), which is where
//! fugue's `Model`/`factor` inference path is actually used. Treat that module,
//! not the default algorithms, as the "deep integration" story.
//!
//! ## Features
//!
//! - **Multiple Algorithms**: SimpleGA, CMA-ES, NSGA-II, Island Model, EDA, Interactive GA (standalone EC)
//! - **Flexible Genomes**: RealVector, BitString, Permutation, TreeGenome
//! - **Modular Operators**: Pluggable selection, crossover, and mutation operators
//! - **Adaptive Hyperparameters**: opt-in Thompson-sampling tuning of operator parameters (`SimpleGABuilder::adaptive_operators` + `SimpleGA::run_adaptive`)
//! - **Optional Fugue integration**: [`fugue_integration`] adds a genuine tempered-SMC / Boltzmann inference path over evolutionary traces
//! - **Production Ready**: Checkpointing (bit-identical resume), parallel evaluation, WASM support
//!
//! ## Core Concepts
//!
//! - **Fitness as Likelihood**: the exp(f/T) ↔ conditioning correspondence, realized concretely by [`BoltzmannSelection`](operators::selection::BoltzmannSelection) and by the tempered-SMC path in [`fugue_integration`]
//! - **Learnable Operators**: opt-in online tuning of operator parameters via a Thompson-sampling bandit (`SimpleGABuilder::adaptive_operators` + `run_adaptive`); the default `run` path uses fixed parameters
//! - **Trace-Based Evolution**: genomes round-trip through fugue [`Trace`](fugue::Trace)s as a data structure; the deeper Fugue-integrated inference (SMC/Boltzmann) is scoped to the [`fugue_integration`] module, not the default algorithms
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
//!     // Run optimization. `real_valued()` pins the genome/fitness types (no
//!     // turbofish) and pre-installs tournament selection, SBX crossover, and
//!     // polynomial mutation as overridable defaults.
//!     let result = SimpleGABuilder::real_valued()
//!         .population_size(100)
//!         .bounds(bounds)
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
