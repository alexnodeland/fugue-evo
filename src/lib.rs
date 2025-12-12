//! # fugue-evo
//!
//! A Probabilistic Genetic Algorithm Library for Rust.
//!
//! This library implements genetic algorithms through the lens of probabilistic programming,
//! treating evolution as Bayesian inference over solution spaces.
//!
//! ## Core Concepts
//!
//! - **Fitness as Likelihood**: Selection pressure maps directly to Bayesian conditioning
//! - **Learnable Operators**: Automatic inference of optimal crossover, mutation, and selection hyperparameters
//! - **Flexible Genomes**: Trait-based abstraction supporting arbitrary genome types
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use fugue_evo::prelude::*;
//! use rand::SeedableRng;
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//!
//! let result = SimpleGA::<RealVector<10>>::builder()
//!     .population_size(100)
//!     .max_generations(500)
//!     .crossover(SbxCrossover::new(20.0))
//!     .mutation(PolynomialMutation::new(20.0))
//!     .selection(TournamentSelection::new(3))
//!     .build(Rastrigin::new(10))?
//!     .run(&mut rng)?;
//! ```

pub mod algorithms;
pub mod diagnostics;
pub mod error;
pub mod fitness;
pub mod fugue_integration;
pub mod genome;
pub mod hyperparameter;
pub mod operators;
pub mod population;
pub mod termination;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::prelude::*;
    pub use crate::diagnostics::prelude::*;
    pub use crate::error::*;
    pub use crate::fitness::prelude::*;
    pub use crate::fugue_integration::prelude::*;
    pub use crate::genome::prelude::*;
    pub use crate::hyperparameter::prelude::*;
    pub use crate::operators::prelude::*;
    pub use crate::population::prelude::*;
    pub use crate::termination::prelude::*;
}
