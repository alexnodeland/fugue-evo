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
//! Evolutionary computation for Rust, in **two layers**:
//!
//! 1. **Classic EC (standalone, no fugue dependency).** SimpleGA, CMA-ES,
//!    NSGA-II, Island Model, Evolution Strategy, EDA/UMDA, SteadyState, the
//!    interactive GA, all operators, checkpointing, and the WASM surface.
//!    Compiles with `--no-default-features --features std,parallel,checkpoint`
//!    with no probabilistic-programming dependency at all.
//! 2. **Evolutionary inference (`ppl` feature, on by default): evolutionary
//!    algorithms *as* probabilistic programs.** The prior over genomes is a
//!    user-written fugue [`Model`](fugue::Model) (a [`GenomePrior`](inference::prior::GenomePrior)),
//!    fitness enters as `factor(β·f(x))`, so the Boltzmann posterior
//!    `π_β(x) ∝ p(x)·exp(β·f(x))` **is a fugue program** — and every sampler
//!    is fugue's own inference machinery:
//!    [`EvolutionChain`](inference::mh::EvolutionChain) (typed single-site MH),
//!    [`EvolutionSMC`](inference::smc::EvolutionSMC) (adaptive tempered SMC
//!    with a population-coupled crossover kernel and a log-evidence estimate),
//!    and [`ArithmeticGrammarPrior`](inference::grammar::ArithmeticGrammarPrior)
//!    (genetic programming over a probabilistic grammar, where subtree
//!    mutation/crossover are generic trace moves).
//!
//! The boundary between the layers is the
//! [`TraceGenome`](genome::trace_genome::TraceGenome) extension trait: classic
//! algorithms require only [`EvolutionaryGenome`](genome::traits::EvolutionaryGenome);
//! genomes that also implement `TraceGenome` can be driven by the inference
//! layer.
//!
//! ## Features
//!
//! - **Multiple Algorithms**: SimpleGA, CMA-ES, NSGA-II, Island Model, EDA, Interactive GA (standalone EC)
//! - **Flexible Genomes**: RealVector, BitString, Permutation, TreeGenome
//! - **Modular Operators**: Pluggable selection, crossover, and mutation operators
//! - **Adaptive Hyperparameters**: opt-in Thompson-sampling tuning of operator parameters
//! - **Evolutionary inference** (`ppl`): priors as programs, tempered SMC over the
//!   Boltzmann posterior, MH with typed proposals, symbolic regression as exact
//!   Bayesian inference
//! - **Production Ready**: Checkpointing (bit-identical resume), parallel evaluation, WASM support
//!
//! ## Quick Start (classic optimization)
//!
//! ```rust,ignore
//! use fugue_evo::prelude::*;
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut rng = StdRng::seed_from_u64(42);
//!     let bounds = MultiBounds::symmetric(5.12, 10);
//!     let result = SimpleGABuilder::real_valued()
//!         .population_size(100)
//!         .bounds(bounds)
//!         .fitness(Sphere::new(10))
//!         .max_generations(200)
//!         .build()?
//!         .run(&mut rng)?;
//!     println!("Best fitness: {:.6}", result.best_fitness);
//!     Ok(())
//! }
//! ```
//!
//! ## Quick Start (evolution as inference, `ppl`)
//!
//! ```rust,ignore
//! use fugue_evo::prelude::*;
//!
//! // Prior as a program; fitness as a likelihood factor; posterior by SMC.
//! let model = EvolutionModel::new(GaussianPrior::new(0.0, 2.0, DIM), fitness);
//! let posterior = EvolutionSMC::run(&mut rng, &model, EvoSmcConfig::default());
//! println!("posterior mean: {}", posterior.weighted_mean(0));
//! println!("log evidence:   {}", posterior.log_evidence);
//! ```
//!
//! ## Module Overview
//!
//! - [`algorithms`]: Classic optimization algorithms (SimpleGA, CMA-ES, NSGA-II, Island Model)
//! - [`genome`]: Genome types, [`EvolutionaryGenome`](genome::traits::EvolutionaryGenome), and (behind `ppl`) [`TraceGenome`](genome::trace_genome::TraceGenome)
//! - [`operators`]: Selection, crossover, and mutation operators
//! - [`fitness`]: Fitness traits and benchmark functions
//! - [`population`]: Population management and individual types
//! - [`termination`]: Stopping criteria
//! - [`hyperparameter`]: Adaptive and Bayesian hyperparameter tuning
//! - [`interactive`]: Human-in-the-loop evolutionary optimization
//! - [`checkpoint`]: State serialization for pause/resume
//! - [`inference`]: Evolution as inference — priors as programs, MH, tempered SMC, grammar GP (`ppl`)
//!
//! ## Examples
//!
//! - `sphere_optimization.rs`, `rastrigin_benchmark.rs`, `cma_es_example.rs`,
//!   `island_model.rs`, `symbolic_regression.rs` (classic GP),
//!   `checkpointing.rs`, `interactive_evolution.rs`: the classic layer
//! - `bayesian_evolution.rs`: the inference layer end-to-end (SMC + MH + adaptive GA)
//! - `symbolic_regression_inference.rs`: **flagship** — symbolic regression as
//!   exact Bayesian inference over a probabilistic grammar

pub mod algorithms;
pub mod checkpoint;
pub mod diagnostics;
pub mod error;
pub mod fitness;
#[cfg(feature = "ppl")]
pub mod inference;

/// Deprecated alias for [`inference`] (the module was renamed in 0.2.0).
#[cfg(feature = "ppl")]
#[deprecated(since = "0.2.0", note = "renamed to `inference`")]
pub use inference as fugue_integration;
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
    pub use crate::genome::prelude::*;
    pub use crate::hyperparameter::prelude::*;
    #[cfg(feature = "ppl")]
    pub use crate::inference::prelude::*;
    pub use crate::interactive::prelude::*;
    pub use crate::operators::prelude::*;
    pub use crate::population::prelude::*;
    pub use crate::termination::prelude::*;
}
