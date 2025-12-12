//! Interactive Genetic Algorithm (IGA) module
//!
//! This module provides support for human-in-the-loop evolutionary optimization,
//! where fitness is derived from user preferences rather than an automated function.
//!
//! # Overview
//!
//! Interactive GAs are useful when:
//! - The fitness function cannot be easily formalized
//! - Human aesthetic judgment is needed (art, design, music generation)
//! - User preferences are subjective and vary per individual
//!
//! # Evaluation Modes
//!
//! The module supports three interaction paradigms:
//!
//! - **Rating**: Users assign numeric scores to individual candidates
//! - **Pairwise Comparison**: Users pick the better of two candidates
//! - **Batch Selection**: Users select their favorites from a presented batch
//!
//! # Example
//!
//! ```rust,ignore
//! use fugue_evo::interactive::prelude::*;
//! use fugue_evo::prelude::*;
//!
//! let mut iga = InteractiveGABuilder::<RealVector>::new()
//!     .population_size(12)
//!     .evaluation_mode(EvaluationMode::BatchSelection)
//!     .batch_size(6)
//!     .bounds(bounds)
//!     .selection(TournamentSelection::new(2))
//!     .crossover(SbxCrossover::new(15.0))
//!     .mutation(PolynomialMutation::new(20.0))
//!     .build()?;
//!
//! loop {
//!     match iga.step(&mut rng) {
//!         StepResult::NeedsEvaluation(request) => {
//!             let response = present_to_user(&request);
//!             iga.provide_response(response);
//!         }
//!         StepResult::GenerationComplete { generation, .. } => {
//!             println!("Generation {} complete", generation);
//!         }
//!         StepResult::Complete(result) => break,
//!     }
//! }
//! ```

pub mod aggregation;
pub mod algorithm;
pub mod evaluator;
pub mod session;
pub mod traits;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::aggregation::{AggregationModel, CandidateStats, FitnessAggregator};
    pub use super::algorithm::{
        InteractiveGA, InteractiveGABuilder, InteractiveGAConfig, InteractiveResult, StepResult,
    };
    pub use super::evaluator::{
        Candidate, CandidateId, EvaluationRequest, EvaluationResponse, RatingScale,
    };
    pub use super::session::{CoverageStats, InteractiveSession};
    pub use super::traits::{EvaluationMode, InteractiveFitness};
}
