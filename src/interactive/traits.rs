//! Interactive fitness traits
//!
//! This module defines the `InteractiveFitness` trait for human-in-the-loop
//! fitness evaluation, as well as supporting types for evaluation modes.

use serde::{Deserialize, Serialize};

use super::aggregation::FitnessAggregator;
use super::evaluator::{Candidate, CandidateId, EvaluationRequest, EvaluationResponse};
use crate::genome::traits::EvolutionaryGenome;

/// Evaluation mode for interactive fitness
///
/// Determines how user feedback is collected during evolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationMode {
    /// User rates each candidate independently on a numeric scale
    ///
    /// Best for: Absolute quality assessment, when users can easily assign scores
    Rating,

    /// User compares pairs of candidates and selects the better one
    ///
    /// Best for: When relative comparisons are easier than absolute ratings,
    /// provides consistent transitive preferences
    Pairwise,

    /// User selects top N favorites from a batch
    ///
    /// Best for: Quick evaluation of many candidates, implicit ranking
    BatchSelection,

    /// System chooses evaluation mode adaptively based on population state
    ///
    /// May switch between modes based on coverage, convergence, or user fatigue
    Adaptive,
}

impl EvaluationMode {
    /// Returns a human-readable description of this mode
    pub fn description(&self) -> &'static str {
        match self {
            Self::Rating => "Rate each candidate on a numeric scale",
            Self::Pairwise => "Compare pairs and select the better one",
            Self::BatchSelection => "Select favorites from a batch",
            Self::Adaptive => "System adapts evaluation method automatically",
        }
    }
}

impl Default for EvaluationMode {
    fn default() -> Self {
        Self::Rating
    }
}

/// Trait for interactive fitness evaluation
///
/// Unlike the synchronous [`Fitness`](crate::fitness::traits::Fitness) trait that returns
/// immediate values, `InteractiveFitness` generates evaluation requests that must
/// be fulfilled by user interaction.
///
/// # Design
///
/// The trait is designed around a request/response pattern:
/// 1. Algorithm calls `request_evaluation()` with candidates needing feedback
/// 2. UI presents the request to the user and collects their response
/// 3. Algorithm calls `process_response()` to update fitness estimates
///
/// # Example Implementation
///
/// ```rust,ignore
/// use fugue_evo::interactive::prelude::*;
///
/// struct ArtFitness {
///     mode: EvaluationMode,
/// }
///
/// impl InteractiveFitness for ArtFitness {
///     type Genome = MyArtGenome;
///
///     fn evaluation_mode(&self) -> EvaluationMode {
///         self.mode
///     }
///
///     fn request_evaluation(
///         &self,
///         candidates: &[Candidate<Self::Genome>],
///     ) -> EvaluationRequest<Self::Genome> {
///         match self.mode {
///             EvaluationMode::Rating => {
///                 EvaluationRequest::rate(candidates.to_vec())
///             }
///             EvaluationMode::BatchSelection => {
///                 EvaluationRequest::select_from_batch(candidates.to_vec(), 3)
///             }
///             _ => unimplemented!()
///         }
///     }
///
///     fn process_response(
///         &mut self,
///         response: EvaluationResponse,
///         aggregator: &mut FitnessAggregator,
///     ) -> Vec<(CandidateId, f64)> {
///         // Delegate to aggregator for standard processing
///         aggregator.process_response(&response)
///     }
/// }
/// ```
pub trait InteractiveFitness: Send + Sync {
    /// The genome type being evaluated
    type Genome: EvolutionaryGenome;

    /// Get the preferred evaluation mode for this fitness function
    fn evaluation_mode(&self) -> EvaluationMode;

    /// Generate an evaluation request for the given candidates
    ///
    /// The returned request will be presented to the user for feedback.
    /// The implementation should select candidates appropriately for the
    /// current evaluation mode.
    fn request_evaluation(
        &self,
        candidates: &[Candidate<Self::Genome>],
    ) -> EvaluationRequest<Self::Genome>;

    /// Process user response and update fitness estimates
    ///
    /// Returns the updated fitness values for affected candidates.
    /// The aggregator maintains cumulative statistics and should be
    /// used for fitness computation.
    fn process_response(
        &mut self,
        response: EvaluationResponse,
        aggregator: &mut FitnessAggregator,
    ) -> Vec<(CandidateId, f64)>;

    /// Optional: Called at the start of each generation
    ///
    /// Allows the fitness function to adjust strategy based on
    /// population state or user fatigue.
    fn on_generation_start(&mut self, _generation: usize, _population_size: usize) {}

    /// Optional: Called when an evaluation is skipped
    ///
    /// Allows tracking of user fatigue or disengagement.
    fn on_evaluation_skipped(&mut self) {}
}

/// Default interactive fitness implementation using a fixed evaluation mode
///
/// This provides a simple implementation that delegates all processing
/// to the fitness aggregator. Suitable for most use cases.
#[derive(Clone, Debug)]
pub struct DefaultInteractiveFitness<G>
where
    G: EvolutionaryGenome,
{
    mode: EvaluationMode,
    batch_size: usize,
    select_count: usize,
    _marker: std::marker::PhantomData<G>,
}

impl<G> DefaultInteractiveFitness<G>
where
    G: EvolutionaryGenome,
{
    /// Create a new default interactive fitness with the given mode
    pub fn new(mode: EvaluationMode) -> Self {
        Self {
            mode,
            batch_size: 6,
            select_count: 2,
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the batch size for batch selection mode
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set how many candidates to select in batch selection mode
    pub fn with_select_count(mut self, count: usize) -> Self {
        self.select_count = count;
        self
    }
}

impl<G> Default for DefaultInteractiveFitness<G>
where
    G: EvolutionaryGenome,
{
    fn default() -> Self {
        Self::new(EvaluationMode::Rating)
    }
}

impl<G> InteractiveFitness for DefaultInteractiveFitness<G>
where
    G: EvolutionaryGenome + Clone + Send + Sync,
{
    type Genome = G;

    fn evaluation_mode(&self) -> EvaluationMode {
        self.mode
    }

    fn request_evaluation(
        &self,
        candidates: &[Candidate<Self::Genome>],
    ) -> EvaluationRequest<Self::Genome> {
        match self.mode {
            EvaluationMode::Rating => EvaluationRequest::rate(candidates.to_vec()),
            EvaluationMode::Pairwise => {
                // Select two candidates for comparison
                if candidates.len() >= 2 {
                    EvaluationRequest::compare(candidates[0].clone(), candidates[1].clone())
                } else if candidates.len() == 1 {
                    // Fall back to rating if only one candidate
                    EvaluationRequest::rate(candidates.to_vec())
                } else {
                    EvaluationRequest::rate(vec![])
                }
            }
            EvaluationMode::BatchSelection => {
                let batch: Vec<_> = candidates.iter().take(self.batch_size).cloned().collect();
                EvaluationRequest::select_from_batch(batch, self.select_count)
            }
            EvaluationMode::Adaptive => {
                // Default to rating for adaptive mode
                EvaluationRequest::rate(candidates.to_vec())
            }
        }
    }

    fn process_response(
        &mut self,
        response: EvaluationResponse,
        aggregator: &mut FitnessAggregator,
    ) -> Vec<(CandidateId, f64)> {
        aggregator.process_response(&response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::interactive::aggregation::AggregationModel;

    #[test]
    fn test_evaluation_mode_default() {
        assert_eq!(EvaluationMode::default(), EvaluationMode::Rating);
    }

    #[test]
    fn test_evaluation_mode_description() {
        assert!(!EvaluationMode::Rating.description().is_empty());
        assert!(!EvaluationMode::Pairwise.description().is_empty());
        assert!(!EvaluationMode::BatchSelection.description().is_empty());
        assert!(!EvaluationMode::Adaptive.description().is_empty());
    }

    #[test]
    fn test_default_interactive_fitness_rating() {
        let fitness: DefaultInteractiveFitness<RealVector> =
            DefaultInteractiveFitness::new(EvaluationMode::Rating);

        let c1 = Candidate::new(CandidateId(0), RealVector::new(vec![1.0]));
        let c2 = Candidate::new(CandidateId(1), RealVector::new(vec![2.0]));

        let request = fitness.request_evaluation(&[c1, c2]);
        match request {
            EvaluationRequest::RateCandidates { candidates, .. } => {
                assert_eq!(candidates.len(), 2);
            }
            _ => panic!("Expected RateCandidates request"),
        }
    }

    #[test]
    fn test_default_interactive_fitness_pairwise() {
        let fitness: DefaultInteractiveFitness<RealVector> =
            DefaultInteractiveFitness::new(EvaluationMode::Pairwise);

        let c1 = Candidate::new(CandidateId(0), RealVector::new(vec![1.0]));
        let c2 = Candidate::new(CandidateId(1), RealVector::new(vec![2.0]));

        let request = fitness.request_evaluation(&[c1, c2]);
        match request {
            EvaluationRequest::PairwiseComparison { .. } => {}
            _ => panic!("Expected PairwiseComparison request"),
        }
    }

    #[test]
    fn test_default_interactive_fitness_batch() {
        let fitness: DefaultInteractiveFitness<RealVector> =
            DefaultInteractiveFitness::new(EvaluationMode::BatchSelection)
                .with_batch_size(4)
                .with_select_count(2);

        let candidates: Vec<_> = (0..6)
            .map(|i| Candidate::new(CandidateId(i), RealVector::new(vec![i as f64])))
            .collect();

        let request = fitness.request_evaluation(&candidates);
        match request {
            EvaluationRequest::BatchSelection {
                candidates,
                select_count,
                ..
            } => {
                assert_eq!(candidates.len(), 4); // batch_size
                assert_eq!(select_count, 2);
            }
            _ => panic!("Expected BatchSelection request"),
        }
    }

    #[test]
    fn test_default_interactive_fitness_process_response() {
        let mut fitness: DefaultInteractiveFitness<RealVector> =
            DefaultInteractiveFitness::new(EvaluationMode::Rating);
        let mut aggregator = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });

        let response =
            EvaluationResponse::ratings(vec![(CandidateId(0), 8.0), (CandidateId(1), 6.0)]);

        let updated = fitness.process_response(response, &mut aggregator);
        assert_eq!(updated.len(), 2);
    }
}
