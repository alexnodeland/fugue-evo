//! Core types for interactive evaluation
//!
//! This module defines the request/response types used for human-in-the-loop
//! fitness evaluation in interactive genetic algorithms.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::genome::traits::EvolutionaryGenome;

/// Unique identifier for a candidate in an interactive session
///
/// Each candidate is assigned a unique ID when created, which remains
/// stable across generations. This allows tracking evaluation history
/// and aggregating feedback over time.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct CandidateId(pub usize);

impl fmt::Display for CandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Candidate({})", self.0)
    }
}

impl From<usize> for CandidateId {
    fn from(id: usize) -> Self {
        Self(id)
    }
}

impl From<CandidateId> for usize {
    fn from(id: CandidateId) -> Self {
        id.0
    }
}

/// A candidate presented for user evaluation
///
/// Wraps a genome with its unique identifier and current fitness estimate.
/// The fitness estimate is updated as user feedback is received and aggregated.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "G: Serialize + for<'a> Deserialize<'a>")]
pub struct Candidate<G>
where
    G: EvolutionaryGenome,
{
    /// Unique identifier for this candidate
    pub id: CandidateId,
    /// The genome of this candidate
    pub genome: G,
    /// Current fitness estimate (updated as feedback arrives)
    pub fitness_estimate: Option<f64>,
    /// Generation when this candidate was created
    pub birth_generation: usize,
    /// Number of times this candidate has been evaluated
    pub evaluation_count: usize,
}

impl<G> Candidate<G>
where
    G: EvolutionaryGenome,
{
    /// Create a new candidate with the given ID and genome
    pub fn new(id: CandidateId, genome: G) -> Self {
        Self {
            id,
            genome,
            fitness_estimate: None,
            birth_generation: 0,
            evaluation_count: 0,
        }
    }

    /// Create a new candidate with birth generation
    pub fn with_generation(id: CandidateId, genome: G, generation: usize) -> Self {
        Self {
            id,
            genome,
            fitness_estimate: None,
            birth_generation: generation,
            evaluation_count: 0,
        }
    }

    /// Set the fitness estimate
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness_estimate = Some(fitness);
    }

    /// Get the fitness estimate, if available
    pub fn fitness(&self) -> Option<f64> {
        self.fitness_estimate
    }

    /// Check if this candidate has been evaluated at least once
    pub fn is_evaluated(&self) -> bool {
        self.evaluation_count > 0
    }

    /// Increment the evaluation count
    pub fn record_evaluation(&mut self) {
        self.evaluation_count += 1;
    }

    /// Get the age of this candidate (generations since birth)
    pub fn age(&self, current_generation: usize) -> usize {
        current_generation.saturating_sub(self.birth_generation)
    }
}

impl<G> PartialEq for Candidate<G>
where
    G: EvolutionaryGenome,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<G> Eq for Candidate<G> where G: EvolutionaryGenome {}

impl<G> Hash for Candidate<G>
where
    G: EvolutionaryGenome,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Rating scale configuration for numeric ratings
///
/// Defines the valid range and behavior for user ratings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RatingScale {
    /// Minimum rating value
    pub min: f64,
    /// Maximum rating value
    pub max: f64,
    /// Whether ties are allowed (for pairwise comparisons)
    pub allow_ties: bool,
    /// Optional step size (e.g., 0.5 for half-star ratings)
    pub step: Option<f64>,
}

impl RatingScale {
    /// Create a new rating scale
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
            allow_ties: true,
            step: None,
        }
    }

    /// Standard 1-10 rating scale
    pub fn one_to_ten() -> Self {
        Self {
            min: 1.0,
            max: 10.0,
            allow_ties: true,
            step: Some(1.0),
        }
    }

    /// Standard 1-5 rating scale (5-star rating)
    pub fn one_to_five() -> Self {
        Self {
            min: 1.0,
            max: 5.0,
            allow_ties: true,
            step: Some(1.0),
        }
    }

    /// Binary like/dislike scale
    pub fn binary() -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            allow_ties: false,
            step: Some(1.0),
        }
    }

    /// Set whether ties are allowed
    pub fn with_ties(mut self, allow: bool) -> Self {
        self.allow_ties = allow;
        self
    }

    /// Set the step size for discrete ratings
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }

    /// Validate a rating against this scale
    pub fn validate(&self, rating: f64) -> bool {
        if rating < self.min || rating > self.max {
            return false;
        }
        if let Some(step) = self.step {
            // Check if rating is a valid step from min
            let steps_from_min = (rating - self.min) / step;
            (steps_from_min - steps_from_min.round()).abs() < 1e-9
        } else {
            true
        }
    }

    /// Clamp a rating to the valid range
    pub fn clamp(&self, rating: f64) -> f64 {
        rating.clamp(self.min, self.max)
    }

    /// Normalize a rating to [0, 1] range
    pub fn normalize(&self, rating: f64) -> f64 {
        (rating - self.min) / (self.max - self.min)
    }

    /// Denormalize a [0, 1] value to this scale
    pub fn denormalize(&self, normalized: f64) -> f64 {
        normalized * (self.max - self.min) + self.min
    }
}

impl Default for RatingScale {
    fn default() -> Self {
        Self::one_to_ten()
    }
}

/// Evaluation request sent to the user
///
/// Represents a request for user feedback on one or more candidates.
/// The type of feedback requested depends on the variant.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "G: Serialize + for<'a> Deserialize<'a>")]
pub enum EvaluationRequest<G>
where
    G: EvolutionaryGenome,
{
    /// Rate individual candidates on a numeric scale
    ///
    /// User assigns a rating to each candidate. Ratings may be partial
    /// (not all candidates need to be rated).
    RateCandidates {
        /// Candidates to rate
        candidates: Vec<Candidate<G>>,
        /// Rating scale to use
        scale: RatingScale,
    },

    /// Compare two candidates - which is better?
    ///
    /// User selects the preferred candidate, or indicates a tie
    /// (if allowed by the scale).
    PairwiseComparison {
        /// First candidate
        candidate_a: Candidate<G>,
        /// Second candidate
        candidate_b: Candidate<G>,
        /// Whether ties are allowed
        allow_tie: bool,
    },

    /// Select N favorites from a batch
    ///
    /// User selects their top candidates from the presented set.
    /// Selection implies preference over non-selected candidates.
    BatchSelection {
        /// Candidates to choose from
        candidates: Vec<Candidate<G>>,
        /// Number of candidates to select
        select_count: usize,
        /// Minimum number required (for partial selections)
        min_select: usize,
    },
}

impl<G> EvaluationRequest<G>
where
    G: EvolutionaryGenome,
{
    /// Create a rate candidates request with default scale
    pub fn rate(candidates: Vec<Candidate<G>>) -> Self {
        Self::RateCandidates {
            candidates,
            scale: RatingScale::default(),
        }
    }

    /// Create a rate candidates request with custom scale
    pub fn rate_with_scale(candidates: Vec<Candidate<G>>, scale: RatingScale) -> Self {
        Self::RateCandidates { candidates, scale }
    }

    /// Create a pairwise comparison request
    pub fn compare(a: Candidate<G>, b: Candidate<G>) -> Self {
        Self::PairwiseComparison {
            candidate_a: a,
            candidate_b: b,
            allow_tie: true,
        }
    }

    /// Create a batch selection request
    pub fn select_from_batch(candidates: Vec<Candidate<G>>, select_count: usize) -> Self {
        Self::BatchSelection {
            candidates,
            select_count,
            min_select: 1,
        }
    }

    /// Get the number of candidates in this request
    pub fn candidate_count(&self) -> usize {
        match self {
            Self::RateCandidates { candidates, .. } => candidates.len(),
            Self::PairwiseComparison { .. } => 2,
            Self::BatchSelection { candidates, .. } => candidates.len(),
        }
    }

    /// Get all candidate IDs in this request
    pub fn candidate_ids(&self) -> Vec<CandidateId> {
        match self {
            Self::RateCandidates { candidates, .. } => candidates.iter().map(|c| c.id).collect(),
            Self::PairwiseComparison {
                candidate_a,
                candidate_b,
                ..
            } => vec![candidate_a.id, candidate_b.id],
            Self::BatchSelection { candidates, .. } => candidates.iter().map(|c| c.id).collect(),
        }
    }
}

/// User response to an evaluation request
///
/// Contains the user's feedback on the presented candidates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EvaluationResponse {
    /// Ratings for candidates
    ///
    /// May be partial (not all candidates rated). Each entry is (candidate_id, rating).
    Ratings(Vec<(CandidateId, f64)>),

    /// Winner of pairwise comparison
    ///
    /// `None` indicates a tie (if allowed).
    PairwiseWinner(Option<CandidateId>),

    /// Selected candidates from batch
    ///
    /// IDs of selected candidates. Order may indicate preference ranking.
    BatchSelected(Vec<CandidateId>),

    /// User chose to skip this evaluation
    ///
    /// No feedback provided for this request.
    Skip,
}

impl EvaluationResponse {
    /// Create a ratings response
    pub fn ratings(ratings: Vec<(CandidateId, f64)>) -> Self {
        Self::Ratings(ratings)
    }

    /// Create a pairwise winner response
    pub fn winner(id: CandidateId) -> Self {
        Self::PairwiseWinner(Some(id))
    }

    /// Create a tie response for pairwise comparison
    pub fn tie() -> Self {
        Self::PairwiseWinner(None)
    }

    /// Create a batch selection response
    pub fn selected(ids: Vec<CandidateId>) -> Self {
        Self::BatchSelected(ids)
    }

    /// Create a skip response
    pub fn skip() -> Self {
        Self::Skip
    }

    /// Check if this is a skip response
    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip)
    }

    /// Get the candidate IDs mentioned in this response
    pub fn mentioned_ids(&self) -> Vec<CandidateId> {
        match self {
            Self::Ratings(ratings) => ratings.iter().map(|(id, _)| *id).collect(),
            Self::PairwiseWinner(Some(id)) => vec![*id],
            Self::PairwiseWinner(None) => vec![],
            Self::BatchSelected(ids) => ids.clone(),
            Self::Skip => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;

    #[test]
    fn test_candidate_id() {
        let id1 = CandidateId(0);
        let id2 = CandidateId(1);
        let id3 = CandidateId(0);

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(format!("{}", id1), "Candidate(0)");
    }

    #[test]
    fn test_candidate_creation() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let candidate: Candidate<RealVector> = Candidate::new(CandidateId(0), genome);

        assert_eq!(candidate.id, CandidateId(0));
        assert!(candidate.fitness_estimate.is_none());
        assert!(!candidate.is_evaluated());
        assert_eq!(candidate.evaluation_count, 0);
    }

    #[test]
    fn test_candidate_evaluation() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let mut candidate: Candidate<RealVector> = Candidate::new(CandidateId(0), genome);

        candidate.set_fitness(7.5);
        candidate.record_evaluation();

        assert_eq!(candidate.fitness(), Some(7.5));
        assert!(candidate.is_evaluated());
        assert_eq!(candidate.evaluation_count, 1);
    }

    #[test]
    fn test_rating_scale_validation() {
        let scale = RatingScale::one_to_ten();

        assert!(scale.validate(1.0));
        assert!(scale.validate(5.0));
        assert!(scale.validate(10.0));
        assert!(!scale.validate(0.0));
        assert!(!scale.validate(11.0));
        assert!(!scale.validate(5.5)); // Step is 1.0
    }

    #[test]
    fn test_rating_scale_normalization() {
        let scale = RatingScale::one_to_ten();

        assert!((scale.normalize(1.0) - 0.0).abs() < 1e-9);
        assert!((scale.normalize(5.5) - 0.5).abs() < 1e-9);
        assert!((scale.normalize(10.0) - 1.0).abs() < 1e-9);

        assert!((scale.denormalize(0.0) - 1.0).abs() < 1e-9);
        assert!((scale.denormalize(0.5) - 5.5).abs() < 1e-9);
        assert!((scale.denormalize(1.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluation_request_rate() {
        let c1: Candidate<RealVector> = Candidate::new(CandidateId(0), RealVector::new(vec![1.0]));
        let c2: Candidate<RealVector> = Candidate::new(CandidateId(1), RealVector::new(vec![2.0]));

        let request = EvaluationRequest::rate(vec![c1, c2]);
        assert_eq!(request.candidate_count(), 2);
        assert_eq!(
            request.candidate_ids(),
            vec![CandidateId(0), CandidateId(1)]
        );
    }

    #[test]
    fn test_evaluation_request_compare() {
        let c1: Candidate<RealVector> = Candidate::new(CandidateId(0), RealVector::new(vec![1.0]));
        let c2: Candidate<RealVector> = Candidate::new(CandidateId(1), RealVector::new(vec![2.0]));

        let request = EvaluationRequest::compare(c1, c2);
        assert_eq!(request.candidate_count(), 2);
    }

    #[test]
    fn test_evaluation_response() {
        let response = EvaluationResponse::winner(CandidateId(0));
        assert_eq!(response.mentioned_ids(), vec![CandidateId(0)]);

        let response = EvaluationResponse::tie();
        assert!(response.mentioned_ids().is_empty());

        let response = EvaluationResponse::skip();
        assert!(response.is_skip());
    }
}
