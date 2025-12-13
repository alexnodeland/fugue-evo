//! Session state management for interactive evolution
//!
//! This module provides serializable session state that allows pausing
//! and resuming interactive evolution sessions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "checkpoint")]
use std::fs::File;
#[cfg(feature = "checkpoint")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "checkpoint")]
use std::path::Path;

use super::aggregation::FitnessAggregator;
use super::evaluator::{Candidate, CandidateId, EvaluationRequest};
use super::uncertainty::FitnessEstimate;
use crate::error::CheckpointError;
use crate::genome::traits::EvolutionaryGenome;

/// Current session format version
pub const SESSION_VERSION: u32 = 1;

/// Statistics about evaluation coverage in a session
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoverageStats {
    /// Fraction of population with at least one evaluation (0.0 to 1.0)
    pub coverage: f64,
    /// Average evaluations per candidate
    pub avg_evaluations: f64,
    /// Minimum evaluations for any candidate
    pub min_evaluations: usize,
    /// Maximum evaluations for any candidate
    pub max_evaluations: usize,
    /// Number of candidates with zero evaluations
    pub unevaluated_count: usize,
    /// Total population size
    pub population_size: usize,
}

impl CoverageStats {
    /// Check if coverage meets minimum threshold
    pub fn meets_threshold(&self, min_coverage: f64) -> bool {
        self.coverage >= min_coverage
    }
}

/// Complete state of an interactive evolution session
///
/// This struct captures all state needed to pause and resume an
/// interactive evolution session, including population, fitness
/// aggregator state, and session metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "G: Serialize + for<'a> Deserialize<'a>")]
pub struct InteractiveSession<G>
where
    G: EvolutionaryGenome,
{
    /// Schema version for forward compatibility
    pub version: u32,
    /// Current population with fitness estimates
    pub population: Vec<Candidate<G>>,
    /// Current generation number
    pub generation: usize,
    /// Total evaluation requests made
    pub evaluations_requested: usize,
    /// Total responses received (excluding skips)
    pub responses_received: usize,
    /// Number of skipped evaluations
    pub skipped: usize,
    /// Fitness aggregator state
    pub aggregator: FitnessAggregator,
    /// History of evaluation requests (limited to recent history)
    pub request_history: Vec<SerializedRequest>,
    /// Custom session metadata
    pub metadata: HashMap<String, String>,
    /// Next candidate ID to assign
    pub next_candidate_id: usize,
}

/// Serialized form of an evaluation request (without genome data)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedRequest {
    /// Type of request
    pub request_type: String,
    /// Candidate IDs involved
    pub candidate_ids: Vec<CandidateId>,
    /// Generation when request was made
    pub generation: usize,
    /// Whether this request was skipped
    pub was_skipped: bool,
}

impl<G> InteractiveSession<G>
where
    G: EvolutionaryGenome,
{
    /// Create a new empty session
    pub fn new(aggregator: FitnessAggregator) -> Self {
        Self {
            version: SESSION_VERSION,
            population: Vec::new(),
            generation: 0,
            evaluations_requested: 0,
            responses_received: 0,
            skipped: 0,
            aggregator,
            request_history: Vec::new(),
            metadata: HashMap::new(),
            next_candidate_id: 0,
        }
    }

    /// Create a new session with initial population
    pub fn with_population(population: Vec<Candidate<G>>, aggregator: FitnessAggregator) -> Self {
        let next_id = population.iter().map(|c| c.id.0).max().unwrap_or(0) + 1;
        Self {
            version: SESSION_VERSION,
            population,
            generation: 0,
            evaluations_requested: 0,
            responses_received: 0,
            skipped: 0,
            aggregator,
            request_history: Vec::new(),
            metadata: HashMap::new(),
            next_candidate_id: next_id,
        }
    }

    /// Get the next candidate ID and increment counter
    pub fn next_id(&mut self) -> CandidateId {
        let id = CandidateId(self.next_candidate_id);
        self.next_candidate_id += 1;
        id
    }

    /// Add a candidate to the population
    pub fn add_candidate(&mut self, genome: G) -> CandidateId {
        let id = self.next_id();
        let candidate = Candidate::with_generation(id, genome, self.generation);
        self.population.push(candidate);
        id
    }

    /// Get a candidate by ID
    pub fn get_candidate(&self, id: CandidateId) -> Option<&Candidate<G>> {
        self.population.iter().find(|c| c.id == id)
    }

    /// Get a mutable reference to a candidate by ID
    pub fn get_candidate_mut(&mut self, id: CandidateId) -> Option<&mut Candidate<G>> {
        self.population.iter_mut().find(|c| c.id == id)
    }

    /// Get all candidates that haven't been evaluated
    pub fn unevaluated_candidates(&self) -> Vec<&Candidate<G>> {
        self.population
            .iter()
            .filter(|c| !c.is_evaluated())
            .collect()
    }

    /// Get candidates sorted by fitness (best first)
    pub fn ranked_candidates(&self) -> Vec<&Candidate<G>> {
        let mut candidates: Vec<_> = self.population.iter().collect();
        candidates.sort_by(|a, b| {
            let fa = a.fitness_estimate.unwrap_or(f64::NEG_INFINITY);
            let fb = b.fitness_estimate.unwrap_or(f64::NEG_INFINITY);
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Get the best candidate
    pub fn best_candidate(&self) -> Option<&Candidate<G>> {
        self.population
            .iter()
            .filter(|c| c.fitness_estimate.is_some())
            .max_by(|a, b| {
                let fa = a.fitness_estimate.unwrap();
                let fb = b.fitness_estimate.unwrap();
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Calculate coverage statistics
    pub fn coverage_stats(&self) -> CoverageStats {
        if self.population.is_empty() {
            return CoverageStats::default();
        }

        let eval_counts: Vec<usize> = self.population.iter().map(|c| c.evaluation_count).collect();

        let evaluated = eval_counts.iter().filter(|&&c| c > 0).count();
        let total_evals: usize = eval_counts.iter().sum();

        CoverageStats {
            coverage: evaluated as f64 / self.population.len() as f64,
            avg_evaluations: total_evals as f64 / self.population.len() as f64,
            min_evaluations: eval_counts.iter().copied().min().unwrap_or(0),
            max_evaluations: eval_counts.iter().copied().max().unwrap_or(0),
            unevaluated_count: self.population.len() - evaluated,
            population_size: self.population.len(),
        }
    }

    /// Record that an evaluation request was made
    pub fn record_request<GG: EvolutionaryGenome>(&mut self, request: &EvaluationRequest<GG>) {
        self.evaluations_requested += 1;

        let serialized = SerializedRequest {
            request_type: match request {
                EvaluationRequest::RateCandidates { .. } => "rating".to_string(),
                EvaluationRequest::PairwiseComparison { .. } => "pairwise".to_string(),
                EvaluationRequest::BatchSelection { .. } => "batch".to_string(),
            },
            candidate_ids: request.candidate_ids(),
            generation: self.generation,
            was_skipped: false,
        };

        // Keep limited history
        const MAX_HISTORY: usize = 1000;
        if self.request_history.len() >= MAX_HISTORY {
            self.request_history.remove(0);
        }
        self.request_history.push(serialized);
    }

    /// Record that a response was received
    pub fn record_response(&mut self, was_skipped: bool) {
        if was_skipped {
            self.skipped += 1;
            if let Some(last) = self.request_history.last_mut() {
                last.was_skipped = true;
            }
        } else {
            self.responses_received += 1;
        }
    }

    /// Advance to the next generation
    pub fn advance_generation(&mut self) {
        self.generation += 1;
        self.aggregator.set_generation(self.generation);
    }

    /// Update fitness estimate for a candidate
    pub fn update_fitness(&mut self, id: CandidateId, fitness: f64) {
        if let Some(candidate) = self.get_candidate_mut(id) {
            candidate.set_fitness(fitness);
            candidate.record_evaluation();
        }
    }

    /// Update fitness with full uncertainty information
    pub fn update_fitness_with_uncertainty(&mut self, id: CandidateId, estimate: FitnessEstimate) {
        if let Some(candidate) = self.get_candidate_mut(id) {
            candidate.set_fitness_with_uncertainty(estimate);
            candidate.record_evaluation();
        }
    }

    /// Sync candidate fitness estimates from the aggregator
    ///
    /// Updates all candidates with their current fitness estimates including uncertainty.
    /// Call this after processing responses to ensure candidates have up-to-date estimates.
    pub fn sync_fitness_estimates(&mut self) {
        for candidate in &mut self.population {
            if let Some(estimate) = self.aggregator.get_fitness_estimate(&candidate.id) {
                candidate.fitness_estimate = Some(estimate.mean);
                candidate.fitness_with_uncertainty = Some(estimate);
            }
        }
    }

    /// Get fitness estimates with uncertainty for all candidates
    ///
    /// Returns a vector of (CandidateId, FitnessEstimate) pairs.
    pub fn all_fitness_estimates(&self) -> Vec<(CandidateId, FitnessEstimate)> {
        self.population
            .iter()
            .filter_map(|c| {
                self.aggregator
                    .get_fitness_estimate(&c.id)
                    .map(|e| (c.id, e))
            })
            .collect()
    }

    /// Get candidates sorted by uncertainty (most uncertain first)
    ///
    /// Useful for identifying which candidates need more evaluation.
    pub fn candidates_by_uncertainty(&self) -> Vec<&Candidate<G>> {
        let mut candidates: Vec<_> = self.population.iter().collect();
        candidates.sort_by(|a, b| {
            let var_a = self
                .aggregator
                .get_fitness_estimate(&a.id)
                .map(|e| e.variance)
                .unwrap_or(f64::INFINITY);
            let var_b = self
                .aggregator
                .get_fitness_estimate(&b.id)
                .map(|e| e.variance)
                .unwrap_or(f64::INFINITY);
            // Sort descending - most uncertain first
            var_b
                .partial_cmp(&var_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Get the average uncertainty across all candidates
    pub fn average_uncertainty(&self) -> f64 {
        let estimates: Vec<_> = self
            .population
            .iter()
            .filter_map(|c| self.aggregator.get_fitness_estimate(&c.id))
            .collect();

        if estimates.is_empty() {
            return f64::INFINITY;
        }

        let total_variance: f64 = estimates
            .iter()
            .map(|e| {
                if e.variance.is_finite() {
                    e.variance
                } else {
                    1e6 // Large but finite for averaging
                }
            })
            .sum();

        total_variance / estimates.len() as f64
    }

    /// Replace the population with new candidates
    pub fn replace_population(&mut self, new_population: Vec<Candidate<G>>) {
        let max_id = new_population.iter().map(|c| c.id.0).max().unwrap_or(0);
        self.next_candidate_id = max_id + 1;
        self.population = new_population;
    }

    /// Add metadata to the session
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get response rate (responses / requests)
    pub fn response_rate(&self) -> f64 {
        if self.evaluations_requested > 0 {
            self.responses_received as f64 / self.evaluations_requested as f64
        } else {
            0.0
        }
    }

    /// Get skip rate (skips / requests)
    pub fn skip_rate(&self) -> f64 {
        if self.evaluations_requested > 0 {
            self.skipped as f64 / self.evaluations_requested as f64
        } else {
            0.0
        }
    }
}

/// File-based session persistence (requires `checkpoint` feature)
#[cfg(feature = "checkpoint")]
impl<G> InteractiveSession<G>
where
    G: EvolutionaryGenome + Serialize + for<'de> Deserialize<'de>,
{
    /// Save session to a file
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| {
            CheckpointError::Serialization(format!("Failed to serialize session: {}", e))
        })?;
        Ok(())
    }

    /// Load session from a file
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let session: Self = serde_json::from_reader(reader).map_err(|e| {
            CheckpointError::Deserialization(format!("Failed to deserialize session: {}", e))
        })?;

        // Check version compatibility
        if session.version > SESSION_VERSION {
            return Err(CheckpointError::VersionTooNew(session.version));
        }

        Ok(session)
    }
}

impl<G> InteractiveSession<G>
where
    G: EvolutionaryGenome + Serialize + for<'de> Deserialize<'de>,
{
    /// Serialize session to JSON string (WASM-compatible)
    pub fn to_json(&self) -> Result<String, CheckpointError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| CheckpointError::Serialization(format!("Failed to serialize session: {}", e)))
    }

    /// Deserialize session from JSON string (WASM-compatible)
    pub fn from_json(json: &str) -> Result<Self, CheckpointError> {
        let session: Self = serde_json::from_str(json)
            .map_err(|e| CheckpointError::Deserialization(format!("Failed to deserialize session: {}", e)))?;

        // Check version compatibility
        if session.version > SESSION_VERSION {
            return Err(CheckpointError::VersionTooNew(session.version));
        }

        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::interactive::aggregation::AggregationModel;

    #[test]
    fn test_session_creation() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        assert_eq!(session.generation, 0);
        assert!(session.population.is_empty());
        assert_eq!(session.evaluations_requested, 0);
    }

    #[test]
    fn test_add_candidate() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let id = session.add_candidate(genome);

        assert_eq!(id, CandidateId(0));
        assert_eq!(session.population.len(), 1);
        assert_eq!(session.get_candidate(id).unwrap().birth_generation, 0);
    }

    #[test]
    fn test_coverage_stats() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        // Add 4 candidates
        for i in 0..4 {
            session.add_candidate(RealVector::new(vec![i as f64]));
        }

        // Evaluate 2 of them
        session.population[0].record_evaluation();
        session.population[1].record_evaluation();
        session.population[1].record_evaluation(); // Evaluate twice

        let stats = session.coverage_stats();

        assert_eq!(stats.population_size, 4);
        assert_eq!(stats.coverage, 0.5);
        assert_eq!(stats.unevaluated_count, 2);
        assert_eq!(stats.min_evaluations, 0);
        assert_eq!(stats.max_evaluations, 2);
    }

    #[test]
    fn test_ranked_candidates() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        for i in 0..3 {
            let id = session.add_candidate(RealVector::new(vec![i as f64]));
            session.update_fitness(id, i as f64 * 10.0);
        }

        let ranked = session.ranked_candidates();
        assert_eq!(ranked[0].fitness_estimate, Some(20.0)); // Best first
        assert_eq!(ranked[2].fitness_estimate, Some(0.0)); // Worst last
    }

    #[test]
    fn test_advance_generation() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        session.advance_generation();
        assert_eq!(session.generation, 1);

        let id = session.add_candidate(RealVector::new(vec![1.0]));
        assert_eq!(session.get_candidate(id).unwrap().birth_generation, 1);
    }

    #[test]
    fn test_response_tracking() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        let c1: Candidate<RealVector> = Candidate::new(CandidateId(0), RealVector::new(vec![1.0]));
        let request = EvaluationRequest::rate(vec![c1]);
        session.record_request(&request);
        session.record_response(false);

        session.record_request(&request);
        session.record_response(true); // Skip

        assert_eq!(session.evaluations_requested, 2);
        assert_eq!(session.responses_received, 1);
        assert_eq!(session.skipped, 1);
        assert_eq!(session.response_rate(), 0.5);
        assert_eq!(session.skip_rate(), 0.5);
    }

    #[test]
    fn test_metadata() {
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        session.set_metadata("experiment", "test_run");
        session.set_metadata("user", "alice");

        assert_eq!(
            session.get_metadata("experiment"),
            Some(&"test_run".to_string())
        );
        assert_eq!(session.get_metadata("user"), Some(&"alice".to_string()));
        assert_eq!(session.get_metadata("missing"), None);
    }

    #[test]
    fn test_session_serialization() {
        let aggregator = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });
        let mut session: InteractiveSession<RealVector> = InteractiveSession::new(aggregator);

        session.add_candidate(RealVector::new(vec![1.0, 2.0]));
        session.add_candidate(RealVector::new(vec![3.0, 4.0]));
        session.set_metadata("test", "value");

        // Serialize to JSON
        let json = serde_json::to_string(&session).expect("Failed to serialize");

        // Deserialize back
        let loaded: InteractiveSession<RealVector> =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(loaded.population.len(), 2);
        assert_eq!(loaded.get_metadata("test"), Some(&"value".to_string()));
    }
}
