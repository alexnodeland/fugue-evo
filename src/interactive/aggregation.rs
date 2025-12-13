//! Fitness aggregation models for interactive evaluation
//!
//! This module provides various statistical models for converting user feedback
//! (ratings, comparisons, selections) into fitness values suitable for evolution.
//!
//! # Available Models
//!
//! - **DirectRating**: Simple average of user ratings
//! - **Elo**: Classic Elo rating system from pairwise comparisons
//! - **BradleyTerry**: Maximum likelihood estimation for pairwise data
//! - **ImplicitRanking**: Bonus/penalty system from batch selections
//!
//! # Uncertainty Quantification
//!
//! All models support uncertainty estimation via `get_fitness_estimate()`,
//! which returns a `FitnessEstimate` with variance and confidence intervals.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::bradley_terry::{BradleyTerryModel, BradleyTerryOptimizer};
use super::evaluator::{CandidateId, EvaluationResponse};
use super::uncertainty::FitnessEstimate;

/// Aggregation model for converting user feedback to fitness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AggregationModel {
    /// Direct rating average
    ///
    /// Simply averages all ratings received for each candidate.
    /// Uses default_rating for candidates with no ratings.
    DirectRating {
        /// Default rating for unevaluated candidates
        default_rating: f64,
    },

    /// Elo rating system
    ///
    /// Classic chess-style rating from pairwise comparisons.
    /// Good for transitive preference modeling.
    Elo {
        /// Initial rating for new candidates
        initial_rating: f64,
        /// K-factor controlling rating volatility
        k_factor: f64,
    },

    /// Bradley-Terry model
    ///
    /// Maximum likelihood estimation for pairwise comparison data.
    /// Provides more statistically principled estimates than Elo.
    /// Now supports proper MLE with Newton-Raphson or MM algorithms.
    BradleyTerry {
        /// Initial strength parameter
        initial_strength: f64,
        /// Optimizer configuration (Newton-Raphson or MM)
        #[serde(default)]
        optimizer: BradleyTerryOptimizer,
    },

    /// Legacy Bradley-Terry model (for backward compatibility)
    ///
    /// Uses the simplified iterative MM approach from earlier versions.
    #[serde(alias = "BradleyTerryLegacy")]
    BradleyTerrySimple {
        /// Initial strength parameter
        initial_strength: f64,
        /// Learning rate for iterative updates
        learning_rate: f64,
        /// Number of iterations
        iterations: usize,
    },

    /// Implicit ranking from batch selections
    ///
    /// Assigns bonuses to selected candidates and penalties to
    /// non-selected candidates in each batch.
    ImplicitRanking {
        /// Fitness bonus for being selected
        selected_bonus: f64,
        /// Fitness penalty for not being selected
        not_selected_penalty: f64,
        /// Base fitness for all candidates
        base_fitness: f64,
    },
}

impl Default for AggregationModel {
    fn default() -> Self {
        Self::DirectRating {
            default_rating: 5.0,
        }
    }
}

/// Statistics tracked for each candidate
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CandidateStats {
    /// Sum of all ratings received
    pub rating_sum: f64,
    /// Sum of squared ratings (for variance calculation)
    #[serde(default)]
    pub rating_sum_squares: f64,
    /// Count of ratings received
    pub rating_count: usize,
    /// Current model-based score (Elo, Bradley-Terry strength, etc.)
    pub model_score: f64,
    /// Variance of the model score (for uncertainty quantification)
    #[serde(default = "default_variance")]
    pub model_variance: f64,
    /// Number of wins in pairwise comparisons
    pub wins: usize,
    /// Number of losses in pairwise comparisons
    pub losses: usize,
    /// Number of ties in pairwise comparisons
    pub ties: usize,
    /// Times selected in batch selection
    pub times_selected: usize,
    /// Times presented but not selected
    pub times_passed: usize,
}

fn default_variance() -> f64 {
    f64::INFINITY
}

impl CandidateStats {
    /// Create new stats with the given initial model score
    pub fn new(initial_score: f64) -> Self {
        Self {
            model_score: initial_score,
            model_variance: f64::INFINITY,
            ..Default::default()
        }
    }

    /// Get the average rating (or None if no ratings)
    pub fn average_rating(&self) -> Option<f64> {
        if self.rating_count > 0 {
            Some(self.rating_sum / self.rating_count as f64)
        } else {
            None
        }
    }

    /// Get the sample variance of ratings
    pub fn rating_variance(&self) -> Option<f64> {
        if self.rating_count < 2 {
            return None;
        }
        let n = self.rating_count as f64;
        let mean = self.rating_sum / n;
        // Var = E[X²] - E[X]²
        let var = (self.rating_sum_squares / n) - (mean * mean);
        // Convert to sample variance (Bessel's correction)
        Some(var * n / (n - 1.0))
    }

    /// Get the variance of the mean (standard error squared)
    pub fn rating_variance_of_mean(&self) -> Option<f64> {
        self.rating_variance()
            .map(|var| var / self.rating_count as f64)
    }

    /// Get total number of comparisons
    pub fn total_comparisons(&self) -> usize {
        self.wins + self.losses + self.ties
    }

    /// Get win rate (0.0 to 1.0)
    pub fn win_rate(&self) -> Option<f64> {
        let total = self.total_comparisons();
        if total > 0 {
            Some(self.wins as f64 / total as f64)
        } else {
            None
        }
    }

    /// Get selection rate (0.0 to 1.0)
    pub fn selection_rate(&self) -> Option<f64> {
        let total = self.times_selected + self.times_passed;
        if total > 0 {
            Some(self.times_selected as f64 / total as f64)
        } else {
            None
        }
    }
}

/// Record of a pairwise comparison
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonRecord {
    /// Winner's ID
    pub winner: CandidateId,
    /// Loser's ID
    pub loser: CandidateId,
    /// Generation when comparison occurred
    pub generation: usize,
}

/// Aggregates partial/incremental feedback into fitness estimates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessAggregator {
    /// The aggregation model to use
    model: AggregationModel,
    /// Per-candidate statistics
    candidate_stats: HashMap<CandidateId, CandidateStats>,
    /// History of pairwise comparisons (for Bradley-Terry updates)
    comparisons: Vec<ComparisonRecord>,
    /// Current generation
    current_generation: usize,
}

impl FitnessAggregator {
    /// Create a new aggregator with the given model
    pub fn new(model: AggregationModel) -> Self {
        Self {
            model,
            candidate_stats: HashMap::new(),
            comparisons: Vec::new(),
            current_generation: 0,
        }
    }

    /// Get the aggregation model
    pub fn model(&self) -> &AggregationModel {
        &self.model
    }

    /// Set the current generation
    pub fn set_generation(&mut self, generation: usize) {
        self.current_generation = generation;
    }

    /// Ensure a candidate has stats initialized
    fn ensure_stats(&mut self, id: CandidateId) {
        if !self.candidate_stats.contains_key(&id) {
            let initial_score = match &self.model {
                AggregationModel::DirectRating { default_rating } => *default_rating,
                AggregationModel::Elo { initial_rating, .. } => *initial_rating,
                AggregationModel::BradleyTerry {
                    initial_strength, ..
                } => *initial_strength,
                AggregationModel::BradleyTerrySimple {
                    initial_strength, ..
                } => *initial_strength,
                AggregationModel::ImplicitRanking { base_fitness, .. } => *base_fitness,
            };
            self.candidate_stats
                .insert(id, CandidateStats::new(initial_score));
        }
    }

    /// Get stats for a candidate
    pub fn get_stats(&self, id: &CandidateId) -> Option<&CandidateStats> {
        self.candidate_stats.get(id)
    }

    /// Get current fitness estimate for a candidate (point estimate only)
    ///
    /// For uncertainty information, use `get_fitness_estimate()` instead.
    pub fn get_fitness(&self, id: &CandidateId) -> Option<f64> {
        let stats = self.candidate_stats.get(id)?;

        Some(match &self.model {
            AggregationModel::DirectRating { default_rating } => {
                stats.average_rating().unwrap_or(*default_rating)
            }
            AggregationModel::Elo { .. } => stats.model_score,
            AggregationModel::BradleyTerry { .. } => stats.model_score,
            AggregationModel::BradleyTerrySimple { .. } => stats.model_score,
            AggregationModel::ImplicitRanking { .. } => {
                // Score is base + cumulative bonuses/penalties
                stats.model_score
            }
        })
    }

    /// Get fitness estimate with uncertainty quantification
    ///
    /// Returns a `FitnessEstimate` containing the point estimate, variance,
    /// and confidence intervals.
    pub fn get_fitness_estimate(&self, id: &CandidateId) -> Option<FitnessEstimate> {
        let stats = self.candidate_stats.get(id)?;

        Some(match &self.model {
            AggregationModel::DirectRating { default_rating } => {
                if stats.rating_count == 0 {
                    FitnessEstimate::uninformative(*default_rating)
                } else {
                    let mean = stats.rating_sum / stats.rating_count as f64;
                    let variance = stats.rating_variance_of_mean().unwrap_or(f64::INFINITY);
                    FitnessEstimate::new(mean, variance, stats.rating_count)
                }
            }
            AggregationModel::Elo { k_factor, .. } => {
                // Elo variance approximation based on K-factor and game count
                let n_games = stats.total_comparisons();
                let variance = if n_games == 0 {
                    f64::INFINITY
                } else {
                    // Approximate variance: decreases with games, proportional to K²
                    let base_var = k_factor * k_factor * 0.25; // Bernoulli variance factor
                    base_var / n_games as f64
                };
                FitnessEstimate::new(stats.model_score, variance, n_games)
            }
            AggregationModel::BradleyTerry { .. } | AggregationModel::BradleyTerrySimple { .. } => {
                // Use stored variance from MLE computation
                let n_comparisons = stats.total_comparisons();
                let variance = if stats.model_variance.is_finite() {
                    stats.model_variance
                } else if n_comparisons == 0 {
                    f64::INFINITY
                } else {
                    // Fallback: approximate variance
                    1.0 / n_comparisons as f64
                };
                FitnessEstimate::new(stats.model_score, variance, n_comparisons)
            }
            AggregationModel::ImplicitRanking { .. } => {
                // Binomial variance on selection rate
                let n = stats.times_selected + stats.times_passed;
                if n == 0 {
                    FitnessEstimate::uninformative(stats.model_score)
                } else {
                    let p = stats.times_selected as f64 / n as f64;
                    let variance = p * (1.0 - p) / n as f64;
                    FitnessEstimate::new(stats.model_score, variance, n)
                }
            }
        })
    }

    /// Get access to comparison records (for Bradley-Terry MLE)
    pub fn comparisons(&self) -> &[ComparisonRecord] {
        &self.comparisons
    }

    /// Record a rating for a candidate
    pub fn record_rating(&mut self, id: CandidateId, rating: f64) {
        self.ensure_stats(id);
        if let Some(stats) = self.candidate_stats.get_mut(&id) {
            stats.rating_sum += rating;
            stats.rating_sum_squares += rating * rating;
            stats.rating_count += 1;
        }
    }

    /// Record a pairwise comparison result
    pub fn record_comparison(&mut self, winner: CandidateId, loser: CandidateId) {
        self.ensure_stats(winner);
        self.ensure_stats(loser);

        // Update stats
        if let Some(winner_stats) = self.candidate_stats.get_mut(&winner) {
            winner_stats.wins += 1;
        }
        if let Some(loser_stats) = self.candidate_stats.get_mut(&loser) {
            loser_stats.losses += 1;
        }

        // Record comparison for history
        self.comparisons.push(ComparisonRecord {
            winner,
            loser,
            generation: self.current_generation,
        });

        // Update model scores
        match &self.model {
            AggregationModel::Elo { k_factor, .. } => {
                self.update_elo(winner, loser, *k_factor);
            }
            AggregationModel::BradleyTerry { .. } => {
                // Bradley-Terry updates are batched via recompute_all()
            }
            _ => {}
        }
    }

    /// Record a tie in pairwise comparison
    pub fn record_tie(&mut self, id_a: CandidateId, id_b: CandidateId) {
        self.ensure_stats(id_a);
        self.ensure_stats(id_b);

        if let Some(stats) = self.candidate_stats.get_mut(&id_a) {
            stats.ties += 1;
        }
        if let Some(stats) = self.candidate_stats.get_mut(&id_b) {
            stats.ties += 1;
        }

        // For Elo, treat tie as half-win each
        if let AggregationModel::Elo { k_factor, .. } = &self.model {
            self.update_elo_draw(id_a, id_b, *k_factor);
        }
    }

    /// Record batch selection results
    pub fn record_batch_selection(
        &mut self,
        selected: &[CandidateId],
        not_selected: &[CandidateId],
    ) {
        if let AggregationModel::ImplicitRanking {
            selected_bonus,
            not_selected_penalty,
            ..
        } = &self.model
        {
            let bonus = *selected_bonus;
            let penalty = *not_selected_penalty;

            for &id in selected {
                self.ensure_stats(id);
                if let Some(stats) = self.candidate_stats.get_mut(&id) {
                    stats.times_selected += 1;
                    stats.model_score += bonus;
                }
            }

            for &id in not_selected {
                self.ensure_stats(id);
                if let Some(stats) = self.candidate_stats.get_mut(&id) {
                    stats.times_passed += 1;
                    stats.model_score -= penalty;
                }
            }
        } else {
            // For other models, just track selection counts
            for &id in selected {
                self.ensure_stats(id);
                if let Some(stats) = self.candidate_stats.get_mut(&id) {
                    stats.times_selected += 1;
                }
            }
            for &id in not_selected {
                self.ensure_stats(id);
                if let Some(stats) = self.candidate_stats.get_mut(&id) {
                    stats.times_passed += 1;
                }
            }
        }
    }

    /// Update Elo ratings after a comparison
    fn update_elo(&mut self, winner: CandidateId, loser: CandidateId, k: f64) {
        let winner_rating = self
            .candidate_stats
            .get(&winner)
            .map(|s| s.model_score)
            .unwrap_or(1500.0);
        let loser_rating = self
            .candidate_stats
            .get(&loser)
            .map(|s| s.model_score)
            .unwrap_or(1500.0);

        // Expected scores
        let exp_winner = 1.0 / (1.0 + 10.0_f64.powf((loser_rating - winner_rating) / 400.0));
        let exp_loser = 1.0 - exp_winner;

        // Update ratings
        if let Some(stats) = self.candidate_stats.get_mut(&winner) {
            stats.model_score += k * (1.0 - exp_winner);
        }
        if let Some(stats) = self.candidate_stats.get_mut(&loser) {
            stats.model_score += k * (0.0 - exp_loser);
        }
    }

    /// Update Elo ratings after a draw
    fn update_elo_draw(&mut self, id_a: CandidateId, id_b: CandidateId, k: f64) {
        let rating_a = self
            .candidate_stats
            .get(&id_a)
            .map(|s| s.model_score)
            .unwrap_or(1500.0);
        let rating_b = self
            .candidate_stats
            .get(&id_b)
            .map(|s| s.model_score)
            .unwrap_or(1500.0);

        // Expected scores
        let exp_a = 1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0));
        let exp_b = 1.0 - exp_a;

        // Update ratings (actual = 0.5 for draw)
        if let Some(stats) = self.candidate_stats.get_mut(&id_a) {
            stats.model_score += k * (0.5 - exp_a);
        }
        if let Some(stats) = self.candidate_stats.get_mut(&id_b) {
            stats.model_score += k * (0.5 - exp_b);
        }
    }

    /// Recompute all fitness estimates from comparison history
    ///
    /// This is useful for Bradley-Terry model which uses batch MLE,
    /// or after loading a session from checkpoint.
    pub fn recompute_all(&mut self) -> HashMap<CandidateId, f64> {
        match &self.model {
            AggregationModel::BradleyTerry { optimizer, .. } => {
                self.recompute_bradley_terry_mle(optimizer.clone());
            }
            AggregationModel::BradleyTerrySimple {
                initial_strength,
                learning_rate,
                iterations,
            } => {
                self.recompute_bradley_terry_simple(*initial_strength, *learning_rate, *iterations);
            }
            _ => {}
        }

        // Return current fitness estimates
        self.candidate_stats
            .keys()
            .filter_map(|id| self.get_fitness(id).map(|f| (*id, f)))
            .collect()
    }

    /// Recompute Bradley-Terry using proper MLE (Newton-Raphson or MM)
    fn recompute_bradley_terry_mle(&mut self, optimizer: BradleyTerryOptimizer) {
        let ids: Vec<CandidateId> = self.candidate_stats.keys().copied().collect();
        if ids.is_empty() || self.comparisons.is_empty() {
            return;
        }

        let model = BradleyTerryModel::new(optimizer);
        let result = model.fit(&self.comparisons, &ids);

        // Update stats with MLE results
        for (&id, &strength) in &result.strengths {
            if let Some(stats) = self.candidate_stats.get_mut(&id) {
                stats.model_score = strength;

                // Update variance from covariance matrix
                if let Some(&idx) = result.id_to_index.get(&id) {
                    if idx < result.covariance.nrows() {
                        stats.model_variance = result.covariance[(idx, idx)];
                    }
                }
            }
        }
    }

    /// Recompute Bradley-Terry using simplified iterative MM (legacy)
    fn recompute_bradley_terry_simple(
        &mut self,
        initial_strength: f64,
        learning_rate: f64,
        iterations: usize,
    ) {
        // Initialize strengths
        let ids: Vec<CandidateId> = self.candidate_stats.keys().copied().collect();
        for &id in &ids {
            if let Some(stats) = self.candidate_stats.get_mut(&id) {
                stats.model_score = initial_strength;
            }
        }

        // Iterative MM algorithm for Bradley-Terry
        for _ in 0..iterations {
            let mut new_scores: HashMap<CandidateId, f64> = HashMap::new();

            for &id in &ids {
                let stats = match self.candidate_stats.get(&id) {
                    Some(s) => s,
                    None => continue,
                };

                let wins = stats.wins as f64;
                if wins == 0.0 {
                    new_scores.insert(id, stats.model_score);
                    continue;
                }

                // Compute denominator: sum of 1/(p_i + p_j) over all comparisons
                let mut denom = 0.0;
                for comparison in &self.comparisons {
                    if comparison.winner == id {
                        let other_score = self
                            .candidate_stats
                            .get(&comparison.loser)
                            .map(|s| s.model_score)
                            .unwrap_or(initial_strength);
                        denom += 1.0 / (stats.model_score + other_score);
                    } else if comparison.loser == id {
                        let other_score = self
                            .candidate_stats
                            .get(&comparison.winner)
                            .map(|s| s.model_score)
                            .unwrap_or(initial_strength);
                        denom += 1.0 / (stats.model_score + other_score);
                    }
                }

                let new_score = if denom > 0.0 {
                    let raw = wins / denom;
                    // Smooth update with learning rate
                    stats.model_score + learning_rate * (raw - stats.model_score)
                } else {
                    stats.model_score
                };

                new_scores.insert(id, new_score.max(0.001)); // Avoid zero strength
            }

            // Apply new scores
            for (id, score) in new_scores {
                if let Some(stats) = self.candidate_stats.get_mut(&id) {
                    stats.model_score = score;
                }
            }
        }
    }

    /// Process an evaluation response and return updated fitness values
    pub fn process_response(&mut self, response: &EvaluationResponse) -> Vec<(CandidateId, f64)> {
        match response {
            EvaluationResponse::Ratings(ratings) => {
                for (id, rating) in ratings {
                    self.record_rating(*id, *rating);
                }
                ratings
                    .iter()
                    .filter_map(|(id, _)| self.get_fitness(id).map(|f| (*id, f)))
                    .collect()
            }
            EvaluationResponse::PairwiseWinner(Some(winner)) => {
                // We need both IDs to record a comparison
                // For now, just return the winner's fitness
                self.ensure_stats(*winner);
                if let Some(f) = self.get_fitness(winner) {
                    vec![(*winner, f)]
                } else {
                    vec![]
                }
            }
            EvaluationResponse::PairwiseWinner(None) => {
                // Tie - nothing to update without both IDs
                vec![]
            }
            EvaluationResponse::BatchSelected(selected) => {
                // Update selection counts
                for id in selected {
                    self.ensure_stats(*id);
                    if let Some(stats) = self.candidate_stats.get_mut(id) {
                        stats.times_selected += 1;
                        if let AggregationModel::ImplicitRanking { selected_bonus, .. } =
                            &self.model
                        {
                            stats.model_score += *selected_bonus;
                        }
                    }
                }
                selected
                    .iter()
                    .filter_map(|id| self.get_fitness(id).map(|f| (*id, f)))
                    .collect()
            }
            EvaluationResponse::Skip => vec![],
        }
    }

    /// Process a pairwise comparison with both candidate IDs
    pub fn process_pairwise(
        &mut self,
        id_a: CandidateId,
        id_b: CandidateId,
        winner: Option<CandidateId>,
    ) -> Vec<(CandidateId, f64)> {
        match winner {
            Some(w) if w == id_a => {
                self.record_comparison(id_a, id_b);
            }
            Some(w) if w == id_b => {
                self.record_comparison(id_b, id_a);
            }
            Some(_) => {
                // Winner ID doesn't match either candidate
            }
            None => {
                self.record_tie(id_a, id_b);
            }
        }

        vec![id_a, id_b]
            .into_iter()
            .filter_map(|id| self.get_fitness(&id).map(|f| (id, f)))
            .collect()
    }

    /// Process batch selection with full context
    pub fn process_batch_selection(
        &mut self,
        all_candidates: &[CandidateId],
        selected: &[CandidateId],
    ) -> Vec<(CandidateId, f64)> {
        let selected_set: std::collections::HashSet<_> = selected.iter().copied().collect();
        let not_selected: Vec<_> = all_candidates
            .iter()
            .copied()
            .filter(|id| !selected_set.contains(id))
            .collect();

        self.record_batch_selection(selected, &not_selected);

        all_candidates
            .iter()
            .filter_map(|id| self.get_fitness(id).map(|f| (*id, f)))
            .collect()
    }

    /// Get all candidate IDs with fitness estimates
    pub fn all_candidates(&self) -> Vec<CandidateId> {
        self.candidate_stats.keys().copied().collect()
    }

    /// Get the number of comparisons recorded
    pub fn comparison_count(&self) -> usize {
        self.comparisons.len()
    }

    /// Clear all recorded data
    pub fn clear(&mut self) {
        self.candidate_stats.clear();
        self.comparisons.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_rating_aggregation() {
        let mut agg = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });

        let id = CandidateId(0);

        // Initially should return default
        agg.ensure_stats(id);
        assert_eq!(agg.get_fitness(&id), Some(5.0));

        // After rating
        agg.record_rating(id, 8.0);
        assert_eq!(agg.get_fitness(&id), Some(8.0));

        // After second rating, should average
        agg.record_rating(id, 6.0);
        assert_eq!(agg.get_fitness(&id), Some(7.0));
    }

    #[test]
    fn test_elo_rating() {
        let mut agg = FitnessAggregator::new(AggregationModel::Elo {
            initial_rating: 1500.0,
            k_factor: 32.0,
        });

        let id_a = CandidateId(0);
        let id_b = CandidateId(1);

        agg.ensure_stats(id_a);
        agg.ensure_stats(id_b);

        // Initial ratings should be equal
        assert_eq!(agg.get_fitness(&id_a), Some(1500.0));
        assert_eq!(agg.get_fitness(&id_b), Some(1500.0));

        // After A beats B
        agg.record_comparison(id_a, id_b);

        let fitness_a = agg.get_fitness(&id_a).unwrap();
        let fitness_b = agg.get_fitness(&id_b).unwrap();

        // Winner should gain rating
        assert!(fitness_a > 1500.0);
        // Loser should lose rating
        assert!(fitness_b < 1500.0);
        // Total rating should be conserved
        assert!((fitness_a + fitness_b - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_elo_draw() {
        let mut agg = FitnessAggregator::new(AggregationModel::Elo {
            initial_rating: 1500.0,
            k_factor: 32.0,
        });

        let id_a = CandidateId(0);
        let id_b = CandidateId(1);

        agg.ensure_stats(id_a);
        agg.ensure_stats(id_b);

        // After tie between equal players, ratings should stay the same
        agg.record_tie(id_a, id_b);

        let fitness_a = agg.get_fitness(&id_a).unwrap();
        let fitness_b = agg.get_fitness(&id_b).unwrap();

        assert!((fitness_a - 1500.0).abs() < 0.01);
        assert!((fitness_b - 1500.0).abs() < 0.01);
    }

    #[test]
    fn test_implicit_ranking() {
        let mut agg = FitnessAggregator::new(AggregationModel::ImplicitRanking {
            selected_bonus: 1.0,
            not_selected_penalty: 0.5,
            base_fitness: 5.0,
        });

        let selected = vec![CandidateId(0), CandidateId(1)];
        let not_selected = vec![CandidateId(2), CandidateId(3)];

        agg.record_batch_selection(&selected, &not_selected);

        // Selected candidates should have bonus
        assert_eq!(agg.get_fitness(&CandidateId(0)), Some(6.0));
        assert_eq!(agg.get_fitness(&CandidateId(1)), Some(6.0));

        // Not selected should have penalty
        assert_eq!(agg.get_fitness(&CandidateId(2)), Some(4.5));
        assert_eq!(agg.get_fitness(&CandidateId(3)), Some(4.5));
    }

    #[test]
    fn test_bradley_terry_simple_recompute() {
        let mut agg = FitnessAggregator::new(AggregationModel::BradleyTerrySimple {
            initial_strength: 1.0,
            learning_rate: 0.5,
            iterations: 10,
        });

        // A beats B multiple times, B beats C
        agg.ensure_stats(CandidateId(0));
        agg.ensure_stats(CandidateId(1));
        agg.ensure_stats(CandidateId(2));

        agg.record_comparison(CandidateId(0), CandidateId(1));
        agg.record_comparison(CandidateId(0), CandidateId(1));
        agg.record_comparison(CandidateId(1), CandidateId(2));

        let fitness = agg.recompute_all();

        // A should have highest strength
        assert!(fitness[&CandidateId(0)] > fitness[&CandidateId(1)]);
        // B should beat C
        assert!(fitness[&CandidateId(1)] > fitness[&CandidateId(2)]);
    }

    #[test]
    fn test_bradley_terry_mle_recompute() {
        use crate::interactive::bradley_terry::BradleyTerryOptimizer;

        let mut agg = FitnessAggregator::new(AggregationModel::BradleyTerry {
            initial_strength: 1.0,
            optimizer: BradleyTerryOptimizer::default(),
        });

        // A beats B multiple times, B beats C
        agg.ensure_stats(CandidateId(0));
        agg.ensure_stats(CandidateId(1));
        agg.ensure_stats(CandidateId(2));

        agg.record_comparison(CandidateId(0), CandidateId(1));
        agg.record_comparison(CandidateId(0), CandidateId(1));
        agg.record_comparison(CandidateId(1), CandidateId(2));

        let fitness = agg.recompute_all();

        // A should have highest strength
        assert!(fitness[&CandidateId(0)] > fitness[&CandidateId(1)]);
        // B should beat C
        assert!(fitness[&CandidateId(1)] > fitness[&CandidateId(2)]);

        // MLE should also provide variance estimates
        let estimate_a = agg.get_fitness_estimate(&CandidateId(0)).unwrap();
        assert!(estimate_a.variance.is_finite());
        assert!(estimate_a.observation_count > 0);
    }

    #[test]
    fn test_fitness_estimate_direct_rating() {
        let mut agg = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });

        let id = CandidateId(0);
        agg.ensure_stats(id);

        // Initially should be uninformative
        let estimate = agg.get_fitness_estimate(&id).unwrap();
        assert_eq!(estimate.mean, 5.0);
        assert!(estimate.variance.is_infinite());

        // After ratings, should have finite variance
        agg.record_rating(id, 8.0);
        agg.record_rating(id, 6.0);
        agg.record_rating(id, 7.0);

        let estimate = agg.get_fitness_estimate(&id).unwrap();
        assert_eq!(estimate.mean, 7.0);
        assert!(estimate.variance.is_finite());
        assert_eq!(estimate.observation_count, 3);
    }

    #[test]
    fn test_candidate_stats() {
        let mut stats = CandidateStats::new(1500.0);

        // Test rating tracking
        stats.rating_sum = 24.0;
        stats.rating_count = 3;
        assert_eq!(stats.average_rating(), Some(8.0));

        // Test win rate
        stats.wins = 3;
        stats.losses = 1;
        assert_eq!(stats.total_comparisons(), 4);
        assert_eq!(stats.win_rate(), Some(0.75));

        // Test selection rate
        stats.times_selected = 2;
        stats.times_passed = 3;
        assert_eq!(stats.selection_rate(), Some(0.4));
    }

    #[test]
    fn test_process_response_ratings() {
        let mut agg = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });

        let response =
            EvaluationResponse::ratings(vec![(CandidateId(0), 8.0), (CandidateId(1), 6.0)]);

        let updated = agg.process_response(&response);

        assert_eq!(updated.len(), 2);
        assert!(updated
            .iter()
            .any(|(id, f)| *id == CandidateId(0) && *f == 8.0));
        assert!(updated
            .iter()
            .any(|(id, f)| *id == CandidateId(1) && *f == 6.0));
    }

    #[test]
    fn test_process_batch_selection() {
        let mut agg = FitnessAggregator::new(AggregationModel::ImplicitRanking {
            selected_bonus: 1.0,
            not_selected_penalty: 0.5,
            base_fitness: 5.0,
        });

        let all = vec![
            CandidateId(0),
            CandidateId(1),
            CandidateId(2),
            CandidateId(3),
        ];
        let selected = vec![CandidateId(0), CandidateId(2)];

        let updated = agg.process_batch_selection(&all, &selected);

        assert_eq!(updated.len(), 4);

        // Check selected got bonus
        let fitness_0 = updated
            .iter()
            .find(|(id, _)| *id == CandidateId(0))
            .unwrap()
            .1;
        assert_eq!(fitness_0, 6.0);

        // Check not selected got penalty
        let fitness_1 = updated
            .iter()
            .find(|(id, _)| *id == CandidateId(1))
            .unwrap()
            .1;
        assert_eq!(fitness_1, 4.5);
    }
}
