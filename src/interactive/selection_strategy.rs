//! Active learning strategies for intelligent candidate selection
//!
//! This module provides strategies for selecting which candidates to present
//! to users for evaluation. Instead of random or sequential selection,
//! active learning strategies prioritize candidates that will provide the
//! most useful information for ranking.
//!
//! # Available Strategies
//!
//! - **Sequential**: Default behavior - simple sequential/round-robin selection
//! - **UncertaintySampling**: Prioritize candidates with highest uncertainty
//! - **ExpectedInformationGain**: Select pairs that maximize information gain
//! - **CoverageAware**: Balance coverage requirements with exploration
//!
//! # Example
//!
//! ```rust,ignore
//! use fugue_evo::interactive::selection_strategy::SelectionStrategy;
//!
//! // Use uncertainty sampling with coverage bonus
//! let strategy = SelectionStrategy::UncertaintySampling {
//!     uncertainty_weight: 1.0,
//! };
//!
//! let selected = strategy.select_batch(&candidates, &aggregator, 4);
//! ```

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::aggregation::FitnessAggregator;
use super::evaluator::Candidate;
use super::uncertainty::FitnessEstimate;
use crate::genome::traits::EvolutionaryGenome;

/// Active learning selection strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Sequential selection (current default behavior)
    ///
    /// Selects candidates in order of their index, cycling through
    /// the population. Simple but not optimal for learning.
    Sequential,

    /// Uncertainty sampling
    ///
    /// Prioritizes candidates with the highest uncertainty (variance)
    /// in their fitness estimates. This helps reduce overall uncertainty
    /// in the ranking.
    UncertaintySampling {
        /// Weight for uncertainty vs coverage balance (default: 1.0)
        /// Higher values prioritize uncertain candidates more strongly
        uncertainty_weight: f64,
    },

    /// Expected information gain (for pairwise comparisons)
    ///
    /// Selects pairs of candidates where the comparison result is
    /// most uncertain (probability close to 0.5). This maximizes
    /// the expected reduction in entropy.
    ExpectedInformationGain {
        /// Temperature for softmax selection (default: 1.0)
        /// Higher values make selection more random
        temperature: f64,
    },

    /// Coverage-aware selection
    ///
    /// Ensures minimum coverage before exploring uncertain candidates.
    /// Good for balancing exploration with ensuring all candidates
    /// are evaluated at least some minimum number of times.
    CoverageAware {
        /// Minimum evaluations before considering a candidate "covered"
        min_evaluations: usize,
        /// Bonus weight for under-evaluated candidates
        exploration_bonus: f64,
    },
}

impl Default for SelectionStrategy {
    fn default() -> Self {
        Self::Sequential
    }
}

impl SelectionStrategy {
    /// Create uncertainty sampling strategy
    pub fn uncertainty_sampling(uncertainty_weight: f64) -> Self {
        Self::UncertaintySampling { uncertainty_weight }
    }

    /// Create expected information gain strategy
    pub fn information_gain(temperature: f64) -> Self {
        Self::ExpectedInformationGain { temperature }
    }

    /// Create coverage-aware strategy
    pub fn coverage_aware(min_evaluations: usize, exploration_bonus: f64) -> Self {
        Self::CoverageAware {
            min_evaluations,
            exploration_bonus,
        }
    }

    /// Select a batch of candidates for evaluation
    ///
    /// # Arguments
    ///
    /// * `candidates` - All candidates in the population
    /// * `aggregator` - Fitness aggregator with current estimates
    /// * `batch_size` - Number of candidates to select
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Indices of selected candidates (into the candidates slice)
    pub fn select_batch<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        batch_size: usize,
        rng: &mut R,
    ) -> Vec<usize>
    where
        G: EvolutionaryGenome,
        R: Rng,
    {
        if candidates.is_empty() || batch_size == 0 {
            return vec![];
        }

        let batch_size = batch_size.min(candidates.len());

        match self {
            Self::Sequential => self.select_sequential(candidates, batch_size),
            Self::UncertaintySampling { uncertainty_weight } => {
                self.select_by_uncertainty(candidates, aggregator, batch_size, *uncertainty_weight)
            }
            Self::ExpectedInformationGain { temperature } => self.select_by_information_gain(
                candidates,
                aggregator,
                batch_size,
                *temperature,
                rng,
            ),
            Self::CoverageAware {
                min_evaluations,
                exploration_bonus,
            } => self.select_coverage_aware(
                candidates,
                aggregator,
                batch_size,
                *min_evaluations,
                *exploration_bonus,
            ),
        }
    }

    /// Select a pair for pairwise comparison
    ///
    /// # Arguments
    ///
    /// * `candidates` - All candidates in the population
    /// * `aggregator` - Fitness aggregator with current estimates
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of indices for the two candidates to compare
    pub fn select_pair<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        rng: &mut R,
    ) -> Option<(usize, usize)>
    where
        G: EvolutionaryGenome,
        R: Rng,
    {
        if candidates.len() < 2 {
            return None;
        }

        match self {
            Self::Sequential => {
                // Simple sequential pairing
                Some((0, 1))
            }
            Self::UncertaintySampling { .. } => {
                // Select two most uncertain candidates
                let scores = self.compute_uncertainty_scores(candidates, aggregator);
                let mut indices: Vec<usize> = (0..candidates.len()).collect();
                indices.sort_by(|&a, &b| {
                    scores[b]
                        .partial_cmp(&scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Some((indices[0], indices[1]))
            }
            Self::ExpectedInformationGain { temperature } => {
                self.select_pair_by_information_gain(candidates, aggregator, *temperature, rng)
            }
            Self::CoverageAware {
                min_evaluations, ..
            } => {
                // Pair candidates with fewest evaluations
                let mut indices: Vec<(usize, usize)> = candidates
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.evaluation_count))
                    .collect();
                indices.sort_by_key(|&(_, count)| count);

                let a = indices[0].0;
                let b = if indices.len() > 1 {
                    // Find candidate with fewest evaluations that's also "close" in ranking
                    let a_eval = candidates[a].evaluation_count;
                    if a_eval < *min_evaluations {
                        // First pass: just pick two under-evaluated
                        indices[1].0
                    } else {
                        // Pick most informative pair among adequately covered
                        self.find_informative_pair(candidates, aggregator, rng)
                    }
                } else {
                    return None;
                };
                Some((a, b))
            }
        }
    }

    /// Sequential selection - first N unevaluated, then first N overall
    fn select_sequential<G>(&self, candidates: &[Candidate<G>], batch_size: usize) -> Vec<usize>
    where
        G: EvolutionaryGenome,
    {
        // First, select unevaluated candidates
        let mut selected: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.evaluation_count == 0)
            .take(batch_size)
            .map(|(i, _)| i)
            .collect();

        // If need more, add from beginning
        if selected.len() < batch_size {
            for i in 0..candidates.len() {
                if selected.len() >= batch_size {
                    break;
                }
                if !selected.contains(&i) {
                    selected.push(i);
                }
            }
        }

        selected
    }

    /// Compute uncertainty scores for all candidates
    fn compute_uncertainty_scores<G>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
    ) -> Vec<f64>
    where
        G: EvolutionaryGenome,
    {
        candidates
            .iter()
            .map(|c| {
                aggregator
                    .get_fitness_estimate(&c.id)
                    .map(|e| {
                        if e.variance.is_infinite() {
                            f64::MAX // Highest priority for unobserved
                        } else {
                            e.variance
                        }
                    })
                    .unwrap_or(f64::MAX)
            })
            .collect()
    }

    /// Select by uncertainty (highest variance first)
    fn select_by_uncertainty<G>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        batch_size: usize,
        uncertainty_weight: f64,
    ) -> Vec<usize>
    where
        G: EvolutionaryGenome,
    {
        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let uncertainty = aggregator
                    .get_fitness_estimate(&c.id)
                    .map(|e| {
                        if e.variance.is_infinite() {
                            f64::MAX
                        } else {
                            e.variance
                        }
                    })
                    .unwrap_or(f64::MAX);

                // Bonus for fewer evaluations
                let coverage_bonus = 1.0 / (c.evaluation_count as f64 + 1.0);

                let score = uncertainty_weight * uncertainty + coverage_bonus;
                (i, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(batch_size)
            .map(|(i, _)| i)
            .collect()
    }

    /// Select by expected information gain
    fn select_by_information_gain<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        batch_size: usize,
        temperature: f64,
        rng: &mut R,
    ) -> Vec<usize>
    where
        G: EvolutionaryGenome,
        R: Rng,
    {
        // For batch selection, use a simplified approach:
        // Score candidates by how uncertain their ranking position is
        let estimates: Vec<Option<FitnessEstimate>> = candidates
            .iter()
            .map(|c| aggregator.get_fitness_estimate(&c.id))
            .collect();

        // Score each candidate by entropy of pairwise comparisons with others
        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let my_est = &estimates[i];
                let score = estimates
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other_est)| pairwise_entropy(my_est.as_ref(), other_est.as_ref()))
                    .sum::<f64>();
                (i, score)
            })
            .collect();

        if temperature > 0.0 {
            // Softmax sampling
            let max_score = scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores
                .iter()
                .map(|(_, s)| ((s - max_score) / temperature).exp())
                .collect();
            let total: f64 = weights.iter().sum();

            let mut selected = Vec::with_capacity(batch_size);
            let mut remaining: Vec<(usize, f64)> = scores
                .iter()
                .zip(weights.iter())
                .map(|((i, _), w)| (*i, *w / total))
                .collect();

            for _ in 0..batch_size {
                if remaining.is_empty() {
                    break;
                }

                let r: f64 = rng.gen();
                let mut cumsum = 0.0;
                let mut chosen_idx = 0;

                for (idx, (_, w)) in remaining.iter().enumerate() {
                    cumsum += w;
                    if r < cumsum {
                        chosen_idx = idx;
                        break;
                    }
                }

                let (i, _) = remaining.remove(chosen_idx);
                selected.push(i);

                // Renormalize remaining weights
                let new_total: f64 = remaining.iter().map(|(_, w)| w).sum();
                if new_total > 0.0 {
                    for (_, w) in &mut remaining {
                        *w /= new_total;
                    }
                }
            }

            selected
        } else {
            // Deterministic: take top scorers
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores
                .into_iter()
                .take(batch_size)
                .map(|(i, _)| i)
                .collect()
        }
    }

    /// Select pair by information gain (for pairwise comparison mode)
    fn select_pair_by_information_gain<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        temperature: f64,
        rng: &mut R,
    ) -> Option<(usize, usize)>
    where
        G: EvolutionaryGenome,
        R: Rng,
    {
        let n = candidates.len();
        if n < 2 {
            return None;
        }

        let estimates: Vec<Option<FitnessEstimate>> = candidates
            .iter()
            .map(|c| aggregator.get_fitness_estimate(&c.id))
            .collect();

        // Compute information gain for each pair
        let mut pair_scores: Vec<((usize, usize), f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let entropy = pairwise_entropy(estimates[i].as_ref(), estimates[j].as_ref());
                pair_scores.push(((i, j), entropy));
            }
        }

        if pair_scores.is_empty() {
            return Some((0, 1));
        }

        if temperature > 0.0 {
            // Softmax selection
            let max_score = pair_scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = pair_scores
                .iter()
                .map(|(_, s)| ((s - max_score) / temperature).exp())
                .collect();
            let total: f64 = weights.iter().sum();

            let r: f64 = rng.gen();
            let mut cumsum = 0.0;

            for ((pair, _), w) in pair_scores.iter().zip(weights.iter()) {
                cumsum += w / total;
                if r < cumsum {
                    return Some(*pair);
                }
            }
        }

        // Return highest scoring pair
        pair_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(pair_scores[0].0)
    }

    /// Coverage-aware selection
    fn select_coverage_aware<G>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        batch_size: usize,
        min_evaluations: usize,
        exploration_bonus: f64,
    ) -> Vec<usize>
    where
        G: EvolutionaryGenome,
    {
        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let score = if c.evaluation_count < min_evaluations {
                    // Must evaluate - infinite priority
                    f64::MAX
                } else {
                    // Base uncertainty
                    let uncertainty = aggregator
                        .get_fitness_estimate(&c.id)
                        .map(|e| {
                            if e.variance.is_infinite() {
                                1e6
                            } else {
                                e.variance
                            }
                        })
                        .unwrap_or(1e6);

                    // Exploration bonus for fewer evaluations
                    let bonus = exploration_bonus / (c.evaluation_count as f64 + 1.0);

                    uncertainty + bonus
                };
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(batch_size)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find an informative pair among adequately covered candidates
    fn find_informative_pair<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        rng: &mut R,
    ) -> usize
    where
        G: EvolutionaryGenome,
        R: Rng,
    {
        // Find candidate whose ranking is most uncertain relative to others
        let estimates: Vec<Option<FitnessEstimate>> = candidates
            .iter()
            .map(|c| aggregator.get_fitness_estimate(&c.id))
            .collect();

        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let score = estimates
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other)| pairwise_entropy(estimates[i].as_ref(), other.as_ref()))
                    .sum::<f64>();
                (i, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Add some randomness to avoid always picking the same
        let top_k = 3.min(scores.len());
        let chosen = rng.gen_range(0..top_k);
        scores[chosen].0
    }
}

/// Compute entropy of pairwise comparison outcome
///
/// Entropy is maximized when P(A beats B) = 0.5 (most uncertain)
fn pairwise_entropy(a: Option<&FitnessEstimate>, b: Option<&FitnessEstimate>) -> f64 {
    match (a, b) {
        (Some(est_a), Some(est_b)) => {
            // Approximate P(A beats B) using normal approximation
            let mean_diff = est_a.mean - est_b.mean;
            let var_diff = est_a.variance + est_b.variance;

            if var_diff.is_infinite() || var_diff <= 0.0 {
                // Maximum entropy when we know nothing
                return 1.0;
            }

            // P(A > B) ≈ Φ((μ_A - μ_B) / sqrt(σ²_A + σ²_B))
            let z = mean_diff / var_diff.sqrt();
            let p = normal_cdf(z);

            // Binary entropy: -p*log(p) - (1-p)*log(1-p)
            binary_entropy(p)
        }
        _ => 1.0, // Maximum entropy for unobserved
    }
}

/// Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
fn binary_entropy(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::interactive::aggregation::{AggregationModel, FitnessAggregator};
    use crate::interactive::evaluator::CandidateId;

    fn make_candidates(n: usize) -> Vec<Candidate<RealVector>> {
        (0..n)
            .map(|i| {
                let mut c = Candidate::new(CandidateId(i), RealVector::new(vec![i as f64]));
                c.evaluation_count = 0;
                c
            })
            .collect()
    }

    #[test]
    fn test_sequential_selection() {
        let candidates = make_candidates(10);
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::Sequential;
        let selected = strategy.select_batch(&candidates, &aggregator, 3, &mut rng);

        assert_eq!(selected.len(), 3);
        // Should select first 3 (all unevaluated)
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_uncertainty_sampling() {
        let mut candidates = make_candidates(5);
        let mut aggregator = FitnessAggregator::new(AggregationModel::DirectRating {
            default_rating: 5.0,
        });
        let mut rng = rand::thread_rng();

        // Give candidate 0 multiple identical ratings (low variance)
        aggregator.record_rating(CandidateId(0), 7.0);
        aggregator.record_rating(CandidateId(0), 7.0);
        aggregator.record_rating(CandidateId(0), 7.0);
        candidates[0].evaluation_count = 3;

        // Give candidate 1 multiple varied ratings (medium variance)
        aggregator.record_rating(CandidateId(1), 4.0);
        aggregator.record_rating(CandidateId(1), 8.0);
        candidates[1].evaluation_count = 2;

        // Candidates 2, 3, 4 are unevaluated (highest uncertainty)

        let strategy = SelectionStrategy::UncertaintySampling {
            uncertainty_weight: 1.0,
        };
        let selected = strategy.select_batch(&candidates, &aggregator, 2, &mut rng);

        // Should select high-uncertainty candidates, NOT the well-evaluated candidate 0
        assert_eq!(selected.len(), 2);
        for &idx in &selected {
            assert!(
                idx != 0,
                "Should not select the well-evaluated candidate with low variance"
            );
        }
    }

    #[test]
    fn test_coverage_aware() {
        let mut candidates = make_candidates(5);
        candidates[0].evaluation_count = 3;
        candidates[1].evaluation_count = 2;
        candidates[2].evaluation_count = 0; // Under min
        candidates[3].evaluation_count = 0; // Under min
        candidates[4].evaluation_count = 1;

        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::CoverageAware {
            min_evaluations: 2,
            exploration_bonus: 1.0,
        };
        let selected = strategy.select_batch(&candidates, &aggregator, 2, &mut rng);

        // Should prioritize candidates 2 and 3 (under min coverage)
        assert!(selected.contains(&2) || selected.contains(&3));
    }

    #[test]
    fn test_select_pair_sequential() {
        let candidates = make_candidates(5);
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::Sequential;
        let pair = strategy.select_pair(&candidates, &aggregator, &mut rng);

        assert!(pair.is_some());
        let (a, b) = pair.unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_select_pair_info_gain() {
        let candidates = make_candidates(5);
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::ExpectedInformationGain { temperature: 1.0 };
        let pair = strategy.select_pair(&candidates, &aggregator, &mut rng);

        assert!(pair.is_some());
        let (a, b) = pair.unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_binary_entropy() {
        // Max entropy at p = 0.5
        let max_entropy = binary_entropy(0.5);
        assert!((max_entropy - std::f64::consts::LN_2).abs() < 1e-6);

        // Zero entropy at p = 0 or 1
        assert!(binary_entropy(0.001) < 0.1);
        assert!(binary_entropy(0.999) < 0.1);
    }

    #[test]
    fn test_normal_cdf() {
        // CDF(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);

        // CDF(-∞) → 0, CDF(+∞) → 1
        assert!(normal_cdf(-10.0) < 0.001);
        assert!(normal_cdf(10.0) > 0.999);

        // Symmetry
        assert!((normal_cdf(1.0) + normal_cdf(-1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_candidates() {
        let candidates: Vec<Candidate<RealVector>> = vec![];
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::default();
        let selected = strategy.select_batch(&candidates, &aggregator, 5, &mut rng);
        assert!(selected.is_empty());

        let pair = strategy.select_pair(&candidates, &aggregator, &mut rng);
        assert!(pair.is_none());
    }

    #[test]
    fn test_single_candidate() {
        let candidates = make_candidates(1);
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();

        let strategy = SelectionStrategy::default();
        let selected = strategy.select_batch(&candidates, &aggregator, 5, &mut rng);
        assert_eq!(selected.len(), 1);

        let pair = strategy.select_pair(&candidates, &aggregator, &mut rng);
        assert!(pair.is_none()); // Can't make a pair from 1 candidate
    }
}
