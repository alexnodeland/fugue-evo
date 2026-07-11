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
                        // Pick most informative partner among adequately covered,
                        // excluding `a` so we can never return a self-pair (EV-68).
                        self.find_informative_pair(candidates, aggregator, Some(a), rng)
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
        // Normalize by the batch's mean variance so the coverage bonus is on a
        // comparable, model-agnostic scale (EV-69).
        let variances: Vec<f64> = candidates
            .iter()
            .map(|c| {
                aggregator
                    .get_fitness_estimate(&c.id)
                    .map(|e| e.variance)
                    .unwrap_or(f64::INFINITY)
            })
            .collect();
        let var_scale = mean_variance_scale(&variances);

        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let score = normalized_uncertainty_score(
                    variances[i],
                    c.evaluation_count,
                    var_scale,
                    uncertainty_weight,
                    1.0,
                );
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
                let weights_now: Vec<f64> = remaining.iter().map(|(_, w)| *w).collect();
                let chosen_idx = inverse_cdf_pick(&weights_now, r);

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

            // Same inverse-CDF sampler as the batch path, with the last-index
            // residual fallback (EV-100) so the two paths behave consistently.
            let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();
            let r: f64 = rng.gen();
            let idx = inverse_cdf_pick(&normalized, r);
            return Some(pair_scores[idx].0);
        }

        // Deterministic (temperature <= 0): return highest scoring pair.
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
        // Normalize by the batch's mean variance so the exploration bonus is on a
        // comparable, model-agnostic scale (EV-69).
        let variances: Vec<f64> = candidates
            .iter()
            .map(|c| {
                aggregator
                    .get_fitness_estimate(&c.id)
                    .map(|e| e.variance)
                    .unwrap_or(f64::INFINITY)
            })
            .collect();
        let var_scale = mean_variance_scale(&variances);

        let mut scores: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let score = if c.evaluation_count < min_evaluations {
                    // Must evaluate - infinite priority (ranks above any covered
                    // candidate, whose normalized score is <= UNOBSERVED + bonus).
                    f64::MAX
                } else {
                    normalized_uncertainty_score(
                        variances[i],
                        c.evaluation_count,
                        var_scale,
                        1.0,
                        exploration_bonus,
                    )
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

    /// Find an informative partner among adequately covered candidates.
    ///
    /// When `exclude` is `Some(a)`, index `a` is never returned, so callers that
    /// have already committed to `a` cannot receive a self-pair `(a, a)` (EV-68).
    fn find_informative_pair<G, R>(
        &self,
        candidates: &[Candidate<G>],
        aggregator: &FitnessAggregator,
        exclude: Option<usize>,
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
            .filter(|(i, _)| Some(*i) != exclude)
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

        // With `exclude` set and >= 2 candidates the caller guarantees at least
        // one remaining index; fall back to the excluded index only if somehow
        // nothing else exists (never on the reachable path).
        if scores.is_empty() {
            return exclude.unwrap_or(0);
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Add some randomness to avoid always picking the same
        let top_k = 3.min(scores.len());
        let chosen = rng.gen_range(0..top_k);
        scores[chosen].0
    }
}

/// Inverse-CDF sample over a (normalized) weight vector.
///
/// Returns the first index whose cumulative weight exceeds `r`. If floating-point
/// round-off leaves the cumulative sum just below `r` so that no prefix qualifies,
/// it falls back to the **last** index — the standard inverse-CDF residual choice
/// (EV-100) — instead of the old behavior of defaulting to index 0, which biased
/// selection toward the first remaining candidate.
fn inverse_cdf_pick(weights: &[f64], r: f64) -> usize {
    let mut cumsum = 0.0;
    for (idx, w) in weights.iter().enumerate() {
        cumsum += w;
        if r < cumsum {
            return idx;
        }
    }
    weights.len().saturating_sub(1)
}

/// Score magnitude assigned to an unobserved (infinite-variance) candidate after
/// normalization.
///
/// Large enough to dominate any normalized *finite* uncertainty (which is ~O(1)
/// after dividing by the mean variance), so unobserved candidates keep top
/// priority, but finite so the coverage bonus can still order equally-unobserved
/// candidates and no `f64::MAX + bonus` overflow occurs.
const UNOBSERVED_NORMALIZED_UNCERTAINTY: f64 = 1e6;

/// Mean of the finite, positive variances in `variances` (EV-69).
///
/// Used to put raw model variances on a comparable, model-agnostic scale before
/// an exploration/coverage bonus is added. Returns `1.0` when nothing finite is
/// available to average, leaving the bonus as the sole differentiator.
fn mean_variance_scale(variances: &[f64]) -> f64 {
    let (sum, count) = variances
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
    if count == 0 {
        1.0
    } else {
        sum / count as f64
    }
}

/// Combine a candidate's raw variance and evaluation count into an acquisition
/// score with a model-agnostic scale (EV-69).
///
/// The variance is divided by the batch's mean variance (`var_scale`) so the
/// additive exploration/coverage bonus has a comparable influence regardless of
/// the aggregation model's native variance magnitude (DirectRating ~O(1), Elo
/// ~O(10²), ImplicitRanking ~O(10⁻²)). Previously the bonus was added to the raw
/// variance and was therefore either inert or dominant depending on the model.
fn normalized_uncertainty_score(
    variance: f64,
    eval_count: usize,
    var_scale: f64,
    uncertainty_weight: f64,
    bonus_coeff: f64,
) -> f64 {
    let normalized = if variance.is_finite() {
        variance / var_scale
    } else {
        UNOBSERVED_NORMALIZED_UNCERTAINTY
    };
    let bonus = bonus_coeff / (eval_count as f64 + 1.0);
    uncertainty_weight * normalized + bonus
}

/// Maximum binary entropy, in **nats** (`ln 2`).
///
/// This is the sentinel returned for pairs whose comparison outcome is entirely
/// unknown (an unobserved candidate / infinite variance). Keeping it equal to
/// the true maximum of [`binary_entropy`] — rather than the mismatched `1.0`
/// (which is 1 *bit*, not 1 nat) — means unobserved and observed pair scores are
/// on the same scale (EV-99). Unobserved pairs still get priority through the
/// coverage/exploration terms of the selection strategies, not through an
/// inflated entropy.
const MAX_BINARY_ENTROPY_NATS: f64 = std::f64::consts::LN_2;

/// Compute the entropy (in nats) of a pairwise comparison outcome.
///
/// Entropy is maximized (`ln 2`) when `P(A beats B) = 0.5` (most uncertain) and
/// approaches `0` as the outcome becomes determined.
fn pairwise_entropy(a: Option<&FitnessEstimate>, b: Option<&FitnessEstimate>) -> f64 {
    match (a, b) {
        (Some(est_a), Some(est_b)) => {
            let mean_diff = est_a.mean - est_b.mean;
            let var_diff = est_a.variance + est_b.variance;

            if var_diff.is_infinite() {
                // At least one estimate is completely unobserved -> outcome
                // genuinely unknown, so report the maximum entropy sentinel.
                return MAX_BINARY_ENTROPY_NATS;
            }

            if var_diff <= 0.0 {
                // Both candidates are perfectly measured (EV-70). The outcome is
                // then fully DETERMINED by the means: entropy ~0 unless the means
                // also tie (a genuine coin flip). Returning the max here — as the
                // old code did — wrongly made the strategy spend comparisons on
                // pairs it is already certain about.
                return if mean_diff.abs() < f64::EPSILON {
                    MAX_BINARY_ENTROPY_NATS
                } else {
                    0.0
                };
            }

            // P(A > B) ≈ Φ((μ_A - μ_B) / sqrt(σ²_A + σ²_B))
            let z = mean_diff / var_diff.sqrt();
            let p = normal_cdf(z);

            binary_entropy(p)
        }
        _ => MAX_BINARY_ENTROPY_NATS, // Unobserved pair
    }
}

/// Binary entropy in **nats**: `H(p) = -p·ln(p) - (1-p)·ln(1-p)`.
///
/// Maximized at `p = 0.5` with value `ln 2 ≈ 0.6931` nats (not `1.0`, which
/// would be 1 bit / log-base-2). See [`MAX_BINARY_ENTROPY_NATS`].
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

    #[test]
    fn test_inverse_cdf_pick_fallback_is_last() {
        // regression: EV-100 — when the cumulative weight never reaches `r`
        // (floating-point residual), the fallback must be the LAST index, not 0.
        let weights = vec![0.3, 0.3, 0.3]; // sums to 0.9 < 1.0
        assert_eq!(inverse_cdf_pick(&weights, 0.95), 2); // pre-fix returned 0
                                                         // Normal inverse-CDF behavior still holds.
        assert_eq!(inverse_cdf_pick(&weights, 0.1), 0);
        assert_eq!(inverse_cdf_pick(&weights, 0.4), 1);
        assert_eq!(inverse_cdf_pick(&weights, 0.7), 2);
    }

    #[test]
    fn test_entropy_sentinel_is_nats() {
        // regression: EV-99 — the unobserved/degenerate sentinel must be the max
        // binary entropy in NATS (ln 2), matching binary_entropy's maximum, not 1.0.
        assert!((MAX_BINARY_ENTROPY_NATS - std::f64::consts::LN_2).abs() < 1e-12);
        let unobserved = pairwise_entropy(None, None);
        assert!((unobserved - binary_entropy(0.5)).abs() < 1e-9);
        assert!((unobserved - std::f64::consts::LN_2).abs() < 1e-9);
        assert!(unobserved < 1.0); // ln 2 = 0.693..., not the old 1.0 (bit)
    }

    #[test]
    fn test_pairwise_entropy_known_below_unknown() {
        // regression: EV-70 — a pair of perfectly-known candidates with different
        // means has a DETERMINED outcome (entropy ~0) and must score BELOW an
        // unobserved pair (max entropy). The old code returned max entropy for the
        // zero-variance case, inverting active-learning priority.
        let known_a = FitnessEstimate::new(9.0, 0.0, 100);
        let known_b = FitnessEstimate::new(1.0, 0.0, 100);
        let known = pairwise_entropy(Some(&known_a), Some(&known_b));

        let unknown_a = FitnessEstimate::uninformative(5.0); // infinite variance
        let unknown_b = FitnessEstimate::uninformative(5.0);
        let unknown = pairwise_entropy(Some(&unknown_a), Some(&unknown_b));

        assert!(
            known < unknown,
            "known {known} should score below unknown {unknown}"
        );
        assert!(known < 1e-6, "determined outcome should be ~0, got {known}");
        // Degenerate: zero variance AND equal means -> genuine coin flip -> max.
        let tie_a = FitnessEstimate::new(5.0, 0.0, 100);
        let tie_b = FitnessEstimate::new(5.0, 0.0, 100);
        assert!(
            (pairwise_entropy(Some(&tie_a), Some(&tie_b)) - std::f64::consts::LN_2).abs() < 1e-9
        );
    }

    #[test]
    fn test_coverage_aware_never_returns_self_pair() {
        // regression: EV-68 — CoverageAware select_pair must never return (a, a),
        // even when the least-evaluated candidate is also the most informative.
        let mut candidates = make_candidates(2);
        candidates[0].evaluation_count = 2;
        candidates[1].evaluation_count = 2; // both >= min_evaluations
        let aggregator = FitnessAggregator::new(AggregationModel::default());
        let mut rng = rand::thread_rng();
        let strategy = SelectionStrategy::CoverageAware {
            min_evaluations: 1,
            exploration_bonus: 1.0,
        };
        for _ in 0..200 {
            let (a, b) = strategy
                .select_pair(&candidates, &aggregator, &mut rng)
                .unwrap();
            assert_ne!(a, b, "select_pair returned a self-pair");
        }
    }

    #[test]
    fn test_normalized_uncertainty_scale_invariant() {
        // regression: EV-69 — the exploration/coverage bonus must have a
        // model-agnostic influence: scaling ALL variances by a constant must not
        // change the ranking. Un-normalized scoring flips the ranking instead.
        let counts = [1usize, 100usize];
        let variances = [1.0, 1.05];
        let var_scale = mean_variance_scale(&variances);
        let s_a = normalized_uncertainty_score(variances[0], counts[0], var_scale, 1.0, 1.0);
        let s_b = normalized_uncertainty_score(variances[1], counts[1], var_scale, 1.0, 1.0);

        let variances_big: Vec<f64> = variances.iter().map(|v| v * 100.0).collect();
        let scale_big = mean_variance_scale(&variances_big);
        let s_a_big =
            normalized_uncertainty_score(variances_big[0], counts[0], scale_big, 1.0, 1.0);
        let s_b_big =
            normalized_uncertainty_score(variances_big[1], counts[1], scale_big, 1.0, 1.0);

        // Ranking preserved across variance scales.
        assert_eq!(s_a > s_b, s_a_big > s_b_big);
        assert!(
            s_a > s_b,
            "the low-count candidate should win via the bonus"
        );

        // Contrast: raw variance + fixed bonus flips the ranking under scaling.
        let raw_a = variances[0] + 1.0 / (counts[0] as f64 + 1.0);
        let raw_b = variances[1] + 1.0 / (counts[1] as f64 + 1.0);
        let raw_a_big = variances_big[0] + 1.0 / (counts[0] as f64 + 1.0);
        let raw_b_big = variances_big[1] + 1.0 / (counts[1] as f64 + 1.0);
        assert!(raw_a > raw_b); // A wins at small scale...
        assert!(raw_a_big < raw_b_big); // ...but B wins at large scale (the bug).
    }
}
