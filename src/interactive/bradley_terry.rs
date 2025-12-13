//! Bradley-Terry model implementation with Maximum Likelihood Estimation
//!
//! This module provides proper MLE-based Bradley-Terry model fitting with two
//! optimization algorithms:
//!
//! - **Newton-Raphson**: Fast convergence, provides Fisher Information for uncertainty
//! - **MM (Minorization-Maximization)**: Simple, guaranteed convergence, uses bootstrap for uncertainty
//!
//! # Bradley-Terry Model
//!
//! The Bradley-Terry model estimates the probability that candidate i beats candidate j as:
//!
//! ```text
//! P(i beats j) = π_i / (π_i + π_j)
//! ```
//!
//! where π_i is the "strength" parameter for candidate i.
//!
//! # Example
//!
//! ```rust,ignore
//! use fugue_evo::interactive::bradley_terry::{BradleyTerryModel, BradleyTerryOptimizer};
//!
//! let comparisons = vec![
//!     ComparisonRecord { winner: CandidateId(0), loser: CandidateId(1), generation: 0 },
//!     ComparisonRecord { winner: CandidateId(0), loser: CandidateId(2), generation: 0 },
//! ];
//!
//! let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
//! let result = model.fit(&comparisons, &candidate_ids);
//!
//! let estimate = result.get_estimate(CandidateId(0));
//! println!("Strength: {:.2} ± {:.2}", estimate.mean, estimate.std_error());
//! ```

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::aggregation::ComparisonRecord;
use super::evaluator::CandidateId;
use super::uncertainty::FitnessEstimate;

/// Internal context for fitting operations
///
/// Groups common parameters for fit operations to reduce function argument count.
struct FitContext<'a> {
    comparisons: &'a [ComparisonRecord],
    candidate_ids: &'a [CandidateId],
    id_to_index: HashMap<CandidateId, usize>,
    n: usize,
}

impl<'a> FitContext<'a> {
    fn new(comparisons: &'a [ComparisonRecord], candidate_ids: &'a [CandidateId]) -> Self {
        let id_to_index: HashMap<CandidateId, usize> = candidate_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        let n = candidate_ids.len();
        Self {
            comparisons,
            candidate_ids,
            id_to_index,
            n,
        }
    }
}

/// Bradley-Terry optimizer configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BradleyTerryOptimizer {
    /// Newton-Raphson optimization with Fisher Information for uncertainty
    ///
    /// Faster convergence, provides analytical covariance matrix from
    /// the inverse Fisher Information (negative Hessian).
    NewtonRaphson {
        /// Maximum iterations (default: 100)
        max_iterations: usize,
        /// Convergence tolerance for gradient norm (default: 1e-8)
        tolerance: f64,
        /// L2 regularization for Hessian stability (default: 1e-6)
        regularization: f64,
    },

    /// MM (Minorization-Maximization) algorithm with bootstrap for uncertainty
    ///
    /// Simpler, guaranteed monotonic likelihood increase, uses bootstrap
    /// resampling to estimate variance.
    MM {
        /// Maximum iterations (default: 100)
        max_iterations: usize,
        /// Convergence tolerance for parameter change (default: 1e-8)
        tolerance: f64,
        /// Number of bootstrap samples for variance estimation (default: 100)
        bootstrap_samples: usize,
    },
}

impl Default for BradleyTerryOptimizer {
    fn default() -> Self {
        Self::NewtonRaphson {
            max_iterations: 100,
            tolerance: 1e-6, // Relaxed for better convergence on small datasets
            regularization: 1e-6,
        }
    }
}

impl BradleyTerryOptimizer {
    /// Create Newton-Raphson optimizer with custom parameters
    pub fn newton_raphson(max_iterations: usize, tolerance: f64, regularization: f64) -> Self {
        Self::NewtonRaphson {
            max_iterations,
            tolerance,
            regularization,
        }
    }

    /// Create MM optimizer with custom parameters
    pub fn mm(max_iterations: usize, tolerance: f64, bootstrap_samples: usize) -> Self {
        Self::MM {
            max_iterations,
            tolerance,
            bootstrap_samples,
        }
    }
}

/// Result of Bradley-Terry MLE optimization
#[derive(Clone, Debug)]
pub struct BradleyTerryResult {
    /// Strength parameters (probability scale, sum to n)
    pub strengths: HashMap<CandidateId, f64>,
    /// Covariance matrix (from Fisher^-1 or bootstrap)
    pub covariance: DMatrix<f64>,
    /// Mapping from CandidateId to matrix index
    pub id_to_index: HashMap<CandidateId, usize>,
    /// Log-likelihood at solution
    pub log_likelihood: f64,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Did the algorithm converge?
    pub converged: bool,
    /// Final gradient norm (Newton-Raphson) or max parameter change (MM)
    pub convergence_metric: f64,
}

impl BradleyTerryResult {
    /// Get fitness estimate for a candidate with uncertainty
    pub fn get_estimate(&self, id: CandidateId) -> Option<FitnessEstimate> {
        let strength = *self.strengths.get(&id)?;
        let idx = *self.id_to_index.get(&id)?;

        // Variance is diagonal element of covariance matrix
        let variance = if idx < self.covariance.nrows() {
            self.covariance[(idx, idx)]
        } else {
            f64::INFINITY
        };

        // Count total comparisons involving this candidate
        let observation_count = self.strengths.len(); // Approximate

        Some(FitnessEstimate::new(strength, variance, observation_count))
    }

    /// Get all estimates as a map
    pub fn all_estimates(&self) -> HashMap<CandidateId, FitnessEstimate> {
        self.strengths
            .keys()
            .filter_map(|&id| self.get_estimate(id).map(|e| (id, e)))
            .collect()
    }

    /// Predict probability that candidate a beats candidate b
    pub fn predict_win_probability(&self, a: CandidateId, b: CandidateId) -> Option<f64> {
        let pa = self.strengths.get(&a)?;
        let pb = self.strengths.get(&b)?;
        Some(pa / (pa + pb))
    }
}

/// Bradley-Terry model for pairwise comparison data
pub struct BradleyTerryModel {
    optimizer: BradleyTerryOptimizer,
}

impl BradleyTerryModel {
    /// Create a new Bradley-Terry model with specified optimizer
    pub fn new(optimizer: BradleyTerryOptimizer) -> Self {
        Self { optimizer }
    }

    /// Fit the model to comparison data
    ///
    /// # Arguments
    ///
    /// * `comparisons` - Historical pairwise comparison records
    /// * `candidate_ids` - All candidate IDs to include (may include uncompared candidates)
    ///
    /// # Returns
    ///
    /// `BradleyTerryResult` with fitted strengths and uncertainty estimates
    pub fn fit(
        &self,
        comparisons: &[ComparisonRecord],
        candidate_ids: &[CandidateId],
    ) -> BradleyTerryResult {
        if candidate_ids.is_empty() || comparisons.is_empty() {
            return self.empty_result(candidate_ids);
        }

        let ctx = FitContext::new(comparisons, candidate_ids);

        match &self.optimizer {
            BradleyTerryOptimizer::NewtonRaphson {
                max_iterations,
                tolerance,
                regularization,
            } => self.fit_newton_raphson(&ctx, *max_iterations, *tolerance, *regularization),
            BradleyTerryOptimizer::MM {
                max_iterations,
                tolerance,
                bootstrap_samples,
            } => self.fit_mm(&ctx, *max_iterations, *tolerance, *bootstrap_samples),
        }
    }

    /// Empty result for edge cases
    fn empty_result(&self, candidate_ids: &[CandidateId]) -> BradleyTerryResult {
        let n = candidate_ids.len();
        let strengths: HashMap<CandidateId, f64> =
            candidate_ids.iter().map(|&id| (id, 1.0)).collect();
        let id_to_index: HashMap<CandidateId, usize> = candidate_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        BradleyTerryResult {
            strengths,
            covariance: DMatrix::from_diagonal_element(n, n, f64::INFINITY),
            id_to_index,
            log_likelihood: 0.0,
            iterations: 0,
            converged: true,
            convergence_metric: 0.0,
        }
    }

    /// Newton-Raphson optimization
    ///
    /// Uses log-parameterization: θ_i = log(π_i)
    /// This makes the optimization unconstrained.
    fn fit_newton_raphson(
        &self,
        ctx: &FitContext,
        max_iterations: usize,
        tolerance: f64,
        regularization: f64,
    ) -> BradleyTerryResult {
        let n = ctx.n;
        let comparisons = ctx.comparisons;
        let candidate_ids = ctx.candidate_ids;
        let id_to_index = &ctx.id_to_index;
        // Initialize log-strengths to zero
        let mut theta = DVector::zeros(n);

        // Count wins for each candidate
        let mut wins = vec![0usize; n];
        for comp in comparisons {
            if let Some(&idx) = id_to_index.get(&comp.winner) {
                wins[idx] += 1;
            }
        }

        let mut converged = false;
        let mut iterations = 0;
        let mut gradient_norm = f64::INFINITY;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Compute gradient and Hessian
            let mut gradient = DVector::zeros(n);
            let mut hessian = DMatrix::zeros(n, n);

            // Gradient: g_i = wins_i - Σ_j n_ij * σ(θ_i - θ_j)
            // Hessian: H_ii = -Σ_j n_ij * σ(θ_i - θ_j) * (1 - σ(θ_i - θ_j))
            //          H_ij = n_ij * σ(θ_i - θ_j) * (1 - σ(θ_i - θ_j))

            for comp in comparisons {
                let i = match id_to_index.get(&comp.winner) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let j = match id_to_index.get(&comp.loser) {
                    Some(&idx) => idx,
                    None => continue,
                };

                // σ(θ_i - θ_j) = P(i beats j)
                let diff = theta[i] - theta[j];
                let p = sigmoid(diff);
                let q = 1.0 - p; // P(j beats i)

                // Gradient contributions
                gradient[i] += q; // = 1 - p
                gradient[j] -= q; // = -(1 - p) = p - 1

                // Hessian contributions (second derivatives of log-likelihood)
                let h = p * q;
                hessian[(i, i)] -= h;
                hessian[(j, j)] -= h;
                hessian[(i, j)] += h;
                hessian[(j, i)] += h;
            }

            // Add regularization to diagonal
            for i in 0..n {
                hessian[(i, i)] -= regularization;
            }

            // Check convergence
            gradient_norm = gradient.norm();
            if gradient_norm < tolerance {
                converged = true;
                break;
            }

            // Newton step: δ = -H^{-1} * g
            // Use LU decomposition for solving
            let neg_hessian = -&hessian;
            let delta = match neg_hessian.clone().lu().solve(&gradient) {
                Some(d) => d,
                None => {
                    // Hessian is singular, add more regularization
                    let mut reg_hessian = neg_hessian;
                    for i in 0..n {
                        reg_hessian[(i, i)] += regularization * 10.0;
                    }
                    match reg_hessian.lu().solve(&gradient) {
                        Some(d) => d,
                        None => break, // Give up
                    }
                }
            };

            // Update with line search backtracking for stability
            let mut step_size = 1.0;
            let current_ll = self.log_likelihood(&theta, comparisons, id_to_index);

            for _ in 0..10 {
                let new_theta = &theta + step_size * &delta;
                let new_ll = self.log_likelihood(&new_theta, comparisons, id_to_index);

                if new_ll > current_ll - 1e-4 * step_size * gradient.dot(&delta) {
                    theta = new_theta;
                    break;
                }
                step_size *= 0.5;
            }

            // Normalize (subtract mean for identifiability)
            let mean_theta = theta.mean();
            theta -= DVector::from_element(n, mean_theta);
        }

        // Convert to probability scale
        let strengths: HashMap<CandidateId, f64> = candidate_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, theta[i].exp()))
            .collect();

        // Covariance from inverse negative Hessian (Fisher Information)
        let mut final_hessian = DMatrix::zeros(n, n);
        for comp in comparisons {
            let i = match id_to_index.get(&comp.winner) {
                Some(&idx) => idx,
                None => continue,
            };
            let j = match id_to_index.get(&comp.loser) {
                Some(&idx) => idx,
                None => continue,
            };

            let p = sigmoid(theta[i] - theta[j]);
            let h = p * (1.0 - p);

            final_hessian[(i, i)] -= h;
            final_hessian[(j, j)] -= h;
            final_hessian[(i, j)] += h;
            final_hessian[(j, i)] += h;
        }

        // Add regularization
        for i in 0..n {
            final_hessian[(i, i)] -= regularization;
        }

        // Covariance = -H^{-1}
        let covariance = match (-&final_hessian).clone().try_inverse() {
            Some(inv) => inv,
            None => DMatrix::from_diagonal_element(n, n, f64::INFINITY),
        };

        let log_likelihood = self.log_likelihood(&theta, comparisons, id_to_index);

        BradleyTerryResult {
            strengths,
            covariance,
            id_to_index: id_to_index.clone(),
            log_likelihood,
            iterations,
            converged,
            convergence_metric: gradient_norm,
        }
    }

    /// MM algorithm optimization
    fn fit_mm(
        &self,
        ctx: &FitContext,
        max_iterations: usize,
        tolerance: f64,
        bootstrap_samples: usize,
    ) -> BradleyTerryResult {
        let n = ctx.n;
        let comparisons = ctx.comparisons;
        let candidate_ids = ctx.candidate_ids;
        let id_to_index = &ctx.id_to_index;

        // Fit point estimates
        let (pi, iterations, converged, max_change) =
            self.mm_core(comparisons, id_to_index, n, max_iterations, tolerance);

        // Bootstrap for variance estimation
        let covariance =
            self.bootstrap_covariance(ctx, max_iterations, tolerance, bootstrap_samples, &pi);

        // Convert to HashMap
        let strengths: HashMap<CandidateId, f64> = candidate_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, pi[i]))
            .collect();

        // Compute log-likelihood
        let theta: DVector<f64> = pi.iter().map(|&p| p.ln()).collect::<Vec<_>>().into();
        let log_likelihood = self.log_likelihood(&theta, comparisons, id_to_index);

        BradleyTerryResult {
            strengths,
            covariance,
            id_to_index: id_to_index.clone(),
            log_likelihood,
            iterations,
            converged,
            convergence_metric: max_change,
        }
    }

    /// Core MM iteration
    fn mm_core(
        &self,
        comparisons: &[ComparisonRecord],
        id_to_index: &HashMap<CandidateId, usize>,
        n: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> (Vec<f64>, usize, bool, f64) {
        // Initialize strengths uniformly
        let mut pi = vec![1.0; n];

        // Count wins
        let mut wins = vec![0usize; n];
        for comp in comparisons {
            if let Some(&idx) = id_to_index.get(&comp.winner) {
                wins[idx] += 1;
            }
        }

        let mut converged = false;
        let mut iterations = 0;
        let mut max_change = f64::INFINITY;

        for iter in 0..max_iterations {
            iterations = iter + 1;
            let mut pi_new = vec![0.0; n];

            for i in 0..n {
                if wins[i] == 0 {
                    // Candidate with no wins - use small regularized value
                    pi_new[i] = 0.01;
                    continue;
                }

                // Compute denominator: Σ_j n_ij / (π_i + π_j)
                let mut denom = 0.0;
                for comp in comparisons {
                    let w_idx = id_to_index.get(&comp.winner).copied();
                    let l_idx = id_to_index.get(&comp.loser).copied();

                    match (w_idx, l_idx) {
                        (Some(wi), Some(li)) if wi == i || li == i => {
                            let other = if wi == i { li } else { wi };
                            denom += 1.0 / (pi[i] + pi[other]);
                        }
                        _ => {}
                    }
                }

                if denom > 0.0 {
                    pi_new[i] = wins[i] as f64 / denom;
                } else {
                    pi_new[i] = pi[i]; // No comparisons, keep current
                }
            }

            // Normalize so strengths sum to n (arbitrary but stable)
            let sum: f64 = pi_new.iter().sum();
            if sum > 0.0 {
                for p in &mut pi_new {
                    *p *= n as f64 / sum;
                }
            }

            // Check convergence
            max_change = pi
                .iter()
                .zip(pi_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            if max_change < tolerance {
                converged = true;
                pi = pi_new;
                break;
            }

            pi = pi_new;
        }

        (pi, iterations, converged, max_change)
    }

    /// Bootstrap resampling for variance estimation
    fn bootstrap_covariance(
        &self,
        ctx: &FitContext,
        max_iterations: usize,
        tolerance: f64,
        bootstrap_samples: usize,
        point_estimate: &[f64],
    ) -> DMatrix<f64> {
        let n = ctx.n;
        let comparisons = ctx.comparisons;
        let id_to_index = &ctx.id_to_index;

        if bootstrap_samples == 0 || comparisons.is_empty() {
            return DMatrix::from_diagonal_element(n, n, f64::INFINITY);
        }

        let mut rng = rand::thread_rng();
        let mut bootstrap_estimates: Vec<Vec<f64>> = Vec::with_capacity(bootstrap_samples);

        for _ in 0..bootstrap_samples {
            // Resample comparisons with replacement
            let resampled: Vec<ComparisonRecord> = (0..comparisons.len())
                .map(|_| comparisons[rng.gen_range(0..comparisons.len())].clone())
                .collect();

            // Fit to resampled data
            let (pi, _, _, _) = self.mm_core(&resampled, id_to_index, n, max_iterations, tolerance);
            bootstrap_estimates.push(pi);
        }

        // Compute covariance matrix from bootstrap samples
        let mut covariance = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let mean_i = point_estimate[i];
                let mean_j = point_estimate[j];

                let cov: f64 = bootstrap_estimates
                    .iter()
                    .map(|est| (est[i] - mean_i) * (est[j] - mean_j))
                    .sum::<f64>()
                    / (bootstrap_samples - 1).max(1) as f64;

                covariance[(i, j)] = cov;
            }
        }

        covariance
    }

    /// Compute log-likelihood
    fn log_likelihood(
        &self,
        theta: &DVector<f64>,
        comparisons: &[ComparisonRecord],
        id_to_index: &HashMap<CandidateId, usize>,
    ) -> f64 {
        let mut ll = 0.0;

        for comp in comparisons {
            let i = match id_to_index.get(&comp.winner) {
                Some(&idx) => idx,
                None => continue,
            };
            let j = match id_to_index.get(&comp.loser) {
                Some(&idx) => idx,
                None => continue,
            };

            // log P(i beats j) = log(σ(θ_i - θ_j)) = θ_i - θ_j - log(1 + exp(θ_i - θ_j))
            let diff = theta[i] - theta[j];
            ll += log_sigmoid(diff);
        }

        ll
    }
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Log sigmoid: log(σ(x)) = -log(1 + exp(-x))
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_comparisons(pairs: &[(usize, usize)]) -> Vec<ComparisonRecord> {
        pairs
            .iter()
            .map(|&(w, l)| ComparisonRecord {
                winner: CandidateId(w),
                loser: CandidateId(l),
                generation: 0,
            })
            .collect()
    }

    #[test]
    fn test_newton_raphson_basic() {
        // Simple case: A beats B twice, B beats C twice
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (1, 2), (1, 2)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1), CandidateId(2)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        assert!(result.converged);

        // A should be strongest, C weakest
        let pa = result.strengths[&CandidateId(0)];
        let pb = result.strengths[&CandidateId(1)];
        let pc = result.strengths[&CandidateId(2)];

        assert!(pa > pb);
        assert!(pb > pc);
    }

    #[test]
    fn test_mm_basic() {
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (1, 2), (1, 2)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1), CandidateId(2)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::mm(100, 1e-8, 50));
        let result = model.fit(&comparisons, &candidate_ids);

        assert!(result.converged);

        let pa = result.strengths[&CandidateId(0)];
        let pb = result.strengths[&CandidateId(1)];
        let pc = result.strengths[&CandidateId(2)];

        assert!(pa > pb);
        assert!(pb > pc);
    }

    #[test]
    fn test_newton_raphson_and_mm_agree() {
        let comparisons = make_comparisons(&[
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 1),
            (1, 0),
            (2, 1),
            (0, 2),
            (0, 2),
        ]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1), CandidateId(2)];

        let nr_model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let mm_model = BradleyTerryModel::new(BradleyTerryOptimizer::mm(100, 1e-8, 0));

        let nr_result = nr_model.fit(&comparisons, &candidate_ids);
        let mm_result = mm_model.fit(&comparisons, &candidate_ids);

        // Rankings should agree
        let nr_ranking: Vec<_> = {
            let mut r: Vec<_> = candidate_ids
                .iter()
                .map(|&id| (id, nr_result.strengths[&id]))
                .collect();
            r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            r.into_iter().map(|(id, _)| id).collect()
        };

        let mm_ranking: Vec<_> = {
            let mut r: Vec<_> = candidate_ids
                .iter()
                .map(|&id| (id, mm_result.strengths[&id]))
                .collect();
            r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            r.into_iter().map(|(id, _)| id).collect()
        };

        assert_eq!(nr_ranking, mm_ranking);
    }

    #[test]
    fn test_get_estimate() {
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (0, 1)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        let estimate = result.get_estimate(CandidateId(0)).unwrap();
        assert!(estimate.variance < f64::INFINITY);
        assert!(estimate.variance > 0.0);
    }

    #[test]
    fn test_predict_win_probability() {
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (0, 1), (0, 1)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        let p = result
            .predict_win_probability(CandidateId(0), CandidateId(1))
            .unwrap();
        assert!(p > 0.5); // A should be favored
        assert!(p < 1.0);
    }

    #[test]
    fn test_empty_comparisons() {
        let comparisons: Vec<ComparisonRecord> = vec![];
        let candidate_ids = vec![CandidateId(0), CandidateId(1)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        // Should return uniform strengths with infinite variance
        assert!(result.converged);
        assert!(result.covariance[(0, 0)].is_infinite());
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);

        // Symmetry: σ(-x) = 1 - σ(x)
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert!((sigmoid(-x) - (1.0 - sigmoid(x))).abs() < 1e-9);
        }
    }

    #[test]
    fn test_log_sigmoid() {
        // log(σ(x)) should be negative
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
            assert!(log_sigmoid(x) <= 0.0);
            assert!((log_sigmoid(x).exp() - sigmoid(x)).abs() < 1e-9);
        }
    }

    #[test]
    fn test_covariance_positive_semidefinite() {
        let comparisons = make_comparisons(&[(0, 1), (0, 2), (1, 2), (0, 1), (1, 2), (0, 2)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1), CandidateId(2)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        // Diagonal should be non-negative
        for i in 0..3 {
            assert!(result.covariance[(i, i)] >= 0.0);
        }
    }
}
