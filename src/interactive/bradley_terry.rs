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
        /// Gaussian prior precision on the log-strengths (L2 penalty
        /// coefficient λ, default: 0.1).
        ///
        /// This is a genuine MAP prior: the penalized objective is
        /// `LL(θ) − (λ/2)·‖θ‖²`, so the prior contributes `−λθ` to the
        /// gradient and `−λ` to the Hessian diagonal. It shrinks the
        /// log-strengths toward `0` (strength `1`), which keeps candidates
        /// that win or lose *all* of their comparisons finite instead of
        /// diverging to ±∞.
        #[serde(alias = "regularization")]
        prior_lambda: f64,
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
            prior_lambda: 0.1,
        }
    }
}

impl BradleyTerryOptimizer {
    /// Create Newton-Raphson optimizer with custom parameters
    ///
    /// `prior_lambda` is the precision of the Gaussian prior on the
    /// log-strengths (see [`BradleyTerryOptimizer::NewtonRaphson`]). A value of
    /// `0.0` recovers the unregularized MLE (which can diverge for
    /// all-win/all-loss candidates); `0.1` is a sensible default.
    pub fn newton_raphson(max_iterations: usize, tolerance: f64, prior_lambda: f64) -> Self {
        Self::NewtonRaphson {
            max_iterations,
            tolerance,
            prior_lambda,
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
///
/// # Scale convention
///
/// Both optimization paths (Newton-Raphson and MM) report the point estimate
/// and its uncertainty on the **strength scale** `π = exp(θ)`:
///
/// - `strengths` holds `π_i` (strictly positive, mean-centered in log-space so
///   `Σ log π_i = 0`).
/// - `covariance` is `Cov(π)`, i.e. the covariance of the *strengths*, not of
///   the log-strengths. Newton-Raphson obtains it by the delta method from the
///   sum-to-zero-constrained Fisher information; MM obtains it by bootstrap.
///   Because both are on the same (strength) scale, downstream consumers such
///   as `CandidateStats::model_variance` and the active-learning acquisition
///   can use them interchangeably.
#[derive(Clone, Debug)]
pub struct BradleyTerryResult {
    /// Strength parameters `π_i = exp(θ_i)` (probability scale, log-strengths
    /// sum to zero)
    pub strengths: HashMap<CandidateId, f64>,
    /// Covariance of the strengths `Cov(π)` (delta-method Fisher⁻¹ or bootstrap)
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
                prior_lambda,
            } => self.fit_newton_raphson(&ctx, *max_iterations, *tolerance, *prior_lambda),
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
    /// Uses log-parameterization: θ_i = log(π_i), so the optimization is
    /// unconstrained. A Gaussian prior on the log-strengths (precision
    /// `prior_lambda`, EV-67) turns this into a MAP estimator: the penalized
    /// objective is `LL(θ) − (λ/2)·‖θ‖²`, whose gradient carries `−λθ` and whose
    /// Hessian diagonal carries `−λ`. The prior keeps all-win / all-loss
    /// candidates finite and makes the (penalized) Hessian strictly negative
    /// definite so the Newton solve never hits the singular all-ones direction.
    fn fit_newton_raphson(
        &self,
        ctx: &FitContext,
        max_iterations: usize,
        tolerance: f64,
        prior_lambda: f64,
    ) -> BradleyTerryResult {
        let n = ctx.n;
        let comparisons = ctx.comparisons;
        let candidate_ids = ctx.candidate_ids;
        let id_to_index = &ctx.id_to_index;
        // Initialize log-strengths to zero
        let mut theta = DVector::zeros(n);

        let mut converged = false;
        let mut iterations = 0;
        let mut gradient_norm = f64::INFINITY;

        for iter in 0..max_iterations {
            iterations = iter + 1;

            // Compute gradient and Hessian of the log-likelihood.
            let mut gradient = DVector::zeros(n);
            let mut hessian = DMatrix::zeros(n, n);

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

            // Add the Gaussian log-strength prior (EV-67): a genuine MAP penalty
            // that contributes -λθ to the gradient AND -λ to the Hessian
            // diagonal (consistent, unlike the previous Hessian-only ridge).
            if prior_lambda > 0.0 {
                for i in 0..n {
                    gradient[i] -= prior_lambda * theta[i];
                    hessian[(i, i)] -= prior_lambda;
                }
            }

            // Check convergence on the penalized gradient.
            gradient_norm = gradient.norm();
            if gradient_norm < tolerance {
                converged = true;
                break;
            }

            // Newton ascent step: δ = (−H)^{-1} g. With the prior, −H = M + λI is
            // positive definite, so the solve is well conditioned.
            let neg_hessian = -&hessian;
            let delta = match neg_hessian.clone().lu().solve(&gradient) {
                Some(d) => d,
                None => {
                    // Extremely ill-conditioned graph: nudge the diagonal and retry.
                    let mut reg_hessian = neg_hessian;
                    let nudge = if prior_lambda > 0.0 {
                        prior_lambda
                    } else {
                        1e-6
                    };
                    for i in 0..n {
                        reg_hessian[(i, i)] += nudge;
                    }
                    match reg_hessian.lu().solve(&gradient) {
                        Some(d) => d,
                        None => break, // Give up
                    }
                }
            };

            // Backtracking line search enforcing the Armijo *sufficient-increase*
            // condition (EV-65): accept only steps that raise the penalized
            // log-likelihood by at least c·t·(gᵀδ).
            let (new_theta, _backtracks) = self.backtracking_line_search(
                &theta,
                &delta,
                &gradient,
                comparisons,
                id_to_index,
                prior_lambda,
            );
            theta = new_theta;

            // Normalize (subtract mean for identifiability); this stays inside the
            // sum-to-zero subspace that the prior also prefers.
            let mean_theta = theta.mean();
            theta -= DVector::from_element(n, mean_theta);
        }

        // Convert to strength scale.
        let strengths: HashMap<CandidateId, f64> = candidate_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, theta[i].exp()))
            .collect();

        // Strength-scale covariance via the delta method from the constrained
        // Fisher information (EV-25 / EV-66).
        let covariance = self.strength_covariance(&theta, comparisons, id_to_index, n);

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

    /// Backtracking line search for the Newton *ascent* step.
    ///
    /// Returns the accepted parameter vector and the number of times the step
    /// was halved. Accepts the first step `t ∈ {1, 1/2, 1/4, …}` satisfying the
    /// Armijo sufficient-increase condition (EV-65)
    ///
    /// ```text
    /// f(θ + t·δ) ≥ f(θ) + c·t·(∇f·δ)
    /// ```
    ///
    /// where `f` is the penalized log-likelihood and `∇f·δ = gᵀ(−H)^{-1}g ≥ 0`
    /// is a genuine ascent slope. If no step in the schedule qualifies (should
    /// not happen for a proper ascent direction) the original `θ` is returned
    /// unchanged so the outer loop can never *decrease* the objective.
    fn backtracking_line_search(
        &self,
        theta: &DVector<f64>,
        delta: &DVector<f64>,
        gradient: &DVector<f64>,
        comparisons: &[ComparisonRecord],
        id_to_index: &HashMap<CandidateId, usize>,
        prior_lambda: f64,
    ) -> (DVector<f64>, usize) {
        const C1: f64 = 1e-4;
        const MAX_BACKTRACKS: usize = 30;

        let dir_deriv = gradient.dot(delta);
        let current = self.penalized_log_likelihood(theta, comparisons, id_to_index, prior_lambda);

        let mut step_size = 1.0;
        for backtracks in 0..MAX_BACKTRACKS {
            let candidate = theta + step_size * delta;
            let candidate_ll =
                self.penalized_log_likelihood(&candidate, comparisons, id_to_index, prior_lambda);

            if armijo_sufficient_increase(current, candidate_ll, step_size, dir_deriv, C1) {
                return (candidate, backtracks);
            }
            step_size *= 0.5;
        }

        // No admissible step found: make no move rather than risk a decrease.
        (theta.clone(), MAX_BACKTRACKS)
    }

    /// Penalized log-likelihood `LL(θ) − (λ/2)·‖θ‖²` (the MAP objective).
    fn penalized_log_likelihood(
        &self,
        theta: &DVector<f64>,
        comparisons: &[ComparisonRecord],
        id_to_index: &HashMap<CandidateId, usize>,
        prior_lambda: f64,
    ) -> f64 {
        self.log_likelihood(theta, comparisons, id_to_index) - 0.5 * prior_lambda * theta.dot(theta)
    }

    /// Strength-scale covariance from the sum-to-zero-constrained Fisher
    /// information (EV-25 / EV-66).
    ///
    /// The BT log-likelihood in log-strengths `θ` is invariant to a global shift
    /// `θ → θ + c·1`, so the Fisher information `M = −H_likelihood` is singular
    /// with the all-ones vector in its null space. Ridge-inverting `(M + reg·I)`
    /// (the previous approach) put a spurious `1/reg` variance along that null
    /// direction, inflating every variance by ~`1/(n·reg)`. Instead we invert `M`
    /// on the sum-to-zero subspace via the Moore-Penrose pseudo-inverse
    /// (equivalently the reduced `(n−1)`-dimensional system), giving the
    /// constrained covariance of `θ`. We then map to the strength scale
    /// `π = exp(θ)` by the delta method, `Cov(π) = diag(π)·Cov(θ)·diag(π)`, so the
    /// reported variance is on the same scale as the reported strengths (matching
    /// the MM bootstrap covariance).
    ///
    /// The prior `λ` regularizes the *point estimate* only; the reported
    /// covariance is the likelihood's constrained observed information, which is
    /// what the numerical regression (`EV-25`) pins against the analytic value.
    fn strength_covariance(
        &self,
        theta: &DVector<f64>,
        comparisons: &[ComparisonRecord],
        id_to_index: &HashMap<CandidateId, usize>,
        n: usize,
    ) -> DMatrix<f64> {
        // Fisher information M = -H of the log-likelihood (no prior, no ridge).
        let mut m = DMatrix::<f64>::zeros(n, n);
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

            m[(i, i)] += h;
            m[(j, j)] += h;
            m[(i, j)] -= h;
            m[(j, i)] -= h;
        }

        // Constrained (sum-to-zero) covariance of θ via the pseudo-inverse.
        let cov_theta = match m.pseudo_inverse(1e-9) {
            Ok(inv) => inv,
            Err(_) => return DMatrix::from_diagonal_element(n, n, f64::INFINITY),
        };

        // Delta method to the strength scale: Cov(π) = diag(π)·Cov(θ)·diag(π).
        let pi: Vec<f64> = (0..n).map(|i| theta[i].exp()).collect();
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                cov[(i, j)] = pi[i] * pi[j] * cov_theta[(i, j)];
            }
        }
        cov
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

                // Regularized MM update (EV-67). This is the closed-form MAP
                // update under a Gamma(1+ε, ε) prior on π (mode at π = 1), which
                // plays the same role for the multiplicative MM iteration that
                // the Gaussian log-strength prior plays for Newton-Raphson: it
                // shrinks toward the neutral strength π = 1 and keeps all-win
                // (ε in the denominator) and all-loss (ε in the numerator)
                // candidates finite, replacing the previous arbitrary 0.01 floor.
                //
                // A literal Gaussian-on-log-strength prior has no closed-form MM
                // update; the Gamma pseudo-count is the mathematically standard
                // regularizer for MM Bradley-Terry (Caron & Doucet, 2012).
                let numerator = wins[i] as f64 + MM_PRIOR_PSEUDOCOUNT;
                let denom = denom + MM_PRIOR_PSEUDOCOUNT;
                pi_new[i] = if denom > 0.0 {
                    numerator / denom
                } else {
                    pi[i]
                };
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

/// Pseudo-count (Gamma-style) prior strength for the MM path.
///
/// Mirrors the Newton-Raphson Gaussian log-strength prior: both shrink toward
/// the neutral strength `π = 1` and keep all-win / all-loss candidates finite.
const MM_PRIOR_PSEUDOCOUNT: f64 = 0.1;

/// Armijo sufficient-*increase* test for maximizing `f` along an ascent
/// direction `δ` (EV-65).
///
/// Accepts the step when `f(θ + t·δ) ≥ f(θ) + c·t·(∇f·δ)`. Because `∇f·δ ≥ 0`
/// for an ascent direction, the acceptance threshold sits *above* the current
/// value, so the guard genuinely enforces monotone progress. (The previous code
/// *subtracted* the directional-derivative term, placing the threshold below the
/// current value and thereby accepting small decreases.)
fn armijo_sufficient_increase(
    current: f64,
    candidate: f64,
    step: f64,
    dir_deriv: f64,
    c: f64,
) -> bool {
    candidate >= current + c * step * dir_deriv
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

    #[test]
    fn test_constrained_fisher_covariance_matches_analytic() {
        // regression: EV-25 / EV-66 — the reported variance must equal the
        // sum-to-zero-constrained Fisher inverse (delta-mapped to the strength
        // scale), NOT a ridge-inflated `1/(n·reg)` value. Balanced 3-candidate
        // round-robin (each unordered pair compared twice, one win each) => the
        // MAP log-strengths are exactly 0, so π = 1 and the delta factor is 1.
        let comparisons = make_comparisons(&[(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]);
        let candidate_ids = vec![CandidateId(0), CandidateId(1), CandidateId(2)];

        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let result = model.fit(&comparisons, &candidate_ids);

        // Analytic constrained Fisher inverse: at θ=0 every h = 0.25, so
        // M = 1.5·I − 0.5·J and M⁺ has diagonal 4/9.
        let n = 3;
        let mut m = DMatrix::<f64>::zeros(n, n);
        for comp in &comparisons {
            let i = comp.winner.0;
            let j = comp.loser.0;
            let h = 0.25;
            m[(i, i)] += h;
            m[(j, j)] += h;
            m[(i, j)] -= h;
            m[(j, i)] -= h;
        }
        let analytic = m.pseudo_inverse(1e-9).unwrap();
        for i in 0..n {
            assert!(
                (result.covariance[(i, i)] - analytic[(i, i)]).abs() < 1e-6,
                "diag {}: got {}, analytic {}",
                i,
                result.covariance[(i, i)],
                analytic[(i, i)]
            );
            assert!((result.covariance[(i, i)] - 4.0 / 9.0).abs() < 1e-6);
        }
        // The old ridge inversion produced ~1/(n·reg) ≈ 3.3e5. We must be O(1).
        assert!(result.covariance[(0, 0)] < 1.0);
    }

    #[test]
    fn test_prior_keeps_all_win_all_loss_finite() {
        // regression: EV-67 — a candidate that wins (or loses) ALL comparisons
        // must stay finite thanks to the log-strength prior, not diverge.
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]);
        let ids = vec![CandidateId(0), CandidateId(1)];

        // Newton-Raphson (Gaussian log-strength prior).
        let nr = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let r = nr.fit(&comparisons, &ids);
        let s0 = r.strengths[&CandidateId(0)];
        let s1 = r.strengths[&CandidateId(1)];
        assert!(s0.is_finite() && s1.is_finite());
        assert!(s0 > s1);
        assert!(s0 < 50.0, "NR strength diverged: {}", s0);
        assert!(s1 > 0.0, "NR loser strength collapsed: {}", s1);

        // MM (Gamma pseudo-count prior).
        let mm = BradleyTerryModel::new(BradleyTerryOptimizer::mm(200, 1e-9, 0));
        let rm = mm.fit(&comparisons, &ids);
        let m0 = rm.strengths[&CandidateId(0)];
        let m1 = rm.strengths[&CandidateId(1)];
        assert!(m0.is_finite() && m0 < 50.0, "MM strength diverged: {}", m0);
        assert!(m0 > m1);
        assert!(m1 > 0.0);
    }

    #[test]
    fn test_armijo_sign_rejects_small_decrease() {
        // regression: EV-65 — the sufficient-increase guard must REJECT a step
        // that decreases the objective. The pre-fix condition subtracted the
        // directional-derivative term and would have ACCEPTED this same step.
        let current = 10.0;
        let candidate = 9.99995; // a tiny decrease
        let step = 1.0;
        let dir_deriv = 1.0; // positive ascent slope
        let c = 1e-4;

        // Correct threshold = 10 + 1e-4 = 10.0001, above the candidate -> reject.
        assert!(!armijo_sufficient_increase(
            current, candidate, step, dir_deriv, c
        ));
        // A sufficiently increasing step is accepted.
        assert!(armijo_sufficient_increase(
            current, 10.5, step, dir_deriv, c
        ));
        // The pre-fix (buggy) predicate used `current - c·t·(∇f·δ)` = 9.9999,
        // which the decreasing candidate exceeds -> it would have been accepted.
        let buggy_threshold = current - c * step * dir_deriv;
        assert!(candidate > buggy_threshold);
    }

    #[test]
    fn test_backtracking_triggers_on_overshoot() {
        // regression: EV-65 — with a deliberately oversized ascent direction the
        // full step overshoots and lowers the penalized log-likelihood, so the
        // line search MUST backtrack and still finish no lower than it started.
        let comparisons = make_comparisons(&[(0, 1), (0, 1), (1, 2), (1, 2)]);
        let ids = [CandidateId(0), CandidateId(1), CandidateId(2)];
        let id_to_index: HashMap<CandidateId, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let model = BradleyTerryModel::new(BradleyTerryOptimizer::default());
        let lambda = 0.1;

        let theta = DVector::from_element(3, 0.0);
        let mut gradient = DVector::zeros(3);
        let mut hessian = DMatrix::zeros(3, 3);
        for comp in &comparisons {
            let i = id_to_index[&comp.winner];
            let j = id_to_index[&comp.loser];
            let p = sigmoid(theta[i] - theta[j]);
            let q = 1.0 - p;
            let h = p * q;
            gradient[i] += q;
            gradient[j] -= q;
            hessian[(i, i)] -= h;
            hessian[(j, j)] -= h;
            hessian[(i, j)] += h;
            hessian[(j, i)] += h;
        }
        for i in 0..3 {
            gradient[i] -= lambda * theta[i];
            hessian[(i, i)] -= lambda;
        }
        let newton = (-&hessian).lu().solve(&gradient).unwrap();
        let big_delta = 50.0 * &newton; // gross overshoot

        let before = model.penalized_log_likelihood(&theta, &comparisons, &id_to_index, lambda);
        let (new_theta, backtracks) = model.backtracking_line_search(
            &theta,
            &big_delta,
            &gradient,
            &comparisons,
            &id_to_index,
            lambda,
        );
        let after = model.penalized_log_likelihood(&new_theta, &comparisons, &id_to_index, lambda);

        assert!(backtracks >= 1, "expected backtracking to trigger");
        assert!(
            after >= before,
            "line search must not decrease the objective"
        );
    }
}
