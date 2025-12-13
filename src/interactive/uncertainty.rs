//! Uncertainty quantification for fitness estimates
//!
//! This module provides types and utilities for representing fitness estimates
//! with associated uncertainty (variance and confidence intervals).

use serde::{Deserialize, Serialize};

/// Full fitness estimate with uncertainty quantification
///
/// Represents a fitness value along with its statistical uncertainty,
/// including variance and confidence intervals. This enables informed
/// decision-making about which candidates need more evaluation.
///
/// # Example
///
/// ```rust
/// use fugue_evo::interactive::uncertainty::FitnessEstimate;
///
/// let estimate = FitnessEstimate::new(7.5, 0.25, 10);
/// println!("Fitness: {:.2} ± {:.2}", estimate.mean, estimate.std_error());
/// println!("95% CI: [{:.2}, {:.2}]", estimate.ci_lower, estimate.ci_upper);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitnessEstimate {
    /// Point estimate (mean fitness)
    pub mean: f64,
    /// Variance of the estimate
    pub variance: f64,
    /// Lower bound of confidence interval (default 95%)
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Number of observations contributing to this estimate
    pub observation_count: usize,
}

impl FitnessEstimate {
    /// Z-score for 95% confidence interval
    const Z_95: f64 = 1.96;

    /// Create a new fitness estimate from mean and variance
    ///
    /// Automatically computes 95% confidence intervals from the variance.
    ///
    /// # Arguments
    ///
    /// * `mean` - Point estimate of fitness
    /// * `variance` - Variance of the estimate (not the population variance)
    /// * `observation_count` - Number of observations used to compute the estimate
    pub fn new(mean: f64, variance: f64, observation_count: usize) -> Self {
        let std_err = variance.sqrt().max(0.0);
        Self {
            mean,
            variance,
            ci_lower: mean - Self::Z_95 * std_err,
            ci_upper: mean + Self::Z_95 * std_err,
            observation_count,
        }
    }

    /// Create estimate with custom confidence level
    ///
    /// # Arguments
    ///
    /// * `mean` - Point estimate
    /// * `variance` - Variance of the estimate
    /// * `observation_count` - Number of observations
    /// * `z_score` - Z-score for desired confidence level (e.g., 1.96 for 95%, 2.576 for 99%)
    pub fn with_confidence(
        mean: f64,
        variance: f64,
        observation_count: usize,
        z_score: f64,
    ) -> Self {
        let std_err = variance.sqrt().max(0.0);
        Self {
            mean,
            variance,
            ci_lower: mean - z_score * std_err,
            ci_upper: mean + z_score * std_err,
            observation_count,
        }
    }

    /// Create an uninformative estimate (infinite variance)
    ///
    /// Used for candidates with no evaluations.
    pub fn uninformative(default_mean: f64) -> Self {
        Self {
            mean: default_mean,
            variance: f64::INFINITY,
            ci_lower: f64::NEG_INFINITY,
            ci_upper: f64::INFINITY,
            observation_count: 0,
        }
    }

    /// Standard error (sqrt of variance)
    pub fn std_error(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Width of the confidence interval
    pub fn ci_width(&self) -> f64 {
        self.ci_upper - self.ci_lower
    }

    /// Check if confidence interval contains a value
    pub fn ci_contains(&self, value: f64) -> bool {
        value >= self.ci_lower && value <= self.ci_upper
    }

    /// Check if this estimate overlaps with another's confidence interval
    pub fn ci_overlaps(&self, other: &FitnessEstimate) -> bool {
        self.ci_lower <= other.ci_upper && self.ci_upper >= other.ci_lower
    }

    /// Check if this estimate is significantly better than another
    ///
    /// Returns true if the lower bound of this estimate's CI is above
    /// the upper bound of the other's CI.
    pub fn significantly_better_than(&self, other: &FitnessEstimate) -> bool {
        self.ci_lower > other.ci_upper
    }

    /// Check if this estimate has high uncertainty (needs more data)
    ///
    /// Returns true if variance is infinite or observation count is below threshold.
    pub fn is_uncertain(&self, min_observations: usize) -> bool {
        self.variance.is_infinite() || self.observation_count < min_observations
    }

    /// Coefficient of variation (relative uncertainty)
    ///
    /// Returns None if mean is zero to avoid division by zero.
    pub fn coefficient_of_variation(&self) -> Option<f64> {
        if self.mean.abs() < f64::EPSILON {
            None
        } else {
            Some(self.std_error() / self.mean.abs())
        }
    }

    /// Merge two independent estimates (weighted by inverse variance)
    ///
    /// Combines two estimates using inverse-variance weighting,
    /// which is optimal for independent normal estimates.
    pub fn merge(&self, other: &FitnessEstimate) -> FitnessEstimate {
        // Handle infinite variance cases
        if self.variance.is_infinite() {
            return other.clone();
        }
        if other.variance.is_infinite() {
            return self.clone();
        }
        if self.variance == 0.0 && other.variance == 0.0 {
            // Both have zero variance - average
            return FitnessEstimate::new(
                (self.mean + other.mean) / 2.0,
                0.0,
                self.observation_count + other.observation_count,
            );
        }

        // Inverse variance weighting
        let w1 = 1.0 / self.variance;
        let w2 = 1.0 / other.variance;
        let w_total = w1 + w2;

        let merged_mean = (w1 * self.mean + w2 * other.mean) / w_total;
        let merged_variance = 1.0 / w_total;
        let merged_count = self.observation_count + other.observation_count;

        FitnessEstimate::new(merged_mean, merged_variance, merged_count)
    }
}

impl Default for FitnessEstimate {
    fn default() -> Self {
        Self::uninformative(0.0)
    }
}

/// Compute sample variance using Welford's online algorithm
///
/// This is numerically stable for computing variance incrementally.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WelfordVariance {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squared differences from mean
}

impl WelfordVariance {
    /// Create a new variance calculator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new observation
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get current count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get sample variance (unbiased, divided by n-1)
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            f64::INFINITY
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get population variance (divided by n)
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Get variance of the mean (standard error squared)
    pub fn variance_of_mean(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            self.sample_variance() / self.count as f64
        }
    }

    /// Convert to FitnessEstimate
    pub fn to_estimate(&self) -> FitnessEstimate {
        FitnessEstimate::new(self.mean, self.variance_of_mean(), self.count)
    }

    /// Merge with another WelfordVariance (for parallel computation)
    pub fn merge(&self, other: &WelfordVariance) -> WelfordVariance {
        if self.count == 0 {
            return other.clone();
        }
        if other.count == 0 {
            return self.clone();
        }

        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * other.count as f64 / count as f64;
        let m2 = self.m2
            + other.m2
            + delta * delta * self.count as f64 * other.count as f64 / count as f64;

        WelfordVariance { count, mean, m2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_estimate_creation() {
        let est = FitnessEstimate::new(5.0, 0.25, 10);
        assert_eq!(est.mean, 5.0);
        assert_eq!(est.variance, 0.25);
        assert_eq!(est.observation_count, 10);
        assert!((est.std_error() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_interval() {
        let est = FitnessEstimate::new(10.0, 1.0, 100);
        // 95% CI with std_err = 1.0: [10 - 1.96, 10 + 1.96]
        assert!((est.ci_lower - 8.04).abs() < 0.01);
        assert!((est.ci_upper - 11.96).abs() < 0.01);
        assert!((est.ci_width() - 3.92).abs() < 0.01);
    }

    #[test]
    fn test_ci_contains() {
        let est = FitnessEstimate::new(10.0, 1.0, 100);
        assert!(est.ci_contains(10.0));
        assert!(est.ci_contains(9.0));
        assert!(est.ci_contains(11.0));
        assert!(!est.ci_contains(5.0));
        assert!(!est.ci_contains(15.0));
    }

    #[test]
    fn test_ci_overlaps() {
        let est1 = FitnessEstimate::new(10.0, 1.0, 100);
        let est2 = FitnessEstimate::new(11.0, 1.0, 100);
        let est3 = FitnessEstimate::new(20.0, 1.0, 100);

        assert!(est1.ci_overlaps(&est2)); // Close estimates overlap
        assert!(!est1.ci_overlaps(&est3)); // Far estimates don't overlap
    }

    #[test]
    fn test_significantly_better_than() {
        let good = FitnessEstimate::new(20.0, 0.1, 100);
        let bad = FitnessEstimate::new(10.0, 0.1, 100);
        let uncertain = FitnessEstimate::new(15.0, 100.0, 5);

        assert!(good.significantly_better_than(&bad));
        assert!(!bad.significantly_better_than(&good));
        assert!(!good.significantly_better_than(&uncertain)); // Uncertain overlaps
    }

    #[test]
    fn test_uninformative_estimate() {
        let est = FitnessEstimate::uninformative(5.0);
        assert_eq!(est.mean, 5.0);
        assert!(est.variance.is_infinite());
        assert!(est.is_uncertain(1));
    }

    #[test]
    fn test_merge_estimates() {
        let est1 = FitnessEstimate::new(10.0, 1.0, 10);
        let est2 = FitnessEstimate::new(12.0, 1.0, 10);

        let merged = est1.merge(&est2);
        assert_eq!(merged.mean, 11.0); // Average when equal weights
        assert!(merged.variance < est1.variance); // Variance decreases
        assert_eq!(merged.observation_count, 20);
    }

    #[test]
    fn test_merge_with_uninformative() {
        let est = FitnessEstimate::new(10.0, 1.0, 10);
        let uninf = FitnessEstimate::uninformative(5.0);

        let merged = est.merge(&uninf);
        assert_eq!(merged.mean, est.mean);
        assert_eq!(merged.variance, est.variance);
    }

    #[test]
    fn test_welford_variance() {
        let mut welford = WelfordVariance::new();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        for v in values {
            welford.update(v);
        }

        assert_eq!(welford.count(), 8);
        assert!((welford.mean() - 5.0).abs() < 1e-9);
        // Sample variance should be 4.571... (32/7)
        assert!((welford.sample_variance() - 32.0 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_welford_merge() {
        let mut w1 = WelfordVariance::new();
        let mut w2 = WelfordVariance::new();

        for v in [1.0, 2.0, 3.0] {
            w1.update(v);
        }
        for v in [4.0, 5.0, 6.0] {
            w2.update(v);
        }

        let merged = w1.merge(&w2);
        assert_eq!(merged.count(), 6);
        assert!((merged.mean() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_welford_to_estimate() {
        let mut welford = WelfordVariance::new();
        for v in [10.0, 11.0, 9.0, 10.0, 10.0] {
            welford.update(v);
        }

        let est = welford.to_estimate();
        assert_eq!(est.mean, welford.mean());
        assert_eq!(est.observation_count, 5);
        assert_eq!(est.variance, welford.variance_of_mean());
    }

    #[test]
    fn test_coefficient_of_variation() {
        let est = FitnessEstimate::new(10.0, 1.0, 100);
        let cv = est.coefficient_of_variation().unwrap();
        assert!((cv - 0.1).abs() < 1e-9); // std_err / mean = 1.0 / 10.0

        let zero_mean = FitnessEstimate::new(0.0, 1.0, 100);
        assert!(zero_mean.coefficient_of_variation().is_none());
    }
}
