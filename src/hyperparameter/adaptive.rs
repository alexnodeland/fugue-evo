//! Adaptive control mechanisms
//!
//! These mechanisms adapt parameters based on feedback from the search process.

use std::collections::VecDeque;
use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

/// Rechenberg's 1/5 success rule for step-size adaptation
///
/// If the success rate is above 1/5, increase step size (more exploration)
/// If the success rate is below 1/5, decrease step size (more exploitation)
///
/// Reference: Rechenberg, I. (1973). Evolutionsstrategie.
#[derive(Clone, Debug)]
pub struct OneFifthRule {
    /// Factor to increase step size (typically 1.22 ≈ e^(1/5))
    pub increase_factor: f64,
    /// Factor to decrease step size (typically 0.82 ≈ e^(-1/5))
    pub decrease_factor: f64,
    /// Window size for computing success rate
    pub window_size: usize,
    /// Target success rate (default: 0.2)
    pub target_success_rate: f64,
    /// History of success/failure outcomes
    success_history: VecDeque<bool>,
}

impl OneFifthRule {
    /// Create a new 1/5 rule adapter with default parameters
    pub fn new() -> Self {
        Self {
            increase_factor: 1.22,
            decrease_factor: 0.82,
            window_size: 10,
            target_success_rate: 0.2,
            success_history: VecDeque::with_capacity(10),
        }
    }

    /// Set custom factors
    pub fn with_factors(mut self, increase: f64, decrease: f64) -> Self {
        self.increase_factor = increase;
        self.decrease_factor = decrease;
        self
    }

    /// Set window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self.success_history = VecDeque::with_capacity(size);
        self
    }

    /// Set target success rate
    pub fn with_target_rate(mut self, rate: f64) -> Self {
        self.target_success_rate = rate;
        self
    }

    /// Record a mutation outcome
    pub fn record(&mut self, success: bool) {
        self.success_history.push_back(success);
        if self.success_history.len() > self.window_size {
            self.success_history.pop_front();
        }
    }

    /// Get current success rate
    pub fn success_rate(&self) -> Option<f64> {
        if self.success_history.is_empty() {
            return None;
        }
        let successes = self.success_history.iter().filter(|&&s| s).count();
        Some(successes as f64 / self.success_history.len() as f64)
    }

    /// Adapt a step size based on current success rate
    pub fn adapt(&self, sigma: f64) -> f64 {
        if self.success_history.len() < self.window_size {
            return sigma;
        }

        let success_rate = self.success_rate().unwrap_or(self.target_success_rate);

        if success_rate > self.target_success_rate {
            sigma * self.increase_factor
        } else if success_rate < self.target_success_rate {
            sigma * self.decrease_factor
        } else {
            sigma
        }
    }

    /// Reset the history
    pub fn reset(&mut self) {
        self.success_history.clear();
    }
}

impl Default for OneFifthRule {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive operator selection using fitness-based credit assignment
///
/// Tracks performance of multiple operators and adjusts selection probabilities
/// based on the fitness improvements they produce.
#[derive(Clone, Debug)]
pub struct AdaptiveOperatorSelection {
    /// Number of operators
    pub num_operators: usize,
    /// Selection weights for each operator
    pub weights: Vec<f64>,
    /// Learning rate for weight updates
    pub learning_rate: f64,
    /// Minimum probability for any operator
    pub min_probability: f64,
    /// Decay factor for old rewards
    pub decay: f64,
}

impl AdaptiveOperatorSelection {
    /// Create a new adaptive operator selection with uniform initial weights
    pub fn new(num_operators: usize) -> Self {
        assert!(num_operators > 0, "Must have at least one operator");
        Self {
            num_operators,
            weights: vec![1.0 / num_operators as f64; num_operators],
            learning_rate: 0.1,
            min_probability: 0.05,
            decay: 0.99,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set minimum probability
    pub fn with_min_probability(mut self, prob: f64) -> Self {
        self.min_probability = prob;
        self
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    /// Select an operator index
    pub fn select<R: Rng>(&self, rng: &mut R) -> usize {
        let dist = WeightedIndex::new(&self.weights).unwrap();
        dist.sample(rng)
    }

    /// Update weights based on fitness improvement from an operator
    pub fn update(&mut self, operator_idx: usize, fitness_improvement: f64) {
        assert!(operator_idx < self.num_operators);

        // Apply decay to all weights
        for w in &mut self.weights {
            *w *= self.decay;
        }

        // Credit assignment based on fitness improvement
        let reward = fitness_improvement.max(0.0);
        self.weights[operator_idx] += self.learning_rate * reward;

        // Normalize and enforce minimum probability
        self.normalize_weights();
    }

    /// Normalize weights to sum to 1 while enforcing minimums
    fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum <= 0.0 {
            // Reset to uniform if weights collapsed
            for w in &mut self.weights {
                *w = 1.0 / self.num_operators as f64;
            }
            return;
        }

        // Normalize
        for w in &mut self.weights {
            *w /= sum;
        }

        // Enforce minimum probability
        let n = self.num_operators as f64;
        let mut deficit = 0.0;
        let mut excess_count = 0;

        for w in &mut self.weights {
            if *w < self.min_probability / n {
                deficit += self.min_probability / n - *w;
                *w = self.min_probability / n;
            } else {
                excess_count += 1;
            }
        }

        // Redistribute deficit from weights above minimum
        if deficit > 0.0 && excess_count > 0 {
            let reduction = deficit / excess_count as f64;
            for w in &mut self.weights {
                if *w > self.min_probability / n + reduction {
                    *w -= reduction;
                }
            }
        }

        // Final normalization
        let sum: f64 = self.weights.iter().sum();
        for w in &mut self.weights {
            *w /= sum;
        }
    }

    /// Get current selection probabilities
    pub fn probabilities(&self) -> &[f64] {
        &self.weights
    }

    /// Reset weights to uniform
    pub fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 1.0 / self.num_operators as f64;
        }
    }
}

/// Sliding window statistics tracker
#[derive(Clone, Debug)]
pub struct SlidingWindowStats {
    /// Window of values
    values: VecDeque<f64>,
    /// Maximum window size
    window_size: usize,
}

impl SlidingWindowStats {
    /// Create a new sliding window tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a value to the window
    pub fn push(&mut self, value: f64) {
        self.values.push_back(value);
        if self.values.len() > self.window_size {
            self.values.pop_front();
        }
    }

    /// Get the mean of values in the window
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    /// Get the variance of values in the window
    pub fn variance(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }
        let mean = self.mean()?;
        let sum_sq: f64 = self.values.iter().map(|v| (v - mean).powi(2)).sum();
        Some(sum_sq / (self.values.len() - 1) as f64)
    }

    /// Get the standard deviation
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Get the minimum value in the window
    pub fn min(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    /// Get the maximum value in the window
    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    /// Check if window is full
    pub fn is_full(&self) -> bool {
        self.values.len() >= self.window_size
    }

    /// Get number of values in window
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear the window
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

/// Fitness-based adaptive mutation rate
///
/// Adapts mutation rate based on whether mutations are producing improvements.
#[derive(Clone, Debug)]
pub struct AdaptiveMutationRate {
    /// Current mutation rate
    pub rate: f64,
    /// Minimum mutation rate
    pub min_rate: f64,
    /// Maximum mutation rate
    pub max_rate: f64,
    /// Increase factor when improvements are rare
    pub increase_factor: f64,
    /// Decrease factor when improvements are common
    pub decrease_factor: f64,
    /// Statistics tracker
    stats: SlidingWindowStats,
    /// Improvement threshold
    improvement_threshold: f64,
}

impl AdaptiveMutationRate {
    /// Create a new adaptive mutation rate
    pub fn new(initial_rate: f64) -> Self {
        Self {
            rate: initial_rate,
            min_rate: 0.001,
            max_rate: 0.5,
            increase_factor: 1.1,
            decrease_factor: 0.9,
            stats: SlidingWindowStats::new(20),
            improvement_threshold: 0.3, // Target 30% improvement rate
        }
    }

    /// Record a mutation outcome
    pub fn record(&mut self, improved: bool) {
        self.stats.push(if improved { 1.0 } else { 0.0 });
    }

    /// Adapt the mutation rate based on recent history
    pub fn adapt(&mut self) {
        if !self.stats.is_full() {
            return;
        }

        let improvement_rate = self.stats.mean().unwrap_or(0.0);

        if improvement_rate < self.improvement_threshold {
            // Not enough improvements, increase mutation rate
            self.rate = (self.rate * self.increase_factor).min(self.max_rate);
        } else if improvement_rate > self.improvement_threshold * 1.5 {
            // Too many improvements (might be too disruptive), decrease
            self.rate = (self.rate * self.decrease_factor).max(self.min_rate);
        }
    }

    /// Get current rate
    pub fn current_rate(&self) -> f64 {
        self.rate
    }
}

/// Population diversity-based parameter adaptation
#[derive(Clone, Debug)]
pub struct DiversityBasedAdaptation {
    /// Window for tracking diversity
    diversity_history: SlidingWindowStats,
    /// Target diversity level
    pub target_diversity: f64,
    /// Tolerance around target
    pub tolerance: f64,
}

impl DiversityBasedAdaptation {
    /// Create a new diversity-based adapter
    pub fn new(target_diversity: f64) -> Self {
        Self {
            diversity_history: SlidingWindowStats::new(10),
            target_diversity,
            tolerance: 0.1,
        }
    }

    /// Record current diversity
    pub fn record_diversity(&mut self, diversity: f64) {
        self.diversity_history.push(diversity);
    }

    /// Get recommended mutation rate multiplier
    ///
    /// Returns > 1.0 if diversity is too low, < 1.0 if too high
    pub fn mutation_multiplier(&self) -> f64 {
        let Some(current_diversity) = self.diversity_history.mean() else {
            return 1.0;
        };

        if current_diversity < self.target_diversity * (1.0 - self.tolerance) {
            // Diversity too low, increase mutation
            1.5
        } else if current_diversity > self.target_diversity * (1.0 + self.tolerance) {
            // Diversity too high, decrease mutation
            0.8
        } else {
            1.0
        }
    }

    /// Get recommended selection pressure multiplier
    ///
    /// Returns > 1.0 if diversity is too high, < 1.0 if too low
    pub fn selection_pressure_multiplier(&self) -> f64 {
        let Some(current_diversity) = self.diversity_history.mean() else {
            return 1.0;
        };

        if current_diversity < self.target_diversity * (1.0 - self.tolerance) {
            // Diversity too low, reduce selection pressure
            0.8
        } else if current_diversity > self.target_diversity * (1.0 + self.tolerance) {
            // Diversity too high, increase selection pressure
            1.2
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_fifth_rule_increase() {
        let mut rule = OneFifthRule::new().with_window_size(5);

        // All successes -> increase
        for _ in 0..5 {
            rule.record(true);
        }

        let sigma = 1.0;
        let new_sigma = rule.adapt(sigma);
        assert!(new_sigma > sigma);
    }

    #[test]
    fn test_one_fifth_rule_decrease() {
        let mut rule = OneFifthRule::new().with_window_size(5);

        // All failures -> decrease
        for _ in 0..5 {
            rule.record(false);
        }

        let sigma = 1.0;
        let new_sigma = rule.adapt(sigma);
        assert!(new_sigma < sigma);
    }

    #[test]
    fn test_one_fifth_rule_at_target() {
        let mut rule = OneFifthRule::new().with_window_size(5).with_target_rate(0.2);

        // Exactly 1/5 success rate
        rule.record(true);
        for _ in 0..4 {
            rule.record(false);
        }

        let sigma = 1.0;
        let new_sigma = rule.adapt(sigma);
        assert!((new_sigma - sigma).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_operator_selection() {
        let mut aos = AdaptiveOperatorSelection::new(3);
        let mut rng = rand::thread_rng();

        // Initially uniform
        assert_eq!(aos.probabilities().len(), 3);
        for &p in aos.probabilities() {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }

        // Update with reward for operator 0
        aos.update(0, 10.0);

        // Operator 0 should have higher weight now
        assert!(aos.probabilities()[0] > aos.probabilities()[1]);

        // Selection should work
        let _ = aos.select(&mut rng);
    }

    #[test]
    fn test_sliding_window_stats() {
        let mut stats = SlidingWindowStats::new(5);

        assert!(stats.mean().is_none());

        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);

        assert!((stats.mean().unwrap() - 2.0).abs() < 1e-10);
        assert!((stats.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((stats.max().unwrap() - 3.0).abs() < 1e-10);

        // Fill window
        stats.push(4.0);
        stats.push(5.0);
        assert!(stats.is_full());

        // Add more, should drop oldest
        stats.push(6.0);
        assert_eq!(stats.len(), 5);
        assert!((stats.min().unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_mutation_rate() {
        let mut amr = AdaptiveMutationRate::new(0.1);

        // Record no improvements
        for _ in 0..25 {
            amr.record(false);
        }
        amr.adapt();

        // Rate should increase
        assert!(amr.current_rate() > 0.1);
    }

    #[test]
    fn test_diversity_based_adaptation() {
        let mut dba = DiversityBasedAdaptation::new(0.5);

        // Record low diversity
        for _ in 0..10 {
            dba.record_diversity(0.2);
        }

        // Should recommend higher mutation
        assert!(dba.mutation_multiplier() > 1.0);
        // Should recommend lower selection pressure
        assert!(dba.selection_pressure_multiplier() < 1.0);
    }
}
