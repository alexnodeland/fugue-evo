//! Convergence detection for evolutionary algorithms
//!
//! This module provides various methods to detect when an evolutionary algorithm
//! has converged or should terminate.

use serde::{Deserialize, Serialize};

/// Result of a convergence check
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Algorithm has not converged
    NotConverged,
    /// Algorithm has converged with a reason
    Converged(ConvergenceReason),
}

impl ConvergenceStatus {
    /// Check if converged
    pub fn is_converged(&self) -> bool {
        matches!(self, Self::Converged(_))
    }
}

/// Reason for convergence
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceReason {
    /// Fitness has not improved for many generations
    FitnessStagnation { generations: usize },
    /// Population diversity is below threshold
    LowDiversity { diversity: u64 }, // Store as bits for Eq
    /// Target fitness reached
    TargetReached { target: u64 }, // Store as bits for Eq
    /// Maximum generations reached
    MaxGenerations { generations: usize },
    /// Maximum evaluations reached
    MaxEvaluations { evaluations: usize },
    /// R-hat statistic indicates convergence
    RhatConverged { rhat: u64 }, // Store as bits for Eq
    /// Multiple criteria satisfied
    MultipleReasons(Vec<ConvergenceReason>),
    /// Custom termination
    Custom(String),
}

impl ConvergenceReason {
    /// Create a fitness stagnation reason
    pub fn fitness_stagnation(generations: usize) -> Self {
        Self::FitnessStagnation { generations }
    }

    /// Create a low diversity reason
    pub fn low_diversity(diversity: f64) -> Self {
        Self::LowDiversity {
            diversity: diversity.to_bits(),
        }
    }

    /// Create a target reached reason
    pub fn target_reached(target: f64) -> Self {
        Self::TargetReached {
            target: target.to_bits(),
        }
    }

    /// Create an R-hat converged reason
    pub fn rhat_converged(rhat: f64) -> Self {
        Self::RhatConverged {
            rhat: rhat.to_bits(),
        }
    }
}

/// Configuration for convergence detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Maximum generations before termination
    pub max_generations: Option<usize>,
    /// Maximum fitness evaluations before termination
    pub max_evaluations: Option<usize>,
    /// Target fitness to reach
    pub target_fitness: Option<f64>,
    /// Tolerance for target fitness comparison
    pub target_tolerance: f64,
    /// Number of generations without improvement before stagnation
    pub stagnation_generations: usize,
    /// Minimum improvement to not count as stagnation
    pub stagnation_threshold: f64,
    /// Diversity threshold below which convergence is detected
    pub diversity_threshold: f64,
    /// R-hat threshold for convergence (typically 1.1)
    pub rhat_threshold: f64,
    /// Whether to use R-hat based convergence
    pub use_rhat: bool,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            max_generations: None,
            max_evaluations: None,
            target_fitness: None,
            target_tolerance: 1e-6,
            stagnation_generations: 50,
            stagnation_threshold: 1e-9,
            diversity_threshold: 0.01,
            rhat_threshold: 1.1,
            use_rhat: false,
        }
    }
}

impl ConvergenceConfig {
    /// Create a new config with max generations
    pub fn with_max_generations(generations: usize) -> Self {
        Self {
            max_generations: Some(generations),
            ..Default::default()
        }
    }

    /// Set max generations
    pub fn max_generations(mut self, generations: usize) -> Self {
        self.max_generations = Some(generations);
        self
    }

    /// Set max evaluations
    pub fn max_evaluations(mut self, evaluations: usize) -> Self {
        self.max_evaluations = Some(evaluations);
        self
    }

    /// Set target fitness
    pub fn target_fitness(mut self, target: f64) -> Self {
        self.target_fitness = Some(target);
        self
    }

    /// Set target tolerance
    pub fn target_tolerance(mut self, tolerance: f64) -> Self {
        self.target_tolerance = tolerance;
        self
    }

    /// Set stagnation detection parameters
    pub fn stagnation(mut self, generations: usize, threshold: f64) -> Self {
        self.stagnation_generations = generations;
        self.stagnation_threshold = threshold;
        self
    }

    /// Set diversity threshold
    pub fn diversity_threshold(mut self, threshold: f64) -> Self {
        self.diversity_threshold = threshold;
        self
    }

    /// Enable R-hat based convergence
    pub fn with_rhat(mut self, threshold: f64) -> Self {
        self.use_rhat = true;
        self.rhat_threshold = threshold;
        self
    }
}

/// Convergence detector that tracks evolution state
#[derive(Clone, Debug)]
pub struct ConvergenceDetector {
    /// Configuration
    config: ConvergenceConfig,
    /// History of best fitness values
    best_fitness_history: Vec<f64>,
    /// History of mean fitness values (for R-hat)
    mean_fitness_history: Vec<f64>,
    /// History of diversity values
    diversity_history: Vec<f64>,
    /// Current generation
    current_generation: usize,
    /// Current evaluations
    current_evaluations: usize,
    /// Best fitness seen so far
    best_fitness_overall: f64,
    /// Generation when best fitness was last improved
    last_improvement_generation: usize,
}

impl ConvergenceDetector {
    /// Create a new convergence detector
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            config,
            best_fitness_history: Vec::new(),
            mean_fitness_history: Vec::new(),
            diversity_history: Vec::new(),
            current_generation: 0,
            current_evaluations: 0,
            best_fitness_overall: f64::NEG_INFINITY,
            last_improvement_generation: 0,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(ConvergenceConfig::default())
    }

    /// Update with generation statistics
    pub fn update(
        &mut self,
        generation: usize,
        evaluations: usize,
        best_fitness: f64,
        mean_fitness: f64,
        diversity: f64,
    ) {
        self.current_generation = generation;
        self.current_evaluations = evaluations;
        self.best_fitness_history.push(best_fitness);
        self.mean_fitness_history.push(mean_fitness);
        self.diversity_history.push(diversity);

        // Track improvement
        if best_fitness > self.best_fitness_overall + self.config.stagnation_threshold {
            self.best_fitness_overall = best_fitness;
            self.last_improvement_generation = generation;
        }
    }

    /// Check if algorithm has converged
    pub fn check(&self) -> ConvergenceStatus {
        let mut reasons = Vec::new();

        // Check max generations
        if let Some(max_gen) = self.config.max_generations {
            if self.current_generation >= max_gen {
                reasons.push(ConvergenceReason::MaxGenerations {
                    generations: self.current_generation,
                });
            }
        }

        // Check max evaluations
        if let Some(max_eval) = self.config.max_evaluations {
            if self.current_evaluations >= max_eval {
                reasons.push(ConvergenceReason::MaxEvaluations {
                    evaluations: self.current_evaluations,
                });
            }
        }

        // Check target fitness
        if let Some(target) = self.config.target_fitness {
            if let Some(&best) = self.best_fitness_history.last() {
                if (best - target).abs() <= self.config.target_tolerance || best >= target {
                    reasons.push(ConvergenceReason::target_reached(best));
                }
            }
        }

        // Check stagnation
        let generations_since_improvement =
            self.current_generation - self.last_improvement_generation;
        if generations_since_improvement >= self.config.stagnation_generations {
            reasons.push(ConvergenceReason::fitness_stagnation(
                generations_since_improvement,
            ));
        }

        // Check diversity
        if let Some(&diversity) = self.diversity_history.last() {
            if diversity < self.config.diversity_threshold {
                reasons.push(ConvergenceReason::low_diversity(diversity));
            }
        }

        // Check R-hat if enabled
        if self.config.use_rhat && self.mean_fitness_history.len() >= 10 {
            // Split history into "chains" for R-hat calculation
            let rhat = self.compute_rhat();
            if rhat < self.config.rhat_threshold {
                reasons.push(ConvergenceReason::rhat_converged(rhat));
            }
        }

        // Return result
        match reasons.len() {
            0 => ConvergenceStatus::NotConverged,
            1 => ConvergenceStatus::Converged(reasons.pop().unwrap()),
            _ => ConvergenceStatus::Converged(ConvergenceReason::MultipleReasons(reasons)),
        }
    }

    /// Compute R-hat statistic from fitness history
    fn compute_rhat(&self) -> f64 {
        // Split history into two "chains"
        let n = self.mean_fitness_history.len();
        let half = n / 2;

        if half < 5 {
            return f64::INFINITY; // Not enough data
        }

        let chain1: Vec<f64> = self.mean_fitness_history[..half].to_vec();
        let chain2: Vec<f64> = self.mean_fitness_history[half..].to_vec();

        evolutionary_rhat(&[chain1, chain2])
    }

    /// Get the best fitness seen
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness_overall
    }

    /// Get generations since last improvement
    pub fn generations_without_improvement(&self) -> usize {
        self.current_generation - self.last_improvement_generation
    }

    /// Get the latest diversity value
    pub fn current_diversity(&self) -> Option<f64> {
        self.diversity_history.last().copied()
    }

    /// Get the fitness history
    pub fn fitness_history(&self) -> &[f64] {
        &self.best_fitness_history
    }

    /// Get the diversity history
    pub fn diversity_history(&self) -> &[f64] {
        &self.diversity_history
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.best_fitness_history.clear();
        self.mean_fitness_history.clear();
        self.diversity_history.clear();
        self.current_generation = 0;
        self.current_evaluations = 0;
        self.best_fitness_overall = f64::NEG_INFINITY;
        self.last_improvement_generation = 0;
    }
}

/// R-hat analog for evolutionary convergence
///
/// Compares fitness distributions across multiple runs/chains.
/// Values close to 1.0 indicate convergence.
pub fn evolutionary_rhat(runs: &[Vec<f64>]) -> f64 {
    if runs.is_empty() || runs[0].is_empty() {
        return f64::INFINITY;
    }

    let m = runs.len() as f64;
    let n = runs.iter().map(|r| r.len()).min().unwrap_or(0) as f64;

    if n < 2.0 || m < 2.0 {
        return f64::INFINITY;
    }

    // Between-chain variance
    let chain_means: Vec<f64> = runs.iter().map(|r| r.iter().sum::<f64>() / n).collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0)
        * chain_means
            .iter()
            .map(|cm| (cm - grand_mean).powi(2))
            .sum::<f64>();

    // Within-chain variance
    let w: f64 = runs
        .iter()
        .map(|r| {
            let mean = r.iter().sum::<f64>() / n;
            r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>()
        / m;

    if w == 0.0 {
        return 1.0; // Perfect convergence
    }

    // Pooled variance estimate
    let var_plus = ((n - 1.0) / n) * w + b / n;

    (var_plus / w).sqrt()
}

/// Effective Sample Size (ESS) for evolutionary SMC
///
/// Measures the effective number of independent samples based on importance weights.
pub fn evolutionary_ess(weights: &[f64]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }

    // Normalize weights
    let sum: f64 = weights.iter().sum();
    if sum == 0.0 {
        return weights.len() as f64;
    }

    let normalized: Vec<f64> = weights.iter().map(|w| w / sum).collect();
    let sum_sq: f64 = normalized.iter().map(|w| w * w).sum();

    if sum_sq == 0.0 {
        weights.len() as f64
    } else {
        1.0 / sum_sq
    }
}

/// Effective Sample Size from log weights
pub fn evolutionary_ess_log(log_weights: &[f64]) -> f64 {
    if log_weights.is_empty() {
        return 0.0;
    }

    // Use log-sum-exp trick for numerical stability
    let max_log = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_log.is_infinite() {
        return log_weights.len() as f64;
    }

    let weights: Vec<f64> = log_weights.iter().map(|lw| (lw - max_log).exp()).collect();
    evolutionary_ess(&weights)
}

/// Detect fitness stagnation in a history of fitness values
///
/// Returns the number of generations the fitness has been stagnant.
pub fn detect_stagnation(fitness_history: &[f64], threshold: f64) -> usize {
    if fitness_history.len() < 2 {
        return 0;
    }

    let best = fitness_history
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Count generations since best was improved
    let mut stagnant_count: usize = 0;
    for &fitness in fitness_history.iter().rev() {
        if (fitness - best).abs() <= threshold {
            stagnant_count += 1;
        } else {
            break;
        }
    }

    stagnant_count.saturating_sub(1) // Don't count the best itself
}

/// Compute population convergence from fitness values
///
/// Returns a value between 0 (no convergence) and 1 (perfect convergence)
/// based on the coefficient of variation of fitness values.
pub fn fitness_convergence(fitness_values: &[f64]) -> f64 {
    if fitness_values.len() < 2 {
        return 1.0;
    }

    let mean = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
    if mean.abs() < f64::EPSILON {
        return 1.0;
    }

    let variance = fitness_values
        .iter()
        .map(|f| (f - mean).powi(2))
        .sum::<f64>()
        / (fitness_values.len() - 1) as f64;
    let std = variance.sqrt();

    // Coefficient of variation (CV)
    let cv = std / mean.abs();

    // Convert to convergence metric (higher = more converged)
    // CV of 0 means perfect convergence
    // Use exponential decay so that small CV gives high convergence
    (-cv).exp()
}

/// Termination criteria for evolutionary algorithms
#[derive(Clone, Debug)]
pub struct TerminationCriteria {
    criteria: Vec<TerminationCriterion>,
    require_all: bool,
}

/// A single termination criterion
#[derive(Clone, Debug)]
pub enum TerminationCriterion {
    /// Maximum generations
    MaxGenerations(usize),
    /// Maximum evaluations
    MaxEvaluations(usize),
    /// Target fitness (maximize)
    TargetFitness(f64, f64), // (target, tolerance)
    /// Fitness stagnation
    Stagnation(usize, f64), // (generations, threshold)
    /// Diversity threshold
    DiversityThreshold(f64),
    /// Time limit in seconds
    TimeLimit(f64),
    /// Custom predicate
    Custom(String), // Description only, evaluation handled externally
}

impl TerminationCriteria {
    /// Create new empty criteria (any criterion triggers termination)
    pub fn new() -> Self {
        Self {
            criteria: Vec::new(),
            require_all: false,
        }
    }

    /// Create criteria where all must be satisfied
    pub fn require_all() -> Self {
        Self {
            criteria: Vec::new(),
            require_all: true,
        }
    }

    /// Add a criterion
    pub fn add(mut self, criterion: TerminationCriterion) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Add max generations criterion
    pub fn max_generations(self, generations: usize) -> Self {
        self.add(TerminationCriterion::MaxGenerations(generations))
    }

    /// Add max evaluations criterion
    pub fn max_evaluations(self, evaluations: usize) -> Self {
        self.add(TerminationCriterion::MaxEvaluations(evaluations))
    }

    /// Add target fitness criterion
    pub fn target_fitness(self, target: f64, tolerance: f64) -> Self {
        self.add(TerminationCriterion::TargetFitness(target, tolerance))
    }

    /// Add stagnation criterion
    pub fn stagnation(self, generations: usize, threshold: f64) -> Self {
        self.add(TerminationCriterion::Stagnation(generations, threshold))
    }

    /// Add diversity threshold criterion
    pub fn diversity_threshold(self, threshold: f64) -> Self {
        self.add(TerminationCriterion::DiversityThreshold(threshold))
    }

    /// Add time limit criterion
    pub fn time_limit(self, seconds: f64) -> Self {
        self.add(TerminationCriterion::TimeLimit(seconds))
    }

    /// Check if termination criteria are met
    pub fn should_terminate(
        &self,
        generation: usize,
        evaluations: usize,
        best_fitness: f64,
        diversity: f64,
        stagnation_generations: usize,
        elapsed_seconds: f64,
    ) -> Option<ConvergenceReason> {
        let mut satisfied = Vec::new();

        for criterion in &self.criteria {
            let met = match criterion {
                TerminationCriterion::MaxGenerations(max) => generation >= *max,
                TerminationCriterion::MaxEvaluations(max) => evaluations >= *max,
                TerminationCriterion::TargetFitness(target, tolerance) => {
                    (best_fitness - target).abs() <= *tolerance || best_fitness >= *target
                }
                TerminationCriterion::Stagnation(gens, _threshold) => {
                    stagnation_generations >= *gens
                }
                TerminationCriterion::DiversityThreshold(thresh) => diversity < *thresh,
                TerminationCriterion::TimeLimit(limit) => elapsed_seconds >= *limit,
                TerminationCriterion::Custom(_) => false, // Handled externally
            };

            if met {
                satisfied.push(criterion.to_reason(
                    generation,
                    evaluations,
                    best_fitness,
                    diversity,
                ));
            }
        }

        if satisfied.is_empty() {
            return None;
        }

        if self.require_all && satisfied.len() < self.criteria.len() {
            return None;
        }

        // Return the reason(s)
        if satisfied.len() == 1 {
            Some(satisfied.pop().unwrap())
        } else {
            Some(ConvergenceReason::MultipleReasons(satisfied))
        }
    }

    /// Get all criteria
    pub fn criteria(&self) -> &[TerminationCriterion] {
        &self.criteria
    }
}

impl Default for TerminationCriteria {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminationCriterion {
    fn to_reason(
        &self,
        generation: usize,
        evaluations: usize,
        best_fitness: f64,
        diversity: f64,
    ) -> ConvergenceReason {
        match self {
            Self::MaxGenerations(_) => ConvergenceReason::MaxGenerations {
                generations: generation,
            },
            Self::MaxEvaluations(_) => ConvergenceReason::MaxEvaluations { evaluations },
            Self::TargetFitness(_, _) => ConvergenceReason::target_reached(best_fitness),
            Self::Stagnation(gens, _) => ConvergenceReason::fitness_stagnation(*gens),
            Self::DiversityThreshold(_) => ConvergenceReason::low_diversity(diversity),
            Self::TimeLimit(t) => ConvergenceReason::Custom(format!("Time limit of {t}s reached")),
            Self::Custom(desc) => ConvergenceReason::Custom(desc.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detector_basic() {
        let config = ConvergenceConfig::with_max_generations(100);
        let mut detector = ConvergenceDetector::new(config);

        // Simulate improving fitness
        for i in 0..50 {
            detector.update(i, i * 10, i as f64, i as f64 * 0.5, 0.5);
        }

        let status = detector.check();
        assert!(!status.is_converged());
    }

    #[test]
    fn test_convergence_detector_max_generations() {
        let config = ConvergenceConfig::with_max_generations(50);
        let mut detector = ConvergenceDetector::new(config);

        for i in 0..60 {
            detector.update(i, i * 10, i as f64, i as f64 * 0.5, 0.5);
        }

        let status = detector.check();
        assert!(status.is_converged());
        if let ConvergenceStatus::Converged(reason) = status {
            assert!(matches!(reason, ConvergenceReason::MaxGenerations { .. }));
        }
    }

    #[test]
    fn test_convergence_detector_target_fitness() {
        let config = ConvergenceConfig::default()
            .target_fitness(100.0)
            .target_tolerance(1.0);
        let mut detector = ConvergenceDetector::new(config);

        detector.update(0, 10, 99.5, 50.0, 0.5);

        let status = detector.check();
        assert!(status.is_converged());
    }

    #[test]
    fn test_convergence_detector_stagnation() {
        let config = ConvergenceConfig::default().stagnation(10, 1e-9);
        let mut detector = ConvergenceDetector::new(config);

        // First improvement
        detector.update(0, 10, 50.0, 50.0, 0.5);

        // Then stagnation
        for i in 1..20 {
            detector.update(i, i * 10, 50.0, 50.0, 0.5);
        }

        let status = detector.check();
        assert!(status.is_converged());
        if let ConvergenceStatus::Converged(reason) = status {
            assert!(matches!(
                reason,
                ConvergenceReason::FitnessStagnation { .. }
            ));
        }
    }

    #[test]
    fn test_convergence_detector_low_diversity() {
        let config = ConvergenceConfig::default().diversity_threshold(0.1);
        let mut detector = ConvergenceDetector::new(config);

        detector.update(0, 10, 50.0, 50.0, 0.05);

        let status = detector.check();
        assert!(status.is_converged());
        if let ConvergenceStatus::Converged(reason) = status {
            assert!(matches!(reason, ConvergenceReason::LowDiversity { .. }));
        }
    }

    #[test]
    fn test_evolutionary_rhat() {
        // Similar chains with some variation should give R-hat close to 1
        let chain1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let chain2 = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let rhat = evolutionary_rhat(&[chain1, chain2]);
        // R-hat should be close to 1 for similar chains (typically < 1.1 for convergence)
        assert!(rhat < 1.2, "R-hat was {}, expected < 1.2", rhat);
    }

    #[test]
    fn test_evolutionary_rhat_divergent() {
        // Very different chains with internal variation should give high R-hat
        let chain1 = vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.8, 1.8, 2.2, 1.3, 2.7];
        let chain2 = vec![
            100.0, 101.0, 100.5, 101.5, 100.2, 101.8, 100.8, 101.2, 100.3, 101.7,
        ];
        let rhat = evolutionary_rhat(&[chain1, chain2]);
        // R-hat should be high for divergent chains
        assert!(rhat > 1.5, "R-hat was {}, expected > 1.5", rhat);
    }

    #[test]
    fn test_evolutionary_ess() {
        // Equal weights should give ESS = n
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let ess = evolutionary_ess(&weights);
        assert!((ess - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_evolutionary_ess_unequal() {
        // One dominant weight should give low ESS
        let weights = vec![1.0, 0.0, 0.0, 0.0];
        let ess = evolutionary_ess(&weights);
        assert!((ess - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evolutionary_ess_log() {
        let log_weights = vec![0.0, 0.0, 0.0, 0.0];
        let ess = evolutionary_ess_log(&log_weights);
        assert!((ess - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_detect_stagnation() {
        let history = vec![10.0, 20.0, 30.0, 30.0, 30.0, 30.0];
        let stagnant = detect_stagnation(&history, 1e-9);
        assert_eq!(stagnant, 3); // 3 generations at max
    }

    #[test]
    fn test_detect_stagnation_improving() {
        let history = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stagnant = detect_stagnation(&history, 1e-9);
        assert_eq!(stagnant, 0);
    }

    #[test]
    fn test_fitness_convergence() {
        // All same fitness = perfect convergence
        let fitness = vec![50.0, 50.0, 50.0, 50.0];
        let conv = fitness_convergence(&fitness);
        assert!((conv - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fitness_convergence_diverse() {
        // Very diverse fitness = low convergence
        let fitness = vec![0.0, 100.0, 0.0, 100.0];
        let conv = fitness_convergence(&fitness);
        assert!(conv < 0.5);
    }

    #[test]
    fn test_termination_criteria_max_gen() {
        let criteria = TerminationCriteria::new().max_generations(100);

        let result = criteria.should_terminate(50, 500, 10.0, 0.5, 0, 10.0);
        assert!(result.is_none());

        let result = criteria.should_terminate(100, 1000, 10.0, 0.5, 0, 20.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_termination_criteria_target() {
        let criteria = TerminationCriteria::new().target_fitness(100.0, 1.0);

        let result = criteria.should_terminate(10, 100, 50.0, 0.5, 0, 5.0);
        assert!(result.is_none());

        let result = criteria.should_terminate(10, 100, 99.5, 0.5, 0, 5.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_termination_criteria_multiple() {
        let criteria = TerminationCriteria::new()
            .max_generations(100)
            .target_fitness(100.0, 1.0);

        // Neither met
        let result = criteria.should_terminate(10, 100, 50.0, 0.5, 0, 5.0);
        assert!(result.is_none());

        // Target met
        let result = criteria.should_terminate(10, 100, 100.0, 0.5, 0, 5.0);
        assert!(result.is_some());

        // Max gen met
        let result = criteria.should_terminate(100, 1000, 50.0, 0.5, 0, 50.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_termination_criteria_require_all() {
        let criteria = TerminationCriteria::require_all()
            .max_generations(100)
            .stagnation(10, 1e-9);

        // Only max gen met
        let result = criteria.should_terminate(100, 1000, 50.0, 0.5, 5, 50.0);
        assert!(result.is_none());

        // Both met
        let result = criteria.should_terminate(100, 1000, 50.0, 0.5, 10, 50.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_convergence_config_builder() {
        let config = ConvergenceConfig::with_max_generations(500)
            .max_evaluations(10000)
            .target_fitness(1.0)
            .target_tolerance(0.01)
            .stagnation(100, 1e-6)
            .diversity_threshold(0.05)
            .with_rhat(1.05);

        assert_eq!(config.max_generations, Some(500));
        assert_eq!(config.max_evaluations, Some(10000));
        assert_eq!(config.target_fitness, Some(1.0));
        assert_eq!(config.target_tolerance, 0.01);
        assert_eq!(config.stagnation_generations, 100);
        assert_eq!(config.stagnation_threshold, 1e-6);
        assert_eq!(config.diversity_threshold, 0.05);
        assert!(config.use_rhat);
        assert_eq!(config.rhat_threshold, 1.05);
    }

    #[test]
    fn test_convergence_detector_reset() {
        let config = ConvergenceConfig::default();
        let mut detector = ConvergenceDetector::new(config);

        detector.update(0, 10, 50.0, 50.0, 0.5);
        detector.update(1, 20, 60.0, 55.0, 0.4);

        assert_eq!(detector.fitness_history().len(), 2);
        assert_eq!(detector.best_fitness(), 60.0);

        detector.reset();

        assert!(detector.fitness_history().is_empty());
        assert_eq!(detector.best_fitness(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_convergence_status_is_converged() {
        let not_converged = ConvergenceStatus::NotConverged;
        assert!(!not_converged.is_converged());

        let converged =
            ConvergenceStatus::Converged(ConvergenceReason::MaxGenerations { generations: 100 });
        assert!(converged.is_converged());
    }
}
