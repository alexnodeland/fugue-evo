//! Bayesian hyperparameter learning
//!
//! Online Bayesian inference for learning optimal hyperparameters using
//! conjugate prior distributions.

use std::collections::VecDeque;
use rand::Rng;
use rand_distr::{Beta, Gamma, Distribution};

/// Beta distribution posterior for probability parameters (e.g., mutation rate)
#[derive(Clone, Debug)]
pub struct BetaPosterior {
    /// Alpha parameter (pseudo-count of successes)
    pub alpha: f64,
    /// Beta parameter (pseudo-count of failures)
    pub beta: f64,
}

impl BetaPosterior {
    /// Create with uniform prior (α = β = 1)
    pub fn uniform() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    /// Create with Jeffreys prior (α = β = 0.5)
    pub fn jeffreys() -> Self {
        Self { alpha: 0.5, beta: 0.5 }
    }

    /// Create with custom prior
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Update posterior with a success observation
    pub fn observe_success(&mut self) {
        self.alpha += 1.0;
    }

    /// Update posterior with a failure observation
    pub fn observe_failure(&mut self) {
        self.beta += 1.0;
    }

    /// Update based on boolean outcome
    pub fn observe(&mut self, success: bool) {
        if success {
            self.observe_success();
        } else {
            self.observe_failure();
        }
    }

    /// Posterior mean
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior mode (for α, β > 1)
    pub fn mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Posterior variance
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    /// Posterior standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample from the posterior
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Beta::new(self.alpha, self.beta)
            .expect("Invalid Beta parameters")
            .sample(rng)
    }

    /// 95% credible interval (approximate using normal approximation for large counts)
    pub fn credible_interval(&self, probability: f64) -> (f64, f64) {
        let mean = self.mean();
        let std = self.std_dev();
        let z = normal_quantile((1.0 + probability) / 2.0);
        let lower = (mean - z * std).max(0.0);
        let upper = (mean + z * std).min(1.0);
        (lower, upper)
    }

    /// Number of observations
    pub fn observations(&self) -> f64 {
        self.alpha + self.beta - 2.0 // Subtract prior pseudo-counts for uniform
    }

    /// Apply decay to move toward prior (for non-stationary environments)
    pub fn decay(&mut self, factor: f64) {
        // Move α and β toward 1 (uniform prior)
        self.alpha = 1.0 + factor * (self.alpha - 1.0);
        self.beta = 1.0 + factor * (self.beta - 1.0);
    }
}

impl Default for BetaPosterior {
    fn default() -> Self {
        Self::uniform()
    }
}

/// Gamma distribution posterior for positive rate parameters (e.g., temperature)
#[derive(Clone, Debug)]
pub struct GammaPosterior {
    /// Shape parameter (α)
    pub shape: f64,
    /// Rate parameter (β)
    pub rate: f64,
}

impl GammaPosterior {
    /// Create with vague prior
    pub fn vague() -> Self {
        Self { shape: 1.0, rate: 0.01 }
    }

    /// Create with custom prior
    pub fn new(shape: f64, rate: f64) -> Self {
        Self { shape, rate }
    }

    /// Update with an observation (conjugate update for exponential likelihood)
    pub fn observe(&mut self, value: f64) {
        self.shape += 1.0;
        self.rate += value;
    }

    /// Posterior mean
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Posterior mode (for shape > 1)
    pub fn mode(&self) -> Option<f64> {
        if self.shape >= 1.0 {
            Some((self.shape - 1.0) / self.rate)
        } else {
            None
        }
    }

    /// Posterior variance
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Sample from the posterior
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Gamma::new(self.shape, 1.0 / self.rate)
            .expect("Invalid Gamma parameters")
            .sample(rng)
    }

    /// Apply decay
    pub fn decay(&mut self, factor: f64) {
        self.shape = 1.0 + factor * (self.shape - 1.0);
        self.rate = 0.01 + factor * (self.rate - 0.01);
    }
}

impl Default for GammaPosterior {
    fn default() -> Self {
        Self::vague()
    }
}

/// Log-normal posterior approximation for step sizes
#[derive(Clone, Debug)]
pub struct LogNormalPosterior {
    /// Mean of log(σ)
    pub mu: f64,
    /// Variance of log(σ)
    pub sigma_sq: f64,
    /// Number of observations
    pub n: usize,
}

impl LogNormalPosterior {
    /// Create with vague prior
    pub fn vague() -> Self {
        Self {
            mu: 0.0,
            sigma_sq: 1.0,
            n: 0,
        }
    }

    /// Create with informative prior
    pub fn new(mu: f64, sigma_sq: f64) -> Self {
        Self { mu, sigma_sq, n: 0 }
    }

    /// Update with an observation of a step size
    pub fn observe(&mut self, sigma: f64) {
        if sigma <= 0.0 {
            return;
        }

        let log_sigma = sigma.ln();
        self.n += 1;

        // Online update of mean and variance
        let delta = log_sigma - self.mu;
        self.mu += delta / self.n as f64;
        // Note: This is a simplified update, not fully Bayesian
        if self.n > 1 {
            self.sigma_sq = (self.sigma_sq * (self.n - 1) as f64 + delta * (log_sigma - self.mu)) / self.n as f64;
        }
    }

    /// Mean of the distribution (in original space)
    pub fn mean(&self) -> f64 {
        (self.mu + self.sigma_sq / 2.0).exp()
    }

    /// Mode of the distribution (in original space)
    pub fn mode(&self) -> f64 {
        (self.mu - self.sigma_sq).exp()
    }

    /// Sample from the posterior
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        use rand_distr::StandardNormal;
        let z: f64 = rng.sample(StandardNormal);
        (self.mu + self.sigma_sq.sqrt() * z).exp()
    }
}

impl Default for LogNormalPosterior {
    fn default() -> Self {
        Self::vague()
    }
}

/// Collection of hyperparameter posteriors
#[derive(Clone, Debug, Default)]
pub struct HyperparameterPosteriors {
    /// Mutation rate posterior
    pub mutation_rate: BetaPosterior,
    /// Crossover probability posterior
    pub crossover_prob: BetaPosterior,
    /// Selection temperature posterior
    pub temperature: GammaPosterior,
    /// SBX distribution index posterior
    pub sbx_eta: GammaPosterior,
    /// Polynomial mutation eta posterior
    pub pm_eta: GammaPosterior,
    /// Step sizes posteriors
    pub step_sizes: Vec<LogNormalPosterior>,
}

impl HyperparameterPosteriors {
    /// Create with default (vague) priors
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with specified number of step size parameters
    pub fn with_step_sizes(n: usize) -> Self {
        Self {
            step_sizes: vec![LogNormalPosterior::vague(); n],
            ..Default::default()
        }
    }

    /// Apply decay to all posteriors
    pub fn decay_all(&mut self, factor: f64) {
        self.mutation_rate.decay(factor);
        self.crossover_prob.decay(factor);
        self.temperature.decay(factor);
        self.sbx_eta.decay(factor);
        self.pm_eta.decay(factor);
    }
}

/// Operator parameters that can be learned
#[derive(Clone, Debug)]
pub struct OperatorParams {
    /// Mutation rate
    pub mutation_rate: f64,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Selection temperature
    pub temperature: f64,
    /// SBX distribution index
    pub sbx_eta: f64,
    /// Polynomial mutation distribution index
    pub pm_eta: f64,
}

impl Default for OperatorParams {
    fn default() -> Self {
        Self {
            mutation_rate: 0.1,
            crossover_prob: 0.9,
            temperature: 1.0,
            sbx_eta: 20.0,
            pm_eta: 20.0,
        }
    }
}

impl OperatorParams {
    /// Sample parameters from posteriors
    pub fn sample_from<R: Rng>(posteriors: &HyperparameterPosteriors, rng: &mut R) -> Self {
        Self {
            mutation_rate: posteriors.mutation_rate.sample(rng),
            crossover_prob: posteriors.crossover_prob.sample(rng),
            temperature: posteriors.temperature.sample(rng).max(0.01),
            sbx_eta: posteriors.sbx_eta.sample(rng).max(1.0),
            pm_eta: posteriors.pm_eta.sample(rng).max(1.0),
        }
    }

    /// Get MAP (maximum a posteriori) estimate from posteriors
    pub fn map_estimate(posteriors: &HyperparameterPosteriors) -> Self {
        Self {
            mutation_rate: posteriors.mutation_rate.mode().unwrap_or(posteriors.mutation_rate.mean()),
            crossover_prob: posteriors.crossover_prob.mode().unwrap_or(posteriors.crossover_prob.mean()),
            temperature: posteriors.temperature.mode().unwrap_or(posteriors.temperature.mean()).max(0.01),
            sbx_eta: posteriors.sbx_eta.mode().unwrap_or(posteriors.sbx_eta.mean()).max(1.0),
            pm_eta: posteriors.pm_eta.mode().unwrap_or(posteriors.pm_eta.mean()).max(1.0),
        }
    }
}

/// Online Bayesian hyperparameter learner
#[derive(Clone, Debug)]
pub struct BayesianHyperparameterLearner {
    /// Hyperparameter posteriors
    pub posteriors: HyperparameterPosteriors,
    /// Sliding window of observations
    history: VecDeque<(OperatorParams, f64)>,
    /// Maximum history size
    window_size: usize,
    /// Decay factor for non-stationary environments
    decay_factor: f64,
}

impl BayesianHyperparameterLearner {
    /// Create a new learner
    pub fn new() -> Self {
        Self {
            posteriors: HyperparameterPosteriors::new(),
            history: VecDeque::new(),
            window_size: 100,
            decay_factor: 1.0, // No decay by default
        }
    }

    /// Set window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set decay factor (< 1.0 for non-stationary environments)
    pub fn with_decay(mut self, factor: f64) -> Self {
        self.decay_factor = factor;
        self
    }

    /// Observe the outcome of applying operators with given parameters
    pub fn observe(&mut self, params: OperatorParams, parent_fitness: f64, child_fitness: f64) {
        let improvement = child_fitness - parent_fitness;
        let success = improvement > 0.0;

        // Update posteriors
        self.posteriors.mutation_rate.observe(success);
        self.posteriors.crossover_prob.observe(success);

        // For continuous parameters, observe them when successful
        if success {
            self.posteriors.temperature.observe(params.temperature);
            self.posteriors.sbx_eta.observe(params.sbx_eta);
            self.posteriors.pm_eta.observe(params.pm_eta);
        }

        // Maintain history
        self.history.push_back((params, improvement));
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        // Apply decay if configured
        if self.decay_factor < 1.0 {
            self.posteriors.decay_all(self.decay_factor);
        }
    }

    /// Sample parameters from current posteriors
    pub fn sample_params<R: Rng>(&self, rng: &mut R) -> OperatorParams {
        OperatorParams::sample_from(&self.posteriors, rng)
    }

    /// Get MAP estimate of parameters
    pub fn map_params(&self) -> OperatorParams {
        OperatorParams::map_estimate(&self.posteriors)
    }

    /// Get current posteriors
    pub fn posteriors(&self) -> &HyperparameterPosteriors {
        &self.posteriors
    }

    /// Get average improvement in history
    pub fn average_improvement(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().map(|(_, imp)| *imp).sum();
        Some(sum / self.history.len() as f64)
    }

    /// Reset the learner
    pub fn reset(&mut self) {
        self.posteriors = HyperparameterPosteriors::new();
        self.history.clear();
    }
}

impl Default for BayesianHyperparameterLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate normal quantile function
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation for normal quantile
    // Good enough for credible interval computation
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let q = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -q } else { q }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_posterior_uniform_prior() {
        let posterior = BetaPosterior::uniform();
        assert!((posterior.mean() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_beta_posterior_update() {
        let mut posterior = BetaPosterior::uniform();

        // Observe 7 successes, 3 failures
        for _ in 0..7 {
            posterior.observe_success();
        }
        for _ in 0..3 {
            posterior.observe_failure();
        }

        // Expected mean: (1 + 7) / (1 + 7 + 1 + 3) = 8/12 = 0.667
        assert!((posterior.mean() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_beta_posterior_sample() {
        let posterior = BetaPosterior::new(5.0, 5.0);
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let sample = posterior.sample(&mut rng);
            assert!(sample >= 0.0 && sample <= 1.0);
        }
    }

    #[test]
    fn test_gamma_posterior() {
        let mut posterior = GammaPosterior::vague();

        for _ in 0..10 {
            posterior.observe(1.0);
        }

        // Mean should be around 1.0 after observing 1.0s
        let mean = posterior.mean();
        assert!(mean > 0.5 && mean < 2.0);
    }

    #[test]
    fn test_log_normal_posterior() {
        let mut posterior = LogNormalPosterior::vague();

        // Observe step sizes around 0.1
        for _ in 0..10 {
            posterior.observe(0.1);
        }

        // Mean should be close to 0.1
        let mean = posterior.mean();
        assert!(mean > 0.05 && mean < 0.5);
    }

    #[test]
    fn test_bayesian_learner() {
        let mut learner = BayesianHyperparameterLearner::new();
        let mut rng = rand::thread_rng();

        // Simulate some observations
        for i in 0..20 {
            let params = OperatorParams::default();
            let parent_fitness = 0.0;
            let child_fitness = if i % 3 == 0 { 1.0 } else { -1.0 };

            learner.observe(params, parent_fitness, child_fitness);
        }

        // Should be able to sample params
        let sampled = learner.sample_params(&mut rng);
        assert!(sampled.mutation_rate >= 0.0 && sampled.mutation_rate <= 1.0);
        assert!(sampled.crossover_prob >= 0.0 && sampled.crossover_prob <= 1.0);
    }

    #[test]
    fn test_operator_params_sample() {
        let posteriors = HyperparameterPosteriors::new();
        let mut rng = rand::thread_rng();

        let params = OperatorParams::sample_from(&posteriors, &mut rng);
        assert!(params.mutation_rate >= 0.0);
        assert!(params.temperature > 0.0);
        assert!(params.sbx_eta >= 1.0);
    }

    #[test]
    fn test_credible_interval() {
        let posterior = BetaPosterior::new(50.0, 50.0);
        let (lower, upper) = posterior.credible_interval(0.95);

        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(lower > 0.0);
        assert!(upper < 1.0);
    }
}
