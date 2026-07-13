//! Bayesian building blocks and an online operator-parameter tuner.
//!
//! This module provides:
//!
//! - Honest conjugate posteriors that say what they are: [`BetaPosterior`]
//!   (Beta-Bernoulli over a probability), [`GammaPosterior`] (Gamma-Exponential
//!   over a positive *rate*), and [`RunningLogMoments`] (a moment tracker for
//!   log-scale quantities — *not* a Bayesian posterior, and named accordingly).
//! - A [`ThompsonSamplingTuner`]: a genuine multi-armed-bandit tuner for GA
//!   operator parameters. Each tunable parameter is discretized into candidate
//!   values ("arms"); each arm carries a Beta posterior over
//!   `P(offspring improves on parents | this value is used)`. Every generation an
//!   arm is Thompson-sampled per parameter and its **value** is applied to the
//!   operators; the observed improvement events are credited back to the arm that
//!   produced them.
//!
//! The Beta draw taken during Thompson sampling is used **only** to pick an arm;
//! it is never returned as the parameter value itself. This is the correct
//! separation that the previous `BayesianHyperparameterLearner` got wrong (it
//! sampled a `P(improvement)` posterior and used the draw *as* the mutation rate).

use rand::Rng;
use rand_distr::{Beta, Distribution, Gamma};

use crate::operators::mutation::{
    BitFlipMutation, GaussianMutation, PolynomialMutation, UniformMutation,
};

/// Beta distribution posterior for a probability parameter (Beta-Bernoulli).
///
/// Models `θ ∈ [0, 1]` with a `Beta(α, β)` prior; observing a success increments
/// `α`, a failure increments `β`. The original prior `(α₀, β₀)` is retained so
/// that [`observations`](Self::observations) can report the true trial count for
/// *any* prior, not just the uniform one.
#[derive(Clone, Debug)]
pub struct BetaPosterior {
    /// Alpha parameter (prior pseudo-successes + observed successes)
    pub alpha: f64,
    /// Beta parameter (prior pseudo-failures + observed failures)
    pub beta: f64,
    /// Prior alpha (α₀) — retained for correct observation counting.
    pub alpha0: f64,
    /// Prior beta (β₀) — retained for correct observation counting.
    pub beta0: f64,
}

impl BetaPosterior {
    /// Create with uniform prior (α = β = 1)
    pub fn uniform() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Create with Jeffreys prior (α = β = 0.5)
    pub fn jeffreys() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Create with a custom prior `Beta(alpha, beta)`.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta,
            alpha0: alpha,
            beta0: beta,
        }
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

    /// Number of *observed* Bernoulli trials, `(α − α₀) + (β − β₀)`.
    ///
    /// This subtracts the stored prior pseudo-counts, so it is correct for any
    /// prior — uniform `Beta(1, 1)`, Jeffreys `Beta(0.5, 0.5)`, or informative.
    pub fn observations(&self) -> f64 {
        (self.alpha - self.alpha0) + (self.beta - self.beta0)
    }

    /// Apply decay to move toward the stored prior (for non-stationary environments)
    pub fn decay(&mut self, factor: f64) {
        self.alpha = self.alpha0 + factor * (self.alpha - self.alpha0);
        self.beta = self.beta0 + factor * (self.beta - self.beta0);
    }
}

impl Default for BetaPosterior {
    fn default() -> Self {
        Self::uniform()
    }
}

/// Gamma posterior for the **rate** `λ` of an `Exponential(λ)` likelihood.
///
/// With a `Gamma(α, β)` prior on `λ` (shape `α`, rate `β`), observing exponential
/// data `xᵢ` gives the conjugate update `α → α + n`, `β → β + Σxᵢ`. Accordingly:
///
/// - [`mean`](Self::mean) `= α / β` is the posterior mean of the **rate** `λ`.
/// - [`posterior_mean_of_mean`](Self::posterior_mean_of_mean) `= β / (α − 1)` is
///   the posterior mean of the **mean** `1/λ` (since `1/λ ~ Inverse-Gamma(α, β)`).
///
/// Note the two differ by a reciprocal: for exponential data with sample mean `m`
/// the rate posterior concentrates near `1/m`, while the mean posterior
/// concentrates near `m`. Callers that want "the average of the observed values"
/// must use [`posterior_mean_of_mean`](Self::posterior_mean_of_mean), not
/// [`mean`](Self::mean). Feeding an arbitrary positive *parameter value* in as if
/// it were exponential data (as the old learner did) is not a coherent model —
/// tune positive parameters with [`ThompsonSamplingTuner`] instead.
#[derive(Clone, Debug)]
pub struct GammaPosterior {
    /// Shape parameter (α)
    pub shape: f64,
    /// Rate parameter (β)
    pub rate: f64,
    /// Prior shape (α₀)
    pub shape0: f64,
    /// Prior rate (β₀)
    pub rate0: f64,
}

impl GammaPosterior {
    /// Create with a vague prior `Gamma(1, 0.01)`.
    pub fn vague() -> Self {
        Self::new(1.0, 0.01)
    }

    /// Create with a custom prior `Gamma(shape, rate)`.
    pub fn new(shape: f64, rate: f64) -> Self {
        Self {
            shape,
            rate,
            shape0: shape,
            rate0: rate,
        }
    }

    /// Conjugate update for one `Exponential(λ)` datum `value`.
    pub fn observe(&mut self, value: f64) {
        self.shape += 1.0;
        self.rate += value;
    }

    /// Posterior mean of the **rate** `λ` (`= α / β`).
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Posterior mean of the **mean** `1/λ` (`= β / (α − 1)`, defined for `α > 1`).
    ///
    /// `1/λ ~ Inverse-Gamma(α, β)` has mean `β / (α − 1)`. This is the quantity to
    /// use when you want the posterior estimate of the average of the observed
    /// positive values.
    pub fn posterior_mean_of_mean(&self) -> Option<f64> {
        if self.shape > 1.0 {
            Some(self.rate / (self.shape - 1.0))
        } else {
            None
        }
    }

    /// Posterior mode of the rate (for shape ≥ 1)
    pub fn mode(&self) -> Option<f64> {
        if self.shape >= 1.0 {
            Some((self.shape - 1.0) / self.rate)
        } else {
            None
        }
    }

    /// Posterior variance of the rate
    pub fn variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Number of observed exponential data points, `α − α₀`.
    pub fn observations(&self) -> f64 {
        self.shape - self.shape0
    }

    /// Sample the **rate** `λ` from the posterior.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Gamma::new(self.shape, 1.0 / self.rate)
            .expect("Invalid Gamma parameters")
            .sample(rng)
    }

    /// Apply decay toward the stored prior.
    pub fn decay(&mut self, factor: f64) {
        self.shape = self.shape0 + factor * (self.shape - self.shape0);
        self.rate = self.rate0 + factor * (self.rate - self.rate0);
    }
}

impl Default for GammaPosterior {
    fn default() -> Self {
        Self::vague()
    }
}

/// Running mean/variance of `ln(x)` for positive quantities such as step sizes.
///
/// This is a numerically-stable Welford moment tracker on the log scale — **not**
/// a Bayesian posterior (the conjugate model for a log-normal with unknown mean
/// and variance would be Normal-Inverse-Gamma). It is named to say exactly that.
///
/// Unlike the previous `LogNormalPosterior`, the variance is a clean population
/// variance with **no prior contamination**: after a single observation the
/// variance is `0`, and there is no spurious `+1.0` sum-of-squares term injected
/// at `n = 2`.
#[derive(Clone, Debug, Default)]
pub struct RunningLogMoments {
    /// Running mean of `ln(x)`.
    mean_log: f64,
    /// Running sum of squared deviations of `ln(x)` (Welford's M2).
    m2: f64,
    /// Number of observations.
    n: usize,
}

impl RunningLogMoments {
    /// Create an empty tracker (no observations, no prior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a positive value (non-positive values are ignored).
    pub fn observe(&mut self, x: f64) {
        if x <= 0.0 {
            return;
        }
        let log_x = x.ln();
        self.n += 1;
        let delta = log_x - self.mean_log;
        self.mean_log += delta / self.n as f64;
        let delta2 = log_x - self.mean_log;
        self.m2 += delta * delta2;
    }

    /// Number of observations.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Mean of `ln(x)`.
    pub fn mean_log(&self) -> f64 {
        self.mean_log
    }

    /// Population variance of `ln(x)` (`M2 / n`, and `0` when `n < 1`).
    pub fn var_log(&self) -> f64 {
        if self.n >= 1 {
            self.m2 / self.n as f64
        } else {
            0.0
        }
    }

    /// Unbiased sample variance of `ln(x)` (`M2 / (n − 1)`, `None` when `n < 2`).
    pub fn sample_var_log(&self) -> Option<f64> {
        if self.n >= 2 {
            Some(self.m2 / (self.n as f64 - 1.0))
        } else {
            None
        }
    }

    /// Mean in the original space, `exp(μ + σ²/2)`.
    pub fn mean(&self) -> f64 {
        (self.mean_log + self.var_log() / 2.0).exp()
    }

    /// Mode in the original space, `exp(μ − σ²)`.
    pub fn mode(&self) -> f64 {
        (self.mean_log - self.var_log()).exp()
    }

    /// Draw a log-normal sample using the current moment estimates.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        use rand_distr::StandardNormal;
        let z: f64 = rng.sample(StandardNormal);
        (self.mean_log + self.var_log().sqrt() * z).exp()
    }
}

/// Mutation operators whose per-gene mutation probability can be set by an online
/// tuner such as [`ThompsonSamplingTuner`].
pub trait TunableMutation {
    /// Set the per-gene mutation probability applied by subsequent `mutate` calls.
    ///
    /// Implementations clamp the value into `[0, 1]`.
    fn set_mutation_probability(&mut self, probability: f64);
}

macro_rules! impl_tunable_mutation {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TunableMutation for $ty {
                fn set_mutation_probability(&mut self, probability: f64) {
                    self.mutation_probability = Some(probability.clamp(0.0, 1.0));
                }
            }
        )+
    };
}

impl_tunable_mutation!(
    PolynomialMutation,
    GaussianMutation,
    UniformMutation,
    BitFlipMutation,
);

/// One discretized candidate value for a tunable parameter, with a Beta posterior
/// over `P(offspring improves on parents | this value is used)`.
#[derive(Clone, Debug)]
pub struct BanditArm {
    /// The concrete parameter value this arm applies.
    pub value: f64,
    /// Posterior over `P(improvement)` when this value is used.
    pub posterior: BetaPosterior,
    /// Number of times this arm has been selected (pulled).
    pub selections: u64,
}

/// A single tunable parameter, discretized into arms and tuned by Thompson
/// sampling over each arm's Beta posterior.
#[derive(Clone, Debug)]
pub struct BanditParameter {
    /// Human-readable parameter name (e.g. `"mutation_rate"`).
    pub name: String,
    arms: Vec<BanditArm>,
    last_selected: Option<usize>,
}

impl BanditParameter {
    /// Create a parameter with the given candidate values and a uniform prior.
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self::with_prior(name, values, BetaPosterior::uniform())
    }

    /// Create a parameter with the given candidate values and a shared Beta prior.
    pub fn with_prior(name: impl Into<String>, values: Vec<f64>, prior: BetaPosterior) -> Self {
        assert!(
            !values.is_empty(),
            "BanditParameter requires at least one arm value"
        );
        let arms = values
            .into_iter()
            .map(|value| BanditArm {
                value,
                posterior: prior.clone(),
                selections: 0,
            })
            .collect();
        Self {
            name: name.into(),
            arms,
            last_selected: None,
        }
    }

    /// Thompson-sample an arm and return its parameter value.
    ///
    /// Draws one probability from each arm's Beta posterior and selects the arm
    /// with the highest draw. The draw is used **only** to choose the arm; the
    /// returned value is the arm's concrete parameter value.
    pub fn select<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let mut best_idx = 0;
        let mut best_draw = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let draw = arm.posterior.sample(rng);
            if draw > best_draw {
                best_draw = draw;
                best_idx = i;
            }
        }
        self.last_selected = Some(best_idx);
        self.arms[best_idx].selections += 1;
        self.arms[best_idx].value
    }

    /// Credit the most recently selected arm with an improvement outcome.
    pub fn observe(&mut self, improved: bool) {
        if let Some(idx) = self.last_selected {
            self.arms[idx].posterior.observe(improved);
        }
    }

    /// The arms of this parameter.
    pub fn arms(&self) -> &[BanditArm] {
        &self.arms
    }

    /// Candidate values, in arm order.
    pub fn values(&self) -> Vec<f64> {
        self.arms.iter().map(|a| a.value).collect()
    }

    /// Posterior mean of `P(improvement)` for each arm, in arm order.
    pub fn posterior_means(&self) -> Vec<f64> {
        self.arms.iter().map(|a| a.posterior.mean()).collect()
    }

    /// Selection (pull) count for each arm, in arm order.
    pub fn selection_counts(&self) -> Vec<u64> {
        self.arms.iter().map(|a| a.selections).collect()
    }

    /// Value chosen at the most recent [`select`](Self::select), if any.
    pub fn selected_value(&self) -> Option<f64> {
        self.last_selected.map(|i| self.arms[i].value)
    }

    /// Arm index chosen at the most recent [`select`](Self::select), if any.
    pub fn selected_index(&self) -> Option<usize> {
        self.last_selected
    }

    /// Index of the arm with the highest posterior mean of `P(improvement)`.
    pub fn best_index(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.posterior
                    .mean()
                    .partial_cmp(&b.posterior.mean())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Value of the arm with the highest posterior mean of `P(improvement)`.
    pub fn best_value(&self) -> f64 {
        self.arms[self.best_index()].value
    }

    /// Total number of improvement events credited across all arms.
    pub fn total_observations(&self) -> f64 {
        self.arms.iter().map(|a| a.posterior.observations()).sum()
    }
}

/// Canonical parameter name for the per-gene mutation probability arm set.
pub const PARAM_MUTATION_RATE: &str = "mutation_rate";
/// Canonical parameter name for the whole-genome crossover probability arm set.
pub const PARAM_CROSSOVER_PROB: &str = "crossover_prob";

/// Configuration for a [`ThompsonSamplingTuner`] wired into a GA.
#[derive(Clone, Debug)]
pub struct ThompsonConfig {
    /// Candidate per-gene mutation probabilities (empty ⇒ mutation not tuned).
    pub mutation_rate_arms: Vec<f64>,
    /// Candidate whole-genome crossover probabilities (empty ⇒ crossover not tuned).
    pub crossover_prob_arms: Vec<f64>,
    /// Beta prior shared by every arm.
    pub prior: BetaPosterior,
    /// Record a per-generation posterior snapshot for later inspection.
    pub record_history: bool,
}

impl Default for ThompsonConfig {
    fn default() -> Self {
        Self {
            mutation_rate_arms: vec![0.01, 0.05, 0.1, 0.2, 0.4],
            crossover_prob_arms: vec![0.5, 0.7, 0.9],
            prior: BetaPosterior::uniform(),
            record_history: false,
        }
    }
}

impl ThompsonConfig {
    /// Build a tuner from this configuration.
    pub fn build_tuner(&self) -> ThompsonSamplingTuner {
        ThompsonSamplingTuner::from_config(self)
    }
}

/// A snapshot of a tuner's per-arm posterior means at one generation.
#[derive(Clone, Debug)]
pub struct TunerSnapshot {
    /// Generation index at which the snapshot was taken.
    pub generation: usize,
    /// One entry per parameter: `(name, value selected this generation, posterior means per arm)`.
    pub parameters: Vec<(String, Option<f64>, Vec<f64>)>,
}

/// Thompson-sampling multi-armed-bandit tuner over GA operator parameters.
///
/// Each parameter ([`BanditParameter`]) is discretized into arms; every generation
/// [`select_all`](Self::select_all) Thompson-samples one arm per parameter, whose
/// **value** the caller applies to its operators, and [`observe`](Self::observe)
/// credits every parameter's selected arm with each improvement event. Distinct
/// parameters keep independent arms and posteriors, so (unlike the previous
/// learner) mutation-rate and crossover-probability posteriors are free to diverge.
#[derive(Clone, Debug)]
pub struct ThompsonSamplingTuner {
    parameters: Vec<BanditParameter>,
    record_history: bool,
    history: Vec<TunerSnapshot>,
    observations: u64,
}

impl ThompsonSamplingTuner {
    /// Create a tuner from an explicit set of parameters.
    pub fn new(parameters: Vec<BanditParameter>) -> Self {
        Self {
            parameters,
            record_history: false,
            history: Vec::new(),
            observations: 0,
        }
    }

    /// Build a tuner from a [`ThompsonConfig`].
    pub fn from_config(cfg: &ThompsonConfig) -> Self {
        let mut parameters = Vec::new();
        if !cfg.mutation_rate_arms.is_empty() {
            parameters.push(BanditParameter::with_prior(
                PARAM_MUTATION_RATE,
                cfg.mutation_rate_arms.clone(),
                cfg.prior.clone(),
            ));
        }
        if !cfg.crossover_prob_arms.is_empty() {
            parameters.push(BanditParameter::with_prior(
                PARAM_CROSSOVER_PROB,
                cfg.crossover_prob_arms.clone(),
                cfg.prior.clone(),
            ));
        }
        Self {
            parameters,
            record_history: cfg.record_history,
            history: Vec::new(),
            observations: 0,
        }
    }

    /// Enable or disable per-generation snapshot recording.
    pub fn with_history(mut self, on: bool) -> Self {
        self.record_history = on;
        self
    }

    /// All tuned parameters.
    pub fn parameters(&self) -> &[BanditParameter] {
        &self.parameters
    }

    /// Look up a parameter by name.
    pub fn parameter(&self, name: &str) -> Option<&BanditParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Look up a parameter by name (mutable).
    pub fn parameter_mut(&mut self, name: &str) -> Option<&mut BanditParameter> {
        self.parameters.iter_mut().find(|p| p.name == name)
    }

    /// Whether the tuner has no parameters to tune.
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Thompson-sample one arm per parameter for the coming generation.
    pub fn select_all<R: Rng>(&mut self, rng: &mut R) {
        for p in &mut self.parameters {
            p.select(rng);
        }
    }

    /// Value selected for `name` at the most recent [`select_all`](Self::select_all).
    pub fn selected(&self, name: &str) -> Option<f64> {
        self.parameter(name).and_then(|p| p.selected_value())
    }

    /// Credit every parameter's currently selected arm with one improvement event.
    pub fn observe(&mut self, improved: bool) {
        for p in &mut self.parameters {
            p.observe(improved);
        }
        self.observations += 1;
    }

    /// Total number of improvement events fed back to the tuner.
    pub fn total_observations(&self) -> u64 {
        self.observations
    }

    /// Record a snapshot of the current posteriors (no-op unless history is enabled).
    pub fn snapshot(&mut self, generation: usize) {
        if !self.record_history {
            return;
        }
        let parameters = self
            .parameters
            .iter()
            .map(|p| (p.name.clone(), p.selected_value(), p.posterior_means()))
            .collect();
        self.history.push(TunerSnapshot {
            generation,
            parameters,
        });
    }

    /// Recorded per-generation snapshots (empty unless history is enabled).
    pub fn history(&self) -> &[TunerSnapshot] {
        &self.history
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

    if p < 0.5 {
        -q
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
            assert!((0.0..=1.0).contains(&sample));
        }
    }

    /// regression: EV-95 — observations() must subtract the *stored* prior, so it
    /// reports the true trial count for Jeffreys/informative priors, not just uniform.
    #[test]
    fn test_beta_observations_uses_stored_prior() {
        // Uniform prior: 10 trials -> 10 observations.
        let mut uniform = BetaPosterior::uniform();
        for i in 0..10 {
            uniform.observe(i % 2 == 0);
        }
        assert!((uniform.observations() - 10.0).abs() < 1e-12);

        // Jeffreys prior Beta(0.5, 0.5): 5 successes + 5 failures -> 10 observations.
        // The pre-fix formula (alpha + beta - 2) would report 11 - 2 = 9 here.
        let mut jeffreys = BetaPosterior::jeffreys();
        for _ in 0..5 {
            jeffreys.observe_success();
        }
        for _ in 0..5 {
            jeffreys.observe_failure();
        }
        assert!((jeffreys.observations() - 10.0).abs() < 1e-12);

        // Informative prior Beta(2, 2): 4 trials -> 4 observations.
        let mut informative = BetaPosterior::new(2.0, 2.0);
        for _ in 0..4 {
            informative.observe_success();
        }
        assert!((informative.observations() - 4.0).abs() < 1e-12);
    }

    /// regression: EV-22 — mean() is the posterior mean of the RATE, and
    /// posterior_mean_of_mean() = β/(α−1) recovers the mean of the observed values.
    #[test]
    fn test_gamma_rate_and_mean_of_mean() {
        // Hand-computed conjugate update: prior Gamma(2, 1), observe [1, 2, 3].
        // Posterior = Gamma(2 + 3, 1 + 6) = Gamma(5, 7).
        let mut posterior = GammaPosterior::new(2.0, 1.0);
        for x in [1.0, 2.0, 3.0] {
            posterior.observe(x);
        }
        assert!((posterior.shape - 5.0).abs() < 1e-12);
        assert!((posterior.rate - 7.0).abs() < 1e-12);
        // Posterior mean of the RATE = 5/7.
        assert!((posterior.mean() - 5.0 / 7.0).abs() < 1e-12);
        // Posterior mean of the MEAN = 7/(5-1) = 1.75.
        assert!((posterior.posterior_mean_of_mean().unwrap() - 1.75).abs() < 1e-12);
        assert!((posterior.observations() - 3.0).abs() < 1e-12);
    }

    /// regression: EV-22 — observing value 20 repeatedly drives the RATE posterior
    /// toward 1/20 (=0.05), while posterior_mean_of_mean() recovers ~20 — the value
    /// the old code could never produce (it returned the reciprocal as the parameter).
    #[test]
    fn test_gamma_recovers_mean_not_reciprocal() {
        let mut posterior = GammaPosterior::vague();
        for _ in 0..1000 {
            posterior.observe(20.0);
        }
        assert!((posterior.mean() - 0.05).abs() < 0.005, "rate ~ 1/20");
        let mean = posterior.posterior_mean_of_mean().unwrap();
        assert!((mean - 20.0).abs() < 0.5, "mean-of-mean ~ 20, got {mean}");
    }

    /// regression: EV-61 — the log-moment tracker must not contaminate the variance
    /// with a prior. After one observation the variance is 0; after two it is the
    /// exact population variance (the old code injected a spurious +1.0 term).
    #[test]
    fn test_running_log_moments_no_prior_contamination() {
        let mut moments = RunningLogMoments::new();

        // Single observation -> variance is exactly 0 (old code left it at 1.0).
        moments.observe(std::f64::consts::E); // ln = 1
        assert!((moments.var_log()).abs() < 1e-12);
        assert!((moments.mean_log() - 1.0).abs() < 1e-12);

        // Second observation ln = 3. Population var of {1, 3} = 1.0, sample var = 2.0.
        // The old (contaminated) recursion produced 1.5.
        moments.observe(std::f64::consts::E.powi(3)); // ln = 3
        assert!((moments.mean_log() - 2.0).abs() < 1e-12);
        assert!((moments.var_log() - 1.0).abs() < 1e-12);
        assert!((moments.sample_var_log().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_running_log_moments_mean_original_space() {
        let mut moments = RunningLogMoments::new();
        for _ in 0..10 {
            moments.observe(0.1);
        }
        // All observations equal -> variance 0 -> mean = 0.1 exactly.
        assert!((moments.mean() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_bandit_parameter_thompson_selects_a_value() {
        let mut param = BanditParameter::new(PARAM_MUTATION_RATE, vec![0.01, 0.1, 0.3]);
        let mut rng = StdRng::seed_from_u64(1);
        let v = param.select(&mut rng);
        assert!([0.01, 0.1, 0.3].contains(&v));
        param.observe(true);
        assert!((param.total_observations() - 1.0).abs() < 1e-12);
    }

    /// regression: EV-23 — an honest Thompson-sampling bandit must concentrate its
    /// pulls on the dominant arm. Here mutation-rate 0.3 truly improves offspring
    /// 55% of the time and 0.01 only 20%; after learning, >70% of late pulls land
    /// on 0.3. (The old design sampled a single P(improvement) posterior and used
    /// the draw *as* the rate, so it had no per-arm concentration at all.)
    #[test]
    fn test_bandit_concentrates_on_better_arm() {
        let mut rng = StdRng::seed_from_u64(20260710);
        let mut param = BanditParameter::new(PARAM_MUTATION_RATE, vec![0.01, 0.3]);

        // True improvement probabilities per arm value.
        let true_p = |v: f64| if v >= 0.3 { 0.55 } else { 0.20 };

        let total_rounds = 2000;
        let late_start = 1500;
        let mut late_good = 0u32;
        let mut late_total = 0u32;

        for round in 0..total_rounds {
            let value = param.select(&mut rng);
            let improved = rng.gen::<f64>() < true_p(value);
            param.observe(improved);
            if round >= late_start {
                late_total += 1;
                if value >= 0.3 {
                    late_good += 1;
                }
            }
        }

        let frac = late_good as f64 / late_total as f64;
        assert!(
            frac > 0.70,
            "expected >70% of late pulls on the better arm, got {:.2}",
            frac
        );
        // The learner should also report 0.3 as the best arm.
        assert!((param.best_value() - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_thompson_tuner_from_config() {
        let cfg = ThompsonConfig::default();
        let mut tuner = cfg.build_tuner();
        assert!(tuner.parameter(PARAM_MUTATION_RATE).is_some());
        assert!(tuner.parameter(PARAM_CROSSOVER_PROB).is_some());

        let mut rng = StdRng::seed_from_u64(7);
        tuner.select_all(&mut rng);
        assert!(tuner.selected(PARAM_MUTATION_RATE).is_some());
        assert!(tuner.selected(PARAM_CROSSOVER_PROB).is_some());

        tuner.observe(true);
        tuner.observe(false);
        assert_eq!(tuner.total_observations(), 2);
    }

    /// The two parameters keep independent posteriors — they cannot be forced
    /// identical the way the old learner's duplicated posteriors were.
    #[test]
    fn test_thompson_tuner_parameters_are_independent() {
        let cfg = ThompsonConfig {
            mutation_rate_arms: vec![0.1, 0.3],
            crossover_prob_arms: vec![0.5, 0.9],
            prior: BetaPosterior::uniform(),
            record_history: false,
        };
        let mut tuner = cfg.build_tuner();
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..50 {
            tuner.select_all(&mut rng);
            tuner.observe(true);
        }
        let mr = tuner.parameter(PARAM_MUTATION_RATE).unwrap();
        let cx = tuner.parameter(PARAM_CROSSOVER_PROB).unwrap();
        // Distinct arm value sets prove they are genuinely separate bandits.
        assert_eq!(mr.values(), vec![0.1, 0.3]);
        assert_eq!(cx.values(), vec![0.5, 0.9]);
    }

    #[test]
    fn test_tunable_mutation_sets_probability() {
        let mut m = GaussianMutation::new(0.1);
        m.set_mutation_probability(0.25);
        assert_eq!(m.mutation_probability, Some(0.25));
        // Clamped into [0, 1].
        m.set_mutation_probability(5.0);
        assert_eq!(m.mutation_probability, Some(1.0));
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
