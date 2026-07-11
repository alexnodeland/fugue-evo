//! Univariate Marginal Distribution Algorithm (UMDA)
//!
//! UMDA is a simple but effective EDA that assumes independence between variables.
//! It estimates univariate marginal distributions for each variable and samples
//! new solutions from the product of these distributions.
//!
//! For continuous problems: estimates mean and variance per dimension
//! For binary problems: estimates probability of 1 per bit

use std::time::Instant;

use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Normal};

use crate::diagnostics::{EvolutionResult, EvolutionStats, GenerationStats, TimingStats};
use crate::error::EvolutionError;
use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::{BinaryGenome, EvolutionaryGenome, RealValuedGenome};
use crate::population::individual::Individual;
use crate::population::population::Population;
use crate::termination::{EvolutionState, MaxGenerations, TerminationCriterion};

/// Select references to the genomes of the `count` best individuals, ordered by
/// [`Individual::is_better_than`] (EV-80).
///
/// Using `is_better_than` rather than the `to_f64`-descending order of
/// `Population::sort_by_fitness` keeps truncation selection correct for every
/// `FitnessValue` — including non-scalar fitnesses whose f64 projection does not
/// agree with their intrinsic ordering.
fn select_top_genomes<G, F>(population: &Population<G, F>, count: usize) -> Vec<&G>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    let mut ranked: Vec<&Individual<G, F>> = population.iter().collect();
    ranked.sort_by(|a, b| {
        if a.is_better_than(b) {
            std::cmp::Ordering::Less
        } else if b.is_better_than(a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    ranked
        .into_iter()
        .take(count)
        .map(|ind| ind.genome())
        .collect()
}

/// Configuration for UMDA.
///
/// The builder ([`UMDABuilder`]) validates these fields in `build()` and returns
/// an error for out-of-range values (EV-79) — it no longer silently clamps them.
/// All fields are public, so constructing `UMDAConfig` directly is a deliberate
/// escape hatch that **bypasses that validation**; if you build the struct by
/// hand, keep the documented ranges yourself.
#[derive(Clone, Debug)]
pub struct UMDAConfig {
    /// Population size (must be >= 1).
    pub population_size: usize,
    /// Selection ratio: the top proportion selected for model learning. Must lie
    /// in the open-closed interval `(0.0, 1.0]`.
    pub selection_ratio: f64,
    /// Minimum variance to prevent collapse (continuous). Must be >= 0.0.
    pub min_variance: f64,
    /// Probability bounds `(min, max)` for binary UMDA (to prevent determinism).
    /// Must satisfy `0.0 < min <= max < 1.0`.
    pub prob_bounds: (f64, f64),
    /// Learning rate for the model update (`1.0` = replace, `<1.0` = blend with
    /// the previous model). Must lie in `(0.0, 1.0]`; `0.0` is rejected because it
    /// makes the update a no-op (the model never learns).
    pub learning_rate: f64,
}

impl Default for UMDAConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            selection_ratio: 0.5,
            min_variance: 0.01,
            prob_bounds: (0.01, 0.99),
            learning_rate: 1.0,
        }
    }
}

impl UMDAConfig {
    /// Validate the configuration ranges (EV-79). Called by `build()`; returns a
    /// [`EvolutionError::Configuration`] describing the first out-of-range field.
    pub fn validate(&self) -> Result<(), EvolutionError> {
        if self.population_size == 0 {
            return Err(EvolutionError::Configuration(
                "population_size must be >= 1".to_string(),
            ));
        }
        if !(self.selection_ratio > 0.0 && self.selection_ratio <= 1.0) {
            return Err(EvolutionError::Configuration(format!(
                "selection_ratio must be in (0.0, 1.0], got {}",
                self.selection_ratio
            )));
        }
        if !(self.learning_rate > 0.0 && self.learning_rate <= 1.0) {
            return Err(EvolutionError::Configuration(format!(
                "learning_rate must be in (0.0, 1.0], got {}",
                self.learning_rate
            )));
        }
        if self.min_variance < 0.0 {
            return Err(EvolutionError::Configuration(format!(
                "min_variance must be >= 0.0, got {}",
                self.min_variance
            )));
        }
        let (pmin, pmax) = self.prob_bounds;
        if !(pmin > 0.0 && pmin <= pmax && pmax < 1.0) {
            return Err(EvolutionError::Configuration(format!(
                "prob_bounds must satisfy 0.0 < min <= max < 1.0, got ({pmin}, {pmax})"
            )));
        }
        Ok(())
    }
}

/// Builder for UMDA
pub struct UMDABuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: UMDAConfig,
    bounds: Option<MultiBounds>,
    fitness: Option<Fit>,
    termination: Option<Term>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F> UMDABuilder<G, F, (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: UMDAConfig::default(),
            bounds: None,
            fitness: None,
            termination: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<G, F> Default for UMDABuilder<G, F, (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, Fit, Term> UMDABuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Set the population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    /// Set the selection ratio (must be in `(0.0, 1.0]`; validated in `build()`).
    pub fn selection_ratio(mut self, ratio: f64) -> Self {
        self.config.selection_ratio = ratio;
        self
    }

    /// Set the minimum variance (must be >= 0.0; validated in `build()`).
    pub fn min_variance(mut self, variance: f64) -> Self {
        self.config.min_variance = variance;
        self
    }

    /// Set probability bounds for binary UMDA (must satisfy
    /// `0.0 < min <= max < 1.0`; validated in `build()`).
    pub fn prob_bounds(mut self, min: f64, max: f64) -> Self {
        self.config.prob_bounds = (min, max);
        self
    }

    /// Set the learning rate for the model update (must be in `(0.0, 1.0]`;
    /// validated in `build()`).
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.config.learning_rate = rate;
        self
    }

    /// Set the search space bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the fitness function
    pub fn fitness<NewFit>(self, fitness: NewFit) -> UMDABuilder<G, F, NewFit, Term>
    where
        NewFit: Fitness<Genome = G, Value = F>,
    {
        UMDABuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: Some(fitness),
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the termination criterion
    pub fn termination<NewTerm>(self, termination: NewTerm) -> UMDABuilder<G, F, Fit, NewTerm>
    where
        NewTerm: TerminationCriterion<G, F>,
    {
        UMDABuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: self.fitness,
            termination: Some(termination),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set max generations (convenience method)
    pub fn max_generations(self, max: usize) -> UMDABuilder<G, F, Fit, MaxGenerations> {
        UMDABuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: self.fitness,
            termination: Some(MaxGenerations::new(max)),
            _phantom: std::marker::PhantomData,
        }
    }
}

// ============================================================================
// Continuous UMDA (RealVector)
// ============================================================================

/// Univariate model for continuous variables
#[derive(Clone, Debug)]
pub struct ContinuousUnivariateModel {
    /// Mean for each dimension
    pub means: Vec<f64>,
    /// Variance for each dimension
    pub variances: Vec<f64>,
}

impl ContinuousUnivariateModel {
    /// Maximum number of rejection retries per coordinate before clamping (EV-39).
    pub const MAX_REJECTION_RETRIES: usize = 100;

    /// Create from bounds (initial uniform-ish distribution)
    pub fn from_bounds(bounds: &MultiBounds) -> Self {
        let means: Vec<f64> = bounds.bounds.iter().map(|b| b.center()).collect();
        let variances: Vec<f64> = bounds
            .bounds
            .iter()
            .map(|b| (b.range() / 4.0).powi(2))
            .collect();
        Self { means, variances }
    }

    /// Update model from selected individuals
    pub fn update(&mut self, selected: &[&RealVector], config: &UMDAConfig) {
        let n = selected.len() as f64;
        if n == 0.0 {
            return;
        }

        for i in 0..self.means.len() {
            // Compute sample mean
            let mean: f64 = selected.iter().map(|g| g.genes()[i]).sum::<f64>() / n;

            // Compute the unbiased (Bessel-corrected, n-1) sample variance (EV-81).
            // The n-1 denominator avoids systematically under-estimating spread,
            // which would otherwise nudge the model toward premature contraction.
            let sum_sq: f64 = selected
                .iter()
                .map(|g| (g.genes()[i] - mean).powi(2))
                .sum::<f64>();
            let variance: f64 = if n > 1.0 { sum_sq / (n - 1.0) } else { 0.0 };

            // Apply learning rate and bounds
            self.means[i] =
                config.learning_rate * mean + (1.0 - config.learning_rate) * self.means[i];
            self.variances[i] = config.learning_rate * variance.max(config.min_variance)
                + (1.0 - config.learning_rate) * self.variances[i];
        }
    }

    /// Sample a new individual from the model.
    ///
    /// EV-39: each coordinate is drawn by rejection from the truncated Gaussian —
    /// out-of-bounds draws are retried up to [`Self::MAX_REJECTION_RETRIES`] times
    /// before falling back to a clamp. For interior-optimum problems this makes
    /// the boundary "atoms" (probability mass piled on a bound by naive clamping)
    /// vanish, so the accepted samples that feed the next variance estimate are
    /// genuine spread rather than collapsed boundary points.
    pub fn sample<R: Rng>(&self, bounds: &MultiBounds, rng: &mut R) -> RealVector {
        let genes: Vec<f64> = self
            .means
            .iter()
            .zip(self.variances.iter())
            .zip(bounds.bounds.iter())
            .map(|((mean, var), bound)| {
                let normal =
                    Normal::new(*mean, var.sqrt()).unwrap_or(Normal::new(*mean, 0.1).unwrap());

                let mut value = normal.sample(rng);
                let mut retries = 0;
                while (value < bound.min || value > bound.max)
                    && retries < Self::MAX_REJECTION_RETRIES
                {
                    value = normal.sample(rng);
                    retries += 1;
                }

                // Fallback only when rejection failed to land in-bounds (e.g. the
                // whole feasible interval is deep in a Gaussian tail): clamp so the
                // returned genome always respects the box constraints.
                value.clamp(bound.min, bound.max)
            })
            .collect();

        RealVector::new(genes)
    }
}

impl<F, Fit, Term> UMDABuilder<RealVector, F, Fit, Term>
where
    F: FitnessValue + Send,
    Fit: Fitness<Genome = RealVector, Value = F> + Sync,
    Term: TerminationCriterion<RealVector, F>,
{
    /// Build the continuous UMDA instance
    pub fn build(self) -> Result<ContinuousUMDA<F, Fit, Term>, EvolutionError> {
        // EV-79: validate configured ranges instead of silently clamping.
        self.config.validate()?;

        let bounds = self
            .bounds
            .ok_or_else(|| EvolutionError::Configuration("Bounds must be specified".to_string()))?;

        let fitness = self.fitness.ok_or_else(|| {
            EvolutionError::Configuration("Fitness function must be specified".to_string())
        })?;

        let termination = self.termination.ok_or_else(|| {
            EvolutionError::Configuration("Termination criterion must be specified".to_string())
        })?;

        Ok(ContinuousUMDA {
            config: self.config,
            bounds,
            fitness,
            termination,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// UMDA for continuous optimization
pub struct ContinuousUMDA<F, Fit, Term>
where
    F: FitnessValue,
{
    config: UMDAConfig,
    bounds: MultiBounds,
    fitness: Fit,
    termination: Term,
    _phantom: std::marker::PhantomData<F>,
}

impl<F, Fit, Term> ContinuousUMDA<F, Fit, Term>
where
    F: FitnessValue + Send,
    Fit: Fitness<Genome = RealVector, Value = F> + Sync,
    Term: TerminationCriterion<RealVector, F>,
{
    /// Create a builder for continuous UMDA
    pub fn builder() -> UMDABuilder<RealVector, F, (), ()> {
        UMDABuilder::new()
    }

    /// Run the UMDA algorithm.
    pub fn run<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<EvolutionResult<RealVector, F>, EvolutionError> {
        self.run_with_model(rng).map(|(result, _model)| result)
    }

    /// Run UMDA, invoking `on_generation(generation, best_fitness)` once per
    /// generation before that generation is sampled/evaluated.
    ///
    /// Returning `false` from the callback cancels the run early and returns the
    /// best-so-far result (AUDIT EV-34: lets the WASM layer report per-generation
    /// progress and support cancellation without a separate step API).
    pub fn run_with_callback<R: Rng, Cb: FnMut(usize, f64) -> bool>(
        &self,
        rng: &mut R,
        on_generation: Cb,
    ) -> Result<EvolutionResult<RealVector, F>, EvolutionError> {
        self.run_with_model_cb(rng, on_generation)
            .map(|(result, _model)| result)
    }

    /// Run the UMDA algorithm and also return the final learned univariate model
    /// (EV-10), so callers/tests can inspect whether the distribution actually
    /// converged toward the optimum rather than merely tracking a best-so-far.
    pub fn run_with_model<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(EvolutionResult<RealVector, F>, ContinuousUnivariateModel), EvolutionError> {
        // No-op observer: identical behavior to the historical `run_with_model`.
        self.run_with_model_cb(rng, |_generation, _best_fitness| true)
    }

    /// Shared UMDA loop body driving both `run_with_model` (no-op callback) and
    /// `run_with_callback` (EV-34 progress/cancel), so there is exactly one copy
    /// of the algorithm.
    fn run_with_model_cb<R: Rng, Cb: FnMut(usize, f64) -> bool>(
        &self,
        rng: &mut R,
        mut on_generation: Cb,
    ) -> Result<(EvolutionResult<RealVector, F>, ContinuousUnivariateModel), EvolutionError> {
        let start_time = Instant::now();

        // Initialize model
        let mut model = ContinuousUnivariateModel::from_bounds(&self.bounds);

        // Initialize population
        let mut population: Population<RealVector, F> =
            Population::random(self.config.population_size, &self.bounds, rng);
        population.evaluate(&self.fitness);

        let mut stats = EvolutionStats::new();
        let mut evaluations = self.config.population_size;
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut generation = 0usize;

        // Track best individual
        let mut best = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation,
                evaluations,
                best_fitness: best.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            // EV-34: per-generation progress/cancel hook. A `false` return cancels
            // the run, returning the best individual found so far.
            if !on_generation(generation, best.fitness_value().to_f64()) {
                break;
            }

            let gen_start = Instant::now();

            // Select the top individuals by is_better_than order (EV-80).
            let select_count =
                (self.config.population_size as f64 * self.config.selection_ratio).ceil() as usize;
            let selected: Vec<&RealVector> = select_top_genomes(&population, select_count);

            // Update model from selected individuals
            model.update(&selected, &self.config);

            // Sample new population from model
            population = Population::with_capacity(self.config.population_size);
            for _ in 0..self.config.population_size {
                let genome = model.sample(&self.bounds, rng);
                population.push(Individual::new(genome));
            }

            // Evaluate
            population.evaluate(&self.fitness);
            evaluations += self.config.population_size;

            // Update best
            if let Some(pop_best) = population.best() {
                if pop_best.is_better_than(&best) {
                    best = pop_best.clone();
                }
            }

            generation += 1;
            population.set_generation(generation);

            // Record statistics
            let timing = TimingStats::new().with_total(gen_start.elapsed());
            let gen_stats = GenerationStats::from_population(&population, generation, evaluations)
                .with_timing(timing);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        let result =
            EvolutionResult::new(best.genome, best.fitness.unwrap(), generation, evaluations)
                .with_stats(stats);
        Ok((result, model))
    }
}

// ============================================================================
// Binary UMDA (BitString)
// ============================================================================

/// Univariate model for binary variables
#[derive(Clone, Debug)]
pub struct BinaryUnivariateModel {
    /// Probability of 1 for each bit position
    pub probabilities: Vec<f64>,
}

impl BinaryUnivariateModel {
    /// Create with uniform 0.5 probability
    pub fn uniform(dimension: usize) -> Self {
        Self {
            probabilities: vec![0.5; dimension],
        }
    }

    /// Update model from selected individuals
    pub fn update(&mut self, selected: &[&BitString], config: &UMDAConfig) {
        let n = selected.len() as f64;
        if n == 0.0 {
            return;
        }

        for i in 0..self.probabilities.len() {
            // Count ones at position i
            let ones: f64 = selected
                .iter()
                .filter(|g| g.bits().get(i).copied().unwrap_or(false))
                .count() as f64;

            // Compute probability with learning rate and bounds
            let new_prob = ones / n;
            let bounded_prob = new_prob.clamp(config.prob_bounds.0, config.prob_bounds.1);
            self.probabilities[i] = config.learning_rate * bounded_prob
                + (1.0 - config.learning_rate) * self.probabilities[i];
        }
    }

    /// Sample a new individual from the model
    pub fn sample<R: Rng>(&self, rng: &mut R) -> BitString {
        let bits: Vec<bool> = self
            .probabilities
            .iter()
            .map(|p| {
                let dist = Bernoulli::new(*p).unwrap_or(Bernoulli::new(0.5).unwrap());
                dist.sample(rng)
            })
            .collect();

        BitString::new(bits)
    }
}

impl<F, Fit, Term> UMDABuilder<BitString, F, Fit, Term>
where
    F: FitnessValue + Send,
    Fit: Fitness<Genome = BitString, Value = F> + Sync,
    Term: TerminationCriterion<BitString, F>,
{
    /// Build the binary UMDA instance
    pub fn build(self) -> Result<BinaryUMDA<F, Fit, Term>, EvolutionError> {
        // EV-79: validate configured ranges instead of silently clamping.
        self.config.validate()?;

        let bounds = self
            .bounds
            .ok_or_else(|| EvolutionError::Configuration("Bounds must be specified".to_string()))?;

        let fitness = self.fitness.ok_or_else(|| {
            EvolutionError::Configuration("Fitness function must be specified".to_string())
        })?;

        let termination = self.termination.ok_or_else(|| {
            EvolutionError::Configuration("Termination criterion must be specified".to_string())
        })?;

        Ok(BinaryUMDA {
            config: self.config,
            bounds,
            fitness,
            termination,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// UMDA for binary optimization
pub struct BinaryUMDA<F, Fit, Term>
where
    F: FitnessValue,
{
    config: UMDAConfig,
    bounds: MultiBounds,
    fitness: Fit,
    termination: Term,
    _phantom: std::marker::PhantomData<F>,
}

impl<F, Fit, Term> BinaryUMDA<F, Fit, Term>
where
    F: FitnessValue + Send,
    Fit: Fitness<Genome = BitString, Value = F> + Sync,
    Term: TerminationCriterion<BitString, F>,
{
    /// Create a builder for binary UMDA
    pub fn builder() -> UMDABuilder<BitString, F, (), ()> {
        UMDABuilder::new()
    }

    /// Run the UMDA algorithm.
    pub fn run<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<EvolutionResult<BitString, F>, EvolutionError> {
        self.run_with_model(rng).map(|(result, _model)| result)
    }

    /// Run the UMDA algorithm and also return the final learned univariate model
    /// (EV-10), so callers/tests can check the learned probabilities converged.
    pub fn run_with_model<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<(EvolutionResult<BitString, F>, BinaryUnivariateModel), EvolutionError> {
        let start_time = Instant::now();

        // Get dimension from bounds
        let dimension = self.bounds.dimension();

        // Initialize model
        let mut model = BinaryUnivariateModel::uniform(dimension);

        // Initialize population
        let mut population: Population<BitString, F> =
            Population::random(self.config.population_size, &self.bounds, rng);
        population.evaluate(&self.fitness);

        let mut stats = EvolutionStats::new();
        let mut evaluations = self.config.population_size;
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut generation = 0usize;

        // Track best individual
        let mut best = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation,
                evaluations,
                best_fitness: best.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            let gen_start = Instant::now();

            // Select the top individuals by is_better_than order (EV-80).
            let select_count =
                (self.config.population_size as f64 * self.config.selection_ratio).ceil() as usize;
            let selected: Vec<&BitString> = select_top_genomes(&population, select_count);

            // Update model from selected individuals
            model.update(&selected, &self.config);

            // Sample new population from model
            population = Population::with_capacity(self.config.population_size);
            for _ in 0..self.config.population_size {
                let genome: BitString = model.sample(rng);
                population.push(Individual::new(genome));
            }

            // Evaluate
            population.evaluate(&self.fitness);
            evaluations += self.config.population_size;

            // Update best
            if let Some(pop_best) = population.best() {
                if pop_best.is_better_than(&best) {
                    best = pop_best.clone();
                }
            }

            generation += 1;
            population.set_generation(generation);

            // Record statistics
            let timing = TimingStats::new().with_total(gen_start.elapsed());
            let gen_stats = GenerationStats::from_population(&population, generation, evaluations)
                .with_timing(timing);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        let result =
            EvolutionResult::new(best.genome, best.fitness.unwrap(), generation, evaluations)
                .with_stats(stats);
        Ok((result, model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::{OneMax, Sphere};
    use crate::genome::bounds::Bounds;
    use crate::termination::MaxEvaluations;
    use rand::SeedableRng;

    #[test]
    fn test_continuous_umda_builder() {
        let bounds = MultiBounds::symmetric(5.0, 10);
        let umda: Result<ContinuousUMDA<f64, _, _>, _> = UMDABuilder::new()
            .population_size(50)
            .selection_ratio(0.3)
            .bounds(bounds)
            .fitness(Sphere::new(10))
            .max_generations(10)
            .build();

        assert!(umda.is_ok());
    }

    /// Best Sphere fitness (higher = better; Sphere is negated) found by pure
    /// uniform random search over `budget` samples — the EV-10 baseline.
    fn random_search_sphere_best<R: Rng>(
        dim: usize,
        half_width: f64,
        budget: usize,
        rng: &mut R,
    ) -> f64 {
        let sphere = Sphere::new(dim);
        let mut best = f64::NEG_INFINITY;
        for _ in 0..budget {
            let genes: Vec<f64> = (0..dim)
                .map(|_| rng.gen_range(-half_width..half_width))
                .collect();
            let f = sphere.evaluate(&RealVector::new(genes));
            if f > best {
                best = f;
            }
        }
        best
    }

    /// Best OneMax fitness found by pure uniform random search — EV-10 baseline.
    fn random_search_onemax_best<R: Rng>(dim: usize, budget: usize, rng: &mut R) -> usize {
        let onemax = OneMax::new(dim);
        let mut best = 0usize;
        for _ in 0..budget {
            let bits: Vec<bool> = (0..dim).map(|_| rng.gen::<bool>()).collect();
            let f = onemax.evaluate(&BitString::new(bits));
            if f > best {
                best = f;
            }
        }
        best
    }

    // regression: EV-10 — a working UMDA must (a) beat a same-budget pure random
    // search by a fixed margin and (b) actually learn a model whose means move to
    // the optimum. A broken UMDA that never converges would still clear the old
    // `best_fitness > -50` bar (random search alone reaches ~-12), so this test
    // fails for pure random search and for a non-learning model.
    #[test]
    fn test_continuous_umda_beats_random_search() {
        let budget = 5000;
        let dim = 10;
        let half_width = 5.12;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
            .population_size(100)
            .selection_ratio(0.3)
            .min_variance(0.001)
            .bounds(MultiBounds::symmetric(half_width, dim))
            .fitness(Sphere::new(dim))
            .termination(MaxEvaluations::new(budget))
            .build()
            .unwrap();
        let (result, model) = umda.run_with_model(&mut rng).unwrap();

        let mut baseline_rng = rand::rngs::StdRng::seed_from_u64(42);
        let random_best = random_search_sphere_best(dim, half_width, budget, &mut baseline_rng);

        assert!(
            result.best_fitness > random_best + 5.0,
            "UMDA best {} should beat random search {} by a clear margin",
            result.best_fitness,
            random_best
        );
        assert!(
            result.best_fitness > -2.0,
            "UMDA should get close to the optimum, got {}",
            result.best_fitness
        );

        // The learned distribution itself must have moved toward the optimum.
        for (i, m) in model.means.iter().enumerate() {
            assert!(
                m.abs() < 0.5,
                "learned mean[{i}] = {m} did not converge toward the optimum (0)"
            );
        }
    }

    // regression: EV-10 — binary UMDA must beat a same-budget random search and
    // learn probabilities that converge toward 1. Random search over 3000 draws
    // tops out around 16 ones, so a broken UMDA cannot clear the >= 19 bar.
    #[test]
    fn test_binary_umda_beats_random_search() {
        let budget = 3000;
        let dim = 20;

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let umda: BinaryUMDA<usize, _, _> = UMDABuilder::new()
            .population_size(100)
            .selection_ratio(0.3)
            .prob_bounds(0.02, 0.98)
            .bounds(MultiBounds::uniform(Bounds::unit(), dim))
            .fitness(OneMax::new(dim))
            .termination(MaxEvaluations::new(budget))
            .build()
            .unwrap();
        let (result, model) = umda.run_with_model(&mut rng).unwrap();

        let mut baseline_rng = rand::rngs::StdRng::seed_from_u64(7);
        let random_best = random_search_onemax_best(dim, budget, &mut baseline_rng);

        assert!(
            result.best_fitness > random_best,
            "UMDA best {} should beat random search {}",
            result.best_fitness,
            random_best
        );
        assert!(
            result.best_fitness >= 19,
            "UMDA should nearly solve OneMax, got {}",
            result.best_fitness
        );

        // The learned probabilities must have converged toward 1.
        for (i, p) in model.probabilities.iter().enumerate() {
            assert!(
                *p > 0.8,
                "learned probability[{i}] = {p} did not converge toward 1"
            );
        }
    }

    // regression: EV-79 — build() validates configured ranges and returns an error
    // instead of silently clamping (or accepting a no-op learning_rate of 0).
    #[test]
    fn test_umda_builder_validation_rejects_out_of_range() {
        let bounds = MultiBounds::symmetric(5.0, 4);

        let bad_ratio: Result<ContinuousUMDA<f64, _, _>, _> = UMDABuilder::new()
            .population_size(50)
            .selection_ratio(1.5)
            .bounds(bounds.clone())
            .fitness(Sphere::new(4))
            .max_generations(5)
            .build();
        assert!(bad_ratio.is_err(), "selection_ratio 1.5 must be rejected");

        let zero_lr: Result<ContinuousUMDA<f64, _, _>, _> = UMDABuilder::new()
            .population_size(50)
            .learning_rate(0.0)
            .bounds(bounds.clone())
            .fitness(Sphere::new(4))
            .max_generations(5)
            .build();
        assert!(
            zero_lr.is_err(),
            "learning_rate 0.0 (a no-op) must be rejected"
        );

        let bad_probs: Result<BinaryUMDA<usize, _, _>, _> = UMDABuilder::new()
            .population_size(50)
            .prob_bounds(0.6, 0.4)
            .bounds(MultiBounds::uniform(Bounds::unit(), 4))
            .fitness(OneMax::new(4))
            .max_generations(5)
            .build();
        assert!(
            bad_probs.is_err(),
            "prob_bounds with min > max must be rejected"
        );

        let ok: Result<ContinuousUMDA<f64, _, _>, _> = UMDABuilder::new()
            .population_size(50)
            .selection_ratio(0.3)
            .learning_rate(0.5)
            .bounds(bounds)
            .fitness(Sphere::new(4))
            .max_generations(5)
            .build();
        assert!(ok.is_ok(), "a valid configuration must still build");
    }

    // regression: EV-81 — the continuous model uses the unbiased (n-1) sample
    // variance. For selected values {1,2,3} at a dimension the Bessel-corrected
    // variance is 1.0, whereas the old biased /n estimate gave 2/3.
    #[test]
    fn test_continuous_variance_is_bessel_corrected() {
        let bounds = MultiBounds::symmetric(5.0, 1);
        let mut model = ContinuousUnivariateModel::from_bounds(&bounds);
        let config = UMDAConfig {
            learning_rate: 1.0,
            min_variance: 1e-12, // don't let the floor mask the estimate
            ..Default::default()
        };

        let g1 = RealVector::new(vec![1.0]);
        let g2 = RealVector::new(vec![2.0]);
        let g3 = RealVector::new(vec![3.0]);
        model.update(&[&g1, &g2, &g3], &config);

        assert!(
            (model.variances[0] - 1.0).abs() < 1e-9,
            "expected Bessel-corrected variance 1.0, got {}",
            model.variances[0]
        );
    }

    #[test]
    fn test_continuous_model_update() {
        let bounds = MultiBounds::symmetric(5.0, 3);
        let mut model = ContinuousUnivariateModel::from_bounds(&bounds);
        let config = UMDAConfig::default();

        // Create some test individuals
        let g1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let g2 = RealVector::new(vec![2.0, 3.0, 4.0]);
        let g3 = RealVector::new(vec![3.0, 4.0, 5.0]);
        let selected = vec![&g1, &g2, &g3];

        model.update(&selected, &config);

        // Mean should be average: (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert!((model.means[0] - 2.0).abs() < 0.01);
        assert!((model.means[1] - 3.0).abs() < 0.01);
        assert!((model.means[2] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_binary_model_update() {
        let mut model = BinaryUnivariateModel::uniform(4);
        let config = UMDAConfig::default();

        // Create test individuals
        let g1 = BitString::new(vec![true, true, false, false]);
        let g2 = BitString::new(vec![true, false, true, false]);
        let g3 = BitString::new(vec![true, false, false, true]);
        let selected = vec![&g1, &g2, &g3];

        model.update(&selected, &config);

        // Position 0: all true -> p=0.99 (bounded)
        // Position 1: 1/3 true -> p=0.33
        // Position 2: 1/3 true -> p=0.33
        // Position 3: 1/3 true -> p=0.33
        assert!(model.probabilities[0] > 0.9);
        assert!(model.probabilities[1] > 0.2 && model.probabilities[1] < 0.5);
    }

    #[test]
    fn test_learning_rate() {
        let bounds = MultiBounds::symmetric(5.0, 2);
        let mut model = ContinuousUnivariateModel::from_bounds(&bounds);
        let config = UMDAConfig {
            learning_rate: 0.5,
            ..Default::default()
        };

        // Initial means should be 0.0 (center of [-5, 5])
        let initial_mean = model.means[0];

        // Create individuals at 2.0
        let g1 = RealVector::new(vec![2.0, 2.0]);
        let selected = vec![&g1];

        model.update(&selected, &config);

        // With learning rate 0.5, new mean = 0.5 * 2.0 + 0.5 * 0.0 = 1.0
        assert!((model.means[0] - (0.5 * 2.0 + 0.5 * initial_mean)).abs() < 0.01);
    }

    // regression: EV-39 — sampling rejects out-of-bounds draws instead of clamping,
    // so a model mean sitting exactly on a bound does not pile ~50% of samples on
    // that boundary (which shrinks the next variance estimate). Rejection yields
    // essentially zero exact-boundary samples; the old clamp yielded ~half.
    #[test]
    fn test_sample_rejection_avoids_boundary_pileup() {
        let bounds = MultiBounds::symmetric(5.0, 1); // [-5, 5]
        let mut model = ContinuousUnivariateModel::from_bounds(&bounds);
        model.means[0] = 5.0; // mean exactly on the upper bound
        model.variances[0] = 1.0; // non-trivial spread

        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let n = 2000;
        let on_boundary = (0..n)
            .filter(|_| {
                let g = model.sample(&bounds, &mut rng);
                (g.genes()[0] - 5.0).abs() < 1e-12
            })
            .count();

        assert!(
            on_boundary <= 2,
            "rejection sampling should not pile samples on the boundary, got {on_boundary}/{n}"
        );
    }

    /// A fitness where LOWER is better while `to_f64` still reports the raw value —
    /// so `is_better_than` and `to_f64`-descending order disagree (mirrors the
    /// ParetoFitness hazard flagged by EV-80).
    #[derive(Clone, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
    struct LowerBetter(f64);

    impl FitnessValue for LowerBetter {
        fn to_f64(&self) -> f64 {
            self.0
        }
        fn is_better_than(&self, other: &Self) -> bool {
            self.0 < other.0
        }
    }

    // regression: EV-80 — truncation selection ranks by is_better_than, not by
    // to_f64-descending order. With a lower-is-better fitness, the top pick must be
    // the lowest value; a to_f64-descending sort (the pre-fix behavior) would pick
    // the highest.
    #[test]
    fn test_select_top_uses_is_better_than() {
        let mut pop: Population<RealVector, LowerBetter> = Population::new();
        pop.push(Individual::with_fitness(
            RealVector::new(vec![30.0]),
            LowerBetter(3.0),
        ));
        pop.push(Individual::with_fitness(
            RealVector::new(vec![10.0]),
            LowerBetter(1.0),
        ));
        pop.push(Individual::with_fitness(
            RealVector::new(vec![20.0]),
            LowerBetter(2.0),
        ));

        let top = select_top_genomes(&pop, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(
            top[0].genes()[0],
            10.0,
            "selection should pick the is_better_than-best (lowest) individual"
        );
    }

    // regression: EV-34 — run_with_callback reports every generation in order and
    // returns the same result as `run` when the callback never cancels.
    #[test]
    fn test_umda_run_with_callback_reports_progress() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
            .population_size(40)
            .selection_ratio(0.3)
            .bounds(MultiBounds::symmetric(5.12, 4))
            .fitness(Sphere::new(4))
            .max_generations(12)
            .build()
            .unwrap();

        let mut seen: Vec<usize> = Vec::new();
        let result = umda
            .run_with_callback(&mut rng, |generation, best| {
                assert!(best.is_finite());
                seen.push(generation);
                true
            })
            .unwrap();

        assert_eq!(seen, (0..result.generations).collect::<Vec<_>>());
        assert!(!seen.is_empty());
    }

    #[test]
    fn test_umda_run_with_callback_cancels_early() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
            .population_size(40)
            .selection_ratio(0.3)
            .bounds(MultiBounds::symmetric(5.12, 4))
            .fitness(Sphere::new(4))
            .max_generations(10_000)
            .build()
            .unwrap();

        let mut calls = 0usize;
        let result = umda
            .run_with_callback(&mut rng, |_generation, _best| {
                calls += 1;
                calls < 6
            })
            .unwrap();

        assert!(calls <= 6, "callback must stop being called after cancel");
        assert!(
            result.generations < 10,
            "run must stop far short of the 10k budget, got {}",
            result.generations
        );
    }
}
