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

/// Configuration for UMDA
#[derive(Clone, Debug)]
pub struct UMDAConfig {
    /// Population size
    pub population_size: usize,
    /// Selection ratio (top proportion selected for model learning)
    pub selection_ratio: f64,
    /// Minimum variance to prevent collapse (for continuous)
    pub min_variance: f64,
    /// Probability bounds for binary (to prevent determinism)
    pub prob_bounds: (f64, f64),
    /// Learning rate for model update (1.0 = replace, <1.0 = blend with previous)
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

    /// Set the selection ratio
    pub fn selection_ratio(mut self, ratio: f64) -> Self {
        self.config.selection_ratio = ratio.clamp(0.1, 0.9);
        self
    }

    /// Set the minimum variance
    pub fn min_variance(mut self, variance: f64) -> Self {
        self.config.min_variance = variance;
        self
    }

    /// Set probability bounds for binary UMDA
    pub fn prob_bounds(mut self, min: f64, max: f64) -> Self {
        self.config.prob_bounds = (min.clamp(0.001, 0.5), max.clamp(0.5, 0.999));
        self
    }

    /// Set the learning rate for model update
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.config.learning_rate = rate.clamp(0.0, 1.0);
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

            // Compute sample variance
            let variance: f64 = selected
                .iter()
                .map(|g| (g.genes()[i] - mean).powi(2))
                .sum::<f64>()
                / n;

            // Apply learning rate and bounds
            self.means[i] =
                config.learning_rate * mean + (1.0 - config.learning_rate) * self.means[i];
            self.variances[i] = config.learning_rate * variance.max(config.min_variance)
                + (1.0 - config.learning_rate) * self.variances[i];
        }
    }

    /// Sample a new individual from the model
    pub fn sample<R: Rng>(&self, bounds: &MultiBounds, rng: &mut R) -> RealVector {
        let genes: Vec<f64> = self
            .means
            .iter()
            .zip(self.variances.iter())
            .zip(bounds.bounds.iter())
            .map(|((mean, var), bound)| {
                let normal =
                    Normal::new(*mean, var.sqrt()).unwrap_or(Normal::new(*mean, 0.1).unwrap());
                let value = normal.sample(rng);
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

    /// Run the UMDA algorithm
    pub fn run<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<EvolutionResult<RealVector, F>, EvolutionError> {
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

            let gen_start = Instant::now();

            // Sort and select top individuals
            population.sort_by_fitness();
            let select_count =
                (self.config.population_size as f64 * self.config.selection_ratio).ceil() as usize;
            let selected: Vec<&RealVector> = population
                .iter()
                .take(select_count)
                .map(|ind| &ind.genome)
                .collect();

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

        Ok(
            EvolutionResult::new(best.genome, best.fitness.unwrap(), generation, evaluations)
                .with_stats(stats),
        )
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

    /// Run the UMDA algorithm
    pub fn run<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<EvolutionResult<BitString, F>, EvolutionError> {
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

            // Sort and select top individuals
            population.sort_by_fitness();
            let select_count =
                (self.config.population_size as f64 * self.config.selection_ratio).ceil() as usize;
            let selected: Vec<&BitString> = population
                .iter()
                .take(select_count)
                .map(|ind| &ind.genome)
                .collect();

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

        Ok(
            EvolutionResult::new(best.genome, best.fitness.unwrap(), generation, evaluations)
                .with_stats(stats),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::{OneMax, Sphere};
    use crate::termination::MaxEvaluations;

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

    #[test]
    fn test_continuous_umda_sphere() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
            .population_size(100)
            .selection_ratio(0.3)
            .min_variance(0.01)
            .bounds(bounds)
            .fitness(Sphere::new(10))
            .termination(MaxEvaluations::new(5000))
            .build()
            .unwrap();

        let result = umda.run(&mut rng).unwrap();

        // Should find improvement
        assert!(
            result.best_fitness > -50.0,
            "Expected fitness > -50, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_binary_umda_onemax() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::uniform(crate::genome::bounds::Bounds::unit(), 20);

        let umda: BinaryUMDA<usize, _, _> = UMDABuilder::new()
            .population_size(100)
            .selection_ratio(0.3)
            .prob_bounds(0.05, 0.95)
            .bounds(bounds)
            .fitness(OneMax::new(20))
            .termination(MaxEvaluations::new(3000))
            .build()
            .unwrap();

        let result = umda.run(&mut rng).unwrap();

        // Should find good solution
        assert!(
            result.best_fitness >= 15,
            "Expected fitness >= 15, got {}",
            result.best_fitness
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
}
