//! Evolution Strategies: (μ+λ)-ES and (μ,λ)-ES
//!
//! This module implements the classic Evolution Strategy algorithms:
//! - (μ+λ)-ES: Parents compete with offspring for survival
//! - (μ,λ)-ES: Only offspring compete, parents are discarded
//!
//! Both support self-adaptive mutation through the `AdaptiveGenome` wrapper.

use std::time::Instant;

use rand::Rng;
use rand_distr::StandardNormal;

use crate::diagnostics::{EvolutionResult, EvolutionStats, GenerationStats, TimingStats};
use crate::error::EvolutionError;
use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{EvolutionaryGenome, RealValuedGenome};
use crate::hyperparameter::self_adaptive::{AdaptiveGenome, StrategyParams};
use crate::population::individual::Individual;
use crate::population::population::Population;
use crate::termination::{EvolutionState, MaxGenerations, TerminationCriterion};

/// Selection strategy for Evolution Strategies
#[derive(Clone, Debug, Default)]
pub enum ESSelectionStrategy {
    /// (μ+λ): Select best μ from parents + offspring combined
    #[default]
    MuPlusLambda,
    /// (μ,λ): Select best μ from offspring only (requires λ ≥ μ)
    MuCommaLambda,
}

/// Configuration for Evolution Strategies
#[derive(Clone, Debug)]
pub struct ESConfig {
    /// Number of parents (μ)
    pub mu: usize,
    /// Number of offspring (λ)
    pub lambda: usize,
    /// Selection strategy
    pub selection: ESSelectionStrategy,
    /// Initial step size (σ)
    pub initial_sigma: f64,
    /// Use self-adaptive mutation (evolve σ)
    pub self_adaptive: bool,
    /// Recombination type
    pub recombination: RecombinationType,
}

impl Default for ESConfig {
    fn default() -> Self {
        Self {
            mu: 15,
            lambda: 100,
            selection: ESSelectionStrategy::MuPlusLambda,
            initial_sigma: 1.0,
            self_adaptive: true,
            recombination: RecombinationType::Intermediate,
        }
    }
}

impl ESConfig {
    /// Create a (μ+λ)-ES configuration
    pub fn mu_plus_lambda(mu: usize, lambda: usize) -> Self {
        Self {
            mu,
            lambda,
            selection: ESSelectionStrategy::MuPlusLambda,
            ..Default::default()
        }
    }

    /// Create a (μ,λ)-ES configuration
    pub fn mu_comma_lambda(mu: usize, lambda: usize) -> Result<Self, EvolutionError> {
        if lambda < mu {
            return Err(EvolutionError::Configuration(format!(
                "For (μ,λ)-ES, λ ({}) must be >= μ ({})",
                lambda, mu
            )));
        }
        Ok(Self {
            mu,
            lambda,
            selection: ESSelectionStrategy::MuCommaLambda,
            ..Default::default()
        })
    }
}

/// Recombination type for ES
#[derive(Clone, Debug, Default)]
pub enum RecombinationType {
    /// No recombination (asexual)
    None,
    /// Discrete recombination (randomly select genes from parents)
    Discrete,
    /// Intermediate recombination (average of parents)
    #[default]
    Intermediate,
    /// Global intermediate (average of all parents)
    GlobalIntermediate,
}

/// Builder for Evolution Strategy
pub struct ESBuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: ESConfig,
    bounds: Option<MultiBounds>,
    fitness: Option<Fit>,
    termination: Option<Term>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F> ESBuilder<G, F, (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: ESConfig::default(),
            bounds: None,
            fitness: None,
            termination: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a (μ+λ)-ES builder
    pub fn mu_plus_lambda(mu: usize, lambda: usize) -> Self {
        Self {
            config: ESConfig::mu_plus_lambda(mu, lambda),
            bounds: None,
            fitness: None,
            termination: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a (μ,λ)-ES builder
    pub fn mu_comma_lambda(mu: usize, lambda: usize) -> Result<Self, EvolutionError> {
        Ok(Self {
            config: ESConfig::mu_comma_lambda(mu, lambda)?,
            bounds: None,
            fitness: None,
            termination: None,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<G, F> Default for ESBuilder<G, F, (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, Fit, Term> ESBuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Set μ (number of parents)
    pub fn mu(mut self, mu: usize) -> Self {
        self.config.mu = mu;
        self
    }

    /// Set λ (number of offspring)
    pub fn lambda(mut self, lambda: usize) -> Self {
        self.config.lambda = lambda;
        self
    }

    /// Set the selection strategy
    pub fn selection_strategy(mut self, strategy: ESSelectionStrategy) -> Self {
        self.config.selection = strategy;
        self
    }

    /// Set the initial step size
    pub fn initial_sigma(mut self, sigma: f64) -> Self {
        self.config.initial_sigma = sigma;
        self
    }

    /// Enable or disable self-adaptive mutation
    pub fn self_adaptive(mut self, enabled: bool) -> Self {
        self.config.self_adaptive = enabled;
        self
    }

    /// Set the recombination type
    pub fn recombination(mut self, recomb: RecombinationType) -> Self {
        self.config.recombination = recomb;
        self
    }

    /// Set the search space bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the fitness function
    pub fn fitness<NewFit>(self, fitness: NewFit) -> ESBuilder<G, F, NewFit, Term>
    where
        NewFit: Fitness<Genome = G, Value = F>,
    {
        ESBuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: Some(fitness),
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the termination criterion
    pub fn termination<NewTerm>(self, termination: NewTerm) -> ESBuilder<G, F, Fit, NewTerm>
    where
        NewTerm: TerminationCriterion<G, F>,
    {
        ESBuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: self.fitness,
            termination: Some(termination),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set max generations (convenience method)
    pub fn max_generations(self, max: usize) -> ESBuilder<G, F, Fit, MaxGenerations> {
        ESBuilder {
            config: self.config,
            bounds: self.bounds,
            fitness: self.fitness,
            termination: Some(MaxGenerations::new(max)),
            _phantom: std::marker::PhantomData,
        }
    }
}

// Parallel version with Send + Sync bounds
#[cfg(feature = "parallel")]
impl<G, F, Fit, Term> ESBuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome + RealValuedGenome + Send + Sync,
    F: FitnessValue + Send,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Build the Evolution Strategy instance
    pub fn build(self) -> Result<EvolutionStrategy<G, F, Fit, Term>, EvolutionError> {
        let bounds = self
            .bounds
            .ok_or_else(|| EvolutionError::Configuration("Bounds must be specified".to_string()))?;

        let fitness = self.fitness.ok_or_else(|| {
            EvolutionError::Configuration("Fitness function must be specified".to_string())
        })?;

        let termination = self.termination.ok_or_else(|| {
            EvolutionError::Configuration("Termination criterion must be specified".to_string())
        })?;

        // Validate (μ,λ) constraint
        if matches!(self.config.selection, ESSelectionStrategy::MuCommaLambda)
            && self.config.lambda < self.config.mu
        {
            return Err(EvolutionError::Configuration(format!(
                "For (μ,λ)-ES, λ ({}) must be >= μ ({})",
                self.config.lambda, self.config.mu
            )));
        }

        Ok(EvolutionStrategy {
            config: self.config,
            bounds,
            fitness,
            termination,
            _phantom: std::marker::PhantomData,
        })
    }
}

// Non-parallel version of build()
#[cfg(not(feature = "parallel"))]
impl<G, F, Fit, Term> ESBuilder<G, F, Fit, Term>
where
    G: EvolutionaryGenome + RealValuedGenome,
    F: FitnessValue,
    Fit: Fitness<Genome = G, Value = F>,
    Term: TerminationCriterion<G, F>,
{
    /// Build the Evolution Strategy instance
    pub fn build(self) -> Result<EvolutionStrategy<G, F, Fit, Term>, EvolutionError> {
        let bounds = self
            .bounds
            .ok_or_else(|| EvolutionError::Configuration("Bounds must be specified".to_string()))?;

        let fitness = self.fitness.ok_or_else(|| {
            EvolutionError::Configuration("Fitness function must be specified".to_string())
        })?;

        let termination = self.termination.ok_or_else(|| {
            EvolutionError::Configuration("Termination criterion must be specified".to_string())
        })?;

        // Validate (μ,λ) constraint
        if matches!(self.config.selection, ESSelectionStrategy::MuCommaLambda)
            && self.config.lambda < self.config.mu
        {
            return Err(EvolutionError::Configuration(format!(
                "For (μ,λ)-ES, λ ({}) must be >= μ ({})",
                self.config.lambda, self.config.mu
            )));
        }

        Ok(EvolutionStrategy {
            config: self.config,
            bounds,
            fitness,
            termination,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Evolution Strategy (μ+λ)-ES or (μ,λ)-ES
///
/// A classic evolutionary algorithm using Gaussian mutation and optional
/// self-adaptive step size control.
pub struct EvolutionStrategy<G, F, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: ESConfig,
    bounds: MultiBounds,
    fitness: Fit,
    termination: Term,
    _phantom: std::marker::PhantomData<(G, F)>,
}

// Parallel version with Send + Sync bounds
#[cfg(feature = "parallel")]
impl<G, F, Fit, Term> EvolutionStrategy<G, F, Fit, Term>
where
    G: EvolutionaryGenome + RealValuedGenome + Send + Sync,
    F: FitnessValue + Send,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Create a builder for Evolution Strategy
    pub fn builder() -> ESBuilder<G, F, (), ()> {
        ESBuilder::new()
    }

    /// Run the evolution strategy
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Initialize population with adaptive genomes
        let mut population: Vec<(AdaptiveGenome<G>, F)> = (0..self.config.mu)
            .map(|_| {
                let genome = G::generate(rng, &self.bounds);
                let adaptive = if self.config.self_adaptive {
                    AdaptiveGenome::new_non_isotropic(
                        genome,
                        vec![self.config.initial_sigma; self.bounds.dimension()],
                    )
                } else {
                    AdaptiveGenome::new_isotropic(genome, self.config.initial_sigma)
                };
                let fitness = self.fitness.evaluate(adaptive.inner());
                (adaptive, fitness)
            })
            .collect();

        // Sort by fitness (descending for maximization)
        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut stats = EvolutionStats::new();
        let mut evaluations = self.config.mu;
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut generation = 0usize;

        // Track best individual
        let mut best = population[0].clone();

        // Create a tracking population for termination checks
        let mut tracking_population: Population<G, F> = Population::with_capacity(self.config.mu);
        for (adaptive, fit) in &population {
            let mut ind = Individual::new(adaptive.inner().clone());
            ind.set_fitness(fit.clone());
            tracking_population.push(ind);
        }

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&tracking_population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main evolution loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation,
                evaluations,
                best_fitness: best.1.to_f64(),
                population: &tracking_population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            let gen_start = Instant::now();

            // Generate offspring
            let mut offspring: Vec<(AdaptiveGenome<G>, F)> = Vec::with_capacity(self.config.lambda);

            for _ in 0..self.config.lambda {
                // Select parent(s) for recombination
                let child = match &self.config.recombination {
                    RecombinationType::None => {
                        // Clone a random parent
                        let parent_idx = rng.gen_range(0..self.config.mu);
                        population[parent_idx].0.clone()
                    }
                    RecombinationType::Discrete => {
                        // Select two parents, randomly pick genes
                        let p1_idx = rng.gen_range(0..self.config.mu);
                        let p2_idx = rng.gen_range(0..self.config.mu);
                        self.discrete_recombination(
                            &population[p1_idx].0,
                            &population[p2_idx].0,
                            rng,
                        )
                    }
                    RecombinationType::Intermediate => {
                        // Average of two parents
                        let p1_idx = rng.gen_range(0..self.config.mu);
                        let p2_idx = rng.gen_range(0..self.config.mu);
                        self.intermediate_recombination(
                            &population[p1_idx].0,
                            &population[p2_idx].0,
                        )
                    }
                    RecombinationType::GlobalIntermediate => {
                        // Average of all parents
                        self.global_intermediate_recombination(&population)
                    }
                };

                // Mutate
                let mutated = self.mutate(child, rng);

                // Evaluate
                let fitness = self.fitness.evaluate(mutated.inner());
                offspring.push((mutated, fitness));
            }
            evaluations += self.config.lambda;

            // Selection
            match self.config.selection {
                ESSelectionStrategy::MuPlusLambda => {
                    // Combine parents and offspring
                    let mut combined = population;
                    combined.extend(offspring);
                    combined
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    population = combined.into_iter().take(self.config.mu).collect();
                }
                ESSelectionStrategy::MuCommaLambda => {
                    // Select only from offspring
                    offspring
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    population = offspring.into_iter().take(self.config.mu).collect();
                }
            }

            // Update best
            if population[0].1.is_better_than(&best.1) {
                best = population[0].clone();
            }

            generation += 1;

            // Update tracking population for statistics
            tracking_population.clear();
            for (adaptive, fit) in &population {
                let mut ind = Individual::new(adaptive.inner().clone());
                ind.set_fitness(fit.clone());
                tracking_population.push(ind);
            }
            tracking_population.set_generation(generation);

            // Record statistics
            let timing = TimingStats::new().with_total(gen_start.elapsed());
            let gen_stats =
                GenerationStats::from_population(&tracking_population, generation, evaluations)
                    .with_timing(timing);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        Ok(
            EvolutionResult::new(best.0.into_inner(), best.1, generation, evaluations)
                .with_stats(stats),
        )
    }

    /// Discrete recombination: randomly select genes from parents
    fn discrete_recombination<R: Rng>(
        &self,
        p1: &AdaptiveGenome<G>,
        p2: &AdaptiveGenome<G>,
        rng: &mut R,
    ) -> AdaptiveGenome<G> {
        let genes1 = p1.inner().genes();
        let genes2 = p2.inner().genes();

        let child_genes: Vec<f64> = genes1
            .iter()
            .zip(genes2.iter())
            .map(|(g1, g2)| if rng.gen_bool(0.5) { *g1 } else { *g2 })
            .collect();

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Recombine strategy parameters
        match (&p1.strategy, &p2.strategy) {
            (StrategyParams::Isotropic(s1), StrategyParams::Isotropic(s2)) => {
                AdaptiveGenome::new_isotropic(child_genome, (s1 * s2).sqrt())
            }
            (StrategyParams::NonIsotropic(s1), StrategyParams::NonIsotropic(s2)) => {
                let sigmas: Vec<f64> = s1
                    .iter()
                    .zip(s2.iter())
                    .map(|(a, b)| if rng.gen_bool(0.5) { *a } else { *b })
                    .collect();
                AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
            }
            _ => AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma),
        }
    }

    /// Intermediate recombination: average of two parents
    fn intermediate_recombination(
        &self,
        p1: &AdaptiveGenome<G>,
        p2: &AdaptiveGenome<G>,
    ) -> AdaptiveGenome<G> {
        let genes1 = p1.inner().genes();
        let genes2 = p2.inner().genes();

        let child_genes: Vec<f64> = genes1
            .iter()
            .zip(genes2.iter())
            .map(|(g1, g2)| (g1 + g2) / 2.0)
            .collect();

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Average strategy parameters
        match (&p1.strategy, &p2.strategy) {
            (StrategyParams::Isotropic(s1), StrategyParams::Isotropic(s2)) => {
                AdaptiveGenome::new_isotropic(child_genome, (s1 * s2).sqrt())
            }
            (StrategyParams::NonIsotropic(s1), StrategyParams::NonIsotropic(s2)) => {
                let sigmas: Vec<f64> = s1
                    .iter()
                    .zip(s2.iter())
                    .map(|(a, b)| (a * b).sqrt())
                    .collect();
                AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
            }
            _ => AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma),
        }
    }

    /// Global intermediate recombination: average of all parents
    fn global_intermediate_recombination(
        &self,
        population: &[(AdaptiveGenome<G>, F)],
    ) -> AdaptiveGenome<G> {
        let n = population.len();
        let dim = population[0].0.inner().genes().len();

        let mut child_genes = vec![0.0; dim];
        for (adaptive, _) in population {
            for (i, gene) in adaptive.inner().genes().iter().enumerate() {
                child_genes[i] += gene / n as f64;
            }
        }

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Average all strategy parameters
        let avg_sigma = if self.config.self_adaptive {
            // Average the sigmas from all parents
            let sigmas: Vec<f64> = (0..dim)
                .map(|i| {
                    let sum: f64 = population
                        .iter()
                        .map(|(a, _)| a.strategy.get_sigma(i))
                        .sum();
                    sum / n as f64
                })
                .collect();
            AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
        } else {
            AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma)
        };

        avg_sigma
    }

    /// Mutate an adaptive genome
    fn mutate<R: Rng>(&self, mut genome: AdaptiveGenome<G>, rng: &mut R) -> AdaptiveGenome<G> {
        let n = genome.inner().genes().len();

        // Self-adaptive: mutate strategy parameters first
        if self.config.self_adaptive {
            genome.strategy.mutate(n, rng);
        }

        // Collect sigmas first to avoid borrow conflicts
        let sigmas: Vec<f64> = (0..n).map(|i| genome.strategy.get_sigma(i)).collect();

        // Then mutate the genome using the (possibly updated) strategy
        let genes = genome.inner_mut().genes_mut();
        for i in 0..genes.len() {
            let perturbation: f64 = rng.sample(StandardNormal);
            genes[i] += sigmas[i] * perturbation;

            // Clamp to bounds
            if let Some(b) = self.bounds.get(i) {
                genes[i] = genes[i].clamp(b.min, b.max);
            }
        }

        genome
    }
}

// Non-parallel version without Send + Sync bounds
#[cfg(not(feature = "parallel"))]
impl<G, F, Fit, Term> EvolutionStrategy<G, F, Fit, Term>
where
    G: EvolutionaryGenome + RealValuedGenome,
    F: FitnessValue,
    Fit: Fitness<Genome = G, Value = F>,
    Term: TerminationCriterion<G, F>,
{
    /// Create a builder for Evolution Strategy
    pub fn builder() -> ESBuilder<G, F, (), ()> {
        ESBuilder::new()
    }

    /// Run the evolution strategy
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Initialize population with adaptive genomes
        let mut population: Vec<(AdaptiveGenome<G>, F)> = (0..self.config.mu)
            .map(|_| {
                let genome = G::generate(rng, &self.bounds);
                let adaptive = if self.config.self_adaptive {
                    AdaptiveGenome::new_non_isotropic(
                        genome,
                        vec![self.config.initial_sigma; self.bounds.dimension()],
                    )
                } else {
                    AdaptiveGenome::new_isotropic(genome, self.config.initial_sigma)
                };
                let fitness = self.fitness.evaluate(adaptive.inner());
                (adaptive, fitness)
            })
            .collect();

        // Sort by fitness (descending for maximization)
        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut stats = EvolutionStats::new();
        let mut evaluations = self.config.mu;
        let mut fitness_history: Vec<f64> = Vec::new();
        let mut generation = 0usize;

        // Track best individual
        let mut best = population[0].clone();

        // Create a tracking population for termination checks
        let mut tracking_population: Population<G, F> = Population::with_capacity(self.config.mu);
        for (adaptive, fit) in &population {
            let mut ind = Individual::new(adaptive.inner().clone());
            ind.set_fitness(fit.clone());
            tracking_population.push(ind);
        }

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&tracking_population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main evolution loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation,
                evaluations,
                best_fitness: best.1.to_f64(),
                population: &tracking_population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            let gen_start = Instant::now();

            // Generate offspring
            let mut offspring: Vec<(AdaptiveGenome<G>, F)> = Vec::with_capacity(self.config.lambda);

            for _ in 0..self.config.lambda {
                // Select parent(s) for recombination
                let child = match &self.config.recombination {
                    RecombinationType::None => {
                        // Clone a random parent
                        let parent_idx = rng.gen_range(0..self.config.mu);
                        population[parent_idx].0.clone()
                    }
                    RecombinationType::Discrete => {
                        // Select two parents, randomly pick genes
                        let p1_idx = rng.gen_range(0..self.config.mu);
                        let p2_idx = rng.gen_range(0..self.config.mu);
                        self.discrete_recombination(
                            &population[p1_idx].0,
                            &population[p2_idx].0,
                            rng,
                        )
                    }
                    RecombinationType::Intermediate => {
                        // Average of two parents
                        let p1_idx = rng.gen_range(0..self.config.mu);
                        let p2_idx = rng.gen_range(0..self.config.mu);
                        self.intermediate_recombination(
                            &population[p1_idx].0,
                            &population[p2_idx].0,
                        )
                    }
                    RecombinationType::GlobalIntermediate => {
                        // Average of all parents
                        self.global_intermediate_recombination(&population)
                    }
                };

                // Mutate
                let mutated = self.mutate(child, rng);

                // Evaluate
                let fitness = self.fitness.evaluate(mutated.inner());
                offspring.push((mutated, fitness));
            }
            evaluations += self.config.lambda;

            // Selection
            match self.config.selection {
                ESSelectionStrategy::MuPlusLambda => {
                    // Combine parents and offspring
                    let mut combined = population;
                    combined.extend(offspring);
                    combined
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    population = combined.into_iter().take(self.config.mu).collect();
                }
                ESSelectionStrategy::MuCommaLambda => {
                    // Select only from offspring
                    offspring
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    population = offspring.into_iter().take(self.config.mu).collect();
                }
            }

            // Update best
            if population[0].1.is_better_than(&best.1) {
                best = population[0].clone();
            }

            generation += 1;

            // Update tracking population for statistics
            tracking_population.clear();
            for (adaptive, fit) in &population {
                let mut ind = Individual::new(adaptive.inner().clone());
                ind.set_fitness(fit.clone());
                tracking_population.push(ind);
            }
            tracking_population.set_generation(generation);

            // Record statistics
            let timing = TimingStats::new().with_total(gen_start.elapsed());
            let gen_stats =
                GenerationStats::from_population(&tracking_population, generation, evaluations)
                    .with_timing(timing);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        Ok(
            EvolutionResult::new(best.0.into_inner(), best.1, generation, evaluations)
                .with_stats(stats),
        )
    }

    /// Discrete recombination: randomly select genes from parents
    fn discrete_recombination<R: Rng>(
        &self,
        p1: &AdaptiveGenome<G>,
        p2: &AdaptiveGenome<G>,
        rng: &mut R,
    ) -> AdaptiveGenome<G> {
        let genes1 = p1.inner().genes();
        let genes2 = p2.inner().genes();

        let child_genes: Vec<f64> = genes1
            .iter()
            .zip(genes2.iter())
            .map(|(g1, g2)| if rng.gen_bool(0.5) { *g1 } else { *g2 })
            .collect();

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Recombine strategy parameters
        match (&p1.strategy, &p2.strategy) {
            (StrategyParams::Isotropic(s1), StrategyParams::Isotropic(s2)) => {
                AdaptiveGenome::new_isotropic(child_genome, (s1 * s2).sqrt())
            }
            (StrategyParams::NonIsotropic(s1), StrategyParams::NonIsotropic(s2)) => {
                let sigmas: Vec<f64> = s1
                    .iter()
                    .zip(s2.iter())
                    .map(|(a, b)| if rng.gen_bool(0.5) { *a } else { *b })
                    .collect();
                AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
            }
            _ => AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma),
        }
    }

    /// Intermediate recombination: average of two parents
    fn intermediate_recombination(
        &self,
        p1: &AdaptiveGenome<G>,
        p2: &AdaptiveGenome<G>,
    ) -> AdaptiveGenome<G> {
        let genes1 = p1.inner().genes();
        let genes2 = p2.inner().genes();

        let child_genes: Vec<f64> = genes1
            .iter()
            .zip(genes2.iter())
            .map(|(g1, g2)| (g1 + g2) / 2.0)
            .collect();

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Average strategy parameters
        match (&p1.strategy, &p2.strategy) {
            (StrategyParams::Isotropic(s1), StrategyParams::Isotropic(s2)) => {
                AdaptiveGenome::new_isotropic(child_genome, (s1 * s2).sqrt())
            }
            (StrategyParams::NonIsotropic(s1), StrategyParams::NonIsotropic(s2)) => {
                let sigmas: Vec<f64> = s1
                    .iter()
                    .zip(s2.iter())
                    .map(|(a, b)| (a * b).sqrt())
                    .collect();
                AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
            }
            _ => AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma),
        }
    }

    /// Global intermediate recombination: average of all parents
    fn global_intermediate_recombination(
        &self,
        population: &[(AdaptiveGenome<G>, F)],
    ) -> AdaptiveGenome<G> {
        let n = population.len();
        let dim = population[0].0.inner().genes().len();

        let mut child_genes = vec![0.0; dim];
        for (adaptive, _) in population {
            for (i, gene) in adaptive.inner().genes().iter().enumerate() {
                child_genes[i] += gene / n as f64;
            }
        }

        let child_genome = G::from_genes(child_genes).expect("Failed to create genome from genes");

        // Average all strategy parameters
        let avg_sigma = if self.config.self_adaptive {
            // Average the sigmas from all parents
            let sigmas: Vec<f64> = (0..dim)
                .map(|i| {
                    let sum: f64 = population
                        .iter()
                        .map(|(a, _)| a.strategy.get_sigma(i))
                        .sum();
                    sum / n as f64
                })
                .collect();
            AdaptiveGenome::new_non_isotropic(child_genome, sigmas)
        } else {
            AdaptiveGenome::new_isotropic(child_genome, self.config.initial_sigma)
        };

        avg_sigma
    }

    /// Mutate an adaptive genome
    fn mutate<R: Rng>(&self, mut genome: AdaptiveGenome<G>, rng: &mut R) -> AdaptiveGenome<G> {
        let n = genome.inner().genes().len();

        // Self-adaptive: mutate strategy parameters first
        if self.config.self_adaptive {
            genome.strategy.mutate(n, rng);
        }

        // Collect sigmas first to avoid borrow conflicts
        let sigmas: Vec<f64> = (0..n).map(|i| genome.strategy.get_sigma(i)).collect();

        // Then mutate the genome using the (possibly updated) strategy
        let genes = genome.inner_mut().genes_mut();
        for i in 0..genes.len() {
            let perturbation: f64 = rng.sample(StandardNormal);
            genes[i] += sigmas[i] * perturbation;

            // Clamp to bounds
            if let Some(b) = self.bounds.get(i) {
                genes[i] = genes[i].clamp(b.min, b.max);
            }
        }

        genome
    }
}

/// Type alias for (μ+λ)-ES
pub type MuPlusLambdaES<G, F, Fit, Term> = EvolutionStrategy<G, F, Fit, Term>;

/// Type alias for (μ,λ)-ES
pub type MuCommaLambdaES<G, F, Fit, Term> = EvolutionStrategy<G, F, Fit, Term>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::genome::real_vector::RealVector;
    use crate::termination::MaxEvaluations;

    #[test]
    fn test_es_builder() {
        let bounds = MultiBounds::symmetric(5.0, 10);
        let es: Result<EvolutionStrategy<RealVector, f64, _, _>, _> = ESBuilder::new()
            .mu(15)
            .lambda(100)
            .bounds(bounds)
            .fitness(Sphere::new(10))
            .max_generations(10)
            .build();

        assert!(es.is_ok());
    }

    #[test]
    fn test_mu_plus_lambda_es() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::mu_plus_lambda(10, 70)
            .initial_sigma(1.0)
            .self_adaptive(true)
            .bounds(bounds)
            .fitness(Sphere::new(10))
            .termination(MaxEvaluations::new(3000))
            .build()
            .unwrap();

        let result = es.run(&mut rng).unwrap();

        // Should find improvement
        assert!(
            result.best_fitness > -50.0,
            "Expected fitness > -50, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_mu_comma_lambda_es() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::mu_comma_lambda(10, 70)
            .unwrap()
            .initial_sigma(1.0)
            .self_adaptive(true)
            .bounds(bounds)
            .fitness(Sphere::new(10))
            .termination(MaxEvaluations::new(3000))
            .build()
            .unwrap();

        let result = es.run(&mut rng).unwrap();

        // Should find improvement (may be less effective than μ+λ without elitism)
        assert!(
            result.best_fitness > -100.0,
            "Expected fitness > -100, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_mu_comma_lambda_constraint() {
        // λ < μ should fail
        let result = ESConfig::mu_comma_lambda(50, 30);
        assert!(result.is_err());
    }

    #[test]
    fn test_recombination_types() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        let recomb_types = vec![
            RecombinationType::None,
            RecombinationType::Discrete,
            RecombinationType::Intermediate,
            RecombinationType::GlobalIntermediate,
        ];

        for recomb in recomb_types {
            let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::new()
                .mu(10)
                .lambda(50)
                .recombination(recomb)
                .bounds(bounds.clone())
                .fitness(Sphere::new(5))
                .termination(MaxEvaluations::new(500))
                .build()
                .unwrap();

            let result = es.run(&mut rng);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_es_self_adaptive_disabled() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::new()
            .mu(10)
            .lambda(50)
            .self_adaptive(false)
            .initial_sigma(0.5)
            .bounds(bounds)
            .fitness(Sphere::new(5))
            .termination(MaxEvaluations::new(500))
            .build()
            .unwrap();

        let result = es.run(&mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_es_bounds_respected() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(2.0, 5);

        let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::new()
            .mu(10)
            .lambda(50)
            .initial_sigma(5.0) // Large sigma to test bounds
            .bounds(bounds.clone())
            .fitness(Sphere::new(5))
            .termination(MaxEvaluations::new(500))
            .build()
            .unwrap();

        let result = es.run(&mut rng).unwrap();

        // All genes should be within bounds
        for gene in result.best_genome.genes() {
            assert!(
                *gene >= -2.0 && *gene <= 2.0,
                "Gene {} outside bounds [-2.0, 2.0]",
                gene
            );
        }
    }
}
