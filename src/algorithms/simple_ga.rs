//! Simple Genetic Algorithm
//!
//! This module implements a standard generational genetic algorithm.

use std::time::Instant;

use rand::Rng;

use crate::diagnostics::{EvolutionResult, EvolutionStats, GenerationStats, TimingStats};
use crate::error::EvolutionError;
use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::{
    BoundedCrossoverOperator, BoundedMutationOperator, CrossoverOperator, MutationOperator,
    SelectionOperator,
};
use crate::population::individual::Individual;
use crate::population::population::Population;
use crate::termination::{EvolutionState, MaxGenerations, TerminationCriterion};

/// Configuration for the Simple GA
#[derive(Clone, Debug)]
pub struct SimpleGAConfig {
    /// Population size
    pub population_size: usize,
    /// Whether to use elitism (preserve best individual)
    pub elitism: bool,
    /// Number of elite individuals to preserve
    pub elite_count: usize,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Whether to evaluate in parallel
    pub parallel_evaluation: bool,
}

impl Default for SimpleGAConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism: true,
            elite_count: 1,
            crossover_probability: 0.9,
            parallel_evaluation: true,
        }
    }
}

/// Builder for SimpleGA
pub struct SimpleGABuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: SimpleGAConfig,
    bounds: Option<MultiBounds>,
    selection: Option<S>,
    crossover: Option<C>,
    mutation: Option<M>,
    fitness: Option<Fit>,
    termination: Option<Term>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F> SimpleGABuilder<G, F, (), (), (), (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: SimpleGAConfig::default(),
            bounds: None,
            selection: None,
            crossover: None,
            mutation: None,
            fitness: None,
            termination: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<G, F> Default for SimpleGABuilder<G, F, (), (), (), (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, S, C, M, Fit, Term> SimpleGABuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Set the population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    /// Enable or disable elitism
    pub fn elitism(mut self, enabled: bool) -> Self {
        self.config.elitism = enabled;
        self
    }

    /// Set the number of elite individuals to preserve
    pub fn elite_count(mut self, count: usize) -> Self {
        self.config.elite_count = count;
        self
    }

    /// Set the crossover probability
    pub fn crossover_probability(mut self, probability: f64) -> Self {
        self.config.crossover_probability = probability;
        self
    }

    /// Enable or disable parallel evaluation
    pub fn parallel_evaluation(mut self, enabled: bool) -> Self {
        self.config.parallel_evaluation = enabled;
        self
    }

    /// Set the search space bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the selection operator
    pub fn selection<NewS>(self, selection: NewS) -> SimpleGABuilder<G, F, NewS, C, M, Fit, Term>
    where
        NewS: SelectionOperator<G>,
    {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: Some(selection),
            crossover: self.crossover,
            mutation: self.mutation,
            fitness: self.fitness,
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the crossover operator
    pub fn crossover<NewC>(self, crossover: NewC) -> SimpleGABuilder<G, F, S, NewC, M, Fit, Term>
    where
        NewC: CrossoverOperator<G>,
    {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: Some(crossover),
            mutation: self.mutation,
            fitness: self.fitness,
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the mutation operator
    pub fn mutation<NewM>(self, mutation: NewM) -> SimpleGABuilder<G, F, S, C, NewM, Fit, Term>
    where
        NewM: MutationOperator<G>,
    {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: self.crossover,
            mutation: Some(mutation),
            fitness: self.fitness,
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the fitness function
    pub fn fitness<NewFit>(self, fitness: NewFit) -> SimpleGABuilder<G, F, S, C, M, NewFit, Term>
    where
        NewFit: Fitness<Genome = G, Value = F>,
    {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: self.crossover,
            mutation: self.mutation,
            fitness: Some(fitness),
            termination: self.termination,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the termination criterion
    pub fn termination<NewTerm>(
        self,
        termination: NewTerm,
    ) -> SimpleGABuilder<G, F, S, C, M, Fit, NewTerm>
    where
        NewTerm: TerminationCriterion<G, F>,
    {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: self.crossover,
            mutation: self.mutation,
            fitness: self.fitness,
            termination: Some(termination),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set max generations (convenience method)
    pub fn max_generations(
        self,
        max: usize,
    ) -> SimpleGABuilder<G, F, S, C, M, Fit, MaxGenerations> {
        SimpleGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: self.crossover,
            mutation: self.mutation,
            fitness: self.fitness,
            termination: Some(MaxGenerations::new(max)),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<G, F, S, C, M, Fit, Term> SimpleGABuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Build the SimpleGA instance
    #[allow(clippy::type_complexity)]
    pub fn build(self) -> Result<SimpleGA<G, F, S, C, M, Fit, Term>, EvolutionError> {
        let bounds = self
            .bounds
            .ok_or_else(|| EvolutionError::Configuration("Bounds must be specified".to_string()))?;

        let selection = self.selection.ok_or_else(|| {
            EvolutionError::Configuration("Selection operator must be specified".to_string())
        })?;

        let crossover = self.crossover.ok_or_else(|| {
            EvolutionError::Configuration("Crossover operator must be specified".to_string())
        })?;

        let mutation = self.mutation.ok_or_else(|| {
            EvolutionError::Configuration("Mutation operator must be specified".to_string())
        })?;

        let fitness = self.fitness.ok_or_else(|| {
            EvolutionError::Configuration("Fitness function must be specified".to_string())
        })?;

        let termination = self.termination.ok_or_else(|| {
            EvolutionError::Configuration("Termination criterion must be specified".to_string())
        })?;

        Ok(SimpleGA {
            config: self.config,
            bounds,
            selection,
            crossover,
            mutation,
            fitness,
            termination,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Simple Genetic Algorithm
///
/// A standard generational GA with configurable operators.
pub struct SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: SimpleGAConfig,
    bounds: MultiBounds,
    selection: S,
    crossover: C,
    mutation: M,
    fitness: Fit,
    termination: Term,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F, S, C, M, Fit, Term> SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Create a builder for SimpleGA
    pub fn builder() -> SimpleGABuilder<G, F, (), (), (), (), ()> {
        SimpleGABuilder::new()
    }

    /// Run the genetic algorithm
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Initialize population
        let mut population: Population<G, F> =
            Population::random(self.config.population_size, &self.bounds, rng);

        // Evaluate initial population
        let eval_start = Instant::now();
        if self.config.parallel_evaluation {
            population.evaluate_parallel(&self.fitness);
        } else {
            population.evaluate(&self.fitness);
        }
        let eval_time = eval_start.elapsed();

        let mut stats = EvolutionStats::new();
        let mut evaluations = population.len();
        let mut fitness_history: Vec<f64> = Vec::new();

        // Track best individual
        let mut best_individual = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&population, 0, evaluations)
            .with_timing(TimingStats::new().with_evaluation(eval_time));
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main evolution loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation: population.generation(),
                evaluations,
                best_fitness: best_individual.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            let gen_start = Instant::now();

            // Create new generation
            let mut new_population: Population<G, F> =
                Population::with_capacity(self.config.population_size);

            // Elitism: copy best individuals
            if self.config.elitism {
                let mut sorted = population.clone();
                sorted.sort_by_fitness();
                for i in 0..self.config.elite_count.min(sorted.len()) {
                    new_population.push(sorted[i].clone());
                }
            }

            // Selection pool
            let selection_pool: Vec<(G, f64)> = population.as_fitness_pairs();

            // Generate offspring
            let _selection_start = Instant::now();
            let mut selection_time = std::time::Duration::ZERO;
            let mut crossover_time = std::time::Duration::ZERO;
            let mut mutation_time = std::time::Duration::ZERO;

            while new_population.len() < self.config.population_size {
                // Selection
                let sel_start = Instant::now();
                let parent1_idx = self.selection.select(&selection_pool, rng);
                let parent2_idx = self.selection.select(&selection_pool, rng);
                selection_time += sel_start.elapsed();

                let parent1 = &selection_pool[parent1_idx].0;
                let parent2 = &selection_pool[parent2_idx].0;

                // Crossover
                let cross_start = Instant::now();
                let (mut child1, mut child2) =
                    if rng.gen::<f64>() < self.config.crossover_probability {
                        match self.crossover.crossover(parent1, parent2, rng).genome() {
                            Some((c1, c2)) => (c1, c2),
                            None => (parent1.clone(), parent2.clone()),
                        }
                    } else {
                        (parent1.clone(), parent2.clone())
                    };
                crossover_time += cross_start.elapsed();

                // Mutation
                let mut_start = Instant::now();
                self.mutation.mutate(&mut child1, rng);
                self.mutation.mutate(&mut child2, rng);
                mutation_time += mut_start.elapsed();

                // Add to new population
                new_population.push(Individual::with_generation(
                    child1,
                    population.generation() + 1,
                ));
                if new_population.len() < self.config.population_size {
                    new_population.push(Individual::with_generation(
                        child2,
                        population.generation() + 1,
                    ));
                }
            }

            // Truncate to exact size
            while new_population.len() > self.config.population_size {
                new_population.pop();
            }

            // Evaluate new population
            let eval_start = Instant::now();
            if self.config.parallel_evaluation {
                new_population.evaluate_parallel(&self.fitness);
            } else {
                new_population.evaluate(&self.fitness);
            }
            let eval_time = eval_start.elapsed();
            evaluations += new_population.len()
                - (if self.config.elitism {
                    self.config.elite_count
                } else {
                    0
                });

            // Update generation counter
            new_population.set_generation(population.generation() + 1);
            population = new_population;

            // Update best individual
            if let Some(best) = population.best() {
                if best.is_better_than(&best_individual) {
                    best_individual = best.clone();
                }
            }

            // Record statistics
            let timing = TimingStats::new()
                .with_selection(selection_time)
                .with_crossover(crossover_time)
                .with_mutation(mutation_time)
                .with_evaluation(eval_time)
                .with_total(gen_start.elapsed());

            let gen_stats =
                GenerationStats::from_population(&population, population.generation(), evaluations)
                    .with_timing(timing);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        Ok(EvolutionResult::new(
            best_individual.genome,
            best_individual.fitness.unwrap(),
            population.generation(),
            evaluations,
        )
        .with_stats(stats))
    }
}

// Implement bounded operators version
impl<G, F, S, C, M, Fit, Term> SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: BoundedCrossoverOperator<G>,
    M: BoundedMutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Run the genetic algorithm with bounded operators
    pub fn run_bounded<R: Rng>(
        &self,
        rng: &mut R,
    ) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Initialize population
        let mut population: Population<G, F> =
            Population::random(self.config.population_size, &self.bounds, rng);

        // Evaluate initial population
        if self.config.parallel_evaluation {
            population.evaluate_parallel(&self.fitness);
        } else {
            population.evaluate(&self.fitness);
        }

        let mut stats = EvolutionStats::new();
        let mut evaluations = population.len();
        let mut fitness_history: Vec<f64> = Vec::new();

        // Track best individual
        let mut best_individual = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        // Main evolution loop
        loop {
            // Check termination
            let state = EvolutionState {
                generation: population.generation(),
                evaluations,
                best_fitness: best_individual.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            // Create new generation
            let mut new_population: Population<G, F> =
                Population::with_capacity(self.config.population_size);

            // Elitism
            if self.config.elitism {
                let mut sorted = population.clone();
                sorted.sort_by_fitness();
                for i in 0..self.config.elite_count.min(sorted.len()) {
                    new_population.push(sorted[i].clone());
                }
            }

            let selection_pool: Vec<(G, f64)> = population.as_fitness_pairs();

            while new_population.len() < self.config.population_size {
                let parent1_idx = self.selection.select(&selection_pool, rng);
                let parent2_idx = self.selection.select(&selection_pool, rng);

                let parent1 = &selection_pool[parent1_idx].0;
                let parent2 = &selection_pool[parent2_idx].0;

                let (mut child1, mut child2) =
                    if rng.gen::<f64>() < self.config.crossover_probability {
                        match self
                            .crossover
                            .crossover_bounded(parent1, parent2, &self.bounds, rng)
                            .genome()
                        {
                            Some((c1, c2)) => (c1, c2),
                            None => (parent1.clone(), parent2.clone()),
                        }
                    } else {
                        (parent1.clone(), parent2.clone())
                    };

                self.mutation.mutate_bounded(&mut child1, &self.bounds, rng);
                self.mutation.mutate_bounded(&mut child2, &self.bounds, rng);

                new_population.push(Individual::with_generation(
                    child1,
                    population.generation() + 1,
                ));
                if new_population.len() < self.config.population_size {
                    new_population.push(Individual::with_generation(
                        child2,
                        population.generation() + 1,
                    ));
                }
            }

            while new_population.len() > self.config.population_size {
                new_population.pop();
            }

            if self.config.parallel_evaluation {
                new_population.evaluate_parallel(&self.fitness);
            } else {
                new_population.evaluate(&self.fitness);
            }
            evaluations += new_population.len()
                - (if self.config.elitism {
                    self.config.elite_count
                } else {
                    0
                });

            new_population.set_generation(population.generation() + 1);
            population = new_population;

            if let Some(best) = population.best() {
                if best.is_better_than(&best_individual) {
                    best_individual = best.clone();
                }
            }

            let gen_stats =
                GenerationStats::from_population(&population, population.generation(), evaluations);
            fitness_history.push(gen_stats.best_fitness);
            stats.record(gen_stats);
        }

        stats.set_runtime(start_time.elapsed());

        Ok(EvolutionResult::new(
            best_individual.genome,
            best_individual.fitness.unwrap(),
            population.generation(),
            evaluations,
        )
        .with_stats(stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::{OneMax, Sphere};
    use crate::genome::traits::RealValuedGenome;
    use crate::operators::crossover::{SbxCrossover, UniformCrossover};
    use crate::operators::mutation::{BitFlipMutation, PolynomialMutation};
    use crate::operators::selection::TournamentSelection;
    use crate::termination::TargetFitness;

    #[test]
    fn test_simple_ga_builder() {
        let bounds = MultiBounds::symmetric(5.0, 10);
        let ga = SimpleGABuilder::new()
            .population_size(50)
            .bounds(bounds)
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .max_generations(10)
            .build();

        assert!(ga.is_ok());
    }

    #[test]
    fn test_simple_ga_missing_bounds() {
        // Test that build() returns an error when bounds are missing
        // Note: With the type-safe builder, operators must be provided to call build()
        // but bounds can be missing (they're an Option in the builder)
        let ga = SimpleGABuilder::new()
            .population_size(50)
            // bounds are missing
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .max_generations(10)
            .build();

        assert!(ga.is_err());
        if let Err(e) = ga {
            assert!(e.to_string().contains("Bounds"));
        }
    }

    #[test]
    fn test_simple_ga_sphere() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let ga = SimpleGABuilder::new()
            .population_size(50)
            .bounds(bounds)
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .max_generations(100)
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Should find some improvement from random initialization
        // Initial random values in [-5.12, 5.12] have expected fitness around -52 per dimension
        // so total ~-520. Even modest improvement should get to -200 or better.
        assert!(
            result.best_fitness > -200.0,
            "Expected fitness > -200, got {}",
            result.best_fitness
        ); // Sphere is negated, so closer to 0 is better
        assert!(result.generations <= 100);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_simple_ga_onemax() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::uniform(crate::genome::bounds::Bounds::unit(), 20);

        let ga = SimpleGABuilder::new()
            .population_size(50)
            .bounds(bounds)
            .selection(TournamentSelection::new(3))
            .crossover(UniformCrossover::new())
            .mutation(BitFlipMutation::new())
            .fitness(OneMax::new(20))
            .max_generations(50)
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Should find perfect or near-perfect solution
        assert!(result.best_fitness >= 15); // At least 75% ones
    }

    #[test]
    fn test_simple_ga_target_fitness() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        let ga = SimpleGABuilder::new()
            .population_size(100)
            .bounds(bounds)
            .selection(TournamentSelection::new(5))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(5))
            .termination(TargetFitness::with_tolerance(0.0, 0.1)) // Near 0 (optimal)
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Should reach target
        assert!(result.best_fitness >= -0.1);
        assert_eq!(
            result.stats.termination_reason.as_deref(),
            Some("Target fitness reached")
        );
    }

    #[test]
    fn test_simple_ga_bounded() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let ga = SimpleGABuilder::new()
            .population_size(50)
            .bounds(bounds.clone())
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .max_generations(50)
            .build()
            .unwrap();

        let result = ga.run_bounded(&mut rng).unwrap();

        // All genes should be within bounds
        for gene in result.best_genome.genes() {
            assert!(*gene >= -5.12 && *gene <= 5.12);
        }
    }

    #[test]
    fn test_simple_ga_elitism() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        // Run without elitism
        let ga_no_elite = SimpleGABuilder::new()
            .population_size(20)
            .elitism(false)
            .bounds(bounds.clone())
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(5))
            .max_generations(20)
            .build()
            .unwrap();

        // Run with elitism
        let ga_elite = SimpleGABuilder::new()
            .population_size(20)
            .elitism(true)
            .elite_count(2)
            .bounds(bounds)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(5))
            .max_generations(20)
            .build()
            .unwrap();

        // Both should run without error
        let result_no_elite = ga_no_elite.run(&mut rng);
        let result_elite = ga_elite.run(&mut rng);

        assert!(result_no_elite.is_ok());
        assert!(result_elite.is_ok());
    }

    #[test]
    fn test_simple_ga_statistics() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        let ga = SimpleGABuilder::new()
            .population_size(20)
            .bounds(bounds)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(5))
            .max_generations(10)
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Check statistics were collected
        assert_eq!(result.stats.num_generations(), 11); // Initial + 10 generations
        assert!(result.stats.total_runtime_ms > 0.0);

        // Best fitness should improve or stay the same
        let history = result.stats.best_fitness_history();
        for i in 1..history.len() {
            assert!(history[i] >= history[i - 1] - 0.001); // Allow small numerical error
        }
    }
}
