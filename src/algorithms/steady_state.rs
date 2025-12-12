//! Steady-State Genetic Algorithm
//!
//! This module implements a steady-state genetic algorithm where only a small
//! number of individuals are replaced in each generation (typically 1 or 2),
//! rather than replacing the entire population.

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

/// Replacement strategy for steady-state GA
#[derive(Clone, Debug, Default)]
pub enum ReplacementStrategy {
    /// Replace the worst individual(s) in the population
    #[default]
    ReplaceWorst,
    /// Replace a randomly selected individual
    ReplaceRandom,
    /// Replace the parent if offspring is better (generational replacement)
    ReplaceParent,
    /// Use tournament selection to choose individual to replace (inverse tournament)
    TournamentWorst(usize),
}

/// Configuration for the Steady-State GA
#[derive(Clone, Debug)]
pub struct SteadyStateConfig {
    /// Population size
    pub population_size: usize,
    /// Number of offspring to generate per step (typically 1 or 2)
    pub offspring_count: usize,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Replacement strategy
    pub replacement: ReplacementStrategy,
    /// Whether to evaluate in parallel
    pub parallel_evaluation: bool,
    /// Number of steps per "generation" for statistics reporting
    pub steps_per_generation: usize,
}

impl Default for SteadyStateConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            offspring_count: 2,
            crossover_probability: 0.9,
            replacement: ReplacementStrategy::ReplaceWorst,
            parallel_evaluation: false,
            steps_per_generation: 50, // Report stats every 50 replacements
        }
    }
}

/// Builder for SteadyStateGA
pub struct SteadyStateBuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: SteadyStateConfig,
    bounds: Option<MultiBounds>,
    selection: Option<S>,
    crossover: Option<C>,
    mutation: Option<M>,
    fitness: Option<Fit>,
    termination: Option<Term>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F> SteadyStateBuilder<G, F, (), (), (), (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: SteadyStateConfig::default(),
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

impl<G, F> Default for SteadyStateBuilder<G, F, (), (), (), (), ()>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F, S, C, M, Fit, Term> SteadyStateBuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Set the population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    /// Set the number of offspring per step
    pub fn offspring_count(mut self, count: usize) -> Self {
        self.config.offspring_count = count;
        self
    }

    /// Set the crossover probability
    pub fn crossover_probability(mut self, probability: f64) -> Self {
        self.config.crossover_probability = probability;
        self
    }

    /// Set the replacement strategy
    pub fn replacement(mut self, strategy: ReplacementStrategy) -> Self {
        self.config.replacement = strategy;
        self
    }

    /// Enable or disable parallel evaluation
    pub fn parallel_evaluation(mut self, enabled: bool) -> Self {
        self.config.parallel_evaluation = enabled;
        self
    }

    /// Set the number of steps per generation (for statistics)
    pub fn steps_per_generation(mut self, steps: usize) -> Self {
        self.config.steps_per_generation = steps;
        self
    }

    /// Set the search space bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the selection operator
    pub fn selection<NewS>(self, selection: NewS) -> SteadyStateBuilder<G, F, NewS, C, M, Fit, Term>
    where
        NewS: SelectionOperator<G>,
    {
        SteadyStateBuilder {
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
    pub fn crossover<NewC>(self, crossover: NewC) -> SteadyStateBuilder<G, F, S, NewC, M, Fit, Term>
    where
        NewC: CrossoverOperator<G>,
    {
        SteadyStateBuilder {
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
    pub fn mutation<NewM>(self, mutation: NewM) -> SteadyStateBuilder<G, F, S, C, NewM, Fit, Term>
    where
        NewM: MutationOperator<G>,
    {
        SteadyStateBuilder {
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
    pub fn fitness<NewFit>(self, fitness: NewFit) -> SteadyStateBuilder<G, F, S, C, M, NewFit, Term>
    where
        NewFit: Fitness<Genome = G, Value = F>,
    {
        SteadyStateBuilder {
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
    ) -> SteadyStateBuilder<G, F, S, C, M, Fit, NewTerm>
    where
        NewTerm: TerminationCriterion<G, F>,
    {
        SteadyStateBuilder {
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
    ) -> SteadyStateBuilder<G, F, S, C, M, Fit, MaxGenerations> {
        SteadyStateBuilder {
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

impl<G, F, S, C, M, Fit, Term> SteadyStateBuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Build the SteadyStateGA instance
    #[allow(clippy::type_complexity)]
    pub fn build(self) -> Result<SteadyStateGA<G, F, S, C, M, Fit, Term>, EvolutionError> {
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

        Ok(SteadyStateGA {
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

/// Steady-State Genetic Algorithm
///
/// A GA variant that replaces only a few individuals per generation step,
/// maintaining a more continuous population transition.
pub struct SteadyStateGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: SteadyStateConfig,
    bounds: MultiBounds,
    selection: S,
    crossover: C,
    mutation: M,
    fitness: Fit,
    termination: Term,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, F, S, C, M, Fit, Term> SteadyStateGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Create a builder for SteadyStateGA
    pub fn builder() -> SteadyStateBuilder<G, F, (), (), (), (), ()> {
        SteadyStateBuilder::new()
    }

    /// Find the index of the worst individual to replace
    fn find_replacement_index<R: Rng>(&self, population: &Population<G, F>, rng: &mut R) -> usize {
        match &self.config.replacement {
            ReplacementStrategy::ReplaceWorst => {
                // Find index of worst fitness
                population
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.fitness_value()
                            .partial_cmp(b.fitness_value())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            ReplacementStrategy::ReplaceRandom => rng.gen_range(0..population.len()),
            ReplacementStrategy::ReplaceParent => {
                // This is handled specially in the main loop
                0
            }
            ReplacementStrategy::TournamentWorst(tournament_size) => {
                // Inverse tournament: select worst from tournament
                let mut worst_idx = rng.gen_range(0..population.len());
                let mut worst_fitness = population[worst_idx].fitness_value().to_f64();

                for _ in 1..*tournament_size {
                    let idx = rng.gen_range(0..population.len());
                    let fitness = population[idx].fitness_value().to_f64();
                    if fitness < worst_fitness {
                        worst_idx = idx;
                        worst_fitness = fitness;
                    }
                }
                worst_idx
            }
        }
    }

    /// Run the steady-state genetic algorithm
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError> {
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
        let mut step_count = 0usize;
        let mut generation = 0usize;

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
                generation,
                evaluations,
                best_fitness: best_individual.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            let step_start = Instant::now();

            // Selection pool
            let selection_pool: Vec<(G, f64)> = population.as_fitness_pairs();

            // Select parents
            let parent1_idx = self.selection.select(&selection_pool, rng);
            let parent2_idx = self.selection.select(&selection_pool, rng);
            let parent1 = &selection_pool[parent1_idx].0;
            let parent2 = &selection_pool[parent2_idx].0;

            // Crossover
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_probability {
                match self.crossover.crossover(parent1, parent2, rng).genome() {
                    Some((c1, c2)) => (c1, c2),
                    None => (parent1.clone(), parent2.clone()),
                }
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Mutation
            self.mutation.mutate(&mut child1, rng);
            self.mutation.mutate(&mut child2, rng);

            // Evaluate offspring
            let mut offspring: Vec<Individual<G, F>> =
                vec![Individual::new(child1), Individual::new(child2)];

            for ind in &mut offspring {
                let fitness = self.fitness.evaluate(&ind.genome);
                ind.set_fitness(fitness);
            }
            evaluations += offspring.len();

            // Replace individuals in population
            for child in offspring.into_iter().take(self.config.offspring_count) {
                let replace_idx = match &self.config.replacement {
                    ReplacementStrategy::ReplaceParent => {
                        // Only replace if child is better than worst parent
                        let p1_fit = selection_pool[parent1_idx].1;
                        let p2_fit = selection_pool[parent2_idx].1;
                        let child_fit = child.fitness_value().to_f64();

                        if child_fit > p1_fit.min(p2_fit) {
                            if p1_fit < p2_fit {
                                parent1_idx
                            } else {
                                parent2_idx
                            }
                        } else {
                            continue; // Don't replace
                        }
                    }
                    _ => self.find_replacement_index(&population, rng),
                };

                // Only replace if new individual is better than the one being replaced
                // (optional: could be configurable)
                if child
                    .fitness_value()
                    .is_better_than(population[replace_idx].fitness_value())
                {
                    population[replace_idx] = child;
                }
            }

            // Update best individual
            if let Some(best) = population.best() {
                if best.is_better_than(&best_individual) {
                    best_individual = best.clone();
                }
            }

            step_count += 1;

            // Record statistics at generation boundaries
            if step_count.is_multiple_of(self.config.steps_per_generation) {
                generation += 1;
                population.set_generation(generation);

                let timing = TimingStats::new()
                    .with_total(step_start.elapsed() * self.config.steps_per_generation as u32);

                let gen_stats =
                    GenerationStats::from_population(&population, generation, evaluations)
                        .with_timing(timing);
                fitness_history.push(gen_stats.best_fitness);
                stats.record(gen_stats);
            }
        }

        stats.set_runtime(start_time.elapsed());

        Ok(EvolutionResult::new(
            best_individual.genome,
            best_individual.fitness.unwrap(),
            generation,
            evaluations,
        )
        .with_stats(stats))
    }
}

// Implement bounded operators version
impl<G, F, S, C, M, Fit, Term> SteadyStateGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
    S: SelectionOperator<G>,
    C: BoundedCrossoverOperator<G>,
    M: BoundedMutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F> + Sync,
    Term: TerminationCriterion<G, F>,
{
    /// Run the steady-state genetic algorithm with bounded operators
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
        let mut step_count = 0usize;
        let mut generation = 0usize;

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
                generation,
                evaluations,
                best_fitness: best_individual.fitness_value().to_f64(),
                population: &population,
                fitness_history: &fitness_history,
            };

            if self.termination.should_terminate(&state) {
                stats.set_termination_reason(self.termination.reason());
                break;
            }

            // Selection pool
            let selection_pool: Vec<(G, f64)> = population.as_fitness_pairs();

            // Select parents
            let parent1_idx = self.selection.select(&selection_pool, rng);
            let parent2_idx = self.selection.select(&selection_pool, rng);
            let parent1 = &selection_pool[parent1_idx].0;
            let parent2 = &selection_pool[parent2_idx].0;

            // Crossover with bounds
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_probability {
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

            // Mutation with bounds
            self.mutation.mutate_bounded(&mut child1, &self.bounds, rng);
            self.mutation.mutate_bounded(&mut child2, &self.bounds, rng);

            // Evaluate offspring
            let mut offspring: Vec<Individual<G, F>> =
                vec![Individual::new(child1), Individual::new(child2)];

            for ind in &mut offspring {
                let fitness = self.fitness.evaluate(&ind.genome);
                ind.set_fitness(fitness);
            }
            evaluations += offspring.len();

            // Replace individuals in population
            for child in offspring.into_iter().take(self.config.offspring_count) {
                let replace_idx = match &self.config.replacement {
                    ReplacementStrategy::ReplaceParent => {
                        let p1_fit = selection_pool[parent1_idx].1;
                        let p2_fit = selection_pool[parent2_idx].1;
                        let child_fit = child.fitness_value().to_f64();

                        if child_fit > p1_fit.min(p2_fit) {
                            if p1_fit < p2_fit {
                                parent1_idx
                            } else {
                                parent2_idx
                            }
                        } else {
                            continue;
                        }
                    }
                    _ => self.find_replacement_index(&population, rng),
                };

                if child
                    .fitness_value()
                    .is_better_than(population[replace_idx].fitness_value())
                {
                    population[replace_idx] = child;
                }
            }

            // Update best individual
            if let Some(best) = population.best() {
                if best.is_better_than(&best_individual) {
                    best_individual = best.clone();
                }
            }

            step_count += 1;

            // Record statistics at generation boundaries
            if step_count.is_multiple_of(self.config.steps_per_generation) {
                generation += 1;
                population.set_generation(generation);

                let gen_stats =
                    GenerationStats::from_population(&population, generation, evaluations);
                fitness_history.push(gen_stats.best_fitness);
                stats.record(gen_stats);
            }
        }

        stats.set_runtime(start_time.elapsed());

        Ok(EvolutionResult::new(
            best_individual.genome,
            best_individual.fitness.unwrap(),
            generation,
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
    use crate::termination::MaxEvaluations;

    #[test]
    fn test_steady_state_builder() {
        let bounds = MultiBounds::symmetric(5.0, 10);
        let ga = SteadyStateBuilder::new()
            .population_size(50)
            .offspring_count(2)
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
    fn test_steady_state_sphere() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let ga = SteadyStateBuilder::new()
            .population_size(50)
            .offspring_count(2)
            .steps_per_generation(25)
            .bounds(bounds)
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .termination(MaxEvaluations::new(2000))
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Should find some improvement
        assert!(
            result.best_fitness > -200.0,
            "Expected fitness > -200, got {}",
            result.best_fitness
        );
        assert!(result.evaluations <= 2000);
    }

    #[test]
    fn test_steady_state_onemax() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::uniform(crate::genome::bounds::Bounds::unit(), 20);

        let ga = SteadyStateBuilder::new()
            .population_size(50)
            .offspring_count(2)
            .steps_per_generation(25)
            .bounds(bounds)
            .selection(TournamentSelection::new(3))
            .crossover(UniformCrossover::new())
            .mutation(BitFlipMutation::new())
            .fitness(OneMax::new(20))
            .termination(MaxEvaluations::new(2000))
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();

        // Should find good solution
        assert!(result.best_fitness >= 15); // At least 75% ones
    }

    #[test]
    fn test_replacement_strategies() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 5);

        let strategies = vec![
            ReplacementStrategy::ReplaceWorst,
            ReplacementStrategy::ReplaceRandom,
            ReplacementStrategy::TournamentWorst(3),
        ];

        for strategy in strategies {
            let ga = SteadyStateBuilder::new()
                .population_size(30)
                .offspring_count(2)
                .replacement(strategy)
                .bounds(bounds.clone())
                .selection(TournamentSelection::new(3))
                .crossover(SbxCrossover::new(20.0))
                .mutation(PolynomialMutation::new(20.0))
                .fitness(Sphere::new(5))
                .termination(MaxEvaluations::new(500))
                .build()
                .unwrap();

            let result = ga.run(&mut rng);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_steady_state_bounded() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.12, 10);

        let ga = SteadyStateBuilder::new()
            .population_size(50)
            .bounds(bounds.clone())
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(10))
            .termination(MaxEvaluations::new(1000))
            .build()
            .unwrap();

        let result = ga.run_bounded(&mut rng).unwrap();

        // All genes should be within bounds
        for gene in result.best_genome.genes() {
            assert!(*gene >= -5.12 && *gene <= 5.12);
        }
    }
}
