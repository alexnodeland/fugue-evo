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

/// Replacement strategy for steady-state GA.
///
/// EV-42: only [`ReplacementStrategy::ReplaceIfBetter`] is elitist (it accepts an
/// offspring only when it improves on the individual it would replace). Every
/// other strategy performs its replacement **unconditionally**, matching its
/// documented meaning — in particular `ReplaceRandom` is a genuine, diversity-
/// preserving stochastic replacement and may accept a worse offspring.
#[derive(Clone, Debug, Default)]
pub enum ReplacementStrategy {
    /// Replace the worst individual in the population, unconditionally. Because the
    /// current best is never the replacement target, the population best is
    /// preserved, but the population mean may temporarily worsen.
    #[default]
    ReplaceWorst,
    /// Replace a uniformly-random individual, unconditionally (non-elitist). A
    /// worse offspring can enter the population, preserving exploratory diversity.
    ReplaceRandom,
    /// Replace the worse parent, but only if the offspring beats it
    /// (generational replacement).
    ReplaceParent,
    /// Replace the worst individual, but only if the offspring is strictly better
    /// than it (the fully elitist, accept-if-better option).
    ReplaceIfBetter,
    /// Use an inverse tournament to choose a (likely poor) individual to replace,
    /// then replace it unconditionally.
    TournamentWorst(usize),
}

impl ReplacementStrategy {
    /// Whether replacement is gated on the offspring improving on its target
    /// (EV-42). Only `ReplaceIfBetter` is elitist here; `ReplaceParent` carries
    /// its own acceptance test in the main loop.
    fn requires_improvement(&self) -> bool {
        matches!(self, ReplacementStrategy::ReplaceIfBetter)
    }
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
    /// EV-44: when true, an offspring whose genome equals an existing population
    /// member is rejected instead of inserted, preventing the population from
    /// collapsing to duplicates. This costs an O(population_size) genome
    /// comparison per candidate offspring.
    pub prevent_duplicates: bool,
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
            prevent_duplicates: false,
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

    /// Enable or disable duplicate avoidance (EV-44).
    ///
    /// When enabled, an offspring whose genome equals any current population
    /// member is rejected rather than inserted. This preserves diversity but adds
    /// an O(population_size) genome comparison per candidate offspring, and
    /// requires the genome type to implement [`PartialEq`].
    pub fn prevent_duplicates(mut self, enabled: bool) -> Self {
        self.config.prevent_duplicates = enabled;
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

    /// Find the index of the individual to replace for the configured strategy.
    fn find_replacement_index<R: Rng>(&self, population: &Population<G, F>, rng: &mut R) -> usize {
        match &self.config.replacement {
            // Both worst-targeting strategies aim at the least-fit individual; the
            // accept-if-better gate for `ReplaceIfBetter` is applied by the caller.
            ReplacementStrategy::ReplaceWorst | ReplacementStrategy::ReplaceIfBetter => {
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

    /// Generate exactly `offspring_count` evaluated children (EV-43).
    ///
    /// Each returned tuple is `(child, (parent1_idx, parent1_fitness),
    /// (parent2_idx, parent2_fitness))`; the parent data drives `ReplaceParent`.
    /// Fresh parents are selected per crossover pair, and only children that will
    /// actually be considered are mutated and evaluated.
    #[allow(clippy::type_complexity)]
    fn generate_offspring<R: Rng>(
        &self,
        selection_pool: &[(G, f64)],
        rng: &mut R,
    ) -> Vec<(Individual<G, F>, (usize, f64), (usize, f64))> {
        let target = self.config.offspring_count;
        let mut offspring = Vec::with_capacity(target);

        while offspring.len() < target {
            let p1_idx = self.selection.select(selection_pool, rng);
            let p2_idx = self.selection.select(selection_pool, rng);
            let p1 = (p1_idx, selection_pool[p1_idx].1);
            let p2 = (p2_idx, selection_pool[p2_idx].1);
            let parent1 = &selection_pool[p1_idx].0;
            let parent2 = &selection_pool[p2_idx].0;

            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_probability {
                match self.crossover.crossover(parent1, parent2, rng).genome() {
                    Some((c1, c2)) => (c1, c2),
                    None => (parent1.clone(), parent2.clone()),
                }
            } else {
                (parent1.clone(), parent2.clone())
            };

            self.mutation.mutate(&mut child1, rng);
            let mut ind1 = Individual::new(child1);
            ind1.set_fitness(self.fitness.evaluate(ind1.genome()));
            offspring.push((ind1, p1, p2));

            if offspring.len() < target {
                self.mutation.mutate(&mut child2, rng);
                let mut ind2 = Individual::new(child2);
                ind2.set_fitness(self.fitness.evaluate(ind2.genome()));
                offspring.push((ind2, p1, p2));
            }
        }

        offspring
    }

    /// Insert an already-evaluated `child` into `population` per the configured
    /// replacement strategy (EV-42) and duplicate policy (EV-44).
    ///
    /// The population may be left unchanged: a duplicate is rejected when
    /// `prevent_duplicates` is set, `ReplaceParent` skips when the child does not
    /// beat its worse parent, and `ReplaceIfBetter` skips when the child does not
    /// beat the individual it targets. All other strategies replace their chosen
    /// index unconditionally.
    fn place_offspring<R: Rng>(
        &self,
        population: &mut Population<G, F>,
        child: Individual<G, F>,
        parent1: (usize, f64),
        parent2: (usize, f64),
        rng: &mut R,
    ) where
        G: PartialEq,
    {
        // EV-44: reject an exact duplicate of an existing genome if configured
        // (O(population_size) genome comparison).
        if self.config.prevent_duplicates
            && population.iter().any(|ind| ind.genome() == child.genome())
        {
            return;
        }

        let replace_idx = match &self.config.replacement {
            ReplacementStrategy::ReplaceParent => {
                let (p1_idx, p1_fit) = parent1;
                let (p2_idx, p2_fit) = parent2;
                let child_fit = child.fitness_value().to_f64();
                if child_fit > p1_fit.min(p2_fit) {
                    if p1_fit < p2_fit {
                        p1_idx
                    } else {
                        p2_idx
                    }
                } else {
                    return; // offspring did not beat its worse parent
                }
            }
            _ => self.find_replacement_index(population, rng),
        };

        // EV-42: only the elitist ReplaceIfBetter gates on improvement. Every
        // other strategy (ReplaceRandom, ReplaceWorst, TournamentWorst) replaces
        // its chosen index unconditionally, as documented.
        if self.config.replacement.requires_improvement()
            && !child
                .fitness_value()
                .is_better_than(population[replace_idx].fitness_value())
        {
            return;
        }

        population[replace_idx] = child;
    }

    /// Run the steady-state genetic algorithm.
    ///
    /// Requires `G: PartialEq` so the optional duplicate-avoidance policy
    /// (`prevent_duplicates`) can compare genomes; all built-in genome types
    /// satisfy this.
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError>
    where
        G: PartialEq,
    {
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

            // EV-43: generate and evaluate EXACTLY offspring_count children — no
            // more, no fewer — so wasted evaluations don't inflate the counter or
            // trip MaxEvaluations early, and offspring_count > 2 is honored.
            let offspring = self.generate_offspring(&selection_pool, rng);
            evaluations += offspring.len();

            // Replace individuals in the population.
            for (child, parent1, parent2) in offspring {
                self.place_offspring(&mut population, child, parent1, parent2, rng);
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
    /// Generate exactly `offspring_count` evaluated children with bounded
    /// operators (EV-43). See [`SteadyStateGA::generate_offspring`].
    #[allow(clippy::type_complexity)]
    fn generate_offspring_bounded<R: Rng>(
        &self,
        selection_pool: &[(G, f64)],
        rng: &mut R,
    ) -> Vec<(Individual<G, F>, (usize, f64), (usize, f64))> {
        let target = self.config.offspring_count;
        let mut offspring = Vec::with_capacity(target);

        while offspring.len() < target {
            let p1_idx = self.selection.select(selection_pool, rng);
            let p2_idx = self.selection.select(selection_pool, rng);
            let p1 = (p1_idx, selection_pool[p1_idx].1);
            let p2 = (p2_idx, selection_pool[p2_idx].1);
            let parent1 = &selection_pool[p1_idx].0;
            let parent2 = &selection_pool[p2_idx].0;

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

            self.mutation.mutate_bounded(&mut child1, &self.bounds, rng);
            let mut ind1 = Individual::new(child1);
            ind1.set_fitness(self.fitness.evaluate(ind1.genome()));
            offspring.push((ind1, p1, p2));

            if offspring.len() < target {
                self.mutation.mutate_bounded(&mut child2, &self.bounds, rng);
                let mut ind2 = Individual::new(child2);
                ind2.set_fitness(self.fitness.evaluate(ind2.genome()));
                offspring.push((ind2, p1, p2));
            }
        }

        offspring
    }

    /// Run the steady-state genetic algorithm with bounded operators.
    ///
    /// Requires `G: PartialEq` for the optional `prevent_duplicates` policy.
    pub fn run_bounded<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError>
    where
        G: PartialEq,
    {
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

            // EV-43: generate and evaluate EXACTLY offspring_count children with
            // bounded operators.
            let offspring = self.generate_offspring_bounded(&selection_pool, rng);
            evaluations += offspring.len();

            // Replace individuals in the population.
            for (child, parent1, parent2) in offspring {
                self.place_offspring(&mut population, child, parent1, parent2, rng);
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
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use crate::operators::crossover::{SbxCrossover, UniformCrossover};
    use crate::operators::mutation::{BitFlipMutation, PolynomialMutation};
    use crate::operators::selection::TournamentSelection;
    use crate::termination::MaxEvaluations;
    use rand::SeedableRng;

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

    // regression: EV-43 — with offspring_count = 1, exactly one child is evaluated
    // per step. Starting from a pop of 10 (10 initial evals) under a budget of 15,
    // the fixed code stops at exactly 15; the pre-fix code evaluated 2 per step and
    // overshot to 16.
    #[test]
    fn test_offspring_count_one_evaluates_one_per_step() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let bounds = MultiBounds::symmetric(5.12, 5);
        let ga = SteadyStateBuilder::new()
            .population_size(10)
            .offspring_count(1)
            .steps_per_generation(1)
            .bounds(bounds)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(5))
            .termination(MaxEvaluations::new(15))
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();
        assert_eq!(result.evaluations, 15);
    }

    // regression: EV-42 — ReplaceRandom replaces unconditionally, so a worse
    // offspring can enter and the population's WORST fitness can degrade across
    // steps. The pre-fix accept-if-better guard kept the worst monotone, so no
    // degradation could ever appear.
    #[test]
    fn test_replace_random_accepts_worse_offspring() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(20);
        let bounds = MultiBounds::symmetric(5.12, 5);
        let ga = SteadyStateBuilder::new()
            .population_size(20)
            .offspring_count(1)
            .steps_per_generation(1)
            .replacement(ReplacementStrategy::ReplaceRandom)
            .bounds(bounds)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(2.0))
            .mutation(PolynomialMutation::new(2.0))
            .fitness(Sphere::new(5))
            .termination(MaxEvaluations::new(400))
            .build()
            .unwrap();

        let result = ga.run(&mut rng).unwrap();
        let worst: Vec<f64> = result
            .stats
            .generations
            .iter()
            .map(|g| g.worst_fitness)
            .collect();
        let degraded = worst.windows(2).any(|w| w[1] < w[0] - 1e-12);
        assert!(
            degraded,
            "ReplaceRandom must let the population worst degrade at least once"
        );
    }

    // regression: EV-44 — with prevent_duplicates, an offspring whose genome equals
    // an existing member is rejected instead of inserted, even if it is fitter.
    #[test]
    fn test_prevent_duplicates_rejects_existing_genome() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let bounds = MultiBounds::symmetric(5.0, 3);
        let ga = SteadyStateBuilder::new()
            .population_size(5)
            .offspring_count(1)
            .replacement(ReplacementStrategy::ReplaceRandom)
            .prevent_duplicates(true)
            .bounds(bounds)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(3))
            .max_generations(1)
            .build()
            .unwrap();

        // A population of distinct genomes.
        let mut pop: Population<RealVector, f64> = Population::new();
        for i in 0..5 {
            let mut ind = Individual::new(RealVector::new(vec![i as f64, 0.0, 0.0]));
            ind.set_fitness(-(i as f64));
            pop.push(ind);
        }

        // A child equal to an existing member is rejected (population unchanged),
        // even though its fitness is better.
        let mut dup = Individual::new(RealVector::new(vec![2.0, 0.0, 0.0]));
        dup.set_fitness(100.0);
        let before: Vec<RealVector> = pop.iter().map(|i| i.genome().clone()).collect();
        ga.place_offspring(&mut pop, dup, (0, 0.0), (0, 0.0), &mut rng);
        let after: Vec<RealVector> = pop.iter().map(|i| i.genome().clone()).collect();
        assert_eq!(before, after, "duplicate offspring must be rejected");

        // A novel genome is accepted (ReplaceRandom, unconditional).
        let mut novel = Individual::new(RealVector::new(vec![42.0, 0.0, 0.0]));
        novel.set_fitness(-999.0);
        let novel_genome = novel.genome().clone();
        ga.place_offspring(&mut pop, novel, (0, 0.0), (0, 0.0), &mut rng);
        assert!(
            pop.iter().any(|i| *i.genome() == novel_genome),
            "a novel genome should be inserted"
        );
    }
}
