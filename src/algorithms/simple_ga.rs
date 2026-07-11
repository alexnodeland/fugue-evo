//! Simple Genetic Algorithm
//!
//! This module implements a standard generational genetic algorithm.

use std::time::Instant;

use rand::Rng;

use crate::diagnostics::{EvolutionResult, EvolutionStats, GenerationStats, TimingStats};
use crate::error::EvolutionError;
use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::permutation::Permutation;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::EvolutionaryGenome;
use crate::hyperparameter::bayesian::{
    ThompsonConfig, ThompsonSamplingTuner, TunableMutation, PARAM_CROSSOVER_PROB,
    PARAM_MUTATION_RATE,
};
use crate::operators::crossover::{OxCrossover, SbxCrossover, UniformCrossover};
use crate::operators::mutation::{BitFlipMutation, PermutationSwapMutation, PolynomialMutation};
use crate::operators::selection::TournamentSelection;
use crate::operators::traits::{
    BoundedCrossoverOperator, BoundedMutationOperator, CrossoverOperator, MutationOperator,
    SelectionOperator,
};
use crate::population::individual::Individual;
use crate::population::population::Population;
use crate::termination::{EvolutionState, MaxGenerations, TerminationCriterion};

/// Validate a builder's configuration and bounds, producing a clear typed error.
///
/// Every operator/fitness/termination slot is enforced at compile time by the
/// type-state builder, so this only needs to cover the runtime-configurable
/// fields (population sizing, probabilities, and bounds).
fn validate_config(config: &SimpleGAConfig, bounds: &MultiBounds) -> Result<(), EvolutionError> {
    if config.population_size == 0 {
        return Err(EvolutionError::Configuration(
            "population_size must be at least 1".to_string(),
        ));
    }
    if bounds.dimension() == 0 {
        return Err(EvolutionError::Configuration(
            "bounds must have at least one dimension".to_string(),
        ));
    }
    if config.elitism && config.elite_count > config.population_size {
        return Err(EvolutionError::Configuration(format!(
            "elite_count ({}) cannot exceed population_size ({})",
            config.elite_count, config.population_size
        )));
    }
    if !(0.0..=1.0).contains(&config.crossover_probability) {
        return Err(EvolutionError::Configuration(format!(
            "crossover_probability must be in [0, 1], got {}",
            config.crossover_probability
        )));
    }
    Ok(())
}

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
    adaptive: Option<ThompsonConfig>,
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
            adaptive: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl
    SimpleGABuilder<RealVector, f64, TournamentSelection, SbxCrossover, PolynomialMutation, (), ()>
{
    /// Ergonomic entry point for real-valued optimization — **no turbofish needed**.
    ///
    /// Pins the genome to [`RealVector`] and the fitness value to `f64`, and
    /// pre-installs sensible operator defaults (tournament selection, SBX
    /// crossover, polynomial mutation). Any default is overridable by calling the
    /// corresponding `.selection()` / `.crossover()` / `.mutation()` method.
    ///
    /// ```no_run
    /// use fugue_evo::prelude::*;
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let result = SimpleGABuilder::real_valued()
    ///     .population_size(100)
    ///     .bounds(MultiBounds::symmetric(5.12, 10))
    ///     .fitness(Sphere::new(10))
    ///     .max_generations(200)
    ///     .build()
    ///     .unwrap()
    ///     .run(&mut rng)
    ///     .unwrap();
    /// println!("best = {:.6}", result.best_fitness);
    /// ```
    pub fn real_valued() -> Self {
        SimpleGABuilder::new()
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
    }
}

impl
    SimpleGABuilder<
        BitString,
        usize,
        TournamentSelection,
        UniformCrossover,
        BitFlipMutation,
        (),
        (),
    >
{
    /// Ergonomic entry point for bit-string optimization — **no turbofish needed**.
    ///
    /// Pins the genome to [`BitString`] and the fitness value to `usize`, with
    /// tournament selection, uniform crossover, and bit-flip mutation as
    /// overridable defaults.
    pub fn bit_string() -> Self {
        SimpleGABuilder::new()
            .selection(TournamentSelection::new(3))
            .crossover(UniformCrossover::new())
            .mutation(BitFlipMutation::new())
    }
}

impl
    SimpleGABuilder<
        Permutation,
        f64,
        TournamentSelection,
        OxCrossover,
        PermutationSwapMutation,
        (),
        (),
    >
{
    /// Ergonomic entry point for permutation optimization — **no turbofish needed**.
    ///
    /// Pins the genome to [`Permutation`] and the fitness value to `f64`, with
    /// tournament selection, order crossover (OX), and swap mutation as
    /// overridable defaults.
    pub fn permutation() -> Self {
        SimpleGABuilder::new()
            .selection(TournamentSelection::new(3))
            .crossover(OxCrossover)
            .mutation(PermutationSwapMutation::default())
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

    /// Opt in to online Thompson-sampling tuning of operator parameters.
    ///
    /// When enabled, [`SimpleGA::run_adaptive`] consults a
    /// [`ThompsonSamplingTuner`] each generation for the per-gene mutation
    /// probability and the whole-genome crossover probability, applies the
    /// sampled arm values, and credits the arms with the observed
    /// parent-vs-offspring improvement events. The mutation operator must
    /// implement [`TunableMutation`].
    pub fn adaptive_operators(mut self, config: ThompsonConfig) -> Self {
        self.adaptive = Some(config);
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
            adaptive: self.adaptive,
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
            adaptive: self.adaptive,
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
            adaptive: self.adaptive,
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
            adaptive: self.adaptive,
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
            adaptive: self.adaptive,
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
            adaptive: self.adaptive,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Builder build() method - parallel version with Send + Sync bounds
#[cfg(feature = "parallel")]
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

        // Validate runtime-configurable fields (operators/fitness/termination are
        // already enforced at compile time by the type-state builder).
        validate_config(&self.config, &bounds)?;

        let tuner = self.adaptive.map(|cfg| cfg.build_tuner());

        Ok(SimpleGA {
            config: self.config,
            bounds,
            selection,
            crossover,
            mutation,
            fitness,
            termination,
            tuner,
            _phantom: std::marker::PhantomData,
        })
    }
}

// Builder build() method - non-parallel version without Send + Sync bounds
#[cfg(not(feature = "parallel"))]
impl<G, F, S, C, M, Fit, Term> SimpleGABuilder<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F>,
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

        // Validate runtime-configurable fields (operators/fitness/termination are
        // already enforced at compile time by the type-state builder).
        validate_config(&self.config, &bounds)?;

        let tuner = self.adaptive.map(|cfg| cfg.build_tuner());

        Ok(SimpleGA {
            config: self.config,
            bounds,
            selection,
            crossover,
            mutation,
            fitness,
            termination,
            tuner,
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
    tuner: Option<ThompsonSamplingTuner>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

// Adaptive (Thompson-sampling) run loop.
//
// Available for any tunable mutation operator regardless of the `parallel`
// feature: it evaluates offspring sequentially so it can read each
// parent-vs-offspring improvement event and feed it back to the tuner.
impl<G, F, S, C, M, Fit, Term> SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G> + TunableMutation + Clone,
    Fit: Fitness<Genome = G, Value = F>,
    Term: TerminationCriterion<G, F>,
{
    /// Access the online tuner.
    ///
    /// Populated once [`run_adaptive`](Self::run_adaptive) has run, or immediately
    /// if the builder opted in via
    /// [`SimpleGABuilder::adaptive_operators`](SimpleGABuilder::adaptive_operators).
    pub fn tuner(&self) -> Option<&ThompsonSamplingTuner> {
        self.tuner.as_ref()
    }

    /// Run the GA with online Thompson-sampling tuning of operator parameters.
    ///
    /// Each generation the tuner Thompson-samples a per-gene mutation probability
    /// and a whole-genome crossover probability; those values drive that
    /// generation's operators, and each offspring's improvement over its parents
    /// is credited back to the arm that produced it. If the builder did not opt in
    /// via [`SimpleGABuilder::adaptive_operators`](SimpleGABuilder::adaptive_operators),
    /// a default [`ThompsonConfig`] tuner is created on first use.
    pub fn run_adaptive<R: Rng>(
        &mut self,
        rng: &mut R,
    ) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Own the tuner locally so the loop can freely borrow `self`'s fields.
        let mut tuner = self
            .tuner
            .take()
            .unwrap_or_else(|| ThompsonConfig::default().build_tuner());

        let mut population: Population<G, F> =
            Population::random(self.config.population_size, &self.bounds, rng);
        population.evaluate(&self.fitness);

        let mut stats = EvolutionStats::new();
        let mut evaluations = population.len();
        let mut fitness_history: Vec<f64> = Vec::new();

        let mut best_individual = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        let gen_stats = GenerationStats::from_population(&population, 0, evaluations);
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);
        tuner.snapshot(0);

        loop {
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

            let next_generation = population.generation() + 1;

            // Thompson-sample this generation's operator parameters.
            tuner.select_all(rng);
            let crossover_prob = tuner
                .selected(PARAM_CROSSOVER_PROB)
                .unwrap_or(self.config.crossover_probability);
            let mut mutation = self.mutation.clone();
            if let Some(mutation_prob) = tuner.selected(PARAM_MUTATION_RATE) {
                mutation.set_mutation_probability(mutation_prob);
            }

            let mut new_population: Population<G, F> =
                Population::with_capacity(self.config.population_size);

            // Elitism: carry the best (already-evaluated) individuals forward.
            if self.config.elitism {
                let mut sorted = population.clone();
                sorted.sort_by_fitness();
                for i in 0..self.config.elite_count.min(sorted.len()) {
                    new_population.push(sorted[i].clone());
                }
            }

            let selection_pool: Vec<(G, f64)> = population.as_fitness_pairs();

            while new_population.len() < self.config.population_size {
                let p1 = self.selection.select(&selection_pool, rng);
                let p2 = self.selection.select(&selection_pool, rng);
                let parent1 = &selection_pool[p1].0;
                let parent2 = &selection_pool[p2].0;
                // Improvement is measured against the better of the two parents.
                let parent_best = selection_pool[p1].1.max(selection_pool[p2].1);

                let (mut child1, mut child2) = if rng.gen::<f64>() < crossover_prob {
                    match self.crossover.crossover(parent1, parent2, rng).genome() {
                        Some((c1, c2)) => (c1, c2),
                        None => (parent1.clone(), parent2.clone()),
                    }
                } else {
                    (parent1.clone(), parent2.clone())
                };

                mutation.mutate(&mut child1, rng);
                mutation.mutate(&mut child2, rng);

                // Evaluate immediately so each improvement event credits the arms.
                let f1 = self.fitness.evaluate(&child1);
                tuner.observe(f1.to_f64() > parent_best);
                evaluations += 1;
                let mut ind1 = Individual::with_fitness(child1, f1);
                ind1.birth_generation = next_generation;
                new_population.push(ind1);

                if new_population.len() < self.config.population_size {
                    let f2 = self.fitness.evaluate(&child2);
                    tuner.observe(f2.to_f64() > parent_best);
                    evaluations += 1;
                    let mut ind2 = Individual::with_fitness(child2, f2);
                    ind2.birth_generation = next_generation;
                    new_population.push(ind2);
                }
            }

            while new_population.len() > self.config.population_size {
                new_population.pop();
            }

            new_population.set_generation(next_generation);
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
            tuner.snapshot(next_generation);
        }

        stats.set_runtime(start_time.elapsed());

        // Store the tuner back so callers can inspect the learned posteriors.
        self.tuner = Some(tuner);

        Ok(EvolutionResult::new(
            best_individual.genome,
            best_individual.fitness.unwrap(),
            population.generation(),
            evaluations,
        )
        .with_stats(stats))
    }
}

// Parallel version with Send + Sync bounds
#[cfg(feature = "parallel")]
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

// Bounded operators version - parallel with Send + Sync
#[cfg(feature = "parallel")]
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

// Non-parallel version without Send + Sync bounds
#[cfg(not(feature = "parallel"))]
impl<G, F, S, C, M, Fit, Term> SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F>,
    Term: TerminationCriterion<G, F>,
{
    /// Create a builder for SimpleGA
    pub fn builder() -> SimpleGABuilder<G, F, (), (), (), (), ()> {
        SimpleGABuilder::new()
    }

    /// Run the genetic algorithm
    ///
    /// This is a thin driver over the incremental stepping API
    /// ([`SimpleGA::init_run`], [`SimpleGA::step_generation`],
    /// [`SimpleGA::finish_run`]); it is guaranteed to produce the same result as
    /// driving those methods manually (AUDIT EV-34).
    pub fn run<R: Rng>(&self, rng: &mut R) -> Result<EvolutionResult<G, F>, EvolutionError> {
        let mut state = self.init_run(rng)?;
        while self.step_generation(&mut state, rng)? {}
        Ok(self.finish_run(state))
    }

    /// Initialize an incremental run: build and evaluate the initial population
    /// and record generation-0 statistics.
    ///
    /// The returned [`SimpleGaRun`] can then be advanced one generation at a time
    /// with [`SimpleGA::step_generation`] and consumed with
    /// [`SimpleGA::finish_run`]. This is the incremental counterpart to
    /// [`SimpleGA::run`], letting callers (e.g. the WASM bindings) report
    /// progress and cancel early (AUDIT EV-34).
    pub fn init_run<R: Rng>(&self, rng: &mut R) -> Result<SimpleGaRun<G, F>, EvolutionError> {
        let start_time = Instant::now();

        // Initialize population
        let mut population: Population<G, F> =
            Population::random(self.config.population_size, &self.bounds, rng);

        // Evaluate initial population (sequential only)
        let eval_start = Instant::now();
        population.evaluate(&self.fitness);
        let eval_time = eval_start.elapsed();

        let mut stats = EvolutionStats::new();
        let evaluations = population.len();
        let mut fitness_history: Vec<f64> = Vec::new();

        // Track best individual
        let best_individual = population
            .best()
            .ok_or(EvolutionError::EmptyPopulation)?
            .clone();

        // Record initial statistics
        let gen_stats = GenerationStats::from_population(&population, 0, evaluations)
            .with_timing(TimingStats::new().with_evaluation(eval_time));
        fitness_history.push(gen_stats.best_fitness);
        stats.record(gen_stats);

        Ok(SimpleGaRun {
            population,
            best_individual,
            evaluations,
            fitness_history,
            stats,
            start_time,
            terminated: false,
        })
    }

    /// Advance an incremental run by a single generation.
    ///
    /// Returns `Ok(true)` if a generation was executed, or `Ok(false)` if the
    /// termination criterion fired (in which case `state` is left otherwise
    /// unchanged and marked terminated). This mirrors exactly one iteration of
    /// [`SimpleGA::run`]'s main loop, including the termination check performed
    /// before the generation body.
    pub fn step_generation<R: Rng>(
        &self,
        state: &mut SimpleGaRun<G, F>,
        rng: &mut R,
    ) -> Result<bool, EvolutionError> {
        if state.terminated {
            return Ok(false);
        }

        // Check termination
        let evo_state = EvolutionState {
            generation: state.population.generation(),
            evaluations: state.evaluations,
            best_fitness: state.best_individual.fitness_value().to_f64(),
            population: &state.population,
            fitness_history: &state.fitness_history,
        };

        if self.termination.should_terminate(&evo_state) {
            state
                .stats
                .set_termination_reason(self.termination.reason());
            state.terminated = true;
            return Ok(false);
        }

        let gen_start = Instant::now();

        // Create new generation
        let mut new_population: Population<G, F> =
            Population::with_capacity(self.config.population_size);

        // Elitism: copy best individuals
        if self.config.elitism {
            let mut sorted = state.population.clone();
            sorted.sort_by_fitness();
            for i in 0..self.config.elite_count.min(sorted.len()) {
                new_population.push(sorted[i].clone());
            }
        }

        // Selection pool
        let selection_pool: Vec<(G, f64)> = state.population.as_fitness_pairs();

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
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_probability {
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
                state.population.generation() + 1,
            ));
            if new_population.len() < self.config.population_size {
                new_population.push(Individual::with_generation(
                    child2,
                    state.population.generation() + 1,
                ));
            }
        }

        // Truncate to exact size
        while new_population.len() > self.config.population_size {
            new_population.pop();
        }

        // Evaluate new population (sequential only)
        let eval_start = Instant::now();
        new_population.evaluate(&self.fitness);
        let eval_time = eval_start.elapsed();
        state.evaluations += new_population.len()
            - (if self.config.elitism {
                self.config.elite_count
            } else {
                0
            });

        // Update generation counter
        new_population.set_generation(state.population.generation() + 1);
        state.population = new_population;

        // Update best individual
        if let Some(best) = state.population.best() {
            if best.is_better_than(&state.best_individual) {
                state.best_individual = best.clone();
            }
        }

        // Record statistics
        let timing = TimingStats::new()
            .with_selection(selection_time)
            .with_crossover(crossover_time)
            .with_mutation(mutation_time)
            .with_evaluation(eval_time)
            .with_total(gen_start.elapsed());

        let gen_stats = GenerationStats::from_population(
            &state.population,
            state.population.generation(),
            state.evaluations,
        )
        .with_timing(timing);
        state.fitness_history.push(gen_stats.best_fitness);
        state.stats.record(gen_stats);

        Ok(true)
    }

    /// Consume an incremental run and produce the final [`EvolutionResult`],
    /// mirroring the tail of [`SimpleGA::run`].
    pub fn finish_run(&self, mut state: SimpleGaRun<G, F>) -> EvolutionResult<G, F> {
        state.stats.set_runtime(state.start_time.elapsed());

        EvolutionResult::new(
            state.best_individual.genome,
            state
                .best_individual
                .fitness
                .expect("best individual is always evaluated"),
            state.population.generation(),
            state.evaluations,
        )
        .with_stats(state.stats)
    }

    /// Inject migrant genomes into an in-progress run, replacing the current
    /// worst individuals.
    ///
    /// Each migrant is evaluated with this GA's own fitness function (so a genome
    /// that emigrated from another island is scored under the receiving island's
    /// objective) and overwrites one of the worst individuals in the population.
    /// The best-so-far individual is refreshed and the evaluation counter is
    /// advanced by the number of migrants accepted. Used to build island-model
    /// migration on top of the incremental stepping API (AUDIT EV-77).
    pub fn inject_migrants(&self, state: &mut SimpleGaRun<G, F>, migrants: Vec<G>) {
        if migrants.is_empty() {
            return;
        }

        // Sort best-first so the worst individuals sit at the tail.
        state.population.sort_by_fitness();
        let n = state.population.len();
        if n == 0 {
            return;
        }
        let k = migrants.len().min(n);

        let evaluated: Vec<Individual<G, F>> = migrants
            .into_iter()
            .take(k)
            .map(|genome| {
                let value = self.fitness.evaluate(&genome);
                Individual::with_fitness(genome, value)
            })
            .collect();

        {
            let individuals = state.population.individuals_mut();
            for (i, individual) in evaluated.into_iter().enumerate() {
                let idx = n - 1 - i; // replace worst-first
                individuals[idx] = individual;
            }
        }
        state.evaluations += k;

        // Refresh the tracked best individual.
        if let Some(best) = state.population.best() {
            if best.is_better_than(&state.best_individual) {
                state.best_individual = best.clone();
            }
        }
    }
}

/// Mutable state for driving a [`SimpleGA`] one generation at a time.
///
/// Produced by [`SimpleGA::init_run`], advanced by
/// [`SimpleGA::step_generation`], and consumed by [`SimpleGA::finish_run`]. This
/// is the incremental counterpart to [`SimpleGA::run`] (AUDIT EV-34): callers can
/// drive the generation loop, read progress via the getters below, and cancel
/// early — all without changing `run`'s behavior, since `run` is implemented in
/// terms of these methods.
#[cfg(not(feature = "parallel"))]
pub struct SimpleGaRun<G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    population: Population<G, F>,
    best_individual: Individual<G, F>,
    evaluations: usize,
    fitness_history: Vec<f64>,
    stats: EvolutionStats,
    start_time: Instant,
    terminated: bool,
}

#[cfg(not(feature = "parallel"))]
impl<G, F> SimpleGaRun<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Number of generations completed so far.
    pub fn generation(&self) -> usize {
        self.population.generation()
    }

    /// Total fitness evaluations performed so far.
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Best fitness found so far (as `f64`).
    pub fn best_fitness(&self) -> f64 {
        self.best_individual.fitness_value().to_f64()
    }

    /// The best genome found so far.
    pub fn best_genome(&self) -> &G {
        &self.best_individual.genome
    }

    /// Per-generation best-so-far fitness trajectory (index 0 is generation 0).
    pub fn fitness_history(&self) -> &[f64] {
        &self.fitness_history
    }

    /// `true` once the termination criterion has fired.
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Clone the best `k` genomes in the current population (best first).
    ///
    /// Used to select emigrants for island-model migration (AUDIT EV-77).
    pub fn best_genomes(&self, k: usize) -> Vec<G> {
        let mut sorted = self.population.clone();
        sorted.sort_by_fitness();
        sorted
            .iter()
            .take(k)
            .map(|ind| ind.genome.clone())
            .collect()
    }
}

// Bounded operators version - non-parallel without Send + Sync
#[cfg(not(feature = "parallel"))]
impl<G, F, S, C, M, Fit, Term> SimpleGA<G, F, S, C, M, Fit, Term>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    S: SelectionOperator<G>,
    C: BoundedCrossoverOperator<G>,
    M: BoundedMutationOperator<G>,
    Fit: Fitness<Genome = G, Value = F>,
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

        // Evaluate initial population (sequential only)
        population.evaluate(&self.fitness);

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

            // Evaluate (sequential only)
            new_population.evaluate(&self.fitness);
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
    use crate::operators::mutation::{BitFlipMutation, GaussianMutation, PolynomialMutation};
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

    // regression: EV-34 — the incremental stepping API (init_run / step_generation
    // / finish_run) must reproduce run() exactly, since run() is implemented in
    // terms of it. Only compiled in the non-parallel build where the API exists.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_step_api_matches_run() {
        use rand::SeedableRng;

        let build = || {
            SimpleGABuilder::new()
                .population_size(30)
                .bounds(MultiBounds::symmetric(5.12, 6))
                .selection(TournamentSelection::new(3))
                .crossover(SbxCrossover::new(20.0))
                .mutation(PolynomialMutation::new(20.0))
                .fitness(Sphere::new(6))
                .max_generations(25)
                .build()
                .unwrap()
        };

        // One-shot run().
        let ga_a = build();
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(2024);
        let run_result = ga_a.run(&mut rng_a).unwrap();

        // Manual stepping with the same seed.
        let ga_b = build();
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(2024);
        let mut state = ga_b.init_run(&mut rng_b).unwrap();
        let mut steps = 0;
        while ga_b.step_generation(&mut state, &mut rng_b).unwrap() {
            steps += 1;
        }
        assert!(state.is_terminated());
        assert_eq!(steps, 25, "should take exactly max_generations steps");
        let step_result = ga_b.finish_run(state);

        assert_eq!(run_result.generations, step_result.generations);
        assert_eq!(run_result.evaluations, step_result.evaluations);
        assert_eq!(run_result.best_fitness, step_result.best_fitness);
        assert_eq!(
            run_result.best_genome.genes(),
            step_result.best_genome.genes()
        );
    }

    // regression: EV-77 — migration helpers used by the WASM island model.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn test_inject_migrants_replaces_worst() {
        use rand::SeedableRng;

        let ga = SimpleGABuilder::new()
            .population_size(20)
            .bounds(MultiBounds::symmetric(5.12, 4))
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(20.0))
            .mutation(PolynomialMutation::new(20.0))
            .fitness(Sphere::new(4))
            .max_generations(10)
            .build()
            .unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut state = ga.init_run(&mut rng).unwrap();
        for _ in 0..3 {
            ga.step_generation(&mut state, &mut rng).unwrap();
        }

        let evals_before = state.evaluations();
        // The optimum for the (negated) Sphere is the origin; inject it and the
        // best-so-far must improve to ~0 and the evaluation count must rise.
        let optimum = RealVector::new(vec![0.0; 4]);
        ga.inject_migrants(&mut state, vec![optimum]);

        assert_eq!(state.evaluations(), evals_before + 1);
        assert!(
            state.best_fitness() > -1e-9,
            "injected optimum should become the best (got {})",
            state.best_fitness()
        );

        // best_genomes returns clones of the top individuals.
        let top = state.best_genomes(2);
        assert_eq!(top.len(), 2);
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

    /// regression: EV-35 — the quickstart must work with ZERO turbofish via the
    /// `real_valued()` entry point (pre-fix, the only path was the 7-parameter
    /// `SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()` turbofish form).
    #[test]
    fn test_real_valued_constructor_no_turbofish() {
        let mut rng = rand::thread_rng();
        let ga = SimpleGABuilder::real_valued()
            .population_size(40)
            .bounds(MultiBounds::symmetric(5.12, 10))
            .fitness(Sphere::new(10))
            .max_generations(20)
            .build()
            .unwrap();
        let result = ga.run(&mut rng).unwrap();
        assert!(result.evaluations > 0);
    }

    /// regression: EV-35 — a default operator installed by `real_valued()` remains
    /// overridable (calling `.mutation(...)` swaps it and still builds).
    #[test]
    fn test_real_valued_constructor_override_operator() {
        let mut rng = rand::thread_rng();
        let ga = SimpleGABuilder::real_valued()
            .mutation(GaussianMutation::new(0.1))
            .population_size(30)
            .bounds(MultiBounds::symmetric(5.12, 5))
            .fitness(Sphere::new(5))
            .max_generations(10)
            .build()
            .unwrap();
        assert!(ga.run(&mut rng).is_ok());
    }

    /// regression: EV-35 — bit-string quickstart with zero turbofish.
    #[test]
    fn test_bit_string_constructor_no_turbofish() {
        let mut rng = rand::thread_rng();
        let ga = SimpleGABuilder::bit_string()
            .population_size(40)
            .bounds(MultiBounds::uniform(
                crate::genome::bounds::Bounds::unit(),
                20,
            ))
            .fitness(OneMax::new(20))
            .max_generations(30)
            .build()
            .unwrap();
        let result = ga.run(&mut rng).unwrap();
        assert!(result.best_fitness >= 10);
    }

    /// regression: EV-35 — permutation quickstart with zero turbofish.
    #[test]
    fn test_permutation_constructor_no_turbofish() {
        use crate::fitness::traits::FnFitness;
        use crate::genome::permutation::Permutation;
        use crate::genome::traits::PermutationGenome;

        let mut rng = rand::thread_rng();
        // Maximized when the permutation equals the identity (sum of |v - i| = 0).
        let fitness = FnFitness::new(|p: &Permutation| -> f64 {
            -(p.permutation()
                .iter()
                .enumerate()
                .map(|(i, &v)| (v as f64 - i as f64).abs())
                .sum::<f64>())
        });
        let ga = SimpleGABuilder::permutation()
            .population_size(30)
            .bounds(MultiBounds::symmetric(1.0, 8))
            .fitness(fitness)
            .max_generations(10)
            .build()
            .unwrap();
        assert!(ga.run(&mut rng).is_ok());
    }

    /// regression: EV-86 — build() must reject an invalid config with a clear
    /// typed `Configuration` error rather than proceeding or panicking.
    #[test]
    fn test_build_rejects_zero_population() {
        let ga = SimpleGABuilder::real_valued()
            .population_size(0)
            .bounds(MultiBounds::symmetric(5.12, 10))
            .fitness(Sphere::new(10))
            .max_generations(10)
            .build();
        assert!(ga.is_err());
        let msg = ga.err().unwrap().to_string();
        assert!(msg.contains("population_size"), "got: {msg}");
    }

    /// regression: EV-86 — build() validates the crossover probability range.
    #[test]
    fn test_build_rejects_out_of_range_crossover_probability() {
        let ga = SimpleGABuilder::real_valued()
            .crossover_probability(1.5)
            .population_size(20)
            .bounds(MultiBounds::symmetric(5.12, 5))
            .fitness(Sphere::new(5))
            .max_generations(5)
            .build();
        assert!(ga.is_err());
        assert!(ga
            .err()
            .unwrap()
            .to_string()
            .contains("crossover_probability"));
    }

    /// regression: EV-86 — elite_count exceeding population_size is rejected.
    #[test]
    fn test_build_rejects_elite_count_exceeding_population() {
        let ga = SimpleGABuilder::real_valued()
            .population_size(10)
            .elite_count(50)
            .bounds(MultiBounds::symmetric(5.12, 5))
            .fitness(Sphere::new(5))
            .max_generations(5)
            .build();
        assert!(ga.is_err());
        assert!(ga.err().unwrap().to_string().contains("elite_count"));
    }

    /// regression: EV-21 — the adaptive tuner is actually wired into the GA: after
    /// `run_adaptive` the tuner has received improvement feedback (pre-fix, no
    /// algorithm consumed the learner at all).
    #[test]
    fn test_run_adaptive_feeds_tuner() {
        use crate::hyperparameter::bayesian::{ThompsonConfig, PARAM_MUTATION_RATE};

        let mut rng = rand::thread_rng();
        let mut ga = SimpleGABuilder::real_valued()
            .population_size(30)
            .bounds(MultiBounds::symmetric(5.12, 8))
            .fitness(Sphere::new(8))
            .max_generations(15)
            .adaptive_operators(ThompsonConfig::default())
            .build()
            .unwrap();

        let result = ga.run_adaptive(&mut rng).unwrap();
        assert!(result.evaluations > 0);

        let tuner = ga
            .tuner()
            .expect("tuner should be present after run_adaptive");
        assert!(
            tuner.total_observations() > 0,
            "tuner must receive improvement feedback"
        );
        let mr = tuner.parameter(PARAM_MUTATION_RATE).unwrap();
        assert!(
            mr.total_observations() > 0.0,
            "mutation-rate arms must accumulate observations"
        );
    }
}
