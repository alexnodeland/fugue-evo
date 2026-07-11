//! Island Model Parallelism
//!
//! Implements a distributed evolutionary algorithm where multiple populations
//! (islands) evolve independently with periodic migration of individuals.
//!
//! Note: This module requires the `parallel` feature to be enabled.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{EvoResult, EvolutionError, OperatorResult};
use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::{CrossoverOperator, MutationOperator, SelectionOperator};
use crate::population::individual::Individual;
use crate::population::population::Population;

/// Migration topology determines which islands exchange individuals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MigrationTopology {
    /// Ring topology: each island sends to the next
    Ring,
    /// Fully connected: every island can send to any other
    FullyConnected,
    /// Random: each island sends to a randomly chosen island
    Random,
    /// Star: all islands send to/receive from a central hub
    Star { hub_index: usize },
}

impl Default for MigrationTopology {
    fn default() -> Self {
        Self::Ring
    }
}

impl MigrationTopology {
    /// Get the target islands for migration from a given source island
    pub fn targets(&self, source: usize, num_islands: usize) -> Vec<usize> {
        match self {
            Self::Ring => vec![(source + 1) % num_islands],
            Self::FullyConnected => (0..num_islands).filter(|&i| i != source).collect(),
            Self::Random => (0..num_islands).filter(|&i| i != source).collect(),
            Self::Star { hub_index } => {
                if source == *hub_index {
                    (0..num_islands).filter(|&i| i != source).collect()
                } else {
                    vec![*hub_index]
                }
            }
        }
    }
}

/// Migration policy determines which and how many individuals migrate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MigrationPolicy {
    /// Best k individuals migrate
    Best(usize),
    /// Random k individuals migrate
    Random(usize),
    /// Best k individuals replace worst k on target
    BestReplaceWorst(usize),
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        Self::Best(1)
    }
}

/// Configuration for the island model
#[derive(Clone, Debug)]
pub struct IslandModelConfig {
    /// Number of islands
    pub num_islands: usize,
    /// Population size per island
    pub island_population_size: usize,
    /// Migration interval (generations between migrations)
    pub migration_interval: usize,
    /// Migration topology
    pub topology: MigrationTopology,
    /// Migration policy
    pub policy: MigrationPolicy,
    /// Bounds for genome generation
    pub bounds: MultiBounds,
    /// Number of elites to preserve per island
    pub elitism: usize,
}

impl Default for IslandModelConfig {
    fn default() -> Self {
        Self {
            num_islands: 4,
            island_population_size: 50,
            migration_interval: 10,
            topology: MigrationTopology::Ring,
            policy: MigrationPolicy::Best(1),
            bounds: MultiBounds::symmetric(5.0, 10),
            elitism: 1,
        }
    }
}

/// State of a single island
pub struct Island<G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Island index
    pub index: usize,
    /// Current population
    pub population: Population<G, F>,
    /// Best individual found on this island
    pub best: Option<Individual<G, F>>,
    /// Current generation (local to island)
    pub generation: usize,
    /// Total fitness evaluations on this island
    pub evaluations: usize,
}

impl<G, F> Island<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new island with a random population
    pub fn new<R: rand::Rng>(index: usize, size: usize, bounds: &MultiBounds, rng: &mut R) -> Self {
        Self {
            index,
            population: Population::random(size, bounds, rng),
            best: None,
            generation: 0,
            evaluations: 0,
        }
    }

    /// Create an island with an existing population
    pub fn with_population(index: usize, population: Population<G, F>) -> Self {
        Self {
            index,
            population,
            best: None,
            generation: 0,
            evaluations: 0,
        }
    }

    /// Run one generation of evolution on this island
    pub fn evolve_one_generation<Fit, Sel, Cross, Mut, R>(
        &mut self,
        fitness: &Fit,
        selection: &Sel,
        crossover: &Cross,
        mutation: &Mut,
        elitism: usize,
        rng: &mut R,
    ) -> EvoResult<()>
    where
        Fit: Fitness<Genome = G, Value = F>,
        Sel: SelectionOperator<G>,
        Cross: CrossoverOperator<G>,
        Mut: MutationOperator<G>,
        R: rand::Rng,
    {
        // Evaluate population. EV-82: count only individuals that actually needed
        // evaluating — carried-over elites are already scored and are skipped by
        // Population::evaluate, so adding the full population length over-counts.
        let newly_evaluated = self.population.len() - self.population.count_evaluated();
        self.population.evaluate(fitness);
        self.evaluations += newly_evaluated;

        // Update best
        if let Some(current_best) = self.population.best() {
            match &self.best {
                None => self.best = Some(current_best.clone()),
                Some(best) if current_best.is_better_than(best) => {
                    self.best = Some(current_best.clone());
                }
                _ => {}
            }
        }

        // Preserve elites. EV-83: cap the elite count at the population size so an
        // over-large `elitism` cannot cause a usize underflow below.
        self.population.sort_by_fitness();
        let pop_len = self.population.len();
        let elite_count = elitism.min(pop_len);
        let target_offspring = pop_len - elite_count;
        let elites: Vec<_> = self.population.iter().take(elite_count).cloned().collect();

        // Prepare selection pool: (genome, fitness) pairs
        let selection_pool: Vec<(G, f64)> = self
            .population
            .iter()
            .filter_map(|ind| {
                ind.fitness
                    .as_ref()
                    .map(|f| (ind.genome.clone(), f.to_f64()))
            })
            .collect();

        if selection_pool.len() < 2 {
            return Err(EvolutionError::EmptyPopulation);
        }

        // Selection and reproduction
        let mut offspring = Vec::with_capacity(target_offspring);

        while offspring.len() < target_offspring {
            // Select parents
            let idx1 = selection.select(&selection_pool, rng);
            let idx2 = selection.select(&selection_pool, rng);
            let parent1 = &selection_pool[idx1].0;
            let parent2 = &selection_pool[idx2].0;

            // Crossover
            let (mut child1, mut child2) = match crossover.crossover(parent1, parent2, rng) {
                OperatorResult::Success((c1, c2)) | OperatorResult::Repaired((c1, c2), _) => {
                    (c1, c2)
                }
                OperatorResult::Failed(_) => (parent1.clone(), parent2.clone()),
            };

            // Mutation
            mutation.mutate(&mut child1, rng);
            mutation.mutate(&mut child2, rng);

            offspring.push(Individual::new(child1));
            if offspring.len() < target_offspring {
                offspring.push(Individual::new(child2));
            }
        }

        // Replace population with elites + offspring
        let mut new_population = Population::new();
        for elite in elites {
            new_population.push(elite);
        }
        for child in offspring {
            new_population.push(child);
        }
        self.population = new_population;

        self.generation += 1;
        Ok(())
    }

    /// Get emigrants for migration (individuals leaving this island)
    pub fn get_emigrants<R: rand::Rng>(
        &self,
        policy: &MigrationPolicy,
        rng: &mut R,
    ) -> Vec<Individual<G, F>> {
        match policy {
            MigrationPolicy::Best(k) => {
                let mut sorted: Vec<_> = self.population.iter().cloned().collect();
                sorted.sort_by(|a, b| match (a.fitness.as_ref(), b.fitness.as_ref()) {
                    (Some(fa), Some(fb)) => fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
                sorted.into_iter().take(*k).collect()
            }
            MigrationPolicy::Random(k) => {
                use rand::seq::SliceRandom;
                let mut individuals: Vec<_> = self.population.iter().cloned().collect();
                individuals.shuffle(rng);
                individuals.into_iter().take(*k).collect()
            }
            MigrationPolicy::BestReplaceWorst(k) => {
                let mut sorted: Vec<_> = self.population.iter().cloned().collect();
                sorted.sort_by(|a, b| match (a.fitness.as_ref(), b.fitness.as_ref()) {
                    (Some(fa), Some(fb)) => fb.partial_cmp(fa).unwrap_or(std::cmp::Ordering::Equal),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                });
                sorted.into_iter().take(*k).collect()
            }
        }
    }

    /// Accept immigrants (individuals coming to this island).
    ///
    /// EV-41: immigrants always replace the island's WORST members, regardless of
    /// the migration policy, so an island's best individual is never overwritten
    /// by an arriving (possibly worse) migrant. `sort_by_fitness` orders the
    /// population best-first, so the worst members occupy the tail.
    pub fn accept_immigrants(&mut self, immigrants: Vec<Individual<G, F>>) {
        if immigrants.is_empty() || self.population.is_empty() {
            return;
        }

        self.population.sort_by_fitness();
        let pop_len = self.population.len();
        for (i, immigrant) in immigrants.into_iter().enumerate() {
            if i >= pop_len {
                break;
            }
            self.population[pop_len - 1 - i] = immigrant;
        }
    }
}

/// Island Model Evolutionary Algorithm
pub struct IslandModel<G, Fit, Sel, Cross, Mut, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Configuration
    pub config: IslandModelConfig,
    /// Islands
    pub islands: Vec<Island<G, F>>,
    /// One persistent RNG per island, derived once from the caller's master RNG
    /// (EV-12). Reusing these across generations makes seeded runs bit-reproducible
    /// even though islands are evolved in parallel.
    island_rngs: Vec<StdRng>,
    /// Fitness function (shared)
    pub fitness: Arc<Fit>,
    /// Selection operator
    pub selection: Sel,
    /// Crossover operator
    pub crossover: Cross,
    /// Mutation operator
    pub mutation: Mut,
    /// Global best individual
    pub global_best: Option<Individual<G, F>>,
    /// Total generations (global)
    pub generation: usize,
    /// Total evaluations across all islands
    pub total_evaluations: usize,
}

impl<G, Fit, Sel, Cross, Mut, F> IslandModel<G, Fit, Sel, Cross, Mut, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    Fit: Fitness<Genome = G, Value = F> + Send + Sync,
    Sel: SelectionOperator<G> + Clone + Send + Sync,
    Cross: CrossoverOperator<G> + Clone + Send + Sync,
    Mut: MutationOperator<G> + Clone + Send + Sync,
{
    /// Create a new island model
    pub fn new<R: rand::Rng>(
        config: IslandModelConfig,
        fitness: Fit,
        selection: Sel,
        crossover: Cross,
        mutation: Mut,
        rng: &mut R,
    ) -> Self {
        let islands: Vec<_> = (0..config.num_islands)
            .map(|i| {
                let mut island_rng = StdRng::from_seed(rng.gen());
                Island::new(
                    i,
                    config.island_population_size,
                    &config.bounds,
                    &mut island_rng,
                )
            })
            .collect();

        // EV-12: draw one persistent working RNG per island from the master RNG
        // once, up front, so per-island search is deterministic under a fixed seed.
        let island_rngs: Vec<StdRng> = (0..config.num_islands)
            .map(|_| StdRng::seed_from_u64(rng.gen()))
            .collect();

        Self {
            config,
            islands,
            island_rngs,
            fitness: Arc::new(fitness),
            selection,
            crossover,
            mutation,
            global_best: None,
            generation: 0,
            total_evaluations: 0,
        }
    }

    /// Run evolution for a specified number of generations
    pub fn run<R: rand::Rng>(
        &mut self,
        max_generations: usize,
        rng: &mut R,
    ) -> EvoResult<&Individual<G, F>> {
        for _ in 0..max_generations {
            self.step(rng)?;
        }

        self.global_best
            .as_ref()
            .ok_or(EvolutionError::EmptyPopulation)
    }

    /// Perform one generation step on all islands
    pub fn step<R: rand::Rng>(&mut self, rng: &mut R) -> EvoResult<()> {
        // Evolve each island independently (can be parallelized)
        let elitism = self.config.elitism;
        let fitness = Arc::clone(&self.fitness);
        let selection = self.selection.clone();
        let crossover = self.crossover.clone();
        let mutation = self.mutation.clone();

        // Parallel evolution of islands. EV-12: each island uses its own
        // persistent, master-seeded RNG (not OS entropy), so a seeded run is
        // reproducible even though evaluation happens in parallel.
        self.islands
            .par_iter_mut()
            .zip(self.island_rngs.par_iter_mut())
            .for_each(|(island, island_rng)| {
                let _ = island.evolve_one_generation(
                    fitness.as_ref(),
                    &selection,
                    &crossover,
                    &mutation,
                    elitism,
                    island_rng,
                );
            });

        // Update global best
        for island in &self.islands {
            if let Some(island_best) = &island.best {
                match &self.global_best {
                    None => self.global_best = Some(island_best.clone()),
                    Some(global) if island_best.is_better_than(global) => {
                        self.global_best = Some(island_best.clone());
                    }
                    _ => {}
                }
            }
        }

        self.generation += 1;
        self.total_evaluations = self.islands.iter().map(|i| i.evaluations).sum();

        // Migration
        if self
            .generation
            .is_multiple_of(self.config.migration_interval)
        {
            self.migrate(rng);
        }

        Ok(())
    }

    /// Perform migration between islands
    fn migrate<R: rand::Rng>(&mut self, rng: &mut R) {
        let num_islands = self.islands.len();
        let policy = self.config.policy.clone();
        let topology = self.config.topology.clone();

        // Collect emigrants from each island using that island's persistent RNG
        // (EV-12), so random-emigrant selection is reproducible under a seed.
        let emigrants: Vec<Vec<Individual<G, F>>> = self
            .islands
            .iter()
            .zip(self.island_rngs.iter_mut())
            .map(|(island, island_rng)| island.get_emigrants(&policy, island_rng))
            .collect();

        // Route emigrants to target islands
        for (source, source_emigrants) in emigrants.into_iter().enumerate() {
            let targets = topology.targets(source, num_islands);

            if targets.is_empty() {
                continue;
            }

            match topology {
                MigrationTopology::Random => {
                    // Pick one random target
                    let target = targets[rng.gen_range(0..targets.len())];
                    self.islands[target].accept_immigrants(source_emigrants);
                }
                _ => {
                    // EV-11: broadcast to EVERY target the topology defines
                    // (FullyConnected reaches all peers; a Star hub reaches all
                    // spokes), not just the first one.
                    for &target in &targets {
                        self.islands[target].accept_immigrants(source_emigrants.clone());
                    }
                }
            }
        }
    }

    /// Get a combined population from all islands
    pub fn combined_population(&self) -> Population<G, F> {
        let mut combined = Population::new();
        for island in &self.islands {
            for individual in island.population.iter() {
                combined.push(individual.clone());
            }
        }
        combined
    }

    /// Get statistics about each island
    pub fn island_statistics(&self) -> Vec<IslandStats<F>> {
        self.islands
            .iter()
            .map(|island| {
                let (sum, best) = island
                    .population
                    .iter()
                    .filter_map(|i| i.fitness.clone())
                    .fold((0.0, None::<F>), |(sum, best), f| {
                        let new_best = match best {
                            None => Some(f.clone()),
                            Some(b) if f.is_better_than(&b) => Some(f.clone()),
                            b => b,
                        };
                        (sum + f.to_f64(), new_best)
                    });

                let count = island
                    .population
                    .iter()
                    .filter(|i| i.fitness.is_some())
                    .count();

                IslandStats {
                    index: island.index,
                    generation: island.generation,
                    evaluations: island.evaluations,
                    population_size: island.population.len(),
                    mean_fitness: if count > 0 { sum / count as f64 } else { 0.0 },
                    best_fitness: best,
                }
            })
            .collect()
    }
}

/// Statistics for a single island
#[derive(Clone, Debug)]
pub struct IslandStats<F: FitnessValue> {
    /// Island index
    pub index: usize,
    /// Current generation
    pub generation: usize,
    /// Total evaluations
    pub evaluations: usize,
    /// Population size
    pub population_size: usize,
    /// Mean fitness
    pub mean_fitness: f64,
    /// Best fitness on this island
    pub best_fitness: Option<F>,
}

/// Builder for IslandModel
pub struct IslandModelBuilder<G, Fit, Sel, Cross, Mut, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    config: IslandModelConfig,
    fitness: Option<Fit>,
    selection: Option<Sel>,
    crossover: Option<Cross>,
    mutation: Option<Mut>,
    _phantom: std::marker::PhantomData<(G, F)>,
}

impl<G, Fit, Sel, Cross, Mut, F> IslandModelBuilder<G, Fit, Sel, Cross, Mut, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    Fit: Fitness<Genome = G, Value = F> + Send + Sync,
    Sel: SelectionOperator<G> + Clone + Send + Sync,
    Cross: CrossoverOperator<G> + Clone + Send + Sync,
    Mut: MutationOperator<G> + Clone + Send + Sync,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: IslandModelConfig::default(),
            fitness: None,
            selection: None,
            crossover: None,
            mutation: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set number of islands
    pub fn num_islands(mut self, n: usize) -> Self {
        self.config.num_islands = n;
        self
    }

    /// Set population size per island
    pub fn island_population_size(mut self, size: usize) -> Self {
        self.config.island_population_size = size;
        self
    }

    /// Set migration interval
    pub fn migration_interval(mut self, interval: usize) -> Self {
        self.config.migration_interval = interval;
        self
    }

    /// Set migration topology
    pub fn topology(mut self, topology: MigrationTopology) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set migration policy
    pub fn migration_policy(mut self, policy: MigrationPolicy) -> Self {
        self.config.policy = policy;
        self
    }

    /// Set bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.config.bounds = bounds;
        self
    }

    /// Set elitism
    pub fn elitism(mut self, n: usize) -> Self {
        self.config.elitism = n;
        self
    }

    /// Set fitness function
    pub fn fitness(mut self, fitness: Fit) -> Self {
        self.fitness = Some(fitness);
        self
    }

    /// Set selection operator
    pub fn selection(mut self, selection: Sel) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Set crossover operator
    pub fn crossover(mut self, crossover: Cross) -> Self {
        self.crossover = Some(crossover);
        self
    }

    /// Set mutation operator
    pub fn mutation(mut self, mutation: Mut) -> Self {
        self.mutation = Some(mutation);
        self
    }

    /// Build the island model
    pub fn build<R: rand::Rng>(
        self,
        rng: &mut R,
    ) -> EvoResult<IslandModel<G, Fit, Sel, Cross, Mut, F>> {
        let fitness = self.fitness.ok_or(EvolutionError::Configuration(
            "Fitness function is required".to_string(),
        ))?;
        let selection = self.selection.ok_or(EvolutionError::Configuration(
            "Selection operator is required".to_string(),
        ))?;
        let crossover = self.crossover.ok_or(EvolutionError::Configuration(
            "Crossover operator is required".to_string(),
        ))?;
        let mutation = self.mutation.ok_or(EvolutionError::Configuration(
            "Mutation operator is required".to_string(),
        ))?;

        Ok(IslandModel::new(
            self.config,
            fitness,
            selection,
            crossover,
            mutation,
            rng,
        ))
    }
}

impl<G, Fit, Sel, Cross, Mut, F> Default for IslandModelBuilder<G, Fit, Sel, Cross, Mut, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    Fit: Fitness<Genome = G, Value = F> + Send + Sync,
    Sel: SelectionOperator<G> + Clone + Send + Sync,
    Cross: CrossoverOperator<G> + Clone + Send + Sync,
    Mut: MutationOperator<G> + Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::genome::real_vector::RealVector;
    use crate::operators::crossover::BlxAlphaCrossover;
    use crate::operators::mutation::GaussianMutation;
    use crate::operators::selection::TournamentSelection;
    use rand::SeedableRng;

    #[test]
    fn test_migration_topology_ring() {
        let topology = MigrationTopology::Ring;
        assert_eq!(topology.targets(0, 4), vec![1]);
        assert_eq!(topology.targets(1, 4), vec![2]);
        assert_eq!(topology.targets(3, 4), vec![0]);
    }

    #[test]
    fn test_migration_topology_fully_connected() {
        let topology = MigrationTopology::FullyConnected;
        let targets = topology.targets(0, 4);
        assert_eq!(targets.len(), 3);
        assert!(!targets.contains(&0));
    }

    #[test]
    fn test_migration_topology_star() {
        let topology = MigrationTopology::Star { hub_index: 0 };
        // Non-hub sends to hub
        assert_eq!(topology.targets(1, 4), vec![0]);
        assert_eq!(topology.targets(2, 4), vec![0]);
        // Hub sends to all
        let hub_targets = topology.targets(0, 4);
        assert_eq!(hub_targets.len(), 3);
    }

    #[test]
    fn test_island_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let bounds = MultiBounds::symmetric(5.0, 5);
        let island: Island<RealVector> = Island::new(0, 10, &bounds, &mut rng);

        assert_eq!(island.index, 0);
        assert_eq!(island.population.len(), 10);
        assert_eq!(island.generation, 0);
    }

    #[test]
    fn test_island_model_builder() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let bounds = MultiBounds::symmetric(5.0, 5);

        let result = IslandModelBuilder::<RealVector, _, _, _, _>::new()
            .num_islands(4)
            .island_population_size(20)
            .migration_interval(5)
            .topology(MigrationTopology::Ring)
            .migration_policy(MigrationPolicy::Best(2))
            .bounds(bounds)
            .elitism(1)
            .fitness(Sphere::new(5))
            .selection(TournamentSelection::new(3))
            .crossover(BlxAlphaCrossover::new(0.5))
            .mutation(GaussianMutation::new(0.1))
            .build(&mut rng);

        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.islands.len(), 4);
    }

    #[test]
    fn test_island_model_evolution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let bounds = MultiBounds::symmetric(5.0, 5);

        let mut model = IslandModelBuilder::<RealVector, _, _, _, _>::new()
            .num_islands(2)
            .island_population_size(20)
            .migration_interval(5)
            .topology(MigrationTopology::Ring)
            .migration_policy(MigrationPolicy::Best(1))
            .bounds(bounds)
            .elitism(1)
            .fitness(Sphere::new(5))
            .selection(TournamentSelection::new(2))
            .crossover(BlxAlphaCrossover::new(0.5))
            .mutation(GaussianMutation::new(0.1))
            .build(&mut rng)
            .unwrap();

        // Run for 10 generations
        let result = model.run(10, &mut rng);
        assert!(result.is_ok());

        assert_eq!(model.generation, 10);
        assert!(model.global_best.is_some());
    }

    #[test]
    fn test_island_statistics() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let bounds = MultiBounds::symmetric(5.0, 5);

        let mut model = IslandModelBuilder::<RealVector, _, _, _, _>::new()
            .num_islands(3)
            .island_population_size(10)
            .migration_interval(10)
            .bounds(bounds)
            .fitness(Sphere::new(5))
            .selection(TournamentSelection::new(2))
            .crossover(BlxAlphaCrossover::new(0.5))
            .mutation(GaussianMutation::new(0.1))
            .build(&mut rng)
            .unwrap();

        model.run(5, &mut rng).unwrap();

        let stats = model.island_statistics();
        assert_eq!(stats.len(), 3);
        for stat in &stats {
            assert_eq!(stat.generation, 5);
            assert_eq!(stat.population_size, 10);
        }
    }

    // regression: EV-12 — two runs with the same master seed must produce
    // identical best-fitness trajectories, even with parallel island evaluation.
    #[test]
    fn test_island_model_reproducible_under_seed() {
        fn trajectory(seed: u64) -> Vec<f64> {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let bounds = MultiBounds::symmetric(5.0, 5);
            let mut model = IslandModelBuilder::<RealVector, _, _, _, _>::new()
                .num_islands(3)
                .island_population_size(20)
                .migration_interval(3)
                .topology(MigrationTopology::Ring)
                .migration_policy(MigrationPolicy::Best(1))
                .bounds(bounds)
                .elitism(1)
                .fitness(Sphere::new(5))
                .selection(TournamentSelection::new(2))
                .crossover(BlxAlphaCrossover::new(0.5))
                .mutation(GaussianMutation::new(0.1))
                .build(&mut rng)
                .unwrap();

            let mut traj = Vec::new();
            for _ in 0..15 {
                model.step(&mut rng).unwrap();
                traj.push(*model.global_best.as_ref().unwrap().fitness_value());
            }
            traj
        }

        let a = trajectory(12345);
        let b = trajectory(12345);
        assert_eq!(a, b, "seeded island runs must be bit-reproducible");

        let c = trajectory(99999);
        assert_ne!(
            a, c,
            "different seeds should produce different trajectories"
        );
    }

    // regression: EV-11 — FullyConnected migration must broadcast each island's
    // emigrant to EVERY other island. With N islands and Best(1) that yields
    // N + N*(N-1) champions in total (N home + N*(N-1) arrivals); the pre-fix code
    // only sent to the first target, giving N + N.
    #[test]
    fn test_fully_connected_broadcasts_to_all_targets() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let bounds = MultiBounds::symmetric(100.0, 1);

        let num_islands = 4;
        let pop = 8;
        let mut model = IslandModelBuilder::<RealVector, _, _, _, _>::new()
            .num_islands(num_islands)
            .island_population_size(pop)
            .topology(MigrationTopology::FullyConnected)
            .migration_policy(MigrationPolicy::Best(1))
            .bounds(bounds)
            .fitness(Sphere::new(1))
            .selection(TournamentSelection::new(2))
            .crossover(BlxAlphaCrossover::new(0.5))
            .mutation(GaussianMutation::new(0.1))
            .build(&mut rng)
            .unwrap();

        // One clearly-best "champion" per island plus clearly-worst filler.
        for (i, island) in model.islands.iter_mut().enumerate() {
            let mut new_pop: Population<RealVector> = Population::new();
            new_pop.push(Individual::with_fitness(
                RealVector::new(vec![i as f64]),
                1000.0,
            ));
            for _ in 1..pop {
                new_pop.push(Individual::with_fitness(
                    RealVector::new(vec![-1.0]),
                    -1000.0,
                ));
            }
            island.population = new_pop;
        }

        model.migrate(&mut rng);

        let champions: usize = model
            .islands
            .iter()
            .flat_map(|isl| isl.population.iter())
            .filter(|ind| (*ind.fitness_value() - 1000.0).abs() < 1e-9)
            .count();

        assert_eq!(champions, num_islands + num_islands * (num_islands - 1));
    }

    // regression: EV-41 — an arriving (worse) migrant must replace a WORST member,
    // never the island best. Pre-fix code replaced a uniformly random member.
    #[test]
    fn test_immigrants_replace_worst_not_best() {
        let mut island_pop: Population<RealVector> = Population::new();
        island_pop.push(Individual::with_fitness(RealVector::new(vec![0.0]), 500.0)); // best
        for _ in 0..9 {
            island_pop.push(Individual::with_fitness(RealVector::new(vec![9.0]), 1.0));
            // worst
        }
        let mut island: Island<RealVector> = Island::with_population(3, island_pop);

        // A migrant worse than the best but better than the worst.
        let migrant = Individual::with_fitness(RealVector::new(vec![7.0]), 50.0);
        island.accept_immigrants(vec![migrant]);

        let best = island
            .population
            .iter()
            .map(|ind| *ind.fitness_value())
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(best, 500.0, "the island best must survive migration");

        let migrant_present = island
            .population
            .iter()
            .any(|ind| (*ind.fitness_value() - 50.0).abs() < 1e-9);
        assert!(migrant_present, "migrant should have been accepted");

        let worst_count = island
            .population
            .iter()
            .filter(|ind| (*ind.fitness_value() - 1.0).abs() < 1e-9)
            .count();
        assert_eq!(
            worst_count, 8,
            "exactly one worst member should be replaced"
        );
    }

    // regression: EV-82 — the evaluation counter must count only genuine fitness
    // calls. From generation 2 on, carried-over elites are already scored, so only
    // (pop - elites) individuals are evaluated, not the whole population.
    #[test]
    fn test_island_evaluation_counter_counts_actual_evaluations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let bounds = MultiBounds::symmetric(5.0, 5);
        let mut island: Island<RealVector> = Island::new(0, 20, &bounds, &mut rng);

        let fitness = Sphere::new(5);
        let selection = TournamentSelection::new(2);
        let crossover = BlxAlphaCrossover::new(0.5);
        let mutation = GaussianMutation::new(0.1);
        let elitism = 2;
        let pop = 20;
        let generations = 5;

        for _ in 0..generations {
            island
                .evolve_one_generation(
                    &fitness, &selection, &crossover, &mutation, elitism, &mut rng,
                )
                .unwrap();
        }

        // gen 1 evaluates all `pop`; gens 2..=G evaluate only (pop - elitism).
        let expected = pop + (generations - 1) * (pop - elitism);
        assert_eq!(island.evaluations, expected);
    }

    // regression: EV-83 — `evolve_one_generation` with `elitism` greater than the
    // population size must not panic. The pre-fix `population.len() - elitism` was
    // an unguarded usize subtraction that underflowed and panicked; the clamp
    // (`elite_count = elitism.min(pop_len)`, `target_offspring = pop_len -
    // elite_count`) makes it carry all members as elites and produce zero
    // offspring, returning a valid same-size population.
    #[test]
    fn test_island_elitism_exceeding_population_does_not_underflow() {
        let mut rng = StdRng::seed_from_u64(99);
        let pop = 20;
        let bounds = MultiBounds::symmetric(5.0, 5);
        let mut island: Island<RealVector> = Island::new(0, pop, &bounds, &mut rng);

        let fitness = Sphere::new(5);
        let selection = TournamentSelection::new(2);
        let crossover = BlxAlphaCrossover::new(0.5);
        let mutation = GaussianMutation::new(0.1);

        // Snapshot the whole population's genomes so we can prove every member is
        // carried over unchanged when everyone is an elite.
        island.population.evaluate(&fitness);
        island.population.sort_by_fitness();
        let before: Vec<RealVector> =
            island.population.iter().map(|ind| ind.genome.clone()).collect();

        // elitism (25) deliberately exceeds pop (20); this panicked pre-fix.
        let elitism = 25;
        let result = island.evolve_one_generation(
            &fitness, &selection, &crossover, &mutation, elitism, &mut rng,
        );
        assert!(
            result.is_ok(),
            "over-large elitism must return Ok, got {result:?}"
        );

        // Population size is preserved: all members carried as elites, zero
        // offspring produced.
        assert_eq!(
            island.population.len(),
            pop,
            "population size must be preserved when everyone is an elite"
        );

        // Every original genome survives (elitism == whole population => no
        // reproduction). Order is preserved because elites are pushed in sorted
        // order and no offspring follow.
        let after: Vec<RealVector> =
            island.population.iter().map(|ind| ind.genome.clone()).collect();
        assert_eq!(
            after.len(),
            before.len(),
            "no offspring should be created when elite_count == pop_len"
        );
        for (a, b) in after.iter().zip(before.iter()) {
            assert_eq!(a.as_vec(), b.as_vec(), "all elites must be carried unchanged");
        }
    }
}
