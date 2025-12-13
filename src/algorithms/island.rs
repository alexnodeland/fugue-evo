//! Island Model Parallelism
//!
//! Implements a distributed evolutionary algorithm where multiple populations
//! (islands) evolve independently with periodic migration of individuals.
//!
//! Note: This module requires the `parallel` feature to be enabled.

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
        // Evaluate population
        self.population.evaluate(fitness);
        self.evaluations += self.population.len();

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

        // Preserve elites
        self.population.sort_by_fitness();
        let elites: Vec<_> = self.population.iter().take(elitism).cloned().collect();

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
        let mut offspring = Vec::with_capacity(self.population.len());

        while offspring.len() < self.population.len() - elitism {
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
            if offspring.len() < self.population.len() - elitism {
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

    /// Accept immigrants (individuals coming to this island)
    pub fn accept_immigrants<R: rand::Rng>(
        &mut self,
        immigrants: Vec<Individual<G, F>>,
        policy: &MigrationPolicy,
        rng: &mut R,
    ) {
        match policy {
            MigrationPolicy::Best(_) | MigrationPolicy::Random(_) => {
                // Replace random individuals
                for immigrant in immigrants {
                    let idx = rng.gen_range(0..self.population.len());
                    self.population[idx] = immigrant;
                }
            }
            MigrationPolicy::BestReplaceWorst(k) => {
                // Replace worst individuals
                self.population.sort_by_fitness();
                let pop_len = self.population.len();
                for (i, immigrant) in immigrants.into_iter().enumerate() {
                    if i < *k && pop_len > i {
                        self.population[pop_len - 1 - i] = immigrant;
                    }
                }
            }
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
                let mut island_rng = rand::rngs::StdRng::from_seed(rng.gen());
                Island::new(
                    i,
                    config.island_population_size,
                    &config.bounds,
                    &mut island_rng,
                )
            })
            .collect();

        Self {
            config,
            islands,
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

        // Parallel evolution of islands
        self.islands.par_iter_mut().for_each(|island| {
            let mut island_rng = rand::rngs::StdRng::from_entropy();
            let _ = island.evolve_one_generation(
                fitness.as_ref(),
                &selection,
                &crossover,
                &mutation,
                elitism,
                &mut island_rng,
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
        let policy = &self.config.policy;
        let topology = &self.config.topology;

        // Collect emigrants from each island
        let emigrants: Vec<Vec<Individual<G, F>>> = self
            .islands
            .iter()
            .map(|island| island.get_emigrants(policy, &mut rand::rngs::StdRng::from_entropy()))
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
                    self.islands[target].accept_immigrants(source_emigrants, policy, rng);
                }
                _ => {
                    // Send to first target
                    if let Some(&target) = targets.first() {
                        self.islands[target].accept_immigrants(
                            source_emigrants.clone(),
                            policy,
                            rng,
                        );
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
}
