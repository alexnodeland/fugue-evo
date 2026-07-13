//! Population type
//!
//! This module provides the Population container type.

use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::fitness::traits::{Fitness, FitnessValue};
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;
use crate::population::individual::Individual;

/// A population of individuals
#[derive(Clone, Debug)]
pub struct Population<G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// The individuals in this population
    individuals: Vec<Individual<G, F>>,
    /// Current generation number
    generation: usize,
}

impl<G, F> Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create an empty population
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
            generation: 0,
        }
    }

    /// Create a population with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            individuals: Vec::with_capacity(capacity),
            generation: 0,
        }
    }

    /// Create a population from a vector of individuals
    pub fn from_individuals(individuals: Vec<Individual<G, F>>) -> Self {
        Self {
            individuals,
            generation: 0,
        }
    }

    /// Create a random population
    pub fn random<R: Rng>(size: usize, bounds: &MultiBounds, rng: &mut R) -> Self {
        let individuals = (0..size)
            .map(|_| Individual::new(G::generate(rng, bounds)))
            .collect();
        Self {
            individuals,
            generation: 0,
        }
    }

    /// Get the current generation
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Increment the generation counter
    pub fn increment_generation(&mut self) {
        self.generation += 1;
    }

    /// Set the generation number
    pub fn set_generation(&mut self, generation: usize) {
        self.generation = generation;
    }

    /// Get the population size
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Check if the population is empty
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Get an individual by index
    pub fn get(&self, index: usize) -> Option<&Individual<G, F>> {
        self.individuals.get(index)
    }

    /// Get a mutable reference to an individual by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Individual<G, F>> {
        self.individuals.get_mut(index)
    }

    /// Add an individual to the population
    pub fn push(&mut self, individual: Individual<G, F>) {
        self.individuals.push(individual);
    }

    /// Remove and return the last individual
    pub fn pop(&mut self) -> Option<Individual<G, F>> {
        self.individuals.pop()
    }

    /// Clear the population
    pub fn clear(&mut self) {
        self.individuals.clear();
    }

    /// Get an iterator over the individuals
    pub fn iter(&self) -> impl Iterator<Item = &Individual<G, F>> {
        self.individuals.iter()
    }

    /// Get a mutable iterator over the individuals
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Individual<G, F>> {
        self.individuals.iter_mut()
    }

    /// Get the underlying vector of individuals
    pub fn individuals(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    /// Get mutable access to the underlying vector
    pub fn individuals_mut(&mut self) -> &mut Vec<Individual<G, F>> {
        &mut self.individuals
    }

    /// Take the individuals out of this population
    pub fn into_individuals(self) -> Vec<Individual<G, F>> {
        self.individuals
    }

    /// Get the best individual (by fitness)
    ///
    /// Ranks by [`FitnessValue::cmp_by_quality`] (which delegates to
    /// `is_better_than` and treats any `NaN` fitness as strictly worst), so it
    /// returns the true best even for fitness types whose `to_f64()` ordering
    /// disagrees with the real ordering (e.g. `ParetoFitness` with infinite
    /// crowding distances) and never returns a `NaN`-fitness individual unless
    /// every evaluated individual is `NaN`.
    pub fn best(&self) -> Option<&Individual<G, F>> {
        self.individuals
            .iter()
            .filter(|i| i.is_evaluated())
            .max_by(|a, b| {
                a.fitness
                    .as_ref()
                    .expect("filtered to evaluated individuals")
                    .cmp_by_quality(
                        b.fitness
                            .as_ref()
                            .expect("filtered to evaluated individuals"),
                    )
            })
    }

    /// Get the worst individual (by fitness)
    ///
    /// Uses the same [`FitnessValue::cmp_by_quality`] total order as
    /// [`best`](Self::best); a `NaN` fitness is ranked strictly worst.
    pub fn worst(&self) -> Option<&Individual<G, F>> {
        self.individuals
            .iter()
            .filter(|i| i.is_evaluated())
            .min_by(|a, b| {
                a.fitness
                    .as_ref()
                    .expect("filtered to evaluated individuals")
                    .cmp_by_quality(
                        b.fitness
                            .as_ref()
                            .expect("filtered to evaluated individuals"),
                    )
            })
    }

    /// Sort the population by fitness (best first)
    ///
    /// Ranks by [`FitnessValue::cmp_by_quality`] rather than a `to_f64()`
    /// scalar. Unevaluated individuals (and, defensively, `NaN` fitness) sort
    /// last.
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| {
            match (a.fitness.as_ref(), b.fitness.as_ref()) {
                // Better first => descending by quality.
                (Some(fa), Some(fb)) => fb.cmp_by_quality(fa),
                // Evaluated individuals precede unevaluated ones.
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });
    }

    /// Truncate the population to the given size, keeping the best individuals
    pub fn truncate_to_best(&mut self, size: usize) {
        self.sort_by_fitness();
        self.individuals.truncate(size);
    }

    /// Check if all individuals have been evaluated
    pub fn all_evaluated(&self) -> bool {
        self.individuals.iter().all(|i| i.is_evaluated())
    }

    /// Count the number of evaluated individuals
    pub fn count_evaluated(&self) -> usize {
        self.individuals.iter().filter(|i| i.is_evaluated()).count()
    }

    /// Get genome-fitness pairs for selection
    pub fn as_selection_pool(&self) -> Vec<(&G, f64)> {
        self.individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref().map(|f| (&i.genome, f.to_f64())))
            .collect()
    }

    /// Get genome-fitness pairs as owned tuples
    pub fn as_fitness_pairs(&self) -> Vec<(G, f64)>
    where
        G: Clone,
    {
        self.individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref().map(|f| (i.genome.clone(), f.to_f64())))
            .collect()
    }

    /// Evaluate all individuals using the given fitness function (sequential)
    pub fn evaluate<Fit>(&mut self, fitness: &Fit)
    where
        Fit: Fitness<Genome = G, Value = F>,
    {
        for individual in &mut self.individuals {
            if !individual.is_evaluated() {
                let f = fitness.evaluate(&individual.genome);
                individual.set_fitness(f);
            }
        }
    }

    /// Compute mean fitness
    pub fn mean_fitness(&self) -> Option<f64> {
        let evaluated: Vec<f64> = self
            .individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref().map(|f| f.to_f64()))
            .collect();

        if evaluated.is_empty() {
            None
        } else {
            Some(evaluated.iter().sum::<f64>() / evaluated.len() as f64)
        }
    }

    /// Compute fitness standard deviation
    pub fn fitness_std(&self) -> Option<f64> {
        let mean = self.mean_fitness()?;
        let evaluated: Vec<f64> = self
            .individuals
            .iter()
            .filter_map(|i| i.fitness.as_ref().map(|f| f.to_f64()))
            .collect();

        if evaluated.len() < 2 {
            return None;
        }

        let variance = evaluated.iter().map(|f| (f - mean).powi(2)).sum::<f64>()
            / (evaluated.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Compute population diversity (average pairwise distance)
    pub fn diversity(&self) -> f64 {
        if self.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..self.len() {
            for j in (i + 1)..self.len() {
                total_distance += self.individuals[i]
                    .genome
                    .distance(&self.individuals[j].genome);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_distance / count as f64
        }
    }
}

/// Parallel evaluation support (requires `parallel` feature)
#[cfg(feature = "parallel")]
impl<G, F> Population<G, F>
where
    G: EvolutionaryGenome + Send + Sync,
    F: FitnessValue + Send,
{
    /// Evaluate all individuals using the given fitness function (parallel)
    pub fn evaluate_parallel<Fit>(&mut self, fitness: &Fit)
    where
        Fit: Fitness<Genome = G, Value = F> + Sync,
    {
        self.individuals
            .par_iter_mut()
            .filter(|i| !i.is_evaluated())
            .for_each(|individual| {
                let f = fitness.evaluate(&individual.genome);
                individual.set_fitness(f);
            });
    }
}

/// Sequential fallback for parallel evaluation (when `parallel` feature is disabled)
#[cfg(not(feature = "parallel"))]
impl<G, F> Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Evaluate all individuals using the given fitness function (sequential fallback)
    ///
    /// Note: This is a sequential implementation used when the `parallel` feature is disabled.
    pub fn evaluate_parallel<Fit>(&mut self, fitness: &Fit)
    where
        Fit: Fitness<Genome = G, Value = F>,
    {
        self.evaluate(fitness);
    }
}

impl<G, F> Default for Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, F> std::ops::Index<usize> for Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    type Output = Individual<G, F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.individuals[index]
    }
}

impl<G, F> std::ops::IndexMut<usize> for Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.individuals[index]
    }
}

impl<G, F> IntoIterator for Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    type Item = Individual<G, F>;
    type IntoIter = std::vec::IntoIter<Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.into_iter()
    }
}

impl<G, F> FromIterator<Individual<G, F>> for Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    fn from_iter<I: IntoIterator<Item = Individual<G, F>>>(iter: I) -> Self {
        Self::from_individuals(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::fitness::traits::ParetoFitness;
    use crate::genome::real_vector::RealVector;

    fn create_test_population() -> Population<RealVector> {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
            Individual::with_fitness(RealVector::new(vec![2.0]), 20.0),
            Individual::with_fitness(RealVector::new(vec![3.0]), 30.0),
            Individual::with_fitness(RealVector::new(vec![4.0]), 40.0),
            Individual::with_fitness(RealVector::new(vec![5.0]), 50.0),
        ];
        Population::from_individuals(individuals)
    }

    #[test]
    fn test_population_new() {
        let pop: Population<RealVector> = Population::new();
        assert!(pop.is_empty());
        assert_eq!(pop.generation(), 0);
    }

    #[test]
    fn test_population_random() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 3);
        let pop: Population<RealVector> = Population::random(10, &bounds, &mut rng);

        assert_eq!(pop.len(), 10);
        assert!(!pop.all_evaluated());
    }

    #[test]
    fn test_population_best_worst() {
        let pop = create_test_population();

        let best = pop.best().unwrap();
        assert_eq!(best.fitness_f64(), 50.0);

        let worst = pop.worst().unwrap();
        assert_eq!(worst.fitness_f64(), 10.0);
    }

    #[test]
    fn test_best_ignores_nan_fitness() {
        // regression: EV-07
        // A NaN fitness must never be reported as the best individual. It is
        // ranked strictly worst, so best() returns the true maximum even when
        // the NaN individual is the LAST element -- the position that used to
        // fool max_by (NaN comparisons returned Equal, treated as "replace").
        let mut individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
            Individual::with_fitness(RealVector::new(vec![3.0]), 30.0),
            Individual::with_fitness(RealVector::new(vec![2.0]), 20.0),
        ];
        // Inject NaN through the public field to model a value slipping past
        // set_fitness's guard (defense in depth), placed last.
        let mut nan_ind = Individual::new(RealVector::new(vec![9.0]));
        nan_ind.fitness = Some(f64::NAN);
        individuals.push(nan_ind);

        let pop = Population::from_individuals(individuals);
        assert_eq!(pop.best().unwrap().fitness_f64(), 30.0);
        // worst() over finite values (NaN is ranked worst and would be picked,
        // but here we confirm the true minimum among the reals is reachable by
        // dropping the NaN): with the NaN present it is the worst.
        assert!(pop.worst().unwrap().fitness_f64().is_nan());
    }

    #[test]
    fn test_best_worst_sort_use_is_better_than_for_pareto() {
        // regression: EV-08
        // ParetoFitness ranks by (rank asc, crowding desc). NSGA-II assigns
        // f64::INFINITY crowding to front-boundary members, so to_f64() =
        // -(rank) + inf*0.001 collapses to +inf for EVERY rank. A to_f64()-based
        // best() therefore cannot distinguish a rank-0 optimum from a rank-50
        // solution; best()/worst()/sort_by_fitness() must delegate to
        // is_better_than().
        let mut good = ParetoFitness::new(vec![0.0, 0.0]);
        good.rank = 0;
        good.crowding_distance = f64::INFINITY;

        let mut bad = ParetoFitness::new(vec![9.0, 9.0]);
        bad.rank = 50;
        bad.crowding_distance = f64::INFINITY;

        assert_eq!(good.to_f64(), bad.to_f64()); // both +inf: to_f64() cannot rank them

        // `bad` placed last so a to_f64()/Equal tie would wrongly "take later".
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![0.0]), good.clone()),
            Individual::with_fitness(RealVector::new(vec![1.0]), bad.clone()),
        ];
        let mut pop: Population<RealVector, ParetoFitness> =
            Population::from_individuals(individuals);

        assert_eq!(pop.best().unwrap().fitness.as_ref().unwrap().rank, 0);
        assert_eq!(pop.worst().unwrap().fitness.as_ref().unwrap().rank, 50);

        pop.sort_by_fitness();
        assert_eq!(pop[0].fitness.as_ref().unwrap().rank, 0); // best first
        assert_eq!(pop[1].fitness.as_ref().unwrap().rank, 50);
    }

    #[test]
    fn test_population_sort_by_fitness() {
        let mut pop = create_test_population();
        pop.sort_by_fitness();

        let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness_f64()).collect();
        assert_eq!(fitnesses, vec![50.0, 40.0, 30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_population_truncate_to_best() {
        let mut pop = create_test_population();
        pop.truncate_to_best(3);

        assert_eq!(pop.len(), 3);
        let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness_f64()).collect();
        assert_eq!(fitnesses, vec![50.0, 40.0, 30.0]);
    }

    #[test]
    fn test_population_mean_fitness() {
        let pop = create_test_population();
        let mean = pop.mean_fitness().unwrap();
        assert_eq!(mean, 30.0); // (10 + 20 + 30 + 40 + 50) / 5
    }

    #[test]
    fn test_population_fitness_std() {
        let pop = create_test_population();
        let std = pop.fitness_std().unwrap();
        // Variance = ((10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2) / 4
        // = (400 + 100 + 0 + 100 + 400) / 4 = 250
        // Std = sqrt(250) ≈ 15.81
        assert!((std - 15.81).abs() < 0.1);
    }

    #[test]
    fn test_population_evaluate() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 3);
        let mut pop: Population<RealVector> = Population::random(5, &bounds, &mut rng);

        let fitness = Sphere::new(3);
        pop.evaluate(&fitness);

        assert!(pop.all_evaluated());
        assert_eq!(pop.count_evaluated(), 5);
    }

    #[test]
    fn test_population_evaluate_parallel() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 3);
        let mut pop: Population<RealVector> = Population::random(100, &bounds, &mut rng);

        let fitness = Sphere::new(3);
        pop.evaluate_parallel(&fitness);

        assert!(pop.all_evaluated());
        assert_eq!(pop.count_evaluated(), 100);
    }

    #[test]
    fn test_population_generation() {
        let mut pop = create_test_population();
        assert_eq!(pop.generation(), 0);

        pop.increment_generation();
        assert_eq!(pop.generation(), 1);

        pop.set_generation(100);
        assert_eq!(pop.generation(), 100);
    }

    #[test]
    fn test_population_push_pop() {
        let mut pop: Population<RealVector> = Population::new();

        pop.push(Individual::with_fitness(RealVector::new(vec![1.0]), 10.0));
        assert_eq!(pop.len(), 1);

        let ind = pop.pop().unwrap();
        assert_eq!(ind.fitness_f64(), 10.0);
        assert!(pop.is_empty());
    }

    #[test]
    fn test_population_indexing() {
        let pop = create_test_population();
        assert_eq!(pop[0].fitness_f64(), 10.0);
        assert_eq!(pop[4].fitness_f64(), 50.0);
    }

    #[test]
    fn test_population_as_selection_pool() {
        let pop = create_test_population();
        let pool = pop.as_selection_pool();

        assert_eq!(pool.len(), 5);
        assert_eq!(pool[0].1, 10.0);
        assert_eq!(pool[4].1, 50.0);
    }

    #[test]
    fn test_population_diversity() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![0.0, 0.0]), 1.0),
            Individual::with_fitness(RealVector::new(vec![1.0, 0.0]), 1.0),
            Individual::with_fitness(RealVector::new(vec![0.0, 1.0]), 1.0),
        ];
        let pop = Population::from_individuals(individuals);

        let diversity = pop.diversity();
        // Average of distances: (1, 1, sqrt(2)) / 3 ≈ 1.14
        assert!(diversity > 1.0 && diversity < 1.2);
    }

    #[test]
    fn test_population_from_iterator() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
            Individual::with_fitness(RealVector::new(vec![2.0]), 20.0),
        ];
        let pop: Population<RealVector> = individuals.into_iter().collect();

        assert_eq!(pop.len(), 2);
    }

    #[test]
    fn test_population_into_iterator() {
        let pop = create_test_population();
        let individuals: Vec<_> = pop.into_iter().collect();

        assert_eq!(individuals.len(), 5);
    }
}
