//! Operator traits
//!
//! This module defines the core operator traits for genetic algorithms.

use rand::Rng;

use crate::error::OperatorResult;
use crate::genome::traits::EvolutionaryGenome;
use crate::genome::bounds::MultiBounds;

/// Selection operator trait
///
/// Selects individuals from a population for reproduction.
pub trait SelectionOperator<G: EvolutionaryGenome>: Send + Sync {
    /// Select a single individual from the population
    ///
    /// Returns the index of the selected individual.
    fn select<R: Rng>(
        &self,
        population: &[(G, f64)], // (genome, fitness) pairs
        rng: &mut R,
    ) -> usize;

    /// Select multiple individuals from the population
    fn select_many<R: Rng>(
        &self,
        population: &[(G, f64)],
        count: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        (0..count).map(|_| self.select(population, rng)).collect()
    }
}

/// Crossover operator trait
///
/// Combines genetic material from two parents to create offspring.
pub trait CrossoverOperator<G: EvolutionaryGenome>: Send + Sync {
    /// Apply crossover to two parents and produce two offspring
    fn crossover<R: Rng>(
        &self,
        parent1: &G,
        parent2: &G,
        rng: &mut R,
    ) -> OperatorResult<(G, G)>;

    /// Get the probability of crossover being applied
    fn crossover_probability(&self) -> f64 {
        1.0
    }
}

/// Mutation operator trait
///
/// Applies random changes to a genome.
pub trait MutationOperator<G: EvolutionaryGenome>: Send + Sync {
    /// Apply mutation to a genome in place
    fn mutate<R: Rng>(&self, genome: &mut G, rng: &mut R);

    /// Get the mutation probability per gene
    fn mutation_probability(&self) -> f64 {
        1.0
    }
}

/// Bounded mutation operator trait
///
/// Mutation operator that respects bounds on gene values.
pub trait BoundedMutationOperator<G: EvolutionaryGenome>: MutationOperator<G> {
    /// Apply bounded mutation to a genome
    fn mutate_bounded<R: Rng>(&self, genome: &mut G, bounds: &MultiBounds, rng: &mut R);
}

/// Bounded crossover operator trait
///
/// Crossover operator that respects bounds on gene values.
pub trait BoundedCrossoverOperator<G: EvolutionaryGenome>: CrossoverOperator<G> {
    /// Apply bounded crossover to two parents
    fn crossover_bounded<R: Rng>(
        &self,
        parent1: &G,
        parent2: &G,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> OperatorResult<(G, G)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::OperatorResult;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::{EvolutionaryGenome, RealValuedGenome};

    // Mock selection operator for testing
    struct MockSelection;

    impl SelectionOperator<RealVector> for MockSelection {
        fn select<R: Rng>(
            &self,
            population: &[(RealVector, f64)],
            rng: &mut R,
        ) -> usize {
            rng.gen_range(0..population.len())
        }
    }

    // Mock crossover operator for testing
    struct MockCrossover;

    impl CrossoverOperator<RealVector> for MockCrossover {
        fn crossover<R: Rng>(
            &self,
            parent1: &RealVector,
            parent2: &RealVector,
            _rng: &mut R,
        ) -> OperatorResult<(RealVector, RealVector)> {
            // Just swap parents as a simple crossover
            OperatorResult::Success((parent2.clone(), parent1.clone()))
        }
    }

    // Mock mutation operator for testing
    struct MockMutation;

    impl MutationOperator<RealVector> for MockMutation {
        fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
            if let Some(genes) = genome.as_mut_slice() {
                for gene in genes.iter_mut() {
                    *gene += rng.gen_range(-0.1..0.1);
                }
            }
        }
    }

    #[test]
    fn test_mock_selection() {
        let mut rng = rand::thread_rng();
        let population: Vec<(RealVector, f64)> = (0..10)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect();

        let selection = MockSelection;
        let idx = selection.select(&population, &mut rng);
        assert!(idx < population.len());
    }

    #[test]
    fn test_mock_selection_many() {
        let mut rng = rand::thread_rng();
        let population: Vec<(RealVector, f64)> = (0..10)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect();

        let selection = MockSelection;
        let indices = selection.select_many(&population, 5, &mut rng);
        assert_eq!(indices.len(), 5);
        for idx in indices {
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_mock_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let parent2 = RealVector::new(vec![4.0, 5.0, 6.0]);

        let crossover = MockCrossover;
        let result = crossover.crossover(&parent1, &parent2, &mut rng);
        assert!(result.is_ok());

        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.genes(), parent2.genes());
        assert_eq!(child2.genes(), parent1.genes());
    }

    #[test]
    fn test_mock_mutation() {
        let mut rng = rand::thread_rng();
        let original = RealVector::new(vec![1.0, 2.0, 3.0]);
        let mut genome = original.clone();

        let mutation = MockMutation;
        mutation.mutate(&mut genome, &mut rng);

        // Genes should have changed
        // (with very high probability, they won't all be exactly the same)
        assert_ne!(genome, original);
    }
}
