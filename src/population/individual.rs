//! Individual wrapper type
//!
//! This module provides the Individual type that wraps a genome with its fitness.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::fitness::traits::FitnessValue;
use crate::genome::traits::EvolutionaryGenome;

/// An individual in the population
///
/// Wraps a genome with its computed fitness value and additional metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Individual<G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// The genome of this individual
    pub genome: G,
    /// The fitness value (None if not yet evaluated)
    pub fitness: Option<F>,
    /// Generation when this individual was created
    pub birth_generation: usize,
    /// Number of offspring produced by this individual
    pub offspring_count: usize,
}

impl<G, F> Individual<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new individual with an unevaluated genome
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: None,
            birth_generation: 0,
            offspring_count: 0,
        }
    }

    /// Create a new individual with a known fitness
    pub fn with_fitness(genome: G, fitness: F) -> Self {
        Self {
            genome,
            fitness: Some(fitness),
            birth_generation: 0,
            offspring_count: 0,
        }
    }

    /// Create a new individual with birth generation
    pub fn with_generation(genome: G, generation: usize) -> Self {
        Self {
            genome,
            fitness: None,
            birth_generation: generation,
            offspring_count: 0,
        }
    }

    /// Check if this individual has been evaluated
    pub fn is_evaluated(&self) -> bool {
        self.fitness.is_some()
    }

    /// Get the fitness value, panicking if not evaluated
    pub fn fitness_value(&self) -> &F {
        self.fitness
            .as_ref()
            .expect("Individual has not been evaluated")
    }

    /// Get the fitness as f64
    pub fn fitness_f64(&self) -> f64 {
        self.fitness_value().to_f64()
    }

    /// Set the fitness value
    pub fn set_fitness(&mut self, fitness: F) {
        self.fitness = Some(fitness);
    }

    /// Take the genome out of this individual
    pub fn into_genome(self) -> G {
        self.genome
    }

    /// Get a reference to the genome
    pub fn genome(&self) -> &G {
        &self.genome
    }

    /// Get a mutable reference to the genome
    pub fn genome_mut(&mut self) -> &mut G {
        &mut self.genome
    }

    /// Check if this individual is better than another
    pub fn is_better_than(&self, other: &Self) -> bool {
        match (&self.fitness, &other.fitness) {
            (Some(f1), Some(f2)) => f1.is_better_than(f2),
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => false,
        }
    }

    /// Age of this individual (generations since birth)
    pub fn age(&self, current_generation: usize) -> usize {
        current_generation.saturating_sub(self.birth_generation)
    }
}

impl<G, F> PartialEq for Individual<G, F>
where
    G: EvolutionaryGenome + PartialEq,
    F: FitnessValue + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.genome == other.genome && self.fitness == other.fitness
    }
}

impl<G, F> PartialOrd for Individual<G, F>
where
    G: EvolutionaryGenome + PartialEq,
    F: FitnessValue + PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (&self.fitness, &other.fitness) {
            (Some(f1), Some(f2)) => f1.partial_cmp(f2),
            (Some(_), None) => Some(Ordering::Greater),
            (None, Some(_)) => Some(Ordering::Less),
            (None, None) => Some(Ordering::Equal),
        }
    }
}

/// A pair of individuals (for crossover results)
pub type IndividualPair<G, F = f64> = (Individual<G, F>, Individual<G, F>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;

    #[test]
    fn test_individual_new() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let individual: Individual<RealVector> = Individual::new(genome);

        assert!(!individual.is_evaluated());
        assert_eq!(individual.birth_generation, 0);
        assert_eq!(individual.offspring_count, 0);
    }

    #[test]
    fn test_individual_with_fitness() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let individual = Individual::with_fitness(genome, 42.0);

        assert!(individual.is_evaluated());
        assert_eq!(individual.fitness_f64(), 42.0);
    }

    #[test]
    fn test_individual_set_fitness() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let mut individual: Individual<RealVector> = Individual::new(genome);

        assert!(!individual.is_evaluated());
        individual.set_fitness(100.0);
        assert!(individual.is_evaluated());
        assert_eq!(individual.fitness_f64(), 100.0);
    }

    #[test]
    fn test_individual_is_better_than() {
        let g1 = RealVector::new(vec![1.0]);
        let g2 = RealVector::new(vec![2.0]);

        let ind1 = Individual::with_fitness(g1, 100.0);
        let ind2 = Individual::with_fitness(g2, 50.0);

        assert!(ind1.is_better_than(&ind2));
        assert!(!ind2.is_better_than(&ind1));
    }

    #[test]
    fn test_individual_is_better_than_unevaluated() {
        let g1 = RealVector::new(vec![1.0]);
        let g2 = RealVector::new(vec![2.0]);

        let ind1 = Individual::with_fitness(g1, 100.0);
        let ind2: Individual<RealVector> = Individual::new(g2);

        assert!(ind1.is_better_than(&ind2));
        assert!(!ind2.is_better_than(&ind1));
    }

    #[test]
    fn test_individual_age() {
        let genome = RealVector::new(vec![1.0]);
        let individual: Individual<RealVector> = Individual::with_generation(genome, 10);

        assert_eq!(individual.age(10), 0);
        assert_eq!(individual.age(15), 5);
        assert_eq!(individual.age(5), 0); // saturating sub
    }

    #[test]
    fn test_individual_partial_ord() {
        let g1 = RealVector::new(vec![1.0]);
        let g2 = RealVector::new(vec![2.0]);

        let ind1 = Individual::with_fitness(g1, 100.0);
        let ind2 = Individual::with_fitness(g2, 50.0);

        assert!(ind1 > ind2);
        assert!(ind2 < ind1);
    }

    #[test]
    fn test_individual_into_genome() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let individual = Individual::with_fitness(genome.clone(), 42.0);

        let recovered = individual.into_genome();
        assert_eq!(recovered, genome);
    }

    #[test]
    fn test_individual_genome_mut() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let mut individual: Individual<RealVector> = Individual::new(genome);

        individual.genome_mut().genes_mut()[0] = 100.0;
        assert_eq!(individual.genome()[0], 100.0);
    }
}
