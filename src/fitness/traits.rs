//! Fitness traits
//!
//! This module defines the fitness evaluation traits.

use std::fmt::Debug;

use serde::{de::DeserializeOwned, Serialize};

use crate::genome::traits::EvolutionaryGenome;

/// Trait bound for fitness values
///
/// Fitness values must be comparable and convertible to f64 for
/// probabilistic selection operations. They must also be serializable
/// for checkpointing.
pub trait FitnessValue:
    PartialOrd + Clone + Send + Sync + Debug + Serialize + DeserializeOwned + 'static
{
    /// Convert fitness to f64 for probabilistic operations
    fn to_f64(&self) -> f64;

    /// Check if this fitness is better than another
    fn is_better_than(&self, other: &Self) -> bool;

    /// Check if this fitness is worse than another
    fn is_worse_than(&self, other: &Self) -> bool {
        other.is_better_than(self)
    }
}

impl FitnessValue for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl FitnessValue for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl FitnessValue for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl FitnessValue for i32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl FitnessValue for usize {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self > other
    }
}

/// Multi-objective fitness value using Pareto ranking
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct ParetoFitness {
    /// Objective values (all to be maximized)
    pub objectives: Vec<f64>,
    /// Pareto rank (0 = non-dominated front)
    pub rank: usize,
    /// Crowding distance for diversity preservation
    pub crowding_distance: f64,
}

impl ParetoFitness {
    /// Create a new Pareto fitness with the given objectives
    pub fn new(objectives: Vec<f64>) -> Self {
        Self {
            objectives,
            rank: usize::MAX,
            crowding_distance: 0.0,
        }
    }

    /// Check if this solution dominates another
    /// (all objectives >= and at least one >)
    pub fn dominates(&self, other: &Self) -> bool {
        let dominated = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .all(|(a, b)| a >= b);
        let strictly_better = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .any(|(a, b)| a > b);
        dominated && strictly_better
    }

    /// Number of objectives
    pub fn num_objectives(&self) -> usize {
        self.objectives.len()
    }
}

impl PartialOrd for ParetoFitness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Compare by rank first, then by crowding distance
        match self.rank.partial_cmp(&other.rank) {
            Some(std::cmp::Ordering::Equal) => {
                // Higher crowding distance is better (more diverse)
                self.crowding_distance.partial_cmp(&other.crowding_distance)
            }
            ord => ord.map(|o| o.reverse()), // Reverse because lower rank is better
        }
    }
}

impl FitnessValue for ParetoFitness {
    fn to_f64(&self) -> f64 {
        // Aggregated scalar for probabilistic interpretation
        // Lower rank is better, so negate it
        -(self.rank as f64) + self.crowding_distance * 0.001
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self.rank < other.rank
            || (self.rank == other.rank && self.crowding_distance > other.crowding_distance)
    }
}

/// Fitness evaluation trait
///
/// Defines how to evaluate the fitness of a genome.
#[cfg(feature = "parallel")]
pub trait Fitness: Send + Sync {
    /// The genome type being evaluated
    type Genome: EvolutionaryGenome;

    /// The fitness value type
    type Value: FitnessValue;

    /// Evaluate fitness (higher = better by convention)
    fn evaluate(&self, genome: &Self::Genome) -> Self::Value;

    /// Convert fitness to log-likelihood for probabilistic selection
    ///
    /// Uses Boltzmann distribution: P(x) ∝ exp(f(x) / T)
    fn as_log_likelihood(&self, genome: &Self::Genome, temperature: f64) -> f64 {
        let fitness = self.evaluate(genome).to_f64();
        fitness / temperature
    }

    /// Optional: Provide gradient for gradient-assisted mutation
    fn gradient(&self, _genome: &Self::Genome) -> Option<Vec<f64>> {
        None
    }
}

/// Fitness evaluation trait (non-parallel version)
///
/// Defines how to evaluate the fitness of a genome.
#[cfg(not(feature = "parallel"))]
pub trait Fitness {
    /// The genome type being evaluated
    type Genome: EvolutionaryGenome;

    /// The fitness value type
    type Value: FitnessValue;

    /// Evaluate fitness (higher = better by convention)
    fn evaluate(&self, genome: &Self::Genome) -> Self::Value;

    /// Convert fitness to log-likelihood for probabilistic selection
    ///
    /// Uses Boltzmann distribution: P(x) ∝ exp(f(x) / T)
    fn as_log_likelihood(&self, genome: &Self::Genome, temperature: f64) -> f64 {
        let fitness = self.evaluate(genome).to_f64();
        fitness / temperature
    }

    /// Optional: Provide gradient for gradient-assisted mutation
    fn gradient(&self, _genome: &Self::Genome) -> Option<Vec<f64>> {
        None
    }
}

/// A wrapper to negate a fitness function (for minimization problems)
pub struct MinimizeFitness<F> {
    inner: F,
}

impl<F> MinimizeFitness<F> {
    /// Create a minimization wrapper around a fitness function
    pub fn new(fitness: F) -> Self {
        Self { inner: fitness }
    }
}

impl<F: Fitness<Value = f64>> Fitness for MinimizeFitness<F> {
    type Genome = F::Genome;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.inner.evaluate(genome)
    }
}

/// A simple function wrapper for fitness evaluation
pub struct FnFitness<G, F, V>
where
    F: Fn(&G) -> V,
{
    f: F,
    _marker: std::marker::PhantomData<(G, V)>,
}

impl<G, F, V> FnFitness<G, F, V>
where
    F: Fn(&G) -> V,
{
    /// Create a new function-based fitness evaluator
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G, F, V> Fitness for FnFitness<G, F, V>
where
    G: EvolutionaryGenome,
    F: Fn(&G) -> V + Send + Sync,
    V: FitnessValue,
{
    type Genome = G;
    type Value = V;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        (self.f)(genome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;

    #[test]
    fn test_f64_fitness_value() {
        let a: f64 = 10.0;
        let b: f64 = 5.0;

        assert!(a.is_better_than(&b));
        assert!(!b.is_better_than(&a));
        assert!(b.is_worse_than(&a));
        assert_eq!(a.to_f64(), 10.0);
    }

    #[test]
    fn test_i32_fitness_value() {
        let a: i32 = 10;
        let b: i32 = 5;

        assert!(a.is_better_than(&b));
        assert!(!b.is_better_than(&a));
        assert_eq!(a.to_f64(), 10.0);
    }

    #[test]
    fn test_usize_fitness_value() {
        let a: usize = 10;
        let b: usize = 5;

        assert!(a.is_better_than(&b));
        assert!(!b.is_better_than(&a));
        assert_eq!(a.to_f64(), 10.0);
    }

    #[test]
    fn test_pareto_fitness_dominates() {
        let a = ParetoFitness::new(vec![5.0, 5.0]);
        let b = ParetoFitness::new(vec![3.0, 3.0]);
        let c = ParetoFitness::new(vec![6.0, 3.0]); // Better in one, worse in other - not dominated by a

        assert!(a.dominates(&b)); // a is better in all objectives
        assert!(!b.dominates(&a)); // b is worse in all objectives
        assert!(!a.dominates(&c)); // c is better in first objective, so not dominated
        assert!(!c.dominates(&a)); // a is better in second objective, so c doesn't dominate a
    }

    #[test]
    fn test_pareto_fitness_is_better_than() {
        let mut a = ParetoFitness::new(vec![5.0, 5.0]);
        a.rank = 0;
        a.crowding_distance = 1.0;

        let mut b = ParetoFitness::new(vec![3.0, 3.0]);
        b.rank = 1;
        b.crowding_distance = 2.0;

        assert!(a.is_better_than(&b)); // Lower rank is better

        let mut c = ParetoFitness::new(vec![4.0, 4.0]);
        c.rank = 0;
        c.crowding_distance = 0.5;

        assert!(a.is_better_than(&c)); // Same rank, higher crowding distance
    }

    #[test]
    fn test_fn_fitness() {
        let fitness = FnFitness::new(|g: &RealVector| -> f64 {
            -g.genes().iter().map(|x| x * x).sum::<f64>()
        });

        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let value = fitness.evaluate(&genome);
        assert_eq!(value, -14.0);
    }

    #[test]
    fn test_minimize_fitness() {
        let fitness = FnFitness::new(|g: &RealVector| -> f64 {
            g.genes().iter().map(|x| x * x).sum::<f64>()
        });
        let minimize = MinimizeFitness::new(fitness);

        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let value = minimize.evaluate(&genome);
        assert_eq!(value, -14.0);
    }

    #[test]
    fn test_as_log_likelihood() {
        let fitness = FnFitness::new(|g: &RealVector| -> f64 {
            -g.genes().iter().map(|x| x * x).sum::<f64>()
        });

        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let log_likelihood = fitness.as_log_likelihood(&genome, 1.0);
        assert_eq!(log_likelihood, -14.0);

        let log_likelihood_scaled = fitness.as_log_likelihood(&genome, 2.0);
        assert_eq!(log_likelihood_scaled, -7.0);
    }
}
