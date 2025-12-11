//! Termination criteria
//!
//! This module provides various termination criteria for evolutionary algorithms.

use crate::fitness::traits::FitnessValue;
use crate::genome::traits::EvolutionaryGenome;
use crate::population::population::Population;

/// Evolution state for termination checking
#[derive(Clone, Debug)]
pub struct EvolutionState<'a, G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Current generation number
    pub generation: usize,
    /// Total fitness evaluations so far
    pub evaluations: usize,
    /// Best fitness found so far
    pub best_fitness: f64,
    /// Reference to the current population
    pub population: &'a Population<G, F>,
    /// History of best fitness values per generation
    pub fitness_history: &'a [f64],
}

/// Termination criterion trait
pub trait TerminationCriterion<G: EvolutionaryGenome, F: FitnessValue = f64>: Send + Sync {
    /// Check if evolution should terminate
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool;

    /// Get a description of why termination occurred
    fn reason(&self) -> &'static str;
}

/// Terminate after a maximum number of generations
#[derive(Clone, Debug)]
pub struct MaxGenerations(pub usize);

impl MaxGenerations {
    /// Create a new max generations criterion
    pub fn new(max: usize) -> Self {
        Self(max)
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for MaxGenerations {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        state.generation >= self.0
    }

    fn reason(&self) -> &'static str {
        "Maximum generations reached"
    }
}

/// Terminate after a maximum number of fitness evaluations
#[derive(Clone, Debug)]
pub struct MaxEvaluations(pub usize);

impl MaxEvaluations {
    /// Create a new max evaluations criterion
    pub fn new(max: usize) -> Self {
        Self(max)
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for MaxEvaluations {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        state.evaluations >= self.0
    }

    fn reason(&self) -> &'static str {
        "Maximum evaluations reached"
    }
}

/// Terminate when fitness improvement stagnates
#[derive(Clone, Debug)]
pub struct FitnessStagnation {
    /// Number of generations to look back
    pub window: usize,
    /// Minimum improvement threshold
    pub epsilon: f64,
}

impl FitnessStagnation {
    /// Create a new fitness stagnation criterion
    pub fn new(window: usize, epsilon: f64) -> Self {
        Self { window, epsilon }
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for FitnessStagnation {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        if state.fitness_history.len() < self.window {
            return false;
        }

        let start_idx = state.fitness_history.len() - self.window;
        let window = &state.fitness_history[start_idx..];

        if window.is_empty() {
            return false;
        }

        let first = window[0];
        let last = window[window.len() - 1];
        let improvement = (last - first).abs();

        improvement < self.epsilon
    }

    fn reason(&self) -> &'static str {
        "Fitness stagnation detected"
    }
}

/// Terminate when target fitness is reached
#[derive(Clone, Debug)]
pub struct TargetFitness {
    /// Target fitness value
    pub target: f64,
    /// Tolerance for reaching target
    pub tolerance: f64,
}

impl TargetFitness {
    /// Create a new target fitness criterion
    pub fn new(target: f64) -> Self {
        Self {
            target,
            tolerance: 0.0,
        }
    }

    /// Create with a tolerance
    pub fn with_tolerance(target: f64, tolerance: f64) -> Self {
        Self { target, tolerance }
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for TargetFitness {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        state.best_fitness >= self.target - self.tolerance
    }

    fn reason(&self) -> &'static str {
        "Target fitness reached"
    }
}

/// Terminate when population diversity drops below threshold
#[derive(Clone, Debug)]
pub struct DiversityThreshold {
    /// Minimum diversity threshold
    pub min_diversity: f64,
}

impl DiversityThreshold {
    /// Create a new diversity threshold criterion
    pub fn new(min_diversity: f64) -> Self {
        Self { min_diversity }
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for DiversityThreshold {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        let diversity = state.population.diversity();
        diversity < self.min_diversity
    }

    fn reason(&self) -> &'static str {
        "Diversity threshold reached"
    }
}

/// Combine criteria with OR logic (any one triggers termination)
pub struct AnyOf<G: EvolutionaryGenome, F: FitnessValue = f64> {
    criteria: Vec<Box<dyn TerminationCriterion<G, F>>>,
}

impl<G: EvolutionaryGenome, F: FitnessValue> AnyOf<G, F> {
    /// Create a new AnyOf combinator
    pub fn new(criteria: Vec<Box<dyn TerminationCriterion<G, F>>>) -> Self {
        Self { criteria }
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for AnyOf<G, F> {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        self.criteria.iter().any(|c| c.should_terminate(state))
    }

    fn reason(&self) -> &'static str {
        "One of multiple criteria met"
    }
}

/// Combine criteria with AND logic (all must trigger for termination)
pub struct AllOf<G: EvolutionaryGenome, F: FitnessValue = f64> {
    criteria: Vec<Box<dyn TerminationCriterion<G, F>>>,
}

impl<G: EvolutionaryGenome, F: FitnessValue> AllOf<G, F> {
    /// Create a new AllOf combinator
    pub fn new(criteria: Vec<Box<dyn TerminationCriterion<G, F>>>) -> Self {
        Self { criteria }
    }
}

impl<G: EvolutionaryGenome, F: FitnessValue> TerminationCriterion<G, F> for AllOf<G, F> {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool {
        !self.criteria.is_empty() && self.criteria.iter().all(|c| c.should_terminate(state))
    }

    fn reason(&self) -> &'static str {
        "All criteria met"
    }
}

pub mod prelude {
    pub use super::{
        AllOf, AnyOf, DiversityThreshold, EvolutionState, FitnessStagnation, MaxEvaluations,
        MaxGenerations, TargetFitness, TerminationCriterion,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::population::individual::Individual;
    use crate::population::population::Population;

    fn create_test_state<'a>(
        generation: usize,
        evaluations: usize,
        best_fitness: f64,
        population: &'a Population<RealVector>,
        fitness_history: &'a [f64],
    ) -> EvolutionState<'a, RealVector> {
        EvolutionState {
            generation,
            evaluations,
            best_fitness,
            population,
            fitness_history,
        }
    }

    #[test]
    fn test_max_generations() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);
        let history = vec![];

        let criterion = MaxGenerations::new(100);

        let state = create_test_state(50, 0, 10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        let state = create_test_state(100, 0, 10.0, &pop, &history);
        assert!(criterion.should_terminate(&state));

        let state = create_test_state(150, 0, 10.0, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }

    #[test]
    fn test_max_evaluations() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);
        let history = vec![];

        let criterion = MaxEvaluations::new(1000);

        let state = create_test_state(0, 500, 10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        let state = create_test_state(0, 1000, 10.0, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }

    #[test]
    fn test_fitness_stagnation() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);

        let criterion = FitnessStagnation::new(5, 0.01);

        // Not enough history
        let history = vec![1.0, 2.0, 3.0];
        let state = create_test_state(0, 0, 3.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // Still improving
        let history = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let state = create_test_state(0, 0, 5.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // Stagnant
        let history = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let state = create_test_state(0, 0, 5.0, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }

    #[test]
    fn test_target_fitness() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);
        let history = vec![];

        let criterion = TargetFitness::new(0.0);

        // Not at target (fitness is negative, we want 0)
        let state = create_test_state(0, 0, -10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // At target
        let state = create_test_state(0, 0, 0.0, &pop, &history);
        assert!(criterion.should_terminate(&state));

        // With tolerance
        let criterion = TargetFitness::with_tolerance(0.0, 0.1);
        let state = create_test_state(0, 0, -0.05, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }

    #[test]
    fn test_any_of() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);
        let history = vec![];

        let criterion = AnyOf::new(vec![
            Box::new(MaxGenerations::new(100)),
            Box::new(TargetFitness::new(0.0)),
        ]);

        // Neither met
        let state = create_test_state(50, 0, -10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // First met
        let state = create_test_state(100, 0, -10.0, &pop, &history);
        assert!(criterion.should_terminate(&state));

        // Second met
        let state = create_test_state(50, 0, 0.0, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }

    #[test]
    fn test_all_of() {
        let individuals = vec![
            Individual::with_fitness(RealVector::new(vec![1.0]), 10.0),
        ];
        let pop = Population::from_individuals(individuals);
        let history = vec![];

        let criterion = AllOf::new(vec![
            Box::new(MaxGenerations::new(100)),
            Box::new(TargetFitness::new(0.0)),
        ]);

        // Neither met
        let state = create_test_state(50, 0, -10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // Only first met
        let state = create_test_state(100, 0, -10.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // Only second met
        let state = create_test_state(50, 0, 0.0, &pop, &history);
        assert!(!criterion.should_terminate(&state));

        // Both met
        let state = create_test_state(100, 0, 0.0, &pop, &history);
        assert!(criterion.should_terminate(&state));
    }
}
