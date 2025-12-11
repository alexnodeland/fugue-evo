//! Diagnostics and statistics
//!
//! This module provides statistics collection and analysis for evolutionary runs.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::fitness::traits::FitnessValue;
use crate::genome::traits::EvolutionaryGenome;
use crate::population::population::Population;

/// Statistics for a single generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Generation number
    pub generation: usize,
    /// Total fitness evaluations so far
    pub evaluations: usize,
    /// Best fitness in this generation
    pub best_fitness: f64,
    /// Worst fitness in this generation
    pub worst_fitness: f64,
    /// Mean fitness
    pub mean_fitness: f64,
    /// Median fitness
    pub median_fitness: f64,
    /// Fitness standard deviation
    pub fitness_std: f64,
    /// Population diversity
    pub diversity: f64,
    /// Timing information
    pub timing: TimingStats,
}

/// Timing statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimingStats {
    /// Time spent on fitness evaluation (ms)
    pub evaluation_ms: f64,
    /// Time spent on selection (ms)
    pub selection_ms: f64,
    /// Time spent on crossover (ms)
    pub crossover_ms: f64,
    /// Time spent on mutation (ms)
    pub mutation_ms: f64,
    /// Total generation time (ms)
    pub total_ms: f64,
}

impl TimingStats {
    /// Create new timing stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Set evaluation time
    pub fn with_evaluation(mut self, duration: Duration) -> Self {
        self.evaluation_ms = duration.as_secs_f64() * 1000.0;
        self
    }

    /// Set selection time
    pub fn with_selection(mut self, duration: Duration) -> Self {
        self.selection_ms = duration.as_secs_f64() * 1000.0;
        self
    }

    /// Set crossover time
    pub fn with_crossover(mut self, duration: Duration) -> Self {
        self.crossover_ms = duration.as_secs_f64() * 1000.0;
        self
    }

    /// Set mutation time
    pub fn with_mutation(mut self, duration: Duration) -> Self {
        self.mutation_ms = duration.as_secs_f64() * 1000.0;
        self
    }

    /// Set total time
    pub fn with_total(mut self, duration: Duration) -> Self {
        self.total_ms = duration.as_secs_f64() * 1000.0;
        self
    }
}

impl GenerationStats {
    /// Compute statistics from a population
    pub fn from_population<G, F>(
        population: &Population<G, F>,
        generation: usize,
        evaluations: usize,
    ) -> Self
    where
        G: EvolutionaryGenome,
        F: FitnessValue,
    {
        let mut fitnesses: Vec<f64> = population
            .iter()
            .filter_map(|i| i.fitness.as_ref().map(|f| f.to_f64()))
            .collect();

        if fitnesses.is_empty() {
            return Self {
                generation,
                evaluations,
                best_fitness: f64::NEG_INFINITY,
                worst_fitness: f64::INFINITY,
                mean_fitness: 0.0,
                median_fitness: 0.0,
                fitness_std: 0.0,
                diversity: 0.0,
                timing: TimingStats::default(),
            };
        }

        fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let best = fitnesses.last().copied().unwrap_or(f64::NEG_INFINITY);
        let worst = fitnesses.first().copied().unwrap_or(f64::INFINITY);
        let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let median = if fitnesses.len() % 2 == 0 {
            (fitnesses[fitnesses.len() / 2 - 1] + fitnesses[fitnesses.len() / 2]) / 2.0
        } else {
            fitnesses[fitnesses.len() / 2]
        };

        let variance = if fitnesses.len() > 1 {
            fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / (fitnesses.len() - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        let diversity = population.diversity();

        Self {
            generation,
            evaluations,
            best_fitness: best,
            worst_fitness: worst,
            mean_fitness: mean,
            median_fitness: median,
            fitness_std: std,
            diversity,
            timing: TimingStats::default(),
        }
    }

    /// Set timing information
    pub fn with_timing(mut self, timing: TimingStats) -> Self {
        self.timing = timing;
        self
    }
}

/// Statistics collector for an entire evolution run
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvolutionStats {
    /// Statistics per generation
    pub generations: Vec<GenerationStats>,
    /// Total runtime in milliseconds
    pub total_runtime_ms: f64,
    /// Reason for termination
    pub termination_reason: Option<String>,
}

impl EvolutionStats {
    /// Create a new stats collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a generation's statistics
    pub fn record(&mut self, stats: GenerationStats) {
        self.generations.push(stats);
    }

    /// Get the number of generations recorded
    pub fn num_generations(&self) -> usize {
        self.generations.len()
    }

    /// Get the best fitness across all generations
    pub fn best_fitness(&self) -> Option<f64> {
        self.generations
            .iter()
            .map(|g| g.best_fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the final best fitness
    pub fn final_best_fitness(&self) -> Option<f64> {
        self.generations.last().map(|g| g.best_fitness)
    }

    /// Get the history of best fitness values
    pub fn best_fitness_history(&self) -> Vec<f64> {
        self.generations.iter().map(|g| g.best_fitness).collect()
    }

    /// Get the history of mean fitness values
    pub fn mean_fitness_history(&self) -> Vec<f64> {
        self.generations.iter().map(|g| g.mean_fitness).collect()
    }

    /// Get the history of diversity values
    pub fn diversity_history(&self) -> Vec<f64> {
        self.generations.iter().map(|g| g.diversity).collect()
    }

    /// Set the termination reason
    pub fn set_termination_reason(&mut self, reason: &str) {
        self.termination_reason = Some(reason.to_string());
    }

    /// Set the total runtime
    pub fn set_runtime(&mut self, duration: Duration) {
        self.total_runtime_ms = duration.as_secs_f64() * 1000.0;
    }

    /// Get a summary of the evolution run
    pub fn summary(&self) -> String {
        let best = self.best_fitness().unwrap_or(f64::NEG_INFINITY);
        let final_best = self.final_best_fitness().unwrap_or(f64::NEG_INFINITY);
        let generations = self.num_generations();
        let runtime = self.total_runtime_ms;

        format!(
            "Evolution Summary:\n\
             - Generations: {}\n\
             - Best fitness: {:.6}\n\
             - Final best: {:.6}\n\
             - Runtime: {:.2}ms\n\
             - Termination: {}",
            generations,
            best,
            final_best,
            runtime,
            self.termination_reason.as_deref().unwrap_or("unknown")
        )
    }
}

/// Result of an evolution run
#[derive(Clone, Debug)]
pub struct EvolutionResult<G, F = f64>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// The best genome found
    pub best_genome: G,
    /// The best fitness value
    pub best_fitness: F,
    /// Number of generations completed
    pub generations: usize,
    /// Total fitness evaluations
    pub evaluations: usize,
    /// Statistics for the run
    pub stats: EvolutionStats,
}

impl<G, F> EvolutionResult<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
{
    /// Create a new evolution result
    pub fn new(best_genome: G, best_fitness: F, generations: usize, evaluations: usize) -> Self {
        Self {
            best_genome,
            best_fitness,
            generations,
            evaluations,
            stats: EvolutionStats::new(),
        }
    }

    /// Add statistics to the result
    pub fn with_stats(mut self, stats: EvolutionStats) -> Self {
        self.stats = stats;
        self
    }
}

pub mod prelude {
    pub use super::{EvolutionResult, EvolutionStats, GenerationStats, TimingStats};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::population::individual::Individual;

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
    fn test_generation_stats_from_population() {
        let pop = create_test_population();
        let stats = GenerationStats::from_population(&pop, 10, 100);

        assert_eq!(stats.generation, 10);
        assert_eq!(stats.evaluations, 100);
        assert_eq!(stats.best_fitness, 50.0);
        assert_eq!(stats.worst_fitness, 10.0);
        assert_eq!(stats.mean_fitness, 30.0);
        assert_eq!(stats.median_fitness, 30.0);
        assert!(stats.fitness_std > 15.0 && stats.fitness_std < 16.0);
    }

    #[test]
    fn test_generation_stats_empty_population() {
        let pop: Population<RealVector> = Population::new();
        let stats = GenerationStats::from_population(&pop, 0, 0);

        assert_eq!(stats.best_fitness, f64::NEG_INFINITY);
        assert_eq!(stats.worst_fitness, f64::INFINITY);
    }

    #[test]
    fn test_evolution_stats_record() {
        let mut stats = EvolutionStats::new();
        let pop = create_test_population();

        for i in 0..5 {
            let gen_stats = GenerationStats::from_population(&pop, i, i * 10);
            stats.record(gen_stats);
        }

        assert_eq!(stats.num_generations(), 5);
        assert_eq!(stats.best_fitness(), Some(50.0));
    }

    #[test]
    fn test_evolution_stats_history() {
        let mut stats = EvolutionStats::new();

        for i in 0..5 {
            stats.record(GenerationStats {
                generation: i,
                evaluations: i * 10,
                best_fitness: (i + 1) as f64 * 10.0,
                worst_fitness: 0.0,
                mean_fitness: (i + 1) as f64 * 5.0,
                median_fitness: 0.0,
                fitness_std: 0.0,
                diversity: 1.0 / (i + 1) as f64,
                timing: TimingStats::default(),
            });
        }

        let best_history = stats.best_fitness_history();
        assert_eq!(best_history, vec![10.0, 20.0, 30.0, 40.0, 50.0]);

        let mean_history = stats.mean_fitness_history();
        assert_eq!(mean_history, vec![5.0, 10.0, 15.0, 20.0, 25.0]);
    }

    #[test]
    fn test_evolution_stats_summary() {
        let mut stats = EvolutionStats::new();
        stats.record(GenerationStats {
            generation: 0,
            evaluations: 100,
            best_fitness: 50.0,
            worst_fitness: 10.0,
            mean_fitness: 30.0,
            median_fitness: 30.0,
            fitness_std: 15.0,
            diversity: 0.5,
            timing: TimingStats::default(),
        });
        stats.set_termination_reason("Target reached");
        stats.set_runtime(Duration::from_millis(1234));

        let summary = stats.summary();
        assert!(summary.contains("Generations: 1"));
        assert!(summary.contains("Best fitness: 50"));
        assert!(summary.contains("Target reached"));
    }

    #[test]
    fn test_timing_stats() {
        let timing = TimingStats::new()
            .with_evaluation(Duration::from_millis(100))
            .with_selection(Duration::from_millis(20))
            .with_crossover(Duration::from_millis(30))
            .with_mutation(Duration::from_millis(10))
            .with_total(Duration::from_millis(160));

        assert!((timing.evaluation_ms - 100.0).abs() < 0.1);
        assert!((timing.selection_ms - 20.0).abs() < 0.1);
        assert!((timing.crossover_ms - 30.0).abs() < 0.1);
        assert!((timing.mutation_ms - 10.0).abs() < 0.1);
        assert!((timing.total_ms - 160.0).abs() < 0.1);
    }

    #[test]
    fn test_evolution_result() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let result = EvolutionResult::new(genome, 42.0, 100, 1000);

        assert_eq!(result.best_fitness, 42.0);
        assert_eq!(result.generations, 100);
        assert_eq!(result.evaluations, 1000);
    }
}
