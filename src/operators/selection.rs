//! Selection operators
//!
//! This module provides various selection operators for genetic algorithms.

use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, WeightedIndex};

use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::SelectionOperator;

/// Tournament selection operator
///
/// Selects the best individual from a random subset of the population.
#[derive(Clone, Debug)]
pub struct TournamentSelection {
    /// Tournament size (number of individuals competing)
    pub tournament_size: usize,
}

impl TournamentSelection {
    /// Create a new tournament selection with the given size
    pub fn new(tournament_size: usize) -> Self {
        assert!(tournament_size >= 1, "Tournament size must be at least 1");
        Self { tournament_size }
    }

    /// Create binary tournament selection (size = 2)
    pub fn binary() -> Self {
        Self::new(2)
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for TournamentSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        let tournament_size = self.tournament_size.min(population.len());

        // Select random individuals for the tournament
        let indices: Vec<usize> = (0..population.len()).collect();
        let tournament: Vec<usize> = indices
            .choose_multiple(rng, tournament_size)
            .copied()
            .collect();

        // Find the best in the tournament
        tournament
            .into_iter()
            .max_by(|&a, &b| {
                population[a]
                    .1
                    .partial_cmp(&population[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }
}

/// Roulette wheel selection (fitness proportionate)
///
/// Selection probability is proportional to fitness.
#[derive(Clone, Debug)]
pub struct RouletteSelection {
    /// Offset to ensure all fitnesses are positive
    offset: f64,
}

impl RouletteSelection {
    /// Create a new roulette selection
    pub fn new() -> Self {
        Self { offset: 0.0 }
    }

    /// Create with a fitness offset (to handle negative fitness)
    pub fn with_offset(offset: f64) -> Self {
        Self { offset }
    }
}

impl Default for RouletteSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for RouletteSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        // Find minimum fitness and compute offset
        let min_fitness = population
            .iter()
            .map(|(_, f)| *f)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let offset = if min_fitness < 0.0 {
            -min_fitness + self.offset + 1.0
        } else {
            self.offset
        };

        // Compute weights
        let weights: Vec<f64> = population.iter().map(|(_, f)| f + offset).collect();

        // Handle case where all weights are zero
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return rng.gen_range(0..population.len());
        }

        // Create weighted distribution
        match WeightedIndex::new(&weights) {
            Ok(dist) => dist.sample(rng),
            Err(_) => rng.gen_range(0..population.len()),
        }
    }
}

/// Truncation selection
///
/// Selects only from the top percentage of the population.
#[derive(Clone, Debug)]
pub struct TruncationSelection {
    /// Fraction of population to select from (0.0 to 1.0)
    pub truncation_ratio: f64,
}

impl TruncationSelection {
    /// Create a new truncation selection
    pub fn new(truncation_ratio: f64) -> Self {
        assert!(
            truncation_ratio > 0.0 && truncation_ratio <= 1.0,
            "Truncation ratio must be in (0, 1]"
        );
        Self { truncation_ratio }
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for TruncationSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        // Sort indices by fitness (descending)
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            population[b]
                .1
                .partial_cmp(&population[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select from top portion
        let cutoff = ((population.len() as f64) * self.truncation_ratio).ceil() as usize;
        let cutoff = cutoff.max(1);

        indices[rng.gen_range(0..cutoff)]
    }
}

/// Rank-based selection
///
/// Selection probability is based on rank rather than raw fitness.
#[derive(Clone, Debug)]
pub struct RankSelection {
    /// Selection pressure (1.0 = uniform, 2.0 = strong pressure)
    pub selection_pressure: f64,
}

impl RankSelection {
    /// Create a new rank selection
    pub fn new(selection_pressure: f64) -> Self {
        assert!(
            selection_pressure >= 1.0 && selection_pressure <= 2.0,
            "Selection pressure must be in [1.0, 2.0]"
        );
        Self { selection_pressure }
    }
}

impl Default for RankSelection {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for RankSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        let n = population.len();
        let sp = self.selection_pressure;

        // Sort indices by fitness (ascending - worst first)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            population[a]
                .1
                .partial_cmp(&population[b].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute rank-based weights using Baker's linear ranking
        // weight(i) = 2 - sp + 2(sp - 1)(rank - 1)/(n - 1)
        let weights: Vec<f64> = (0..n)
            .map(|rank| {
                if n == 1 {
                    1.0
                } else {
                    2.0 - sp + 2.0 * (sp - 1.0) * (rank as f64) / ((n - 1) as f64)
                }
            })
            .collect();

        match WeightedIndex::new(&weights) {
            Ok(dist) => indices[dist.sample(rng)],
            Err(_) => indices[rng.gen_range(0..n)],
        }
    }
}

/// Boltzmann selection (temperature-based)
///
/// Uses softmax of fitness values scaled by temperature.
#[derive(Clone, Debug)]
pub struct BoltzmannSelection {
    /// Temperature parameter (higher = more uniform, lower = more greedy)
    pub temperature: f64,
}

impl BoltzmannSelection {
    /// Create a new Boltzmann selection
    pub fn new(temperature: f64) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        Self { temperature }
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for BoltzmannSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        // Use log-sum-exp trick for numerical stability
        let scaled: Vec<f64> = population.iter().map(|(_, f)| f / self.temperature).collect();
        let max_scaled = scaled
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let weights: Vec<f64> = scaled.iter().map(|s| (s - max_scaled).exp()).collect();

        match WeightedIndex::new(&weights) {
            Ok(dist) => dist.sample(rng),
            Err(_) => rng.gen_range(0..population.len()),
        }
    }
}

/// Random selection (uniform)
#[derive(Clone, Debug, Default)]
pub struct RandomSelection;

impl RandomSelection {
    /// Create a new random selection
    pub fn new() -> Self {
        Self
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for RandomSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");
        rng.gen_range(0..population.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;

    fn create_population(size: usize) -> Vec<(RealVector, f64)> {
        (0..size)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect()
    }

    #[test]
    fn test_tournament_selection_selects_valid_index() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = TournamentSelection::new(3);

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_tournament_selection_binary() {
        let selection = TournamentSelection::binary();
        assert_eq!(selection.tournament_size, 2);
    }

    #[test]
    fn test_tournament_selection_prefers_fitter() {
        let mut rng = rand::thread_rng();
        // Population with clear fitness difference
        let population: Vec<(RealVector, f64)> = vec![
            (RealVector::new(vec![0.0]), 0.0),
            (RealVector::new(vec![1.0]), 100.0), // Much fitter
            (RealVector::new(vec![2.0]), 0.0),
        ];

        let selection = TournamentSelection::new(3); // Full tournament

        let mut best_count = 0;
        let trials = 100;
        for _ in 0..trials {
            let idx = selection.select(&population, &mut rng);
            if idx == 1 {
                best_count += 1;
            }
        }

        // With full tournament, should always select the best
        assert_eq!(best_count, trials);
    }

    #[test]
    fn test_roulette_selection_selects_valid_index() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = RouletteSelection::new();

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_roulette_selection_handles_negative_fitness() {
        let mut rng = rand::thread_rng();
        let population: Vec<(RealVector, f64)> = vec![
            (RealVector::new(vec![0.0]), -10.0),
            (RealVector::new(vec![1.0]), -5.0),
            (RealVector::new(vec![2.0]), -1.0),
        ];

        let selection = RouletteSelection::new();

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_truncation_selection_selects_from_top() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = TruncationSelection::new(0.2); // Top 20%

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            // Top 20% of [0..9] should be indices 8 or 9
            assert!(idx >= 8);
        }
    }

    #[test]
    fn test_rank_selection_selects_valid_index() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = RankSelection::new(1.5);

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_boltzmann_selection_selects_valid_index() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = BoltzmannSelection::new(1.0);

        for _ in 0..100 {
            let idx = selection.select(&population, &mut rng);
            assert!(idx < population.len());
        }
    }

    #[test]
    fn test_boltzmann_selection_temperature_effect() {
        let mut rng = rand::thread_rng();
        // Population with clear fitness difference
        let population: Vec<(RealVector, f64)> = vec![
            (RealVector::new(vec![0.0]), 0.0),
            (RealVector::new(vec![1.0]), 10.0),
        ];

        // Low temperature = more greedy
        let low_temp = BoltzmannSelection::new(0.1);
        // High temperature = more uniform
        let high_temp = BoltzmannSelection::new(100.0);

        let mut low_best_count = 0;
        let mut high_best_count = 0;
        let trials = 1000;

        for _ in 0..trials {
            if low_temp.select(&population, &mut rng) == 1 {
                low_best_count += 1;
            }
            if high_temp.select(&population, &mut rng) == 1 {
                high_best_count += 1;
            }
        }

        // Low temperature should select best more often
        assert!(low_best_count > high_best_count);
    }

    #[test]
    fn test_random_selection_uniform() {
        let mut rng = rand::thread_rng();
        let population = create_population(2);
        let selection = RandomSelection::new();

        let mut counts = [0, 0];
        let trials = 1000;

        for _ in 0..trials {
            counts[selection.select(&population, &mut rng)] += 1;
        }

        // Should be roughly 50-50 (with some variance)
        let ratio = counts[0] as f64 / counts[1] as f64;
        assert!(ratio > 0.8 && ratio < 1.2);
    }

    #[test]
    fn test_select_many() {
        let mut rng = rand::thread_rng();
        let population = create_population(10);
        let selection = TournamentSelection::new(3);

        let indices = selection.select_many(&population, 5, &mut rng);
        assert_eq!(indices.len(), 5);
        for idx in indices {
            assert!(idx < population.len());
        }
    }

    #[test]
    #[should_panic(expected = "Tournament size must be at least 1")]
    fn test_tournament_size_zero() {
        TournamentSelection::new(0);
    }

    #[test]
    #[should_panic(expected = "Truncation ratio must be in (0, 1]")]
    fn test_truncation_ratio_zero() {
        TruncationSelection::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_boltzmann_temperature_zero() {
        BoltzmannSelection::new(0.0);
    }
}
