//! Selection operators
//!
//! This module provides various selection operators for genetic algorithms.

use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::SelectionOperator;

/// Tournament selection operator
///
/// Selects the best individual from a randomly sampled subset (the
/// "tournament") of the population.
///
/// # Sampling with vs. without replacement
///
/// By default competitors are sampled **with replacement** (each of the `k`
/// competitors is drawn independently and uniformly from the whole
/// population). This is the textbook model, for which the probability that the
/// individual of rank `i` wins has the clean closed form and selection pressure
/// grows smoothly with `k`; the same individual may appear more than once in a
/// tournament. Crucially, `tournament_size >= population size` does **not**
/// make selection deterministic under this model — a draw of `k = n`
/// competitors with replacement includes the global best only with probability
/// `1 - ((n-1)/n)^k` (≈ 0.63 at `k = n`).
///
/// The without-replacement variant (constructed via
/// [`without_replacement`](Self::without_replacement)) instead draws `k`
/// *distinct* individuals; there the tournament size is capped at the
/// population size and `k >= n` degenerates to fully elitist selection (the
/// global best is chosen every call).
#[derive(Clone, Debug)]
pub struct TournamentSelection {
    /// Tournament size (number of individuals competing)
    pub tournament_size: usize,
    /// Whether competitors are sampled with replacement (canonical default) or
    /// as distinct individuals.
    pub with_replacement: bool,
}

impl TournamentSelection {
    /// Create a new tournament selection with the given size (sampling with
    /// replacement, the canonical model).
    pub fn new(tournament_size: usize) -> Self {
        assert!(tournament_size >= 1, "Tournament size must be at least 1");
        Self {
            tournament_size,
            with_replacement: true,
        }
    }

    /// Create binary tournament selection (size = 2, with replacement)
    pub fn binary() -> Self {
        Self::new(2)
    }

    /// Create a tournament selection that samples `tournament_size` **distinct**
    /// competitors (without replacement).
    ///
    /// Note that with this variant `tournament_size >= population size` selects
    /// the global best deterministically every call.
    pub fn without_replacement(tournament_size: usize) -> Self {
        assert!(tournament_size >= 1, "Tournament size must be at least 1");
        Self {
            tournament_size,
            with_replacement: false,
        }
    }
}

impl<G: EvolutionaryGenome> SelectionOperator<G> for TournamentSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        assert!(!population.is_empty(), "Population cannot be empty");

        let n = population.len();

        let tournament: Vec<usize> = if self.with_replacement {
            // Canonical: k competitors drawn i.i.d. with replacement.
            (0..self.tournament_size)
                .map(|_| rng.gen_range(0..n))
                .collect()
        } else {
            // Distinct competitors; cannot draw more than the population size.
            let k = self.tournament_size.min(n);
            (0..n)
                .collect::<Vec<usize>>()
                .choose_multiple(rng, k)
                .copied()
                .collect()
        };

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
            (1.0..=2.0).contains(&selection_pressure),
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
        let scaled: Vec<f64> = population
            .iter()
            .map(|(_, f)| f / self.temperature)
            .collect();
        let max_scaled = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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

        // Without-replacement full tournament draws all distinct individuals,
        // so it selects the best deterministically.
        let selection = TournamentSelection::without_replacement(3);

        let mut best_count = 0;
        let trials = 100;
        for _ in 0..trials {
            let idx = selection.select(&population, &mut rng);
            if idx == 1 {
                best_count += 1;
            }
        }

        // With a full without-replacement tournament, should always select the best
        assert_eq!(best_count, trials);
    }

    #[test]
    fn test_tournament_with_replacement_is_not_deterministic_at_full_size() {
        // regression: EV-104 — the default (with-replacement) tournament must
        // NOT collapse to fully elitist selection when tournament_size ==
        // population size. Pre-fix, sampling was without replacement and
        // clamped to the population, so k >= n selected the global best on
        // every call (best_count would equal trials). With canonical
        // with-replacement sampling the global best is included only with
        // probability 1 - ((n-1)/n)^k, so it is selected only part of the time.
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        // Unique global maximum at index 4.
        let population: Vec<(RealVector, f64)> = (0..5)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect();

        let selection = TournamentSelection::new(5); // k == n, with replacement
        assert!(selection.with_replacement);

        let trials = 300;
        let mut best_count = 0;
        for _ in 0..trials {
            if selection.select(&population, &mut rng) == 4 {
                best_count += 1;
            }
        }

        // Expected P(best selected) = 1 - (4/5)^5 ≈ 0.672, so the count must be
        // strictly below `trials` (the pre-fix without-replacement code would
        // give exactly `trials`).
        assert!(
            best_count < trials,
            "with-replacement tournament should not be deterministic at k == n (got {best_count}/{trials})"
        );
        // ...but the best should still win the clear majority of the time.
        assert!(
            best_count > trials / 2,
            "expected the fittest to win most tournaments (got {best_count}/{trials})"
        );
    }

    #[test]
    fn test_tournament_without_replacement_full_size_is_elitist() {
        // The retained without-replacement variant selects the global best
        // deterministically when tournament_size >= population size.
        let mut rng = rand::thread_rng();
        let population: Vec<(RealVector, f64)> = (0..5)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect();

        let selection = TournamentSelection::without_replacement(5);
        for _ in 0..50 {
            assert_eq!(selection.select(&population, &mut rng), 4);
        }
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
