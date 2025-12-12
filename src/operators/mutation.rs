//! Mutation operators
//!
//! This module provides various mutation operators for genetic algorithms.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::permutation::Permutation;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::{
    BinaryGenome, EvolutionaryGenome, PermutationGenome, RealValuedGenome,
};
use crate::operators::traits::{BoundedMutationOperator, MutationOperator};

/// Polynomial mutation (bounded)
///
/// Uses the polynomial probability distribution to perturb genes.
/// Respects bounds and is commonly used with NSGA-II.
///
/// Reference: Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
#[derive(Clone, Debug)]
pub struct PolynomialMutation {
    /// Distribution index (typically 20-100)
    /// Higher values = smaller mutations
    pub eta_m: f64,
    /// Per-gene mutation probability (default: 1/n)
    pub mutation_probability: Option<f64>,
}

impl PolynomialMutation {
    /// Create a new polynomial mutation with the given distribution index
    pub fn new(eta_m: f64) -> Self {
        assert!(eta_m >= 0.0, "Distribution index must be non-negative");
        Self {
            eta_m,
            mutation_probability: None,
        }
    }

    /// Set a fixed mutation probability per gene
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be in [0, 1]"
        );
        self.mutation_probability = Some(probability);
        self
    }

    /// Apply polynomial mutation to a gene
    fn mutate_gene<R: Rng>(&self, gene: f64, min: f64, max: f64, rng: &mut R) -> f64 {
        let range = max - min;
        if range <= 0.0 {
            return gene;
        }

        let delta1 = (gene - min) / range;
        let delta2 = (max - gene) / range;

        let u = rng.gen::<f64>();
        let delta_q = if u <= 0.5 {
            let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(self.eta_m + 1.0);
            val.powf(1.0 / (self.eta_m + 1.0)) - 1.0
        } else {
            let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(self.eta_m + 1.0);
            1.0 - val.powf(1.0 / (self.eta_m + 1.0))
        };

        (gene + delta_q * range).clamp(min, max)
    }
}

impl MutationOperator<RealVector> for PolynomialMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        // Use default bounds if not specified
        let default_bounds = MultiBounds::symmetric(1e10, genome.dimension());
        self.mutate_bounded(genome, &default_bounds, rng);
    }

    fn mutation_probability(&self) -> f64 {
        self.mutation_probability.unwrap_or(1.0)
    }
}

impl BoundedMutationOperator<RealVector> for PolynomialMutation {
    fn mutate_bounded<R: Rng>(&self, genome: &mut RealVector, bounds: &MultiBounds, rng: &mut R) {
        let n = genome.dimension();
        let prob = self.mutation_probability.unwrap_or(1.0 / n as f64);

        for i in 0..n {
            if rng.gen::<f64>() < prob {
                if let Some(bound) = bounds.get(i) {
                    genome.genes_mut()[i] =
                        self.mutate_gene(genome.genes()[i], bound.min, bound.max, rng);
                }
            }
        }
    }
}

/// Gaussian mutation
///
/// Adds Gaussian noise to each gene.
#[derive(Clone, Debug)]
pub struct GaussianMutation {
    /// Standard deviation of the Gaussian noise
    pub sigma: f64,
    /// Per-gene mutation probability
    pub mutation_probability: Option<f64>,
}

impl GaussianMutation {
    /// Create a new Gaussian mutation with the given standard deviation
    pub fn new(sigma: f64) -> Self {
        assert!(sigma >= 0.0, "Sigma must be non-negative");
        Self {
            sigma,
            mutation_probability: None,
        }
    }

    /// Set a fixed mutation probability per gene
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be in [0, 1]"
        );
        self.mutation_probability = Some(probability);
        self
    }
}

impl MutationOperator<RealVector> for GaussianMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        let n = genome.dimension();
        let prob = self.mutation_probability.unwrap_or(1.0 / n as f64);
        let normal = Normal::new(0.0, self.sigma).unwrap();

        for gene in genome.genes_mut() {
            if rng.gen::<f64>() < prob {
                *gene += normal.sample(rng);
            }
        }
    }

    fn mutation_probability(&self) -> f64 {
        self.mutation_probability.unwrap_or(1.0)
    }
}

impl BoundedMutationOperator<RealVector> for GaussianMutation {
    fn mutate_bounded<R: Rng>(&self, genome: &mut RealVector, bounds: &MultiBounds, rng: &mut R) {
        let n = genome.dimension();
        let prob = self.mutation_probability.unwrap_or(1.0 / n as f64);
        let normal = Normal::new(0.0, self.sigma).unwrap();

        for i in 0..n {
            if rng.gen::<f64>() < prob {
                genome.genes_mut()[i] += normal.sample(rng);
                if let Some(bound) = bounds.get(i) {
                    genome.genes_mut()[i] = bound.clamp(genome.genes()[i]);
                }
            }
        }
    }
}

/// Uniform mutation
///
/// Replaces genes with random values within bounds.
#[derive(Clone, Debug)]
pub struct UniformMutation {
    /// Per-gene mutation probability
    pub mutation_probability: Option<f64>,
}

impl UniformMutation {
    /// Create a new uniform mutation
    pub fn new() -> Self {
        Self {
            mutation_probability: None,
        }
    }

    /// Set a fixed mutation probability per gene
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be in [0, 1]"
        );
        self.mutation_probability = Some(probability);
        self
    }
}

impl Default for UniformMutation {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperator<RealVector> for UniformMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        let default_bounds = MultiBounds::symmetric(1.0, genome.dimension());
        self.mutate_bounded(genome, &default_bounds, rng);
    }
}

impl BoundedMutationOperator<RealVector> for UniformMutation {
    fn mutate_bounded<R: Rng>(&self, genome: &mut RealVector, bounds: &MultiBounds, rng: &mut R) {
        let n = genome.dimension();
        let prob = self.mutation_probability.unwrap_or(1.0 / n as f64);

        for i in 0..n {
            if rng.gen::<f64>() < prob {
                if let Some(bound) = bounds.get(i) {
                    genome.genes_mut()[i] = rng.gen_range(bound.min..=bound.max);
                }
            }
        }
    }
}

/// Bit-flip mutation for bit strings
///
/// Flips each bit with a given probability.
#[derive(Clone, Debug)]
pub struct BitFlipMutation {
    /// Per-bit mutation probability (default: 1/n)
    pub mutation_probability: Option<f64>,
}

impl BitFlipMutation {
    /// Create a new bit-flip mutation
    pub fn new() -> Self {
        Self {
            mutation_probability: None,
        }
    }

    /// Set a fixed mutation probability per bit
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be in [0, 1]"
        );
        self.mutation_probability = Some(probability);
        self
    }
}

impl Default for BitFlipMutation {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperator<BitString> for BitFlipMutation {
    fn mutate<R: Rng>(&self, genome: &mut BitString, rng: &mut R) {
        let n = genome.len();
        let prob = self.mutation_probability.unwrap_or(1.0 / n as f64);

        for i in 0..n {
            if rng.gen::<f64>() < prob {
                genome.flip(i);
            }
        }
    }

    fn mutation_probability(&self) -> f64 {
        self.mutation_probability.unwrap_or(1.0)
    }
}

/// Swap mutation for permutation genomes (also works on any genome)
///
/// Swaps two random positions in the genome.
#[derive(Clone, Debug, Default)]
pub struct SwapMutation {
    /// Number of swaps to perform
    pub num_swaps: usize,
}

impl SwapMutation {
    /// Create a new swap mutation with a single swap
    pub fn new() -> Self {
        Self { num_swaps: 1 }
    }

    /// Create with multiple swaps
    pub fn with_swaps(num_swaps: usize) -> Self {
        Self { num_swaps }
    }
}

impl MutationOperator<BitString> for SwapMutation {
    fn mutate<R: Rng>(&self, genome: &mut BitString, rng: &mut R) {
        let n = genome.len();
        if n < 2 {
            return;
        }

        for _ in 0..self.num_swaps {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j {
                let temp = genome.bits()[i];
                genome.bits_mut()[i] = genome.bits()[j];
                genome.bits_mut()[j] = temp;
            }
        }
    }
}

impl MutationOperator<RealVector> for SwapMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        let n = genome.dimension();
        if n < 2 {
            return;
        }

        for _ in 0..self.num_swaps {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j {
                genome.genes_mut().swap(i, j);
            }
        }
    }
}

/// Scramble mutation
///
/// Scrambles a random segment of the genome.
#[derive(Clone, Debug, Default)]
pub struct ScrambleMutation;

impl ScrambleMutation {
    /// Create a new scramble mutation
    pub fn new() -> Self {
        Self
    }
}

impl MutationOperator<BitString> for ScrambleMutation {
    fn mutate<R: Rng>(&self, genome: &mut BitString, rng: &mut R) {
        use rand::seq::SliceRandom;

        let n = genome.len();
        if n < 2 {
            return;
        }

        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        // Extract segment, shuffle, and put back
        let segment: Vec<bool> = (start..=end).map(|i| genome.bits()[i]).collect();
        let mut shuffled = segment;
        shuffled.shuffle(rng);

        for (i, val) in shuffled.into_iter().enumerate() {
            genome.bits_mut()[start + i] = val;
        }
    }
}

impl MutationOperator<RealVector> for ScrambleMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        use rand::seq::SliceRandom;

        let n = genome.dimension();
        if n < 2 {
            return;
        }

        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        // Shuffle the segment in place
        let slice = &mut genome.genes_mut()[start..=end];
        slice.shuffle(rng);
    }
}

// =============================================================================
// Permutation Mutation Operators
// =============================================================================

/// Swap mutation for permutation genomes
///
/// Swaps two random positions in the permutation.
/// This is one of the simplest and most commonly used permutation mutations.
#[derive(Clone, Debug, Default)]
pub struct PermutationSwapMutation {
    /// Number of swaps to perform
    pub num_swaps: usize,
}

impl PermutationSwapMutation {
    /// Create a new swap mutation with a single swap
    pub fn new() -> Self {
        Self { num_swaps: 1 }
    }

    /// Create with multiple swaps
    pub fn with_swaps(num_swaps: usize) -> Self {
        Self { num_swaps }
    }
}

impl MutationOperator<Permutation> for PermutationSwapMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        let n = genome.dimension();
        if n < 2 {
            return;
        }

        for _ in 0..self.num_swaps {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j {
                genome.swap(i, j);
            }
        }
    }
}

/// Insert mutation for permutation genomes
///
/// Removes an element from one position and inserts it at another.
/// This preserves adjacencies better than swap mutation.
#[derive(Clone, Debug, Default)]
pub struct InsertMutation {
    /// Number of insert operations to perform
    pub num_inserts: usize,
}

impl InsertMutation {
    /// Create a new insert mutation with a single insert
    pub fn new() -> Self {
        Self { num_inserts: 1 }
    }

    /// Create with multiple inserts
    pub fn with_inserts(num_inserts: usize) -> Self {
        Self { num_inserts }
    }
}

impl MutationOperator<Permutation> for InsertMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        let n = genome.dimension();
        if n < 2 {
            return;
        }

        for _ in 0..self.num_inserts {
            let from = rng.gen_range(0..n);
            let to = rng.gen_range(0..n);
            if from != to {
                genome.insert(from, to);
            }
        }
    }
}

/// Inversion mutation (2-opt) for permutation genomes
///
/// Reverses a random segment of the permutation.
/// This is particularly effective for TSP-like problems as it can
/// remove crossing edges.
#[derive(Clone, Debug, Default)]
pub struct InversionMutation;

impl InversionMutation {
    /// Create a new inversion mutation
    pub fn new() -> Self {
        Self
    }
}

impl MutationOperator<Permutation> for InversionMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        let n = genome.dimension();
        if n < 2 {
            return;
        }

        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        genome.reverse_segment(start, end);
    }
}

/// Scramble mutation for permutation genomes
///
/// Shuffles a random segment of the permutation.
/// More disruptive than inversion, but still preserves some structure.
#[derive(Clone, Debug, Default)]
pub struct PermutationScrambleMutation;

impl PermutationScrambleMutation {
    /// Create a new scramble mutation
    pub fn new() -> Self {
        Self
    }
}

impl MutationOperator<Permutation> for PermutationScrambleMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        use rand::seq::SliceRandom;

        let n = genome.dimension();
        if n < 2 {
            return;
        }

        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        // Shuffle the segment
        let perm = genome.permutation_mut();
        perm[start..=end].shuffle(rng);
    }
}

/// Displacement mutation for permutation genomes
///
/// Selects a segment, removes it, and inserts it at a random position.
/// This is similar to insert mutation but operates on segments.
#[derive(Clone, Debug, Default)]
pub struct DisplacementMutation;

impl DisplacementMutation {
    /// Create a new displacement mutation
    pub fn new() -> Self {
        Self
    }
}

impl MutationOperator<Permutation> for DisplacementMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        let n = genome.dimension();
        if n < 3 {
            return;
        }

        // Select segment
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        let segment_len = end - start + 1;
        if segment_len >= n {
            return; // Can't displace the entire permutation
        }

        // Extract segment
        let perm = genome.permutation_mut();
        let segment: Vec<usize> = perm[start..=end].to_vec();

        // Remove segment
        let remaining: Vec<usize> = perm[..start]
            .iter()
            .chain(perm[end + 1..].iter())
            .copied()
            .collect();

        // Choose insertion point in remaining
        let insert_pos = rng.gen_range(0..=remaining.len());

        // Rebuild permutation
        let new_perm: Vec<usize> = remaining[..insert_pos]
            .iter()
            .chain(segment.iter())
            .chain(remaining[insert_pos..].iter())
            .copied()
            .collect();

        perm.copy_from_slice(&new_perm);
    }
}

/// Adaptive mutation rate for permutation genomes
///
/// Combines multiple mutation operators with configurable probabilities.
#[derive(Clone, Debug)]
pub struct AdaptivePermutationMutation {
    /// Probability of swap mutation
    pub swap_prob: f64,
    /// Probability of insert mutation
    pub insert_prob: f64,
    /// Probability of inversion mutation
    pub inversion_prob: f64,
    /// Probability of scramble mutation
    pub scramble_prob: f64,
}

impl AdaptivePermutationMutation {
    /// Create with default probabilities (each equally likely)
    pub fn new() -> Self {
        Self {
            swap_prob: 0.25,
            insert_prob: 0.25,
            inversion_prob: 0.25,
            scramble_prob: 0.25,
        }
    }

    /// Create with custom probabilities
    pub fn with_probs(swap: f64, insert: f64, inversion: f64, scramble: f64) -> Self {
        Self {
            swap_prob: swap,
            insert_prob: insert,
            inversion_prob: inversion,
            scramble_prob: scramble,
        }
    }
}

impl Default for AdaptivePermutationMutation {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationOperator<Permutation> for AdaptivePermutationMutation {
    fn mutate<R: Rng>(&self, genome: &mut Permutation, rng: &mut R) {
        let total = self.swap_prob + self.insert_prob + self.inversion_prob + self.scramble_prob;
        if total <= 0.0 {
            return;
        }

        let r = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;

        cumulative += self.swap_prob;
        if r < cumulative {
            PermutationSwapMutation::new().mutate(genome, rng);
            return;
        }

        cumulative += self.insert_prob;
        if r < cumulative {
            InsertMutation::new().mutate(genome, rng);
            return;
        }

        cumulative += self.inversion_prob;
        if r < cumulative {
            InversionMutation::new().mutate(genome, rng);
            return;
        }

        PermutationScrambleMutation::new().mutate(genome, rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_mutation_respects_bounds() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 10);

        for _ in 0..100 {
            let mut genome = RealVector::generate(&mut rng, &bounds);
            let mutation = PolynomialMutation::new(20.0).with_probability(1.0);
            mutation.mutate_bounded(&mut genome, &bounds, &mut rng);

            for (i, &gene) in genome.genes().iter().enumerate() {
                let bound = bounds.get(i).unwrap();
                assert!(
                    gene >= bound.min && gene <= bound.max,
                    "Gene {} out of bounds: {} not in [{}, {}]",
                    i,
                    gene,
                    bound.min,
                    bound.max
                );
            }
        }
    }

    #[test]
    fn test_polynomial_mutation_changes_genome() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 10);
        let original = RealVector::zeros(10);
        let mut genome = original.clone();

        let mutation = PolynomialMutation::new(20.0).with_probability(1.0);
        mutation.mutate_bounded(&mut genome, &bounds, &mut rng);

        // At least some genes should have changed
        let changed = genome
            .genes()
            .iter()
            .zip(original.genes())
            .filter(|(&a, &b)| a != b)
            .count();
        assert!(changed > 0, "No genes were mutated");
    }

    #[test]
    fn test_polynomial_mutation_eta_effect() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, 1);

        // Low eta = larger mutations
        let low_eta = PolynomialMutation::new(1.0).with_probability(1.0);
        // High eta = smaller mutations
        let high_eta = PolynomialMutation::new(100.0).with_probability(1.0);

        let mut low_total_change = 0.0;
        let mut high_total_change = 0.0;
        let trials = 1000;

        for _ in 0..trials {
            let mut genome_low = RealVector::new(vec![0.0]);
            let mut genome_high = RealVector::new(vec![0.0]);

            low_eta.mutate_bounded(&mut genome_low, &bounds, &mut rng);
            high_eta.mutate_bounded(&mut genome_high, &bounds, &mut rng);

            low_total_change += genome_low[0].abs();
            high_total_change += genome_high[0].abs();
        }

        assert!(
            low_total_change > high_total_change,
            "Low eta should produce larger average changes"
        );
    }

    #[test]
    fn test_gaussian_mutation_changes_genome() {
        let mut rng = rand::thread_rng();
        let original = RealVector::zeros(10);
        let mut genome = original.clone();

        let mutation = GaussianMutation::new(0.1).with_probability(1.0);
        mutation.mutate(&mut genome, &mut rng);

        assert_ne!(genome, original);
    }

    #[test]
    fn test_gaussian_mutation_bounded() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, 10);

        for _ in 0..100 {
            let mut genome = RealVector::zeros(10);
            let mutation = GaussianMutation::new(10.0).with_probability(1.0);
            mutation.mutate_bounded(&mut genome, &bounds, &mut rng);

            for (i, &gene) in genome.genes().iter().enumerate() {
                let bound = bounds.get(i).unwrap();
                assert!(gene >= bound.min && gene <= bound.max);
            }
        }
    }

    #[test]
    fn test_uniform_mutation() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, 10);

        for _ in 0..100 {
            let mut genome = RealVector::zeros(10);
            let mutation = UniformMutation::new().with_probability(1.0);
            mutation.mutate_bounded(&mut genome, &bounds, &mut rng);

            for (i, &gene) in genome.genes().iter().enumerate() {
                let bound = bounds.get(i).unwrap();
                assert!(gene >= bound.min && gene <= bound.max);
            }
        }
    }

    #[test]
    fn test_bit_flip_mutation() {
        let mut rng = rand::thread_rng();
        let original = BitString::zeros(100);
        let mut genome = original.clone();

        let mutation = BitFlipMutation::new().with_probability(0.5);
        mutation.mutate(&mut genome, &mut rng);

        // About half should be flipped
        let flipped = genome.count_ones();
        assert!(
            flipped > 20 && flipped < 80,
            "Expected ~50 flips, got {}",
            flipped
        );
    }

    #[test]
    fn test_bit_flip_mutation_default_probability() {
        let mut rng = rand::thread_rng();
        let original = BitString::zeros(100);
        let mut genome = original.clone();

        let mutation = BitFlipMutation::new(); // 1/n probability
        mutation.mutate(&mut genome, &mut rng);

        // With 1/100 probability, expect ~1 flip on average
        // But due to randomness, we just check some change occurred
        // over multiple trials
        let mut total_flips = 0;
        for _ in 0..100 {
            let mut g = BitString::zeros(100);
            mutation.mutate(&mut g, &mut rng);
            total_flips += g.count_ones();
        }

        // Average should be close to 1
        let avg = total_flips as f64 / 100.0;
        assert!(avg > 0.5 && avg < 2.0, "Expected avg ~1, got {}", avg);
    }

    #[test]
    fn test_swap_mutation() {
        let mut rng = rand::thread_rng();
        let mut genome = RealVector::new(vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let mutation = SwapMutation::new();
        mutation.mutate(&mut genome, &mut rng);

        // The sum should be preserved
        let sum: f64 = genome.genes().iter().sum();
        assert_relative_eq!(sum, 10.0);
    }

    #[test]
    fn test_swap_mutation_multiple() {
        let mut rng = rand::thread_rng();
        let original: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut genome = RealVector::new(original.clone());

        let mutation = SwapMutation::with_swaps(5);
        mutation.mutate(&mut genome, &mut rng);

        // Sum should be preserved
        let sum: f64 = genome.genes().iter().sum();
        assert_relative_eq!(sum, 45.0);
    }

    #[test]
    fn test_scramble_mutation() {
        let mut rng = rand::thread_rng();
        let original: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut genome = RealVector::new(original.clone());

        let mutation = ScrambleMutation::new();
        mutation.mutate(&mut genome, &mut rng);

        // Sum should be preserved
        let sum: f64 = genome.genes().iter().sum();
        assert_relative_eq!(sum, 45.0);

        // All values should still be present
        let mut sorted: Vec<f64> = genome.genes().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, original);
    }

    #[test]
    fn test_scramble_mutation_bitstring() {
        let mut rng = rand::thread_rng();
        let original = BitString::new(vec![true, true, true, false, false, false, true, false]);
        let mut genome = original.clone();

        let mutation = ScrambleMutation::new();
        mutation.mutate(&mut genome, &mut rng);

        // Count should be preserved
        assert_eq!(genome.count_ones(), original.count_ones());
    }

    // =========================================================================
    // Permutation Mutation Tests
    // =========================================================================

    #[test]
    fn test_permutation_swap_mutation() {
        let mut rng = rand::thread_rng();
        let original = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let mut genome = original.clone();

        let mutation = PermutationSwapMutation::new();
        mutation.mutate(&mut genome, &mut rng);

        // Should still be valid permutation
        assert!(genome.is_valid_permutation());
        assert_eq!(genome.dimension(), 8);
    }

    #[test]
    fn test_permutation_swap_mutation_multiple() {
        let mut rng = rand::thread_rng();
        let original = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut genome = original.clone();

        let mutation = PermutationSwapMutation::with_swaps(5);
        mutation.mutate(&mut genome, &mut rng);

        // Should still be valid permutation
        assert!(genome.is_valid_permutation());
        assert_eq!(genome.dimension(), 10);
    }

    #[test]
    fn test_insert_mutation() {
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);

            let mutation = InsertMutation::new();
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
            assert_eq!(genome.dimension(), 8);
        }
    }

    #[test]
    fn test_inversion_mutation() {
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);

            let mutation = InversionMutation::new();
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
            assert_eq!(genome.dimension(), 8);
        }
    }

    #[test]
    fn test_inversion_mutation_reverses_segment() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let mutation = InversionMutation::new();
        mutation.mutate(&mut genome, &mut rng);

        // Should still be valid permutation
        assert!(genome.is_valid_permutation());
    }

    #[test]
    fn test_permutation_scramble_mutation() {
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);

            let mutation = PermutationScrambleMutation::new();
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
            assert_eq!(genome.dimension(), 8);
        }
    }

    #[test]
    fn test_displacement_mutation() {
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

            let mutation = DisplacementMutation::new();
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
            assert_eq!(genome.dimension(), 10);
        }
    }

    #[test]
    fn test_adaptive_permutation_mutation() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);

            let mutation = AdaptivePermutationMutation::new();
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
            assert_eq!(genome.dimension(), 8);
        }
    }

    #[test]
    fn test_adaptive_permutation_mutation_custom_probs() {
        let mut rng = rand::thread_rng();

        // Test with only inversion mutation
        let mutation = AdaptivePermutationMutation::with_probs(0.0, 0.0, 1.0, 0.0);

        for _ in 0..50 {
            let mut genome = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
            mutation.mutate(&mut genome, &mut rng);

            assert!(genome.is_valid_permutation());
        }
    }
}
