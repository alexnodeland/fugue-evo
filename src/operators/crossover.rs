//! Crossover operators
//!
//! This module provides various crossover operators for genetic algorithms.

use rand::Rng;

use crate::error::{OperatorError, OperatorResult};
use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::{BinaryGenome, EvolutionaryGenome, RealValuedGenome};
use crate::operators::traits::{BoundedCrossoverOperator, CrossoverOperator};

/// Simulated Binary Crossover (SBX)
///
/// SBX generates offspring from parents using a spread factor that
/// simulates single-point crossover for binary strings.
///
/// Reference: Deb, K., & Agrawal, R. B. (1995). Simulated Binary Crossover
/// for Continuous Search Space.
#[derive(Clone, Debug)]
pub struct SbxCrossover {
    /// Distribution index (typically 2-20)
    /// Higher values = offspring closer to parents
    pub eta: f64,
    /// Per-gene crossover probability
    pub crossover_probability: f64,
}

impl SbxCrossover {
    /// Create a new SBX crossover with the given distribution index
    pub fn new(eta: f64) -> Self {
        assert!(eta >= 0.0, "Distribution index must be non-negative");
        Self {
            eta,
            crossover_probability: 0.9,
        }
    }

    /// Set the per-gene crossover probability
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            probability >= 0.0 && probability <= 1.0,
            "Probability must be in [0, 1]"
        );
        self.crossover_probability = probability;
        self
    }

    /// Compute the spread factor β from a uniform random value
    fn spread_factor(&self, u: f64) -> f64 {
        if u <= 0.5 {
            (2.0 * u).powf(1.0 / (self.eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta + 1.0))
        }
    }

    /// Apply SBX crossover to two f64 slices
    fn apply_sbx<R: Rng>(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        bounds: Option<&MultiBounds>,
        rng: &mut R,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut child1: Vec<f64> = parent1.to_vec();
        let mut child2: Vec<f64> = parent2.to_vec();

        for i in 0..parent1.len() {
            if rng.gen::<f64>() < self.crossover_probability {
                let x1 = parent1[i];
                let x2 = parent2[i];

                // Only apply if parents differ sufficiently
                if (x1 - x2).abs() > 1e-14 {
                    let u = rng.gen::<f64>();
                    let beta = self.spread_factor(u);

                    child1[i] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
                    child2[i] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2);

                    // Apply bounds if provided
                    if let Some(b) = bounds {
                        if let Some(bound) = b.get(i) {
                            child1[i] = bound.clamp(child1[i]);
                            child2[i] = bound.clamp(child2[i]);
                        }
                    }
                }
            }
        }

        (child1, child2)
    }
}

impl CrossoverOperator<RealVector> for SbxCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let (child1_genes, child2_genes) =
            self.apply_sbx(parent1.genes(), parent2.genes(), None, rng);

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }

    fn crossover_probability(&self) -> f64 {
        self.crossover_probability
    }
}

impl BoundedCrossoverOperator<RealVector> for SbxCrossover {
    fn crossover_bounded<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let (child1_genes, child2_genes) =
            self.apply_sbx(parent1.genes(), parent2.genes(), Some(bounds), rng);

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Blend Crossover (BLX-α)
///
/// Creates offspring within an extended range defined by parents.
#[derive(Clone, Debug)]
pub struct BlxAlphaCrossover {
    /// Extension factor (typically 0.5)
    pub alpha: f64,
}

impl BlxAlphaCrossover {
    /// Create a new BLX-α crossover
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0.0, "Alpha must be non-negative");
        Self { alpha }
    }

    /// Create with default alpha = 0.5
    pub fn default_alpha() -> Self {
        Self::new(0.5)
    }
}

impl CrossoverOperator<RealVector> for BlxAlphaCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let mut child1_genes = Vec::with_capacity(parent1.dimension());
        let mut child2_genes = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            let x1 = parent1[i];
            let x2 = parent2[i];

            let min_val = x1.min(x2);
            let max_val = x1.max(x2);
            let range = max_val - min_val;

            let low = min_val - self.alpha * range;
            let high = max_val + self.alpha * range;

            child1_genes.push(rng.gen_range(low..=high));
            child2_genes.push(rng.gen_range(low..=high));
        }

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Uniform crossover for bit strings
///
/// Each bit is independently chosen from either parent with equal probability.
#[derive(Clone, Debug)]
pub struct UniformCrossover {
    /// Probability of choosing from parent1 (default: 0.5)
    pub bias: f64,
}

impl UniformCrossover {
    /// Create a new uniform crossover
    pub fn new() -> Self {
        Self { bias: 0.5 }
    }

    /// Create with a specific bias towards parent1
    pub fn with_bias(bias: f64) -> Self {
        assert!(
            bias >= 0.0 && bias <= 1.0,
            "Bias must be in [0, 1]"
        );
        Self { bias }
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossoverOperator<BitString> for UniformCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let mut child1_bits = Vec::with_capacity(parent1.dimension());
        let mut child2_bits = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            if rng.gen::<f64>() < self.bias {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// One-point crossover for bit strings
#[derive(Clone, Debug, Default)]
pub struct OnePointCrossover;

impl OnePointCrossover {
    /// Create a new one-point crossover
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<BitString> for OnePointCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let n = parent1.dimension();
        if n == 0 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        let crossover_point = rng.gen_range(0..n);

        let mut child1_bits = Vec::with_capacity(n);
        let mut child2_bits = Vec::with_capacity(n);

        for i in 0..n {
            if i < crossover_point {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Two-point crossover for bit strings
#[derive(Clone, Debug, Default)]
pub struct TwoPointCrossover;

impl TwoPointCrossover {
    /// Create a new two-point crossover
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<BitString> for TwoPointCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let n = parent1.dimension();
        if n < 2 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        let mut point1 = rng.gen_range(0..n);
        let mut point2 = rng.gen_range(0..n);
        if point1 > point2 {
            std::mem::swap(&mut point1, &mut point2);
        }

        let mut child1_bits = Vec::with_capacity(n);
        let mut child2_bits = Vec::with_capacity(n);

        for i in 0..n {
            if i < point1 || i >= point2 {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Arithmetic crossover for real-valued genomes
///
/// Creates offspring as weighted averages of parents.
#[derive(Clone, Debug)]
pub struct ArithmeticCrossover {
    /// Weight for parent1 (parent2 weight = 1 - weight)
    pub weight: f64,
}

impl ArithmeticCrossover {
    /// Create a new arithmetic crossover with the given weight
    pub fn new(weight: f64) -> Self {
        assert!(
            weight >= 0.0 && weight <= 1.0,
            "Weight must be in [0, 1]"
        );
        Self { weight }
    }

    /// Create with uniform weight (0.5)
    pub fn uniform() -> Self {
        Self::new(0.5)
    }
}

impl CrossoverOperator<RealVector> for ArithmeticCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        _rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let w = self.weight;
        let mut child1_genes = Vec::with_capacity(parent1.dimension());
        let mut child2_genes = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            child1_genes.push(w * parent1[i] + (1.0 - w) * parent2[i]);
            child2_genes.push((1.0 - w) * parent1[i] + w * parent2[i]);
        }

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sbx_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0, 1.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.dimension(), 3);
        assert_eq!(child2.dimension(), 3);
    }

    #[test]
    fn test_sbx_with_bounds() {
        let mut rng = rand::thread_rng();
        // Use parents within bounds
        let parent1 = RealVector::new(vec![-0.3, -0.2]);
        let parent2 = RealVector::new(vec![0.3, 0.4]);
        let bounds = MultiBounds::symmetric(0.5, 2);

        let sbx = SbxCrossover::new(2.0).with_probability(1.0); // Low eta = more spread, always crossover

        // Run multiple times and check bounds
        for _ in 0..100 {
            let result = sbx.crossover_bounded(&parent1, &parent2, &bounds, &mut rng);
            let (child1, child2) = result.genome().unwrap();

            for gene in child1.genes() {
                assert!(
                    *gene >= -0.5 && *gene <= 0.5,
                    "gene {} out of bounds [-0.5, 0.5]",
                    gene
                );
            }
            for gene in child2.genes() {
                assert!(
                    *gene >= -0.5 && *gene <= 0.5,
                    "gene {} out of bounds [-0.5, 0.5]",
                    gene
                );
            }
        }
    }

    #[test]
    fn test_sbx_spread_factor() {
        let sbx = SbxCrossover::new(20.0);

        // At u = 0.5, β should be 1.0
        let beta = sbx.spread_factor(0.5);
        assert_relative_eq!(beta, 1.0, epsilon = 1e-10);

        // β should be symmetric around 0.5
        let beta_low = sbx.spread_factor(0.25);
        let beta_high = sbx.spread_factor(0.75);
        assert_relative_eq!(beta_low, 1.0 / beta_high, epsilon = 1e-10);
    }

    #[test]
    fn test_sbx_identical_parents() {
        let mut rng = rand::thread_rng();
        let parent = RealVector::new(vec![1.0, 2.0, 3.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent, &parent, &mut rng);

        let (child1, child2) = result.genome().unwrap();
        // With identical parents, children should equal parents
        assert_eq!(child1.genes(), parent.genes());
        assert_eq!(child2.genes(), parent.genes());
    }

    #[test]
    fn test_sbx_dimension_mismatch() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0]);
        let parent2 = RealVector::new(vec![1.0, 2.0, 3.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent1, &parent2, &mut rng);

        assert!(!result.is_ok());
    }

    #[test]
    fn test_blx_alpha_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0]);

        let blx = BlxAlphaCrossover::new(0.5);
        let result = blx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.dimension(), 2);
        assert_eq!(child2.dimension(), 2);
    }

    #[test]
    fn test_blx_alpha_range() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0]);
        let parent2 = RealVector::new(vec![1.0]);

        let blx = BlxAlphaCrossover::new(0.0); // No extension

        for _ in 0..100 {
            let result = blx.crossover(&parent1, &parent2, &mut rng);
            let (child1, child2) = result.genome().unwrap();

            // With α = 0, offspring should be in [0, 1]
            assert!(child1[0] >= 0.0 && child1[0] <= 1.0);
            assert!(child2[0] >= 0.0 && child2[0] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_crossover_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::new(vec![true, true, true, true]);
        let parent2 = BitString::new(vec![false, false, false, false]);

        let ux = UniformCrossover::new();
        let result = ux.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.len(), 4);
        assert_eq!(child2.len(), 4);
    }

    #[test]
    fn test_uniform_crossover_complementary() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::new(vec![true, true, true, true]);
        let parent2 = BitString::new(vec![false, false, false, false]);

        let ux = UniformCrossover::new();
        let result = ux.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // Children should be complementary
        for i in 0..4 {
            assert_ne!(child1[i], child2[i]);
        }
    }

    #[test]
    fn test_one_point_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::ones(10);
        let parent2 = BitString::zeros(10);

        let opx = OnePointCrossover::new();
        let result = opx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();

        // Children should have a contiguous segment from each parent
        // and the children should be complementary
        for i in 0..10 {
            assert_ne!(child1[i], child2[i]);
        }
    }

    #[test]
    fn test_two_point_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::ones(10);
        let parent2 = BitString::zeros(10);

        let tpx = TwoPointCrossover::new();
        let result = tpx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.len(), 10);
        assert_eq!(child2.len(), 10);
    }

    #[test]
    fn test_arithmetic_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0]);

        let ax = ArithmeticCrossover::new(0.5);
        let result = ax.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // With 0.5 weight, both children should be at midpoint
        for gene in child1.genes() {
            assert_relative_eq!(*gene, 0.5);
        }
        for gene in child2.genes() {
            assert_relative_eq!(*gene, 0.5);
        }
    }

    #[test]
    fn test_arithmetic_crossover_weighted() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0]);
        let parent2 = RealVector::new(vec![1.0]);

        let ax = ArithmeticCrossover::new(0.75);
        let result = ax.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // child1 = 0.75 * 0 + 0.25 * 1 = 0.25
        // child2 = 0.25 * 0 + 0.75 * 1 = 0.75
        assert_relative_eq!(child1[0], 0.25);
        assert_relative_eq!(child2[0], 0.75);
    }
}
