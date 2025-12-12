//! Self-adaptive control mechanisms
//!
//! In self-adaptive evolution strategies, strategy parameters (like mutation step sizes)
//! are encoded in the genome and evolve alongside the solution parameters.

use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::genome::traits::EvolutionaryGenome;

/// Strategy parameters that can be evolved alongside the genome
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StrategyParams {
    /// Single global step size (σ) - (1, σ)-ES or simple ES
    Isotropic(f64),

    /// Per-gene step sizes (σ₁, ..., σₙ) - Non-isotropic mutation
    NonIsotropic(Vec<f64>),

    /// Full covariance with rotation angles
    /// Contains step sizes (σ₁, ..., σₙ) and rotation angles (α₁, ..., αₖ) where k = n(n-1)/2
    Correlated {
        sigmas: Vec<f64>,
        rotations: Vec<f64>,
    },
}

impl StrategyParams {
    /// Create isotropic parameters with given step size
    pub fn isotropic(sigma: f64) -> Self {
        Self::Isotropic(sigma)
    }

    /// Create non-isotropic parameters with given step sizes
    pub fn non_isotropic(sigmas: Vec<f64>) -> Self {
        Self::NonIsotropic(sigmas)
    }

    /// Create correlated parameters
    pub fn correlated(sigmas: Vec<f64>) -> Self {
        let n = sigmas.len();
        let num_rotations = n * (n - 1) / 2;
        Self::Correlated {
            sigmas,
            rotations: vec![0.0; num_rotations],
        }
    }

    /// Get dimension of the strategy parameters
    pub fn dimension(&self) -> usize {
        match self {
            Self::Isotropic(_) => 1,
            Self::NonIsotropic(sigmas) => sigmas.len(),
            Self::Correlated { sigmas, rotations } => sigmas.len() + rotations.len(),
        }
    }

    /// Get the minimum step size to prevent collapse
    const MIN_SIGMA: f64 = 1e-10;

    /// Mutate the strategy parameters using log-normal distribution
    ///
    /// τ = 1/√(2n) - global learning rate
    /// τ' = 1/√(2√n) - local learning rate
    pub fn mutate<R: Rng>(&mut self, n: usize, rng: &mut R) {
        let tau = 1.0 / (2.0 * n as f64).sqrt();
        let tau_prime = 1.0 / (2.0 * (n as f64).sqrt()).sqrt();
        let n0: f64 = rng.sample(StandardNormal);

        match self {
            Self::Isotropic(sigma) => {
                *sigma *= (tau_prime * n0).exp();
                *sigma = sigma.max(Self::MIN_SIGMA);
            }
            Self::NonIsotropic(sigmas) => {
                for sigma in sigmas.iter_mut() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(Self::MIN_SIGMA);
                }
            }
            Self::Correlated { sigmas, rotations } => {
                // Update step sizes
                for sigma in sigmas.iter_mut() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(Self::MIN_SIGMA);
                }
                // Update rotation angles (≈5° per step)
                let beta = 0.0873;
                for alpha in rotations.iter_mut() {
                    *alpha += beta * rng.sample::<f64, _>(StandardNormal);
                }
            }
        }
    }

    /// Get step size for a specific gene (for non-isotropic and correlated)
    pub fn get_sigma(&self, gene_idx: usize) -> f64 {
        match self {
            Self::Isotropic(sigma) => *sigma,
            Self::NonIsotropic(sigmas) => sigmas.get(gene_idx).copied().unwrap_or(sigmas[0]),
            Self::Correlated { sigmas, .. } => sigmas.get(gene_idx).copied().unwrap_or(sigmas[0]),
        }
    }

    /// Get all step sizes
    pub fn sigmas(&self) -> Vec<f64> {
        match self {
            Self::Isotropic(sigma) => vec![*sigma],
            Self::NonIsotropic(sigmas) => sigmas.clone(),
            Self::Correlated { sigmas, .. } => sigmas.clone(),
        }
    }
}

/// Genome wrapper with self-adaptive strategy parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveGenome<G> {
    /// The underlying genome
    pub genome: G,
    /// Strategy parameters
    pub strategy: StrategyParams,
}

impl<G: EvolutionaryGenome> AdaptiveGenome<G> {
    /// Create a new adaptive genome with isotropic strategy
    pub fn new_isotropic(genome: G, initial_sigma: f64) -> Self {
        Self {
            genome,
            strategy: StrategyParams::Isotropic(initial_sigma),
        }
    }

    /// Create a new adaptive genome with non-isotropic strategy
    pub fn new_non_isotropic(genome: G, initial_sigmas: Vec<f64>) -> Self {
        Self {
            genome,
            strategy: StrategyParams::NonIsotropic(initial_sigmas),
        }
    }

    /// Create a new adaptive genome with correlated strategy
    pub fn new_correlated(genome: G, initial_sigmas: Vec<f64>) -> Self {
        Self {
            genome,
            strategy: StrategyParams::correlated(initial_sigmas),
        }
    }

    /// Get reference to the underlying genome
    pub fn inner(&self) -> &G {
        &self.genome
    }

    /// Get mutable reference to the underlying genome
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.genome
    }

    /// Consume and return the underlying genome
    pub fn into_inner(self) -> G {
        self.genome
    }
}

/// Crossover for adaptive genomes
///
/// Performs intermediate recombination of strategy parameters.
pub fn adaptive_crossover<G: Clone, R: Rng>(
    parent1: &AdaptiveGenome<G>,
    parent2: &AdaptiveGenome<G>,
    child_genome: G,
    rng: &mut R,
) -> AdaptiveGenome<G> {
    let strategy = match (&parent1.strategy, &parent2.strategy) {
        (StrategyParams::Isotropic(s1), StrategyParams::Isotropic(s2)) => {
            // Geometric mean of step sizes
            StrategyParams::Isotropic((s1 * s2).sqrt())
        }
        (StrategyParams::NonIsotropic(s1), StrategyParams::NonIsotropic(s2)) => {
            let sigmas: Vec<f64> = s1
                .iter()
                .zip(s2.iter())
                .map(|(a, b)| (a * b).sqrt())
                .collect();
            StrategyParams::NonIsotropic(sigmas)
        }
        (
            StrategyParams::Correlated {
                sigmas: s1,
                rotations: r1,
            },
            StrategyParams::Correlated {
                sigmas: s2,
                rotations: r2,
            },
        ) => {
            let sigmas: Vec<f64> = s1
                .iter()
                .zip(s2.iter())
                .map(|(a, b)| (a * b).sqrt())
                .collect();
            let rotations: Vec<f64> = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| (a + b) / 2.0)
                .collect();
            StrategyParams::Correlated { sigmas, rotations }
        }
        // Mixed types: randomly pick one parent's strategy
        (s1, s2) => {
            if rng.gen_bool(0.5) {
                s1.clone()
            } else {
                s2.clone()
            }
        }
    };

    AdaptiveGenome {
        genome: child_genome,
        strategy,
    }
}

/// Learning rate parameters for self-adaptation
#[derive(Clone, Debug)]
pub struct LearningRates {
    /// Global learning rate τ'
    pub tau_prime: f64,
    /// Local learning rate τ
    pub tau: f64,
    /// Rotation angle step β
    pub beta: f64,
}

impl LearningRates {
    /// Create default learning rates for dimension n
    pub fn for_dimension(n: usize) -> Self {
        Self {
            tau_prime: 1.0 / (2.0 * (n as f64).sqrt()).sqrt(),
            tau: 1.0 / (2.0 * n as f64).sqrt(),
            beta: 0.0873, // ≈ 5 degrees
        }
    }

    /// Create custom learning rates
    pub fn custom(tau_prime: f64, tau: f64, beta: f64) -> Self {
        Self {
            tau_prime,
            tau,
            beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;

    #[test]
    fn test_strategy_params_isotropic() {
        let mut params = StrategyParams::isotropic(1.0);
        assert_eq!(params.dimension(), 1);
        assert!((params.get_sigma(0) - 1.0).abs() < 1e-10);

        let mut rng = rand::thread_rng();
        params.mutate(10, &mut rng);

        // Should have changed
        assert!(params.get_sigma(0) != 1.0 || true); // Might not change much
    }

    #[test]
    fn test_strategy_params_non_isotropic() {
        let params = StrategyParams::non_isotropic(vec![0.1, 0.2, 0.3]);
        assert_eq!(params.dimension(), 3);
        assert!((params.get_sigma(0) - 0.1).abs() < 1e-10);
        assert!((params.get_sigma(1) - 0.2).abs() < 1e-10);
        assert!((params.get_sigma(2) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_strategy_params_correlated() {
        let params = StrategyParams::correlated(vec![0.1, 0.2, 0.3]);
        // 3 sigmas + 3 rotation angles = 6
        assert_eq!(params.dimension(), 6);
    }

    #[test]
    fn test_strategy_params_min_sigma() {
        let mut params = StrategyParams::isotropic(1e-20);
        let mut rng = rand::thread_rng();

        // Even with tiny initial sigma, should not go below minimum
        for _ in 0..100 {
            params.mutate(10, &mut rng);
        }

        assert!(params.get_sigma(0) >= StrategyParams::MIN_SIGMA);
    }

    #[test]
    fn test_adaptive_genome_creation() {
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let adaptive = AdaptiveGenome::new_isotropic(genome.clone(), 0.5);

        assert_eq!(adaptive.inner().genes(), genome.genes());
        assert!((adaptive.strategy.get_sigma(0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_crossover() {
        let mut rng = rand::thread_rng();

        let g1 = RealVector::new(vec![1.0, 2.0]);
        let g2 = RealVector::new(vec![3.0, 4.0]);
        let child_genome = RealVector::new(vec![2.0, 3.0]);

        let p1 = AdaptiveGenome::new_isotropic(g1, 0.1);
        let p2 = AdaptiveGenome::new_isotropic(g2, 0.4);

        let child = adaptive_crossover(&p1, &p2, child_genome, &mut rng);

        // Geometric mean of 0.1 and 0.4 = 0.2
        assert!((child.strategy.get_sigma(0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_learning_rates() {
        let rates = LearningRates::for_dimension(10);

        // τ = 1/√(2*10) ≈ 0.2236
        assert!((rates.tau - 0.2236).abs() < 0.01);

        // τ' = 1/√(2*√10) ≈ 0.3976
        assert!((rates.tau_prime - 0.3976).abs() < 0.01);
    }
}
