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

    /// Absolute numerical-underflow floor for step sizes.
    ///
    /// This is *only* a guard against σ decaying to a literal zero /
    /// degenerate delta distribution — it does **not** prevent premature
    /// convergence (1e-10 is many orders of magnitude below typical domain
    /// scales, so the population has long since stagnated before this floor
    /// ever engages). For a floor that actually guards against premature
    /// step-size collapse, pass a problem-scaled `min_sigma` to
    /// [`StrategyParams::mutate`] (regression: EV-62).
    const SIGMA_UNDERFLOW_FLOOR: f64 = 1e-10;

    /// Mutate the strategy parameters using the log-normal self-adaptation rule.
    ///
    /// Following Beyer & Schwefel (2002) and Eiben & Smith (*Introduction to
    /// Evolutionary Computing* §4.4.2), the uncorrelated-mutation update with
    /// `n` step sizes is
    ///
    /// ```text
    /// σ_i' = σ_i · exp(τ' · N(0,1) + τ · N_i(0,1))
    /// ```
    ///
    /// where `N(0,1)` is drawn **once per individual** (shared across all
    /// coordinates) and `N_i(0,1)` is drawn independently per coordinate, with:
    /// * `τ' = 1/√(2n)` on the shared/global deviate — the *smaller* coefficient,
    ///   because its effect is coherent across all `n` coordinates, and
    /// * `τ = 1/√(2√n)` on the per-coordinate deviate — the *larger* coefficient.
    ///
    /// For the single-step-size (isotropic) case the standard rule is
    /// `σ' = σ · exp(τ₀ · N(0,1))` with `τ₀ = 1/√n` (Schwefel).
    ///
    /// `min_sigma` is a caller-supplied, problem-scaled lower bound on the step
    /// size; the effective floor is `max(min_sigma, SIGMA_UNDERFLOW_FLOOR)`.
    pub fn mutate<R: Rng>(&mut self, n: usize, min_sigma: f64, rng: &mut R) {
        // Global (once-per-individual) learning rate τ' = 1/√(2n).
        let tau_prime = 1.0 / (2.0 * n as f64).sqrt();
        // Per-coordinate learning rate τ = 1/√(2√n).
        let tau = 1.0 / (2.0 * (n as f64).sqrt()).sqrt();
        // Single-step-size (isotropic) learning rate τ₀ = 1/√n.
        let tau_0 = 1.0 / (n as f64).sqrt();
        let floor = min_sigma.max(Self::SIGMA_UNDERFLOW_FLOOR);
        let n0: f64 = rng.sample(StandardNormal);

        match self {
            Self::Isotropic(sigma) => {
                *sigma *= (tau_0 * n0).exp();
                *sigma = sigma.max(floor);
            }
            Self::NonIsotropic(sigmas) => {
                for sigma in sigmas.iter_mut() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(floor);
                }
            }
            Self::Correlated { sigmas, rotations } => {
                // Update step sizes
                for sigma in sigmas.iter_mut() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(floor);
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
    /// Global (once-per-individual) learning rate τ' = 1/√(2n)
    pub tau_prime: f64,
    /// Per-coordinate learning rate τ = 1/√(2√n)
    pub tau: f64,
    /// Rotation angle step β
    pub beta: f64,
}

impl LearningRates {
    /// Create default learning rates for dimension n.
    ///
    /// Per Beyer & Schwefel (2002): the global (shared) deviate carries the
    /// smaller rate `τ' = 1/√(2n)`, and each per-coordinate deviate carries the
    /// larger rate `τ = 1/√(2√n)` (regression: EV-05, EV-24).
    pub fn for_dimension(n: usize) -> Self {
        Self {
            tau_prime: 1.0 / (2.0 * n as f64).sqrt(),
            tau: 1.0 / (2.0 * (n as f64).sqrt()).sqrt(),
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
        params.mutate(10, 0.0, &mut rng);

        // Sigma should remain positive after mutation
        assert!(params.get_sigma(0) > 0.0);
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
    fn test_strategy_params_underflow_floor() {
        let mut params = StrategyParams::isotropic(1e-20);
        let mut rng = rand::thread_rng();

        // Even with tiny initial sigma and no problem-scaled floor, sigma must
        // not underflow below the absolute floor.
        for _ in 0..100 {
            params.mutate(10, 0.0, &mut rng);
        }

        assert!(params.get_sigma(0) >= StrategyParams::SIGMA_UNDERFLOW_FLOOR);
    }

    /// regression: EV-62 — a configurable `min_sigma` must actually floor the
    /// step size (the old code only had an absolute 1e-10 underflow guard that
    /// did nothing to prevent premature collapse). Pre-fix `mutate` had no
    /// `min_sigma` parameter and would let σ decay far below any meaningful
    /// problem scale.
    #[test]
    fn test_configurable_min_sigma_prevents_collapse() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let min_sigma = 0.1;

        // Isotropic
        let mut iso = StrategyParams::isotropic(1.0);
        for _ in 0..10_000 {
            iso.mutate(10, min_sigma, &mut rng);
            assert!(
                iso.get_sigma(0) >= min_sigma,
                "isotropic sigma {} fell below configured min_sigma {}",
                iso.get_sigma(0),
                min_sigma
            );
        }

        // Non-isotropic: every coordinate must respect the floor.
        let mut aniso = StrategyParams::non_isotropic(vec![1.0; 5]);
        for _ in 0..10_000 {
            aniso.mutate(5, min_sigma, &mut rng);
            for i in 0..5 {
                assert!(aniso.get_sigma(i) >= min_sigma);
            }
        }
    }

    /// regression: EV-05 / EV-24 — the shared (once-per-individual) deviate must
    /// carry the smaller coefficient τ' = 1/√(2n) and each per-coordinate
    /// deviate the larger τ = 1/√(2√n). The cross-coordinate covariance of
    /// ln(σ_i'/σ_i) equals (coefficient on the shared deviate)²; with the
    /// pre-fix swapped rates it would be 1/(2√n) ≈ 0.158 for n=10 instead of the
    /// correct 1/(2n) = 0.05.
    #[test]
    fn test_non_isotropic_learning_rate_assignment() {
        use rand::SeedableRng;
        let n = 10usize;
        let samples = 60_000usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);

        // Collect ln-ratios for coordinates 0 and 1 across many independent
        // single mutations (each starts from σ = 1).
        let mut r0 = Vec::with_capacity(samples);
        let mut r1 = Vec::with_capacity(samples);
        for _ in 0..samples {
            let mut p = StrategyParams::non_isotropic(vec![1.0; n]);
            p.mutate(n, 0.0, &mut rng);
            r0.push(p.get_sigma(0).ln());
            r1.push(p.get_sigma(1).ln());
        }

        let mean0 = r0.iter().sum::<f64>() / samples as f64;
        let mean1 = r1.iter().sum::<f64>() / samples as f64;
        let cov: f64 = r0
            .iter()
            .zip(r1.iter())
            .map(|(a, b)| (a - mean0) * (b - mean1))
            .sum::<f64>()
            / samples as f64;

        // Cov equals (coefficient on shared deviate)². Correct: 1/(2n) = 0.05.
        // Buggy (swapped): 1/(2√n) ≈ 0.158.
        let expected = 1.0 / (2.0 * n as f64);
        assert!(
            (cov - expected).abs() < 0.02,
            "cross-coordinate covariance {} should be ≈ {} (shared deviate rate²), \
             not the swapped 1/(2√n) ≈ {}",
            cov,
            expected,
            1.0 / (2.0 * (n as f64).sqrt())
        );
    }

    /// regression: EV-97 — the isotropic (single-step-size) case must use
    /// τ₀ = 1/√n, so Var(ln σ'/σ) = 1/n. The pre-fix code used 1/√(2√n), giving
    /// variance 1/(2√n) ≈ 0.158 for n=10 instead of the correct 0.1.
    #[test]
    fn test_isotropic_learning_rate() {
        use rand::SeedableRng;
        let n = 10usize;
        let samples = 60_000usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        let mut r = Vec::with_capacity(samples);
        for _ in 0..samples {
            let mut p = StrategyParams::isotropic(1.0);
            p.mutate(n, 0.0, &mut rng);
            r.push(p.get_sigma(0).ln());
        }
        let mean = r.iter().sum::<f64>() / samples as f64;
        let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples as f64;

        let expected = 1.0 / n as f64; // τ₀² = 1/n
        assert!(
            (var - expected).abs() < 0.02,
            "Var(ln σ'/σ) {} should be ≈ 1/n = {} (τ₀ = 1/√n), not 1/(2√n) ≈ {}",
            var,
            expected,
            1.0 / (2.0 * (n as f64).sqrt())
        );
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

    /// regression: EV-05 / EV-24 — the global rate τ' must be the *smaller*
    /// 1/√(2n) and the per-coordinate rate τ the *larger* 1/√(2√n). The pre-fix
    /// `for_dimension` had these two values swapped between the fields.
    #[test]
    fn test_learning_rates() {
        let rates = LearningRates::for_dimension(10);

        // Global (shared) rate τ' = 1/√(2*10) ≈ 0.2236
        assert!((rates.tau_prime - 0.2236).abs() < 0.01);

        // Per-coordinate rate τ = 1/√(2*√10) ≈ 0.3976
        assert!((rates.tau - 0.3976).abs() < 0.01);

        // The global (shared) rate must be strictly smaller than the
        // per-coordinate rate.
        assert!(rates.tau_prime < rates.tau);
    }
}
