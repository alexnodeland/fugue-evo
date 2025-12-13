//! Checkpoint state structures
//!
//! Complete evolution state for checkpointing and recovery.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::diagnostics::GenerationStats;
use crate::genome::traits::EvolutionaryGenome;
use crate::population::individual::Individual;

/// Current checkpoint format version
pub const CHECKPOINT_VERSION: u32 = 1;

/// Complete evolution state for checkpointing
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Checkpoint<G>
where
    G: Clone + Serialize + EvolutionaryGenome,
{
    /// Schema version for forward compatibility
    pub version: u32,
    /// Current generation
    pub generation: usize,
    /// Total fitness evaluations
    pub evaluations: usize,
    /// Population with fitness values
    pub population: Vec<Individual<G>>,
    /// RNG state for reproducibility (serialized bytes)
    pub rng_state: Option<Vec<u8>>,
    /// Best individual found so far
    pub best: Option<Individual<G>>,
    /// Algorithm-specific state
    pub algorithm_state: AlgorithmState,
    /// Hyperparameter state if using adaptive learning
    pub hyperparameter_state: Option<HyperparameterState>,
    /// Statistics history
    pub statistics: Vec<GenerationStats>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl<G> Checkpoint<G>
where
    G: Clone + Serialize + EvolutionaryGenome,
{
    /// Create a new checkpoint
    pub fn new(generation: usize, population: Vec<Individual<G>>) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            generation,
            evaluations: 0,
            population,
            rng_state: None,
            best: None,
            algorithm_state: AlgorithmState::SimpleGA,
            hyperparameter_state: None,
            statistics: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the number of evaluations
    pub fn with_evaluations(mut self, evaluations: usize) -> Self {
        self.evaluations = evaluations;
        self
    }

    /// Set the best individual
    pub fn with_best(mut self, best: Individual<G>) -> Self {
        self.best = Some(best);
        self
    }

    /// Set algorithm-specific state
    pub fn with_algorithm_state(mut self, state: AlgorithmState) -> Self {
        self.algorithm_state = state;
        self
    }

    /// Set hyperparameter state
    pub fn with_hyperparameter_state(mut self, state: HyperparameterState) -> Self {
        self.hyperparameter_state = Some(state);
        self
    }

    /// Add statistics history
    pub fn with_statistics(mut self, stats: Vec<GenerationStats>) -> Self {
        self.statistics = stats;
        self
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set RNG state
    pub fn with_rng_state(mut self, state: Vec<u8>) -> Self {
        self.rng_state = Some(state);
        self
    }

    /// Check if checkpoint is compatible with current version
    pub fn is_compatible(&self) -> bool {
        self.version <= CHECKPOINT_VERSION
    }

    /// Get the checkpoint version
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// Algorithm-specific state variants
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AlgorithmState {
    /// Simple generational GA (no additional state)
    SimpleGA,
    /// Steady-state GA
    SteadyState { replacement_count: usize },
    /// CMA-ES state
    CmaEs(CmaEsCheckpointState),
    /// NSGA-II state
    Nsga2 { pareto_front_indices: Vec<usize> },
    /// HBGA state
    Hbga {
        population_params: Vec<f64>,
        temperature: f64,
    },
    /// Island model state
    Island {
        island_populations: Vec<Vec<usize>>,
        migration_count: usize,
    },
    /// Interactive GA state
    Interactive {
        /// Serialized aggregator state (JSON)
        aggregator_state: String,
        /// Number of pending evaluations
        pending_evaluations: usize,
        /// Evaluation mode
        evaluation_mode: String,
    },
    /// Custom algorithm state (JSON serialized)
    Custom(String),
}

/// CMA-ES checkpoint state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmaEsCheckpointState {
    /// Mean vector
    pub mean: Vec<f64>,
    /// Global step size
    pub sigma: f64,
    /// Covariance matrix (flattened row-major)
    pub covariance: Vec<f64>,
    /// Evolution path for sigma
    pub path_sigma: Vec<f64>,
    /// Evolution path for covariance
    pub path_c: Vec<f64>,
    /// Dimension
    pub dimension: usize,
}

/// Hyperparameter learning state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperparameterState {
    /// Mutation rate posterior (alpha, beta for Beta distribution)
    pub mutation_rate_posterior: Option<(f64, f64)>,
    /// Crossover probability posterior
    pub crossover_prob_posterior: Option<(f64, f64)>,
    /// Selection temperature posterior (shape, rate for Gamma)
    pub temperature_posterior: Option<(f64, f64)>,
    /// Step size posteriors (mu, sigma_sq for LogNormal)
    pub step_size_posteriors: Vec<(f64, f64)>,
    /// Operator selection weights
    pub operator_weights: Vec<f64>,
    /// History window for learning
    pub history_size: usize,
}

impl Default for HyperparameterState {
    fn default() -> Self {
        Self {
            mutation_rate_posterior: None,
            crossover_prob_posterior: None,
            temperature_posterior: None,
            step_size_posteriors: Vec::new(),
            operator_weights: Vec::new(),
            history_size: 100,
        }
    }
}

/// Builder for creating checkpoints
pub struct CheckpointBuilder<G>
where
    G: Clone + Serialize + EvolutionaryGenome,
{
    checkpoint: Checkpoint<G>,
}

impl<G> CheckpointBuilder<G>
where
    G: Clone + Serialize + EvolutionaryGenome,
{
    /// Create a new checkpoint builder
    pub fn new(generation: usize, population: Vec<Individual<G>>) -> Self {
        Self {
            checkpoint: Checkpoint::new(generation, population),
        }
    }

    /// Set evaluations count
    pub fn evaluations(mut self, count: usize) -> Self {
        self.checkpoint.evaluations = count;
        self
    }

    /// Set best individual
    pub fn best(mut self, individual: Individual<G>) -> Self {
        self.checkpoint.best = Some(individual);
        self
    }

    /// Set algorithm state
    pub fn algorithm_state(mut self, state: AlgorithmState) -> Self {
        self.checkpoint.algorithm_state = state;
        self
    }

    /// Set hyperparameter state
    pub fn hyperparameters(mut self, state: HyperparameterState) -> Self {
        self.checkpoint.hyperparameter_state = Some(state);
        self
    }

    /// Set statistics
    pub fn statistics(mut self, stats: Vec<GenerationStats>) -> Self {
        self.checkpoint.statistics = stats;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.checkpoint.metadata.insert(key.into(), value.into());
        self
    }

    /// Set RNG state
    pub fn rng_state(mut self, state: Vec<u8>) -> Self {
        self.checkpoint.rng_state = Some(state);
        self
    }

    /// Build the checkpoint
    pub fn build(self) -> Checkpoint<G> {
        self.checkpoint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;

    #[test]
    fn test_checkpoint_creation() {
        let population: Vec<Individual<RealVector>> = vec![
            Individual::new(RealVector::new(vec![1.0, 2.0])),
            Individual::new(RealVector::new(vec![3.0, 4.0])),
        ];

        let checkpoint = Checkpoint::new(10, population.clone());

        assert_eq!(checkpoint.version, CHECKPOINT_VERSION);
        assert_eq!(checkpoint.generation, 10);
        assert_eq!(checkpoint.population.len(), 2);
    }

    #[test]
    fn test_checkpoint_builder() {
        let population: Vec<Individual<RealVector>> =
            vec![Individual::new(RealVector::new(vec![1.0]))];

        let checkpoint = CheckpointBuilder::new(5, population)
            .evaluations(1000)
            .algorithm_state(AlgorithmState::SimpleGA)
            .metadata("experiment", "test_run")
            .build();

        assert_eq!(checkpoint.generation, 5);
        assert_eq!(checkpoint.evaluations, 1000);
        assert_eq!(
            checkpoint.metadata.get("experiment"),
            Some(&"test_run".to_string())
        );
    }

    #[test]
    fn test_checkpoint_compatibility() {
        let population: Vec<Individual<RealVector>> = vec![];
        let checkpoint = Checkpoint::new(0, population);

        assert!(checkpoint.is_compatible());
    }

    #[test]
    fn test_cmaes_checkpoint_state() {
        let state = CmaEsCheckpointState {
            mean: vec![0.0, 0.0],
            sigma: 1.0,
            covariance: vec![1.0, 0.0, 0.0, 1.0],
            path_sigma: vec![0.0, 0.0],
            path_c: vec![0.0, 0.0],
            dimension: 2,
        };

        let alg_state = AlgorithmState::CmaEs(state);
        if let AlgorithmState::CmaEs(s) = alg_state {
            assert_eq!(s.dimension, 2);
            assert_eq!(s.sigma, 1.0);
        } else {
            panic!("Expected CmaEs state");
        }
    }

    #[test]
    fn test_checkpoint_with_methods() {
        let population: Vec<Individual<RealVector>> =
            vec![Individual::new(RealVector::new(vec![1.0, 2.0]))];
        let best = Individual::with_fitness(RealVector::new(vec![0.5, 0.5]), 10.0);

        let checkpoint = Checkpoint::new(5, population)
            .with_evaluations(500)
            .with_best(best.clone())
            .with_algorithm_state(AlgorithmState::SteadyState {
                replacement_count: 10,
            })
            .with_metadata("run_id", "test123")
            .with_rng_state(vec![1, 2, 3, 4]);

        assert_eq!(checkpoint.evaluations, 500);
        assert!(checkpoint.best.is_some());
        assert_eq!(
            checkpoint.metadata.get("run_id"),
            Some(&"test123".to_string())
        );
        assert!(checkpoint.rng_state.is_some());
    }

    #[test]
    fn test_checkpoint_with_hyperparameters() {
        let population: Vec<Individual<RealVector>> = vec![];
        let hp_state = HyperparameterState {
            mutation_rate_posterior: Some((2.0, 8.0)),
            crossover_prob_posterior: Some((5.0, 5.0)),
            ..Default::default()
        };

        let checkpoint = Checkpoint::new(0, population).with_hyperparameter_state(hp_state);

        assert!(checkpoint.hyperparameter_state.is_some());
        let hp = checkpoint.hyperparameter_state.unwrap();
        assert_eq!(hp.mutation_rate_posterior, Some((2.0, 8.0)));
    }

    #[test]
    fn test_checkpoint_with_statistics() {
        use crate::diagnostics::{GenerationStats, TimingStats};

        let population: Vec<Individual<RealVector>> = vec![];
        let stats = vec![
            GenerationStats {
                generation: 0,
                evaluations: 100,
                best_fitness: 10.0,
                worst_fitness: 1.0,
                mean_fitness: 5.0,
                median_fitness: 5.0,
                fitness_std: 2.0,
                diversity: 0.5,
                timing: TimingStats::default(),
            },
            GenerationStats {
                generation: 1,
                evaluations: 200,
                best_fitness: 15.0,
                worst_fitness: 2.0,
                mean_fitness: 7.0,
                median_fitness: 7.0,
                fitness_std: 1.5,
                diversity: 0.4,
                timing: TimingStats::default(),
            },
        ];

        let checkpoint = Checkpoint::new(2, population).with_statistics(stats.clone());

        assert_eq!(checkpoint.statistics.len(), 2);
    }

    #[test]
    fn test_checkpoint_version() {
        let population: Vec<Individual<RealVector>> = vec![];
        let checkpoint = Checkpoint::new(0, population);

        assert_eq!(checkpoint.version(), CHECKPOINT_VERSION);
    }

    #[test]
    fn test_checkpoint_builder_full() {
        use crate::diagnostics::{GenerationStats, TimingStats};

        let population: Vec<Individual<RealVector>> =
            vec![Individual::new(RealVector::new(vec![1.0]))];
        let best = Individual::with_fitness(RealVector::new(vec![0.0]), 100.0);
        let hp_state = HyperparameterState::default();
        let stats = vec![GenerationStats {
            generation: 0,
            evaluations: 100,
            best_fitness: 100.0,
            worst_fitness: 10.0,
            mean_fitness: 50.0,
            median_fitness: 50.0,
            fitness_std: 10.0,
            diversity: 0.5,
            timing: TimingStats::default(),
        }];

        let checkpoint = CheckpointBuilder::new(10, population)
            .evaluations(5000)
            .best(best)
            .algorithm_state(AlgorithmState::Nsga2 {
                pareto_front_indices: vec![0, 1, 2],
            })
            .hyperparameters(hp_state)
            .statistics(stats)
            .metadata("version", "1.0")
            .rng_state(vec![0, 1, 2, 3])
            .build();

        assert_eq!(checkpoint.generation, 10);
        assert_eq!(checkpoint.evaluations, 5000);
        assert!(checkpoint.best.is_some());
        assert!(checkpoint.hyperparameter_state.is_some());
        assert_eq!(checkpoint.statistics.len(), 1);
        assert!(checkpoint.rng_state.is_some());
    }

    #[test]
    fn test_algorithm_state_variants() {
        // Test all algorithm state variants
        let simple_ga = AlgorithmState::SimpleGA;
        let steady_state = AlgorithmState::SteadyState {
            replacement_count: 5,
        };
        let nsga2 = AlgorithmState::Nsga2 {
            pareto_front_indices: vec![0, 1],
        };
        let hbga = AlgorithmState::Hbga {
            population_params: vec![1.0, 2.0],
            temperature: 1.5,
        };
        let island = AlgorithmState::Island {
            island_populations: vec![vec![0, 1], vec![2, 3]],
            migration_count: 3,
        };
        let interactive = AlgorithmState::Interactive {
            aggregator_state: "{}".to_string(),
            pending_evaluations: 10,
            evaluation_mode: "pairwise".to_string(),
        };
        let custom = AlgorithmState::Custom("custom_state".to_string());

        // Verify they're all different variants (pattern matching)
        assert!(matches!(simple_ga, AlgorithmState::SimpleGA));
        assert!(matches!(steady_state, AlgorithmState::SteadyState { .. }));
        assert!(matches!(nsga2, AlgorithmState::Nsga2 { .. }));
        assert!(matches!(hbga, AlgorithmState::Hbga { .. }));
        assert!(matches!(island, AlgorithmState::Island { .. }));
        assert!(matches!(interactive, AlgorithmState::Interactive { .. }));
        assert!(matches!(custom, AlgorithmState::Custom(_)));
    }

    #[test]
    fn test_hyperparameter_state_default() {
        let hp = HyperparameterState::default();

        assert!(hp.mutation_rate_posterior.is_none());
        assert!(hp.crossover_prob_posterior.is_none());
        assert!(hp.temperature_posterior.is_none());
        assert!(hp.step_size_posteriors.is_empty());
        assert!(hp.operator_weights.is_empty());
        assert_eq!(hp.history_size, 100);
    }
}
