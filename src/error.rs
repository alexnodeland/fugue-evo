//! Error types for fugue-evo
//!
//! This module defines all error types used throughout the library.

use thiserror::Error;

/// Address type for genome trace operations
pub type Address = String;

/// Error type for genome operations
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GenomeError {
    /// A required address was missing from the trace
    #[error("Missing address in trace: {0}")]
    MissingAddress(Address),

    /// Type mismatch when converting from trace
    #[error("Type mismatch at address {address}: expected {expected}, got {actual}")]
    TypeMismatch {
        address: Address,
        expected: String,
        actual: String,
    },

    /// Invalid genome structure
    #[error("Invalid genome structure: {0}")]
    InvalidStructure(String),

    /// Constraint violation in genome
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Error type for operator failures
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OperatorError {
    /// Crossover operation failed
    #[error("Crossover failed: {0}")]
    CrossoverFailed(String),

    /// Mutation operation failed
    #[error("Mutation failed: {0}")]
    MutationFailed(String),

    /// Selection operation failed
    #[error("Selection failed: {0}")]
    SelectionFailed(String),

    /// Invalid operator configuration
    #[error("Invalid operator configuration: {0}")]
    InvalidConfiguration(String),
}

/// Error type for checkpoint operations
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// IO error during checkpoint (native only)
    #[cfg(not(target_arch = "wasm32"))]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Storage error (WASM-compatible alternative to IO)
    #[cfg(target_arch = "wasm32")]
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Checkpoint version mismatch
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    /// Checkpoint version is too new
    #[error("Checkpoint version {0} is newer than supported")]
    VersionTooNew(u32),

    /// Checkpoint version is too old
    #[error("Checkpoint version {0} is too old to load")]
    VersionTooOld(u32),

    /// Checkpoint file not found
    #[error("Checkpoint not found: {0}")]
    NotFound(String),

    /// Corrupted checkpoint data
    #[error("Corrupted checkpoint: {0}")]
    Corrupted(String),
}

/// Top-level error type for evolution operations
#[derive(Debug, Error)]
pub enum EvolutionError {
    /// Genome error
    #[error("Genome error: {0}")]
    Genome(#[from] GenomeError),

    /// Operator error
    #[error("Operator error: {0}")]
    Operator(#[from] OperatorError),

    /// Fitness evaluation failed
    #[error("Fitness evaluation failed: {0}")]
    FitnessEvaluation(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Configuration(String),

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(#[from] CheckpointError),

    /// Numerical instability
    #[error("Numerical instability: {0}")]
    Numerical(String),

    /// Empty population
    #[error("Empty population")]
    EmptyPopulation,

    /// Interactive evaluation error
    #[error("Interactive evaluation error: {0}")]
    InteractiveEvaluation(String),

    /// Insufficient evaluation coverage
    #[error("Insufficient coverage: {coverage:.1}% (need {required:.1}%)")]
    InsufficientCoverage {
        /// Actual coverage achieved
        coverage: f64,
        /// Required coverage threshold
        required: f64,
    },
}

/// Result type alias for evolution operations
pub type EvoResult<T> = Result<T, EvolutionError>;

/// Repair information when an operator needs to fix a constraint violation
#[derive(Debug, Clone)]
pub struct RepairInfo {
    /// List of constraint violations that were repaired
    pub constraint_violations: Vec<String>,
    /// Method used to repair the genome
    pub repair_method: &'static str,
}

/// Result of an operator application with optional repair information
#[derive(Debug, Clone)]
pub enum OperatorResult<G> {
    /// Operation succeeded without repairs
    Success(G),
    /// Operation succeeded but required repairs
    Repaired(G, RepairInfo),
    /// Operation failed unrecoverably
    Failed(OperatorError),
}

impl<G> OperatorResult<G> {
    /// Returns the genome if successful or repaired, None if failed
    pub fn genome(self) -> Option<G> {
        match self {
            Self::Success(g) | Self::Repaired(g, _) => Some(g),
            Self::Failed(_) => None,
        }
    }

    /// Returns true if the operation was successful (with or without repairs)
    pub fn is_ok(&self) -> bool {
        !matches!(self, Self::Failed(_))
    }

    /// Returns true if repairs were needed
    pub fn was_repaired(&self) -> bool {
        matches!(self, Self::Repaired(_, _))
    }

    /// Maps the genome type
    pub fn map<U, F: FnOnce(G) -> U>(self, f: F) -> OperatorResult<U> {
        match self {
            Self::Success(g) => OperatorResult::Success(f(g)),
            Self::Repaired(g, info) => OperatorResult::Repaired(f(g), info),
            Self::Failed(e) => OperatorResult::Failed(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_error_display() {
        let err = GenomeError::MissingAddress("gene_0".to_string());
        assert_eq!(err.to_string(), "Missing address in trace: gene_0");

        let err = GenomeError::TypeMismatch {
            address: "gene_1".to_string(),
            expected: "f64".to_string(),
            actual: "bool".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Type mismatch at address gene_1: expected f64, got bool"
        );

        let err = GenomeError::DimensionMismatch {
            expected: 10,
            actual: 5,
        };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 10, got 5");
    }

    #[test]
    fn test_operator_error_display() {
        let err = OperatorError::CrossoverFailed("incompatible parents".to_string());
        assert_eq!(err.to_string(), "Crossover failed: incompatible parents");

        let err = OperatorError::InvalidConfiguration("eta must be positive".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid operator configuration: eta must be positive"
        );
    }

    #[test]
    fn test_evolution_error_from_genome_error() {
        let genome_err = GenomeError::InvalidStructure("bad shape".to_string());
        let evo_err: EvolutionError = genome_err.into();
        assert!(matches!(evo_err, EvolutionError::Genome(_)));
    }

    #[test]
    fn test_operator_result_success() {
        let result: OperatorResult<i32> = OperatorResult::Success(42);
        assert!(result.is_ok());
        assert!(!result.was_repaired());
        assert_eq!(result.genome(), Some(42));
    }

    #[test]
    fn test_operator_result_repaired() {
        let repair_info = RepairInfo {
            constraint_violations: vec!["out of bounds".to_string()],
            repair_method: "clamp",
        };
        let result: OperatorResult<i32> = OperatorResult::Repaired(42, repair_info);
        assert!(result.is_ok());
        assert!(result.was_repaired());
        assert_eq!(result.genome(), Some(42));
    }

    #[test]
    fn test_operator_result_failed() {
        let result: OperatorResult<i32> =
            OperatorResult::Failed(OperatorError::MutationFailed("test".to_string()));
        assert!(!result.is_ok());
        assert!(!result.was_repaired());
        assert_eq!(result.genome(), None);
    }

    #[test]
    fn test_operator_result_map() {
        let result: OperatorResult<i32> = OperatorResult::Success(42);
        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.genome(), Some(84));
    }
}
