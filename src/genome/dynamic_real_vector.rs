//! Dynamic (variable-length) real-valued vector genome
//!
//! This module provides a variable-length real-valued vector genome type
//! for problems where the solution dimension is not fixed.

use fugue::{addr, ChoiceValue, Trace};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{EvolutionaryGenome, RealValuedGenome};

/// Variable-length real-valued vector genome
///
/// This genome type represents optimization problems where the solution
/// dimension can vary. Useful for problems like:
/// - Neural network architecture search (variable hidden layer sizes)
/// - Variable-length feature selection
/// - Adaptive-dimensional optimization
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicRealVector {
    /// The genes (values) of this genome
    genes: Vec<f64>,
    /// Minimum allowed length
    min_length: usize,
    /// Maximum allowed length
    max_length: usize,
}

impl DynamicRealVector {
    /// Create a new dynamic real vector with the given genes and length constraints
    pub fn new(genes: Vec<f64>, min_length: usize, max_length: usize) -> Result<Self, GenomeError> {
        if genes.len() < min_length || genes.len() > max_length {
            return Err(GenomeError::InvalidStructure(format!(
                "Gene length {} outside bounds [{}, {}]",
                genes.len(),
                min_length,
                max_length
            )));
        }
        if min_length > max_length {
            return Err(GenomeError::InvalidStructure(format!(
                "min_length ({}) > max_length ({})",
                min_length, max_length
            )));
        }
        Ok(Self {
            genes,
            min_length,
            max_length,
        })
    }

    /// Create with default length constraints (1 to usize::MAX)
    pub fn with_defaults(genes: Vec<f64>) -> Self {
        Self {
            genes,
            min_length: 1,
            max_length: usize::MAX,
        }
    }

    /// Create a zero-filled vector of the given dimension
    pub fn zeros(
        dimension: usize,
        min_length: usize,
        max_length: usize,
    ) -> Result<Self, GenomeError> {
        Self::new(vec![0.0; dimension], min_length, max_length)
    }

    /// Get the minimum allowed length
    pub fn min_length(&self) -> usize {
        self.min_length
    }

    /// Get the maximum allowed length
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get the underlying vector
    pub fn into_inner(self) -> Vec<f64> {
        self.genes
    }

    /// Get a reference to the genes
    pub fn as_vec(&self) -> &Vec<f64> {
        &self.genes
    }

    /// Calculate Euclidean norm
    pub fn norm(&self) -> f64 {
        self.genes.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Calculate squared Euclidean norm
    pub fn norm_squared(&self) -> f64 {
        self.genes.iter().map(|x| x * x).sum::<f64>()
    }

    /// Add a gene to the end (if within bounds)
    pub fn push(&mut self, gene: f64) -> Result<(), GenomeError> {
        if self.genes.len() >= self.max_length {
            return Err(GenomeError::ConstraintViolation(format!(
                "Cannot add gene: would exceed max_length {}",
                self.max_length
            )));
        }
        self.genes.push(gene);
        Ok(())
    }

    /// Remove the last gene (if within bounds)
    pub fn pop(&mut self) -> Result<f64, GenomeError> {
        if self.genes.len() <= self.min_length {
            return Err(GenomeError::ConstraintViolation(format!(
                "Cannot remove gene: would go below min_length {}",
                self.min_length
            )));
        }
        Ok(self.genes.pop().unwrap())
    }

    /// Insert a gene at a specific position
    pub fn insert(&mut self, index: usize, gene: f64) -> Result<(), GenomeError> {
        if self.genes.len() >= self.max_length {
            return Err(GenomeError::ConstraintViolation(format!(
                "Cannot insert gene: would exceed max_length {}",
                self.max_length
            )));
        }
        if index > self.genes.len() {
            return Err(GenomeError::InvalidStructure(format!(
                "Insert index {} out of bounds for length {}",
                index,
                self.genes.len()
            )));
        }
        self.genes.insert(index, gene);
        Ok(())
    }

    /// Remove a gene at a specific position
    pub fn remove(&mut self, index: usize) -> Result<f64, GenomeError> {
        if self.genes.len() <= self.min_length {
            return Err(GenomeError::ConstraintViolation(format!(
                "Cannot remove gene: would go below min_length {}",
                self.min_length
            )));
        }
        if index >= self.genes.len() {
            return Err(GenomeError::InvalidStructure(format!(
                "Remove index {} out of bounds for length {}",
                index,
                self.genes.len()
            )));
        }
        Ok(self.genes.remove(index))
    }

    /// Check if a gene can be added
    pub fn can_grow(&self) -> bool {
        self.genes.len() < self.max_length
    }

    /// Check if a gene can be removed
    pub fn can_shrink(&self) -> bool {
        self.genes.len() > self.min_length
    }

    /// Element-wise addition (requires same length)
    pub fn add(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.genes.len() != other.genes.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.genes.len(),
                actual: other.genes.len(),
            });
        }
        Self::new(
            self.genes
                .iter()
                .zip(other.genes.iter())
                .map(|(a, b)| a + b)
                .collect(),
            self.min_length,
            self.max_length,
        )
    }

    /// Element-wise subtraction (requires same length)
    pub fn sub(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.genes.len() != other.genes.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.genes.len(),
                actual: other.genes.len(),
            });
        }
        Self::new(
            self.genes
                .iter()
                .zip(other.genes.iter())
                .map(|(a, b)| a - b)
                .collect(),
            self.min_length,
            self.max_length,
        )
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            genes: self.genes.iter().map(|x| x * scalar).collect(),
            min_length: self.min_length,
            max_length: self.max_length,
        }
    }
}

impl EvolutionaryGenome for DynamicRealVector {
    type Allele = f64;
    type Phenotype = Vec<f64>;

    /// Convert DynamicRealVector to Fugue trace.
    ///
    /// Stores genes at "gene#i" and metadata at "meta#min_length" and "meta#max_length".
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();
        for (i, &gene) in self.genes.iter().enumerate() {
            trace.insert_choice(addr!("gene", i), ChoiceValue::F64(gene), 0.0);
        }
        // Store length constraints in trace
        trace.insert_choice(
            addr!("meta", "min_length"),
            ChoiceValue::I64(self.min_length as i64),
            0.0,
        );
        trace.insert_choice(
            addr!("meta", "max_length"),
            ChoiceValue::I64(self.max_length as i64),
            0.0,
        );
        trace.insert_choice(
            addr!("meta", "length"),
            ChoiceValue::I64(self.genes.len() as i64),
            0.0,
        );
        trace
    }

    /// Reconstruct DynamicRealVector from Fugue trace.
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let min_length = trace
            .get_i64(&addr!("meta", "min_length"))
            .map(|v| v as usize)
            .unwrap_or(1);
        let max_length = trace
            .get_i64(&addr!("meta", "max_length"))
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        let expected_length = trace.get_i64(&addr!("meta", "length")).map(|v| v as usize);

        let mut genes = Vec::new();
        let mut i = 0;
        while let Some(val) = trace.get_f64(&addr!("gene", i)) {
            genes.push(val);
            i += 1;
            // Stop if we've reached expected length (handles sparse traces)
            if let Some(len) = expected_length {
                if i >= len {
                    break;
                }
            }
        }

        if genes.is_empty() {
            return Err(GenomeError::InvalidStructure(
                "No genes found in trace".to_string(),
            ));
        }

        Self::new(genes, min_length, max_length)
    }

    fn decode(&self) -> Self::Phenotype {
        self.genes.clone()
    }

    fn dimension(&self) -> usize {
        self.genes.len()
    }

    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        // Generate with random length within the bounds
        let min_len = 1;
        let max_len = bounds.dimension();
        let length = if min_len == max_len {
            min_len
        } else {
            rng.gen_range(min_len..=max_len)
        };

        let genes: Vec<f64> = (0..length)
            .map(|i| {
                let b = if i < bounds.dimension() {
                    bounds.get(i).unwrap()
                } else {
                    // Use first bound if we exceed bounds dimension
                    bounds.get(0).unwrap()
                };
                rng.gen_range(b.min..=b.max)
            })
            .collect();

        Self {
            genes,
            min_length: min_len,
            max_length: max_len,
        }
    }

    fn distance(&self, other: &Self) -> f64 {
        // Euclidean distance for common genes, plus penalty for different lengths
        let common_len = self.genes.len().min(other.genes.len());
        let mut dist_sq = 0.0;

        for i in 0..common_len {
            let diff = self.genes[i] - other.genes[i];
            dist_sq += diff * diff;
        }

        // Add penalty for length difference
        let length_penalty = (self.genes.len() as f64 - other.genes.len() as f64).abs();
        dist_sq.sqrt() + length_penalty
    }

    fn trace_prefix() -> &'static str {
        "dyn_gene"
    }
}

impl RealValuedGenome for DynamicRealVector {
    fn genes(&self) -> &[f64] {
        &self.genes
    }

    fn genes_mut(&mut self) -> &mut [f64] {
        &mut self.genes
    }

    fn from_genes(genes: Vec<f64>) -> Result<Self, GenomeError> {
        if genes.is_empty() {
            return Err(GenomeError::InvalidStructure(
                "Cannot create DynamicRealVector with empty genes".to_string(),
            ));
        }
        Ok(Self::with_defaults(genes))
    }

    fn apply_bounds(&mut self, bounds: &MultiBounds) {
        for (i, gene) in self.genes.iter_mut().enumerate() {
            if let Some(b) = bounds.get(i) {
                *gene = gene.clamp(b.min, b.max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_real_vector_creation() {
        let genes = vec![1.0, 2.0, 3.0];
        let genome = DynamicRealVector::new(genes.clone(), 1, 10).unwrap();
        assert_eq!(genome.genes(), &genes[..]);
        assert_eq!(genome.dimension(), 3);
    }

    #[test]
    fn test_dynamic_real_vector_length_constraints() {
        // Too short
        let result = DynamicRealVector::new(vec![1.0], 2, 10);
        assert!(result.is_err());

        // Too long
        let result = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 1, 2);
        assert!(result.is_err());

        // Invalid bounds
        let result = DynamicRealVector::new(vec![1.0, 2.0], 5, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_push_pop() {
        let mut genome = DynamicRealVector::new(vec![1.0, 2.0], 1, 5).unwrap();

        // Push
        genome.push(3.0).unwrap();
        assert_eq!(genome.dimension(), 3);
        assert_eq!(genome.genes()[2], 3.0);

        // Pop
        let val = genome.pop().unwrap();
        assert_eq!(val, 3.0);
        assert_eq!(genome.dimension(), 2);
    }

    #[test]
    fn test_push_pop_bounds() {
        let mut genome = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 3, 3).unwrap();

        // Cannot push (at max)
        assert!(genome.push(4.0).is_err());

        // Cannot pop (at min)
        assert!(genome.pop().is_err());
    }

    #[test]
    fn test_insert_remove() {
        let mut genome = DynamicRealVector::new(vec![1.0, 3.0], 1, 5).unwrap();

        // Insert in middle
        genome.insert(1, 2.0).unwrap();
        assert_eq!(genome.genes(), &[1.0, 2.0, 3.0]);

        // Remove from middle
        let val = genome.remove(1).unwrap();
        assert_eq!(val, 2.0);
        assert_eq!(genome.genes(), &[1.0, 3.0]);
    }

    #[test]
    fn test_can_grow_shrink() {
        let genome = DynamicRealVector::new(vec![1.0, 2.0], 1, 3).unwrap();
        assert!(genome.can_grow());
        assert!(genome.can_shrink());

        let genome_at_max = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        assert!(!genome_at_max.can_grow());
        assert!(genome_at_max.can_shrink());

        let genome_at_min = DynamicRealVector::new(vec![1.0], 1, 3).unwrap();
        assert!(genome_at_min.can_grow());
        assert!(!genome_at_min.can_shrink());
    }

    #[test]
    fn test_trace_roundtrip() {
        let genome = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 2, 5).unwrap();
        let trace = genome.to_trace();
        let restored = DynamicRealVector::from_trace(&trace).unwrap();

        assert_eq!(genome.genes(), restored.genes());
        assert_eq!(genome.min_length(), restored.min_length());
        assert_eq!(genome.max_length(), restored.max_length());
    }

    #[test]
    fn test_distance() {
        let g1 = DynamicRealVector::new(vec![0.0, 0.0], 1, 10).unwrap();
        let g2 = DynamicRealVector::new(vec![3.0, 4.0], 1, 10).unwrap();

        // Euclidean distance should be 5
        let dist = g1.distance(&g2);
        assert!((dist - 5.0).abs() < 0.001);

        // Different lengths add penalty
        let g3 = DynamicRealVector::new(vec![0.0, 0.0, 0.0], 1, 10).unwrap();
        let dist_with_penalty = g1.distance(&g3);
        assert!(dist_with_penalty > 0.0);
    }

    #[test]
    fn test_generate_random() {
        let bounds = MultiBounds::symmetric(5.0, 5);
        let mut rng = rand::thread_rng();

        let genome = DynamicRealVector::generate(&mut rng, &bounds);

        assert!(genome.dimension() >= 1);
        assert!(genome.dimension() <= 5);
        for gene in genome.genes() {
            assert!(*gene >= -5.0 && *gene <= 5.0);
        }
    }

    #[test]
    fn test_arithmetic_operations() {
        let g1 = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 1, 10).unwrap();
        let g2 = DynamicRealVector::new(vec![4.0, 5.0, 6.0], 1, 10).unwrap();

        let sum = g1.add(&g2).unwrap();
        assert_eq!(sum.genes(), &[5.0, 7.0, 9.0]);

        let diff = g2.sub(&g1).unwrap();
        assert_eq!(diff.genes(), &[3.0, 3.0, 3.0]);

        let scaled = g1.scale(2.0);
        assert_eq!(scaled.genes(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_norm() {
        let genome = DynamicRealVector::new(vec![3.0, 4.0], 1, 10).unwrap();
        assert!((genome.norm() - 5.0).abs() < 0.001);
        assert!((genome.norm_squared() - 25.0).abs() < 0.001);
    }
}
