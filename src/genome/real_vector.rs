//! Real-valued vector genome
//!
//! This module provides a fixed-length real-valued vector genome type
//! with Fugue trace integration for probabilistic operations.

use fugue::{addr, ChoiceValue, Trace};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{EvolutionaryGenome, RealValuedGenome};

/// Fixed-length real-valued vector genome
///
/// This genome type represents continuous optimization problems where
/// solutions are vectors of real numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealVector {
    /// The genes (values) of this genome
    genes: Vec<f64>,
}

impl RealVector {
    /// Create a new real vector with the given genes
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes }
    }

    /// Create a zero-filled vector of the given dimension
    pub fn zeros(dimension: usize) -> Self {
        Self {
            genes: vec![0.0; dimension],
        }
    }

    /// Create a vector filled with a constant value
    pub fn filled(dimension: usize, value: f64) -> Self {
        Self {
            genes: vec![value; dimension],
        }
    }

    /// Create from an iterator
    pub fn collect_from<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
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

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.genes.len() != other.genes.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.genes.len(),
                actual: other.genes.len(),
            });
        }
        Ok(Self {
            genes: self
                .genes
                .iter()
                .zip(other.genes.iter())
                .map(|(a, b)| a + b)
                .collect(),
        })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.genes.len() != other.genes.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.genes.len(),
                actual: other.genes.len(),
            });
        }
        Ok(Self {
            genes: self
                .genes
                .iter()
                .zip(other.genes.iter())
                .map(|(a, b)| a - b)
                .collect(),
        })
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> Self {
        Self {
            genes: self.genes.iter().map(|x| x * scalar).collect(),
        }
    }
}

impl EvolutionaryGenome for RealVector {
    type Allele = f64;
    type Phenotype = Vec<f64>;

    /// Convert RealVector to Fugue trace.
    ///
    /// Each gene is stored at address "gene#i" where i is the index.
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();
        for (i, &gene) in self.genes.iter().enumerate() {
            trace.insert_choice(addr!("gene", i), ChoiceValue::F64(gene), 0.0);
        }
        trace
    }

    /// Reconstruct RealVector from Fugue trace.
    ///
    /// Reads genes from addresses "gene#0", "gene#1", ... until no more are found.
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let mut genes = Vec::new();
        let mut i = 0;
        while let Some(val) = trace.get_f64(&addr!("gene", i)) {
            genes.push(val);
            i += 1;
        }
        if genes.is_empty() {
            return Err(GenomeError::InvalidStructure(
                "No genes found in trace".to_string(),
            ));
        }
        Ok(Self { genes })
    }

    fn decode(&self) -> Self::Phenotype {
        self.genes.clone()
    }

    fn dimension(&self) -> usize {
        self.genes.len()
    }

    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        let genes = bounds
            .bounds
            .iter()
            .map(|b| rng.gen_range(b.min..=b.max))
            .collect();
        Self { genes }
    }

    fn as_slice(&self) -> Option<&[f64]> {
        Some(&self.genes)
    }

    fn as_mut_slice(&mut self) -> Option<&mut [f64]> {
        Some(&mut self.genes)
    }

    fn distance(&self, other: &Self) -> f64 {
        self.genes
            .iter()
            .zip(other.genes.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl RealValuedGenome for RealVector {
    fn genes(&self) -> &[f64] {
        &self.genes
    }

    fn genes_mut(&mut self) -> &mut [f64] {
        &mut self.genes
    }

    fn from_genes(genes: Vec<f64>) -> Result<Self, GenomeError> {
        Ok(Self { genes })
    }
}

impl std::ops::Index<usize> for RealVector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl std::ops::IndexMut<usize> for RealVector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.genes[index]
    }
}

impl From<Vec<f64>> for RealVector {
    fn from(genes: Vec<f64>) -> Self {
        Self { genes }
    }
}

impl From<RealVector> for Vec<f64> {
    fn from(genome: RealVector) -> Self {
        genome.genes
    }
}

impl<const N: usize> From<[f64; N]> for RealVector {
    fn from(arr: [f64; N]) -> Self {
        Self {
            genes: arr.to_vec(),
        }
    }
}

impl IntoIterator for RealVector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl<'a> IntoIterator for &'a RealVector {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fugue::addr;

    #[test]
    fn test_real_vector_new() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.genes(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_vector_zeros() {
        let v = RealVector::zeros(5);
        assert_eq!(v.dimension(), 5);
        assert!(v.genes().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_real_vector_filled() {
        let v = RealVector::filled(3, 42.0);
        assert_eq!(v.genes(), &[42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_real_vector_from_array() {
        let v: RealVector = [1.0, 2.0, 3.0].into();
        assert_eq!(v.genes(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_vector_decode() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        let phenotype = v.decode();
        assert_eq!(phenotype, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_vector_generate() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 10);
        let v = RealVector::generate(&mut rng, &bounds);

        assert_eq!(v.dimension(), 10);
        for gene in v.genes() {
            assert!(*gene >= -5.0 && *gene <= 5.0);
        }
    }

    #[test]
    fn test_real_vector_norm() {
        let v = RealVector::new(vec![3.0, 4.0]);
        assert_relative_eq!(v.norm(), 5.0);
        assert_relative_eq!(v.norm_squared(), 25.0);
    }

    #[test]
    fn test_real_vector_distance() {
        let v1 = RealVector::new(vec![0.0, 0.0]);
        let v2 = RealVector::new(vec![3.0, 4.0]);
        assert_relative_eq!(v1.distance(&v2), 5.0);
    }

    #[test]
    fn test_real_vector_add() {
        let v1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let v2 = RealVector::new(vec![4.0, 5.0, 6.0]);
        let sum = v1.add(&v2).unwrap();
        assert_eq!(sum.genes(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_real_vector_add_dimension_mismatch() {
        let v1 = RealVector::new(vec![1.0, 2.0]);
        let v2 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let result = v1.add(&v2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GenomeError::DimensionMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn test_real_vector_sub() {
        let v1 = RealVector::new(vec![5.0, 7.0, 9.0]);
        let v2 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let diff = v1.sub(&v2).unwrap();
        assert_eq!(diff.genes(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_real_vector_scale() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        let scaled = v.scale(2.0);
        assert_eq!(scaled.genes(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_real_vector_indexing() {
        let mut v = RealVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);

        v[1] = 42.0;
        assert_eq!(v[1], 42.0);
    }

    #[test]
    fn test_real_vector_apply_bounds() {
        let mut v = RealVector::new(vec![-10.0, 0.0, 10.0]);
        let bounds = MultiBounds::symmetric(5.0, 3);
        v.apply_bounds(&bounds);
        assert_eq!(v.genes(), &[-5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_real_vector_iteration() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        let sum: f64 = v.into_iter().sum();
        assert_relative_eq!(sum, 6.0);
    }

    #[test]
    fn test_real_vector_into_inner() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        let inner: Vec<f64> = v.into_inner();
        assert_eq!(inner, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_vector_serialization() {
        let v = RealVector::new(vec![1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&v).unwrap();
        let deserialized: RealVector = serde_json::from_str(&serialized).unwrap();
        assert_eq!(v, deserialized);
    }

    #[test]
    fn test_real_vector_to_trace() {
        let v = RealVector::new(vec![1.5, 2.5, 3.5]);
        let trace = v.to_trace();

        assert_eq!(trace.get_f64(&addr!("gene", 0)), Some(1.5));
        assert_eq!(trace.get_f64(&addr!("gene", 1)), Some(2.5));
        assert_eq!(trace.get_f64(&addr!("gene", 2)), Some(3.5));
        assert_eq!(trace.get_f64(&addr!("gene", 3)), None);
    }

    #[test]
    fn test_real_vector_from_trace() {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("gene", 0), ChoiceValue::F64(1.0), 0.0);
        trace.insert_choice(addr!("gene", 1), ChoiceValue::F64(2.0), 0.0);
        trace.insert_choice(addr!("gene", 2), ChoiceValue::F64(3.0), 0.0);

        let v = RealVector::from_trace(&trace).unwrap();
        assert_eq!(v.genes(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_vector_trace_roundtrip() {
        let original = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let trace = original.to_trace();
        let recovered = RealVector::from_trace(&trace).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_real_vector_from_trace_empty() {
        let trace = Trace::default();
        let result = RealVector::from_trace(&trace);
        assert!(result.is_err());
    }
}
