//! Core genome traits
//!
//! This module defines the `EvolutionaryGenome` trait and related types,
//! with integration to Fugue PPL for probabilistic operations.

use fugue::{addr, Address, Trace};

// Re-export ChoiceValue for use in genome implementations
pub use fugue::ChoiceValue;
use rand::Rng;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;

/// Core genome abstraction with Fugue integration for evolutionary algorithms.
///
/// This trait defines the interface for evolvable solution representations,
/// with explicit Fugue trace integration for probabilistic operations.
/// Genomes must be cloneable, serializable, and thread-safe.
///
/// # Fugue Integration
///
/// The key insight is that Fugue's traces can represent genomes. Each gene
/// is stored at an indexed address, enabling:
/// - Trace-based mutation (selective resampling)
/// - Trace-based crossover (merging parent traces)
/// - Probabilistic interpretation of genetic operators
pub trait EvolutionaryGenome: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {
    /// The allele type for individual genes
    type Allele: Clone + Send;

    /// The phenotype or decoded solution type
    type Phenotype;

    /// Convert genome to Fugue trace for probabilistic operations.
    ///
    /// Each gene is stored at an indexed address (e.g., "gene#0", "gene#1", ...),
    /// enabling trace-based evolutionary operators.
    fn to_trace(&self) -> Trace;

    /// Reconstruct genome from Fugue trace after evolutionary operations.
    ///
    /// This is the inverse of `to_trace()`, extracting gene values from
    /// the trace's choice map.
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;

    /// Decode genome into phenotype for fitness evaluation
    fn decode(&self) -> Self::Phenotype;

    /// Compute dimensionality for adaptive operators
    fn dimension(&self) -> usize;

    /// Generate a random genome within the given bounds.
    ///
    /// # Interpretation of `bounds`
    ///
    /// `MultiBounds` semantically describes a set of per-dimension numeric
    /// `[min, max]` intervals, but only the real-valued genome types
    /// ([`RealVector`](crate::genome::real_vector::RealVector),
    /// [`DynamicRealVector`](crate::genome::dynamic_real_vector::DynamicRealVector))
    /// actually consult the `min`/`max` fields. For every other built-in genome
    /// type, **only `bounds.dimension()` (the *count* of intervals) is
    /// consulted** — it is repurposed as a stand-in for the genome's structural
    /// size, and the `min`/`max` values are ignored:
    ///
    /// - [`BitString`](crate::genome::bit_string::BitString): `dimension()` is
    ///   the number of bits. Prefer
    ///   [`BitString::generate_with_len`](crate::genome::bit_string::BitString::generate_with_len).
    /// - [`Permutation`](crate::genome::permutation::Permutation): `dimension()`
    ///   is the permutation length. Prefer
    ///   [`Permutation::generate_with_len`](crate::genome::permutation::Permutation::generate_with_len).
    /// - [`TreeGenome`](crate::genome::tree::TreeGenome): `dimension()` is
    ///   remapped to a maximum tree depth. Prefer
    ///   [`TreeGenome::generate_with_depth`](crate::genome::tree::TreeGenome::generate_with_depth).
    ///
    /// When you are not generating real-valued genomes, use the per-type honest
    /// constructors listed above; they make the size/depth parameter explicit
    /// instead of overloading `MultiBounds`.
    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self;

    /// Get the genome's genes as a slice (for numeric genomes)
    fn as_slice(&self) -> Option<&[Self::Allele]> {
        None
    }

    /// Get the genome's genes as a mutable slice (for numeric genomes)
    fn as_mut_slice(&mut self) -> Option<&mut [Self::Allele]> {
        None
    }

    /// Distance metric between two genomes.
    ///
    /// This is a **required** method: there is deliberately no default
    /// implementation, because a silent fallback (e.g. always `0.0`) would make
    /// every pair of genomes look identical and silently break diversity-driven
    /// mechanisms (niching, crowding, speciation).
    ///
    /// # Panics
    ///
    /// Implementations panic when the two genomes are structurally incompatible
    /// (for fixed-structure genomes, this means different lengths / dimensions).
    /// A structural mismatch is an invariant violation by the caller rather than
    /// a recoverable condition. Use [`try_distance`](Self::try_distance) when a
    /// fallible comparison is required.
    ///
    /// (Genome types whose comparison is meaningfully defined across differing
    /// structures — e.g. [`DynamicRealVector`](crate::genome::dynamic_real_vector::DynamicRealVector),
    /// which adds a length penalty, and [`TreeGenome`](crate::genome::tree::TreeGenome),
    /// which compares size/depth — never panic.)
    fn distance(&self, other: &Self) -> f64;

    /// Fallible distance metric.
    ///
    /// Returns `Err(GenomeError::DimensionMismatch { .. })` (or another
    /// [`GenomeError`]) when the two genomes are structurally incompatible,
    /// instead of panicking as [`distance`](Self::distance) does. For genome
    /// types whose distance is defined across differing structures this always
    /// returns `Ok`.
    fn try_distance(&self, other: &Self) -> Result<f64, GenomeError>;

    /// Get the address prefix used for trace storage (default: "gene")
    fn trace_prefix() -> &'static str {
        "gene"
    }
}

/// Trait for genomes that can be represented as real vectors
pub trait RealValuedGenome: EvolutionaryGenome<Allele = f64> {
    /// Get the genes as a slice of f64 values
    fn genes(&self) -> &[f64];

    /// Get the genes as a mutable slice of f64 values
    fn genes_mut(&mut self) -> &mut [f64];

    /// Create from a vector of genes
    fn from_genes(genes: Vec<f64>) -> Result<Self, GenomeError>;

    /// Apply bounds to all genes
    fn apply_bounds(&mut self, bounds: &MultiBounds) {
        bounds.clamp_vec(self.genes_mut());
    }
}

/// Trait for genomes that can be represented as bit strings
pub trait BinaryGenome: EvolutionaryGenome<Allele = bool> {
    /// Get the bits as a slice
    fn bits(&self) -> &[bool];

    /// Get the bits as a mutable slice
    fn bits_mut(&mut self) -> &mut [bool];

    /// Create from a vector of bits
    fn from_bits(bits: Vec<bool>) -> Result<Self, GenomeError>;

    /// Count the number of true bits (ones)
    fn count_ones(&self) -> usize {
        self.bits().iter().filter(|&&b| b).count()
    }

    /// Count the number of false bits (zeros)
    fn count_zeros(&self) -> usize {
        self.bits().iter().filter(|&&b| !b).count()
    }
}

/// Trait for genomes that represent permutations
pub trait PermutationGenome: EvolutionaryGenome<Allele = usize> {
    /// Get the permutation as a slice
    fn permutation(&self) -> &[usize];

    /// Get the permutation as a mutable slice
    fn permutation_mut(&mut self) -> &mut [usize];

    /// Create from a vector of indices
    fn from_permutation(perm: Vec<usize>) -> Result<Self, GenomeError>;

    /// Check if the genome represents a valid permutation
    fn is_valid_permutation(&self) -> bool {
        let perm = self.permutation();
        let n = perm.len();
        if n == 0 {
            return true;
        }

        let mut seen = vec![false; n];
        for &idx in perm {
            if idx >= n || seen[idx] {
                return false;
            }
            seen[idx] = true;
        }
        true
    }
}

/// Helper function to create a gene address for trace storage
pub fn gene_address(prefix: &str, index: usize) -> Address {
    addr!(prefix, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // Mock genome for testing the trait
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct MockGenome {
        genes: Vec<f64>,
    }

    impl EvolutionaryGenome for MockGenome {
        type Allele = f64;
        type Phenotype = Vec<f64>;

        fn to_trace(&self) -> Trace {
            let mut trace = Trace::default();
            for (i, &gene) in self.genes.iter().enumerate() {
                trace.insert_choice(addr!("gene", i), ChoiceValue::F64(gene), 0.0);
            }
            trace
        }

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
            self.try_distance(other)
                .expect("distance: genomes have mismatched dimension")
        }

        fn try_distance(&self, other: &Self) -> Result<f64, GenomeError> {
            if self.genes.len() != other.genes.len() {
                return Err(GenomeError::DimensionMismatch {
                    expected: self.genes.len(),
                    actual: other.genes.len(),
                });
            }
            Ok(self
                .genes
                .iter()
                .zip(other.genes.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt())
        }
    }

    impl RealValuedGenome for MockGenome {
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

    #[test]
    fn test_mock_genome_decode() {
        let genome = MockGenome {
            genes: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(genome.decode(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mock_genome_dimension() {
        let genome = MockGenome {
            genes: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(genome.dimension(), 3);
    }

    #[test]
    fn test_mock_genome_generate() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 3);
        let genome = MockGenome::generate(&mut rng, &bounds);
        assert_eq!(genome.dimension(), 3);
        for gene in genome.genes() {
            assert!(*gene >= -5.0 && *gene <= 5.0);
        }
    }

    #[test]
    fn test_mock_genome_distance() {
        let g1 = MockGenome {
            genes: vec![0.0, 0.0, 0.0],
        };
        let g2 = MockGenome {
            genes: vec![3.0, 4.0, 0.0],
        };
        assert_eq!(g1.distance(&g2), 5.0);
    }

    #[test]
    fn test_real_valued_genome_apply_bounds() {
        let mut genome = MockGenome {
            genes: vec![-10.0, 0.0, 10.0],
        };
        let bounds = MultiBounds::symmetric(5.0, 3);
        genome.apply_bounds(&bounds);
        assert_eq!(genome.genes, vec![-5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_mock_genome_to_trace() {
        let genome = MockGenome {
            genes: vec![1.0, 2.0, 3.0],
        };
        let trace = genome.to_trace();

        assert_eq!(trace.get_f64(&addr!("gene", 0)), Some(1.0));
        assert_eq!(trace.get_f64(&addr!("gene", 1)), Some(2.0));
        assert_eq!(trace.get_f64(&addr!("gene", 2)), Some(3.0));
    }

    #[test]
    fn test_mock_genome_from_trace() {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("gene", 0), ChoiceValue::F64(1.5), 0.0);
        trace.insert_choice(addr!("gene", 1), ChoiceValue::F64(2.5), 0.0);
        trace.insert_choice(addr!("gene", 2), ChoiceValue::F64(3.5), 0.0);

        let genome = MockGenome::from_trace(&trace).unwrap();
        assert_eq!(genome.genes, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_mock_genome_trace_roundtrip() {
        let original = MockGenome {
            genes: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let trace = original.to_trace();
        let recovered = MockGenome::from_trace(&trace).unwrap();
        assert_eq!(original, recovered);
    }

    // Mock binary genome for testing
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct MockBinaryGenome {
        bits: Vec<bool>,
    }

    impl EvolutionaryGenome for MockBinaryGenome {
        type Allele = bool;
        type Phenotype = Vec<bool>;

        fn to_trace(&self) -> Trace {
            let mut trace = Trace::default();
            for (i, &bit) in self.bits.iter().enumerate() {
                trace.insert_choice(addr!("bit", i), ChoiceValue::Bool(bit), 0.0);
            }
            trace
        }

        fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
            let mut bits = Vec::new();
            let mut i = 0;
            while let Some(val) = trace.get_bool(&addr!("bit", i)) {
                bits.push(val);
                i += 1;
            }
            if bits.is_empty() {
                return Err(GenomeError::InvalidStructure(
                    "No bits found in trace".to_string(),
                ));
            }
            Ok(Self { bits })
        }

        fn decode(&self) -> Self::Phenotype {
            self.bits.clone()
        }

        fn dimension(&self) -> usize {
            self.bits.len()
        }

        fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
            let bits = (0..bounds.dimension()).map(|_| rng.gen()).collect();
            Self { bits }
        }

        fn distance(&self, other: &Self) -> f64 {
            self.try_distance(other)
                .expect("distance: genomes have mismatched dimension")
        }

        fn try_distance(&self, other: &Self) -> Result<f64, GenomeError> {
            if self.bits.len() != other.bits.len() {
                return Err(GenomeError::DimensionMismatch {
                    expected: self.bits.len(),
                    actual: other.bits.len(),
                });
            }
            Ok(self
                .bits
                .iter()
                .zip(other.bits.iter())
                .filter(|(a, b)| a != b)
                .count() as f64)
        }

        fn trace_prefix() -> &'static str {
            "bit"
        }
    }

    impl BinaryGenome for MockBinaryGenome {
        fn bits(&self) -> &[bool] {
            &self.bits
        }

        fn bits_mut(&mut self) -> &mut [bool] {
            &mut self.bits
        }

        fn from_bits(bits: Vec<bool>) -> Result<Self, GenomeError> {
            Ok(Self { bits })
        }
    }

    #[test]
    fn test_binary_genome_count() {
        let genome = MockBinaryGenome {
            bits: vec![true, false, true, true, false],
        };
        assert_eq!(genome.count_ones(), 3);
        assert_eq!(genome.count_zeros(), 2);
    }

    #[test]
    fn test_binary_genome_trace_roundtrip() {
        let original = MockBinaryGenome {
            bits: vec![true, false, true, false, true],
        };
        let trace = original.to_trace();
        let recovered = MockBinaryGenome::from_trace(&trace).unwrap();
        assert_eq!(original, recovered);
    }

    // Mock permutation genome for testing
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct MockPermGenome {
        perm: Vec<usize>,
    }

    impl EvolutionaryGenome for MockPermGenome {
        type Allele = usize;
        type Phenotype = Vec<usize>;

        fn to_trace(&self) -> Trace {
            let mut trace = Trace::default();
            for (i, &val) in self.perm.iter().enumerate() {
                trace.insert_choice(addr!("perm", i), ChoiceValue::Usize(val), 0.0);
            }
            trace
        }

        fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
            let mut perm = Vec::new();
            let mut i = 0;
            while let Some(val) = trace.get_usize(&addr!("perm", i)) {
                perm.push(val);
                i += 1;
            }
            if perm.is_empty() {
                return Err(GenomeError::InvalidStructure(
                    "No permutation found in trace".to_string(),
                ));
            }
            Ok(Self { perm })
        }

        fn decode(&self) -> Self::Phenotype {
            self.perm.clone()
        }

        fn dimension(&self) -> usize {
            self.perm.len()
        }

        fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
            use rand::seq::SliceRandom;
            let n = bounds.dimension();
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(rng);
            Self { perm }
        }

        fn distance(&self, other: &Self) -> f64 {
            self.try_distance(other)
                .expect("distance: genomes have mismatched dimension")
        }

        fn try_distance(&self, other: &Self) -> Result<f64, GenomeError> {
            if self.perm.len() != other.perm.len() {
                return Err(GenomeError::DimensionMismatch {
                    expected: self.perm.len(),
                    actual: other.perm.len(),
                });
            }
            // Hamming-style positional disagreement (sufficient for the mock).
            Ok(self
                .perm
                .iter()
                .zip(other.perm.iter())
                .filter(|(a, b)| a != b)
                .count() as f64)
        }

        fn trace_prefix() -> &'static str {
            "perm"
        }
    }

    impl PermutationGenome for MockPermGenome {
        fn permutation(&self) -> &[usize] {
            &self.perm
        }

        fn permutation_mut(&mut self) -> &mut [usize] {
            &mut self.perm
        }

        fn from_permutation(perm: Vec<usize>) -> Result<Self, GenomeError> {
            Ok(Self { perm })
        }
    }

    #[test]
    fn test_permutation_genome_is_valid() {
        let valid = MockPermGenome {
            perm: vec![2, 0, 1, 3],
        };
        assert!(valid.is_valid_permutation());

        let invalid_dup = MockPermGenome {
            perm: vec![0, 1, 1, 3],
        };
        assert!(!invalid_dup.is_valid_permutation());

        let invalid_range = MockPermGenome {
            perm: vec![0, 1, 5, 3],
        };
        assert!(!invalid_range.is_valid_permutation());

        let empty = MockPermGenome { perm: vec![] };
        assert!(empty.is_valid_permutation());
    }

    #[test]
    fn test_permutation_genome_trace_roundtrip() {
        let original = MockPermGenome {
            perm: vec![3, 1, 4, 0, 2],
        };
        let trace = original.to_trace();
        let recovered = MockPermGenome::from_trace(&trace).unwrap();
        assert_eq!(original, recovered);
    }
}
