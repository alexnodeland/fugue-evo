//! Bit string genome
//!
//! This module provides a fixed-length bit string genome type for combinatorial optimization.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{BinaryGenome, EvolutionaryGenome};

/// Fixed-length bit string genome
///
/// This genome type represents binary optimization problems where
/// solutions are vectors of boolean values.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitString {
    /// The bits of this genome
    bits: Vec<bool>,
}

impl BitString {
    /// Create a new bit string with the given bits
    pub fn new(bits: Vec<bool>) -> Self {
        Self { bits }
    }

    /// Create an all-zeros bit string of the given length
    pub fn zeros(length: usize) -> Self {
        Self {
            bits: vec![false; length],
        }
    }

    /// Create an all-ones bit string of the given length
    pub fn ones(length: usize) -> Self {
        Self {
            bits: vec![true; length],
        }
    }

    /// Create a bit string from a u64 with the given length
    pub fn from_u64(value: u64, length: usize) -> Self {
        assert!(length <= 64, "Length must be <= 64 for u64 conversion");
        let bits = (0..length).map(|i| (value >> i) & 1 == 1).collect();
        Self { bits }
    }

    /// Convert to a u64 (only valid for length <= 64)
    pub fn to_u64(&self) -> Option<u64> {
        if self.bits.len() > 64 {
            return None;
        }
        let mut value = 0u64;
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                value |= 1 << i;
            }
        }
        Some(value)
    }

    /// Get the length of the bit string
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if the bit string is empty
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Get a specific bit
    pub fn get(&self, index: usize) -> Option<bool> {
        self.bits.get(index).copied()
    }

    /// Set a specific bit
    pub fn set(&mut self, index: usize, value: bool) {
        if let Some(bit) = self.bits.get_mut(index) {
            *bit = value;
        }
    }

    /// Flip a specific bit
    pub fn flip(&mut self, index: usize) {
        if let Some(bit) = self.bits.get_mut(index) {
            *bit = !*bit;
        }
    }

    /// Flip all bits
    pub fn flip_all(&mut self) {
        for bit in &mut self.bits {
            *bit = !*bit;
        }
    }

    /// Get the complement (all bits flipped)
    pub fn complement(&self) -> Self {
        Self {
            bits: self.bits.iter().map(|b| !b).collect(),
        }
    }

    /// Hamming distance to another bit string
    pub fn hamming_distance(&self, other: &Self) -> usize {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Bitwise AND with another bit string
    pub fn and(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.bits.len() != other.bits.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.bits.len(),
                actual: other.bits.len(),
            });
        }
        Ok(Self {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(a, b)| *a && *b)
                .collect(),
        })
    }

    /// Bitwise OR with another bit string
    pub fn or(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.bits.len() != other.bits.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.bits.len(),
                actual: other.bits.len(),
            });
        }
        Ok(Self {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(a, b)| *a || *b)
                .collect(),
        })
    }

    /// Bitwise XOR with another bit string
    pub fn xor(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.bits.len() != other.bits.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.bits.len(),
                actual: other.bits.len(),
            });
        }
        Ok(Self {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(a, b)| *a ^ *b)
                .collect(),
        })
    }
}

impl EvolutionaryGenome for BitString {
    type Allele = bool;
    type Phenotype = Vec<bool>;

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

    fn as_slice(&self) -> Option<&[bool]> {
        Some(&self.bits)
    }

    fn as_mut_slice(&mut self) -> Option<&mut [bool]> {
        Some(&mut self.bits)
    }

    fn distance(&self, other: &Self) -> f64 {
        self.hamming_distance(other) as f64
    }
}

impl BinaryGenome for BitString {
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

impl std::ops::Index<usize> for BitString {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.bits[index]
    }
}

impl From<Vec<bool>> for BitString {
    fn from(bits: Vec<bool>) -> Self {
        Self { bits }
    }
}

impl From<BitString> for Vec<bool> {
    fn from(genome: BitString) -> Self {
        genome.bits
    }
}

impl<const N: usize> From<[bool; N]> for BitString {
    fn from(arr: [bool; N]) -> Self {
        Self { bits: arr.to_vec() }
    }
}

impl IntoIterator for BitString {
    type Item = bool;
    type IntoIter = std::vec::IntoIter<bool>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.into_iter()
    }
}

impl<'a> IntoIterator for &'a BitString {
    type Item = &'a bool;
    type IntoIter = std::slice::Iter<'a, bool>;

    fn into_iter(self) -> Self::IntoIter {
        self.bits.iter()
    }
}

impl std::fmt::Display for BitString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for bit in &self.bits {
            write!(f, "{}", if *bit { '1' } else { '0' })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_string_new() {
        let bs = BitString::new(vec![true, false, true]);
        assert_eq!(bs.len(), 3);
        assert_eq!(bs.bits(), &[true, false, true]);
    }

    #[test]
    fn test_bit_string_zeros() {
        let bs = BitString::zeros(5);
        assert_eq!(bs.len(), 5);
        assert!(bs.bits().iter().all(|&b| !b));
        assert_eq!(bs.count_ones(), 0);
        assert_eq!(bs.count_zeros(), 5);
    }

    #[test]
    fn test_bit_string_ones() {
        let bs = BitString::ones(5);
        assert_eq!(bs.len(), 5);
        assert!(bs.bits().iter().all(|&b| b));
        assert_eq!(bs.count_ones(), 5);
        assert_eq!(bs.count_zeros(), 0);
    }

    #[test]
    fn test_bit_string_from_u64() {
        let bs = BitString::from_u64(0b101, 4);
        assert_eq!(bs.bits(), &[true, false, true, false]);
    }

    #[test]
    fn test_bit_string_to_u64() {
        let bs = BitString::new(vec![true, false, true, false]);
        assert_eq!(bs.to_u64(), Some(0b0101));

        let long_bs = BitString::zeros(100);
        assert_eq!(long_bs.to_u64(), None);
    }

    #[test]
    fn test_bit_string_get_set() {
        let mut bs = BitString::zeros(3);
        assert_eq!(bs.get(0), Some(false));
        assert_eq!(bs.get(3), None);

        bs.set(1, true);
        assert_eq!(bs.get(1), Some(true));
    }

    #[test]
    fn test_bit_string_flip() {
        let mut bs = BitString::zeros(3);
        bs.flip(1);
        assert_eq!(bs.bits(), &[false, true, false]);
    }

    #[test]
    fn test_bit_string_flip_all() {
        let mut bs = BitString::new(vec![true, false, true]);
        bs.flip_all();
        assert_eq!(bs.bits(), &[false, true, false]);
    }

    #[test]
    fn test_bit_string_complement() {
        let bs = BitString::new(vec![true, false, true]);
        let comp = bs.complement();
        assert_eq!(comp.bits(), &[false, true, false]);
    }

    #[test]
    fn test_bit_string_hamming_distance() {
        let bs1 = BitString::new(vec![true, false, true, false]);
        let bs2 = BitString::new(vec![true, true, false, false]);
        assert_eq!(bs1.hamming_distance(&bs2), 2);
    }

    #[test]
    fn test_bit_string_and() {
        let bs1 = BitString::new(vec![true, true, false, false]);
        let bs2 = BitString::new(vec![true, false, true, false]);
        let result = bs1.and(&bs2).unwrap();
        assert_eq!(result.bits(), &[true, false, false, false]);
    }

    #[test]
    fn test_bit_string_or() {
        let bs1 = BitString::new(vec![true, true, false, false]);
        let bs2 = BitString::new(vec![true, false, true, false]);
        let result = bs1.or(&bs2).unwrap();
        assert_eq!(result.bits(), &[true, true, true, false]);
    }

    #[test]
    fn test_bit_string_xor() {
        let bs1 = BitString::new(vec![true, true, false, false]);
        let bs2 = BitString::new(vec![true, false, true, false]);
        let result = bs1.xor(&bs2).unwrap();
        assert_eq!(result.bits(), &[false, true, true, false]);
    }

    #[test]
    fn test_bit_string_dimension_mismatch() {
        let bs1 = BitString::zeros(3);
        let bs2 = BitString::zeros(4);
        assert!(bs1.and(&bs2).is_err());
        assert!(bs1.or(&bs2).is_err());
        assert!(bs1.xor(&bs2).is_err());
    }

    #[test]
    fn test_bit_string_generate() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, 10);
        let bs = BitString::generate(&mut rng, &bounds);
        assert_eq!(bs.len(), 10);
    }

    #[test]
    fn test_bit_string_distance() {
        let bs1 = BitString::new(vec![true, false, true, false]);
        let bs2 = BitString::new(vec![true, true, false, false]);
        assert_eq!(bs1.distance(&bs2), 2.0);
    }

    #[test]
    fn test_bit_string_display() {
        let bs = BitString::new(vec![true, false, true, true]);
        assert_eq!(format!("{}", bs), "1011");
    }

    #[test]
    fn test_bit_string_indexing() {
        let bs = BitString::new(vec![true, false, true]);
        assert!(bs[0]);
        assert!(!bs[1]);
        assert!(bs[2]);
    }

    #[test]
    fn test_bit_string_from_array() {
        let bs: BitString = [true, false, true].into();
        assert_eq!(bs.bits(), &[true, false, true]);
    }

    #[test]
    fn test_bit_string_serialization() {
        let bs = BitString::new(vec![true, false, true]);
        let serialized = serde_json::to_string(&bs).unwrap();
        let deserialized: BitString = serde_json::from_str(&serialized).unwrap();
        assert_eq!(bs, deserialized);
    }

    #[test]
    fn test_bit_string_decode() {
        let bs = BitString::new(vec![true, false, true]);
        let phenotype = bs.decode();
        assert_eq!(phenotype, vec![true, false, true]);
    }
}
