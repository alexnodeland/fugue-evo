//! Permutation genome
//!
//! This module provides a permutation genome type for ordering problems
//! (e.g., TSP, scheduling) with Fugue trace integration.

use fugue::{addr, ChoiceValue, Trace};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{EvolutionaryGenome, PermutationGenome};

/// Permutation genome for ordering problems
///
/// Represents a permutation of indices 0..n, commonly used for:
/// - Traveling Salesman Problem (TSP)
/// - Job Shop Scheduling
/// - Vehicle Routing Problems
/// - Any problem where the solution is an ordering of elements
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permutation {
    /// The permutation (indices 0..n in some order)
    perm: Vec<usize>,
}

impl Permutation {
    /// Create a new permutation from a vector of indices
    ///
    /// # Panics
    /// Panics if the input is not a valid permutation of 0..n
    pub fn new(perm: Vec<usize>) -> Self {
        let result = Self { perm };
        assert!(
            result.is_valid_permutation(),
            "Input must be a valid permutation of 0..n"
        );
        result
    }

    /// Create a permutation from a vector without validation
    ///
    /// # Safety
    /// The caller must ensure the input is a valid permutation of 0..n
    pub fn new_unchecked(perm: Vec<usize>) -> Self {
        Self { perm }
    }

    /// Try to create a permutation, returning an error if invalid
    pub fn try_new(perm: Vec<usize>) -> Result<Self, GenomeError> {
        let result = Self { perm };
        if result.is_valid_permutation() {
            Ok(result)
        } else {
            Err(GenomeError::InvalidStructure(
                "Input is not a valid permutation of 0..n".to_string(),
            ))
        }
    }

    /// Create the identity permutation [0, 1, 2, ..., n-1]
    pub fn identity(n: usize) -> Self {
        Self {
            perm: (0..n).collect(),
        }
    }

    /// Create a random permutation of size n
    pub fn random<R: Rng>(n: usize, rng: &mut R) -> Self {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(rng);
        Self { perm }
    }

    /// Get the length of the permutation
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Check if the permutation is empty
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Get the element at index i
    pub fn get(&self, i: usize) -> Option<usize> {
        self.perm.get(i).copied()
    }

    /// Get the inverse permutation
    ///
    /// If `perm[i] = j`, then `inverse[j] = i`
    pub fn inverse(&self) -> Self {
        let n = self.perm.len();
        let mut inv = vec![0; n];
        for (i, &j) in self.perm.iter().enumerate() {
            inv[j] = i;
        }
        Self { perm: inv }
    }

    /// Compose this permutation with another
    ///
    /// Returns a permutation where `result[i] = other[self[i]]`
    pub fn compose(&self, other: &Self) -> Result<Self, GenomeError> {
        if self.perm.len() != other.perm.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.perm.len(),
                actual: other.perm.len(),
            });
        }
        let composed: Vec<usize> = self.perm.iter().map(|&i| other.perm[i]).collect();
        Ok(Self { perm: composed })
    }

    /// Swap two elements at positions i and j
    pub fn swap(&mut self, i: usize, j: usize) {
        self.perm.swap(i, j);
    }

    /// Reverse a segment from start to end (inclusive)
    pub fn reverse_segment(&mut self, start: usize, end: usize) {
        if start < end && end < self.perm.len() {
            self.perm[start..=end].reverse();
        }
    }

    /// Insert element at position `from` to position `to`
    pub fn insert(&mut self, from: usize, to: usize) {
        if from == to || from >= self.perm.len() || to >= self.perm.len() {
            return;
        }
        let elem = self.perm.remove(from);
        self.perm.insert(to, elem);
    }

    /// Calculate the number of inversions (disorder measure)
    ///
    /// An inversion is a pair (i, j) where i < j but `perm[i] > perm[j]`.
    /// Returns a value in `[0, n*(n-1)/2]` where 0 means sorted.
    pub fn inversions(&self) -> usize {
        let n = self.perm.len();
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.perm[i] > self.perm[j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Calculate Kendall tau distance to another permutation
    ///
    /// Counts the number of pairwise disagreements (i.e., pairs that are
    /// in different order in the two permutations).
    pub fn kendall_tau_distance(&self, other: &Self) -> Result<usize, GenomeError> {
        if self.perm.len() != other.perm.len() {
            return Err(GenomeError::DimensionMismatch {
                expected: self.perm.len(),
                actual: other.perm.len(),
            });
        }

        // Compose with inverse of other to get relative order
        let other_inv = other.inverse();
        let composed = self.compose(&other_inv)?;

        // Count inversions in the composed permutation
        Ok(composed.inversions())
    }

    /// Check if this is a cyclic permutation (single cycle)
    pub fn is_cyclic(&self) -> bool {
        if self.perm.is_empty() {
            return true;
        }

        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut current = 0;
        let mut cycle_len = 0;

        while !visited[current] {
            visited[current] = true;
            current = self.perm[current];
            cycle_len += 1;
        }

        cycle_len == n && current == 0
    }

    /// Get the underlying vector
    pub fn into_inner(self) -> Vec<usize> {
        self.perm
    }

    /// Get a reference to the underlying slice
    pub fn as_slice(&self) -> &[usize] {
        &self.perm
    }
}

impl EvolutionaryGenome for Permutation {
    type Allele = usize;
    type Phenotype = Vec<usize>;

    /// Convert Permutation to Fugue trace.
    ///
    /// Each position is stored at address "perm#i" where i is the index.
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();
        for (i, &val) in self.perm.iter().enumerate() {
            trace.insert_choice(addr!("perm", i), ChoiceValue::Usize(val), 0.0);
        }
        trace
    }

    /// Reconstruct Permutation from Fugue trace.
    ///
    /// Reads values from addresses "perm#0", "perm#1", ... until no more are found.
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
        Self::try_new(perm)
    }

    fn decode(&self) -> Self::Phenotype {
        self.perm.clone()
    }

    fn dimension(&self) -> usize {
        self.perm.len()
    }

    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        let n = bounds.dimension();
        Self::random(n, rng)
    }

    fn distance(&self, other: &Self) -> f64 {
        self.kendall_tau_distance(other).unwrap_or(0) as f64
    }

    fn trace_prefix() -> &'static str {
        "perm"
    }
}

impl PermutationGenome for Permutation {
    fn permutation(&self) -> &[usize] {
        &self.perm
    }

    fn permutation_mut(&mut self) -> &mut [usize] {
        &mut self.perm
    }

    fn from_permutation(perm: Vec<usize>) -> Result<Self, GenomeError> {
        Self::try_new(perm)
    }
}

impl std::ops::Index<usize> for Permutation {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.perm[index]
    }
}

impl From<Vec<usize>> for Permutation {
    fn from(perm: Vec<usize>) -> Self {
        Self::new(perm)
    }
}

impl From<Permutation> for Vec<usize> {
    fn from(p: Permutation) -> Self {
        p.perm
    }
}

impl IntoIterator for Permutation {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.perm.into_iter()
    }
}

impl<'a> IntoIterator for &'a Permutation {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.perm.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::traits::PermutationGenome;

    #[test]
    fn test_permutation_new() {
        let p = Permutation::new(vec![2, 0, 1, 3]);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0], 2);
        assert_eq!(p[1], 0);
    }

    #[test]
    #[should_panic(expected = "valid permutation")]
    fn test_permutation_new_invalid_duplicate() {
        Permutation::new(vec![0, 1, 1, 3]);
    }

    #[test]
    #[should_panic(expected = "valid permutation")]
    fn test_permutation_new_invalid_out_of_range() {
        Permutation::new(vec![0, 1, 5, 3]);
    }

    #[test]
    fn test_permutation_try_new() {
        assert!(Permutation::try_new(vec![2, 0, 1, 3]).is_ok());
        assert!(Permutation::try_new(vec![0, 1, 1, 3]).is_err());
    }

    #[test]
    fn test_permutation_identity() {
        let p = Permutation::identity(5);
        assert_eq!(p.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_permutation_random() {
        let mut rng = rand::thread_rng();
        let p = Permutation::random(10, &mut rng);
        assert!(p.is_valid_permutation());
        assert_eq!(p.len(), 10);
    }

    #[test]
    fn test_permutation_inverse() {
        let p = Permutation::new(vec![2, 0, 3, 1]);
        let inv = p.inverse();
        // If p[i] = j, then inv[j] = i
        // p[0]=2 -> inv[2]=0, p[1]=0 -> inv[0]=1, p[2]=3 -> inv[3]=2, p[3]=1 -> inv[1]=3
        assert_eq!(inv.as_slice(), &[1, 3, 0, 2]);

        // Composing with inverse should give identity
        let composed = p.compose(&inv).unwrap();
        assert_eq!(composed.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_permutation_compose() {
        let p1 = Permutation::new(vec![1, 2, 0]);
        let p2 = Permutation::new(vec![2, 0, 1]);
        let composed = p1.compose(&p2).unwrap();
        // composed[i] = p2[p1[i]]
        // composed[0] = p2[1] = 0, composed[1] = p2[2] = 1, composed[2] = p2[0] = 2
        assert_eq!(composed.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn test_permutation_swap() {
        let mut p = Permutation::new(vec![0, 1, 2, 3]);
        p.swap(0, 3);
        assert_eq!(p.as_slice(), &[3, 1, 2, 0]);
    }

    #[test]
    fn test_permutation_reverse_segment() {
        let mut p = Permutation::new(vec![0, 1, 2, 3, 4]);
        p.reverse_segment(1, 3);
        assert_eq!(p.as_slice(), &[0, 3, 2, 1, 4]);
    }

    #[test]
    fn test_permutation_insert() {
        let mut p = Permutation::new(vec![0, 1, 2, 3, 4]);
        p.insert(1, 4);
        assert_eq!(p.as_slice(), &[0, 2, 3, 4, 1]);
    }

    #[test]
    fn test_permutation_inversions() {
        // Sorted: 0 inversions
        let p1 = Permutation::identity(5);
        assert_eq!(p1.inversions(), 0);

        // Reversed: n*(n-1)/2 inversions
        let p2 = Permutation::new(vec![4, 3, 2, 1, 0]);
        assert_eq!(p2.inversions(), 10); // 5*4/2 = 10

        // Single swap: 1 inversion
        let p3 = Permutation::new(vec![1, 0, 2, 3, 4]);
        assert_eq!(p3.inversions(), 1);
    }

    #[test]
    fn test_permutation_kendall_tau() {
        let p1 = Permutation::new(vec![0, 1, 2, 3]);
        let p2 = Permutation::new(vec![0, 1, 2, 3]);
        assert_eq!(p1.kendall_tau_distance(&p2).unwrap(), 0);

        let p3 = Permutation::new(vec![0, 1, 3, 2]);
        assert_eq!(p1.kendall_tau_distance(&p3).unwrap(), 1);

        let p4 = Permutation::new(vec![3, 2, 1, 0]);
        assert_eq!(p1.kendall_tau_distance(&p4).unwrap(), 6);
    }

    #[test]
    fn test_permutation_is_cyclic() {
        // Single cycle (3 -> 1 -> 2 -> 0 -> 3)
        let cyclic = Permutation::new(vec![3, 2, 0, 1]);
        // Let's trace: 0 -> 3 -> 1 -> 2 -> 0, that's 4 elements in one cycle
        assert!(cyclic.is_cyclic());

        // Not a single cycle: identity has n fixed points (1-cycles)
        let identity = Permutation::identity(4);
        assert!(!identity.is_cyclic()); // Each element maps to itself

        // Empty is trivially cyclic
        let empty = Permutation::identity(0);
        assert!(empty.is_cyclic());
    }

    #[test]
    fn test_permutation_decode() {
        let p = Permutation::new(vec![2, 0, 1]);
        assert_eq!(p.decode(), vec![2, 0, 1]);
    }

    #[test]
    fn test_permutation_dimension() {
        let p = Permutation::new(vec![2, 0, 1, 3, 4]);
        assert_eq!(p.dimension(), 5);
    }

    #[test]
    fn test_permutation_generate() {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, 10);
        let p = Permutation::generate(&mut rng, &bounds);
        assert_eq!(p.dimension(), 10);
        assert!(p.is_valid_permutation());
    }

    #[test]
    fn test_permutation_distance() {
        let p1 = Permutation::new(vec![0, 1, 2, 3]);
        let p2 = Permutation::new(vec![3, 2, 1, 0]);
        assert_eq!(p1.distance(&p2), 6.0);
    }

    #[test]
    fn test_permutation_to_trace() {
        let p = Permutation::new(vec![2, 0, 1]);
        let trace = p.to_trace();

        assert_eq!(trace.get_usize(&addr!("perm", 0)), Some(2));
        assert_eq!(trace.get_usize(&addr!("perm", 1)), Some(0));
        assert_eq!(trace.get_usize(&addr!("perm", 2)), Some(1));
        assert_eq!(trace.get_usize(&addr!("perm", 3)), None);
    }

    #[test]
    fn test_permutation_from_trace() {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("perm", 0), ChoiceValue::Usize(1), 0.0);
        trace.insert_choice(addr!("perm", 1), ChoiceValue::Usize(2), 0.0);
        trace.insert_choice(addr!("perm", 2), ChoiceValue::Usize(0), 0.0);

        let p = Permutation::from_trace(&trace).unwrap();
        assert_eq!(p.as_slice(), &[1, 2, 0]);
    }

    #[test]
    fn test_permutation_trace_roundtrip() {
        let original = Permutation::new(vec![4, 2, 0, 3, 1]);
        let trace = original.to_trace();
        let recovered = Permutation::from_trace(&trace).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_permutation_from_trace_invalid() {
        let mut trace = Trace::default();
        trace.insert_choice(addr!("perm", 0), ChoiceValue::Usize(0), 0.0);
        trace.insert_choice(addr!("perm", 1), ChoiceValue::Usize(0), 0.0); // duplicate!

        let result = Permutation::from_trace(&trace);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_from_trace_empty() {
        let trace = Trace::default();
        let result = Permutation::from_trace(&trace);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_serialization() {
        let p = Permutation::new(vec![2, 0, 1, 3]);
        let serialized = serde_json::to_string(&p).unwrap();
        let deserialized: Permutation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(p, deserialized);
    }

    #[test]
    fn test_permutation_iteration() {
        let p = Permutation::new(vec![2, 0, 1]);
        let collected: Vec<usize> = p.into_iter().collect();
        assert_eq!(collected, vec![2, 0, 1]);
    }

    #[test]
    fn test_permutation_ref_iteration() {
        let p = Permutation::new(vec![2, 0, 1]);
        let sum: usize = p.into_iter().sum();
        assert_eq!(sum, 3);
    }

    #[test]
    fn test_permutation_into_inner() {
        let p = Permutation::new(vec![2, 0, 1]);
        let v: Vec<usize> = p.into_inner();
        assert_eq!(v, vec![2, 0, 1]);
    }

    #[test]
    fn test_permutation_from_vec() {
        let p: Permutation = vec![1, 0, 2].into();
        assert_eq!(p.as_slice(), &[1, 0, 2]);
    }
}
