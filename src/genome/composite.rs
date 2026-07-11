//! Composite genome for mixed-representation problems
//!
//! This module provides a genome type that combines two different genome types,
//! enabling optimization over heterogeneous solution spaces.

use fugue::{Address, Trace};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;

/// A composite genome combining two different genome types
///
/// This is useful for problems that require multiple representations,
/// such as:
/// - Continuous parameters + discrete choices
/// - Feature selection (binary) + feature weights (continuous)
/// - Topology (permutation) + parameters (continuous)
///
/// # Type Parameters
/// - `A`: The first genome type
/// - `B`: The second genome type
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    bound = "A: Serialize + for<'de2> Deserialize<'de2>, B: Serialize + for<'de2> Deserialize<'de2>"
)]
pub struct CompositeGenome<A, B>
where
    A: EvolutionaryGenome,
    B: EvolutionaryGenome,
{
    /// First component genome
    pub first: A,
    /// Second component genome
    pub second: B,
}

impl<A, B> CompositeGenome<A, B>
where
    A: EvolutionaryGenome,
    B: EvolutionaryGenome,
{
    /// Create a new composite genome from two components
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Get a reference to the first component
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Get a mutable reference to the first component
    pub fn first_mut(&mut self) -> &mut A {
        &mut self.first
    }

    /// Get a reference to the second component
    pub fn second(&self) -> &B {
        &self.second
    }

    /// Get a mutable reference to the second component
    pub fn second_mut(&mut self) -> &mut B {
        &mut self.second
    }

    /// Consume and return the components
    pub fn into_parts(self) -> (A, B) {
        (self.first, self.second)
    }

    /// Map a function over the first component
    pub fn map_first<F, C>(self, f: F) -> CompositeGenome<C, B>
    where
        F: FnOnce(A) -> C,
        C: EvolutionaryGenome,
    {
        CompositeGenome {
            first: f(self.first),
            second: self.second,
        }
    }

    /// Map a function over the second component
    pub fn map_second<F, C>(self, f: F) -> CompositeGenome<A, C>
    where
        F: FnOnce(B) -> C,
        C: EvolutionaryGenome,
    {
        CompositeGenome {
            first: self.first,
            second: f(self.second),
        }
    }
}

impl<A, B> EvolutionaryGenome for CompositeGenome<A, B>
where
    A: EvolutionaryGenome + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
    B: EvolutionaryGenome + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>,
{
    type Allele = (A::Allele, B::Allele);
    type Phenotype = (A::Phenotype, B::Phenotype);

    /// Convert composite genome to Fugue trace.
    ///
    /// Each component is delegated to its own [`to_trace`](EvolutionaryGenome::to_trace),
    /// and every entry of the resulting nested trace is copied verbatim under a
    /// namespace prefix (`"first/"` / `"second/"`). This preserves full fidelity
    /// for *any* component genome type — including `Permutation`, `TreeGenome`
    /// and future encodings — rather than special-casing a fixed set of address
    /// name literals.
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();
        namespace_into(&self.first.to_trace(), "first", &mut trace);
        namespace_into(&self.second.to_trace(), "second", &mut trace);
        trace
    }

    /// Reconstruct composite genome from Fugue trace.
    ///
    /// Strips the `"first/"` / `"second/"` namespace back off each entry to
    /// rebuild the two component sub-traces, then delegates to each component's
    /// own [`from_trace`](EvolutionaryGenome::from_trace).
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let (first_trace, saw_first) = extract_namespace(trace, "first");
        let (second_trace, saw_second) = extract_namespace(trace, "second");

        if !saw_first {
            return Err(GenomeError::MissingAddress("first/*".to_string()));
        }
        if !saw_second {
            return Err(GenomeError::MissingAddress("second/*".to_string()));
        }

        let first = A::from_trace(&first_trace)?;
        let second = B::from_trace(&second_trace)?;

        Ok(Self { first, second })
    }

    fn decode(&self) -> Self::Phenotype {
        (self.first.decode(), self.second.decode())
    }

    fn dimension(&self) -> usize {
        self.first.dimension() + self.second.dimension()
    }

    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self {
        // Split bounds between components
        // This is a simplification - in practice, you'd want separate bounds
        let first_dim = bounds.dimension() / 2;
        let second_dim = bounds.dimension() - first_dim;

        let first_bounds =
            MultiBounds::new(bounds.bounds.iter().take(first_dim).cloned().collect());
        let second_bounds = MultiBounds::new(
            bounds
                .bounds
                .iter()
                .skip(first_dim)
                .take(second_dim)
                .cloned()
                .collect(),
        );

        Self {
            first: A::generate(rng, &first_bounds),
            second: B::generate(rng, &second_bounds),
        }
    }

    fn distance(&self, other: &Self) -> f64 {
        // Combined distance (could be weighted)
        self.first.distance(&other.first) + self.second.distance(&other.second)
    }

    fn try_distance(&self, other: &Self) -> Result<f64, GenomeError> {
        Ok(self.first.try_distance(&other.first)? + self.second.try_distance(&other.second)?)
    }

    fn trace_prefix() -> &'static str {
        "composite"
    }
}

/// Copy every entry of `src` into `dst`, prefixing each address with
/// `"<namespace>/"`. Used to merge a component's own trace into the composite
/// trace without interpreting the component's address scheme.
fn namespace_into(src: &Trace, namespace: &str, dst: &mut Trace) {
    for (addr, choice) in &src.choices {
        dst.insert_choice(
            Address(format!("{}/{}", namespace, addr.0)),
            choice.value.clone(),
            choice.logp,
        );
    }
}

/// Inverse of [`namespace_into`]: collect every entry whose address begins with
/// `"<namespace>/"` into a fresh trace with the prefix stripped back off.
/// Returns the reconstructed sub-trace and whether any entry was found.
fn extract_namespace(trace: &Trace, namespace: &str) -> (Trace, bool) {
    let prefix = format!("{namespace}/");
    let mut sub = Trace::default();
    let mut found = false;
    for (addr, choice) in &trace.choices {
        if let Some(rest) = addr.0.strip_prefix(&prefix) {
            sub.insert_choice(Address(rest.to_string()), choice.value.clone(), choice.logp);
            found = true;
        }
    }
    (sub, found)
}

/// Builder for composite bounds that tracks bounds for each component
#[derive(Clone, Debug)]
pub struct CompositeBounds {
    /// Bounds for the first component
    pub first_bounds: MultiBounds,
    /// Bounds for the second component
    pub second_bounds: MultiBounds,
}

impl CompositeBounds {
    /// Create composite bounds from two separate MultiBounds
    pub fn new(first_bounds: MultiBounds, second_bounds: MultiBounds) -> Self {
        Self {
            first_bounds,
            second_bounds,
        }
    }

    /// Get combined bounds (concatenated)
    pub fn combined(&self) -> MultiBounds {
        let mut all_bounds: Vec<_> = self.first_bounds.bounds.to_vec();
        all_bounds.extend(self.second_bounds.bounds.iter().cloned());
        MultiBounds::new(all_bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::bit_string::BitString;
    use crate::genome::permutation::Permutation;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::{BinaryGenome, PermutationGenome, RealValuedGenome};

    #[test]
    fn test_composite_creation() {
        let real = RealVector::new(vec![1.0, 2.0, 3.0]);
        let binary = BitString::new(vec![true, false, true, false]);

        let composite = CompositeGenome::new(real.clone(), binary.clone());

        assert_eq!(composite.first().genes(), real.genes());
        assert_eq!(composite.second().bits(), binary.bits());
    }

    #[test]
    fn test_composite_dimension() {
        let real = RealVector::new(vec![1.0, 2.0, 3.0]);
        let binary = BitString::new(vec![true, false, true, false]);

        let composite = CompositeGenome::new(real, binary);

        // 3 real + 4 binary = 7
        assert_eq!(composite.dimension(), 7);
    }

    #[test]
    fn test_composite_decode() {
        let real = RealVector::new(vec![1.0, 2.0, 3.0]);
        let binary = BitString::new(vec![true, false, true, false]);

        let composite = CompositeGenome::new(real, binary);
        let (decoded_real, decoded_binary) = composite.decode();

        assert_eq!(decoded_real, vec![1.0, 2.0, 3.0]);
        assert_eq!(decoded_binary, vec![true, false, true, false]);
    }

    #[test]
    fn test_composite_into_parts() {
        let real = RealVector::new(vec![1.0, 2.0]);
        let binary = BitString::new(vec![true, true, false]);

        let composite = CompositeGenome::new(real.clone(), binary.clone());
        let (r, b) = composite.into_parts();

        assert_eq!(r.genes(), real.genes());
        assert_eq!(b.bits(), binary.bits());
    }

    #[test]
    fn test_composite_map() {
        let real = RealVector::new(vec![1.0, 2.0]);
        let binary = BitString::new(vec![true, false]);

        let composite = CompositeGenome::new(real, binary);

        // Map first to double values
        let mapped = composite.map_first(|r| r.scale(2.0));
        assert_eq!(mapped.first().genes(), &[2.0, 4.0]);
    }

    #[test]
    fn test_composite_distance() {
        let c1 = CompositeGenome::new(
            RealVector::new(vec![0.0, 0.0]),
            BitString::new(vec![true, false]),
        );

        let c2 = CompositeGenome::new(
            RealVector::new(vec![3.0, 4.0]),
            BitString::new(vec![false, true]),
        );

        let dist = c1.distance(&c2);

        // Real distance = 5.0, binary distance = 2 (Hamming)
        assert!(dist > 0.0);
    }

    #[test]
    fn test_composite_generate() {
        let bounds = MultiBounds::symmetric(5.0, 6); // Split as 3+3
        let mut rng = rand::thread_rng();

        // This test requires that both component types can generate from bounds
        // For simplicity, test with two RealVectors
        let composite: CompositeGenome<RealVector, RealVector> =
            CompositeGenome::generate(&mut rng, &bounds);

        assert_eq!(composite.dimension(), 6);
    }

    #[test]
    fn test_composite_bounds() {
        use crate::genome::bounds::Bounds;

        let first_bounds = MultiBounds::symmetric(5.0, 3);
        let second_bounds = MultiBounds::uniform(Bounds::unit(), 4);

        let composite_bounds = CompositeBounds::new(first_bounds, second_bounds);
        let combined = composite_bounds.combined();

        assert_eq!(combined.dimension(), 7);
    }

    #[test]
    fn test_composite_first_mut() {
        let real = RealVector::new(vec![1.0, 2.0, 3.0]);
        let binary = BitString::new(vec![true, false]);

        let mut composite = CompositeGenome::new(real, binary);

        // Modify first component through mutable reference
        composite.first_mut().genes_mut()[0] = 10.0;

        assert_eq!(composite.first().genes()[0], 10.0);
    }

    #[test]
    fn test_composite_second_mut() {
        let real = RealVector::new(vec![1.0, 2.0]);
        let binary = BitString::new(vec![true, false, true]);

        let mut composite = CompositeGenome::new(real, binary);

        // Modify second component through mutable reference
        composite.second_mut().bits_mut()[0] = false;

        assert!(!composite.second().bits()[0]);
    }

    #[test]
    fn test_composite_map_second() {
        let real = RealVector::new(vec![1.0, 2.0]);
        let second_real = RealVector::new(vec![3.0, 4.0]);

        let composite = CompositeGenome::new(real, second_real);

        // Map second to double values
        let mapped = composite.map_second(|r| r.scale(2.0));
        assert_eq!(mapped.second().genes(), &[6.0, 8.0]);
    }

    #[test]
    fn test_composite_trace_roundtrip_real_vectors() {
        let first = RealVector::new(vec![1.5, 2.5, 3.5]);
        let second = RealVector::new(vec![4.5, 5.5]);

        let composite = CompositeGenome::new(first.clone(), second.clone());
        let trace = composite.to_trace();
        let recovered: CompositeGenome<RealVector, RealVector> =
            CompositeGenome::from_trace(&trace).expect("Should deserialize");

        assert_eq!(recovered.first().genes(), first.genes());
        assert_eq!(recovered.second().genes(), second.genes());
    }

    #[test]
    fn test_composite_trace_roundtrip_mixed() {
        // regression: EV-03 — full round-trip fidelity for a mixed composite.
        let real = RealVector::new(vec![1.0, 2.0]);
        let binary = BitString::new(vec![true, false, true]);

        let composite = CompositeGenome::new(real.clone(), binary.clone());
        let trace = composite.to_trace();
        let recovered: CompositeGenome<RealVector, BitString> =
            CompositeGenome::from_trace(&trace).expect("mixed composite should round-trip");

        assert_eq!(recovered.first().genes(), real.genes());
        assert_eq!(recovered.second().bits(), binary.bits());
    }

    #[test]
    fn test_composite_trace_roundtrip_permutation_realvector() {
        // regression: EV-03 — previously to_trace dropped every permutation value
        // (no "perm" prefix was recognized) and from_trace always failed. The
        // module doc explicitly showcases "topology (permutation) + parameters".
        let perm = Permutation::new(vec![2, 0, 3, 1]);
        let real = RealVector::new(vec![1.5, -2.5, 3.5]);

        let composite = CompositeGenome::new(perm.clone(), real.clone());
        let trace = composite.to_trace();
        let recovered: CompositeGenome<Permutation, RealVector> =
            CompositeGenome::from_trace(&trace)
                .expect("permutation+real composite should round-trip");

        assert_eq!(recovered.first().permutation(), perm.permutation());
        assert_eq!(recovered.second().genes(), real.genes());
        assert_eq!(recovered, composite);
    }

    #[test]
    fn test_composite_trace_roundtrip_bitstring_permutation() {
        // regression: EV-03 — round-trip for BitString + Permutation.
        let bits = BitString::new(vec![true, false, true, true]);
        let perm = Permutation::new(vec![1, 3, 0, 2]);

        let composite = CompositeGenome::new(bits.clone(), perm.clone());
        let trace = composite.to_trace();
        let recovered: CompositeGenome<BitString, Permutation> =
            CompositeGenome::from_trace(&trace)
                .expect("bitstring+permutation composite should round-trip");

        assert_eq!(recovered.first().bits(), bits.bits());
        assert_eq!(recovered.second().permutation(), perm.permutation());
        assert_eq!(recovered, composite);
    }

    #[test]
    fn test_composite_trace_prefix() {
        assert_eq!(
            <CompositeGenome<RealVector, BitString>>::trace_prefix(),
            "composite"
        );
    }

    #[test]
    fn test_composite_from_trace_missing_dim_error() {
        use fugue::Trace;
        let empty_trace = Trace::default();

        let result: Result<CompositeGenome<RealVector, RealVector>, _> =
            CompositeGenome::from_trace(&empty_trace);

        assert!(result.is_err());
    }
}
