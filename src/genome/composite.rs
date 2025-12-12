//! Composite genome for mixed-representation problems
//!
//! This module provides a genome type that combines two different genome types,
//! enabling optimization over heterogeneous solution spaces.

use fugue::{addr, ChoiceValue, Trace};
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
#[serde(bound = "A: Serialize + for<'de2> Deserialize<'de2>, B: Serialize + for<'de2> Deserialize<'de2>")]
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
    /// Stores dimensions and serialized data for each component.
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::default();

        // Store dimensions
        trace.insert_choice(
            addr!("composite", "first_dim"),
            ChoiceValue::Usize(self.first.dimension()),
            0.0,
        );
        trace.insert_choice(
            addr!("composite", "second_dim"),
            ChoiceValue::Usize(self.second.dimension()),
            0.0,
        );

        // Store first component's trace entries with prefix
        let first_trace = self.first.to_trace();
        for i in 0..self.first.dimension() {
            if let Some(val) = first_trace.get_f64(&addr!("gene", i)) {
                trace.insert_choice(addr!("first_gene", i), ChoiceValue::F64(val), 0.0);
            } else if let Some(val) = first_trace.get_bool(&addr!("bit", i)) {
                trace.insert_choice(addr!("first_bit", i), ChoiceValue::Bool(val), 0.0);
            } else if let Some(val) = first_trace.get_usize(&addr!("element", i)) {
                trace.insert_choice(addr!("first_element", i), ChoiceValue::Usize(val), 0.0);
            }
        }

        // Store second component's trace entries with prefix
        let second_trace = self.second.to_trace();
        for i in 0..self.second.dimension() {
            if let Some(val) = second_trace.get_f64(&addr!("gene", i)) {
                trace.insert_choice(addr!("second_gene", i), ChoiceValue::F64(val), 0.0);
            } else if let Some(val) = second_trace.get_bool(&addr!("bit", i)) {
                trace.insert_choice(addr!("second_bit", i), ChoiceValue::Bool(val), 0.0);
            } else if let Some(val) = second_trace.get_usize(&addr!("element", i)) {
                trace.insert_choice(addr!("second_element", i), ChoiceValue::Usize(val), 0.0);
            }
        }

        trace
    }

    /// Reconstruct composite genome from Fugue trace.
    ///
    /// Note: This is a simplified implementation that may lose some type information.
    /// For full fidelity, use serde serialization directly.
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        // Get dimensions
        let first_dim = trace
            .get_usize(&addr!("composite", "first_dim"))
            .ok_or_else(|| GenomeError::MissingAddress("composite#first_dim".to_string()))?;
        let second_dim = trace
            .get_usize(&addr!("composite", "second_dim"))
            .ok_or_else(|| GenomeError::MissingAddress("composite#second_dim".to_string()))?;

        // Reconstruct first component's trace
        let mut first_trace = Trace::default();
        for i in 0..first_dim {
            if let Some(val) = trace.get_f64(&addr!("first_gene", i)) {
                first_trace.insert_choice(addr!("gene", i), ChoiceValue::F64(val), 0.0);
            } else if let Some(val) = trace.get_bool(&addr!("first_bit", i)) {
                first_trace.insert_choice(addr!("bit", i), ChoiceValue::Bool(val), 0.0);
            } else if let Some(val) = trace.get_usize(&addr!("first_element", i)) {
                first_trace.insert_choice(addr!("element", i), ChoiceValue::Usize(val), 0.0);
            }
        }

        // Reconstruct second component's trace
        let mut second_trace = Trace::default();
        for i in 0..second_dim {
            if let Some(val) = trace.get_f64(&addr!("second_gene", i)) {
                second_trace.insert_choice(addr!("gene", i), ChoiceValue::F64(val), 0.0);
            } else if let Some(val) = trace.get_bool(&addr!("second_bit", i)) {
                second_trace.insert_choice(addr!("bit", i), ChoiceValue::Bool(val), 0.0);
            } else if let Some(val) = trace.get_usize(&addr!("second_element", i)) {
                second_trace.insert_choice(addr!("element", i), ChoiceValue::Usize(val), 0.0);
            }
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

    fn trace_prefix() -> &'static str {
        "composite"
    }
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
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::{BinaryGenome, RealValuedGenome};

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
}
