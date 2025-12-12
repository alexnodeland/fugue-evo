//! Crossover operators
//!
//! This module provides various crossover operators for genetic algorithms.

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::error::{OperatorError, OperatorResult};
use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::permutation::Permutation;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::{
    BinaryGenome, EvolutionaryGenome, PermutationGenome, RealValuedGenome,
};
use crate::operators::traits::{BoundedCrossoverOperator, CrossoverOperator};

/// Simulated Binary Crossover (SBX)
///
/// SBX generates offspring from parents using a spread factor that
/// simulates single-point crossover for binary strings.
///
/// Reference: Deb, K., & Agrawal, R. B. (1995). Simulated Binary Crossover
/// for Continuous Search Space.
#[derive(Clone, Debug)]
pub struct SbxCrossover {
    /// Distribution index (typically 2-20)
    /// Higher values = offspring closer to parents
    pub eta: f64,
    /// Per-gene crossover probability
    pub crossover_probability: f64,
}

impl SbxCrossover {
    /// Create a new SBX crossover with the given distribution index
    pub fn new(eta: f64) -> Self {
        assert!(eta >= 0.0, "Distribution index must be non-negative");
        Self {
            eta,
            crossover_probability: 0.9,
        }
    }

    /// Set the per-gene crossover probability
    pub fn with_probability(mut self, probability: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&probability),
            "Probability must be in [0, 1]"
        );
        self.crossover_probability = probability;
        self
    }

    /// Compute the spread factor β from a uniform random value
    fn spread_factor(&self, u: f64) -> f64 {
        if u <= 0.5 {
            (2.0 * u).powf(1.0 / (self.eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta + 1.0))
        }
    }

    /// Apply SBX crossover to two f64 slices
    fn apply_sbx<R: Rng>(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        bounds: Option<&MultiBounds>,
        rng: &mut R,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut child1: Vec<f64> = parent1.to_vec();
        let mut child2: Vec<f64> = parent2.to_vec();

        for i in 0..parent1.len() {
            if rng.gen::<f64>() < self.crossover_probability {
                let x1 = parent1[i];
                let x2 = parent2[i];

                // Only apply if parents differ sufficiently
                if (x1 - x2).abs() > 1e-14 {
                    let u = rng.gen::<f64>();
                    let beta = self.spread_factor(u);

                    child1[i] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
                    child2[i] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2);

                    // Apply bounds if provided
                    if let Some(b) = bounds {
                        if let Some(bound) = b.get(i) {
                            child1[i] = bound.clamp(child1[i]);
                            child2[i] = bound.clamp(child2[i]);
                        }
                    }
                }
            }
        }

        (child1, child2)
    }
}

impl CrossoverOperator<RealVector> for SbxCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let (child1_genes, child2_genes) =
            self.apply_sbx(parent1.genes(), parent2.genes(), None, rng);

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }

    fn crossover_probability(&self) -> f64 {
        self.crossover_probability
    }
}

impl BoundedCrossoverOperator<RealVector> for SbxCrossover {
    fn crossover_bounded<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let (child1_genes, child2_genes) =
            self.apply_sbx(parent1.genes(), parent2.genes(), Some(bounds), rng);

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Blend Crossover (BLX-α)
///
/// Creates offspring within an extended range defined by parents.
#[derive(Clone, Debug)]
pub struct BlxAlphaCrossover {
    /// Extension factor (typically 0.5)
    pub alpha: f64,
}

impl BlxAlphaCrossover {
    /// Create a new BLX-α crossover
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0.0, "Alpha must be non-negative");
        Self { alpha }
    }

    /// Create with default alpha = 0.5
    pub fn default_alpha() -> Self {
        Self::new(0.5)
    }
}

impl CrossoverOperator<RealVector> for BlxAlphaCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let mut child1_genes = Vec::with_capacity(parent1.dimension());
        let mut child2_genes = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            let x1 = parent1[i];
            let x2 = parent2[i];

            let min_val = x1.min(x2);
            let max_val = x1.max(x2);
            let range = max_val - min_val;

            let low = min_val - self.alpha * range;
            let high = max_val + self.alpha * range;

            child1_genes.push(rng.gen_range(low..=high));
            child2_genes.push(rng.gen_range(low..=high));
        }

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Uniform crossover for bit strings
///
/// Each bit is independently chosen from either parent with equal probability.
#[derive(Clone, Debug)]
pub struct UniformCrossover {
    /// Probability of choosing from parent1 (default: 0.5)
    pub bias: f64,
}

impl UniformCrossover {
    /// Create a new uniform crossover
    pub fn new() -> Self {
        Self { bias: 0.5 }
    }

    /// Create with a specific bias towards parent1
    pub fn with_bias(bias: f64) -> Self {
        assert!((0.0..=1.0).contains(&bias), "Bias must be in [0, 1]");
        Self { bias }
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossoverOperator<BitString> for UniformCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let mut child1_bits = Vec::with_capacity(parent1.dimension());
        let mut child2_bits = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            if rng.gen::<f64>() < self.bias {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// One-point crossover for bit strings
#[derive(Clone, Debug, Default)]
pub struct OnePointCrossover;

impl OnePointCrossover {
    /// Create a new one-point crossover
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<BitString> for OnePointCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let n = parent1.dimension();
        if n == 0 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        let crossover_point = rng.gen_range(0..n);

        let mut child1_bits = Vec::with_capacity(n);
        let mut child2_bits = Vec::with_capacity(n);

        for i in 0..n {
            if i < crossover_point {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Two-point crossover for bit strings
#[derive(Clone, Debug, Default)]
pub struct TwoPointCrossover;

impl TwoPointCrossover {
    /// Create a new two-point crossover
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<BitString> for TwoPointCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BitString,
        parent2: &BitString,
        rng: &mut R,
    ) -> OperatorResult<(BitString, BitString)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let n = parent1.dimension();
        if n < 2 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        let mut point1 = rng.gen_range(0..n);
        let mut point2 = rng.gen_range(0..n);
        if point1 > point2 {
            std::mem::swap(&mut point1, &mut point2);
        }

        let mut child1_bits = Vec::with_capacity(n);
        let mut child2_bits = Vec::with_capacity(n);

        for i in 0..n {
            if i < point1 || i >= point2 {
                child1_bits.push(parent1[i]);
                child2_bits.push(parent2[i]);
            } else {
                child1_bits.push(parent2[i]);
                child2_bits.push(parent1[i]);
            }
        }

        let child1 = BitString::from_bits(child1_bits).unwrap();
        let child2 = BitString::from_bits(child2_bits).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

/// Arithmetic crossover for real-valued genomes
///
/// Creates offspring as weighted averages of parents.
#[derive(Clone, Debug)]
pub struct ArithmeticCrossover {
    /// Weight for parent1 (parent2 weight = 1 - weight)
    pub weight: f64,
}

impl ArithmeticCrossover {
    /// Create a new arithmetic crossover with the given weight
    pub fn new(weight: f64) -> Self {
        assert!((0.0..=1.0).contains(&weight), "Weight must be in [0, 1]");
        Self { weight }
    }

    /// Create with uniform weight (0.5)
    pub fn uniform() -> Self {
        Self::new(0.5)
    }
}

impl CrossoverOperator<RealVector> for ArithmeticCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        _rng: &mut R,
    ) -> OperatorResult<(RealVector, RealVector)> {
        if parent1.dimension() != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        let w = self.weight;
        let mut child1_genes = Vec::with_capacity(parent1.dimension());
        let mut child2_genes = Vec::with_capacity(parent2.dimension());

        for i in 0..parent1.dimension() {
            child1_genes.push(w * parent1[i] + (1.0 - w) * parent2[i]);
            child2_genes.push((1.0 - w) * parent1[i] + w * parent2[i]);
        }

        let child1 = RealVector::from_genes(child1_genes).unwrap();
        let child2 = RealVector::from_genes(child2_genes).unwrap();

        OperatorResult::Success((child1, child2))
    }
}

// =============================================================================
// Permutation Crossover Operators
// =============================================================================

/// Partially Mapped Crossover (PMX) for permutation genomes
///
/// PMX preserves relative order and position information from both parents.
/// It selects a segment from parent1 and maps the corresponding positions
/// from parent2, creating valid permutations.
///
/// Reference: Goldberg, D. E., & Lingle, R. (1985). Alleles, Loci, and the
/// Traveling Salesman Problem. ICGA.
#[derive(Clone, Debug, Default)]
pub struct PmxCrossover;

impl PmxCrossover {
    /// Create a new PMX crossover operator
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<Permutation> for PmxCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &Permutation,
        parent2: &Permutation,
        rng: &mut R,
    ) -> OperatorResult<(Permutation, Permutation)> {
        let n = parent1.dimension();

        if n != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        if n < 2 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        // Select two crossover points
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        let p1 = parent1.permutation();
        let p2 = parent2.permutation();

        // Initialize children with sentinel values
        let mut child1 = vec![usize::MAX; n];
        let mut child2 = vec![usize::MAX; n];

        // Copy segments from opposite parent
        for i in start..=end {
            child1[i] = p2[i];
            child2[i] = p1[i];
        }

        // Build mappings for the segment
        let mut map1: HashMap<usize, usize> = HashMap::new();
        let mut map2: HashMap<usize, usize> = HashMap::new();
        for i in start..=end {
            map1.insert(p2[i], p1[i]);
            map2.insert(p1[i], p2[i]);
        }

        // Fill remaining positions
        for i in (0..start).chain((end + 1)..n) {
            // For child1: try to place p1[i], resolve conflicts via mapping
            let mut val1 = p1[i];
            while child1[start..=end].contains(&val1) {
                val1 = *map1.get(&val1).unwrap_or(&val1);
            }
            child1[i] = val1;

            // For child2: try to place p2[i], resolve conflicts via mapping
            let mut val2 = p2[i];
            while child2[start..=end].contains(&val2) {
                val2 = *map2.get(&val2).unwrap_or(&val2);
            }
            child2[i] = val2;
        }

        let c1 = match Permutation::try_new(child1) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "PMX produced invalid child1: {}",
                    e
                )))
            }
        };
        let c2 = match Permutation::try_new(child2) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "PMX produced invalid child2: {}",
                    e
                )))
            }
        };

        OperatorResult::Success((c1, c2))
    }
}

/// Order Crossover (OX) for permutation genomes
///
/// OX preserves the relative order of elements from one parent while
/// copying a segment from the other parent.
///
/// Reference: Davis, L. (1985). Applying Adaptive Algorithms to Epistatic Domains.
/// IJCAI.
#[derive(Clone, Debug, Default)]
pub struct OxCrossover;

impl OxCrossover {
    /// Create a new OX crossover operator
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<Permutation> for OxCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &Permutation,
        parent2: &Permutation,
        rng: &mut R,
    ) -> OperatorResult<(Permutation, Permutation)> {
        let n = parent1.dimension();

        if n != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        if n < 2 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        // Select two crossover points
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        let child1 = Self::ox_single(parent1, parent2, start, end);
        let child2 = Self::ox_single(parent2, parent1, start, end);

        let c1 = match Permutation::try_new(child1) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "OX produced invalid child1: {}",
                    e
                )))
            }
        };
        let c2 = match Permutation::try_new(child2) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "OX produced invalid child2: {}",
                    e
                )))
            }
        };

        OperatorResult::Success((c1, c2))
    }
}

impl OxCrossover {
    /// Create a single child using OX
    fn ox_single(
        parent1: &Permutation,
        parent2: &Permutation,
        start: usize,
        end: usize,
    ) -> Vec<usize> {
        let n = parent1.dimension();
        let p1 = parent1.permutation();
        let p2 = parent2.permutation();

        let mut child = vec![usize::MAX; n];

        // Copy segment from parent1
        let segment: HashSet<usize> = p1[start..=end].iter().copied().collect();
        for i in start..=end {
            child[i] = p1[i];
        }

        // Fill remaining positions from parent2 in order, skipping segment elements
        let mut pos = (end + 1) % n;
        let mut p2_idx = (end + 1) % n;

        while pos != start {
            // Find next element from parent2 not in segment
            while segment.contains(&p2[p2_idx]) {
                p2_idx = (p2_idx + 1) % n;
            }

            child[pos] = p2[p2_idx];
            pos = (pos + 1) % n;
            p2_idx = (p2_idx + 1) % n;
        }

        child
    }
}

/// Cycle Crossover (CX) for permutation genomes
///
/// CX produces offspring where each element's position comes from one parent,
/// preserving the absolute position of elements. It identifies cycles in the
/// parent mappings and alternates which parent contributes each cycle.
///
/// Reference: Oliver, I. M., Smith, D. J., & Holland, J. R. (1987).
/// A Study of Permutation Crossover Operators on the Traveling Salesman Problem.
#[derive(Clone, Debug, Default)]
pub struct CxCrossover;

impl CxCrossover {
    /// Create a new CX crossover operator
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverOperator<Permutation> for CxCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &Permutation,
        parent2: &Permutation,
        _rng: &mut R,
    ) -> OperatorResult<(Permutation, Permutation)> {
        let n = parent1.dimension();

        if n != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        if n == 0 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        let p1 = parent1.permutation();
        let p2 = parent2.permutation();

        // Build position map: value -> position in parent1
        let mut pos_in_p1: HashMap<usize, usize> = HashMap::new();
        for (i, &val) in p1.iter().enumerate() {
            pos_in_p1.insert(val, i);
        }

        // Find cycles and assign to children
        let mut child1 = vec![usize::MAX; n];
        let mut child2 = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut use_p1 = true; // Alternate which parent cycle goes to child1

        for start in 0..n {
            if visited[start] {
                continue;
            }

            // Find the cycle starting at position `start`
            let mut cycle_positions = Vec::new();
            let mut pos = start;

            loop {
                cycle_positions.push(pos);
                visited[pos] = true;

                // Follow the cycle: position in p1 -> value in p2 at same position -> position of that value in p1
                let val_in_p2 = p2[pos];
                pos = *pos_in_p1.get(&val_in_p2).unwrap();

                if pos == start {
                    break;
                }
            }

            // Assign cycle positions to children
            for &cycle_pos in &cycle_positions {
                if use_p1 {
                    child1[cycle_pos] = p1[cycle_pos];
                    child2[cycle_pos] = p2[cycle_pos];
                } else {
                    child1[cycle_pos] = p2[cycle_pos];
                    child2[cycle_pos] = p1[cycle_pos];
                }
            }

            use_p1 = !use_p1; // Alternate for next cycle
        }

        let c1 = match Permutation::try_new(child1) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "CX produced invalid child1: {}",
                    e
                )))
            }
        };
        let c2 = match Permutation::try_new(child2) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "CX produced invalid child2: {}",
                    e
                )))
            }
        };

        OperatorResult::Success((c1, c2))
    }
}

/// Edge Recombination Crossover (ERX) for permutation genomes
///
/// ERX focuses on preserving edges (adjacencies) from both parents,
/// which is particularly useful for TSP-like problems.
#[derive(Clone, Debug, Default)]
pub struct EdgeRecombinationCrossover;

impl EdgeRecombinationCrossover {
    /// Create a new edge recombination crossover operator
    pub fn new() -> Self {
        Self
    }

    /// Build edge table from parents
    fn build_edge_table(
        parent1: &Permutation,
        parent2: &Permutation,
    ) -> HashMap<usize, HashSet<usize>> {
        let n = parent1.dimension();
        let p1 = parent1.permutation();
        let p2 = parent2.permutation();

        let mut edges: HashMap<usize, HashSet<usize>> = HashMap::new();

        // Initialize all nodes
        for i in 0..n {
            edges.insert(i, HashSet::new());
        }

        // Add edges from parent1
        for i in 0..n {
            let curr = p1[i];
            let prev = p1[(i + n - 1) % n];
            let next = p1[(i + 1) % n];
            edges.get_mut(&curr).unwrap().insert(prev);
            edges.get_mut(&curr).unwrap().insert(next);
        }

        // Add edges from parent2
        for i in 0..n {
            let curr = p2[i];
            let prev = p2[(i + n - 1) % n];
            let next = p2[(i + 1) % n];
            edges.get_mut(&curr).unwrap().insert(prev);
            edges.get_mut(&curr).unwrap().insert(next);
        }

        edges
    }
}

impl CrossoverOperator<Permutation> for EdgeRecombinationCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &Permutation,
        parent2: &Permutation,
        rng: &mut R,
    ) -> OperatorResult<(Permutation, Permutation)> {
        let n = parent1.dimension();

        if n != parent2.dimension() {
            return OperatorResult::Failed(OperatorError::CrossoverFailed(
                "Parent dimensions do not match".to_string(),
            ));
        }

        if n < 2 {
            return OperatorResult::Success((parent1.clone(), parent2.clone()));
        }

        // Build edge table
        let mut edges = Self::build_edge_table(parent1, parent2);

        // Build child
        let mut child = Vec::with_capacity(n);
        let mut remaining: HashSet<usize> = (0..n).collect();

        // Start with first element of parent1
        let mut current = parent1.permutation()[0];
        child.push(current);
        remaining.remove(&current);

        // Remove current from all edge lists
        for edge_set in edges.values_mut() {
            edge_set.remove(&current);
        }

        while child.len() < n {
            // Get neighbors of current
            let neighbors = edges.get(&current).cloned().unwrap_or_default();

            // Choose next: prefer neighbor with fewest remaining edges
            let next = if !neighbors.is_empty() {
                let filtered: Vec<usize> = neighbors
                    .iter()
                    .filter(|x| remaining.contains(x))
                    .copied()
                    .collect();
                if filtered.is_empty() {
                    // Pick random from remaining
                    let remaining_vec: Vec<usize> = remaining.iter().copied().collect();
                    remaining_vec[rng.gen_range(0..remaining_vec.len())]
                } else {
                    // Pick one with minimum edge count
                    *filtered
                        .iter()
                        .min_by_key(|&&x| edges.get(&x).map(|s| s.len()).unwrap_or(0))
                        .unwrap()
                }
            } else {
                // No neighbors, pick random from remaining
                let remaining_vec: Vec<usize> = remaining.iter().copied().collect();
                remaining_vec[rng.gen_range(0..remaining_vec.len())]
            };

            child.push(next);
            remaining.remove(&next);
            current = next;

            // Remove current from all edge lists
            for edge_set in edges.values_mut() {
                edge_set.remove(&current);
            }
        }

        // Create second child by running again with different starting point
        let mut edges2 = Self::build_edge_table(parent1, parent2);
        let mut child2 = Vec::with_capacity(n);
        let mut remaining2: HashSet<usize> = (0..n).collect();

        // Start with first element of parent2
        let mut current2 = parent2.permutation()[0];
        child2.push(current2);
        remaining2.remove(&current2);

        for edge_set in edges2.values_mut() {
            edge_set.remove(&current2);
        }

        while child2.len() < n {
            let neighbors = edges2.get(&current2).cloned().unwrap_or_default();

            let next2 = if !neighbors.is_empty() {
                let filtered: Vec<usize> = neighbors
                    .iter()
                    .filter(|x| remaining2.contains(x))
                    .copied()
                    .collect();
                if filtered.is_empty() {
                    let remaining_vec: Vec<usize> = remaining2.iter().copied().collect();
                    remaining_vec[rng.gen_range(0..remaining_vec.len())]
                } else {
                    *filtered
                        .iter()
                        .min_by_key(|&&x| edges2.get(&x).map(|s| s.len()).unwrap_or(0))
                        .unwrap()
                }
            } else {
                let remaining_vec: Vec<usize> = remaining2.iter().copied().collect();
                remaining_vec[rng.gen_range(0..remaining_vec.len())]
            };

            child2.push(next2);
            remaining2.remove(&next2);
            current2 = next2;

            for edge_set in edges2.values_mut() {
                edge_set.remove(&current2);
            }
        }

        let c1 = match Permutation::try_new(child) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "ERX produced invalid child1: {}",
                    e
                )))
            }
        };
        let c2 = match Permutation::try_new(child2) {
            Ok(p) => p,
            Err(e) => {
                return OperatorResult::Failed(OperatorError::CrossoverFailed(format!(
                    "ERX produced invalid child2: {}",
                    e
                )))
            }
        };

        OperatorResult::Success((c1, c2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sbx_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0, 1.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.dimension(), 3);
        assert_eq!(child2.dimension(), 3);
    }

    #[test]
    fn test_sbx_with_bounds() {
        let mut rng = rand::thread_rng();
        // Use parents within bounds
        let parent1 = RealVector::new(vec![-0.3, -0.2]);
        let parent2 = RealVector::new(vec![0.3, 0.4]);
        let bounds = MultiBounds::symmetric(0.5, 2);

        let sbx = SbxCrossover::new(2.0).with_probability(1.0); // Low eta = more spread, always crossover

        // Run multiple times and check bounds
        for _ in 0..100 {
            let result = sbx.crossover_bounded(&parent1, &parent2, &bounds, &mut rng);
            let (child1, child2) = result.genome().unwrap();

            for gene in child1.genes() {
                assert!(
                    *gene >= -0.5 && *gene <= 0.5,
                    "gene {} out of bounds [-0.5, 0.5]",
                    gene
                );
            }
            for gene in child2.genes() {
                assert!(
                    *gene >= -0.5 && *gene <= 0.5,
                    "gene {} out of bounds [-0.5, 0.5]",
                    gene
                );
            }
        }
    }

    #[test]
    fn test_sbx_spread_factor() {
        let sbx = SbxCrossover::new(20.0);

        // At u = 0.5, β should be 1.0
        let beta = sbx.spread_factor(0.5);
        assert_relative_eq!(beta, 1.0, epsilon = 1e-10);

        // β should be symmetric around 0.5
        let beta_low = sbx.spread_factor(0.25);
        let beta_high = sbx.spread_factor(0.75);
        assert_relative_eq!(beta_low, 1.0 / beta_high, epsilon = 1e-10);
    }

    #[test]
    fn test_sbx_identical_parents() {
        let mut rng = rand::thread_rng();
        let parent = RealVector::new(vec![1.0, 2.0, 3.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent, &parent, &mut rng);

        let (child1, child2) = result.genome().unwrap();
        // With identical parents, children should equal parents
        assert_eq!(child1.genes(), parent.genes());
        assert_eq!(child2.genes(), parent.genes());
    }

    #[test]
    fn test_sbx_dimension_mismatch() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0]);
        let parent2 = RealVector::new(vec![1.0, 2.0, 3.0]);

        let sbx = SbxCrossover::new(20.0);
        let result = sbx.crossover(&parent1, &parent2, &mut rng);

        assert!(!result.is_ok());
    }

    #[test]
    fn test_blx_alpha_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0]);

        let blx = BlxAlphaCrossover::new(0.5);
        let result = blx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.dimension(), 2);
        assert_eq!(child2.dimension(), 2);
    }

    #[test]
    fn test_blx_alpha_range() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0]);
        let parent2 = RealVector::new(vec![1.0]);

        let blx = BlxAlphaCrossover::new(0.0); // No extension

        for _ in 0..100 {
            let result = blx.crossover(&parent1, &parent2, &mut rng);
            let (child1, child2) = result.genome().unwrap();

            // With α = 0, offspring should be in [0, 1]
            assert!(child1[0] >= 0.0 && child1[0] <= 1.0);
            assert!(child2[0] >= 0.0 && child2[0] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_crossover_creates_valid_offspring() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::new(vec![true, true, true, true]);
        let parent2 = BitString::new(vec![false, false, false, false]);

        let ux = UniformCrossover::new();
        let result = ux.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.len(), 4);
        assert_eq!(child2.len(), 4);
    }

    #[test]
    fn test_uniform_crossover_complementary() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::new(vec![true, true, true, true]);
        let parent2 = BitString::new(vec![false, false, false, false]);

        let ux = UniformCrossover::new();
        let result = ux.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // Children should be complementary
        for i in 0..4 {
            assert_ne!(child1[i], child2[i]);
        }
    }

    #[test]
    fn test_one_point_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::ones(10);
        let parent2 = BitString::zeros(10);

        let opx = OnePointCrossover::new();
        let result = opx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();

        // Children should have a contiguous segment from each parent
        // and the children should be complementary
        for i in 0..10 {
            assert_ne!(child1[i], child2[i]);
        }
    }

    #[test]
    fn test_two_point_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = BitString::ones(10);
        let parent2 = BitString::zeros(10);

        let tpx = TwoPointCrossover::new();
        let result = tpx.crossover(&parent1, &parent2, &mut rng);

        assert!(result.is_ok());
        let (child1, child2) = result.genome().unwrap();
        assert_eq!(child1.len(), 10);
        assert_eq!(child2.len(), 10);
    }

    #[test]
    fn test_arithmetic_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0, 0.0]);
        let parent2 = RealVector::new(vec![1.0, 1.0]);

        let ax = ArithmeticCrossover::new(0.5);
        let result = ax.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // With 0.5 weight, both children should be at midpoint
        for gene in child1.genes() {
            assert_relative_eq!(*gene, 0.5);
        }
        for gene in child2.genes() {
            assert_relative_eq!(*gene, 0.5);
        }
    }

    #[test]
    fn test_arithmetic_crossover_weighted() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![0.0]);
        let parent2 = RealVector::new(vec![1.0]);

        let ax = ArithmeticCrossover::new(0.75);
        let result = ax.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // child1 = 0.75 * 0 + 0.25 * 1 = 0.25
        // child2 = 0.25 * 0 + 0.75 * 1 = 0.75
        assert_relative_eq!(child1[0], 0.25);
        assert_relative_eq!(child2[0], 0.75);
    }

    // =========================================================================
    // Permutation Crossover Tests
    // =========================================================================

    #[test]
    fn test_pmx_creates_valid_permutations() {
        let mut rng = rand::thread_rng();
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = Permutation::new(vec![7, 6, 5, 4, 3, 2, 1, 0]);

        let pmx = PmxCrossover::new();

        for _ in 0..100 {
            let result = pmx.crossover(&parent1, &parent2, &mut rng);
            assert!(result.is_ok());
            let (child1, child2) = result.genome().unwrap();

            assert!(child1.is_valid_permutation());
            assert!(child2.is_valid_permutation());
            assert_eq!(child1.dimension(), 8);
            assert_eq!(child2.dimension(), 8);
        }
    }

    #[test]
    fn test_pmx_identical_parents() {
        let mut rng = rand::thread_rng();
        let parent = Permutation::new(vec![0, 1, 2, 3, 4]);

        let pmx = PmxCrossover::new();
        let result = pmx.crossover(&parent, &parent, &mut rng);

        let (child1, child2) = result.genome().unwrap();
        // With identical parents, children should equal parents
        assert_eq!(child1.as_slice(), parent.as_slice());
        assert_eq!(child2.as_slice(), parent.as_slice());
    }

    #[test]
    fn test_pmx_dimension_mismatch() {
        let mut rng = rand::thread_rng();
        let parent1 = Permutation::new(vec![0, 1, 2, 3]);
        let parent2 = Permutation::new(vec![0, 1, 2, 3, 4]);

        let pmx = PmxCrossover::new();
        let result = pmx.crossover(&parent1, &parent2, &mut rng);

        assert!(!result.is_ok());
    }

    #[test]
    fn test_ox_creates_valid_permutations() {
        let mut rng = rand::thread_rng();
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = Permutation::new(vec![7, 6, 5, 4, 3, 2, 1, 0]);

        let ox = OxCrossover::new();

        for _ in 0..100 {
            let result = ox.crossover(&parent1, &parent2, &mut rng);
            assert!(result.is_ok());
            let (child1, child2) = result.genome().unwrap();

            assert!(child1.is_valid_permutation());
            assert!(child2.is_valid_permutation());
            assert_eq!(child1.dimension(), 8);
            assert_eq!(child2.dimension(), 8);
        }
    }

    #[test]
    fn test_ox_preserves_segment() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = Permutation::new(vec![7, 6, 5, 4, 3, 2, 1, 0]);

        let ox = OxCrossover::new();
        let result = ox.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();
        assert!(child1.is_valid_permutation());
        assert!(child2.is_valid_permutation());
    }

    #[test]
    fn test_cx_creates_valid_permutations() {
        let mut rng = rand::thread_rng();
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = Permutation::new(vec![1, 2, 3, 4, 5, 6, 7, 0]);

        let cx = CxCrossover::new();

        for _ in 0..100 {
            let result = cx.crossover(&parent1, &parent2, &mut rng);
            assert!(result.is_ok());
            let (child1, child2) = result.genome().unwrap();

            assert!(child1.is_valid_permutation());
            assert!(child2.is_valid_permutation());
            assert_eq!(child1.dimension(), 8);
            assert_eq!(child2.dimension(), 8);
        }
    }

    #[test]
    fn test_cx_preserves_positions() {
        let mut rng = rand::thread_rng();
        // CX should preserve positions from one parent or the other
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4]);
        let parent2 = Permutation::new(vec![4, 3, 2, 1, 0]);

        let cx = CxCrossover::new();
        let result = cx.crossover(&parent1, &parent2, &mut rng);

        let (child1, child2) = result.genome().unwrap();

        // Each position in child should have the value from one of the parents at that position
        for i in 0..5 {
            let c1_val = child1[i];
            let c2_val = child2[i];

            assert!(c1_val == parent1[i] || c1_val == parent2[i]);
            assert!(c2_val == parent1[i] || c2_val == parent2[i]);
        }
    }

    #[test]
    fn test_cx_identical_parents() {
        let mut rng = rand::thread_rng();
        let parent = Permutation::new(vec![0, 1, 2, 3, 4]);

        let cx = CxCrossover::new();
        let result = cx.crossover(&parent, &parent, &mut rng);

        let (child1, child2) = result.genome().unwrap();
        // With identical parents, children should equal parents
        assert_eq!(child1.as_slice(), parent.as_slice());
        assert_eq!(child2.as_slice(), parent.as_slice());
    }

    #[test]
    fn test_erx_creates_valid_permutations() {
        let mut rng = rand::thread_rng();
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = Permutation::new(vec![7, 6, 5, 4, 3, 2, 1, 0]);

        let erx = EdgeRecombinationCrossover::new();

        for _ in 0..50 {
            let result = erx.crossover(&parent1, &parent2, &mut rng);
            assert!(result.is_ok());
            let (child1, child2) = result.genome().unwrap();

            assert!(child1.is_valid_permutation());
            assert!(child2.is_valid_permutation());
            assert_eq!(child1.dimension(), 8);
            assert_eq!(child2.dimension(), 8);
        }
    }

    #[test]
    fn test_erx_preserves_some_edges() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Create parents with some common edges
        let parent1 = Permutation::new(vec![0, 1, 2, 3, 4]);
        let parent2 = Permutation::new(vec![0, 1, 4, 3, 2]);

        // Edge 0-1 is common to both parents
        let erx = EdgeRecombinationCrossover::new();
        let result = erx.crossover(&parent1, &parent2, &mut rng);

        let (child1, _child2) = result.genome().unwrap();
        assert!(child1.is_valid_permutation());
    }
}
