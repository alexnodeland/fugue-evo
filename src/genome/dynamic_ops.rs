//! Variation operators for [`DynamicRealVector`].
//!
//! [`DynamicRealVector`] is a variable-length real-valued genome, but the
//! fixed-length real operators in `crate::operators` are written concretely for
//! [`RealVector`](crate::genome::real_vector::RealVector) and therefore cannot
//! vary a genome's length. This module provides the missing length-aware
//! operators so a population of `DynamicRealVector` genomes can actually be
//! evolved:
//!
//! - [`cut_and_splice`]: a crossover that recombines two variable-length parents
//!   and yields two children whose lengths may differ from either parent.
//! - [`DynamicGaussianMutation`]: a mutation that combines per-gene Gaussian
//!   perturbation with length-changing insert/delete.
//!
//! These live under `crate::genome` (not `crate::operators`) so they stay
//! next to the genome type they are specialized for, and they operate directly
//! on [`DynamicRealVector`] rather than through the operator traits.

use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::error::GenomeError;
use crate::genome::bounds::MultiBounds;
use crate::genome::dynamic_real_vector::DynamicRealVector;
use crate::genome::traits::{EvolutionaryGenome, RealValuedGenome};

/// Clamp a gene vector's length into `[min_length, max_length]`.
///
/// Overlong vectors are truncated; overshort vectors are padded by repeating
/// their last gene (or `0.0` when empty), so padding values stay within the
/// value range of the existing genes.
fn repair_length(mut genes: Vec<f64>, min_length: usize, max_length: usize) -> Vec<f64> {
    if genes.len() > max_length {
        genes.truncate(max_length);
    }
    while genes.len() < min_length {
        let fill = genes.last().copied().unwrap_or(0.0);
        genes.push(fill);
    }
    genes
}

/// Cut-and-splice crossover for variable-length real vectors.
///
/// Each parent is cut at an independent, uniformly chosen point; the head of one
/// parent is spliced with the tail of the other (and vice versa), producing two
/// children whose lengths may differ from both parents. Children are then
/// repaired into the shared length window `[min_length, max_length]`.
///
/// The children's length constraints are the intersection of the parents'
/// constraints (`max(min_length)` .. `min(max_length)`); if that window is empty
/// this returns [`GenomeError::InvalidStructure`].
pub fn cut_and_splice<R: Rng>(
    parent1: &DynamicRealVector,
    parent2: &DynamicRealVector,
    rng: &mut R,
) -> Result<(DynamicRealVector, DynamicRealVector), GenomeError> {
    let min_length = parent1.min_length().max(parent2.min_length());
    let max_length = parent1.max_length().min(parent2.max_length());
    if min_length > max_length {
        return Err(GenomeError::InvalidStructure(format!(
            "cut_and_splice: parents have incompatible length constraints \
             (combined min_length {min_length} > max_length {max_length})"
        )));
    }

    let g1 = parent1.genes();
    let g2 = parent2.genes();
    // Cut points range over `0..=len` so a head or tail may be empty.
    let cut1 = rng.gen_range(0..=g1.len());
    let cut2 = rng.gen_range(0..=g2.len());

    let mut child1: Vec<f64> = g1[..cut1].to_vec();
    child1.extend_from_slice(&g2[cut2..]);
    let mut child2: Vec<f64> = g2[..cut2].to_vec();
    child2.extend_from_slice(&g1[cut1..]);

    let child1 = repair_length(child1, min_length, max_length);
    let child2 = repair_length(child2, min_length, max_length);

    Ok((
        DynamicRealVector::new(child1, min_length, max_length)?,
        DynamicRealVector::new(child2, min_length, max_length)?,
    ))
}

/// Length-aware mutation for [`DynamicRealVector`].
///
/// Combines two independent effects, applied in this order:
/// 1. **Gaussian perturbation** — each gene is, with probability
///    `gene_mutation_prob`, perturbed by a sample from `N(0, sigma^2)` and then
///    clamped back into the supplied bounds.
/// 2. **Length change** — with probability `grow_prob` a new gene (sampled
///    within bounds) is inserted at a random position, and with probability
///    `shrink_prob` a random gene is removed. Both changes respect the genome's
///    own `[min_length, max_length]` window via
///    [`can_grow`](DynamicRealVector::can_grow) /
///    [`can_shrink`](DynamicRealVector::can_shrink), so the length always stays
///    in range.
#[derive(Clone, Copy, Debug)]
pub struct DynamicGaussianMutation {
    /// Standard deviation of the per-gene Gaussian perturbation.
    pub sigma: f64,
    /// Per-gene probability of applying the Gaussian perturbation.
    pub gene_mutation_prob: f64,
    /// Probability of inserting a new gene (subject to `can_grow`).
    pub grow_prob: f64,
    /// Probability of removing a gene (subject to `can_shrink`).
    pub shrink_prob: f64,
}

impl DynamicGaussianMutation {
    /// Create a new mutation operator.
    ///
    /// # Panics
    /// Panics if any probability is outside `[0, 1]` or if `sigma` is negative.
    pub fn new(sigma: f64, gene_mutation_prob: f64, grow_prob: f64, shrink_prob: f64) -> Self {
        assert!(sigma >= 0.0, "sigma must be non-negative");
        for (name, p) in [
            ("gene_mutation_prob", gene_mutation_prob),
            ("grow_prob", grow_prob),
            ("shrink_prob", shrink_prob),
        ] {
            assert!(
                (0.0..=1.0).contains(&p),
                "{name} must be in [0, 1], got {p}"
            );
        }
        Self {
            sigma,
            gene_mutation_prob,
            grow_prob,
            shrink_prob,
        }
    }

    /// Apply the mutation in place.
    pub fn mutate<R: Rng>(
        &self,
        genome: &mut DynamicRealVector,
        bounds: &MultiBounds,
        rng: &mut R,
    ) {
        // 1. Per-gene Gaussian perturbation.
        if self.sigma > 0.0 && self.gene_mutation_prob > 0.0 {
            let normal = Normal::new(0.0, self.sigma).expect("sigma > 0");
            for gene in genome.genes_mut().iter_mut() {
                if rng.gen::<f64>() < self.gene_mutation_prob {
                    *gene += normal.sample(rng);
                }
            }
            genome.apply_bounds(bounds);
        }

        // 2a. Length-increasing insert.
        if self.grow_prob > 0.0 && rng.gen::<f64>() < self.grow_prob && genome.can_grow() {
            let len = genome.dimension();
            let index = rng.gen_range(0..=len);
            let value = match bounds.get(index).or_else(|| bounds.get(0)) {
                Some(b) => rng.gen_range(b.min..=b.max),
                None => 0.0,
            };
            // `can_grow` guarantees room; ignore the (impossible) error path.
            let _ = genome.insert(index, value);
        }

        // 2b. Length-decreasing delete.
        if self.shrink_prob > 0.0 && rng.gen::<f64>() < self.shrink_prob && genome.can_shrink() {
            let len = genome.dimension();
            if len > 0 {
                let index = rng.gen_range(0..len);
                let _ = genome.remove(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cut_and_splice_lengths_and_values_within_range(
            g1 in prop::collection::vec(-5.0..5.0f64, 1..12),
            g2 in prop::collection::vec(-5.0..5.0f64, 1..12),
        ) {
            // Property (EV-57): children lengths stay within [min, max] and
            // values stay within bounds (they come from the parents, which are
            // within [-5, 5], or from in-range padding).
            let min_length = 1;
            let max_length = 16;
            let p1 = DynamicRealVector::new(g1, min_length, max_length).unwrap();
            let p2 = DynamicRealVector::new(g2, min_length, max_length).unwrap();
            let mut rng = rand::thread_rng();

            let (c1, c2) = cut_and_splice(&p1, &p2, &mut rng).unwrap();

            prop_assert!(c1.dimension() >= min_length && c1.dimension() <= max_length);
            prop_assert!(c2.dimension() >= min_length && c2.dimension() <= max_length);
            for &v in c1.genes() {
                prop_assert!((-5.0..=5.0).contains(&v));
            }
            for &v in c2.genes() {
                prop_assert!((-5.0..=5.0).contains(&v));
            }
        }

        #[test]
        fn mutation_preserves_length_window_and_bounds(
            genes in prop::collection::vec(-4.0..4.0f64, 2..10),
        ) {
            // Property (EV-57): after any number of mutations the length stays in
            // [min, max] and all values stay within the configured bounds.
            let min_length = 1;
            let max_length = 12;
            let mut genome = DynamicRealVector::new(genes, min_length, max_length).unwrap();
            let bounds = MultiBounds::symmetric(5.0, max_length);
            let op = DynamicGaussianMutation::new(0.75, 0.5, 0.5, 0.5);
            let mut rng = rand::thread_rng();

            for _ in 0..64 {
                op.mutate(&mut genome, &bounds, &mut rng);
                prop_assert!(
                    genome.dimension() >= min_length && genome.dimension() <= max_length
                );
                for &v in genome.genes() {
                    prop_assert!((-5.0..=5.0).contains(&v));
                }
            }
        }
    }

    #[test]
    fn cut_and_splice_incompatible_constraints_errors() {
        // Combined window is max(2,1)=2 .. min(4,1)=1, which is empty.
        let p1 = DynamicRealVector::new(vec![1.0, 2.0, 3.0], 2, 4).unwrap();
        let p2 = DynamicRealVector::new(vec![9.0], 1, 1).unwrap();
        let mut rng = rand::thread_rng();
        assert!(cut_and_splice(&p1, &p2, &mut rng).is_err());
    }

    #[test]
    fn cut_and_splice_is_usable_for_evolution() {
        // Smoke test: two parents recombine into two valid children.
        let p1 = DynamicRealVector::new(vec![1.0, 2.0, 3.0, 4.0], 1, 8).unwrap();
        let p2 = DynamicRealVector::new(vec![-1.0, -2.0], 1, 8).unwrap();
        let mut rng = rand::thread_rng();
        let (c1, c2) = cut_and_splice(&p1, &p2, &mut rng).unwrap();
        assert!(c1.dimension() >= 1 && c1.dimension() <= 8);
        assert!(c2.dimension() >= 1 && c2.dimension() <= 8);
    }

    #[test]
    fn mutation_can_grow_and_shrink() {
        // With grow-only settings the genome should be able to reach max length;
        // with shrink-only settings it should reach min length.
        let bounds = MultiBounds::symmetric(5.0, 8);
        let mut rng = rand::thread_rng();

        let grow = DynamicGaussianMutation::new(0.1, 0.0, 1.0, 0.0);
        let mut g = DynamicRealVector::new(vec![0.0, 0.0], 2, 8).unwrap();
        for _ in 0..50 {
            grow.mutate(&mut g, &bounds, &mut rng);
        }
        assert_eq!(g.dimension(), 8);

        let shrink = DynamicGaussianMutation::new(0.1, 0.0, 0.0, 1.0);
        let mut s = DynamicRealVector::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 8).unwrap();
        for _ in 0..50 {
            shrink.mutate(&mut s, &bounds, &mut rng);
        }
        assert_eq!(s.dimension(), 2);
    }
}
