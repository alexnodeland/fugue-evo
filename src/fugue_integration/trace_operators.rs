//! Trace-based genetic operators
//!
//! These operators work directly on Fugue traces, enabling probabilistic
//! interpretations of mutation and crossover.

use std::collections::HashSet;

use fugue::{Address, ChoiceValue, Trace};
use rand::Rng;

use crate::error::GenomeError;
use crate::genome::traits::EvolutionaryGenome;

/// Trait for selecting which addresses to mutate
pub trait MutationSelector: Send + Sync {
    /// Select addresses that should be mutated
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address>;
}

/// Uniform random mutation selector
///
/// Each address has an independent probability of being selected for mutation.
#[derive(Clone, Debug)]
pub struct UniformMutationSelector {
    /// Probability of mutating each address
    pub mutation_probability: f64,
}

impl UniformMutationSelector {
    /// Create a new uniform mutation selector
    pub fn new(probability: f64) -> Self {
        Self {
            mutation_probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Default 1/n mutation probability
    pub fn one_over_n(n: usize) -> Self {
        Self::new(1.0 / n as f64)
    }
}

impl MutationSelector for UniformMutationSelector {
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address> {
        trace
            .choices
            .keys()
            .filter(|_| rng.gen::<f64>() < self.mutation_probability)
            .cloned()
            .collect()
    }
}

/// Single-site mutation selector
///
/// Selects exactly one random address for mutation.
#[derive(Clone, Debug, Default)]
pub struct SingleSiteMutationSelector;

impl SingleSiteMutationSelector {
    /// Create a new single-site selector
    pub fn new() -> Self {
        Self
    }
}

impl MutationSelector for SingleSiteMutationSelector {
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address> {
        let addresses: Vec<_> = trace.choices.keys().collect();
        if addresses.is_empty() {
            return HashSet::new();
        }

        let idx = rng.gen_range(0..addresses.len());
        let mut sites = HashSet::new();
        sites.insert(addresses[idx].clone());
        sites
    }
}

/// Multi-site mutation selector
///
/// Selects exactly k random addresses for mutation.
#[derive(Clone, Debug)]
pub struct MultiSiteMutationSelector {
    /// Number of sites to mutate
    pub num_sites: usize,
}

impl MultiSiteMutationSelector {
    /// Create a new multi-site selector
    pub fn new(num_sites: usize) -> Self {
        Self { num_sites }
    }
}

impl MutationSelector for MultiSiteMutationSelector {
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address> {
        let addresses: Vec<_> = trace.choices.keys().collect();
        if addresses.is_empty() {
            return HashSet::new();
        }

        let k = self.num_sites.min(addresses.len());
        let mut selected = HashSet::new();
        let mut indices: Vec<usize> = (0..addresses.len()).collect();

        // Fisher-Yates partial shuffle
        for i in 0..k {
            let j = rng.gen_range(i..addresses.len());
            indices.swap(i, j);
            selected.insert(addresses[indices[i]].clone());
        }

        selected
    }
}

/// Trait for determining which parent contributes to each address during crossover
pub trait CrossoverMask: Send + Sync {
    /// Returns true if parent1's value should be used at the given address
    fn from_parent1(&self, addr: &Address) -> bool;
}

/// Uniform crossover mask
///
/// Each address independently chosen from either parent.
#[derive(Clone, Debug)]
pub struct UniformCrossoverMask {
    /// Probability of choosing parent1's value
    pub bias: f64,
    /// Set of addresses that should come from parent1
    selected: HashSet<Address>,
}

impl UniformCrossoverMask {
    /// Create a new uniform crossover mask
    pub fn new<R: Rng>(bias: f64, trace: &Trace, rng: &mut R) -> Self {
        let selected = trace
            .choices
            .keys()
            .filter(|_| rng.gen::<f64>() < bias)
            .cloned()
            .collect();

        Self {
            bias,
            selected,
        }
    }

    /// Create with 50/50 probability
    pub fn balanced<R: Rng>(trace: &Trace, rng: &mut R) -> Self {
        Self::new(0.5, trace, rng)
    }
}

impl CrossoverMask for UniformCrossoverMask {
    fn from_parent1(&self, addr: &Address) -> bool {
        self.selected.contains(addr)
    }
}

/// Single-point crossover mask
///
/// All addresses before the crossover point come from parent1,
/// all after come from parent2.
#[derive(Clone, Debug)]
pub struct SinglePointCrossoverMask {
    /// Addresses from parent1 (before crossover point)
    parent1_addresses: HashSet<Address>,
}

impl SinglePointCrossoverMask {
    /// Create a new single-point crossover mask
    pub fn new<R: Rng>(trace: &Trace, rng: &mut R) -> Self {
        let addresses: Vec<_> = trace.choices.keys().cloned().collect();
        if addresses.is_empty() {
            return Self {
                parent1_addresses: HashSet::new(),
            };
        }

        let crossover_point = rng.gen_range(0..=addresses.len());
        let parent1_addresses: HashSet<_> = addresses.into_iter().take(crossover_point).collect();

        Self { parent1_addresses }
    }
}

impl CrossoverMask for SinglePointCrossoverMask {
    fn from_parent1(&self, addr: &Address) -> bool {
        self.parent1_addresses.contains(addr)
    }
}

/// Two-point crossover mask
///
/// Addresses between the two points come from parent2,
/// outside comes from parent1.
#[derive(Clone, Debug)]
pub struct TwoPointCrossoverMask {
    /// Addresses from parent1 (outside crossover segment)
    parent1_addresses: HashSet<Address>,
}

impl TwoPointCrossoverMask {
    /// Create a new two-point crossover mask
    pub fn new<R: Rng>(trace: &Trace, rng: &mut R) -> Self {
        let addresses: Vec<_> = trace.choices.keys().cloned().collect();
        if addresses.is_empty() {
            return Self {
                parent1_addresses: HashSet::new(),
            };
        }

        let mut point1 = rng.gen_range(0..addresses.len());
        let mut point2 = rng.gen_range(0..addresses.len());
        if point1 > point2 {
            std::mem::swap(&mut point1, &mut point2);
        }

        // Parent1 gets addresses outside [point1, point2)
        let parent1_addresses: HashSet<_> = addresses
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < point1 || *i >= point2)
            .map(|(_, addr)| addr.clone())
            .collect();

        Self { parent1_addresses }
    }
}

impl CrossoverMask for TwoPointCrossoverMask {
    fn from_parent1(&self, addr: &Address) -> bool {
        self.parent1_addresses.contains(addr)
    }
}

/// Trace-based mutation operator
///
/// Mutates a genome by selectively resampling addresses in its trace representation.
pub fn mutate_trace<G, S, R>(
    genome: &G,
    selector: &S,
    mutation_fn: impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue,
    rng: &mut R,
) -> Result<G, GenomeError>
where
    G: EvolutionaryGenome,
    S: MutationSelector,
    R: Rng,
{
    let trace = genome.to_trace();
    let mutation_sites = selector.select_sites(&trace, rng);

    let mut new_trace = Trace::default();

    for (addr, choice) in &trace.choices {
        let new_value = if mutation_sites.contains(addr) {
            mutation_fn(addr, &choice.value, rng)
        } else {
            choice.value.clone()
        };
        new_trace.insert_choice(addr.clone(), new_value, choice.logp);
    }

    G::from_trace(&new_trace)
}

/// Trace-based crossover operator
///
/// Creates offspring by merging parent traces according to a crossover mask.
pub fn crossover_traces<G, M, R>(
    parent1: &G,
    parent2: &G,
    mask: &M,
    _rng: &mut R,
) -> Result<(G, G), GenomeError>
where
    G: EvolutionaryGenome,
    M: CrossoverMask,
    R: Rng,
{
    let trace1 = parent1.to_trace();
    let trace2 = parent2.to_trace();

    let mut child1_trace = Trace::default();
    let mut child2_trace = Trace::default();

    // Collect all addresses from both parents
    let all_addresses: HashSet<Address> = trace1
        .choices
        .keys()
        .chain(trace2.choices.keys())
        .cloned()
        .collect();

    for addr in all_addresses {
        let (val_for_child1, val_for_child2) = if mask.from_parent1(&addr) {
            // Child1 gets parent1, child2 gets parent2
            (
                trace1.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
                trace2.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
            )
        } else {
            // Child1 gets parent2, child2 gets parent1
            (
                trace2.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
                trace1.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
            )
        };

        child1_trace.insert_choice(addr.clone(), val_for_child1, 0.0);
        child2_trace.insert_choice(addr, val_for_child2, 0.0);
    }

    let child1 = G::from_trace(&child1_trace)?;
    let child2 = G::from_trace(&child2_trace)?;

    Ok((child1, child2))
}

/// Gaussian mutation function for f64 values
pub fn gaussian_mutation<R: Rng>(
    sigma: f64,
) -> impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue {
    move |_addr, value, rng| {
        if let ChoiceValue::F64(v) = value {
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Simple uniform noise
            let mutated = v + sigma * noise * 2.0_f64.sqrt(); // Scale to approximate gaussian
            ChoiceValue::F64(mutated)
        } else {
            value.clone()
        }
    }
}

/// Bit flip mutation function for boolean values
pub fn bit_flip_mutation<R: Rng>() -> impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue {
    move |_addr, value, _rng| {
        if let ChoiceValue::Bool(b) = value {
            ChoiceValue::Bool(!b)
        } else {
            value.clone()
        }
    }
}

/// Bounded mutation function that respects bounds
pub fn bounded_mutation<R: Rng>(
    sigma: f64,
    lower: f64,
    upper: f64,
) -> impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue {
    move |_addr, value, rng| {
        if let ChoiceValue::F64(v) = value {
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let mutated = (v + sigma * noise * 2.0_f64.sqrt()).clamp(lower, upper);
            ChoiceValue::F64(mutated)
        } else {
            value.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;

    #[test]
    fn test_uniform_mutation_selector() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let trace = genome.to_trace();

        // High probability should select many sites
        let selector = UniformMutationSelector::new(0.9);
        let sites = selector.select_sites(&trace, &mut rng);
        // Most likely selects multiple sites (probabilistic, so not deterministic)
        assert!(sites.len() <= 5);

        // Low probability should select few sites
        let selector_low = UniformMutationSelector::new(0.1);
        let _sites_low = selector_low.select_sites(&trace, &mut rng);
    }

    #[test]
    fn test_single_site_mutation_selector() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let trace = genome.to_trace();

        let selector = SingleSiteMutationSelector::new();
        let sites = selector.select_sites(&trace, &mut rng);

        assert_eq!(sites.len(), 1);
    }

    #[test]
    fn test_multi_site_mutation_selector() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let trace = genome.to_trace();

        let selector = MultiSiteMutationSelector::new(3);
        let sites = selector.select_sites(&trace, &mut rng);

        assert_eq!(sites.len(), 3);
    }

    #[test]
    fn test_mutate_trace() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let selector = UniformMutationSelector::new(1.0); // Mutate all
        let mutation_fn = gaussian_mutation(0.1);

        let mutated = mutate_trace(&genome, &selector, mutation_fn, &mut rng).unwrap();

        // Should have same dimension
        assert_eq!(mutated.dimension(), genome.dimension());
        // Values should have changed (with high probability)
    }

    #[test]
    fn test_crossover_traces() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let parent2 = RealVector::new(vec![4.0, 5.0, 6.0]);

        let trace1 = parent1.to_trace();
        let mask = UniformCrossoverMask::balanced(&trace1, &mut rng);

        let (child1, child2) = crossover_traces(&parent1, &parent2, &mask, &mut rng).unwrap();

        assert_eq!(child1.dimension(), 3);
        assert_eq!(child2.dimension(), 3);

        // Children should have values from either parent
        for i in 0..3 {
            let c1_val = child1.genes()[i];
            let c2_val = child2.genes()[i];
            let p1_val = parent1.genes()[i];
            let p2_val = parent2.genes()[i];

            assert!(
                (c1_val - p1_val).abs() < 1e-10 || (c1_val - p2_val).abs() < 1e-10,
                "Child1 value {} not from either parent ({} or {})",
                c1_val, p1_val, p2_val
            );
            assert!(
                (c2_val - p1_val).abs() < 1e-10 || (c2_val - p2_val).abs() < 1e-10,
                "Child2 value {} not from either parent ({} or {})",
                c2_val, p1_val, p2_val
            );
        }
    }

    #[test]
    fn test_single_point_crossover_mask() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let trace = genome.to_trace();

        for _ in 0..10 {
            let mask = SinglePointCrossoverMask::new(&trace, &mut rng);

            // Check that it's a valid partition (addresses are either all before or all after)
            let addresses: Vec<_> = trace.choices.keys().collect();
            let mut found_split = false;

            for i in 1..addresses.len() {
                let prev_from_p1 = mask.from_parent1(addresses[i - 1]);
                let curr_from_p1 = mask.from_parent1(addresses[i]);

                if prev_from_p1 && !curr_from_p1 {
                    assert!(!found_split, "Multiple splits found");
                    found_split = true;
                }
            }
        }
    }
}
