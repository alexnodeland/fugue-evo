# Custom Operators

This guide shows how to create custom selection, crossover, and mutation operators.

## Selection Operators

Selection chooses parents for reproduction based on fitness.

### Trait Definition

```rust,ignore
pub trait SelectionOperator<G>: Send + Sync {
    /// Select an individual from the population
    /// Returns the index of the selected individual
    fn select<R: Rng>(
        &self,
        population: &[(G, f64)], // (genome, fitness) pairs
        rng: &mut R,
    ) -> usize;
}
```

### Example: Boltzmann Selection

Temperature-controlled selection pressure:

```rust,ignore
use fugue_evo::prelude::*;

pub struct BoltzmannSelection {
    temperature: f64,
}

impl BoltzmannSelection {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl<G> SelectionOperator<G> for BoltzmannSelection {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize {
        // Compute Boltzmann probabilities
        let max_fitness = population.iter()
            .map(|(_, f)| *f)
            .fold(f64::NEG_INFINITY, f64::max);

        let weights: Vec<f64> = population.iter()
            .map(|(_, f)| ((f - max_fitness) / self.temperature).exp())
            .collect();

        let total: f64 = weights.iter().sum();

        // Roulette wheel selection
        let mut target = rng.gen::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            target -= w;
            if target <= 0.0 {
                return i;
            }
        }

        population.len() - 1
    }
}
```

**Usage:**

```rust,ignore
let selection = BoltzmannSelection::new(1.0); // Higher temp = more random
```

## Crossover Operators

Crossover combines two parents to create offspring.

### Trait Definition

```rust,ignore
pub trait CrossoverOperator<G>: Send + Sync {
    type Output;

    fn crossover<R: Rng>(
        &self,
        parent1: &G,
        parent2: &G,
        rng: &mut R,
    ) -> Self::Output;
}
```

### Example: Blend Crossover (BLX-α)

Creates offspring in an expanded range around parents:

```rust,ignore
pub struct BlendCrossover {
    alpha: f64,
}

impl BlendCrossover {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl CrossoverOperator<RealVector> for BlendCrossover {
    type Output = CrossoverResult<RealVector>;

    fn crossover<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        rng: &mut R,
    ) -> Self::Output {
        let g1 = parent1.genes();
        let g2 = parent2.genes();

        let mut child1_genes = Vec::with_capacity(g1.len());
        let mut child2_genes = Vec::with_capacity(g1.len());

        for i in 0..g1.len() {
            let min_val = g1[i].min(g2[i]);
            let max_val = g1[i].max(g2[i]);
            let range = max_val - min_val;

            // Expanded range: [min - α*range, max + α*range]
            let low = min_val - self.alpha * range;
            let high = max_val + self.alpha * range;

            child1_genes.push(rng.gen_range(low..=high));
            child2_genes.push(rng.gen_range(low..=high));
        }

        CrossoverResult::new(
            RealVector::new(child1_genes),
            RealVector::new(child2_genes),
        )
    }
}
```

**Usage:**

```rust,ignore
let crossover = BlendCrossover::new(0.5); // α = 0.5 is common
```

### Bounded Crossover

For crossover that needs bounds information:

```rust,ignore
pub trait BoundedCrossoverOperator<G>: Send + Sync {
    type Output;

    fn crossover_bounded<R: Rng>(
        &self,
        parent1: &G,
        parent2: &G,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> Self::Output;
}

impl BoundedCrossoverOperator<RealVector> for BlendCrossover {
    type Output = CrossoverResult<RealVector>;

    fn crossover_bounded<R: Rng>(
        &self,
        parent1: &RealVector,
        parent2: &RealVector,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> Self::Output {
        let result = self.crossover(parent1, parent2, rng);

        // Clamp to bounds
        let child1 = clamp_to_bounds(result.genome().unwrap().0, bounds);
        let child2 = clamp_to_bounds(result.genome().unwrap().1, bounds);

        CrossoverResult::new(child1, child2)
    }
}
```

## Mutation Operators

Mutation introduces random variation.

### Trait Definition

```rust,ignore
pub trait MutationOperator<G>: Send + Sync {
    fn mutate<R: Rng>(&self, genome: &mut G, rng: &mut R);
}
```

### Example: Cauchy Mutation

Heavy-tailed mutation for escaping local optima:

```rust,ignore
use rand_distr::{Cauchy, Distribution};

pub struct CauchyMutation {
    scale: f64,
    probability: f64,
}

impl CauchyMutation {
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            probability: 1.0,
        }
    }

    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p;
        self
    }
}

impl MutationOperator<RealVector> for CauchyMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        let cauchy = Cauchy::new(0.0, self.scale).unwrap();

        for gene in genome.genes_mut() {
            if rng.gen::<f64>() < self.probability {
                *gene += cauchy.sample(rng);
            }
        }
    }
}
```

### Adaptive Mutation

Mutation that changes based on progress:

```rust,ignore
pub struct AdaptiveMutation {
    initial_sigma: f64,
    final_sigma: f64,
    current_generation: usize,
    max_generations: usize,
}

impl AdaptiveMutation {
    pub fn new(initial: f64, final_val: f64, max_gen: usize) -> Self {
        Self {
            initial_sigma: initial,
            final_sigma: final_val,
            current_generation: 0,
            max_generations: max_gen,
        }
    }

    pub fn set_generation(&mut self, gen: usize) {
        self.current_generation = gen;
    }

    fn current_sigma(&self) -> f64 {
        let progress = self.current_generation as f64 / self.max_generations as f64;
        self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress
    }
}

impl MutationOperator<RealVector> for AdaptiveMutation {
    fn mutate<R: Rng>(&self, genome: &mut RealVector, rng: &mut R) {
        let sigma = self.current_sigma();
        let normal = rand_distr::Normal::new(0.0, sigma).unwrap();

        for gene in genome.genes_mut() {
            *gene += normal.sample(rng);
        }
    }
}
```

## Problem-Specific Operators

### Permutation: Order Crossover (OX)

```rust,ignore
pub struct OrderCrossover;

impl CrossoverOperator<Permutation> for OrderCrossover {
    type Output = CrossoverResult<Permutation>;

    fn crossover<R: Rng>(
        &self,
        parent1: &Permutation,
        parent2: &Permutation,
        rng: &mut R,
    ) -> Self::Output {
        let n = parent1.len();
        let p1 = parent1.as_slice();
        let p2 = parent2.as_slice();

        // Select crossover segment
        let mut start = rng.gen_range(0..n);
        let mut end = rng.gen_range(0..n);
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        // Child 1: segment from p1, rest from p2 in order
        let mut child1 = vec![usize::MAX; n];
        let mut used1: HashSet<usize> = HashSet::new();

        // Copy segment
        for i in start..=end {
            child1[i] = p1[i];
            used1.insert(p1[i]);
        }

        // Fill rest from p2
        let mut pos = (end + 1) % n;
        for &val in p2.iter().cycle().skip(end + 1).take(n) {
            if !used1.contains(&val) {
                child1[pos] = val;
                used1.insert(val);
                pos = (pos + 1) % n;
                if pos == start {
                    break;
                }
            }
        }

        // Similarly for child2...
        let child2 = /* symmetric construction */;

        CrossoverResult::new(
            Permutation::new(child1),
            Permutation::new(child2),
        )
    }
}
```

### BitString: Intelligent Flip

```rust,ignore
pub struct IntelligentBitFlip {
    /// Probability of flipping 0→1
    p_set: f64,
    /// Probability of flipping 1→0
    p_clear: f64,
}

impl MutationOperator<BitString> for IntelligentBitFlip {
    fn mutate<R: Rng>(&self, genome: &mut BitString, rng: &mut R) {
        for bit in genome.bits_mut() {
            let p = if *bit { self.p_clear } else { self.p_set };
            if rng.gen::<f64>() < p {
                *bit = !*bit;
            }
        }
    }
}
```

## Combining Operators

### Composite Mutation

Apply multiple mutations:

```rust,ignore
pub struct CompositeMutation<M1, M2> {
    mutation1: M1,
    mutation2: M2,
    p1: f64, // Probability of using mutation1
}

impl<G, M1, M2> MutationOperator<G> for CompositeMutation<M1, M2>
where
    M1: MutationOperator<G>,
    M2: MutationOperator<G>,
{
    fn mutate<R: Rng>(&self, genome: &mut G, rng: &mut R) {
        if rng.gen::<f64>() < self.p1 {
            self.mutation1.mutate(genome, rng);
        } else {
            self.mutation2.mutate(genome, rng);
        }
    }
}
```

## Testing Operators

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossover_preserves_genes() {
        let p1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let p2 = RealVector::new(vec![4.0, 5.0, 6.0]);

        let crossover = BlendCrossover::new(0.0); // No expansion
        let mut rng = StdRng::seed_from_u64(42);

        let result = crossover.crossover(&p1, &p2, &mut rng);
        let (c1, c2) = result.genome().unwrap();

        // Children should be within parent ranges
        for i in 0..3 {
            let min = p1.genes()[i].min(p2.genes()[i]);
            let max = p1.genes()[i].max(p2.genes()[i]);
            assert!(c1.genes()[i] >= min && c1.genes()[i] <= max);
        }
    }

    #[test]
    fn test_mutation_changes_genome() {
        let mut genome = RealVector::new(vec![0.0; 10]);
        let mutation = CauchyMutation::new(1.0);
        let mut rng = StdRng::seed_from_u64(42);

        let original = genome.clone();
        mutation.mutate(&mut genome, &mut rng);

        assert_ne!(genome.genes(), original.genes());
    }
}
```

## Next Steps

- [Custom Genome Types](./custom-genome.md) - Create genomes for your operators
- [Hyperparameter Learning](../tutorials/hyperparameter-learning.md) - Learn operator parameters
