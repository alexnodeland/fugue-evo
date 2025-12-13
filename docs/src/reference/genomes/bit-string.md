# BitString Reference

Fixed-length string of bits for binary/discrete optimization.

## Module

```rust,ignore
use fugue_evo::genome::bit_string::BitString;
```

## Construction

```rust,ignore
// From vector
let genome = BitString::new(vec![true, false, true, true]);

// All zeros
let genome = BitString::zeros(10);

// All ones
let genome = BitString::ones(10);

// Random
let genome = BitString::random(10, &mut rng);

// Random with probability
let genome = BitString::random_with_probability(10, 0.3, &mut rng);
```

## Methods

### Access

| Method | Return | Description |
|--------|--------|-------------|
| `bits()` | `&[bool]` | Immutable bit access |
| `bits_mut()` | `&mut [bool]` | Mutable bit access |
| `length()` | `usize` | Number of bits |
| `get(i)` | `Option<bool>` | Get bit at index |

### Bit Operations

| Method | Description |
|--------|-------------|
| `flip(i)` | Flip bit at index |
| `set(i, val)` | Set bit to value |
| `count_ones()` | Number of true bits |
| `count_zeros()` | Number of false bits |

### Binary Operations

| Method | Description |
|--------|-------------|
| `and(&other)` | Bitwise AND |
| `or(&other)` | Bitwise OR |
| `xor(&other)` | Bitwise XOR |
| `not()` | Bitwise NOT |

### Distance

| Method | Description |
|--------|-------------|
| `hamming_distance(&other)` | Number of differing bits |

## Trace Representation

Bits stored as 0.0 or 1.0:

```rust,ignore
addr!("bit", 0) → 0.0 or 1.0
addr!("bit", 1) → 0.0 or 1.0
// ...
```

## Recommended Operators

### Crossover

```rust,ignore
// Uniform: each bit from random parent
UniformCrossover::new()

// One-point crossover
OnePointCrossover::new()

// Two-point crossover
TwoPointCrossover::new()
```

### Mutation

```rust,ignore
// Flip each bit with probability
BitFlipMutation::new(0.01)  // 1% per bit
```

## Example: Knapsack Problem

```rust,ignore
use fugue_evo::prelude::*;

struct KnapsackFitness {
    values: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
}

impl Fitness<BitString> for KnapsackFitness {
    type Value = f64;

    fn evaluate(&self, genome: &BitString) -> f64 {
        let mut total_value = 0.0;
        let mut total_weight = 0.0;

        for (i, &selected) in genome.bits().iter().enumerate() {
            if selected {
                total_value += self.values[i];
                total_weight += self.weights[i];
            }
        }

        if total_weight > self.capacity {
            0.0  // Infeasible
        } else {
            total_value
        }
    }
}

// Usage
let fitness = KnapsackFitness {
    values: vec![60.0, 100.0, 120.0],
    weights: vec![10.0, 20.0, 30.0],
    capacity: 50.0,
};

let result = SimpleGABuilder::<BitString, f64, _, _, _, _, _>::new()
    .population_size(50)
    .initial_population((0..50).map(|_| BitString::random(3, &mut rng)).collect())
    .selection(TournamentSelection::new(3))
    .crossover(UniformCrossover::new())
    .mutation(BitFlipMutation::new(0.05))
    .fitness(fitness)
    .max_generations(100)
    .build()?
    .run(&mut rng)?;
```

## See Also

- [Custom Fitness Functions](../../how-to/custom-fitness.md)
- [Custom Operators](../../how-to/custom-operators.md)
