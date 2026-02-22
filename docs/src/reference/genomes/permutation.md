# Permutation Reference

Ordered arrangement of elements for ordering/scheduling problems.

## Module

```rust,ignore
use fugue_evo::genome::permutation::Permutation;
```

## Construction

```rust,ignore
// From vector
let genome = Permutation::new(vec![2, 0, 3, 1]);

// Identity permutation [0, 1, 2, ..., n-1]
let genome = Permutation::identity(5);

// Random permutation
let genome = Permutation::random(5, &mut rng);
```

## Methods

### Access

| Method | Return | Description |
|--------|--------|-------------|
| `as_slice()` | `&[usize]` | Immutable element access |
| `as_mut_slice()` | `&mut [usize]` | Mutable element access |
| `len()` | `usize` | Number of elements |
| `get(i)` | `Option<usize>` | Get element at position |

### Permutation Operations

| Method | Description |
|--------|-------------|
| `swap(i, j)` | Swap elements at positions |
| `reverse_segment(i, j)` | Reverse segment [i, j] |
| `insert(from, to)` | Remove from `from`, insert at `to` |
| `inverse()` | Compute inverse permutation |
| `compose(&other)` | Compose permutations |

### Validation

| Method | Return | Description |
|--------|--------|-------------|
| `is_valid()` | `bool` | Check if valid permutation |
| `contains(elem)` | `bool` | Check if contains element |

### Distance

| Method | Description |
|--------|-------------|
| `kendall_tau_distance(&other)` | Number of pairwise disagreements |

## Trace Representation

Elements stored at indexed addresses:

```rust,ignore
addr!("pos", 0) → permutation[0]
addr!("pos", 1) → permutation[1]
// ...
```

## Recommended Operators

### Crossover

```rust,ignore
// Order Crossover (OX)
OrderCrossover::new()

// Partially Mapped Crossover (PMX)
PmxCrossover::new()

// Cycle Crossover
CycleCrossover::new()
```

### Mutation

```rust,ignore
// Swap two random positions
SwapMutation::new(0.1)

// Reverse a segment
InversionMutation::new(0.1)

// Move element to new position
InsertionMutation::new(0.1)
```

## Example: TSP

```rust,ignore
use fugue_evo::prelude::*;

struct TSPFitness {
    distances: Vec<Vec<f64>>,  // Distance matrix
}

impl Fitness<Permutation> for TSPFitness {
    type Value = f64;

    fn evaluate(&self, genome: &Permutation) -> f64 {
        let tour = genome.as_slice();
        let n = tour.len();

        let mut total_distance = 0.0;
        for i in 0..n {
            let from = tour[i];
            let to = tour[(i + 1) % n];
            total_distance += self.distances[from][to];
        }

        -total_distance  // Negate for maximization
    }
}

// Usage
let n_cities = 10;
let distances = generate_distance_matrix(n_cities);
let fitness = TSPFitness { distances };

let result = SimpleGABuilder::<Permutation, f64, _, _, _, _, _>::new()
    .population_size(100)
    .initial_population((0..100).map(|_| Permutation::random(n_cities, &mut rng)).collect())
    .selection(TournamentSelection::new(5))
    .crossover(OrderCrossover::new())
    .mutation(InversionMutation::new(0.2))
    .fitness(fitness)
    .max_generations(500)
    .build()?
    .run(&mut rng)?;

println!("Best tour: {:?}", result.best_genome.as_slice());
println!("Tour length: {}", -result.best_fitness);
```

## Validity Guarantees

Permutation-specific operators maintain validity:
- Every element appears exactly once
- No duplicates
- Range is [0, n)

## See Also

- [Custom Operators](../../how-to/custom-operators.md)
- [Custom Genome Types](../../how-to/custom-genome.md)
