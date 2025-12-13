# Quick Start

This guide walks you through running your first optimization with fugue-evo in under 5 minutes.

## Prerequisites

Ensure you have [installed fugue-evo](./installation.md).

## The Problem: Minimize the Sphere Function

The **Sphere function** is a classic optimization benchmark:

```
f(x) = x₁² + x₂² + ... + xₙ²
```

The global minimum is at the origin (all zeros) with a value of 0.

## Full Example

Here's the complete code to optimize the Sphere function:

```rust,ignore
{{#include ../../../examples/sphere_optimization.rs}}
```

> **Source**: [`examples/sphere_optimization.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/sphere_optimization.rs)

## Running the Example

Run the example directly:

```bash
cargo run --example sphere_optimization
```

Expected output:

```
=== Sphere Function Optimization ===

Optimization complete!
  Best fitness: -0.000023
  Generations:  200
  Evaluations:  20000

Best solution:
  x[0] = 0.001234
  x[1] = -0.000567
  ...

Distance from optimum: 0.004567
```

## Code Breakdown

### 1. Imports and Setup

```rust,ignore
use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

let mut rng = StdRng::seed_from_u64(42);
```

The prelude imports everything you need. We use a seeded RNG for reproducibility.

### 2. Define the Problem

```rust,ignore
const DIM: usize = 10;
let fitness = Sphere::new(DIM);
let bounds = MultiBounds::symmetric(5.12, DIM);
```

- `DIM`: Number of variables to optimize
- `Sphere::new(DIM)`: Built-in benchmark function
- `MultiBounds::symmetric(5.12, DIM)`: Search space [-5.12, 5.12] per dimension

### 3. Configure the Algorithm

```rust,ignore
let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .bounds(bounds)
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(fitness)
    .max_generations(200)
    .build()?
    .run(&mut rng)?;
```

| Setting | Value | Purpose |
|---------|-------|---------|
| `population_size` | 100 | Number of candidate solutions |
| `selection` | Tournament(3) | Select best of 3 random individuals |
| `crossover` | SBX(20.0) | Simulated Binary Crossover |
| `mutation` | Polynomial(20.0) | Polynomial mutation |
| `max_generations` | 200 | When to stop |

### 4. Analyze Results

```rust,ignore
println!("Best fitness: {:.6}", result.best_fitness);
println!("Generations:  {}", result.generations);

for (i, val) in result.best_genome.genes().iter().enumerate() {
    println!("  x[{}] = {:.6}", i, val);
}
```

## Understanding the Output

The fitness value should be close to 0 (the global minimum). The solution values should be close to 0 (the optimal point).

### What if Results Aren't Good?

If the solution isn't converging well:

1. **Increase population size**: More diversity helps exploration
2. **Increase generations**: More time to converge
3. **Adjust mutation**: Higher rates for exploration, lower for exploitation
4. **Try different selection pressure**: Higher tournament size = more exploitation

## Next Steps

- [Your First Optimization](./first-optimization.md) - Build a custom fitness function
- [Continuous Optimization Tutorial](../tutorials/continuous-optimization.md) - Deep dive into real-valued optimization
- [Choosing an Algorithm](../how-to/choosing-algorithm.md) - When to use different algorithms
