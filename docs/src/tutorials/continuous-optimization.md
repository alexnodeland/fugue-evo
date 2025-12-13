# Continuous Optimization Tutorial

This tutorial covers optimizing functions with real-valued variables. You'll learn how to set up, run, and analyze continuous optimization problems.

## The Sphere Function

We'll start with the **Sphere function**, a simple unimodal benchmark:

```
f(x) = Σxᵢ² = x₁² + x₂² + ... + xₙ²
```

**Properties:**
- Global minimum: 0 at origin (all zeros)
- Unimodal: Single optimum, no local minima
- Convex: Any local minimum is global
- Separable: Each variable is independent

## Complete Example

```rust,ignore
{{#include ../../../examples/sphere_optimization.rs}}
```

> **Source**: [`examples/sphere_optimization.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/sphere_optimization.rs)

## Running the Example

```bash
cargo run --example sphere_optimization
```

## Understanding the Components

### Search Bounds

```rust,ignore
let bounds = MultiBounds::symmetric(5.12, DIM);
```

This creates bounds [-5.12, 5.12] for each of the 10 dimensions. The Sphere function is typically tested in this range.

### Operators for Real-Valued Optimization

**Simulated Binary Crossover (SBX)**
```rust,ignore
SbxCrossover::new(20.0)
```

SBX creates offspring similar to parent values. The distribution index (eta = 20.0) controls spread:
- Higher eta → offspring closer to parents (exploitation)
- Lower eta → more diverse offspring (exploration)

**Polynomial Mutation**
```rust,ignore
PolynomialMutation::new(20.0)
```

Adds bounded perturbations to gene values. The distribution index controls mutation magnitude.

### Fitness Function

```rust,ignore
let fitness = Sphere::new(DIM);
```

Built-in `Sphere` computes the sum of squares. Since GAs maximize by default, the function is negated internally.

## Analyzing Results

```rust,ignore
println!("Best fitness: {:.6}", result.best_fitness);
println!("Generations:  {}", result.generations);
println!("Evaluations:  {}", result.evaluations);
```

Key metrics:
- **Best fitness**: Should approach 0 (the global minimum)
- **Generations**: How many evolutionary cycles completed
- **Evaluations**: Total fitness function calls

### Convergence Statistics

```rust,ignore
println!("{}", result.stats.summary());
```

The summary shows:
- Mean and standard deviation per generation
- Diversity metrics
- Improvement trends

## Parameter Tuning

### Population Size

```rust,ignore
.population_size(100)
```

| Size | Trade-off |
|------|-----------|
| Small (20-50) | Fast generations, less diversity |
| Medium (100-200) | Good balance |
| Large (500+) | More exploration, slower per generation |

For Sphere (unimodal), smaller populations work well. Multimodal functions need larger populations.

### Operator Parameters

**SBX Distribution Index**
```rust,ignore
.crossover(SbxCrossover::new(eta))
```

| eta | Effect |
|-----|--------|
| 1-5 | High exploration |
| 15-25 | Balanced |
| 50+ | High exploitation |

**Mutation Probability**
```rust,ignore
.mutation(PolynomialMutation::new(20.0).with_probability(0.1))
```

Default is 1/dimension. Increase for more exploration.

### Elitism

```rust,ignore
.elitism(true)
.elite_count(2)
```

Elitism preserves the best individuals across generations, preventing loss of good solutions.

## Common Issues

### Premature Convergence

**Symptoms**: Population converges too quickly, stuck at suboptimal solution

**Solutions**:
1. Increase population size
2. Lower selection pressure (smaller tournament size)
3. Increase mutation probability

### Slow Convergence

**Symptoms**: Fitness improves very slowly

**Solutions**:
1. Increase selection pressure
2. Reduce mutation (more exploitation)
3. Use elitism to preserve progress

## Exercises

1. **Change dimensions**: Modify `DIM` to 30 and observe how convergence changes
2. **Adjust operators**: Try eta values of 5 and 50 for crossover
3. **Compare selections**: Replace `TournamentSelection` with `RouletteWheelSelection`

## Next Steps

- [Multimodal Optimization](./multimodal-optimization.md) - Handle functions with local optima
- [CMA-ES Tutorial](./cmaes.md) - State-of-the-art continuous optimization
