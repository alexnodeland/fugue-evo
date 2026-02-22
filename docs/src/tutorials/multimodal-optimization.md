# Multimodal Optimization Tutorial

This tutorial tackles the **Rastrigin function**, a challenging multimodal benchmark with many local optima. You'll learn strategies for escaping local optima and finding global solutions.

## The Rastrigin Function

The Rastrigin function is defined as:

```text
f(x) = 10n + Σ[xᵢ² - 10cos(2πxᵢ)]
```

**Properties:**
- Global minimum: 0 at origin
- **Highly multimodal**: ~10ⁿ local minima!
- Non-separable: Variables interact through cosine terms
- Deceptive: Local minima look similar to global minimum

## The Challenge

With 20 dimensions, the Rastrigin function has approximately 10²⁰ local minima. A naive optimization will likely get trapped in one of these local minima.

## Complete Example

```rust,ignore
{{#include ../../../examples/rastrigin_benchmark.rs}}
```

> **Source**: [`examples/rastrigin_benchmark.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/rastrigin_benchmark.rs)

## Running the Example

```bash
cargo run --example rastrigin_benchmark
```

## Key Strategies

### 1. Larger Population

```rust,ignore
.population_size(200)
```

More individuals means more parallel exploration of the search space. For multimodal functions, this helps maintain diversity and cover more basins of attraction.

### 2. Higher Selection Pressure

```rust,ignore
.selection(TournamentSelection::new(5))
```

Tournament size of 5 (vs. 3 for Sphere) increases selection pressure, helping the population converge faster on good regions once found.

### 3. More Exploration in Crossover

```rust,ignore
.crossover(SbxCrossover::new(15.0))
```

Lower distribution index (15 vs. 20) creates more diverse offspring, helping explore new regions.

### 4. Elitism

```rust,ignore
.elitism(true)
.elite_count(2)
```

Critical for multimodal optimization! Without elitism, the best solution can be lost due to selection randomness.

### 5. More Generations

```rust,ignore
.max_generations(500)
```

Multimodal problems need more time to find and refine global optima.

## Understanding Results

### Tracking Progress

```rust,ignore
let history = result.stats.best_fitness_history();
for (i, fitness) in history.iter().enumerate() {
    if i % 50 == 0 {
        println!("  Gen {:4}: {:.6}", i, fitness);
    }
}
```

Watch for:
- **Rapid early improvement**: Finding good basins
- **Plateaus**: Stuck in local optima
- **Jumps**: Escaping to better regions

### Solution Quality

```rust,ignore
let max_deviation = result
    .best_genome
    .genes()
    .iter()
    .map(|x| x.abs())
    .fold(0.0f64, |a, b| a.max(b));

println!("Max deviation from origin: {:.6}", max_deviation);
```

For Rastrigin, each gene should be close to 0. Large deviations indicate the solution is in a local optimum.

## Alternative Approaches

### Island Model

For heavily multimodal problems, consider the Island Model:

```rust,ignore
// Multiple populations with periodic migration
let result = IslandModelBuilder::<RealVector, _, _, _, _, f64>::new()
    .num_islands(4)
    .island_population_size(50)
    .migration_interval(25)
    .migration_policy(MigrationPolicy::Best(2))
    // ... operators
    .build(&mut rng)?
    .run(200, &mut rng)?;
```

See [Island Model Tutorial](./island-model.md) for details.

### Restart Strategy

Manual restarts can help escape deep local optima:

```rust,ignore
let mut best_overall = f64::NEG_INFINITY;
let mut best_genome = None;

for restart in 0..5 {
    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        // ... configuration
        .build()?
        .run(&mut rng)?;

    if result.best_fitness > best_overall {
        best_overall = result.best_fitness;
        best_genome = Some(result.best_genome);
    }

    println!("Restart {}: {:.6}", restart, result.best_fitness);
}
```

## Parameter Guidelines for Multimodal Functions

| Parameter | Unimodal | Multimodal |
|-----------|----------|------------|
| Population | 50-100 | 150-300 |
| Tournament size | 2-3 | 4-7 |
| SBX eta | 15-25 | 10-15 |
| Mutation probability | 1/n | 1.5/n - 2/n |
| Generations | 100-300 | 300-1000 |
| Elitism | Optional | Essential |

## Diagnosing Problems

### Stuck in Local Optima

**Symptoms**:
- Fitness plateaus early
- Solution values are multiples of π (Rastrigin local minima)

**Solutions**:
1. Increase mutation probability
2. Use Island Model
3. Try different random seeds
4. Add restarts

### Loss of Diversity

**Symptoms**:
- All individuals become similar
- No improvement despite many generations

**Solutions**:
1. Increase population size
2. Lower selection pressure
3. Add diversity maintenance (niching)

## Exercises

1. **Vary dimensions**: Compare results for DIM = 10, 20, 30
2. **Compare seeds**: Run with 5 different seeds and analyze variance
3. **Population study**: Try populations of 50, 100, 200, 400

## Next Steps

- [Island Model Tutorial](./island-model.md) - Parallel populations for multimodal optimization
- [CMA-ES Tutorial](./cmaes.md) - Alternative algorithm for continuous optimization
