# Advanced Algorithms

Fugue-evo includes several advanced optimization algorithms beyond the Simple GA. This guide provides an overview and helps you choose the right algorithm for your problem.

## Algorithm Comparison

| Algorithm | Best For | Key Advantage |
|-----------|----------|---------------|
| **SimpleGA** | General-purpose | Flexible, well-understood |
| **CMA-ES** | Continuous (non-separable) | Learns problem structure |
| **NSGA-II** | Multi-objective | Pareto front discovery |
| **Island Model** | Multimodal | Maintains diversity |
| **EDA/UMDA** | Separable problems | Distribution learning |

## CMA-ES

**Covariance Matrix Adaptation Evolution Strategy** is a state-of-the-art algorithm for continuous optimization.

**When to use:**
- Continuous real-valued optimization
- Non-separable problems (variables interact)
- Problems where second-order information helps
- When you don't need custom operators

**Key features:**
- Adapts step size automatically
- Learns covariance structure (variable correlations)
- Typically requires fewer evaluations than GA

See [CMA-ES Tutorial](./cmaes.md) for details.

## NSGA-II

**Non-dominated Sorting Genetic Algorithm II** is designed for multi-objective optimization.

**When to use:**
- Multiple conflicting objectives
- Need the full Pareto front
- Trade-off analysis between objectives

**Key features:**
- Pareto dominance ranking
- Crowding distance for diversity
- Constraint handling support

See [Multi-Objective Optimization](../how-to/multi-objective.md) for details.

## Island Model

**Island Model** runs multiple populations in parallel with periodic migration.

**When to use:**
- Highly multimodal problems
- Parallel computation available
- Need to maintain diversity

**Key features:**
- Multiple independent populations
- Configurable topology (Ring, Star, Complete)
- Migration policies

See [Island Model Tutorial](./island-model.md) for details.

## EDA/UMDA

**Estimation of Distribution Algorithm** builds a probabilistic model of good solutions.

**When to use:**
- Separable problems
- When crossover disrupts good solutions
- Discrete optimization

**Key features:**
- No crossover operator needed
- Builds statistical model of population
- Good for problems with building blocks

## Algorithm Selection Guide

```
                    ┌─────────────────────┐
                    │   Is the problem    │
                    │   multi-objective?  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │ Yes                             │ No
              ▼                                 ▼
        ┌───────────┐                 ┌─────────────────────┐
        │  NSGA-II  │                 │ Is it continuous    │
        └───────────┘                 │ real-valued?        │
                                      └──────────┬──────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         │ Yes                                           │ No
                         ▼                                               ▼
               ┌─────────────────────┐                         ┌─────────────────┐
               │ Is it non-separable │                         │    SimpleGA     │
               │ or ill-conditioned? │                         │ (with BitString │
               └──────────┬──────────┘                         │ or Permutation) │
                          │                                    └─────────────────┘
          ┌───────────────┼───────────────┐
          │ Yes                           │ No
          ▼                               ▼
    ┌───────────┐               ┌─────────────────────┐
    │  CMA-ES   │               │ Is it highly        │
    └───────────┘               │ multimodal?         │
                                └──────────┬──────────┘
                                           │
                       ┌───────────────────┼───────────────────┐
                       │ Yes                                   │ No
                       ▼                                       ▼
               ┌─────────────────┐                    ┌─────────────────┐
               │  Island Model   │                    │    SimpleGA     │
               │   or Restarts   │                    └─────────────────┘
               └─────────────────┘
```

## Combining Algorithms

Sometimes the best approach is to combine algorithms:

### 1. CMA-ES After GA

Use GA for global exploration, then CMA-ES for local refinement:

```rust,ignore
// Phase 1: GA for exploration
let ga_result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .max_generations(50)
    // ...
    .run(&mut rng)?;

// Phase 2: CMA-ES from best GA solution
let initial_mean = ga_result.best_genome.genes().to_vec();
let mut cmaes = CmaEs::new(initial_mean, 0.5);
let final_result = cmaes.run_generations(&fitness, 100, &mut rng)?;
```

### 2. Island Model with Different Algorithms

Run different configurations on different islands:

```rust,ignore
// Create islands with varied parameters
let model = IslandModelBuilder::<RealVector, _, _, _, _, f64>::new()
    .num_islands(4)
    .migration_interval(25)
    // Each island evolves differently due to RNG
    .build(&mut rng)?;
```

## Performance Considerations

### Evaluation Cost

If fitness evaluation is expensive:
- **CMA-ES**: Typically needs fewer evaluations
- **Smaller populations**: Fewer evaluations per generation
- **Surrogate models**: Approximate fitness for pre-selection

### Parallelism

If you have multiple CPU cores:
- **Island Model**: Natural parallelism
- **Parallel evaluation**: Enable with `parallel` feature
- **Multiple runs**: Compare results from different seeds

### Memory

For very high-dimensional problems:
- **CMA-ES**: O(n²) memory for covariance matrix
- **SimpleGA**: O(population × n) memory
- **Consider dimensionality reduction** for n > 1000

## Next Steps

Dive into specific algorithms:
- [CMA-ES Tutorial](./cmaes.md)
- [Island Model Tutorial](./island-model.md)
