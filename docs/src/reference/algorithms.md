# Algorithms Reference

This section provides detailed reference documentation for all optimization algorithms in fugue-evo.

## Algorithm Overview

| Algorithm | Module | Use Case |
|-----------|--------|----------|
| [SimpleGA](./algorithms/simple-ga.md) | `algorithms::simple_ga` | General-purpose optimization |
| [CMA-ES](./algorithms/cmaes.md) | `algorithms::cmaes` | Continuous non-separable optimization |
| [NSGA-II](./algorithms/nsga2.md) | `algorithms::nsga2` | Multi-objective optimization |
| [Island Model](./algorithms/island-model.md) | `algorithms::island` | Parallel multimodal optimization |
| [EDA/UMDA](./algorithms/eda.md) | `algorithms::eda` | Probabilistic model-based optimization |

## Common Patterns

### Builder Pattern

All algorithms use type-safe builders:

```rust,ignore
let result = SomeAlgorithmBuilder::new()
    .population_size(100)
    .max_generations(200)
    // ... configuration
    .build()?
    .run(&mut rng)?;
```

### Result Structure

Most algorithms return an `EvolutionResult`:

```rust,ignore
pub struct EvolutionResult<G, F> {
    pub best_genome: G,
    pub best_fitness: F,
    pub generations: usize,
    pub evaluations: usize,
    pub stats: EvolutionStats,
}
```

### Termination

Algorithms support pluggable termination criteria:

```rust,ignore
.max_generations(200)
// or
.termination(TargetFitness::new(-0.001))
// or
.termination(AnyOf::new(vec![
    Box::new(MaxGenerations::new(500)),
    Box::new(FitnessStagnation::new(50)),
]))
```

## Algorithm Selection

```text
                  Problem Type
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
Multi-Objective    Continuous        Discrete/Other
    │                  │                  │
    ▼                  ▼                  ▼
 NSGA-II          Separable?          SimpleGA
                       │
              ┌────────┼────────┐
              │ Yes    │        │ No
              ▼        │        ▼
          SimpleGA     │     CMA-ES
           or EDA      │
                       │
                  Multimodal?
                       │
              ┌────────┼────────┐
              │ Yes            │ No
              ▼                ▼
        Island Model       SimpleGA
```

## See Also

- [Choosing an Algorithm](../how-to/choosing-algorithm.md) - Decision guide
- [API Documentation](../api-docs.md) - Full rustdoc reference
