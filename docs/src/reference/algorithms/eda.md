# EDA/UMDA Reference

Estimation of Distribution Algorithms using probabilistic models.

## Module

```rust,ignore
use fugue_evo::algorithms::eda::{Umda, UmdaBuilder, UmdaConfig};
```

## Overview

EDAs replace crossover/mutation with probabilistic model building:

1. Select promising individuals
2. Build probabilistic model from selected
3. Sample new individuals from model
4. Repeat

**UMDA** (Univariate Marginal Distribution Algorithm) assumes variables are independent.

## Builder API

### Required Configuration

| Method | Type | Description |
|--------|------|-------------|
| `bounds(bounds)` | `MultiBounds` | Search space bounds |
| `fitness(fit)` | `impl Fitness` | Fitness function |

### Optional Configuration

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `population_size(n)` | `usize` | 100 | Population size |
| `selection_size(n)` | `usize` | 50 | Individuals for model building |
| `max_generations(n)` | `usize` | 100 | Max generations |

## Usage

### Basic Example

```rust,ignore
use fugue_evo::prelude::*;

let result = UmdaBuilder::<RealVector, f64, _>::new()
    .population_size(100)
    .selection_size(30)  // Top 30% for model
    .bounds(MultiBounds::symmetric(5.12, 10))
    .fitness(Sphere::new(10))
    .max_generations(200)
    .build()?
    .run(&mut rng)?;
```

## Algorithm Details

### Model Building (Univariate)

For continuous variables:
```text
μᵢ = mean of selected individuals' gene i
σᵢ = std dev of selected individuals' gene i
```

### Sampling

New individuals sampled from:
```text
xᵢ ~ N(μᵢ, σᵢ²)
```

### When to Use

**Good for:**
- Separable problems (variables independent)
- Problems where crossover is disruptive
- Understanding variable distributions

**Not good for:**
- Non-separable problems (use CMA-ES instead)
- Problems with variable interactions

## Comparison with GA

| Aspect | GA | EDA (UMDA) |
|--------|-----|------------|
| Variation | Crossover + Mutation | Model sampling |
| Interactions | Crossover preserves building blocks | Assumes independence |
| Parameters | Crossover rate, mutation rate | Selection ratio |

## See Also

- [Choosing an Algorithm](../../how-to/choosing-algorithm.md)
- [CMA-ES](./cmaes.md) - For non-separable problems
