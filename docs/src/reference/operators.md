# Operators Reference

Genetic operators for selection, crossover, and mutation.

## Overview

| Category | Purpose | Trait |
|----------|---------|-------|
| [Selection](./operators/selection.md) | Choose parents | `SelectionOperator` |
| [Crossover](./operators/crossover.md) | Combine parents | `CrossoverOperator` |
| [Mutation](./operators/mutation.md) | Add variation | `MutationOperator` |

## Operator Traits

### Selection

```rust,ignore
pub trait SelectionOperator<G>: Send + Sync {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize;
}
```

### Crossover

```rust,ignore
pub trait CrossoverOperator<G>: Send + Sync {
    type Output;
    fn crossover<R: Rng>(&self, p1: &G, p2: &G, rng: &mut R) -> Self::Output;
}

// For bounded genomes
pub trait BoundedCrossoverOperator<G>: Send + Sync {
    type Output;
    fn crossover_bounded<R: Rng>(
        &self, p1: &G, p2: &G, bounds: &MultiBounds, rng: &mut R
    ) -> Self::Output;
}
```

### Mutation

```rust,ignore
pub trait MutationOperator<G>: Send + Sync {
    fn mutate<R: Rng>(&self, genome: &mut G, rng: &mut R);
}

// For bounded genomes
pub trait BoundedMutationOperator<G>: Send + Sync {
    fn mutate_bounded<R: Rng>(
        &self, genome: &mut G, bounds: &MultiBounds, rng: &mut R
    );
}
```

## Operator Selection Guide

### For RealVector

| Operation | Recommended | Alternative |
|-----------|-------------|-------------|
| Selection | `TournamentSelection(3-5)` | `RankSelection` |
| Crossover | `SbxCrossover(15-25)` | `BlendCrossover(0.5)` |
| Mutation | `PolynomialMutation(15-25)` | `GaussianMutation(0.1)` |

### For BitString

| Operation | Recommended | Alternative |
|-----------|-------------|-------------|
| Selection | `TournamentSelection(3)` | `RouletteWheelSelection` |
| Crossover | `UniformCrossover` | `OnePointCrossover` |
| Mutation | `BitFlipMutation(0.01)` | - |

### For Permutation

| Operation | Recommended | Alternative |
|-----------|-------------|-------------|
| Selection | `TournamentSelection(5)` | - |
| Crossover | `OrderCrossover` | `PmxCrossover` |
| Mutation | `InversionMutation` | `SwapMutation` |

## See Also

- [Custom Operators](../how-to/custom-operators.md)
- [API Documentation](../api-docs.md)
