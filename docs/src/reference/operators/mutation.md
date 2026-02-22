# Mutation Operators Reference

Mutation operators add random variation to genomes.

## Module

```rust,ignore
use fugue_evo::operators::mutation::{
    PolynomialMutation, GaussianMutation, BitFlipMutation
};
```

## Polynomial Mutation

For real-valued genomes. Bounded perturbations.

```rust,ignore
let mutation = PolynomialMutation::new(eta);
let mutation = PolynomialMutation::new(eta).with_probability(0.1);
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `eta` | `f64` | 1-100+ | Distribution index |
| `probability` | `f64` | 0-1 | Per-gene mutation probability |

**Distribution index effects:**
- Low eta (1-5): Large mutations
- Medium eta (15-25): Balanced
- High eta (50+): Small mutations

**Default probability:** 1/n (one gene on average)

## Gaussian Mutation

Adds Gaussian noise to genes.

```rust,ignore
let mutation = GaussianMutation::new(sigma);
let mutation = GaussianMutation::new(sigma).with_probability(0.1);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `sigma` | `f64` | Standard deviation of noise |
| `probability` | `f64` | Per-gene mutation probability |

**Note:** May violate bounds. Use with `BoundedMutationOperator` or clamp manually.

## Bit-Flip Mutation

For binary genomes. Flips each bit with probability.

```rust,ignore
let mutation = BitFlipMutation::new(probability);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `probability` | `f64` | Per-bit flip probability |

**Typical values:** 0.01-0.05 (1-5% per bit)

## Swap Mutation

For permutations. Swaps two random positions.

```rust,ignore
let mutation = SwapMutation::new(probability);
```

## Inversion Mutation

For permutations. Reverses a random segment.

```rust,ignore
let mutation = InversionMutation::new(probability);
```

```text
Before: [1, 2, 3, 4, 5, 6]
              ↓ reverse [2,5]
After:  [1, 5, 4, 3, 2, 6]
```

## Insertion Mutation

For permutations. Moves element to new position.

```rust,ignore
let mutation = InsertionMutation::new(probability);
```

## Mutation Probability Guidelines

| Genome | Operator | Typical Probability |
|--------|----------|---------------------|
| RealVector | Polynomial | 1/n or 0.1 |
| RealVector | Gaussian | 1/n or 0.1 |
| BitString | BitFlip | 0.01-0.05 |
| Permutation | Swap | 0.1-0.3 |
| Permutation | Inversion | 0.1-0.3 |

## Usage

```rust,ignore
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .mutation(PolynomialMutation::new(20.0).with_probability(0.1))
    // ...
```

## Bounded Mutation

For mutations respecting bounds:

```rust,ignore
// Polynomial automatically respects bounds when used with SimpleGA

// Manual bounded mutation
mutation.mutate_bounded(&mut genome, &bounds, &mut rng);
```

## See Also

- [Custom Operators](../../how-to/custom-operators.md)
- [Hyperparameter Learning](../../tutorials/hyperparameter-learning.md) - Learn mutation rates
