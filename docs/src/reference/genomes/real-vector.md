# RealVector Reference

Fixed-length vector of real-valued genes for continuous optimization.

## Module

```rust,ignore
use fugue_evo::genome::real_vector::RealVector;
```

## Construction

```rust,ignore
// From vector
let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

// With dimension
let genome = RealVector::zeros(10);

// Random within bounds
let genome = RealVector::random(&bounds, &mut rng);
```

## Methods

### Access

| Method | Return | Description |
|--------|--------|-------------|
| `genes()` | `&[f64]` | Immutable gene access |
| `genes_mut()` | `&mut [f64]` | Mutable gene access |
| `dimension()` | `usize` | Number of genes |
| `get(i)` | `Option<f64>` | Get gene at index |

### Vector Operations

| Method | Description |
|--------|-------------|
| `add(&other)` | Element-wise addition |
| `sub(&other)` | Element-wise subtraction |
| `scale(factor)` | Multiply all genes by factor |
| `norm()` | Euclidean norm |
| `euclidean_distance(&other)` | Distance between vectors |
| `dot(&other)` | Dot product |

### Statistics

| Method | Description |
|--------|-------------|
| `mean()` | Mean of genes |
| `variance()` | Variance of genes |
| `sum()` | Sum of genes |

## Trace Representation

Genes are stored at indexed addresses:

```rust,ignore
// Trace structure
addr!("gene", 0) → genes[0]
addr!("gene", 1) → genes[1]
// ...
```

## Usage with Operators

### Recommended Crossover

```rust,ignore
// Simulated Binary Crossover
SbxCrossover::new(20.0)

// Blend Crossover
BlendCrossover::new(0.5)

// Uniform Crossover
UniformCrossover::new()
```

### Recommended Mutation

```rust,ignore
// Polynomial Mutation
PolynomialMutation::new(20.0)

// Gaussian Mutation
GaussianMutation::new(0.1)
```

## Example

```rust,ignore
use fugue_evo::prelude::*;

let bounds = MultiBounds::symmetric(5.12, 10);
let mut genome = RealVector::random(&bounds, &mut rng);

// Modify genes
genome.genes_mut()[0] = 1.0;

// Vector operations
let other = RealVector::random(&bounds, &mut rng);
let distance = genome.euclidean_distance(&other);

// Use in GA
let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .bounds(bounds)
    // ...
```

## See Also

- [Continuous Optimization Tutorial](../../tutorials/continuous-optimization.md)
- [Custom Genome Types](../../how-to/custom-genome.md)
