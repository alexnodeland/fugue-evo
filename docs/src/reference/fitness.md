# Fitness Functions Reference

Fitness functions evaluate solution quality.

## Module

```rust,ignore
use fugue_evo::fitness::{Fitness, FitnessValue};
use fugue_evo::fitness::benchmarks::{Sphere, Rastrigin, Rosenbrock, Ackley};
```

## Fitness Trait

```rust,ignore
pub trait Fitness<G>: Send + Sync {
    type Value: FitnessValue;

    fn evaluate(&self, genome: &G) -> Self::Value;
}
```

## FitnessValue Trait

```rust,ignore
pub trait FitnessValue: Clone + PartialOrd + Send + Sync {
    fn to_f64(&self) -> f64;
}
```

Implemented for: `f64`, `f32`, `i64`, `i32`, `ParetoFitness`

## Built-in Benchmarks

### Sphere

```rust,ignore
let fitness = Sphere::new(dim);
```

Formula: `f(x) = Σxᵢ²`

| Property | Value |
|----------|-------|
| Optimum | 0 at origin |
| Modality | Unimodal |
| Separable | Yes |
| Bounds | [-5.12, 5.12] |

### Rastrigin

```rust,ignore
let fitness = Rastrigin::new(dim);
```

Formula: `f(x) = 10n + Σ[xᵢ² - 10cos(2πxᵢ)]`

| Property | Value |
|----------|-------|
| Optimum | 0 at origin |
| Modality | Highly multimodal (~10ⁿ local minima) |
| Separable | Yes |
| Bounds | [-5.12, 5.12] |

### Rosenbrock

```rust,ignore
let fitness = Rosenbrock::new(dim);
```

Formula: `f(x) = Σ[100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]`

| Property | Value |
|----------|-------|
| Optimum | 0 at (1,1,...,1) |
| Modality | Unimodal (curved valley) |
| Separable | No |
| Bounds | [-5, 10] |

### Ackley

```rust,ignore
let fitness = Ackley::new(dim);
```

| Property | Value |
|----------|-------|
| Optimum | 0 at origin |
| Modality | Multimodal |
| Separable | No |
| Bounds | [-32.768, 32.768] |

## ZDT Test Suite (Multi-Objective)

```rust,ignore
use fugue_evo::fitness::benchmarks::{Zdt1, Zdt2, Zdt3};

let fitness = Zdt1::new(30);  // 30 variables
```

| Function | Pareto Front Shape |
|----------|-------------------|
| ZDT1 | Convex |
| ZDT2 | Non-convex |
| ZDT3 | Disconnected |

## Custom Fitness

### Single Objective

```rust,ignore
struct MyFitness;

impl Fitness<RealVector> for MyFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        // Higher = better
        let genes = genome.genes();
        -genes.iter().map(|x| x * x).sum::<f64>()
    }
}
```

### Multi-Objective

```rust,ignore
struct MultiObjFitness;

impl Fitness<RealVector> for MultiObjFitness {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        ParetoFitness::new(vec![obj1, obj2])
    }
}
```

## CMA-ES Fitness

CMA-ES uses a separate trait (minimization):

```rust,ignore
pub trait CmaEsFitness: Send + Sync {
    fn evaluate(&self, x: &RealVector) -> f64;  // Returns value to MINIMIZE
}
```

## See Also

- [Custom Fitness Functions](../how-to/custom-fitness.md)
- [Multi-Objective Optimization](../how-to/multi-objective.md)
