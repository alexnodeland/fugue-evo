# CMA-ES Reference

Covariance Matrix Adaptation Evolution Strategy for continuous optimization.

## Module

```rust,ignore
use fugue_evo::algorithms::cmaes::{CmaEs, CmaEsState, CmaEsFitness};
```

## Constructor

```rust,ignore
pub fn new(initial_mean: Vec<f64>, initial_sigma: f64) -> Self
```

| Parameter | Description |
|-----------|-------------|
| `initial_mean` | Starting point (center of search distribution) |
| `initial_sigma` | Initial step size (standard deviation) |

## Configuration Methods

| Method | Description |
|--------|-------------|
| `with_bounds(bounds)` | Set search space bounds |
| `with_population_size(n)` | Override default population size |

## Fitness Trait

CMA-ES uses a different fitness trait (minimization):

```rust,ignore
pub trait CmaEsFitness: Send + Sync {
    /// Evaluate solution - return value to MINIMIZE
    fn evaluate(&self, x: &RealVector) -> f64;
}
```

## Usage

### Basic Example

```rust,ignore
use fugue_evo::prelude::*;

struct MyFitness;

impl CmaEsFitness for MyFitness {
    fn evaluate(&self, x: &RealVector) -> f64 {
        // Return value to minimize
        x.genes().iter().map(|g| g * g).sum()
    }
}

let mut cmaes = CmaEs::new(vec![0.0; 10], 0.5)
    .with_bounds(MultiBounds::symmetric(5.0, 10));

let best = cmaes.run_generations(&MyFitness, 1000, &mut rng)?;

println!("Best fitness: {}", best.fitness_value());
```

### Step-by-Step

```rust,ignore
let mut cmaes = CmaEs::new(vec![0.0; 10], 1.0);

for _ in 0..100 {
    cmaes.step(&fitness, &mut rng)?;

    println!("Generation {}: sigma = {:.6}",
        cmaes.state.generation,
        cmaes.state.sigma);

    // Early stopping
    if cmaes.state.sigma < 1e-10 {
        break;
    }
}
```

## State Structure

```rust,ignore
pub struct CmaEsState {
    pub generation: usize,
    pub evaluations: usize,
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub covariance: Vec<Vec<f64>>,
    // ... internal adaptation parameters
}
```

## Algorithm Parameters

CMA-ES automatically sets most parameters based on problem dimension:

| Parameter | Default | Formula |
|-----------|---------|---------|
| λ (population) | auto | 4 + floor(3 ln(n)) |
| μ (parents) | auto | λ / 2 |
| c_σ | auto | (μ_eff + 2) / (n + μ_eff + 5) |
| d_σ | auto | 1 + 2 max(0, √((μ_eff-1)/(n+1)) - 1) + c_σ |

## Convergence Criteria

CMA-ES converges when:
- `sigma` becomes very small (< 1e-12)
- Condition number of covariance matrix is too high
- No improvement for many generations

## Memory Usage

CMA-ES stores a full covariance matrix:

| Dimensions | Memory (approx) |
|------------|-----------------|
| 10 | ~1 KB |
| 100 | ~80 KB |
| 1000 | ~8 MB |
| 10000 | ~800 MB |

For high dimensions (>1000), consider alternatives.

## Best Practices

1. **Initial sigma**: Should cover ~1/3 of the search range
2. **Initial mean**: Start near expected optimum if known
3. **Bounds**: Use soft bounds (CMA-ES handles them gracefully)
4. **Restarts**: For multimodal problems, use multiple restarts

## See Also

- [CMA-ES Tutorial](../../tutorials/cmaes.md)
- [Choosing an Algorithm](../../how-to/choosing-algorithm.md)
