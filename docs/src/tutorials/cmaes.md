# CMA-ES Tutorial

**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** is one of the most powerful algorithms for continuous optimization, especially for non-separable and ill-conditioned problems.

## When to Use CMA-ES

**Ideal for:**
- Non-separable problems (variables are correlated)
- Ill-conditioned problems (different scaling per variable)
- Medium-dimensional problems (10-100 variables)
- When you want automatic parameter adaptation

**Not ideal for:**
- Very high-dimensional problems (n > 1000)
- Discrete or combinatorial problems
- Problems requiring custom genetic operators

## How CMA-ES Works

CMA-ES maintains a multivariate normal distribution over the search space and iteratively:

1. **Sample** new solutions from the distribution
2. **Evaluate** and rank solutions by fitness
3. **Update** the distribution mean toward better solutions
4. **Adapt** the covariance matrix to learn variable correlations
5. **Adjust** the step size (sigma) based on progress

The covariance matrix captures the shape of the fitness landscape, allowing efficient search even when variables are correlated.

## Complete Example

```rust,ignore
{{#include ../../../examples/cma_es_example.rs}}
```

> **Source**: [`examples/cma_es_example.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/cma_es_example.rs)

## Running the Example

```bash
cargo run --example cma_es_example
```

## Key Components

### Initialization

```rust,ignore
let initial_mean = vec![0.0; DIM];  // Starting point
let initial_sigma = 0.5;             // Initial step size
let bounds = MultiBounds::symmetric(5.0, DIM);

let mut cmaes = CmaEs::new(initial_mean, initial_sigma)
    .with_bounds(bounds);
```

**Parameters:**
- `initial_mean`: Starting center of the search distribution
- `initial_sigma`: Initial standard deviation (step size)
- `bounds`: Optional constraints on the search space

### Fitness Function

CMA-ES uses a different fitness trait than GA:

```rust,ignore
struct RosenbrockCmaEs { dim: usize }

impl CmaEsFitness for RosenbrockCmaEs {
    fn evaluate(&self, x: &RealVector) -> f64 {
        // Return value to MINIMIZE (not maximize!)
        let genes = x.genes();
        let mut sum = 0.0;
        for i in 0..self.dim - 1 {
            let term1 = genes[i + 1] - genes[i] * genes[i];
            let term2 = 1.0 - genes[i];
            sum += 100.0 * term1 * term1 + term2 * term2;
        }
        sum
    }
}
```

**Important**: CMA-ES **minimizes** by default (unlike GA which maximizes).

### Running Optimization

```rust,ignore
let best = cmaes.run_generations(&fitness, 1000, &mut rng)?;

println!("Best fitness: {:.10}", best.fitness_value());
println!("Final sigma: {:.6}", cmaes.state.sigma);
```

## The Rosenbrock Function

The example optimizes the **Rosenbrock function**:

```text
f(x) = Σ[100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]
```

**Properties:**
- Global minimum: f(1,1,...,1) = 0
- Curved valley: The optimum lies in a narrow, curved valley
- Non-separable: Variables are strongly coupled
- Tests algorithm's ability to learn correlations

CMA-ES excels at Rosenbrock because it learns the valley's shape through covariance adaptation.

## Understanding CMA-ES Output

### Sigma (Step Size)

```rust,ignore
println!("Final sigma: {:.6}", cmaes.state.sigma);
```

- **High sigma**: Still exploring, large steps
- **Low sigma**: Converging, fine-tuning
- **Oscillating sigma**: May indicate problem with scale

### Generations vs Evaluations

```rust,ignore
println!("Generations: {}", cmaes.state.generation);
println!("Evaluations: {}", cmaes.state.evaluations);
```

CMA-ES typically samples λ individuals per generation:
- Default λ ≈ 4 + 3ln(n) for n dimensions
- Total evaluations ≈ generations × λ

## Tuning CMA-ES

### Initial Step Size (Sigma)

```rust,ignore
let initial_sigma = 0.5;
```

**Guidelines:**
- Should cover roughly 1/3 of the search space
- Too small: Slow convergence, may miss global optimum
- Too large: Wasted evaluations exploring infeasible regions

For bounds [-5, 5], sigma = (5 - (-5)) / 6 ≈ 1.67 or smaller.

### Initial Mean

```rust,ignore
let initial_mean = vec![0.0; DIM];
```

If you have prior knowledge about good regions, start there:

```rust,ignore
let initial_mean = vec![1.0; DIM];  // Near expected optimum
```

### Population Size

Default population is typically sufficient, but can be increased for multimodal problems:

```rust,ignore
let mut cmaes = CmaEs::new(initial_mean, initial_sigma)
    .with_population_size(100);  // More exploration
```

## Comparison with GA

| Aspect | CMA-ES | Simple GA |
|--------|--------|-----------|
| Operators | Automatic adaptation | Manual configuration |
| Correlations | Learns automatically | Requires special operators |
| Evaluations | Usually fewer | Usually more |
| Flexibility | Fixed framework | Highly customizable |
| Memory | O(n²) | O(population × n) |
| Best for | Non-separable continuous | General purpose |

## Advanced Usage

### Early Stopping

```rust,ignore
for gen in 0..max_generations {
    cmaes.step(&fitness, &mut rng)?;

    // Check for convergence
    if cmaes.state.sigma < 1e-8 {
        println!("Converged at generation {}", gen);
        break;
    }
}
```

### Accessing Distribution Parameters

```rust,ignore
let mean = &cmaes.state.mean;  // Current distribution center
let sigma = cmaes.state.sigma;  // Current step size
let generation = cmaes.state.generation;
```

## Exercises

1. **Compare with GA**: Run both CMA-ES and SimpleGA on Rosenbrock, compare evaluations needed
2. **Step size study**: Try initial sigma values of 0.1, 0.5, 2.0, 5.0
3. **Higher dimensions**: Increase DIM to 20, 50 and observe scaling

## Next Steps

- [Island Model Tutorial](./island-model.md) - Parallel populations for multimodal problems
- [Multi-Objective Optimization](../how-to/multi-objective.md) - When you have multiple objectives
