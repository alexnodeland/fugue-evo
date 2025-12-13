# Custom Fitness Functions

This guide shows how to implement fitness functions for your optimization problems.

## Basic Pattern

Implement the `Fitness` trait for your fitness function:

```rust,ignore
use fugue_evo::prelude::*;

struct MyFitness {
    // Your fitness function parameters
}

impl Fitness<RealVector> for MyFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        // Your evaluation logic
        // Return higher values for better solutions
        let genes = genome.genes();
        // ...compute fitness...
        fitness_value
    }
}
```

## Minimization vs Maximization

Fugue-evo **maximizes** by default. For minimization problems, negate the objective:

```rust,ignore
impl Fitness<RealVector> for MinimizeProblem {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let objective = compute_objective(genome); // Value to minimize
        -objective // Negate for maximization
    }
}
```

## Constrained Optimization

### Penalty Method

Add penalties for constraint violations:

```rust,ignore
struct ConstrainedProblem {
    penalty_weight: f64,
}

impl Fitness<RealVector> for ConstrainedProblem {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let genes = genome.genes();

        // Objective to maximize
        let objective = compute_objective(genes);

        // Constraints (should be >= 0 for feasible solutions)
        let g1 = genes[0] + genes[1] - 10.0; // x0 + x1 <= 10
        let g2 = genes[0] * genes[1] - 5.0;  // x0 * x1 >= 5

        // Penalty for violations
        let violation1 = g1.max(0.0);           // Penalize if > 0
        let violation2 = (-g2).max(0.0);        // Penalize if < 0

        let total_penalty = violation1.powi(2) + violation2.powi(2);

        objective - self.penalty_weight * total_penalty
    }
}
```

**Choosing penalty weight:**
- Too small: Constraints ignored
- Too large: Search biased toward feasibility over quality
- Typical: 100-10000 depending on objective scale

### Death Penalty

Reject infeasible solutions entirely:

```rust,ignore
impl Fitness<RealVector> for StrictConstraints {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let genes = genome.genes();

        // Check constraints
        if !self.is_feasible(genes) {
            return f64::NEG_INFINITY; // Or a very low value
        }

        compute_objective(genes)
    }

    fn is_feasible(&self, genes: &[f64]) -> bool {
        genes[0] + genes[1] <= 10.0 && genes[0] * genes[1] >= 5.0
    }
}
```

## External Simulations

When fitness requires running external code:

```rust,ignore
struct SimulationFitness {
    simulator_path: PathBuf,
}

impl Fitness<RealVector> for SimulationFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        // Write parameters to file
        let params_file = write_params(genome.genes());

        // Run external simulation
        let output = std::process::Command::new(&self.simulator_path)
            .arg(&params_file)
            .output()
            .expect("Failed to run simulator");

        // Parse result
        let result: f64 = parse_output(&output.stdout);

        // Clean up
        std::fs::remove_file(params_file).ok();

        result
    }
}
```

## Data-Driven Fitness

When fitness is computed from data:

```rust,ignore
struct RegressionFitness {
    x_data: Vec<f64>,
    y_data: Vec<f64>,
}

impl RegressionFitness {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len());
        Self { x_data: x, y_data: y }
    }
}

impl Fitness<RealVector> for RegressionFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let genes = genome.genes();
        let a = genes[0];
        let b = genes[1];

        // Model: y = a*x + b
        let mse: f64 = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| {
                let predicted = a * x + b;
                (y - predicted).powi(2)
            })
            .sum::<f64>() / self.x_data.len() as f64;

        -mse // Minimize MSE
    }
}
```

## Multi-Objective Fitness

For NSGA-II, return multiple objectives:

```rust,ignore
struct MultiObjective;

impl Fitness<RealVector> for MultiObjective {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        let genes = genome.genes();

        // Two conflicting objectives
        let obj1 = genes.iter().sum::<f64>();      // Maximize sum
        let obj2 = -genes.iter().product::<f64>(); // Maximize product

        ParetoFitness::new(vec![obj1, obj2])
    }
}
```

## Caching Fitness

For expensive evaluations, cache results:

```rust,ignore
use std::collections::HashMap;
use std::sync::RwLock;

struct CachedFitness<F> {
    inner: F,
    cache: RwLock<HashMap<Vec<u64>, f64>>,
}

impl<F: Fitness<RealVector, Value = f64>> CachedFitness<F> {
    fn cache_key(genome: &RealVector) -> Vec<u64> {
        genome.genes().iter()
            .map(|f| f.to_bits())
            .collect()
    }
}

impl<F: Fitness<RealVector, Value = f64>> Fitness<RealVector> for CachedFitness<F> {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let key = Self::cache_key(genome);

        // Check cache
        if let Ok(cache) = self.cache.read() {
            if let Some(&cached) = cache.get(&key) {
                return cached;
            }
        }

        // Compute and cache
        let result = self.inner.evaluate(genome);

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key, result);
        }

        result
    }
}
```

## Fitness for Different Genome Types

### BitString

```rust,ignore
impl Fitness<BitString> for KnapsackFitness {
    type Value = f64;

    fn evaluate(&self, genome: &BitString) -> f64 {
        let mut total_value = 0.0;
        let mut total_weight = 0.0;

        for (i, bit) in genome.bits().iter().enumerate() {
            if *bit {
                total_value += self.values[i];
                total_weight += self.weights[i];
            }
        }

        if total_weight > self.capacity {
            0.0 // Infeasible
        } else {
            total_value
        }
    }
}
```

### Permutation

```rust,ignore
impl Fitness<Permutation> for TSPFitness {
    type Value = f64;

    fn evaluate(&self, genome: &Permutation) -> f64 {
        let order = genome.as_slice();
        let mut total_distance = 0.0;

        for i in 0..order.len() {
            let from = order[i];
            let to = order[(i + 1) % order.len()];
            total_distance += self.distance_matrix[from][to];
        }

        -total_distance // Minimize distance
    }
}
```

## Testing Fitness Functions

Always test your fitness function:

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_optimum() {
        let fitness = MyFitness::new();
        let optimum = RealVector::new(vec![0.0, 0.0, 0.0]);
        assert!((fitness.evaluate(&optimum) - expected_value).abs() < 1e-6);
    }

    #[test]
    fn test_fitness_ordering() {
        let fitness = MyFitness::new();
        let good = RealVector::new(vec![0.1, 0.1, 0.1]);
        let bad = RealVector::new(vec![5.0, 5.0, 5.0]);
        assert!(fitness.evaluate(&good) > fitness.evaluate(&bad));
    }

    #[test]
    fn test_constraints() {
        let fitness = ConstrainedProblem::new();
        let feasible = RealVector::new(vec![3.0, 2.0]);
        let infeasible = RealVector::new(vec![10.0, 10.0]);
        assert!(fitness.evaluate(&feasible) > fitness.evaluate(&infeasible));
    }
}
```

## Next Steps

- [Custom Genome Types](./custom-genome.md) - Create problem-specific representations
- [Parallel Evolution](./parallel-evolution.md) - Speed up expensive fitness evaluations
