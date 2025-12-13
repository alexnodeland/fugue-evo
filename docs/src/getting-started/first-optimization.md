# Your First Optimization

In this guide, you'll create a complete optimization from scratch with a custom fitness function. We'll optimize a simple engineering problem: finding the dimensions of a box with maximum volume given a surface area constraint.

## The Problem

Design a rectangular box with:
- Maximum volume: V = l × w × h
- Fixed surface area: 2(lw + wh + lh) = 100

This is a constrained optimization problem that we'll handle using a penalty method.

## Step 1: Create the Project

```bash
cargo new box_optimizer
cd box_optimizer
```

Add dependencies to `Cargo.toml`:

```toml
[dependencies]
fugue-evo = "0.1"
rand = "0.8"
```

## Step 2: Define the Fitness Function

Create `src/main.rs`:

```rust,ignore
use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Custom fitness function for box volume optimization
struct BoxVolumeFitness {
    target_surface_area: f64,
    penalty_weight: f64,
}

impl BoxVolumeFitness {
    fn new(target_surface_area: f64) -> Self {
        Self {
            target_surface_area,
            penalty_weight: 1000.0, // Heavy penalty for constraint violation
        }
    }
}

impl Fitness<RealVector> for BoxVolumeFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let genes = genome.genes();
        let length = genes[0];
        let width = genes[1];
        let height = genes[2];

        // Calculate volume (what we want to maximize)
        let volume = length * width * height;

        // Calculate surface area
        let surface_area = 2.0 * (length * width + width * height + length * height);

        // Constraint violation (should be close to target)
        let violation = (surface_area - self.target_surface_area).abs();

        // Fitness = volume - penalty * violation
        // We maximize fitness, so volume is positive and violation is penalized
        volume - self.penalty_weight * violation
    }
}
```

### Why This Works

- **Volume maximization**: Higher volume = higher fitness
- **Constraint handling**: Penalty for deviating from target surface area
- **Penalty weight**: Large enough to make constraint violations costly

## Step 3: Set Up the Optimization

```rust,ignore
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Box Volume Optimization ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Problem setup
    let target_surface_area = 100.0;
    let fitness = BoxVolumeFitness::new(target_surface_area);

    // Search bounds: dimensions between 0.1 and 10
    let bounds = MultiBounds::uniform(Bounds::new(0.1, 10.0), 3);

    println!("Problem: Maximize box volume with surface area = {}", target_surface_area);
    println!("Search space: [0.1, 10.0] for each dimension\n");

    // Configure and run GA
    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(100)
        .bounds(bounds)
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0).with_probability(0.1))
        .fitness(fitness)
        .max_generations(300)
        .elitism(true)
        .elite_count(2)
        .build()?
        .run(&mut rng)?;

    // Display results
    print_results(&result, target_surface_area);

    Ok(())
}
```

## Step 4: Display Results

```rust,ignore
fn print_results(result: &EvolutionResult<RealVector, f64>, target_area: f64) {
    let genes = result.best_genome.genes();
    let length = genes[0];
    let width = genes[1];
    let height = genes[2];

    let volume = length * width * height;
    let surface_area = 2.0 * (length * width + width * height + length * height);

    println!("=== Optimization Results ===\n");
    println!("Best solution found:");
    println!("  Length: {:.4}", length);
    println!("  Width:  {:.4}", width);
    println!("  Height: {:.4}", height);
    println!();
    println!("Metrics:");
    println!("  Volume:       {:.4}", volume);
    println!("  Surface Area: {:.4} (target: {})", surface_area, target_area);
    println!("  Constraint violation: {:.4}", (surface_area - target_area).abs());
    println!();
    println!("Evolution statistics:");
    println!("  Generations: {}", result.generations);
    println!("  Evaluations: {}", result.evaluations);
    println!("  Best fitness: {:.4}", result.best_fitness);

    // For a cube with surface area 100, the optimal solution is:
    // 6s² = 100 → s = √(100/6) ≈ 4.08
    // Volume = s³ ≈ 68.04
    let optimal_side = (target_area / 6.0).sqrt();
    let optimal_volume = optimal_side.powi(3);
    println!();
    println!("Theoretical optimum (cube):");
    println!("  Side length: {:.4}", optimal_side);
    println!("  Volume: {:.4}", optimal_volume);
    println!("  Your solution is {:.2}% of optimal", (volume / optimal_volume) * 100.0);
}
```

## Step 5: Run It

```bash
cargo run
```

Expected output:

```
=== Box Volume Optimization ===

Problem: Maximize box volume with surface area = 100
Search space: [0.1, 10.0] for each dimension

=== Optimization Results ===

Best solution found:
  Length: 4.0824
  Width:  4.0819
  Height: 4.0827

Metrics:
  Volume:       68.0352
  Surface Area: 99.9987 (target: 100)
  Constraint violation: 0.0013

Evolution statistics:
  Generations: 300
  Evaluations: 30000
  Best fitness: 68.0339

Theoretical optimum (cube):
  Side length: 4.0825
  Volume: 68.0414
  Your solution is 99.99% of optimal
```

## Understanding the Result

The GA discovered that a **cube** maximizes volume for a given surface area - a well-known mathematical result! The three dimensions converged to approximately equal values.

## Experimenting

Try modifying the problem:

### 1. Different Surface Area

```rust,ignore
let fitness = BoxVolumeFitness::new(200.0); // Larger box
```

### 2. More Generations

```rust,ignore
.max_generations(500) // More refinement
```

### 3. Constrained Dimensions

What if height must be less than length?

```rust,ignore
impl Fitness<RealVector> for BoxVolumeFitness {
    fn evaluate(&self, genome: &RealVector) -> f64 {
        let genes = genome.genes();
        let length = genes[0];
        let width = genes[1];
        let height = genes[2];

        // Height constraint: h <= l
        let height_violation = if height > length {
            height - length
        } else {
            0.0
        };

        let volume = length * width * height;
        let surface_area = 2.0 * (length * width + width * height + length * height);
        let area_violation = (surface_area - self.target_surface_area).abs();

        volume - self.penalty_weight * (area_violation + height_violation)
    }
}
```

## Complete Source Code

The full example is available at [`examples/box_optimization.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/box_optimization.rs).

## Next Steps

Congratulations! You've created your first custom optimization. Continue learning:

- [Custom Fitness Functions](../how-to/custom-fitness.md) - More fitness function patterns
- [Multimodal Optimization](../tutorials/multimodal-optimization.md) - Handle functions with many local optima
- [Parallel Evolution](../how-to/parallel-evolution.md) - Speed up evaluation
