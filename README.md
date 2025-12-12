# fugue-evo

A Probabilistic Genetic Algorithm Library for Rust.

This library implements genetic algorithms through the lens of probabilistic programming, treating evolution as Bayesian inference over solution spaces.

## Features

- **Multiple Algorithms**: Simple GA, CMA-ES, NSGA-II, Island Model
- **Flexible Genomes**: Real-valued vectors, bit strings, permutations, and GP trees
- **Rich Operators**: SBX crossover, polynomial mutation, tournament selection, and more
- **Probabilistic Integration**: Fugue PPL integration for trace-based evolutionary operators
- **Bayesian Learning**: Online hyperparameter adaptation using conjugate priors
- **Production Ready**: Checkpointing, convergence detection, parallel evaluation

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fugue-evo = "0.1"
```

Basic optimization example:

```rust
use fugue_evo::prelude::*;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Optimize the 10-D Sphere function
    let fitness = Sphere::new(10);
    let bounds = MultiBounds::symmetric(5.12, 10);

    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(100)
        .bounds(bounds)
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(fitness)
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    println!("Best fitness: {}", result.best_fitness);
    Ok(())
}
```

## Examples

The `examples/` directory contains demonstrations of various features:

- `sphere_optimization.rs` - Basic continuous optimization
- `rastrigin_benchmark.rs` - Multimodal function optimization
- `cma_es_example.rs` - CMA-ES for Rosenbrock function
- `island_model.rs` - Parallel island model evolution
- `checkpointing.rs` - Save and restore evolution state
- `symbolic_regression.rs` - Genetic programming with tree genomes
- `hyperparameter_learning.rs` - Bayesian hyperparameter adaptation

Run an example:

```bash
cargo run --example sphere_optimization
```

## Core Concepts

### Fitness as Likelihood

Selection pressure maps directly to Bayesian conditioning. Higher fitness increases the probability of selection, analogous to likelihood weighting in probabilistic inference.

### Learnable Operators

The library supports automatic inference of optimal crossover, mutation, and selection hyperparameters using Bayesian conjugate priors that update online during evolution.

### Flexible Genomes

The `EvolutionaryGenome` trait provides a unified abstraction supporting:
- `RealVector` - Continuous optimization
- `BitString` - Binary/combinatorial problems
- `Permutation` - Ordering problems (TSP, scheduling)
- `TreeGenome` - Genetic programming

### Fugue Integration

Genomes can be converted to Fugue PPL traces for probabilistic operations:

```rust
let trace = genome.to_trace();
let recovered = RealVector::from_trace(&trace)?;
```

## Algorithms

### Simple GA

Standard generational genetic algorithm with configurable operators.

### CMA-ES

Covariance Matrix Adaptation Evolution Strategy for continuous optimization. Adapts the full covariance matrix of a multivariate normal distribution.

### NSGA-II

Non-dominated Sorting Genetic Algorithm II for multi-objective optimization. Finds Pareto-optimal solutions.

### Island Model

Parallel evolution with multiple subpopulations and periodic migration. Supports ring, fully-connected, and star topologies.

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
