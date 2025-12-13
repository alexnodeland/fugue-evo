# Introduction

**fugue-evo** is a probabilistic genetic algorithm library for Rust that treats evolution as Bayesian inference over solution spaces. It provides a flexible, type-safe framework for optimization problems ranging from simple continuous optimization to complex genetic programming.

## Why Fugue-Evo?

Fugue-evo differentiates itself from traditional GA libraries through several key design principles:

### Probabilistic Foundations

Traditional genetic algorithms use heuristic operators, but fugue-evo views evolution through a probabilistic lens:

- **Fitness as Likelihood**: Selection pressure maps directly to Bayesian conditioning
- **Learnable Operators**: Automatic inference of optimal crossover, mutation, and selection hyperparameters using conjugate priors
- **Trace-Based Evolution**: Deep integration with the [fugue-ppl](https://github.com/fugue-ppl/fugue) probabilistic programming library enables novel trace-based genetic operators

### Type Safety

Rust's type system ensures correctness at compile time:

- **Trait-Based Abstraction**: The `EvolutionaryGenome` trait provides a flexible foundation for any genome type
- **Builder Patterns**: Algorithm configuration uses type-safe builders that catch misconfigurations at compile time
- **Generic Operators**: Selection, crossover, and mutation operators work across genome types with proper type constraints

### Production Ready

Built for real-world use:

- **Checkpointing**: Save and restore evolution state for long-running optimizations
- **Parallelism**: Optional Rayon-based parallel evaluation
- **WASM Support**: Run optimizations in the browser with WebAssembly bindings
- **Interactive Evolution**: Human-in-the-loop optimization with multiple evaluation modes

## Algorithms

Fugue-evo includes several optimization algorithms:

| Algorithm | Best For | Key Features |
|-----------|----------|--------------|
| **SimpleGA** | General-purpose optimization | Flexible operators, elitism, parallelism |
| **CMA-ES** | Continuous optimization | Covariance adaptation, state-of-the-art performance |
| **NSGA-II** | Multi-objective optimization | Pareto ranking, crowding distance |
| **Island Model** | Escaping local optima | Parallel populations, migration |
| **EDA/UMDA** | Probabilistic model building | Distribution estimation |
| **Interactive GA** | Subjective optimization | Human feedback integration |

## Genome Types

Built-in genome representations for different problem domains:

- **RealVector**: Continuous optimization (function minimization)
- **BitString**: Combinatorial optimization (knapsack, feature selection)
- **Permutation**: Ordering problems (TSP, scheduling)
- **TreeGenome**: Genetic programming (symbolic regression)

## Getting Started

Ready to dive in? Head to the [Installation](./getting-started/installation.md) guide to add fugue-evo to your project, then follow the [Quick Start](./getting-started/quickstart.md) to run your first optimization.

For a deeper understanding of the library's design, explore the [Core Concepts](./getting-started/concepts.md) section.

## Example

Here's a taste of what optimization looks like with fugue-evo:

```rust,ignore
use fugue_evo::prelude::*;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Define search bounds: 10 dimensions in [-5.12, 5.12]
    let bounds = MultiBounds::symmetric(5.12, 10);

    // Run optimization
    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(100)
        .bounds(bounds)
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(Sphere::new(10))
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    println!("Best fitness: {:.6}", result.best_fitness);
    Ok(())
}
```

## Documentation Structure

This documentation is organized into several sections:

- **Getting Started**: Installation, concepts, and first steps
- **Tutorials**: Step-by-step walkthroughs of example problems
- **How-To Guides**: Task-oriented guides for specific use cases
- **Reference**: Detailed API documentation for algorithms, genomes, and operators
- **Architecture**: Design philosophy and system internals
