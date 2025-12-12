# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a single test by name
cargo test test_name

# Run tests in a specific module
cargo test module_name::

# Run tests with output
cargo test -- --nocapture

# Run property-based tests
cargo test --test property_tests

# Check for warnings and lint issues
cargo clippy

# Format code
cargo fmt

# Run an example
cargo run --example sphere_optimization
```

## Architecture

fugue-evo is a probabilistic genetic algorithm library that treats evolution as Bayesian inference. It integrates with `fugue-ppl` (a probabilistic programming library) for trace-based evolutionary operators.

### Core Abstraction: `EvolutionaryGenome` Trait

The central abstraction is `EvolutionaryGenome` (src/genome/traits.rs), which requires genomes to convert to/from Fugue traces. This enables:
- **Trace-based mutation**: Selective resampling of addresses
- **Trace-based crossover**: Merging parent traces with constraints

Built-in genome types: `RealVector`, `BitString`, `Permutation`, `TreeGenome`

### Module Organization

- **algorithms/**: Evolution algorithms (SimpleGA, CMA-ES, NSGA-II, Island Model)
- **genome/**: Genome types and the `EvolutionaryGenome` trait
- **operators/**: Selection, crossover, mutation operators with trait bounds
- **fitness/**: `Fitness` trait and benchmark functions (Sphere, Rastrigin, Rosenbrock)
- **hyperparameter/**: Adaptive and Bayesian hyperparameter tuning (schedules, self-adaptive, conjugate priors)
- **fugue_integration/**: Trace operators and effect handlers for probabilistic evolution
- **checkpoint/**: State serialization for pausing/resuming evolution
- **termination/**: Convergence criteria (max generations, fitness threshold, stagnation)

### Type Patterns

Algorithms use builder patterns with extensive generics. Example:
```rust
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .bounds(bounds)
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(fitness)
    .max_generations(200)
    .build()?
```

### Operator Traits

Operators implement traits like `SelectionOperator`, `CrossoverOperator`, `MutationOperator`. Bounded variants (`BoundedCrossoverOperator`, `BoundedMutationOperator`) receive bounds information for constraint handling.

### Fugue Integration

Genomes convert to `fugue::Trace` objects where genes are stored at indexed addresses (e.g., `addr!("gene", 0)`). This enables probabilistic interpretation of genetic operators through the `fugue_integration` module's effect handlers.
