# Changelog

All notable changes to fugue-evo will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive mdbook documentation
- Getting Started guide
- Tutorials for all major features
- How-To guides for common tasks
- Reference documentation for all modules
- Architecture documentation

## [0.1.0] - Initial Release

### Added

#### Algorithms
- SimpleGA: General-purpose genetic algorithm with builder pattern
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- NSGA-II: Non-dominated Sorting Genetic Algorithm II for multi-objective optimization
- Island Model: Parallel evolution with migration
- EDA/UMDA: Estimation of Distribution Algorithm
- Steady-State GA: Non-generational evolution
- Evolution Strategy: (μ+λ) and (μ,λ) strategies
- Interactive GA: Human-in-the-loop optimization

#### Genome Types
- RealVector: Continuous optimization
- BitString: Binary optimization
- Permutation: Ordering problems
- TreeGenome: Genetic programming
- DynamicRealVector: Variable-length real vectors
- Composite genomes

#### Operators
- Selection: Tournament, Roulette Wheel, Rank-based
- Crossover: SBX, Uniform, One-point, Two-point, Order (OX), PMX
- Mutation: Polynomial, Gaussian, Bit-flip, Swap, Inversion

#### Fitness
- Fitness trait with generic value types
- ParetoFitness for multi-objective
- Benchmark functions: Sphere, Rastrigin, Rosenbrock, Ackley, ZDT suite

#### Features
- Checkpointing and recovery
- Parallel evaluation via Rayon
- Termination criteria combinators
- Hyperparameter adaptation (schedules, Bayesian learning)
- Fugue PPL integration for trace-based operators

#### WASM Support
- fugue-evo-wasm package for browser/Node.js
- JavaScript bindings via wasm-bindgen

### Dependencies
- nalgebra 0.33 for linear algebra
- rand 0.8 for random number generation
- rayon 1.10 for parallelism
- serde for serialization
- fugue-ppl 0.1.0 for probabilistic programming integration

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | TBD | Initial release |

## Upgrading

### From Pre-release to 0.1.0

If you were using a pre-release version:

1. Update `Cargo.toml`:
   ```toml
   fugue-evo = "0.1"
   ```

2. Check for API changes in builders
3. Update any custom operators to new trait signatures

## Deprecations

None currently.

## Security

No security advisories.
