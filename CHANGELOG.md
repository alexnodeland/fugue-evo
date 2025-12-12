# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-12

### Added

- **Core Genetic Algorithm Framework**
  - `SimpleGA` builder pattern for easy algorithm configuration
  - Generational evolution with configurable operators
  - Elitism support for preserving best individuals

- **Genome Types**
  - `RealVector` for continuous optimization
  - `BitString` for binary/combinatorial problems
  - `Permutation` for ordering problems (TSP, scheduling)
  - `TreeGenome` for genetic programming
  - Unified `EvolutionaryGenome` trait abstraction

- **Selection Operators**
  - `TournamentSelection` with configurable tournament size
  - `RouletteWheelSelection` (fitness-proportionate)
  - `TruncationSelection` for steady-state evolution
  - `RankSelection` for rank-based selection
  - `BoltzmannSelection` with temperature parameter

- **Crossover Operators**
  - `SbxCrossover` (Simulated Binary Crossover) for real-valued genomes
  - `UniformCrossover` for bit strings
  - `SinglePointCrossover` and `TwoPointCrossover`
  - `OrderCrossover` (OX) for permutations
  - `SubtreeCrossover` for tree genomes

- **Mutation Operators**
  - `PolynomialMutation` for real-valued genomes
  - `GaussianMutation` with adaptive step sizes
  - `BitFlipMutation` for bit strings
  - `SwapMutation` and `InsertMutation` for permutations
  - `PointMutation` and `SubtreeMutation` for trees

- **Advanced Algorithms**
  - `CmaEs` (Covariance Matrix Adaptation Evolution Strategy)
  - `NSGA2` for multi-objective optimization with Pareto fronts
  - `IslandModel` for parallel evolution with migration

- **Fugue PPL Integration**
  - `to_trace()` and `from_trace()` for probabilistic programming interop
  - Trace-based evolutionary operators
  - Bayesian hyperparameter learning with `BetaPosterior`

- **Production Features**
  - Checkpointing with `CheckpointManager` (JSON, Binary, Compressed)
  - Convergence detection with configurable criteria
  - Evolution statistics tracking
  - Termination conditions (max generations, target fitness, stagnation)

- **Benchmark Functions**
  - `Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`, `Griewank`
  - `OneMax`, `LeadingOnes` for bit strings
  - `SymbolicRegression` for GP benchmarks

- **Examples**
  - `sphere_optimization.rs` - Basic continuous optimization
  - `rastrigin_benchmark.rs` - Multimodal function optimization
  - `cma_es_example.rs` - CMA-ES for Rosenbrock
  - `island_model.rs` - Parallel island model
  - `checkpointing.rs` - Save/restore evolution state
  - `symbolic_regression.rs` - Genetic programming
  - `hyperparameter_learning.rs` - Bayesian adaptation

- **Testing**
  - Comprehensive unit tests (370+ tests)
  - Property-based tests with proptest (21 tests)
