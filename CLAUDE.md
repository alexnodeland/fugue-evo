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

fugue-evo is a **two-layer** evolutionary-computation library:

1. **Classic EC layer** (no fugue dependency): all algorithms, operators, population machinery, checkpointing, WASM. Compiles with `--no-default-features --features std,parallel,checkpoint`.
2. **Inference layer** (`ppl` feature, default on; `src/inference/`): evolutionary algorithms as probabilistic programs. The prior over genomes is a fugue `Model<G>` (`GenomePrior`), fitness enters as `factor(β·f)`, and the Boltzmann posterior is sampled by fugue's own MH/SMC engines (`EvolutionChain`, `EvolutionSMC`). `ArithmeticGrammarPrior` does genetic programming over a probabilistic grammar — subtree mutation/crossover are generic trace moves.

### Core Abstractions

- `EvolutionaryGenome` (src/genome/traits.rs): the classic, fugue-free genome trait (decode/dimension/generate/distance).
- `TraceGenome` (src/genome/trace_genome.rs, `ppl`): extension trait adding `to_trace`/`from_trace`/`trace_prefix` — the boundary into the inference layer. `Permutation` uses a Lehmer-code (rank) encoding so single-site MH moves stay valid.
- `GenomePrior` (src/inference/prior.rs): a prior as a program — `fn model(&self) -> fugue::Model<G>` returning the decoded genome, plus `trace_of` (encode a genome under the prior's address scheme; grammar prior overrides it). Built-ins: `UniformBoxPrior`, `GaussianPrior`, `BitStringPrior`, `PermutationPrior`, `ArithmeticGrammarPrior`.
- `GenomeLikelihood` (src/inference/likelihood.rs): an observation program `p(data|g)` — observes, factors, latent nuisance sites (jointly inferred). `FactorFitness` is the black-box Gibbs-posterior adapter; `MemoizedFitness` caches expensive evaluations.
- Optimizer mode: `EvolutionSMC::anneal` tempers past beta=1. Multi-objective: `ParetoScalarization` (src/inference/pareto.rs) — scalarization weight as a latent site; posterior traces the Pareto front.
- Feature matrix: `classic` gates the EC toolkit; `ppl` gates inference; each builds without the other (`std,ppl` and `std,parallel,checkpoint,classic` are both CI-relevant configs).

Built-in genome types: `RealVector`, `BitString`, `Permutation`, `TreeGenome`

### Module Organization

- **algorithms/**: Evolution algorithms (SimpleGA, CMA-ES, NSGA-II, Island Model)
- **genome/**: Genome types and the `EvolutionaryGenome` trait
- **operators/**: Selection, crossover, mutation operators with trait bounds
- **fitness/**: `Fitness` trait and benchmark functions (Sphere, Rastrigin, Rosenbrock)
- **hyperparameter/**: Adaptive and Bayesian hyperparameter tuning (schedules, self-adaptive, conjugate priors)
- **inference/**: (`ppl`) priors as programs, `EvolutionModel` (Boltzmann target as a fugue program), `EvolutionChain` (MH), `EvolutionSMC` (tempered SMC + crossover kernel + log-evidence), `ArithmeticGrammarPrior` (grammar GP), effect handlers, trace operators
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

### Inference layer invariants

- The SMC path uses `EvolutionModel::smc_model()` (untempered `factor(f)`): β is applied exactly once by fugue's adaptive tempering. Never bake β into the SMC factor.
- All densities come from running/replaying the target program (`ScoreGivenTrace`); there is deliberately no hand-written density code in this crate.
- Regression anchors that must stay green: EV-16 (conjugate SMC posterior + analytic evidence), EV-52 (weighted trace = β·f), EV-90 (MH truncated-exponential mean), the dead-chain regressions (`test_bitstring_chain_moves`, `test_permutation_chain_moves`), and `test_symreg_recovers_known_expression`.
