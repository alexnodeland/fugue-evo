<div align="center">

<img src="assets/fugue-evo-mark.png" alt="Fugue Evo" width="140" height="140">

# Fugue Evo

**Two layers: classical evolutionary algorithms (standalone), and evolutionary inference — evolutionary algorithms *as* probabilistic programs (tempered SMC in trace space, built on [Fugue](https://github.com/alexnodeland/fugue))**

*Populations hunting real landscapes, live in your browser: every figure in the docs at [evo.fugue.run](https://evo.fugue.run) runs the actual crate, compiled to WASM.*

[![Crates.io](https://img.shields.io/crates/v/fugue-evo.svg)](https://crates.io/crates/fugue-evo)
[![Dev Docs](https://docs.rs/fugue-evo/badge.svg)](https://docs.rs/fugue-evo)
[![User Docs](https://img.shields.io/badge/guides-evo.fugue.run-blue)](https://evo.fugue.run)
[![CI](https://github.com/alexnodeland/fugue-evo/actions/workflows/ci.yml/badge.svg)](https://github.com/alexnodeland/fugue-evo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

An evolutionary-computation library for Rust with an architectural split that keeps both halves honest:

- **Classic EC (no fugue dependency).** SimpleGA, CMA-ES, NSGA-II, Island Model, Evolution Strategy, EDA/UMDA, SteadyState, all operators, checkpointing, and the WASM surface. Build with `--no-default-features --features std,parallel,checkpoint` and there is no probabilistic-programming dependency at all.
- **Evolutionary inference (`ppl` feature, on by default): evolutionary algorithms *as* probabilistic programs.** The prior over genomes is a user-written fugue `Model<G>` (a `GenomePrior`), fitness enters as `factor(β·f(x))`, so the Boltzmann posterior `π_β(x) ∝ p(x)·exp(β·f(x))` **is a fugue program** — and every sampler is fugue's own inference machinery: `EvolutionChain` (typed single-site MH), `EvolutionSMC` (adaptive tempered SMC with a population-coupled crossover kernel and an unbiased log-evidence estimate), and `ArithmeticGrammarPrior` (genetic programming over a probabilistic grammar, where subtree mutation and crossover are generic trace moves). See `examples/symbolic_regression_inference.rs` — symbolic regression as exact Bayesian inference.

## Features

- **Multiple Algorithms**: Simple GA, CMA-ES, NSGA-II, Island Model
- **Flexible Genomes**: Real-valued vectors, bit strings, permutations, and GP trees
- **Rich Operators**: SBX crossover, polynomial mutation, tournament selection, and more
- **Evolutionary inference**: priors as programs (`GenomePrior` → `fugue::Model<G>`), adaptive tempered SMC over the Boltzmann posterior with log-evidence, typed-proposal MH, and grammar-based GP as exact inference (`examples/bayesian_evolution.rs`, `examples/symbolic_regression_inference.rs`)
- **Bayesian Learning**: opt-in online hyperparameter tuning via a Thompson-sampling multi-armed bandit over conjugate `Beta`/`Gamma` posteriors (`SimpleGABuilder::adaptive_operators` + `run_adaptive`; see `examples/hyperparameter_learning.rs`)
- **Production Ready**: checkpointing with bit-identical resume (ChaCha RNG family), convergence detection, parallel evaluation

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

    // `real_valued()` pins the genome/fitness types (no turbofish) and
    // pre-installs tournament selection, SBX crossover, and polynomial mutation
    // as overridable defaults.
    let result = SimpleGABuilder::real_valued()
        .population_size(100)
        .bounds(bounds)
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
- `checkpointing.rs` - Save and restore evolution state with bit-identical resume
- `symbolic_regression.rs` - Genetic programming with tree genomes
- `hyperparameter_learning.rs` - Opt-in Thompson-sampling operator-parameter tuning
- `bayesian_evolution.rs` - End-to-end inference pipeline: tempered SMC over the Boltzmann posterior, MH chain, plus the Bayesian adaptive GA
- `symbolic_regression_inference.rs` - **Flagship**: symbolic regression as exact Bayesian inference over a probabilistic grammar (subtree moves as generic trace machinery, model comparison by Bayes factor)

Run an example:

```bash
cargo run --example sphere_optimization
```

## Documentation

- **[User Guide](https://evo.fugue.run/)** - Tutorials, how-to guides, and reference, with live WASM-backed figures throughout
- **[Playground](https://evo.fugue.run/playground.html)** - Drive all five algorithms in the browser: SimpleGA, CMA-ES, NSGA-II, islands, UMDA
- **[API Reference](https://docs.rs/fugue-evo)** - Complete API documentation

## Core Concepts

### Fitness as Likelihood

The `exp(f/T)` selection weight corresponds to Bayesian conditioning on fitness. In this crate that correspondence is realized concretely in two places: `BoltzmannSelection` (a standalone softmax-of-`f/T` selection operator in the classic layer), and the `inference` module, where the Boltzmann/Gibbs posterior `π_β(x) ∝ p(x)·exp(β·f(x))` is assembled as a literal fugue program (`prior.model().bind(|g| factor(β·f(g)))`) and sampled by fugue's MH and tempered-SMC engines. The other default selection operators (tournament, roulette, rank) are ordinary EC and do not perform inference.

### Learnable Operators

Operator parameters (per-gene mutation probability, crossover probability) can optionally be tuned online by a Thompson-sampling multi-armed bandit: each candidate value is an arm with a conjugate `Beta` posterior over "did this arm's value improve the offspring", and the arm actually applied each generation is Thompson-sampled from those posteriors. Opt in with `SimpleGABuilder::adaptive_operators(ThompsonConfig)` and `SimpleGA::run_adaptive` (the default `run` path uses fixed operator parameters).

### Flexible Genomes

The `EvolutionaryGenome` trait provides a unified abstraction supporting:
- `RealVector` - Continuous optimization
- `BitString` - Binary/combinatorial problems
- `Permutation` - Ordering problems (TSP, scheduling)
- `TreeGenome` - Genetic programming

### Evolution as inference (`ppl`)

Genomes implementing the `TraceGenome` extension trait can be encoded as fugue
traces (`use fugue_evo::genome::trace_genome::TraceGenome`):

```rust
let trace = genome.to_trace();
let recovered = RealVector::from_trace(&trace)?;
```

The real story is the `inference` module: the prior is any fugue program
returning the decoded genome, fitness is a likelihood factor, and the
posterior is sampled by fugue's engines —

```rust
let model = EvolutionModel::new(GaussianPrior::new(0.0, 2.0, DIM), fitness);
let posterior = EvolutionSMC::run(&mut rng, &model, EvoSmcConfig::default());
// posterior.weighted_mean(0), posterior.log_evidence, posterior.best(..)
```

Adaptive ESS-driven tempering from the prior (β = 0) to the posterior
(β = 1), typed single-site MH rejuvenation (all site kinds move, including
bits and permutation ranks), a population-coupled crossover kernel, decode-
replay genome recovery, and an unbiased log-evidence estimate for Bayesian
model comparison. See `examples/bayesian_evolution.rs` and the flagship
`examples/symbolic_regression_inference.rs`.

## Algorithms

### Simple GA

Standard generational genetic algorithm with configurable operators.

### CMA-ES

Covariance Matrix Adaptation Evolution Strategy for continuous optimization. Adapts the full covariance matrix of a multivariate normal distribution.

### NSGA-II

Non-dominated Sorting Genetic Algorithm II for multi-objective optimization. Finds Pareto-optimal solutions.

### Island Model

Parallel evolution with multiple subpopulations and periodic migration. Supports ring, fully-connected, and star topologies.

## Development

fugue-evo depends on the co-developed sibling `fugue` crate via a path
dependency, so its probabilistic-programming bridge is built and tested
against the actual co-developed source rather than a registry release the two
crates were never exercised against together:

```toml
[dependencies]
fugue-ppl = { path = "../fugue", version = "0.2.1", optional = true }
```

Both crates live side by side under the same `fugue-ecosystem` parent
directory and are audited together (audit finding EV-30). This became the
committed default once `fugue`'s own 2026-07 audit remediation landed with a
green full-test gate; earlier in the remediation the sibling checkout was
frequently mid-edit and momentarily uncompilable, which is why the dependency
had been pinned to the published registry release until the sibling stabilized.
The `version = "0.2.0"` field is honored if `fugue-ppl` is ever resolved from
crates.io instead (e.g. the sibling checkout is absent).

To build against the published crates.io release rather than your local
`../fugue` checkout, replace the dependency with:

```toml
[dependencies]
fugue-ppl = "0.2.0"
```

Run `cargo check` after switching either way to confirm the resolved `fugue`
version actually satisfies `fugue-evo`'s usage.

## License

Licensed under the [MIT license](LICENSE).
