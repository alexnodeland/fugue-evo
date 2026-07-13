# fugue-evo

A broad evolutionary-computation library for Rust, with an optional probabilistic-programming bridge to [Fugue](https://github.com/fugue-ppl/fugue).

The default flagship algorithms (SimpleGA, CMA-ES, NSGA-II, Island Model, Evolution Strategy, EDA/UMDA, SteadyState) are standalone evolutionary computation: they use Fugue's `Trace` only as an address→value data container for the optional `to_trace`/`from_trace` round-trip, not for inference. The genuine "evolution as Bayesian inference over solution spaces" story — a tempered Sequential Monte Carlo pipeline over Fugue's `Model`/`Handler`/`factor` machinery — lives in the `fugue_integration` module (`EvolutionarySMC`/`EvolutionStep`/`BayesianAdaptiveGA`), demonstrated by `examples/bayesian_evolution.rs`. Reach for that module, not the default algorithms, when you want the PPL-powered inference path (EV-17).

## Features

- **Multiple Algorithms**: Simple GA, CMA-ES, NSGA-II, Island Model
- **Flexible Genomes**: Real-valued vectors, bit strings, permutations, and GP trees
- **Rich Operators**: SBX crossover, polynomial mutation, tournament selection, and more
- **Probabilistic Integration**: a genuine tempered Sequential Monte Carlo pipeline over Fugue's `Model`/`Handler`/`factor` machinery, targeting the Boltzmann/Gibbs posterior `π_β(x) ∝ p(x) · exp(β·f(x))` (see `examples/bayesian_evolution.rs`)
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
- `bayesian_evolution.rs` - Flagship end-to-end pipeline: tempered SMC over the Boltzmann posterior, plus the Bayesian adaptive GA

Run an example:

```bash
cargo run --example sphere_optimization
```

## Core Concepts

### Fitness as Likelihood

The `exp(f/T)` selection weight corresponds to Bayesian conditioning on fitness. In this crate that correspondence is realized concretely in two places: `BoltzmannSelection` (a standalone softmax-of-`f/T` selection operator), and the tempered-SMC path in `fugue_integration`, which targets the Boltzmann/Gibbs posterior `π_β(x) ∝ p(x)·exp(β·f(x))` using Fugue's `factor` machinery. The other default selection operators (tournament, roulette, rank) are ordinary EC and do not perform inference.

### Learnable Operators

Operator parameters (per-gene mutation probability, crossover probability) can optionally be tuned online by a Thompson-sampling multi-armed bandit: each candidate value is an arm with a conjugate `Beta` posterior over "did this arm's value improve the offspring", and the arm actually applied each generation is Thompson-sampled from those posteriors. Opt in with `SimpleGABuilder::adaptive_operators(ThompsonConfig)` and `SimpleGA::run_adaptive` (the default `run` path uses fixed operator parameters).

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

Beyond trace conversion, the `fugue_integration` module runs a genuine tempered
Sequential Monte Carlo sampler (`EvolutionarySMC`) over Fugue's
`Model`/`Handler`/`factor` machinery, targeting the Boltzmann/Gibbs posterior
`π_β(x) ∝ p(x) · exp(β·f(x))` from the prior (`β = 0`) to the full posterior
(`β = 1`), using trace-based mutation/crossover as `π_β`-invariant
Metropolis–Hastings rejuvenation moves. See `examples/bayesian_evolution.rs`
for the end-to-end pipeline.

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
fugue-ppl = { path = "../fugue", version = "0.1.0" }
```

Both crates live side by side under the same `fugue-ecosystem` parent
directory and are audited together (audit finding EV-30). This became the
committed default once `fugue`'s own 2026-07 audit remediation landed with a
green full-test gate; earlier in the remediation the sibling checkout was
frequently mid-edit and momentarily uncompilable, which is why the dependency
had been pinned to the published registry release until the sibling stabilized.
The `version = "0.1.0"` field is honored if `fugue-ppl` is ever resolved from
crates.io instead (e.g. the sibling checkout is absent).

To build against the published crates.io release rather than your local
`../fugue` checkout, replace the dependency with:

```toml
[dependencies]
fugue-ppl = "0.1.0"
```

Run `cargo check` after switching either way to confirm the resolved `fugue`
version actually satisfies `fugue-evo`'s usage.

## License

Licensed under the [MIT license](LICENSE).
