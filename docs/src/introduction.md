<div class="fugue-hero">

<div class="fugue-mark fugue-mark-evo"><span>e</span></div>

# Fugue Evo

<p class="fugue-hero-tag">Evolution as Bayesian inference — a probabilistic, type-safe evolutionary computation library for Rust.</p>

<div class="fugue-badge-row">
<a href="https://crates.io/crates/fugue-evo"><img src="https://img.shields.io/crates/v/fugue-evo.svg" alt="Crates.io"></a>
<a href="https://docs.rs/fugue-evo"><img src="https://docs.rs/fugue-evo/badge.svg" alt="Dev Docs"></a>
<a href="https://github.com/alexnodeland/fugue-evo/actions/workflows/ci.yml"><img src="https://github.com/alexnodeland/fugue-evo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

</div>

<div class="fugue-explorable fv-inline" data-viz="mini-convergence" data-landscape="rastrigin" data-seed="11" data-caption="This page is alive — a real genetic algorithm, compiled to WASM, is optimizing Rastrigin right now."></div>

<div class="fugue-cards">
<a class="fugue-card" href="./getting-started/installation.html"><span class="fugue-card-title">Getting Started</span><span class="fugue-card-desc">Install fugue-evo and run your first optimization in minutes.</span></a>
<a class="fugue-card" href="./tutorials/continuous-optimization.html"><span class="fugue-card-title">Tutorials</span><span class="fugue-card-desc">Watch a population hunt a landscape, step by step — every figure runs the real crate.</span></a>
<a class="fugue-card" href="./playground.html"><span class="fugue-card-title">Playground</span><span class="fugue-card-desc">Drive all five algorithms live in the browser: GA, CMA-ES, NSGA-II, islands, UMDA.</span></a>
<a class="fugue-card" href="./reference/algorithms.html"><span class="fugue-card-title">Reference</span><span class="fugue-card-desc">Algorithms, genome types, operators, fitness, and termination criteria.</span></a>
<a class="fugue-card" href="./architecture/philosophy.html"><span class="fugue-card-title">Architecture</span><span class="fugue-card-desc">Design philosophy and system internals — evolution through a probabilistic lens.</span></a>
<a class="fugue-card" href="./api/fugue_evo/index.html"><span class="fugue-card-title">API Reference</span><span class="fugue-card-desc">Complete rustdoc for the <code>fugue-evo</code> crate.</span></a>
</div>

<a class="fugue-eco" href="https://fugue.run"><span class="fugue-card-title">Fugue <span class="fugue-brand-arrow">↗</span></span><span class="fugue-card-desc">The probabilistic programming library underneath — compose models in direct style and run MH, HMC, and SMC, with its own interactive explorables.</span></a>

## Why Fugue-Evo?

Fugue-evo differentiates itself from traditional GA libraries through several key design principles:

### Probabilistic Foundations

Traditional genetic algorithms use heuristic operators, but fugue-evo views evolution through a probabilistic lens:

- **Fitness as Likelihood**: Selection pressure maps directly to Bayesian conditioning
- **Learnable Operators**: Automatic inference of optimal crossover, mutation, and selection hyperparameters using conjugate priors
- **Trace-Based Evolution**: Deep integration with the [fugue-ppl](https://fugue.run) probabilistic programming library enables novel trace-based genetic operators

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

## Getting Started

Ready to dive in? Head to the [Installation](./getting-started/installation.md) guide to add fugue-evo to your project, then follow the [Quick Start](./getting-started/quickstart.md) to run your first optimization.

For a deeper understanding of the library's design, explore the [Core Concepts](./getting-started/concepts.md) section.
