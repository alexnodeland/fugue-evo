# fugue-evo Ergonomics Overhaul

## Overview

This document outlines a comprehensive plan to improve the developer experience (DX) of fugue-evo. The library has excellent algorithmic depth but suffers from API friction that limits adoption. This plan addresses the core issues while preserving the library's advanced capabilities.

**Goal:** Make simple things simple, complex things possible.

---

## Current State Analysis

### Primary Pain Points

| Issue | Severity | User Impact |
|-------|----------|-------------|
| 7-parameter builder with 5 wildcards | Critical | IDE breaks, intimidating first impression |
| No clear entry point / happy path | High | Users don't know where to start |
| 40+ line boilerplate for custom evolution | High | Copy-paste errors, maintenance burden |
| Forced Fugue/trace integration | Medium | Coupling to PPL even for simple optimization |
| No sensible defaults | Medium | Users must research every parameter |
| Feature-gated trait differences | Medium | Silent breakage on feature toggle |

### Current API (Problematic)

```rust
// This is the FIRST thing users see in README
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
```

---

## Proposed Architecture

### Tiered API Design

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 1: fugue_evo::quick                                      │
│  "I just want to optimize something"                            │
│  - Closure-based fitness                                        │
│  - Sensible defaults                                            │
│  - Minimal configuration                                        │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: fugue_evo::prelude                                    │
│  "I want control over operators"                                │
│  - Current builder API (improved)                               │
│  - Custom operators                                             │
│  - Callbacks and iterators                                      │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: fugue_evo::{algorithms, operators, ...}               │
│  "I want full control"                                          │
│  - Direct module access                                         │
│  - Fugue/trace integration                                      │
│  - Probabilistic models (SMC, MCMC)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Quick API Module

**Priority:** P0
**Estimated Scope:** New module, ~500-800 lines
**Files:** `src/quick/mod.rs`, `src/quick/minimize.rs`, `src/quick/maximize.rs`, `src/quick/config.rs`

#### Target API

```rust
use fugue_evo::quick::minimize;

// Simplest case: closure + dimensions
let result = minimize(|x: &[f64]| {
    x.iter().map(|v| v * v).sum()
})
.dimensions(10)
.run()?;

println!("Solution: {:?}", result.solution);
println!("Fitness: {}", result.fitness);
```

```rust
// With bounds
let result = minimize(|x: &[f64]| rastrigin(x))
    .dimensions(10)
    .bounds(-5.12, 5.12)
    .run()?;
```

```rust
// With configuration
use fugue_evo::quick::{minimize, Config};

let result = minimize(expensive_simulation)
    .dimensions(20)
    .bounds(-10.0, 10.0)
    .config(Config {
        population_size: 200,
        max_generations: 1000,
        target_fitness: Some(0.001),
        seed: Some(42),
        ..Default::default()
    })
    .run()?;
```

```rust
// Maximize variant
use fugue_evo::quick::maximize;

let result = maximize(|x: &[f64]| -x.iter().map(|v| v * v).sum::<f64>())
    .dimensions(10)
    .run()?;
```

#### Implementation Details

**`src/quick/mod.rs`:**
```rust
//! Quick and easy optimization API
//!
//! This module provides a simplified interface for common optimization tasks.
//! For more control, use the full API via `fugue_evo::prelude`.
//!
//! # Examples
//!
//! ```rust
//! use fugue_evo::quick::minimize;
//!
//! let result = minimize(|x: &[f64]| x.iter().map(|v| v*v).sum())
//!     .dimensions(10)
//!     .run()
//!     .unwrap();
//! ```

mod config;
mod minimize;
mod maximize;
mod result;

pub use config::Config;
pub use minimize::minimize;
pub use maximize::maximize;
pub use result::QuickResult;
```

**`src/quick/config.rs`:**
```rust
/// Configuration for quick optimization
#[derive(Clone, Debug)]
pub struct Config {
    /// Population size (default: 10 * dimensions)
    pub population_size: Option<usize>,

    /// Maximum generations (default: 500)
    pub max_generations: usize,

    /// Stop when fitness reaches this value
    pub target_fitness: Option<f64>,

    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Tournament size for selection (default: 3)
    pub tournament_size: usize,

    /// Enable parallel fitness evaluation
    pub parallel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: None,  // Will use 10 * dim
            max_generations: 500,
            target_fitness: None,
            seed: None,
            tournament_size: 3,
            parallel: false,
        }
    }
}
```

**`src/quick/minimize.rs`:**
```rust
use crate::quick::{Config, QuickResult};

/// Builder for minimization problems
pub struct MinimizeBuilder<F> {
    fitness_fn: F,
    dimensions: Option<usize>,
    lower_bound: f64,
    upper_bound: f64,
    config: Config,
}

/// Start building a minimization problem
///
/// # Example
/// ```rust
/// use fugue_evo::quick::minimize;
///
/// let result = minimize(|x: &[f64]| x.iter().sum::<f64>().powi(2))
///     .dimensions(5)
///     .run()
///     .unwrap();
/// ```
pub fn minimize<F>(fitness_fn: F) -> MinimizeBuilder<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    MinimizeBuilder {
        fitness_fn,
        dimensions: None,
        lower_bound: -10.0,
        upper_bound: 10.0,
        config: Config::default(),
    }
}

impl<F> MinimizeBuilder<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    /// Set the number of dimensions (required)
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.dimensions = Some(dim);
        self
    }

    /// Set symmetric bounds [-b, b] for all dimensions
    pub fn bounds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_bound = lower;
        self.upper_bound = upper;
        self
    }

    /// Set configuration options
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Set random seed for reproducibility
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = Some(seed);
        self
    }

    /// Set maximum generations
    pub fn max_generations(mut self, gens: usize) -> Self {
        self.config.max_generations = gens;
        self
    }

    /// Run the optimization
    pub fn run(self) -> Result<QuickResult, crate::error::EvolutionError> {
        let dim = self.dimensions
            .ok_or_else(|| crate::error::EvolutionError::Configuration(
                "dimensions() must be called before run()".into()
            ))?;

        // Use internal machinery with sensible defaults
        let pop_size = self.config.population_size.unwrap_or(dim * 10);

        // Delegate to SimpleGA with wrapped fitness function
        // ... implementation details

        todo!("Implement using SimpleGA internally")
    }
}
```

**`src/quick/result.rs`:**
```rust
/// Result of a quick optimization run
#[derive(Clone, Debug)]
pub struct QuickResult {
    /// The best solution found
    pub solution: Vec<f64>,

    /// Fitness value of the best solution
    pub fitness: f64,

    /// Number of generations run
    pub generations: usize,

    /// Total fitness evaluations
    pub evaluations: usize,
}
```

#### Acceptance Criteria

- [ ] `minimize()` and `maximize()` functions work with closures
- [ ] `.dimensions()`, `.bounds()`, `.config()` chainable methods
- [ ] Sensible defaults that work without tuning
- [ ] Clear error messages for missing required fields
- [ ] Documentation with examples
- [ ] Integration tests covering common use cases

---

### Phase 2: Builder Type Parameter Reduction

**Priority:** P0
**Estimated Scope:** Refactor existing builders, ~300-500 lines changed
**Files:** `src/algorithms/simple_ga.rs`, `src/algorithms/island.rs`, etc.

#### Current Problem

```rust
pub struct SimpleGABuilder<G, F, S, C, M, Fit, Term> {
    // 7 type parameters!
}
```

#### Solution A: Runtime Validation with Boxed Traits

Replace typestate pattern with optional fields:

```rust
pub struct SimpleGABuilder<G: EvolutionaryGenome> {
    population_size: usize,
    bounds: Option<MultiBounds>,
    selection: Option<Box<dyn SelectionOperator<G>>>,
    crossover: Option<Box<dyn CrossoverOperator<G>>>,
    mutation: Option<Box<dyn MutationOperator<G>>>,
    fitness: Option<Box<dyn Fitness<Genome = G, Value = f64>>>,
    termination: Option<Box<dyn TerminationCriterion<G, f64>>>,
}

impl<G: EvolutionaryGenome> SimpleGABuilder<G> {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            bounds: None,
            selection: None,
            crossover: None,
            mutation: None,
            fitness: None,
            termination: None,
        }
    }

    pub fn selection<S: SelectionOperator<G> + 'static>(mut self, s: S) -> Self {
        self.selection = Some(Box::new(s));
        self
    }

    // ... other setters

    pub fn build(self) -> Result<SimpleGA<G>, BuilderError> {
        Ok(SimpleGA {
            population_size: self.population_size,
            bounds: self.bounds.ok_or(BuilderError::MissingField("bounds"))?,
            selection: self.selection.ok_or(BuilderError::MissingField("selection"))?,
            // ...
        })
    }
}
```

**Usage becomes:**
```rust
// No more turbofish with wildcards!
let ga = SimpleGABuilder::<RealVector>::new()
    .population_size(100)
    .bounds(bounds)
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(sphere)
    .build()?;
```

#### Solution B: Macro for Common Cases (Complementary)

```rust
/// Macro for quick GA setup
#[macro_export]
macro_rules! simple_ga {
    (
        genome: $genome:ty,
        fitness: $fitness:expr,
        bounds: $bounds:expr
        $(, population: $pop:expr)?
        $(, generations: $gens:expr)?
        $(, selection: $sel:expr)?
        $(, crossover: $cross:expr)?
        $(, mutation: $mut:expr)?
        $(,)?
    ) => {{
        use $crate::prelude::*;

        SimpleGABuilder::<$genome>::new()
            .fitness($fitness)
            .bounds($bounds)
            $(.population_size($pop))?
            $(.max_generations($gens))?
            $(.selection($sel))?
            $(.crossover($cross))?
            $(.mutation($mut))?
            .with_defaults()
            .build()
    }};
}

// Usage:
let ga = simple_ga! {
    genome: RealVector,
    fitness: Sphere::new(10),
    bounds: MultiBounds::symmetric(5.12, 10),
    generations: 200,
};
```

#### Acceptance Criteria

- [ ] Builder requires only 1 type parameter (genome type)
- [ ] All builder methods work without explicit type annotations
- [ ] `build()` returns clear errors for missing required fields
- [ ] Backward compatibility via deprecated type aliases (if needed)
- [ ] Macro works for common configurations

---

### Phase 3: Sensible Defaults and Recommendations

**Priority:** P1
**Estimated Scope:** ~200-300 lines across operator files
**Files:** `src/operators/selection/*.rs`, `src/operators/crossover/*.rs`, `src/operators/mutation/*.rs`, builders

#### Add `recommended()` Constructors

```rust
// src/operators/selection/tournament.rs
impl TournamentSelection {
    /// Create tournament selection with recommended size (3)
    ///
    /// Tournament size of 3 provides good selection pressure
    /// for most problems without excessive loss of diversity.
    pub fn recommended() -> Self {
        Self::new(3)
    }
}

// src/operators/crossover/sbx.rs
impl SbxCrossover {
    /// Create SBX crossover with recommended distribution index (20.0)
    ///
    /// η = 20 produces offspring relatively close to parents,
    /// suitable for most real-coded GA applications.
    pub fn recommended() -> Self {
        Self::new(20.0)
    }
}

// src/operators/mutation/polynomial.rs
impl PolynomialMutation {
    /// Create polynomial mutation with recommended distribution index (20.0)
    ///
    /// η_m = 20 produces small mutations, suitable for fine-tuning.
    /// For more exploration, use lower values (5-10).
    pub fn recommended() -> Self {
        Self::new(20.0)
    }
}
```

#### Add `with_defaults()` to Builders

```rust
impl<G: EvolutionaryGenome> SimpleGABuilder<G> {
    /// Fill in missing operators with recommended defaults
    ///
    /// Uses:
    /// - Tournament selection (size 3)
    /// - SBX crossover (η = 20)
    /// - Polynomial mutation (η_m = 20)
    /// - Max generations termination (500)
    pub fn with_defaults(mut self) -> Self {
        if self.selection.is_none() {
            self.selection = Some(Box::new(TournamentSelection::recommended()));
        }
        if self.crossover.is_none() {
            self.crossover = Some(Box::new(SbxCrossover::recommended()));
        }
        if self.mutation.is_none() {
            self.mutation = Some(Box::new(PolynomialMutation::recommended()));
        }
        if self.termination.is_none() {
            self.termination = Some(Box::new(MaxGenerations::new(500)));
        }
        self
    }
}
```

#### Add Heuristic Helpers

```rust
// src/lib.rs or src/utils/heuristics.rs

/// Heuristics for parameter selection
pub mod heuristics {
    /// Recommended population size based on problem dimension
    ///
    /// Returns max(50, 10 * dimension) which works well for most problems.
    pub fn population_size(dimension: usize) -> usize {
        (dimension * 10).max(50)
    }

    /// Recommended mutation probability
    ///
    /// Returns 1/dimension, the standard choice for real-coded GAs.
    pub fn mutation_probability(dimension: usize) -> f64 {
        1.0 / dimension as f64
    }

    /// Recommended tournament size for given population
    ///
    /// Returns 2-5 depending on population size.
    pub fn tournament_size(population_size: usize) -> usize {
        match population_size {
            0..=50 => 2,
            51..=200 => 3,
            201..=500 => 4,
            _ => 5,
        }
    }
}
```

#### Acceptance Criteria

- [ ] All major operators have `recommended()` constructors
- [ ] `with_defaults()` fills in sensible values
- [ ] Heuristic functions documented with rationale
- [ ] Examples updated to show simplified usage

---

### Phase 4: Iterator and Callback APIs

**Priority:** P1
**Estimated Scope:** ~400-600 lines
**Files:** `src/algorithms/simple_ga.rs`, new `src/algorithms/stepper.rs`

#### Iterator API

```rust
/// Step-by-step evolution iterator
pub struct EvolutionSteps<'a, G, F, S, C, M, Fit> {
    ga: &'a mut SimpleGA<G, F, S, C, M, Fit>,
    rng: &'a mut dyn RngCore,
    generation: usize,
}

/// State returned at each generation
pub struct GenerationState<G: EvolutionaryGenome> {
    pub generation: usize,
    pub best_genome: G,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub diversity: f64,
    pub population_size: usize,
}

impl<G, ...> SimpleGA<G, ...> {
    /// Returns an iterator over generations
    ///
    /// # Example
    /// ```rust
    /// let mut ga = SimpleGA::builder()...build()?;
    ///
    /// for state in ga.steps(&mut rng).take(100) {
    ///     println!("Gen {}: best = {:.4}", state.generation, state.best_fitness);
    ///
    ///     // Early termination
    ///     if state.best_fitness < 0.001 {
    ///         break;
    ///     }
    /// }
    ///
    /// let result = ga.into_result();
    /// ```
    pub fn steps<'a, R: RngCore>(&'a mut self, rng: &'a mut R) -> EvolutionSteps<'a, ...> {
        EvolutionSteps {
            ga: self,
            rng,
            generation: 0,
        }
    }
}

impl<'a, G, ...> Iterator for EvolutionSteps<'a, G, ...> {
    type Item = GenerationState<G>;

    fn next(&mut self) -> Option<Self::Item> {
        self.generation += 1;
        let state = self.ga.step_once(self.rng);
        Some(state)
    }
}
```

**Usage:**
```rust
// Simple iteration with early stopping
for state in ga.steps(&mut rng) {
    if state.generation >= 500 || state.best_fitness < 0.001 {
        break;
    }
}

// Functional style
let final_state = ga.steps(&mut rng)
    .take(500)
    .take_while(|s| s.best_fitness > 0.001)
    .last();

// With progress reporting
for state in ga.steps(&mut rng).take(100) {
    if state.generation % 10 == 0 {
        println!("Progress: gen={}, best={:.4}", state.generation, state.best_fitness);
    }
}
```

#### Callback API

```rust
/// Callbacks for evolution events
pub trait EvolutionCallbacks<G: EvolutionaryGenome> {
    /// Called at the start of each generation
    fn on_generation_start(&mut self, _gen: usize, _population: &Population<G>) {}

    /// Called at the end of each generation
    fn on_generation_end(&mut self, _gen: usize, _stats: &GenerationStats) {}

    /// Called when a new best is found
    fn on_new_best(&mut self, _gen: usize, _genome: &G, _fitness: f64) {}

    /// Called for each offspring created
    fn on_offspring(&mut self, _parent1: &G, _parent2: &G, _child: &G) {}

    /// Return true to stop evolution early
    fn should_stop(&self) -> bool { false }
}

/// Simple callback for logging
pub struct LoggingCallback {
    pub log_interval: usize,
}

impl<G: EvolutionaryGenome> EvolutionCallbacks<G> for LoggingCallback {
    fn on_generation_end(&mut self, gen: usize, stats: &GenerationStats) {
        if gen % self.log_interval == 0 {
            println!("Gen {}: best={:.4}, mean={:.4}", gen, stats.best_fitness, stats.mean_fitness);
        }
    }
}

impl<G, ...> SimpleGA<G, ...> {
    /// Run with callbacks
    pub fn run_with_callbacks<R, CB>(
        &mut self,
        rng: &mut R,
        callbacks: &mut CB,
    ) -> Result<EvolutionResult<G>, EvolutionError>
    where
        R: RngCore,
        CB: EvolutionCallbacks<G>,
    {
        // ... implementation
    }
}
```

**Usage:**
```rust
// Custom callback
struct MyCallback {
    best_history: Vec<f64>,
}

impl<G: EvolutionaryGenome> EvolutionCallbacks<G> for MyCallback {
    fn on_new_best(&mut self, gen: usize, _genome: &G, fitness: f64) {
        self.best_history.push(fitness);
        println!("New best at gen {}: {:.4}", gen, fitness);
    }

    fn should_stop(&self) -> bool {
        self.best_history.last().map(|&f| f < 0.001).unwrap_or(false)
    }
}

let mut callback = MyCallback { best_history: vec![] };
ga.run_with_callbacks(&mut rng, &mut callback)?;
```

#### Acceptance Criteria

- [ ] `.steps()` returns a proper `Iterator` impl
- [ ] `GenerationState` contains all useful per-generation info
- [ ] Callbacks fire at appropriate times
- [ ] Early termination works via both iterator and callback
- [ ] Examples demonstrate both patterns

---

### Phase 5: Trait Decoupling (Genome vs TracedGenome)

**Priority:** P1
**Estimated Scope:** ~300-400 lines refactoring
**Files:** `src/genome/traits.rs`, algorithm files

#### Current Problem

```rust
pub trait EvolutionaryGenome: Clone + Send + Sync + Serialize + DeserializeOwned + 'static {
    fn to_trace(&self) -> Trace;  // Required even if never using Fugue
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;
    // ...
}
```

#### Solution: Split Traits

```rust
// src/genome/traits.rs

/// Core genome trait - no Fugue dependency
pub trait Genome: Clone + 'static {
    /// The type of individual genes/alleles
    type Allele: Clone;

    /// Number of dimensions/genes
    fn dimension(&self) -> usize;

    /// Generate a random genome within bounds
    fn generate<R: Rng>(rng: &mut R, bounds: &MultiBounds) -> Self;

    /// Optional: slice access for efficient operations
    fn as_slice(&self) -> Option<&[Self::Allele]> { None }
    fn as_mut_slice(&mut self) -> Option<&mut [Self::Allele]> { None }

    /// Distance metric for diversity calculations
    fn distance(&self, other: &Self) -> f64 { 0.0 }
}

/// Extension trait for Fugue/trace integration
pub trait TracedGenome: Genome {
    /// Convert to a Fugue trace
    fn to_trace(&self) -> Trace;

    /// Reconstruct from a Fugue trace
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;

    /// Address prefix for trace entries
    fn trace_prefix() -> &'static str { "gene" }
}

/// Marker trait for genomes that support parallel evaluation
pub trait SendGenome: Genome + Send + Sync {}
impl<G: Genome + Send + Sync> SendGenome for G {}

/// Marker trait for serializable genomes (checkpointing)
pub trait SerializableGenome: Genome + Serialize + DeserializeOwned {}
impl<G: Genome + Serialize + DeserializeOwned> SerializableGenome for G {}
```

#### Update Algorithm Bounds

```rust
// Standard algorithms only need Genome
impl<G: Genome, ...> SimpleGA<G, ...> {
    pub fn run(...) { ... }
}

// SMC/MCMC need TracedGenome
impl<G: TracedGenome, ...> EvolutionarySMC<G, ...> {
    pub fn run(...) { ... }
}

// Checkpointing needs SerializableGenome
impl<G: SerializableGenome, ...> SimpleGA<G, ...> {
    pub fn save_checkpoint(...) { ... }
    pub fn load_checkpoint(...) { ... }
}
```

#### Backward Compatibility

```rust
/// Type alias for backward compatibility
#[deprecated(note = "Use Genome or TracedGenome instead")]
pub trait EvolutionaryGenome: TracedGenome + SendGenome + SerializableGenome {}
impl<G: TracedGenome + SendGenome + SerializableGenome> EvolutionaryGenome for G {}
```

#### Acceptance Criteria

- [ ] `Genome` trait has no Fugue dependency
- [ ] Standard algorithms work with `Genome` only
- [ ] SMC/MCMC require `TracedGenome`
- [ ] Checkpointing requires `SerializableGenome`
- [ ] Existing code compiles with deprecation warnings
- [ ] Built-in genomes implement all relevant traits

---

### Phase 6: Unified Parallel/Non-Parallel Traits

**Priority:** P2
**Estimated Scope:** ~200 lines refactoring
**Files:** `src/fitness/traits.rs`, algorithm files

#### Current Problem

```rust
#[cfg(feature = "parallel")]
pub trait Fitness: Send + Sync { ... }

#[cfg(not(feature = "parallel"))]
pub trait Fitness { ... }
```

Different trait bounds based on feature flags cause silent breakage.

#### Solution: Always Require Send + Sync, Feature-Gate Execution

```rust
// src/fitness/traits.rs

/// Fitness function trait
///
/// Always requires Send + Sync for consistency.
/// Use `run()` for single-threaded or `run_parallel()` for parallel execution.
pub trait Fitness: Send + Sync {
    type Genome: Genome;
    type Value: FitnessValue;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value;
}

// src/algorithms/simple_ga.rs

impl<G, F, ...> SimpleGA<G, F, ...> {
    /// Run evolution (single-threaded)
    pub fn run<R: RngCore>(&mut self, rng: &mut R) -> Result<EvolutionResult<G>, EvolutionError> {
        self.run_internal(rng, false)
    }

    /// Run evolution with parallel fitness evaluation
    #[cfg(feature = "parallel")]
    pub fn run_parallel<R: RngCore>(&mut self, rng: &mut R) -> Result<EvolutionResult<G>, EvolutionError> {
        self.run_internal(rng, true)
    }

    fn run_internal<R: RngCore>(&mut self, rng: &mut R, parallel: bool) -> Result<...> {
        // ... shared implementation
    }
}
```

#### Acceptance Criteria

- [ ] Single `Fitness` trait definition (always Send + Sync)
- [ ] `run()` always works
- [ ] `run_parallel()` available when feature enabled
- [ ] No code duplication between parallel/non-parallel

---

### Phase 7: Improved Error Messages

**Priority:** P2
**Estimated Scope:** ~100-150 lines
**Files:** `src/error.rs`

#### Enhance Error Types

```rust
// src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum EvolutionError {
    #[error("Configuration error: {message}\n\nHelp: {help}")]
    Configuration {
        message: String,
        help: String,
    },

    #[error("Missing required field '{field}' in builder\n\nHelp: Call .{field}(...) before .build()")]
    MissingBuilderField {
        field: &'static str,
    },

    #[error("Dimension mismatch: expected {expected}, got {actual}\n\nHelp: {help}")]
    DimensionMismatch {
        expected: usize,
        actual: usize,
        help: String,
    },

    #[error("Trace error at address '{address}': {message}\n\nHelp: Ensure genome was created with the same configuration")]
    TraceError {
        address: String,
        message: String,
    },

    #[error("Population is empty\n\nHelp: Ensure population_size > 0 and initialization succeeded")]
    EmptyPopulation,
}

impl EvolutionError {
    pub fn missing_field(field: &'static str) -> Self {
        Self::MissingBuilderField { field }
    }

    pub fn dimension_mismatch(expected: usize, actual: usize, context: &str) -> Self {
        Self::DimensionMismatch {
            expected,
            actual,
            help: format!("Check that {} has the correct dimension", context),
        }
    }
}
```

#### Acceptance Criteria

- [ ] All errors include actionable help text
- [ ] Builder errors specify which field is missing
- [ ] Dimension errors include context about what mismatched

---

## Testing Strategy

### Unit Tests

Each phase should include:
- Happy path tests
- Error case tests
- Edge case tests (empty inputs, single element, etc.)

### Integration Tests

```rust
// tests/quick_api.rs
#[test]
fn test_quick_minimize_sphere() {
    let result = minimize(|x: &[f64]| x.iter().map(|v| v*v).sum())
        .dimensions(10)
        .seed(42)
        .run()
        .unwrap();

    assert!(result.fitness < 0.1);
}

// tests/iterator_api.rs
#[test]
fn test_evolution_iterator() {
    let mut ga = SimpleGA::builder()...build().unwrap();

    let states: Vec<_> = ga.steps(&mut rng).take(10).collect();
    assert_eq!(states.len(), 10);
    assert!(states[9].best_fitness <= states[0].best_fitness);
}
```

### Documentation Tests

All public APIs should have doctests that compile and run.

---

## Documentation Updates

### README.md

Update to show the new quick API first:

```markdown
# fugue-evo

Probabilistic genetic algorithms in Rust.

## Quick Start

```rust
use fugue_evo::quick::minimize;

let result = minimize(|x: &[f64]| {
    x.iter().map(|v| v * v).sum()
})
.dimensions(10)
.run()?;

println!("Best solution: {:?}", result.solution);
```

## More Control

For custom operators and advanced features:

```rust
use fugue_evo::prelude::*;

let ga = SimpleGA::builder()
    .fitness(MyFitness::new())
    .bounds(bounds)
    .with_defaults()
    .build()?;

for state in ga.steps(&mut rng).take(500) {
    // ...
}
```
```

### API Documentation

- Add module-level docs explaining the tiered API
- Add "See also" links between related items
- Add examples to all public items

---

## Migration Guide

For existing users upgrading:

```markdown
# Migration Guide

## Builder Changes

Before:
```rust
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
```

After:
```rust
SimpleGA::builder()  // or SimpleGABuilder::<RealVector>::new()
```

## Trait Changes

Before:
```rust
impl EvolutionaryGenome for MyGenome {
    fn to_trace(&self) -> Trace { ... }  // Required
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> { ... }  // Required
}
```

After:
```rust
impl Genome for MyGenome {
    // Core methods only
}

// Only if using SMC/MCMC:
impl TracedGenome for MyGenome {
    fn to_trace(&self) -> Trace { ... }
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> { ... }
}
```
```

---

## Success Metrics

1. **Reduced boilerplate:** Simple optimization in <10 lines
2. **No turbofish:** Common usage requires no explicit type annotations
3. **Discoverability:** Clear path from simple to advanced usage
4. **Compilation speed:** No regression from trait changes
5. **Documentation:** All public APIs have examples

---

## File Summary

| Phase | New/Modified Files |
|-------|-------------------|
| 1 | `src/quick/mod.rs`, `src/quick/minimize.rs`, `src/quick/maximize.rs`, `src/quick/config.rs`, `src/quick/result.rs` |
| 2 | `src/algorithms/simple_ga.rs`, `src/algorithms/island.rs`, `src/algorithms/nsga2.rs` |
| 3 | `src/operators/selection/*.rs`, `src/operators/crossover/*.rs`, `src/operators/mutation/*.rs`, `src/lib.rs` |
| 4 | `src/algorithms/simple_ga.rs`, `src/algorithms/stepper.rs` (new), `src/algorithms/callbacks.rs` (new) |
| 5 | `src/genome/traits.rs`, all algorithm files |
| 6 | `src/fitness/traits.rs`, algorithm files |
| 7 | `src/error.rs` |

---

## Open Questions

1. **Backward compatibility:** Should we maintain the old builder API behind a feature flag, or deprecate entirely?

2. **Quick API scope:** Should `quick` module support multi-objective (NSGA-II) or keep it single-objective only?

3. **Macro naming:** `simple_ga!` vs `ga!` vs `evolve!`?

4. **Iterator ownership:** Should `.steps()` take `&mut self` or consume `self`?

5. **Callback granularity:** How fine-grained should callbacks be? Per-offspring? Per-mutation?

---

## References

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Builder pattern in Rust](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html)
- [Iterator patterns](https://doc.rust-lang.org/std/iter/index.html)
