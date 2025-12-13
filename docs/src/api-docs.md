# API Documentation

The complete API documentation is generated from rustdoc comments in the source code.

## Viewing API Docs

### Online

Visit the docs.rs page (when published):

```
https://docs.rs/fugue-evo
```

### Local Generation

Generate and view locally:

```bash
cargo doc --open
```

Or generate without opening:

```bash
cargo doc --no-deps
```

Docs will be in `target/doc/fugue_evo/index.html`.

## Module Structure

```
fugue_evo
├── algorithms          # Optimization algorithms
│   ├── simple_ga       # Simple Genetic Algorithm
│   ├── cmaes           # CMA-ES
│   ├── nsga2           # NSGA-II (multi-objective)
│   ├── island          # Island Model
│   └── eda             # EDA/UMDA
├── genome              # Genome types
│   ├── real_vector     # Continuous optimization
│   ├── bit_string      # Binary optimization
│   ├── permutation     # Ordering problems
│   ├── tree            # Genetic programming
│   └── bounds          # Search space bounds
├── operators           # Genetic operators
│   ├── selection       # Selection operators
│   ├── crossover       # Crossover operators
│   └── mutation        # Mutation operators
├── fitness             # Fitness functions
│   ├── traits          # Fitness trait
│   └── benchmarks      # Standard benchmarks
├── population          # Population management
├── termination         # Stopping criteria
├── hyperparameter      # Parameter adaptation
├── interactive         # Human-in-the-loop
├── checkpoint          # State persistence
├── diagnostics         # Statistics and tracking
├── fugue_integration   # PPL integration
└── error               # Error types
```

## Prelude

For convenience, import everything from the prelude:

```rust,ignore
use fugue_evo::prelude::*;
```

This imports:
- All algorithm builders
- All genome types
- All operators
- All fitness traits and benchmarks
- Common utility types

## Key Entry Points

### Algorithms

| Type | Description |
|------|-------------|
| `SimpleGABuilder` | General-purpose GA |
| `CmaEs` | CMA-ES optimizer |
| `Nsga2Builder` | Multi-objective NSGA-II |
| `IslandModelBuilder` | Parallel island model |
| `InteractiveGABuilder` | Human-in-the-loop |

### Genomes

| Type | Description |
|------|-------------|
| `RealVector` | Real-valued vector |
| `BitString` | Binary string |
| `Permutation` | Ordering |
| `TreeGenome` | Tree structure |

### Operators

| Type | Description |
|------|-------------|
| `TournamentSelection` | Tournament selection |
| `SbxCrossover` | Simulated binary crossover |
| `PolynomialMutation` | Polynomial mutation |

## Documentation Conventions

### Example Code

All examples in rustdoc are tested (where possible):

```rust,ignore
/// # Example
///
/// ```
/// use fugue_evo::prelude::*;
/// let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
/// assert_eq!(genome.dimension(), 3);
/// ```
```

### Feature Gates

Feature-gated items are marked:

```rust,ignore
/// Saves checkpoint to file.
///
/// **Requires feature:** `checkpoint`
#[cfg(feature = "checkpoint")]
pub fn save(&self, path: &Path) -> Result<(), Error>
```

### See Also

Cross-references to related items:

```rust,ignore
/// Tournament selection operator.
///
/// # See Also
///
/// - [`RouletteWheelSelection`] - Alternative selection
/// - [`RankSelection`] - Rank-based alternative
```
