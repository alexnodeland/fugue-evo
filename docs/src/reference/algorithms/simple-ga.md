# SimpleGA Reference

The Simple Genetic Algorithm is a flexible, general-purpose evolutionary optimization algorithm.

## Module

```rust,ignore
use fugue_evo::algorithms::simple_ga::{SimpleGA, SimpleGABuilder, SimpleGAConfig};
```

## Builder API

### Required Configuration

| Method | Type | Description |
|--------|------|-------------|
| `bounds(bounds)` | `MultiBounds` | Search space bounds |
| `selection(sel)` | `impl SelectionOperator` | Selection operator |
| `crossover(cx)` | `impl CrossoverOperator` | Crossover operator |
| `mutation(mut)` | `impl MutationOperator` | Mutation operator |
| `fitness(fit)` | `impl Fitness` | Fitness function |

### Optional Configuration

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `population_size(n)` | `usize` | 100 | Population size |
| `max_generations(n)` | `usize` | 100 | Max generations |
| `elitism(b)` | `bool` | false | Enable elitism |
| `elite_count(n)` | `usize` | 1 | Number of elites |
| `parallel(b)` | `bool` | false | Parallel evaluation |
| `initial_population(pop)` | `Vec<G>` | Random | Custom initial population |

## Usage

### Basic Example

```rust,ignore
use fugue_evo::prelude::*;

let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .bounds(MultiBounds::symmetric(5.12, 10))
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(Sphere::new(10))
    .max_generations(200)
    .elitism(true)
    .build()?
    .run(&mut rng)?;
```

### With Custom Termination

```rust,ignore
let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    // ... configuration
    .termination(AnyOf::new(vec![
        Box::new(MaxGenerations::new(1000)),
        Box::new(TargetFitness::new(-0.001)),
        Box::new(FitnessStagnation::new(50)),
    ]))
    .build()?
    .run(&mut rng)?;
```

### Step-by-Step Execution

```rust,ignore
let mut ga = SimpleGABuilder::new()
    // ... configuration
    .build()?;

ga.initialize(&mut rng)?;

while !ga.should_terminate() {
    ga.step(&mut rng)?;

    // Access current state
    println!("Generation {}: best = {:.6}",
        ga.generation(),
        ga.best_fitness().unwrap_or(0.0));
}

let result = ga.into_result();
```

## Configuration Struct

```rust,ignore
pub struct SimpleGAConfig {
    pub population_size: usize,
    pub elitism: bool,
    pub elite_count: usize,
    pub parallel: bool,
}
```

## Generic Parameters

The builder has extensive generics for type safety:

```rust,ignore
SimpleGABuilder::<G, F, S, C, M, Fit, Term>
```

| Parameter | Constraint | Description |
|-----------|------------|-------------|
| `G` | `EvolutionaryGenome` | Genome type |
| `F` | `FitnessValue` | Fitness value type |
| `S` | `SelectionOperator<G>` | Selection operator |
| `C` | `CrossoverOperator<G>` | Crossover operator |
| `M` | `MutationOperator<G>` | Mutation operator |
| `Fit` | `Fitness<G, Value=F>` | Fitness function |
| `Term` | `TerminationCriterion` | Termination criterion |

## Algorithm Flow

```
┌─────────────────────────────────────────┐
│           SimpleGA Flow                  │
│                                          │
│  1. Initialize random population         │
│  2. Evaluate fitness                     │
│  3. while not terminated:                │
│     a. Select parents                    │
│     b. Apply crossover                   │
│     c. Apply mutation                    │
│     d. Evaluate offspring                │
│     e. Replace population (with elitism) │
│     f. Update statistics                 │
│  4. Return best solution                 │
└─────────────────────────────────────────┘
```

## Error Handling

```rust,ignore
let result = SimpleGABuilder::new()
    .population_size(100)
    // Missing required configuration
    .build(); // Returns Err(ConfigurationError)

match result {
    Ok(ga) => { /* run */ }
    Err(e) => eprintln!("Configuration error: {}", e),
}
```

## Performance Tips

1. **Population size**: Start with 100, increase for multimodal problems
2. **Selection pressure**: Higher tournament size = faster convergence
3. **Elitism**: Almost always beneficial
4. **Parallelism**: Enable for expensive fitness functions

## See Also

- [Continuous Optimization Tutorial](../../tutorials/continuous-optimization.md)
- [Custom Operators](../../how-to/custom-operators.md)
- [API Documentation](../../api-docs.md)
