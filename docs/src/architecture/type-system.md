# Type System

Fugue-evo uses Rust's type system to ensure correctness and provide good developer experience.

## Generic Algorithm Builders

Algorithm builders use extensive generics:

```rust,ignore
pub struct SimpleGABuilder<G, F, S, C, M, Fit, Term> {
    // G: Genome type
    // F: Fitness value type
    // S: Selection operator
    // C: Crossover operator
    // M: Mutation operator
    // Fit: Fitness function
    // Term: Termination criterion
}
```

### Why So Many Generics?

**Type safety**: Each component's compatibility is checked at compile time.

```rust,ignore
// This compiles: SBX works with RealVector
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .crossover(SbxCrossover::new(20.0))

// This wouldn't compile: SBX doesn't implement CrossoverOperator<BitString>
SimpleGABuilder::<BitString, f64, _, _, _, _, _>::new()
    .crossover(SbxCrossover::new(20.0))  // Compile error!
```

### Type Inference

Rust often infers generic parameters:

```rust,ignore
// Full explicit types (verbose)
SimpleGABuilder::<
    RealVector,
    f64,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
    Sphere,
    MaxGenerations,
>::new()

// With inference (same thing, cleaner)
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    // Types inferred from usage
```

## Trait Bounds

### Genome Constraints

```rust,ignore
pub trait EvolutionaryGenome:
    Clone           // Population copying
    + Send          // Thread safety
    + Sync          // Thread safety
    + Serialize     // Checkpointing
    + DeserializeOwned
{
    // ...
}
```

### Operator Constraints

```rust,ignore
pub trait SelectionOperator<G>: Send + Sync {
    fn select<R: Rng>(&self, population: &[(G, f64)], rng: &mut R) -> usize;
}
```

The `Send + Sync` bounds enable parallel evaluation.

### Fitness Constraints

```rust,ignore
pub trait FitnessValue:
    Clone           // Result copying
    + PartialOrd    // Comparison for selection
    + Send + Sync   // Thread safety
{
    fn to_f64(&self) -> f64;
}
```

## Associated Types

Some traits use associated types for output:

```rust,ignore
pub trait Fitness<G>: Send + Sync {
    type Value: FitnessValue;  // Associated type

    fn evaluate(&self, genome: &G) -> Self::Value;
}

// Usage
impl Fitness<RealVector> for Sphere {
    type Value = f64;  // Sphere returns f64
    // ...
}

impl Fitness<RealVector> for MultiObjective {
    type Value = ParetoFitness;  // Multi-objective returns ParetoFitness
    // ...
}
```

## Error Handling

### Result Types

Operations that can fail return `Result`:

```rust,ignore
pub type EvoResult<T> = Result<T, EvoError>;

pub fn build(self) -> EvoResult<SimpleGA<G, F, S, C, M, Fit, Term>>;
```

### Error Types

```rust,ignore
#[derive(Debug, thiserror::Error)]
pub enum EvoError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Genome error: {0}")]
    Genome(#[from] GenomeError),

    #[error("Operator error: {0}")]
    Operator(String),

    // ...
}
```

## Marker Traits

Some traits are markers for capabilities:

```rust,ignore
// Indicates genome supports bounds
pub trait BoundedGenome: EvolutionaryGenome {
    fn clamp_to_bounds(&mut self, bounds: &MultiBounds);
}

// Operators that respect bounds
pub trait BoundedMutationOperator<G: BoundedGenome>: MutationOperator<G> {
    fn mutate_bounded<R: Rng>(
        &self,
        genome: &mut G,
        bounds: &MultiBounds,
        rng: &mut R,
    );
}
```

## Phantom Types

For type-level information without runtime cost:

```rust,ignore
pub struct Population<G, F> {
    individuals: Vec<Individual<G>>,
    _fitness: PhantomData<F>,  // Track fitness type without storing
}
```

## Conditional Compilation

Features enable/disable functionality:

```rust,ignore
#[cfg(feature = "parallel")]
impl<G, F, Fit> Population<G, F>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    Fit: Fitness<G, Value = F>,
{
    pub fn evaluate_parallel(&mut self, fitness: &Fit) {
        // Rayon parallel implementation
    }
}
```

## Type Aliases

For common patterns:

```rust,ignore
// Convenience aliases
pub type RealGA = SimpleGA<RealVector, f64, TournamentSelection, SbxCrossover, PolynomialMutation, _, MaxGenerations>;

pub type BinaryGA = SimpleGA<BitString, f64, TournamentSelection, UniformCrossover, BitFlipMutation, _, MaxGenerations>;
```

## Best Practices

### 1. Let Inference Work

```rust,ignore
// Good: minimal type annotations
let ga = SimpleGABuilder::new()
    .bounds(bounds)
    .fitness(Sphere::new(10))
    // types inferred

// Verbose: unnecessary annotations
let ga: SimpleGA<RealVector, f64, ...> = SimpleGABuilder::new()
```

### 2. Use Concrete Types in Applications

```rust,ignore
// In library code: generic
fn run_optimization<G, F, Fit>(fitness: &Fit) -> EvoResult<G>
where
    G: EvolutionaryGenome,
    F: FitnessValue,
    Fit: Fitness<G, Value = F>,

// In application code: concrete
fn optimize_my_problem() -> EvoResult<RealVector> {
    let fitness = MyFitness::new();
    // ...
}
```

### 3. Understand Error Messages

Generic-heavy code can produce complex error messages. Key hints:
- "trait bound not satisfied" = wrong operator for genome type
- "cannot infer type" = add type annotations
- "lifetime" issues = usually Send/Sync bounds

## See Also

- [Design Philosophy](./philosophy.md)
- [API Documentation](../api-docs.md)
