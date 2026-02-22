# Design Philosophy

Fugue-evo is built on several core design principles that distinguish it from traditional GA libraries.

## Evolution as Bayesian Inference

The central insight of fugue-evo is that evolutionary algorithms can be understood through a probabilistic lens:

### Fitness as Likelihood

In Bayesian terms:
- **Prior**: Initial population distribution
- **Likelihood**: Fitness function (how well does this solution explain our objective?)
- **Posterior**: Population after selection

Selection acts as **conditioning**: we observe that solutions should have high fitness, and update our population distribution accordingly.

### Learnable Operators

Traditional GAs use fixed operators with hand-tuned parameters. Fugue-evo treats operator parameters as **uncertain quantities** that can be learned:

```rust,ignore
// Traditional: fixed mutation rate
let mutation_rate = 0.1;

// Fugue-evo: learned mutation rate
let mut posterior = BetaPosterior::new(2.0, 2.0);
// ... observe outcomes, update posterior
let learned_rate = posterior.mean();
```

## Type Safety

Rust's type system enables compile-time correctness guarantees.

### Generic Constraints

Operators are typed to their applicable genomes:

```rust,ignore
// SBX only works with real-valued genomes
impl CrossoverOperator<RealVector> for SbxCrossover { ... }

// Order crossover only works with permutations
impl CrossoverOperator<Permutation> for OrderCrossover { ... }
```

### Builder Pattern

Configuration errors are caught at compile time:

```rust,ignore
let ga = SimpleGABuilder::new()
    .population_size(100)
    // Missing required configuration = compile error
    .build()?;  // Only compiles when complete
```

## Trait-Based Abstraction

Core abstractions are defined as traits, enabling extensibility.

### EvolutionaryGenome

Any type can be a genome if it implements:

```rust,ignore
pub trait EvolutionaryGenome: Clone + Send + Sync {
    fn to_trace(&self) -> Trace;
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;
}
```

### Operator Traits

Custom operators implement standard traits:

```rust,ignore
pub trait MutationOperator<G>: Send + Sync {
    fn mutate<R: Rng>(&self, genome: &mut G, rng: &mut R);
}
```

## Composability

Components are designed to compose cleanly.

### Operator Composition

```rust,ignore
// Combine mutations
let mutation = CompositeMutation::new(
    GaussianMutation::new(0.1),
    PolynomialMutation::new(20.0),
    0.5, // 50% chance of each
);
```

### Termination Composition

```rust,ignore
// Complex stopping conditions
let term = AnyOf::new(vec![
    Box::new(MaxGenerations::new(1000)),
    Box::new(AllOf::new(vec![
        Box::new(MinGenerations::new(100)),
        Box::new(FitnessStagnation::new(20)),
    ])),
]);
```

## Reproducibility

Scientific use requires reproducibility.

### Seeded RNG

All randomness flows through explicit RNG:

```rust,ignore
let mut rng = StdRng::seed_from_u64(42);
let result = ga.run(&mut rng)?;
// Same seed = same result
```

### Checkpointing

State can be saved and restored:

```rust,ignore
manager.save(&checkpoint)?;
// Later...
let checkpoint = load_checkpoint(&path)?;
```

## Performance

### Optional Parallelism

Parallelism is opt-in and doesn't affect correctness:

```rust,ignore
// Sequential (deterministic)
.parallel(false)

// Parallel (faster, same final result distribution)
.parallel(true)
```

### Efficient Representations

Genomes use efficient storage:

```rust,ignore
// RealVector: contiguous Vec<f64>
// BitString: Vec<bool> (could use bitpacking)
// Permutation: Vec<usize>
```

## Interoperability

### Fugue Integration

Deep integration with probabilistic programming:

```rust,ignore
// Genomes convert to traces
let trace = genome.to_trace();

// Enables trace-based operators
let mutated = trace_mutation(&trace, &mut rng);
```

### WASM Support

Same algorithms run in browser:

```javascript
const optimizer = new SimpleGAOptimizer(config);
const result = optimizer.run(fitnessFunction);
```

## Simplicity

### Avoid Over-Engineering

- Minimal dependencies
- Clear, focused APIs
- Documentation over abstraction

### Progressive Disclosure

Simple cases are simple:

```rust,ignore
// Simplest usage
let result = SimpleGABuilder::new()
    .population_size(100)
    .bounds(bounds)
    .fitness(fitness)
    .max_generations(200)
    .defaults()  // Use sensible defaults
    .run(&mut rng)?;
```

Advanced features are available when needed:

```rust,ignore
// Full control
let result = SimpleGABuilder::new()
    .population_size(100)
    .bounds(bounds)
    .selection(custom_selection)
    .crossover(custom_crossover)
    .mutation(custom_mutation)
    .fitness(custom_fitness)
    .termination(complex_termination)
    .parallel(true)
    .elitism(true)
    .elite_count(5)
    .build()?
    .run(&mut rng)?;
```
