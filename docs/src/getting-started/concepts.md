# Core Concepts

This guide explains the fundamental concepts and abstractions in fugue-evo. Understanding these will help you use the library effectively and create custom components.

## The Evolutionary Loop

At its core, genetic algorithms follow an iterative process:

```text
┌─────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY LOOP                      │
│                                                           │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │Initialize│ -> │ Evaluate  │ -> │    Selection     │  │
│  │Population│    │  Fitness  │    │                  │  │
│  └──────────┘    └───────────┘    └────────┬─────────┘  │
│                                             │            │
│  ┌──────────┐    ┌───────────┐    ┌────────▼─────────┐  │
│  │  Check   │ <- │  Replace  │ <- │    Crossover     │  │
│  │Terminate │    │Population │    │   & Mutation     │  │
│  └──────────┘    └───────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Fugue-evo provides abstractions for each step while handling the loop orchestration.

## Genomes

A **genome** represents a candidate solution to your optimization problem. The `EvolutionaryGenome` trait is the core abstraction:

```rust,ignore
pub trait EvolutionaryGenome: Clone + Send + Sync {
    /// Convert genome to a Fugue trace for probabilistic operations
    fn to_trace(&self) -> Trace;

    /// Reconstruct genome from a trace
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;
}
```

### Built-in Genome Types

| Type | Use Case | Example |
|------|----------|---------|
| `RealVector` | Continuous optimization | Function minimization |
| `BitString` | Binary optimization | Feature selection, knapsack |
| `Permutation` | Ordering problems | TSP, job scheduling |
| `TreeGenome` | Genetic programming | Symbolic regression |

### Example: RealVector

```rust,ignore
use fugue_evo::prelude::*;

// Create a 5-dimensional real vector
let genome = RealVector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Access genes
println!("Genes: {:?}", genome.genes());
println!("Dimension: {}", genome.dimension());

// Vector operations
let other = RealVector::new(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
let distance = genome.euclidean_distance(&other);
```

## Fitness Functions

A **fitness function** evaluates how good a solution is. Higher fitness = better solution.

```rust,ignore
pub trait Fitness<G>: Send + Sync {
    type Value: FitnessValue;

    fn evaluate(&self, genome: &G) -> Self::Value;
}
```

### Built-in Benchmarks

Fugue-evo includes standard optimization benchmarks:

```rust,ignore
// Sphere function: f(x) = Σxᵢ²
// Global minimum at origin
let sphere = Sphere::new(10);

// Rastrigin function: highly multimodal
let rastrigin = Rastrigin::new(10);

// Rosenbrock function: curved valley
let rosenbrock = Rosenbrock::new(10);
```

### Custom Fitness

Implement `Fitness` for your own evaluation:

```rust,ignore
struct MyFitness;

impl Fitness<RealVector> for MyFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        // Your evaluation logic here
        let genes = genome.genes();
        // Return fitness (higher = better)
        -genes.iter().map(|x| x * x).sum::<f64>()
    }
}
```

## Operators

Operators manipulate genomes to explore the solution space.

### Selection

Selection chooses parents for reproduction based on fitness:

```rust,ignore
// Tournament: randomly sample k individuals, select best
let selection = TournamentSelection::new(3);

// Roulette wheel: probability proportional to fitness
let selection = RouletteWheelSelection::new();

// Rank-based: probability based on fitness rank
let selection = RankSelection::new();
```

### Crossover

Crossover combines two parents to create offspring:

```rust,ignore
// SBX: Simulated Binary Crossover for real values
let crossover = SbxCrossover::new(20.0); // eta parameter

// Uniform: swap genes with 50% probability
let crossover = UniformCrossover::new();

// One-point: single crossover point
let crossover = OnePointCrossover::new();
```

### Mutation

Mutation introduces random variation:

```rust,ignore
// Polynomial mutation for real values
let mutation = PolynomialMutation::new(20.0);

// Gaussian noise
let mutation = GaussianMutation::new(0.1); // std dev

// Bit-flip for binary genomes
let mutation = BitFlipMutation::new(0.01); // per-bit rate
```

## Bounds

For continuous optimization, bounds constrain the search space:

```rust,ignore
// Symmetric bounds: [-5.12, 5.12] for all dimensions
let bounds = MultiBounds::symmetric(5.12, 10);

// Asymmetric bounds per dimension
let bounds = MultiBounds::new(vec![
    Bounds::new(-10.0, 10.0),
    Bounds::new(0.0, 100.0),
    Bounds::new(-1.0, 1.0),
]);

// Uniform bounds for all dimensions
let bounds = MultiBounds::uniform(Bounds::new(0.0, 1.0), 5);
```

## Population

A **population** is a collection of individuals being evolved:

```rust,ignore
// Create random population within bounds
let mut population: Population<RealVector, f64> =
    Population::random(100, &bounds, &mut rng);

// Evaluate all individuals
population.evaluate(&fitness);

// Access best individual
if let Some(best) = population.best() {
    println!("Best fitness: {}", best.fitness_value());
}
```

## Termination Criteria

Define when evolution should stop:

```rust,ignore
// Fixed number of generations
let term = MaxGenerations::new(500);

// Target fitness achieved
let term = TargetFitness::new(-0.001); // for minimization

// Fitness stagnation (no improvement for N generations)
let term = FitnessStagnation::new(50);

// Combine criteria
let term = AnyOf::new(vec![
    Box::new(MaxGenerations::new(1000)),
    Box::new(TargetFitness::new(0.0)),
]);
```

## Builder Pattern

Algorithms are configured using type-safe builders:

```rust,ignore
let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)      // Population size
    .bounds(bounds)            // Search space
    .selection(selection)      // Selection operator
    .crossover(crossover)      // Crossover operator
    .mutation(mutation)        // Mutation operator
    .fitness(fitness)          // Fitness function
    .max_generations(200)      // Termination criterion
    .elitism(true)             // Keep best individual
    .elite_count(2)            // Number of elites
    .build()?                  // Build algorithm
    .run(&mut rng)?;           // Execute
```

## Results

After evolution completes, results provide comprehensive statistics:

```rust,ignore
let result = ga.run(&mut rng)?;

println!("Best fitness: {}", result.best_fitness);
println!("Best genome: {:?}", result.best_genome);
println!("Generations: {}", result.generations);
println!("Evaluations: {}", result.evaluations);

// Detailed statistics
println!("{}", result.stats.summary());
```

## Next Steps

Now that you understand the core concepts:

- [Quick Start](./quickstart.md) - See these concepts in action
- [Your First Optimization](./first-optimization.md) - Build a complete example
