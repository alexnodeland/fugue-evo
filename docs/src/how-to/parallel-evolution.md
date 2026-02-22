# Parallel Evolution

This guide shows how to speed up evolution using parallel computation.

## Enabling Parallelism

Parallel evaluation is enabled by default with the `parallel` feature:

```toml
[dependencies]
fugue-evo = { version = "0.1", features = ["parallel"] }
```

## Parallel Fitness Evaluation

The simplest form of parallelism: evaluate multiple individuals simultaneously.

### Using SimpleGA

```rust,ignore
let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .parallel(true)  // Enable parallel evaluation
    // ... other configuration
    .build()?
    .run(&mut rng)?;
```

### Manual Parallel Evaluation

Using Rayon directly:

```rust,ignore
use rayon::prelude::*;

// Parallel evaluation
let fitnesses: Vec<f64> = population
    .par_iter()  // Parallel iterator
    .map(|individual| fitness.evaluate(individual.genome()))
    .collect();

// Update population with fitnesses
for (individual, fit) in population.iter_mut().zip(fitnesses) {
    individual.set_fitness(fit);
}
```

## When Parallelism Helps

### Good Candidates

| Scenario | Speedup Potential |
|----------|-------------------|
| Expensive fitness (>10ms) | High |
| Large population (>100) | Medium-High |
| Independent evaluations | High |
| Many CPU cores | Scales with cores |

### Poor Candidates

| Scenario | Why |
|----------|-----|
| Very cheap fitness (<1ms) | Parallelization overhead dominates |
| Small population (<20) | Not enough work to distribute |
| Dependent evaluations | Can't parallelize |
| Memory-bound fitness | Cores compete for memory |

## Tuning Parallelism

### Thread Pool Size

```rust,ignore
// Set number of threads
rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build_global()
    .unwrap();
```

### Chunk Size

For fine-grained control:

```rust,ignore
use rayon::prelude::*;

// Process in chunks
population
    .par_chunks(10)  // Process 10 at a time
    .for_each(|chunk| {
        for individual in chunk {
            // Evaluate
        }
    });
```

## Island Model Parallelism

Islands naturally parallelize across populations:

```rust,ignore
use rayon::prelude::*;

// Each island evolves independently
islands.par_iter_mut().for_each(|island| {
    island.evolve_generation(&mut thread_rng());
});

// Synchronous migration
migrate_between_islands(&mut islands);
```

## Thread Safety Considerations

### Fitness Functions

Your fitness function must be thread-safe (`Send + Sync`):

```rust,ignore
// Good: Stateless fitness
struct MyFitness;

impl Fitness<RealVector> for MyFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        // Pure computation, no shared state
        genome.genes().iter().sum()
    }
}

// Good: Shared read-only data
struct DataDrivenFitness {
    data: Arc<Vec<f64>>,  // Shared immutable data
}

// Bad: Mutable shared state
struct BadFitness {
    counter: Cell<usize>,  // Not thread-safe!
}
```

### Thread-Local State

For fitness functions needing mutable state:

```rust,ignore
use std::cell::RefCell;

thread_local! {
    static CACHE: RefCell<HashMap<u64, f64>> = RefCell::new(HashMap::new());
}

impl Fitness<RealVector> for CachedFitness {
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        let key = hash(genome);

        CACHE.with(|cache| {
            if let Some(&cached) = cache.borrow().get(&key) {
                return cached;
            }

            let result = expensive_compute(genome);
            cache.borrow_mut().insert(key, result);
            result
        })
    }
}
```

## Benchmarking Parallelism

Measure actual speedup:

```rust,ignore
use std::time::Instant;

fn benchmark_parallel_vs_sequential(population: &[Individual<RealVector>], fitness: &impl Fitness<RealVector>) {
    // Sequential
    let start = Instant::now();
    for ind in population {
        fitness.evaluate(ind.genome());
    }
    let sequential_time = start.elapsed();

    // Parallel
    let start = Instant::now();
    population.par_iter().for_each(|ind| {
        fitness.evaluate(ind.genome());
    });
    let parallel_time = start.elapsed();

    println!("Sequential: {:?}", sequential_time);
    println!("Parallel: {:?}", parallel_time);
    println!("Speedup: {:.2}x", sequential_time.as_secs_f64() / parallel_time.as_secs_f64());
}
```

## GPU Acceleration

For very expensive computations, consider GPU:

```rust,ignore
// Pseudocode for GPU fitness
struct GpuFitness {
    device: CudaDevice,
    kernel: CompiledKernel,
}

impl GpuFitness {
    fn evaluate_batch(&self, genomes: &[RealVector]) -> Vec<f64> {
        // Copy genomes to GPU
        let gpu_data = self.device.copy_to_device(genomes);

        // Run kernel
        let gpu_results = self.kernel.run(&gpu_data);

        // Copy results back
        self.device.copy_from_device(&gpu_results)
    }
}
```

## Parallel Evolution Patterns

### Pattern 1: Parallel Evaluation Only

```rust,ignore
for gen in 0..max_generations {
    // Sequential selection and reproduction
    let offspring = create_offspring(&population, &mut rng);

    // Parallel evaluation
    offspring.par_iter_mut().for_each(|ind| {
        let fit = fitness.evaluate(ind.genome());
        ind.set_fitness(fit);
    });

    population = offspring;
}
```

### Pattern 2: Multiple Independent Runs

```rust,ignore
let results: Vec<_> = (0..num_runs)
    .into_par_iter()
    .map(|run| {
        let mut rng = StdRng::seed_from_u64(run as u64);
        run_evolution(&fitness, &mut rng)
    })
    .collect();

let best = results.iter()
    .max_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap());
```

### Pattern 3: Speculative Execution

Run multiple strategies, keep best:

```rust,ignore
let strategies = vec![
    Strategy::HighMutation,
    Strategy::HighCrossover,
    Strategy::Balanced,
];

let results: Vec<_> = strategies
    .par_iter()
    .map(|strategy| {
        run_with_strategy(&fitness, strategy, &mut thread_rng())
    })
    .collect();
```

## Common Issues

### Race Conditions

**Symptom**: Non-deterministic results, crashes

**Solution**: Ensure thread-safe fitness functions

### High Overhead

**Symptom**: Parallel slower than sequential

**Solution**:
- Increase population size
- Use more expensive fitness
- Reduce thread count

### Memory Pressure

**Symptom**: Slowdown with many threads

**Solution**:
- Limit thread count
- Reduce population size
- Stream results instead of collecting

## Next Steps

- [Island Model Tutorial](../tutorials/island-model.md) - Natural parallelism
- [Checkpointing](./checkpointing.md) - Save parallel runs
