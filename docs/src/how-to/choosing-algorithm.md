# Choosing an Algorithm

This guide helps you select the right optimization algorithm for your problem.

## Quick Decision Guide

### By Problem Type

| Problem Type | Recommended | Alternative |
|-------------|-------------|-------------|
| Continuous, unimodal | SimpleGA or CMA-ES | - |
| Continuous, multimodal | Island Model | CMA-ES with restarts |
| Multi-objective | NSGA-II | - |
| Discrete/binary | SimpleGA + BitString | EDA |
| Permutation | SimpleGA + Permutation | - |
| Symbolic/GP | SimpleGA + TreeGenome | - |
| Subjective fitness | Interactive GA | - |

### By Problem Size

| Dimensions | Recommended | Notes |
|------------|-------------|-------|
| 1-10 | Any | All algorithms work well |
| 10-100 | CMA-ES | Learns correlations efficiently |
| 100-1000 | SimpleGA | CMA-ES memory-intensive |
| 1000+ | SimpleGA + parallel | Consider dimensionality reduction |

### By Evaluation Cost

| Cost | Recommended | Strategy |
|------|-------------|----------|
| Very cheap (< 1ms) | Any | Large populations OK |
| Cheap (1-100ms) | Any with parallel | Enable Rayon parallelism |
| Expensive (1-10s) | CMA-ES | Fewer evaluations needed |
| Very expensive (> 10s) | Surrogate + CMA-ES | Approximate fitness |

## Algorithm Details

### SimpleGA

**Best for:** General-purpose optimization with custom requirements

```rust,ignore
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .population_size(100)
    .bounds(bounds)
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(fitness)
    .max_generations(200)
    .build()?
```

**Pros:**
- Highly configurable
- Works with any genome type
- Custom operators supported

**Cons:**
- Requires operator selection
- May need parameter tuning

### CMA-ES

**Best for:** Non-separable continuous optimization

```rust,ignore
let mut cmaes = CmaEs::new(initial_mean, initial_sigma)
    .with_bounds(bounds);
let best = cmaes.run_generations(&fitness, 1000, &mut rng)?;
```

**Pros:**
- Learns problem structure automatically
- Often needs fewer evaluations
- Self-adapting parameters

**Cons:**
- O(n²) memory for covariance matrix
- Only works with continuous problems
- Fixed algorithm structure

### NSGA-II

**Best for:** Multi-objective optimization

```rust,ignore
let result = Nsga2Builder::new()
    .population_size(100)
    .objectives(objectives)
    .max_generations(200)
    .build()?
    .run(&mut rng)?;

// Access Pareto front
for individual in result.pareto_front {
    println!("{:?}", individual.objectives);
}
```

**Pros:**
- Finds entire Pareto front
- Maintains diversity
- Well-established algorithm

**Cons:**
- Only for multi-objective
- More complex setup

### Island Model

**Best for:** Highly multimodal problems

```rust,ignore
let model = IslandModelBuilder::new()
    .num_islands(4)
    .island_population_size(50)
    .topology(MigrationTopology::Ring)
    .migration_interval(25)
    .build(&mut rng)?;
```

**Pros:**
- Maintains diversity
- Natural parallelism
- Escapes local optima

**Cons:**
- More parameters to tune
- Higher total population
- Migration overhead

### Interactive GA

**Best for:** Subjective or hard-to-formalize fitness

```rust,ignore
let mut iga = InteractiveGABuilder::new()
    .population_size(12)
    .evaluation_mode(EvaluationMode::Rating)
    .batch_size(4)
    .build()?;
```

**Pros:**
- No fitness function needed
- Captures human preferences
- Creative exploration

**Cons:**
- Slow (human in loop)
- Limited evaluations
- Inconsistent feedback

## Decision Flowchart

```text
START
  │
  ▼
Is fitness subjective/aesthetic?
  │
  ├─ Yes → Interactive GA
  │
  └─ No
      │
      ▼
    Multiple conflicting objectives?
      │
      ├─ Yes → NSGA-II
      │
      └─ No
          │
          ▼
        Problem type?
          │
          ├─ Continuous → Is it non-separable or ill-conditioned?
          │                 │
          │                 ├─ Yes → CMA-ES
          │                 │
          │                 └─ No → Is it highly multimodal?
          │                           │
          │                           ├─ Yes → Island Model or CMA-ES + restarts
          │                           │
          │                           └─ No → SimpleGA
          │
          ├─ Discrete/Binary → SimpleGA + BitString
          │
          ├─ Permutation → SimpleGA + Permutation
          │
          └─ Trees/Programs → SimpleGA + TreeGenome
```

## Combining Algorithms

Sometimes the best approach combines multiple algorithms:

### CMA-ES After GA

Use GA for global exploration, CMA-ES for local refinement:

```rust,ignore
// Phase 1: Broad search with GA
let ga_result = SimpleGABuilder::new()
    .max_generations(50)
    .build()?.run(&mut rng)?;

// Phase 2: Refine best solution with CMA-ES
let mut cmaes = CmaEs::new(ga_result.best_genome.genes().to_vec(), 0.1);
let final_result = cmaes.run_generations(&fitness, 100, &mut rng)?;
```

### Multiple Restarts

Run algorithm multiple times with different seeds:

```rust,ignore
let mut best_overall = f64::NEG_INFINITY;
for seed in 0..10 {
    let mut rng = StdRng::seed_from_u64(seed);
    let result = run_optimization(&mut rng)?;
    best_overall = best_overall.max(result.best_fitness);
}
```

## Performance Benchmarks

Typical performance on standard benchmarks (your results may vary):

| Function | Dims | SimpleGA | CMA-ES | Island |
|----------|------|----------|--------|--------|
| Sphere | 10 | ~200 gen | ~50 gen | ~150 gen |
| Rastrigin | 10 | ~300 gen | ~100 gen | ~200 gen |
| Rosenbrock | 10 | ~500 gen | ~80 gen | ~400 gen |

Note: Generations aren't directly comparable (different evaluation counts).

## Next Steps

Once you've chosen an algorithm:

- [Custom Fitness Functions](./custom-fitness.md) - Implement your objective
- [Custom Operators](./custom-operators.md) - Tune genetic operators
- [Parallel Evolution](./parallel-evolution.md) - Speed up evaluation
