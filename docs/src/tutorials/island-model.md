# Island Model Tutorial

The **Island Model** runs multiple populations (islands) in parallel with periodic migration of individuals between them. This approach helps maintain diversity and escape local optima in multimodal problems.

## When to Use Island Model

**Ideal for:**
- Highly multimodal problems
- When single-population GA gets trapped
- Parallel computation environments
- Problems requiring diversity maintenance

**Trade-offs:**
- More complex setup
- Migration parameters require tuning
- Higher total population size

## How It Works

```text
┌─────────────────────────────────────────────────────────────┐
│                       ISLAND MODEL                           │
│                                                              │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│   │ Island 1│◄────►│ Island 2│◄────►│ Island 3│           │
│   │  Pop=50 │      │  Pop=50 │      │  Pop=50 │           │
│   └────┬────┘      └─────────┘      └────┬────┘           │
│        │                                  │                 │
│        └──────────────────────────────────┘                │
│                      Migration                              │
│                  (every N generations)                      │
└─────────────────────────────────────────────────────────────┘
```

Each island:
1. Evolves independently for several generations
2. Periodically sends/receives individuals to/from neighbors
3. Continues evolving with new genetic material

## Complete Example

```rust,ignore
{{#include ../../../examples/island_model.rs}}
```

> **Source**: [`examples/island_model.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/island_model.rs)

## Running the Example

```bash
cargo run --example island_model
```

## Key Components

### Configuration

```rust,ignore
let mut island_model = IslandModelBuilder::<RealVector, _, _, _, _, f64>::new()
    .num_islands(4)                            // Number of populations
    .island_population_size(50)                // Size per island
    .topology(MigrationTopology::Ring)         // Connection pattern
    .migration_interval(25)                    // Generations between migrations
    .migration_policy(MigrationPolicy::Best(2)) // What to migrate
    .bounds(bounds.clone())
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(15.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(fitness)
    .build(&mut rng)?;
```

### Migration Topologies

```rust,ignore
MigrationTopology::Ring
```

| Topology | Pattern | Best For |
|----------|---------|----------|
| `Ring` | Each island connects to 2 neighbors | General purpose, good diversity |
| `Star` | Central hub connects to all | Fast information spread |
| `FullyConnected` | Everyone connects to everyone | Maximum mixing |

**Ring Topology:**
```text
    1 ←→ 2
    ↕     ↕
    4 ←→ 3
```

**Star Topology:**
```text
    2   3
     \ /
      1
     / \
    5   4
```

### Migration Policies

```rust,ignore
MigrationPolicy::Best(2)
```

| Policy | Description |
|--------|-------------|
| `Best(n)` | Send n best individuals |
| `Random(n)` | Send n random individuals |
| `Worst(n)` | Replace n worst with immigrants |

### Migration Interval

```rust,ignore
.migration_interval(25)
```

- **Short interval (5-10)**: Frequent mixing, faster convergence
- **Long interval (50-100)**: More independent evolution, more diversity
- **Typical**: 20-50 generations

## Understanding the Comparison

The example compares Island Model with a single population:

```rust,ignore
// Island Model: 4 islands × 50 = 200 total
let mut island_model = IslandModelBuilder::new()
    .num_islands(4)
    .island_population_size(50)
    // ...

// Single Population: 200 total
let single_result = SimpleGABuilder::new()
    .population_size(200)
    // ...
```

Same total population size, different structure. For multimodal problems like Rastrigin, islands often find better solutions because:

1. **Diversity preservation**: Islands explore different regions
2. **Niching effect**: Each island can specialize in a local optimum
3. **Genetic variety**: Migration introduces new genetic material

## Tuning Island Model

### Number of Islands

```rust,ignore
.num_islands(4)
```

**Guidelines:**
- 2-4 islands: Good for most problems
- 4-8 islands: Highly multimodal problems
- More islands = more diversity but slower convergence

### Island Population Size

```rust,ignore
.island_population_size(50)
```

Each island should be large enough to:
- Maintain genetic diversity
- Support effective selection
- Typically 30-100 individuals

### Migration Rate

The effective migration rate is:

```text
migration_rate = migrants_per_interval / (island_size × interval)
```

For `Best(2)` with size 50 and interval 25:
```text
rate = 2 / (50 × 25) = 0.16%
```

Too high: Islands become homogeneous
Too low: Islands don't share discoveries

## Advanced Patterns

### Heterogeneous Islands

Run different configurations on each island:

```rust,ignore
// This is a conceptual example
// Each island uses different operator parameters
let island_configs = vec![
    SbxCrossover::new(10.0),  // Explorative
    SbxCrossover::new(20.0),  // Balanced
    SbxCrossover::new(30.0),  // Exploitative
    SbxCrossover::new(15.0),  // Balanced
];
```

### Adaptive Migration

Adjust migration based on progress:

```rust,ignore
for gen in 0..max_generations {
    island_model.step(&mut rng)?;

    // Migrate more frequently if stuck
    if gen % check_interval == 0 {
        let improvement = check_improvement(&island_model);
        if improvement < threshold {
            island_model.force_migration(&mut rng);
        }
    }
}
```

## Performance Comparison

For the 20-D Rastrigin function:

| Configuration | Typical Best Fitness | Notes |
|---------------|---------------------|-------|
| Single Pop (200) | -15 to -25 | Often stuck in local optima |
| Island 4×50 | -5 to -15 | Better exploration |
| Island 8×25 | -8 to -18 | More diversity, slower convergence |

Results vary by run due to randomness.

## Exercises

1. **Topology comparison**: Compare Ring, Star, and FullyConnected on Rastrigin
2. **Migration interval**: Try intervals of 10, 25, 50, 100
3. **Island count**: Compare 2, 4, 8, 16 islands with same total population

## Next Steps

- [Genetic Programming Tutorial](./genetic-programming.md) - Tree-based evolution
- [Hyperparameter Learning](./hyperparameter-learning.md) - Adaptive parameter tuning
