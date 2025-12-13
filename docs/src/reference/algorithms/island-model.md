# Island Model Reference

Parallel evolution with multiple populations and migration.

## Module

```rust,ignore
use fugue_evo::algorithms::island::{
    IslandModel, IslandModelBuilder, IslandConfig,
    MigrationTopology, MigrationPolicy
};
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

### Island Configuration

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `num_islands(n)` | `usize` | 4 | Number of islands |
| `island_population_size(n)` | `usize` | 50 | Population per island |
| `topology(t)` | `MigrationTopology` | Ring | Connection pattern |
| `migration_interval(n)` | `usize` | 25 | Generations between migrations |
| `migration_policy(p)` | `MigrationPolicy` | Best(2) | What to migrate |

## Migration Topology

```rust,ignore
pub enum MigrationTopology {
    /// Each island connects to 2 neighbors
    Ring,
    /// Central hub connects to all
    Star,
    /// Every island connects to every other
    FullyConnected,
}
```

### Ring Topology
```
1 ←→ 2
↕     ↕
4 ←→ 3
```

### Star Topology
```
  2   3
   \ /
    1 (hub)
   / \
  5   4
```

### Fully Connected
```
1 ←→ 2
↕ ✕ ↕
4 ←→ 3
```

## Migration Policy

```rust,ignore
pub enum MigrationPolicy {
    /// Send n best individuals
    Best(usize),
    /// Send n random individuals
    Random(usize),
    /// Replace n worst with immigrants
    ReplaceWorst(usize),
}
```

## Usage

### Basic Example

```rust,ignore
use fugue_evo::prelude::*;

let mut model = IslandModelBuilder::<RealVector, _, _, _, _, f64>::new()
    .num_islands(4)
    .island_population_size(50)
    .topology(MigrationTopology::Ring)
    .migration_interval(25)
    .migration_policy(MigrationPolicy::Best(2))
    .bounds(MultiBounds::symmetric(5.12, 20))
    .selection(TournamentSelection::new(3))
    .crossover(SbxCrossover::new(15.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(Rastrigin::new(20))
    .build(&mut rng)?;

let best = model.run(200, &mut rng)?;
println!("Best fitness: {}", best.fitness_value());
```

### Step-by-Step

```rust,ignore
let mut model = IslandModelBuilder::new()
    // ... configuration
    .build(&mut rng)?;

for gen in 0..max_generations {
    model.step(&mut rng)?;

    if gen % 50 == 0 {
        println!("Generation {}: best on each island:", gen);
        for (i, island) in model.islands.iter().enumerate() {
            let best = island.best().unwrap();
            println!("  Island {}: {:.6}", i, best.fitness_value());
        }
    }
}
```

### Force Migration

```rust,ignore
// Trigger migration outside normal interval
model.migrate(&mut rng);
```

## State Access

```rust,ignore
pub struct IslandModel<G, F, S, C, M, Fit> {
    pub islands: Vec<Population<G, F>>,
    pub generation: usize,
    pub config: IslandConfig,
    // ...
}
```

## Configuration Struct

```rust,ignore
pub struct IslandConfig {
    pub num_islands: usize,
    pub island_population_size: usize,
    pub topology: MigrationTopology,
    pub migration_interval: usize,
    pub migration_policy: MigrationPolicy,
}
```

## Migration Mechanics

Each migration event:
1. Each island selects emigrants (per policy)
2. Emigrants sent to neighbors (per topology)
3. Immigrants replace worst individuals

Total population remains constant.

## Performance Tips

1. **Island count**: 4-8 typically sufficient
2. **Migration interval**: 20-50 generations
3. **Migration size**: 1-5 individuals
4. **Topology**: Ring for diversity, Fully Connected for speed

## See Also

- [Island Model Tutorial](../../tutorials/island-model.md)
- [Parallel Evolution](../../how-to/parallel-evolution.md)
