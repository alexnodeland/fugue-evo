# Checkpointing & Recovery

This guide shows how to save and restore evolution state for long-running optimizations.

## Why Checkpointing?

- **Resume interrupted runs**: Continue after crashes or shutdowns
- **Experiment branching**: Try different strategies from the same point
- **Progress monitoring**: Analyze intermediate states
- **Resource limits**: Work within time-limited environments

## Basic Checkpointing

### Creating Checkpoints

```rust,ignore
use fugue_evo::prelude::*;
use std::path::PathBuf;

// Create checkpoint manager
let checkpoint_dir = PathBuf::from("./checkpoints");
let mut manager = CheckpointManager::new(&checkpoint_dir, "my_evolution")
    .every(50)    // Save every 50 generations
    .keep(3);     // Keep last 3 checkpoints

// In evolution loop
for gen in 0..max_generations {
    // ... evolution step ...

    if manager.should_save(gen + 1) {
        let individuals: Vec<Individual<RealVector>> = population.iter().cloned().collect();
        let checkpoint = Checkpoint::new(gen + 1, individuals)
            .with_evaluations((gen + 1) * population_size);

        manager.save(&checkpoint)?;
        println!("Saved checkpoint at generation {}", gen + 1);
    }
}
```

### Loading Checkpoints

```rust,ignore
use fugue_evo::checkpoint::load_checkpoint;

// Load specific checkpoint
let checkpoint: Checkpoint<RealVector> = load_checkpoint("./checkpoints/my_evolution_gen_100.ckpt")?;

println!("Loaded generation: {}", checkpoint.generation);
println!("Population size: {}", checkpoint.population.len());
println!("Evaluations: {}", checkpoint.evaluations);

// Reconstruct population
let mut population: Population<RealVector, f64> =
    Population::with_capacity(checkpoint.population.len());
for ind in checkpoint.population {
    population.push(ind);
}
```

## Complete Example

```rust,ignore
{{#include ../../../examples/checkpointing.rs}}
```

> **Source**: [`examples/checkpointing.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/checkpointing.rs)

## Running the Example

```bash
cargo run --example checkpointing
```

## Checkpoint Manager Options

### Save Frequency

```rust,ignore
// Every N generations
CheckpointManager::new(&dir, "name").every(50);

// Only at specific generations
CheckpointManager::new(&dir, "name").at_generations(&[100, 200, 500]);
```

### Retention Policy

```rust,ignore
// Keep last N checkpoints
CheckpointManager::new(&dir, "name").keep(3);

// Keep all checkpoints
CheckpointManager::new(&dir, "name").keep_all();

// Custom retention
CheckpointManager::new(&dir, "name").keep_every(100); // Keep every 100th
```

### Custom Naming

```rust,ignore
// Default: name_gen_N.ckpt
let manager = CheckpointManager::new(&dir, "experiment_1");
// Creates: experiment_1_gen_50.ckpt, experiment_1_gen_100.ckpt, etc.
```

## Checkpoint Contents

The `Checkpoint` struct stores:

```rust,ignore
pub struct Checkpoint<G: EvolutionaryGenome> {
    /// Current generation number
    pub generation: usize,

    /// Full population with fitness values
    pub population: Vec<Individual<G>>,

    /// Total fitness evaluations so far
    pub evaluations: usize,

    /// Optional metadata
    pub metadata: Option<CheckpointMetadata>,
}
```

### Adding Metadata

```rust,ignore
let checkpoint = Checkpoint::new(gen, individuals)
    .with_evaluations(evaluations)
    .with_metadata(CheckpointMetadata {
        timestamp: chrono::Utc::now(),
        best_fitness: population.best().map(|b| *b.fitness_value()),
        config: serde_json::to_string(&config).ok(),
    });
```

## Resume Strategy

### Find Latest Checkpoint

```rust,ignore
fn find_latest_checkpoint(dir: &Path, prefix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(prefix) && n.ends_with(".ckpt"))
                .unwrap_or(false)
        })
        .max_by_key(|e| e.path())
        .map(|e| e.path())
}
```

### Resume or Start Fresh

```rust,ignore
fn run_evolution(checkpoint_dir: &Path) -> Result<(), Box<dyn Error>> {
    let latest = find_latest_checkpoint(checkpoint_dir, "my_evolution");

    let (mut population, start_gen) = if let Some(path) = latest {
        println!("Resuming from {:?}", path);
        let ckpt: Checkpoint<RealVector> = load_checkpoint(&path)?;
        let pop = reconstruct_population(ckpt.population);
        (pop, ckpt.generation)
    } else {
        println!("Starting fresh");
        let pop = Population::random(100, &bounds, &mut rng);
        (pop, 0)
    };

    // Continue evolution from start_gen
    for gen in start_gen..max_generations {
        // ... evolution ...
    }

    Ok(())
}
```

## Saving Algorithm State

For algorithms with internal state (like CMA-ES):

```rust,ignore
#[derive(Serialize, Deserialize)]
struct CmaEsCheckpoint {
    generation: usize,
    mean: Vec<f64>,
    sigma: f64,
    covariance: Vec<Vec<f64>>,
    // ... other CMA-ES state
}

impl CmaEsCheckpoint {
    fn from_cmaes(cmaes: &CmaEs) -> Self {
        Self {
            generation: cmaes.state.generation,
            mean: cmaes.state.mean.clone(),
            sigma: cmaes.state.sigma,
            covariance: cmaes.state.covariance.clone(),
        }
    }

    fn restore(&self) -> CmaEs {
        let mut cmaes = CmaEs::new(self.mean.clone(), self.sigma);
        cmaes.state.generation = self.generation;
        cmaes.state.covariance = self.covariance.clone();
        cmaes
    }
}
```

## Error Handling

```rust,ignore
match load_checkpoint::<RealVector>(&path) {
    Ok(checkpoint) => {
        println!("Loaded successfully");
    }
    Err(CheckpointError::FileNotFound(path)) => {
        println!("Checkpoint not found: {:?}", path);
    }
    Err(CheckpointError::DeserializationFailed(err)) => {
        println!("Corrupted checkpoint: {}", err);
    }
    Err(e) => {
        println!("Unknown error: {}", e);
    }
}
```

## Best Practices

### 1. Checkpoint Frequently Enough

```rust,ignore
// For long runs, checkpoint every ~5-10% of expected runtime
let interval = max_generations / 20;
manager.every(interval);
```

### 2. Verify Checkpoints

```rust,ignore
// After saving, verify it can be loaded
manager.save(&checkpoint)?;
let verified: Checkpoint<RealVector> = load_checkpoint(&manager.latest_path())?;
assert_eq!(verified.generation, checkpoint.generation);
```

### 3. Include Random State

For reproducible resumption, save RNG state:

```rust,ignore
use rand::SeedableRng;

#[derive(Serialize, Deserialize)]
struct FullCheckpoint<G> {
    evolution: Checkpoint<G>,
    rng_seed: u64, // Or full RNG state
}
```

### 4. Use Atomic Writes

Prevent corruption from interrupted saves:

```rust,ignore
// Write to temp file, then rename
let temp_path = path.with_extension("tmp");
write_checkpoint(&checkpoint, &temp_path)?;
std::fs::rename(temp_path, path)?;
```

## Feature Flag

Checkpointing requires the `checkpoint` feature:

```toml
[dependencies]
fugue-evo = { version = "0.1", features = ["checkpoint"] }
```

## Next Steps

- [Parallel Evolution](./parallel-evolution.md) - Speed up evolution
- [Custom Genome Types](./custom-genome.md) - Ensure your genomes are serializable
