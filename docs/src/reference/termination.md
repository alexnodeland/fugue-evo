# Termination Criteria Reference

Termination criteria define when evolution should stop.

## Module

```rust,ignore
use fugue_evo::termination::{
    MaxGenerations, MaxEvaluations, TargetFitness,
    FitnessStagnation, DiversityThreshold, AnyOf, AllOf
};
```

## Trait

```rust,ignore
pub trait TerminationCriterion<G, F>: Send + Sync {
    fn should_terminate(&self, state: &EvolutionState<G, F>) -> bool;
}
```

## Built-in Criteria

### MaxGenerations

Stop after N generations.

```rust,ignore
let term = MaxGenerations::new(500);
```

### MaxEvaluations

Stop after N fitness evaluations.

```rust,ignore
let term = MaxEvaluations::new(100000);
```

### TargetFitness

Stop when fitness threshold reached.

```rust,ignore
let term = TargetFitness::new(threshold);
```

For minimization (negated fitness):
```rust,ignore
let term = TargetFitness::new(-0.001);  // Stop when fitness >= -0.001
```

### FitnessStagnation

Stop if no improvement for N generations.

```rust,ignore
let term = FitnessStagnation::new(generations);
```

| Parameter | Description |
|-----------|-------------|
| `generations` | Generations without improvement |

### DiversityThreshold

Stop when population diversity falls below threshold.

```rust,ignore
let term = DiversityThreshold::new(threshold);
```

## Combinators

### AnyOf

Stop when ANY criterion is met (OR).

```rust,ignore
let term = AnyOf::new(vec![
    Box::new(MaxGenerations::new(1000)),
    Box::new(TargetFitness::new(-0.001)),
    Box::new(FitnessStagnation::new(50)),
]);
```

### AllOf

Stop when ALL criteria are met (AND).

```rust,ignore
let term = AllOf::new(vec![
    Box::new(MinGenerations::new(100)),  // At least 100 generations
    Box::new(FitnessStagnation::new(20)), // And stagnated
]);
```

## Usage

### With Builder

```rust,ignore
SimpleGABuilder::new()
    .max_generations(200)  // Shorthand for MaxGenerations
    // ...
```

### Custom Termination

```rust,ignore
SimpleGABuilder::new()
    .termination(AnyOf::new(vec![
        Box::new(MaxGenerations::new(1000)),
        Box::new(TargetFitness::new(-0.001)),
    ]))
    // ...
```

## Termination Reason

Results include termination reason:

```rust,ignore
let result = ga.run(&mut rng)?;
println!("Stopped because: {:?}", result.termination_reason);
```

## Custom Criteria

```rust,ignore
struct TimeBudget {
    start: Instant,
    budget: Duration,
}

impl<G, F> TerminationCriterion<G, F> for TimeBudget {
    fn should_terminate(&self, _state: &EvolutionState<G, F>) -> bool {
        self.start.elapsed() >= self.budget
    }
}

// Usage
let term = TimeBudget {
    start: Instant::now(),
    budget: Duration::from_secs(60),
};
```

## Recommendations

| Scenario | Recommended |
|----------|-------------|
| Unknown runtime | `AnyOf(MaxGen, Stagnation)` |
| Known optimum | `AnyOf(MaxGen, TargetFitness)` |
| Time-limited | Custom `TimeBudget` |
| Research | `MaxGenerations` for reproducibility |

## See Also

- [SimpleGA Reference](./algorithms/simple-ga.md)
- [Checkpointing](../how-to/checkpointing.md) - Resume after termination
