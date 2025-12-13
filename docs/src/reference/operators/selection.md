# Selection Operators Reference

Selection operators choose parents for reproduction.

## Module

```rust,ignore
use fugue_evo::operators::selection::{
    TournamentSelection, RouletteWheelSelection, RankSelection
};
```

## Tournament Selection

Select best from k random individuals.

```rust,ignore
let selection = TournamentSelection::new(k);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `usize` | Tournament size |

**Characteristics:**
- Higher k = more selection pressure
- k=2 is minimal pressure
- k=5-7 for high pressure
- Efficient: O(k) per selection

**Example:**
```rust,ignore
let selection = TournamentSelection::new(3);
// Selects best of 3 random individuals
```

## Roulette Wheel Selection

Probability proportional to fitness.

```rust,ignore
let selection = RouletteWheelSelection::new();
```

**Characteristics:**
- Fitness proportionate
- Can lose selection pressure with similar fitnesses
- Sensitive to fitness scaling
- O(n) setup, O(log n) per selection

**Requirements:**
- Fitness must be positive
- Works best with transformed fitness

## Rank Selection

Probability based on fitness rank.

```rust,ignore
let selection = RankSelection::new();
// or with pressure parameter
let selection = RankSelection::with_pressure(2.0);
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pressure` | `f64` | 2.0 | Selection pressure (1.0-2.0) |

**Characteristics:**
- Robust to fitness scaling
- Maintains selection pressure
- O(n log n) for ranking
- Good for multimodal problems

## Comparison

| Operator | Pressure Control | Scaling Sensitive | Efficiency |
|----------|------------------|-------------------|------------|
| Tournament | Tournament size | No | O(k) |
| Roulette | Fitness transformation | Yes | O(log n) |
| Rank | Pressure parameter | No | O(n log n) |

## Usage

```rust,ignore
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .selection(TournamentSelection::new(3))
    // ...
```

## See Also

- [Custom Operators](../../how-to/custom-operators.md) - Create custom selection
