# Crossover Operators Reference

Crossover operators combine two parents to create offspring.

## Module

```rust,ignore
use fugue_evo::operators::crossover::{
    SbxCrossover, UniformCrossover, OnePointCrossover, TwoPointCrossover
};
```

## SBX (Simulated Binary Crossover)

For real-valued genomes. Simulates single-point crossover behavior.

```rust,ignore
let crossover = SbxCrossover::new(eta);
```

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `eta` | `f64` | 1-100+ | Distribution index |

**Distribution index effects:**
- Low eta (1-5): Large spread, more exploration
- Medium eta (15-25): Balanced
- High eta (50+): Children near parents

**Example:**
```rust,ignore
let crossover = SbxCrossover::new(20.0);
```

## Uniform Crossover

Each gene from random parent with 50% probability.

```rust,ignore
let crossover = UniformCrossover::new();
// or with custom probability
let crossover = UniformCrossover::with_probability(0.6);
```

**Works with:** `RealVector`, `BitString`

## One-Point Crossover

Single crossover point, swap tails.

```rust,ignore
let crossover = OnePointCrossover::new();
```

```
Parent 1: A A A | B B B
Parent 2: 1 1 1 | 2 2 2
                ↓
Child 1:  A A A | 2 2 2
Child 2:  1 1 1 | B B B
```

**Works with:** `RealVector`, `BitString`

## Two-Point Crossover

Two crossover points, swap middle segment.

```rust,ignore
let crossover = TwoPointCrossover::new();
```

```
Parent 1: A A | B B B | C C
Parent 2: 1 1 | 2 2 2 | 3 3
              ↓
Child 1:  A A | 2 2 2 | C C
Child 2:  1 1 | B B B | 3 3
```

## Order Crossover (OX)

For permutations. Preserves relative ordering.

```rust,ignore
let crossover = OrderCrossover::new();
```

**Preserves:** Relative order of elements not in copied segment.

## PMX (Partially Mapped Crossover)

For permutations. Preserves absolute positions.

```rust,ignore
let crossover = PmxCrossover::new();
```

**Preserves:** Absolute positions where possible.

## Crossover Result

All crossover operators return:

```rust,ignore
pub struct CrossoverResult<G> {
    child1: Option<G>,
    child2: Option<G>,
}

impl<G> CrossoverResult<G> {
    pub fn genome(self) -> Option<(G, G)>;
    pub fn first(self) -> Option<G>;
    pub fn second(self) -> Option<G>;
}
```

## Usage

```rust,ignore
SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
    .crossover(SbxCrossover::new(20.0))
    // ...
```

## Comparison

| Operator | Genome Type | Exploration | Preserves |
|----------|-------------|-------------|-----------|
| SBX | RealVector | Controllable | Value ranges |
| Uniform | Any | High | Nothing specific |
| One-Point | Sequence | Medium | Segments |
| Two-Point | Sequence | Medium | Segments |
| OX | Permutation | Medium | Relative order |
| PMX | Permutation | Medium | Positions |

## See Also

- [Custom Operators](../../how-to/custom-operators.md)
