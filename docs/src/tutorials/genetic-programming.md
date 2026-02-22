# Genetic Programming Tutorial

**Genetic Programming (GP)** evolves tree-structured programs rather than fixed-length vectors. This tutorial demonstrates symbolic regression - discovering mathematical expressions that fit data.

## When to Use Genetic Programming

**Ideal for:**
- Symbolic regression (finding formulas)
- Automated program synthesis
- Discovery of mathematical relationships
- Feature construction for machine learning

**Challenges:**
- Bloat (trees grow unnecessarily large)
- More complex operators
- Larger search spaces

## The Problem: Symbolic Regression

Given input-output pairs, find a mathematical expression that fits the data.

**Target function:**
```text
f(x) = x² + 2x + 1
```

**Training data:**
```text
x: -5, -4, ..., 4, 5
y: f(x) for each x
```

**Goal:** Rediscover the formula (or an equivalent one) from data alone.

## Complete Example

```rust,ignore
{{#include ../../../examples/symbolic_regression.rs}}
```

> **Source**: [`examples/symbolic_regression.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/symbolic_regression.rs)

## Running the Example

```bash
cargo run --example symbolic_regression
```

## Key Components

### Tree Genome

```rust,ignore
TreeGenome<ArithmeticTerminal, ArithmeticFunction>
```

A tree genome consists of:
- **Terminals**: Leaf nodes (variables like `X`, constants like `3.14`)
- **Functions**: Internal nodes (operations like `+`, `*`, `sin`)

**Example tree for `x² + 1`:**
```text
       +
      / \
     *   1
    / \
   X   X
```

### Tree Generation

```rust,ignore
// Full method: all leaves at same depth
let tree = TreeGenome::generate_full(&mut rng, 3, 6);

// Grow method: variable depth
let tree = TreeGenome::generate_grow(&mut rng, 5, 0.3);
```

**Full method:**
- All branches have exactly the specified depth
- Creates bushy, complete trees
- Good initial diversity

**Grow method:**
- Terminates probabilistically
- Creates varied shapes
- More natural tree sizes

### Fitness for Symbolic Regression

```rust,ignore
struct SymbolicRegressionFitness {
    data: Vec<(f64, f64)>,  // (x, y) pairs
}

impl SymbolicRegressionFitness {
    fn evaluate(&self, tree: &TreeGenome<...>) -> f64 {
        let mse: f64 = self.data.iter()
            .map(|(x, y)| {
                let predicted = tree.evaluate(&[*x]);
                (y - predicted).powi(2)
            })
            .sum::<f64>() / self.data.len() as f64;

        // Negate MSE (we maximize fitness)
        // Add parsimony pressure for simpler trees
        let complexity_penalty = tree.size() as f64 * 0.001;
        -mse - complexity_penalty
    }
}
```

**Key elements:**
- Mean Squared Error (MSE) measures fit
- Parsimony pressure discourages bloat
- Negated because GA maximizes

### Genetic Operators for Trees

**Subtree Crossover:**
```rust,ignore
fn subtree_crossover(parent1: &Tree, parent2: &Tree, rng: &mut Rng) -> Tree {
    // 1. Select random node in parent1
    // 2. Select random node in parent2
    // 3. Replace subtree at pos1 with subtree from pos2
}
```

**Point Mutation:**
```rust,ignore
fn point_mutate(tree: &mut Tree, rng: &mut Rng) {
    // 1. Select random node
    // 2. Replace with new random subtree
}
```

## Understanding GP Output

### Expression Representation

```rust,ignore
println!("Expression: {}", tree.to_sexpr());
```

S-expression format (Lisp-like):
- `(+ X 1)` means `X + 1`
- `(* X X)` means `X * X`
- `(+ (* X X) (+ (* 2 X) 1))` means `X² + 2X + 1`

### Tree Metrics

```rust,ignore
println!("Tree size: {} nodes", tree.size());
println!("Tree depth: {}", tree.depth());
```

- **Size**: Total number of nodes
- **Depth**: Longest path from root to leaf

Watch for bloat - trees growing without fitness improvement.

## Controlling Bloat

### 1. Parsimony Pressure

Penalize large trees in fitness:

```rust,ignore
let fitness = raw_fitness - tree.size() as f64 * penalty;
```

### 2. Depth Limits

Reject trees exceeding maximum depth:

```rust,ignore
if child.depth() <= 10 {
    new_pop.push(child);
} else {
    new_pop.push(parent.clone());
}
```

### 3. Tournament with Parsimony

Prefer smaller trees when fitness is similar:

```rust,ignore
fn tournament_with_parsimony(pop: &[Tree], k: usize) -> &Tree {
    let contestants = sample(pop, k);
    contestants.iter()
        .max_by(|a, b| {
            let fitness_cmp = a.fitness.cmp(&b.fitness);
            if fitness_cmp == Equal {
                b.size().cmp(&a.size())  // Prefer smaller
            } else {
                fitness_cmp
            }
        })
}
```

## Available Functions and Terminals

### Built-in Arithmetic

**Terminals:**
- `X`: Input variable
- `Constant(f64)`: Numeric constants

**Functions:**
- `Add`: Binary addition
- `Sub`: Binary subtraction
- `Mul`: Binary multiplication
- `Div`: Protected division (returns 1 if divisor is 0)

### Custom Function Sets

Define problem-specific primitives:

```rust,ignore
enum MyFunction {
    Add, Sub, Mul, Div,
    Sin, Cos, Exp, Log,
}

enum MyTerminal {
    X,
    Y,  // Multiple variables
    Constant(f64),
}
```

## GP Parameters

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| Population | 100-1000 | More = better exploration |
| Tournament size | 3-7 | Selection pressure |
| Crossover rate | 0.7-0.9 | Combination vs mutation |
| Mutation rate | 0.1-0.3 | Exploration |
| Max depth | 6-17 | Complexity limit |
| Parsimony coefficient | 0.0001-0.01 | Bloat control |

## Common Issues

### Bloat

**Symptoms:** Trees grow huge with no fitness improvement

**Solutions:**
- Increase parsimony pressure
- Stricter depth limits
- Use tournament with size tie-breaker

### Premature Convergence

**Symptoms:** Population becomes homogeneous early

**Solutions:**
- Larger population
- Higher mutation rate
- Lower selection pressure

### Numeric Instability

**Symptoms:** `NaN` or `Inf` in evaluations

**Solutions:**
- Use protected division
- Clamp outputs to reasonable range
- Penalize extreme predictions

## Exercises

1. **Different target**: Try f(x) = sin(x) or f(x) = x³
2. **Multiple variables**: Extend to f(x, y) = x² + y²
3. **More functions**: Add `sin`, `cos`, `exp`

## Next Steps

- [Interactive Evolution](./interactive-evolution.md) - Human-guided optimization
- [Custom Genome Types](../how-to/custom-genome.md) - Create your own genomes
