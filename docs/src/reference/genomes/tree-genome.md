# TreeGenome Reference

Tree-structured genome for genetic programming.

## Module

```rust,ignore
use fugue_evo::genome::tree::{TreeGenome, TreeNode, ArithmeticTerminal, ArithmeticFunction};
```

## Construction

```rust,ignore
// Generate full tree (all branches same depth)
let tree = TreeGenome::generate_full(&mut rng, min_depth, max_depth);

// Generate grow tree (variable depth)
let tree = TreeGenome::generate_grow(&mut rng, max_depth, terminal_prob);

// From root node
let root = TreeNode::function(Add, vec![
    TreeNode::terminal(X),
    TreeNode::terminal(Constant(1.0)),
]);
let tree = TreeGenome::new(root, 10);
```

## TreeNode Types

```rust,ignore
pub enum TreeNode<T, F> {
    Terminal(T),           // Leaf node
    Function(F, Vec<Self>), // Internal node with children
}
```

### Built-in Terminals

```rust,ignore
pub enum ArithmeticTerminal {
    X,              // Variable
    Constant(f64),  // Numeric constant
}
```

### Built-in Functions

```rust,ignore
pub enum ArithmeticFunction {
    Add,  // Binary +
    Sub,  // Binary -
    Mul,  // Binary *
    Div,  // Protected division
}
```

## Methods

### Tree Properties

| Method | Return | Description |
|--------|--------|-------------|
| `size()` | `usize` | Total number of nodes |
| `depth()` | `usize` | Maximum depth |
| `height()` | `usize` | Same as depth |

### Evaluation

| Method | Description |
|--------|-------------|
| `evaluate(inputs)` | Evaluate tree with input values |

### Representation

| Method | Return | Description |
|--------|--------|-------------|
| `to_sexpr()` | `String` | S-expression format |
| `to_infix()` | `String` | Infix notation |

### Navigation

| Method | Description |
|--------|-------------|
| `root.positions()` | Get all node positions |
| `root.get_subtree(pos)` | Get subtree at position |
| `root.replace_subtree(pos, new)` | Replace subtree |

## Operators

### Crossover

```rust,ignore
fn subtree_crossover<T, F>(
    parent1: &TreeGenome<T, F>,
    parent2: &TreeGenome<T, F>,
    rng: &mut impl Rng,
) -> TreeGenome<T, F> {
    // Select random subtrees and exchange
}
```

### Mutation

```rust,ignore
// Point mutation: change single node
fn point_mutate<T, F>(tree: &mut TreeGenome<T, F>, rng: &mut impl Rng);

// Subtree mutation: replace subtree with random
fn subtree_mutate<T, F>(tree: &mut TreeGenome<T, F>, rng: &mut impl Rng);

// Hoist mutation: replace tree with subtree
fn hoist_mutate<T, F>(tree: &mut TreeGenome<T, F>, rng: &mut impl Rng);
```

## Custom Primitives

Define your own terminals and functions:

```rust,ignore
#[derive(Clone, Debug)]
pub enum MyTerminal {
    X, Y,  // Two variables
    Constant(f64),
}

#[derive(Clone, Debug)]
pub enum MyFunction {
    Add, Sub, Mul, Div,
    Sin, Cos, Exp, Log,
}

impl MyFunction {
    fn arity(&self) -> usize {
        match self {
            Add | Sub | Mul | Div => 2,
            Sin | Cos | Exp | Log => 1,
        }
    }

    fn apply(&self, args: &[f64]) -> f64 {
        match self {
            Add => args[0] + args[1],
            Sub => args[0] - args[1],
            Mul => args[0] * args[1],
            Div => if args[1].abs() < 1e-10 { 1.0 } else { args[0] / args[1] },
            Sin => args[0].sin(),
            Cos => args[0].cos(),
            Exp => args[0].exp().min(1e10),
            Log => if args[0] > 0.0 { args[0].ln() } else { 0.0 },
        }
    }
}
```

## Bloat Control

### Depth Limiting

```rust,ignore
if child.depth() > max_depth {
    child = parent.clone();  // Reject oversized trees
}
```

### Parsimony Pressure

```rust,ignore
let fitness = raw_fitness - tree.size() as f64 * parsimony_coeff;
```

## Example

See [Genetic Programming Tutorial](../../tutorials/genetic-programming.md) for a complete symbolic regression example.

## See Also

- [Genetic Programming Tutorial](../../tutorials/genetic-programming.md)
- [Custom Genome Types](../../how-to/custom-genome.md)
