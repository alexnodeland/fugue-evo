# NSGA-II Reference

Non-dominated Sorting Genetic Algorithm II for multi-objective optimization.

## Module

```rust,ignore
use fugue_evo::algorithms::nsga2::{Nsga2, Nsga2Builder, Nsga2Result};
use fugue_evo::fitness::ParetoFitness;
```

## Builder API

### Required Configuration

| Method | Type | Description |
|--------|------|-------------|
| `bounds(bounds)` | `MultiBounds` | Search space bounds |
| `selection(sel)` | `impl SelectionOperator` | Selection operator |
| `crossover(cx)` | `impl CrossoverOperator` | Crossover operator |
| `mutation(mut)` | `impl MutationOperator` | Mutation operator |
| `fitness(fit)` | `impl Fitness<Value=ParetoFitness>` | Multi-objective fitness |

### Optional Configuration

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `population_size(n)` | `usize` | 100 | Population size |
| `max_generations(n)` | `usize` | 100 | Max generations |

## ParetoFitness

Multi-objective fitness values:

```rust,ignore
pub struct ParetoFitness {
    objectives: Vec<f64>,
    constraint_violation: Option<f64>,
}

impl ParetoFitness {
    pub fn new(objectives: Vec<f64>) -> Self;
    pub fn with_constraint_violation(self, violation: f64) -> Self;
    pub fn objectives(&self) -> &[f64];
    pub fn dominates(&self, other: &Self) -> bool;
}
```

## Usage

### Basic Example

```rust,ignore
use fugue_evo::prelude::*;

struct BiObjective;

impl Fitness<RealVector> for BiObjective {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        let g = genome.genes();
        ParetoFitness::new(vec![
            g.iter().sum(),        // Maximize sum
            g.iter().product(),    // Maximize product
        ])
    }
}

let result = Nsga2Builder::<RealVector, _, _, _, _>::new()
    .population_size(100)
    .bounds(MultiBounds::symmetric(5.0, 10))
    .selection(TournamentSelection::new(2))
    .crossover(SbxCrossover::new(20.0))
    .mutation(PolynomialMutation::new(20.0))
    .fitness(BiObjective)
    .max_generations(200)
    .build()?
    .run(&mut rng)?;

// Access Pareto front
for solution in &result.pareto_front {
    println!("{:?}", solution.fitness_value().objectives());
}
```

## Result Structure

```rust,ignore
pub struct Nsga2Result<G> {
    pub pareto_front: Vec<Individual<G>>,
    pub generations: usize,
    pub evaluations: usize,
}
```

## Algorithm Details

### Non-dominated Sorting

Ranks population into fronts:
- Front 0: All non-dominated solutions
- Front 1: Non-dominated after removing Front 0
- etc.

### Crowding Distance

Within each front, calculates spacing:
- Higher crowding distance = more isolated
- Preserves diversity on Pareto front

### Selection

Binary tournament using:
1. Lower rank preferred
2. If same rank, higher crowding distance preferred

## Constraints

Handle constraints via constraint-dominance:

```rust,ignore
impl Fitness<RealVector> for ConstrainedBiObjective {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        let g = genome.genes();
        let violation = compute_violation(g);

        ParetoFitness::new(vec![obj1, obj2])
            .with_constraint_violation(violation)
    }
}
```

Constraint-dominance rules:
1. Feasible always dominates infeasible
2. Between infeasible: lower violation wins
3. Between feasible: normal Pareto dominance

## See Also

- [Multi-Objective Optimization](../../how-to/multi-objective.md)
- [Choosing an Algorithm](../../how-to/choosing-algorithm.md)
