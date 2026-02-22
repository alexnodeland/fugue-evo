# Multi-Objective Optimization

This guide shows how to optimize problems with multiple conflicting objectives using NSGA-II.

## When to Use Multi-Objective Optimization

Use multi-objective optimization when:

- You have 2+ objectives that conflict
- You need the full trade-off surface (Pareto front)
- No single "best" solution exists
- Stakeholders need to choose among alternatives

**Examples:**
- Cost vs. quality
- Speed vs. accuracy
- Performance vs. power consumption

## Pareto Dominance

Solution A **dominates** solution B if:
- A is at least as good as B in all objectives
- A is strictly better than B in at least one objective

The **Pareto front** contains all non-dominated solutions.

```text
Objective 2
    ↑
    │   ★ ← Non-dominated (Pareto front)
    │  ★
    │ ★   ● ← Dominated
    │★      ●
    │         ●
    └──────────→ Objective 1

★ = Pareto-optimal solutions
● = Dominated solutions
```

## Basic NSGA-II Usage

```rust,ignore
use fugue_evo::prelude::*;

// Define multi-objective fitness
struct BiObjective;

impl Fitness<RealVector> for BiObjective {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        let genes = genome.genes();

        // Objective 1: Minimize sum of squares
        let obj1 = -genes.iter().map(|x| x * x).sum::<f64>();

        // Objective 2: Minimize distance from (1,1,...)
        let obj2 = -genes.iter().map(|x| (x - 1.0).powi(2)).sum::<f64>();

        ParetoFitness::new(vec![obj1, obj2])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let bounds = MultiBounds::symmetric(5.0, 10);
    let fitness = BiObjective;

    let result = Nsga2Builder::<RealVector, _, _, _, _>::new()
        .population_size(100)
        .bounds(bounds)
        .selection(TournamentSelection::new(2))
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(fitness)
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    // Access Pareto front
    println!("Pareto front size: {}", result.pareto_front.len());
    for (i, solution) in result.pareto_front.iter().enumerate() {
        println!(
            "Solution {}: objectives = {:?}",
            i, solution.fitness_value().objectives()
        );
    }

    Ok(())
}
```

## ZDT Test Problems

Fugue-evo includes standard ZDT test problems:

```rust,ignore
// ZDT1: Convex Pareto front
let fitness = Zdt1::new(30);

// ZDT2: Non-convex Pareto front
let fitness = Zdt2::new(30);

// ZDT3: Disconnected Pareto front
let fitness = Zdt3::new(30);
```

## Analyzing Results

### Pareto Front Coverage

```rust,ignore
// Spread metric: how well-distributed are solutions?
let spread = compute_spread(&result.pareto_front);

// Hypervolume: area dominated by Pareto front
let reference_point = vec![0.0, 0.0]; // Worst case
let hypervolume = compute_hypervolume(&result.pareto_front, &reference_point);
```

### Visualizing Trade-offs

```rust,ignore
// Extract objectives for plotting
let points: Vec<(f64, f64)> = result.pareto_front.iter()
    .map(|sol| {
        let objs = sol.fitness_value().objectives();
        (objs[0], objs[1])
    })
    .collect();

// Export to CSV for plotting
let mut file = File::create("pareto_front.csv")?;
writeln!(file, "obj1,obj2")?;
for (o1, o2) in points {
    writeln!(file, "{},{}", o1, o2)?;
}
```

## Decision Making

After obtaining the Pareto front, choose a solution:

### Weighted Sum

```rust,ignore
fn weighted_solution(pareto_front: &[Individual<RealVector>], weights: &[f64]) -> &Individual<RealVector> {
    pareto_front.iter()
        .max_by(|a, b| {
            let score_a: f64 = a.fitness_value().objectives().iter()
                .zip(weights)
                .map(|(o, w)| o * w)
                .sum();
            let score_b: f64 = b.fitness_value().objectives().iter()
                .zip(weights)
                .map(|(o, w)| o * w)
                .sum();
            score_a.partial_cmp(&score_b).unwrap()
        })
        .unwrap()
}

// Equal weight to both objectives
let balanced = weighted_solution(&result.pareto_front, &[0.5, 0.5]);
```

### Knee Point

Find the "knee" - maximum curvature in Pareto front:

```rust,ignore
fn find_knee(pareto_front: &[Individual<RealVector>]) -> &Individual<RealVector> {
    // Simplified: find point furthest from line connecting extremes
    let min_obj1 = pareto_front.iter()
        .min_by(|a, b| a.objectives()[0].partial_cmp(&b.objectives()[0]).unwrap())
        .unwrap();
    let min_obj2 = pareto_front.iter()
        .min_by(|a, b| a.objectives()[1].partial_cmp(&b.objectives()[1]).unwrap())
        .unwrap();

    // Find point with maximum perpendicular distance to line
    pareto_front.iter()
        .max_by(|a, b| {
            let dist_a = perpendicular_distance(a, min_obj1, min_obj2);
            let dist_b = perpendicular_distance(b, min_obj1, min_obj2);
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .unwrap()
}
```

### Constraint-Based Selection

```rust,ignore
fn select_with_constraints(
    pareto_front: &[Individual<RealVector>],
    obj1_threshold: f64,
) -> Vec<&Individual<RealVector>> {
    pareto_front.iter()
        .filter(|sol| sol.objectives()[0] >= obj1_threshold)
        .collect()
}
```

## Constrained Multi-Objective

Handle constraints with penalty or constraint-dominance:

```rust,ignore
impl Fitness<RealVector> for ConstrainedMultiObj {
    type Value = ParetoFitness;

    fn evaluate(&self, genome: &RealVector) -> ParetoFitness {
        let genes = genome.genes();

        // Objectives
        let obj1 = compute_obj1(genes);
        let obj2 = compute_obj2(genes);

        // Constraint violation
        let violation = compute_constraint_violation(genes);

        if violation > 0.0 {
            // Infeasible: assign poor objectives
            ParetoFitness::new(vec![f64::NEG_INFINITY, f64::NEG_INFINITY])
                .with_constraint_violation(violation)
        } else {
            ParetoFitness::new(vec![obj1, obj2])
        }
    }
}
```

## Many-Objective Optimization

For 3+ objectives, NSGA-II may struggle. Consider:

### Reference Point Methods

```rust,ignore
// Define reference directions
let reference_directions = generate_reference_directions(3, 12);

// Use reference-point based selection
let result = Nsga3Builder::new()
    .reference_directions(reference_directions)
    // ...
    .run(&mut rng)?;
```

### Decomposition

Convert to multiple single-objective problems:

```rust,ignore
let weight_vectors = vec![
    vec![1.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
    vec![0.33, 0.33, 0.34],
    // ... more weight combinations
];

// Run separate optimizations
let results: Vec<_> = weight_vectors.par_iter()
    .map(|weights| {
        let scalarized = ScalarizedFitness::new(&multi_obj, weights);
        run_single_objective(&scalarized, &mut thread_rng())
    })
    .collect();
```

## Performance Tips

### Population Sizing

More objectives need larger populations:

| Objectives | Recommended Population |
|------------|----------------------|
| 2 | 100-200 |
| 3 | 200-500 |
| 4+ | 500-1000+ |

### Archive Strategy

Maintain external archive of non-dominated solutions:

```rust,ignore
let mut archive: Vec<Individual<RealVector>> = Vec::new();

for gen in 0..max_generations {
    // Evolution step...

    // Update archive with new non-dominated solutions
    for solution in population.iter() {
        if !is_dominated_by_archive(solution, &archive) {
            // Remove solutions dominated by new one
            archive.retain(|a| !solution.dominates(a));
            archive.push(solution.clone());
        }
    }
}
```

## Next Steps

- [Choosing an Algorithm](./choosing-algorithm.md) - When to use NSGA-II
- [Custom Fitness Functions](./custom-fitness.md) - Implement multi-objective fitness
