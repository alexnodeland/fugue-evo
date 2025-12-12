//! Symbolic Regression with Genetic Programming
//!
//! This example demonstrates genetic programming using tree genomes
//! to discover symbolic expressions that fit data.
//!
//! The goal is to find a mathematical expression that approximates
//! a target function from input-output examples.

use fugue_evo::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Symbolic Regression with GP ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Generate training data from target function: f(x) = x^2 + 2*x + 1
    let training_data: Vec<(f64, f64)> = (-5..=5)
        .map(|i| {
            let x = i as f64;
            let y = x * x + 2.0 * x + 1.0;  // Target function
            (x, y)
        })
        .collect();

    println!("Target function: f(x) = x^2 + 2*x + 1");
    println!("Training points: {}", training_data.len());
    println!();

    // Create fitness function for symbolic regression
    let fitness = SymbolicRegressionFitness::new(training_data.clone());

    // Create initial population of random trees
    let mut population: Vec<TreeGenome<ArithmeticTerminal, ArithmeticFunction>> = Vec::new();
    for _ in 0..100 {
        // Mix of full and grow initialization
        let tree = if rng.gen::<bool>() {
            TreeGenome::generate_full(&mut rng, 3, 6)
        } else {
            TreeGenome::generate_grow(&mut rng, 5, 0.3)
        };
        population.push(tree);
    }

    // Evaluate initial population
    let mut pop_with_fitness: Vec<(TreeGenome<ArithmeticTerminal, ArithmeticFunction>, f64)> =
        population
            .iter()
            .map(|tree| (tree.clone(), fitness.evaluate(tree)))
            .collect();

    pop_with_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Initial population stats:");
    println!("  Best fitness: {:.6}", pop_with_fitness[0].1);
    println!("  Best expr: {}", pop_with_fitness[0].0.to_sexpr());
    println!();

    // Simple evolution loop
    let max_generations = 50;
    let tournament_size = 5;
    let mutation_rate = 0.2;
    let crossover_rate = 0.8;

    for gen in 0..max_generations {
        let mut new_pop = Vec::new();

        // Elitism: keep best
        new_pop.push(pop_with_fitness[0].0.clone());

        while new_pop.len() < 100 {
            // Tournament selection
            let parent1 = tournament_select(&pop_with_fitness, tournament_size, &mut rng);
            let parent2 = tournament_select(&pop_with_fitness, tournament_size, &mut rng);

            let mut child = if rng.gen::<f64>() < crossover_rate {
                // Subtree crossover
                subtree_crossover(&parent1, &parent2, &mut rng)
            } else {
                parent1.clone()
            };

            // Point mutation
            if rng.gen::<f64>() < mutation_rate {
                point_mutate(&mut child, &mut rng);
            }

            // Limit tree depth
            if child.depth() <= 10 {
                new_pop.push(child);
            } else {
                new_pop.push(parent1.clone());
            }
        }

        // Evaluate new population
        pop_with_fitness = new_pop
            .iter()
            .map(|tree| (tree.clone(), fitness.evaluate(tree)))
            .collect();

        pop_with_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if (gen + 1) % 10 == 0 {
            println!(
                "Gen {:3}: Best = {:.6}, Size = {}, Expr = {}",
                gen + 1,
                pop_with_fitness[0].1,
                pop_with_fitness[0].0.size(),
                truncate_expr(&pop_with_fitness[0].0.to_sexpr(), 50)
            );
        }
    }

    println!("\n=== Final Result ===");
    let best = &pop_with_fitness[0];
    println!("Best fitness: {:.6}", best.1);
    println!("Best expression: {}", best.0.to_sexpr());
    println!("Tree size: {} nodes", best.0.size());
    println!("Tree depth: {}", best.0.depth());

    // Test the discovered function
    println!("\nComparison on test points:");
    println!("{:>6} {:>12} {:>12} {:>12}", "x", "Target", "Predicted", "Error");
    for x in [-3.5, -1.0, 0.0, 1.0, 2.5] {
        let target = x * x + 2.0 * x + 1.0;
        let predicted = best.0.evaluate(&[x]);
        let error = (target - predicted).abs();
        println!("{:6.1} {:12.4} {:12.4} {:12.6}", x, target, predicted, error);
    }

    Ok(())
}

fn truncate_expr(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

fn tournament_select<'a>(
    pop: &'a [(TreeGenome<ArithmeticTerminal, ArithmeticFunction>, f64)],
    size: usize,
    rng: &mut StdRng,
) -> &'a TreeGenome<ArithmeticTerminal, ArithmeticFunction> {
    use rand::seq::SliceRandom;
    let contestants: Vec<_> = pop.choose_multiple(rng, size).collect();
    let best = contestants.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    &best.0
}

fn subtree_crossover(
    parent1: &TreeGenome<ArithmeticTerminal, ArithmeticFunction>,
    parent2: &TreeGenome<ArithmeticTerminal, ArithmeticFunction>,
    rng: &mut StdRng,
) -> TreeGenome<ArithmeticTerminal, ArithmeticFunction> {
    let positions1 = parent1.root.positions();
    let positions2 = parent2.root.positions();

    if positions1.is_empty() || positions2.is_empty() {
        return parent1.clone();
    }

    let pos1 = &positions1[rng.gen_range(0..positions1.len())];
    let pos2 = &positions2[rng.gen_range(0..positions2.len())];

    if let Some(subtree) = parent2.root.get_subtree(pos2) {
        let mut new_root = parent1.root.clone();
        new_root.replace_subtree(pos1, subtree.clone());
        TreeGenome::new(new_root, parent1.max_depth.max(parent2.max_depth))
    } else {
        parent1.clone()
    }
}

fn point_mutate(
    tree: &mut TreeGenome<ArithmeticTerminal, ArithmeticFunction>,
    rng: &mut StdRng,
) {
    let positions = tree.root.positions();
    if positions.is_empty() {
        return;
    }

    let pos = &positions[rng.gen_range(0..positions.len())];

    // Replace with random subtree
    let new_subtree = if rng.gen::<bool>() {
        TreeNode::terminal(ArithmeticTerminal::random(rng))
    } else {
        let func = ArithmeticFunction::random(rng);
        let arity = func.arity();
        let children: Vec<_> = (0..arity)
            .map(|_| TreeNode::terminal(ArithmeticTerminal::random(rng)))
            .collect();
        TreeNode::function(func, children)
    };

    tree.root.replace_subtree(pos, new_subtree);
}

/// Fitness function for symbolic regression
struct SymbolicRegressionFitness {
    data: Vec<(f64, f64)>,
}

impl SymbolicRegressionFitness {
    fn new(data: Vec<(f64, f64)>) -> Self {
        Self { data }
    }

    fn evaluate(&self, tree: &TreeGenome<ArithmeticTerminal, ArithmeticFunction>) -> f64 {
        let mse: f64 = self
            .data
            .iter()
            .map(|(x, y)| {
                let predicted = tree.evaluate(&[*x]);
                if predicted.is_finite() {
                    (y - predicted).powi(2)
                } else {
                    1e6  // Penalty for invalid values
                }
            })
            .sum::<f64>()
            / self.data.len() as f64;

        // Negate MSE (we maximize fitness)
        // Add small parsimony pressure to prefer simpler trees
        let complexity_penalty = tree.size() as f64 * 0.001;
        -mse - complexity_penalty
    }
}
