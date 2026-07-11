//! Sphere Function Optimization
//!
//! This example demonstrates basic continuous optimization using the Simple GA
//! to minimize the Sphere function (sum of squares).
//!
//! The Sphere function is a simple unimodal, convex, and separable benchmark
//! that's easy to optimize but useful for verifying the GA is working correctly.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sphere Function Optimization ===\n");

    // Create a seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Define the problem dimension
    const DIM: usize = 10;

    // Create the fitness function. `Sphere` already reports higher-is-better
    // fitness (it internally negates the sum of squares), so a fitness of 0 at
    // the origin is optimal and no negation is needed here.
    let fitness = Sphere::new(DIM);

    // Define search bounds: each dimension in [-5.12, 5.12]
    let bounds = MultiBounds::symmetric(5.12, DIM);

    // Build and run the Simple GA.
    //
    // The `real_valued()` entry point pins the genome/fitness types and installs
    // sensible operator defaults (tournament selection, SBX crossover, polynomial
    // mutation), so the quickstart needs no turbofish. Any default is overridable
    // with `.selection(..)` / `.crossover(..)` / `.mutation(..)`.
    let result = SimpleGABuilder::real_valued()
        .population_size(100)
        .bounds(bounds)
        .fitness(fitness)
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    // Print results. `result.best_fitness` is the GA's internal (negated,
    // higher-is-better) value, so we flip its sign back here to report the
    // textbook Sphere objective (sum of squares, minimized at 0.0).
    println!("Optimization complete!");
    println!("  Best fitness (sum of squares): {:.6}", -result.best_fitness);
    println!("  Generations:  {}", result.generations);
    println!("  Evaluations:  {}", result.evaluations);
    println!("\nBest solution:");
    for (i, val) in result.best_genome.genes().iter().enumerate() {
        println!("  x[{}] = {:.6}", i, val);
    }

    // The optimal solution is at the origin with fitness 0
    let distance_from_optimum: f64 = result
        .best_genome
        .genes()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    println!("\nDistance from optimum: {:.6}", distance_from_optimum);

    // Show convergence statistics
    println!("\n{}", result.stats.summary());

    Ok(())
}
