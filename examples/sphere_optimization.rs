//! Sphere Function Optimization
//!
//! This example demonstrates basic continuous optimization using the Simple GA
//! to minimize the Sphere function (sum of squares).
//!
//! The Sphere function is a simple unimodal, convex, and separable benchmark
//! that's easy to optimize but useful for verifying the GA is working correctly.

use fugue_evo::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sphere Function Optimization ===\n");

    // Create a seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Define the problem dimension
    const DIM: usize = 10;

    // Create the fitness function (Sphere minimizes to 0 at origin)
    // We negate because fugue-evo maximizes by default
    let fitness = Sphere::new(DIM);

    // Define search bounds: each dimension in [-5.12, 5.12]
    let bounds = MultiBounds::symmetric(5.12, DIM);

    // Build and run the Simple GA
    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(100)
        .bounds(bounds)
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(fitness)
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    // Print results
    println!("Optimization complete!");
    println!("  Best fitness: {:.6}", result.best_fitness);
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
