//! Rastrigin Function Benchmark
//!
//! This example demonstrates optimization of the highly multimodal Rastrigin
//! function, a challenging benchmark that tests the GA's ability to escape
//! local optima.
//!
//! The Rastrigin function has many local minima but a single global minimum
//! at the origin.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rastrigin Function Benchmark ===\n");

    let mut rng = StdRng::seed_from_u64(12345);

    const DIM: usize = 20;

    // Rastrigin is a highly multimodal function
    let fitness = Rastrigin::new(DIM);
    let bounds = MultiBounds::symmetric(5.12, DIM);

    println!("Problem: {} dimensions", DIM);
    println!("Search space: [-5.12, 5.12]^{}", DIM);
    println!("Global optimum: 0.0 at origin\n");

    // Run GA with larger population for multimodal landscape
    let result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(200)
        .bounds(bounds)
        .selection(TournamentSelection::new(5)) // Higher pressure
        .crossover(SbxCrossover::new(15.0)) // More exploration
        .mutation(PolynomialMutation::new(20.0).with_probability(0.1))
        .fitness(fitness)
        .max_generations(500)
        .elitism(true)
        .elite_count(2)
        .build()?
        .run(&mut rng)?;

    println!("Results:");
    println!("  Best fitness:   {:.6}", result.best_fitness);
    println!("  Generations:    {}", result.generations);
    println!("  Evaluations:    {}", result.evaluations);

    // Check how close we got to the global optimum
    let max_deviation = result
        .best_genome
        .genes()
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, |a, b| a.max(b));

    println!("\nSolution quality:");
    println!("  Max deviation from origin: {:.6}", max_deviation);

    // Print fitness progress
    let history = result.stats.best_fitness_history();
    println!("\nFitness progression (every 50 generations):");
    for (i, fitness) in history.iter().enumerate() {
        if i % 50 == 0 {
            println!("  Gen {:4}: {:.6}", i, fitness);
        }
    }

    Ok(())
}
