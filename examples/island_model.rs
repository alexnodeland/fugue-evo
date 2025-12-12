//! Island Model Parallelism
//!
//! This example demonstrates the island model for parallel evolution,
//! where multiple subpopulations evolve independently with periodic
//! migration of individuals between islands.
//!
//! Island models can help maintain diversity and escape local optima.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Island Model Parallelism ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Rastrigin - multimodal function benefits from island diversity
    const DIM: usize = 20;
    let fitness = Rastrigin::new(DIM);
    let bounds = MultiBounds::symmetric(5.12, DIM);

    println!("Problem: {}-D Rastrigin", DIM);
    println!("Configuration:");
    println!("  Islands: 4");
    println!("  Population per island: 50");
    println!("  Migration interval: 25 generations");
    println!("  Migration policy: Best(2)\n");

    // Create island model
    let mut island_model = IslandModelBuilder::<RealVector, _, _, _, _, f64>::new()
        .num_islands(4)
        .island_population_size(50)
        .topology(MigrationTopology::Ring)
        .migration_interval(25)
        .migration_policy(MigrationPolicy::Best(2))
        .bounds(bounds.clone())
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(15.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(fitness)
        .build(&mut rng)?;

    // Run the island model
    let best = island_model.run(200, &mut rng)?;
    let best_fitness = *best.fitness_value();
    let generations = island_model.generation;

    println!("Results:");
    println!("  Best fitness: {:.6}", best_fitness);
    println!("  Generations:  {}", generations);

    // Compare with single-population GA
    println!("\n--- Comparison with single population ---");
    let mut rng2 = StdRng::seed_from_u64(42);
    let fitness2 = Rastrigin::new(DIM);
    let bounds2 = MultiBounds::symmetric(5.12, DIM);

    let single_result = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(200) // Same total population
        .bounds(bounds2)
        .selection(TournamentSelection::new(3))
        .crossover(SbxCrossover::new(15.0))
        .mutation(PolynomialMutation::new(20.0))
        .fitness(fitness2)
        .max_generations(200)
        .build()?
        .run(&mut rng2)?;

    println!("Single population best: {:.6}", single_result.best_fitness);
    println!("Island model best:      {:.6}", best_fitness);

    if best_fitness > single_result.best_fitness {
        println!("\nIsland model found better solution!");
    } else {
        println!("\nSingle population found better solution.");
    }

    Ok(())
}
