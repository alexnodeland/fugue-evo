//! Checkpointing and Recovery
//!
//! This example demonstrates how to save and restore evolution state
//! using checkpoints. This is essential for long-running optimizations
//! that may need to be interrupted and resumed.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Checkpointing and Recovery ===\n");

    let checkpoint_dir = PathBuf::from("/tmp/fugue_evo_checkpoints");

    // Clean up any existing checkpoints first
    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
    }

    // Run with checkpoints
    run_with_checkpoints(&checkpoint_dir)?;

    // Demonstrate resuming (in real usage, this would be after a restart)
    println!("\n--- Simulating resume from checkpoint ---\n");
    resume_from_checkpoint(&checkpoint_dir)?;

    // Clean up
    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
        println!("\nCheckpoint directory cleaned up.");
    }

    Ok(())
}

fn run_with_checkpoints(checkpoint_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);

    const DIM: usize = 10;
    let fitness = Sphere::new(DIM);
    let bounds = MultiBounds::symmetric(5.12, DIM);

    // Run evolution with periodic checkpoints
    let mut population: Population<RealVector, f64> = Population::random(100, &bounds, &mut rng);
    population.evaluate(&fitness);

    let selection = TournamentSelection::new(3);
    let crossover = SbxCrossover::new(20.0);
    let mutation = PolynomialMutation::new(20.0);

    // Create checkpoint manager
    let mut manager = CheckpointManager::new(checkpoint_dir, "evolution")
        .every(50) // Save every 50 generations
        .keep(3); // Keep last 3 checkpoints

    let max_generations = 200;

    for gen in 0..max_generations {
        // Evolution step
        let selection_pool: Vec<_> = population.as_fitness_pairs();
        let mut new_pop: Population<RealVector, f64> = Population::with_capacity(100);

        // Elitism
        if let Some(best) = population.best() {
            new_pop.push(best.clone());
        }

        while new_pop.len() < 100 {
            let p1_idx = selection.select(&selection_pool, &mut rng);
            let p2_idx = selection.select(&selection_pool, &mut rng);

            let (mut c1, mut c2) = crossover
                .crossover(
                    &selection_pool[p1_idx].0,
                    &selection_pool[p2_idx].0,
                    &mut rng,
                )
                .genome()
                .unwrap_or_else(|| {
                    (
                        selection_pool[p1_idx].0.clone(),
                        selection_pool[p2_idx].0.clone(),
                    )
                });

            mutation.mutate(&mut c1, &mut rng);
            mutation.mutate(&mut c2, &mut rng);

            new_pop.push(Individual::new(c1));
            if new_pop.len() < 100 {
                new_pop.push(Individual::new(c2));
            }
        }

        new_pop.evaluate(&fitness);
        new_pop.set_generation(gen + 1);
        population = new_pop;

        // Save checkpoint periodically
        if manager.should_save(gen + 1) {
            let best = population.best().unwrap();
            println!(
                "Gen {:3}: Best = {:.6} - Saving checkpoint...",
                gen + 1,
                best.fitness_value()
            );

            // Create checkpoint with current population
            let individuals: Vec<Individual<RealVector>> = population.iter().cloned().collect();
            let checkpoint =
                Checkpoint::new(gen + 1, individuals).with_evaluations((gen + 1) * 100);

            manager.save(&checkpoint)?;
        }
    }

    let best = population.best().unwrap();
    println!("\nFinal result:");
    println!("  Best fitness: {:.6}", best.fitness_value());
    println!("  Generations:  {}", max_generations);

    Ok(())
}

fn resume_from_checkpoint(checkpoint_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Find the latest checkpoint file
    let entries: Vec<_> = std::fs::read_dir(checkpoint_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "ckpt"))
        .collect();

    if entries.is_empty() {
        println!("No checkpoint files found!");
        return Ok(());
    }

    // Sort by name to get the latest
    let mut paths: Vec<_> = entries.iter().map(|e| e.path()).collect();
    paths.sort();
    let latest_checkpoint = paths.last().unwrap();

    println!("Loading checkpoint: {:?}", latest_checkpoint);

    // Load checkpoint
    let checkpoint: Checkpoint<RealVector> = load_checkpoint(latest_checkpoint)?;

    println!("Loaded checkpoint:");
    println!("  Generation: {}", checkpoint.generation);
    println!("  Population size: {}", checkpoint.population.len());
    println!("  Evaluations: {}", checkpoint.evaluations);

    // Find best in loaded population
    let best_individual = checkpoint
        .population
        .iter()
        .filter_map(|ind| ind.fitness.as_ref().map(|f| (ind, f.to_f64())))
        .max_by(|(_, f1), (_, f2)| f1.partial_cmp(f2).unwrap());

    if let Some((_best, fitness)) = best_individual {
        println!("  Best fitness at checkpoint: {:.6}", fitness);
    }

    // Continue evolution...
    let mut rng = StdRng::seed_from_u64(12345); // Different seed for continuation
    let fitness = Sphere::new(10);

    // Reconstruct population from checkpoint
    let mut population: Population<RealVector, f64> =
        Population::with_capacity(checkpoint.population.len());
    for ind in checkpoint.population {
        population.push(ind);
    }

    let selection = TournamentSelection::new(3);
    let crossover = SbxCrossover::new(20.0);
    let mutation = PolynomialMutation::new(20.0);

    let remaining_gens = 200 - checkpoint.generation;
    println!("\nContinuing for {} more generations...\n", remaining_gens);

    for gen in checkpoint.generation..200 {
        let selection_pool: Vec<_> = population.as_fitness_pairs();
        let mut new_pop: Population<RealVector, f64> = Population::with_capacity(100);

        if let Some(best) = population.best() {
            new_pop.push(best.clone());
        }

        while new_pop.len() < 100 {
            let p1_idx = selection.select(&selection_pool, &mut rng);
            let p2_idx = selection.select(&selection_pool, &mut rng);

            let (mut c1, mut c2) = crossover
                .crossover(
                    &selection_pool[p1_idx].0,
                    &selection_pool[p2_idx].0,
                    &mut rng,
                )
                .genome()
                .unwrap_or_else(|| {
                    (
                        selection_pool[p1_idx].0.clone(),
                        selection_pool[p2_idx].0.clone(),
                    )
                });

            mutation.mutate(&mut c1, &mut rng);
            mutation.mutate(&mut c2, &mut rng);

            new_pop.push(Individual::new(c1));
            if new_pop.len() < 100 {
                new_pop.push(Individual::new(c2));
            }
        }

        new_pop.evaluate(&fitness);
        new_pop.set_generation(gen + 1);
        population = new_pop;

        if (gen + 1) % 50 == 0 {
            let best = population.best().unwrap();
            println!("Gen {:3}: Best = {:.6}", gen + 1, best.fitness_value());
        }
    }

    let best = population.best().unwrap();
    println!("\nFinal result after resumption:");
    println!("  Best fitness: {:.6}", best.fitness_value());

    Ok(())
}
