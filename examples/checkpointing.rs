//! Checkpointing and Reproducible Recovery
//!
//! This example demonstrates how to save and restore evolution state using
//! checkpoints so that a long-running optimization can be interrupted and
//! resumed **bit-identically** - i.e. a run that is checkpointed, restored, and
//! continued reaches the *exact same* result as a run that was never
//! interrupted.
//!
//! The key ingredient (EV-02) is capturing the RNG state alongside the
//! population. Reproducible resume requires a snapshot-able ChaCha RNG
//! (`ChaCha8Rng`/`ChaCha12Rng`/`ChaCha20Rng`); generic generators such as
//! `StdRng`/`ThreadRng` cannot be serialized and therefore cannot reproduce the
//! stochastic trajectory.

use fugue_evo::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const DIM: usize = 10;
const POP_SIZE: usize = 100;
const SEED: u64 = 42;
const TOTAL_GENERATIONS: usize = 20;
const CHECKPOINT_AT: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Checkpointing and Reproducible Recovery ===\n");

    let checkpoint_dir = PathBuf::from("/tmp/fugue_evo_checkpoints");
    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
    }
    std::fs::create_dir_all(&checkpoint_dir)?;

    // (1) Straight run: TOTAL_GENERATIONS with no interruption.
    let straight_best = run_straight(TOTAL_GENERATIONS);
    println!("Straight run ({TOTAL_GENERATIONS} gens):            best = {straight_best:.12}");

    // (2) Interrupted run: CHECKPOINT_AT gens, checkpoint (population + RNG) to
    //     disk, restore, then continue for the remaining gens.
    let resumed_best = run_with_checkpoint(&checkpoint_dir, TOTAL_GENERATIONS, CHECKPOINT_AT)?;
    println!(
        "Resumed run ({CHECKPOINT_AT} + {} gens via disk): best = {resumed_best:.12}",
        TOTAL_GENERATIONS - CHECKPOINT_AT
    );

    // (3) The two must be bit-identical.
    println!();
    if straight_best.to_bits() == resumed_best.to_bits() {
        println!("SUCCESS: resumed run is bit-identical to the uninterrupted run.");
    } else {
        return Err(format!(
            "reproducibility broken: straight={straight_best} resumed={resumed_best}"
        )
        .into());
    }

    if checkpoint_dir.exists() {
        std::fs::remove_dir_all(&checkpoint_dir)?;
        println!("\nCheckpoint directory cleaned up.");
    }

    Ok(())
}

/// Build the fixed operator set / fitness used by both runs.
fn make_setup() -> (
    Sphere,
    MultiBounds,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
) {
    (
        Sphere::new(DIM),
        MultiBounds::symmetric(5.12, DIM),
        TournamentSelection::new(3),
        SbxCrossover::new(20.0),
        PolynomialMutation::new(20.0),
    )
}

/// One generation of evolution. Both the straight and resumed runs call this
/// identical function, so they consume the RNG in exactly the same order.
fn evolve_one_generation<R: Rng>(
    population: &Population<RealVector, f64>,
    fitness: &Sphere,
    selection: &TournamentSelection,
    crossover: &SbxCrossover,
    mutation: &PolynomialMutation,
    rng: &mut R,
) -> Population<RealVector, f64> {
    let selection_pool: Vec<_> = population.as_fitness_pairs();
    let mut new_pop: Population<RealVector, f64> = Population::with_capacity(POP_SIZE);

    // Elitism
    if let Some(best) = population.best() {
        new_pop.push(best.clone());
    }

    while new_pop.len() < POP_SIZE {
        let p1_idx = selection.select(&selection_pool, rng);
        let p2_idx = selection.select(&selection_pool, rng);

        let (mut c1, mut c2) = crossover
            .crossover(&selection_pool[p1_idx].0, &selection_pool[p2_idx].0, rng)
            .genome()
            .unwrap_or_else(|| {
                (
                    selection_pool[p1_idx].0.clone(),
                    selection_pool[p2_idx].0.clone(),
                )
            });

        mutation.mutate(&mut c1, rng);
        mutation.mutate(&mut c2, rng);

        new_pop.push(Individual::new(c1));
        if new_pop.len() < POP_SIZE {
            new_pop.push(Individual::new(c2));
        }
    }

    new_pop.evaluate(fitness);
    new_pop
}

/// Run `generations` generations start-to-finish and return the best fitness.
fn run_straight(generations: usize) -> f64 {
    let (fitness, bounds, selection, crossover, mutation) = make_setup();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut population: Population<RealVector, f64> =
        Population::random(POP_SIZE, &bounds, &mut rng);
    population.evaluate(&fitness);

    for gen in 0..generations {
        population = evolve_one_generation(
            &population,
            &fitness,
            &selection,
            &crossover,
            &mutation,
            &mut rng,
        );
        population.set_generation(gen + 1);
    }

    *population.best().unwrap().fitness_value()
}

/// Run `checkpoint_at` generations, persist a checkpoint (population + RNG) to
/// disk, then load it back and continue to `total` generations. Returns the
/// final best fitness.
fn run_with_checkpoint(
    checkpoint_dir: &Path,
    total: usize,
    checkpoint_at: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let (fitness, bounds, selection, crossover, mutation) = make_setup();

    // --- Phase 1: run up to the checkpoint ---
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut population: Population<RealVector, f64> =
        Population::random(POP_SIZE, &bounds, &mut rng);
    population.evaluate(&fitness);

    for gen in 0..checkpoint_at {
        population = evolve_one_generation(
            &population,
            &fitness,
            &selection,
            &crossover,
            &mutation,
            &mut rng,
        );
        population.set_generation(gen + 1);
    }

    // Persist the checkpoint, capturing BOTH population and RNG state.
    let individuals: Vec<Individual<RealVector>> = population.iter().cloned().collect();
    let checkpoint = Checkpoint::new(checkpoint_at, individuals)
        .with_evaluations(checkpoint_at * POP_SIZE)
        .with_rng(&rng)?; // <-- the load-bearing line: capture the ChaCha RNG

    let path = checkpoint_dir.join("resume.ckpt");
    save_checkpoint(&checkpoint, &path, CheckpointFormat::Binary)?;
    println!("Saved checkpoint at generation {checkpoint_at} -> {path:?}");

    // --- Phase 2: simulate a restart and resume from disk ---
    let checkpoint: Checkpoint<RealVector> = load_checkpoint(&path)?;

    // Restore the exact RNG state captured above.
    let mut rng: ChaCha8Rng = checkpoint
        .restore_rng()?
        .expect("checkpoint must contain a captured RNG for reproducible resume");

    // Rebuild the population exactly as it was.
    let mut population: Population<RealVector, f64> =
        Population::with_capacity(checkpoint.population.len());
    for ind in checkpoint.population {
        population.push(ind);
    }

    for gen in checkpoint.generation..total {
        population = evolve_one_generation(
            &population,
            &fitness,
            &selection,
            &crossover,
            &mutation,
            &mut rng,
        );
        population.set_generation(gen + 1);
    }

    Ok(*population.best().unwrap().fitness_value())
}
