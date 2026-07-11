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
//!
//! Note that the whole example is expressed against the **library** resume API:
//! [`SimpleGA::init_run`]/[`SimpleGA::step_generation`] drive an incremental
//! run, [`SimpleGA::checkpoint_run`] snapshots it (population + RNG),
//! [`save_checkpoint`]/[`load_checkpoint`] round-trip it through disk, and
//! [`SimpleGA::run_from_checkpoint`] resumes it — no hand-rolled generation loop.

use fugue_evo::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::{Path, PathBuf};

const DIM: usize = 10;
const POP_SIZE: usize = 100;
const SEED: u64 = 42;
const TOTAL_GENERATIONS: usize = 20;
const CHECKPOINT_AT: usize = 10;

type SphereGa = SimpleGA<
    RealVector,
    f64,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
    Sphere,
    MaxGenerations,
>;

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
    //     disk, restore, then continue for the remaining gens — all via the
    //     library resume API.
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

/// Build the GA used by both runs. `real_valued()` fixes the genome/fitness
/// types and installs tournament selection, SBX crossover, and polynomial
/// mutation as defaults (no turbofish); elitism mirrors a typical config.
fn build_ga(total: usize) -> SphereGa {
    SimpleGABuilder::real_valued()
        .population_size(POP_SIZE)
        .bounds(MultiBounds::symmetric(5.12, DIM))
        .fitness(Sphere::new(DIM))
        .elitism(true)
        .elite_count(1)
        .max_generations(total)
        .build()
        .expect("build GA")
}

/// Run `generations` generations start-to-finish via the incremental API and
/// return the best fitness.
fn run_straight(generations: usize) -> f64 {
    let ga = build_ga(generations);
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut state = ga.init_run(&mut rng).expect("init_run");
    while ga.step_generation(&mut state, &mut rng).expect("step") {}
    ga.finish_run(state).best_fitness
}

/// Run `checkpoint_at` generations, persist a checkpoint (population + RNG) to
/// disk via the library, then load it back and continue to `total` generations
/// with [`SimpleGA::run_from_checkpoint`]. Returns the final best fitness.
fn run_with_checkpoint(
    checkpoint_dir: &Path,
    total: usize,
    checkpoint_at: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let ga = build_ga(total);

    // --- Phase 1: run up to the checkpoint ---
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut state = ga.init_run(&mut rng)?;
    for _ in 0..checkpoint_at {
        ga.step_generation(&mut state, &mut rng)?;
    }

    // Snapshot the in-progress run — the algorithm captures the population,
    // best, evaluations, statistics AND the ChaCha RNG state for us.
    let checkpoint = ga.checkpoint_run(&state, &rng)?;
    let path = checkpoint_dir.join("resume.ckpt");
    save_checkpoint(&checkpoint, &path, CheckpointFormat::Binary)?;
    println!("Saved checkpoint at generation {checkpoint_at} -> {path:?}");

    // --- Phase 2: simulate a restart and resume from disk ---
    let checkpoint: Checkpoint<RealVector> = load_checkpoint(&path)?;
    let resumed = ga.run_from_checkpoint::<ChaCha8Rng>(&checkpoint)?;

    Ok(resumed.best_fitness)
}
