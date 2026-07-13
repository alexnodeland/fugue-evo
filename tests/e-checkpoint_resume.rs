//! Integration test for reproducible checkpoint resume.
//!
//! regression: EV-02 - a run that is checkpointed to disk (population + RNG),
//! restored, and continued must reach a *bit-identical* result to an
//! uninterrupted run. Before the fix the RNG state was never captured, so a
//! resumed run silently diverged.

use fugue_evo::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::tempdir;

const DIM: usize = 6;
const POP_SIZE: usize = 24;
const SEED: u64 = 20260710;

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

    if let Some(best) = population.best() {
        new_pop.push(best.clone());
    }

    while new_pop.len() < POP_SIZE {
        let p1 = selection.select(&selection_pool, rng);
        let p2 = selection.select(&selection_pool, rng);
        let (mut c1, mut c2) = crossover
            .crossover(&selection_pool[p1].0, &selection_pool[p2].0, rng)
            .genome()
            .unwrap_or_else(|| (selection_pool[p1].0.clone(), selection_pool[p2].0.clone()));
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

fn run_with_checkpoint(dir: &std::path::Path, total: usize, checkpoint_at: usize) -> f64 {
    let (fitness, bounds, selection, crossover, mutation) = make_setup();

    // Phase 1
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

    // Persist through disk, capturing population + RNG.
    let individuals: Vec<Individual<RealVector>> = population.iter().cloned().collect();
    let checkpoint = Checkpoint::new(checkpoint_at, individuals)
        .with_rng(&rng)
        .expect("capture rng");
    let path = dir.join("resume.ckpt");
    save_checkpoint(&checkpoint, &path, CheckpointFormat::Binary).expect("save");

    // Phase 2: restore from disk.
    let checkpoint: Checkpoint<RealVector> = load_checkpoint(&path).expect("load");
    let mut rng: ChaCha8Rng = checkpoint
        .restore_rng()
        .expect("restore rng result")
        .expect("rng present in checkpoint");
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
    *population.best().unwrap().fitness_value()
}

#[test]
fn resume_is_bit_identical_to_uninterrupted_run() {
    let dir = tempdir().unwrap();
    let total = 12;
    let checkpoint_at = 6;

    let straight = run_straight(total);
    let resumed = run_with_checkpoint(dir.path(), total, checkpoint_at);

    assert_eq!(
        straight.to_bits(),
        resumed.to_bits(),
        "resumed run diverged: straight={straight} resumed={resumed}"
    );
}

// ---------------------------------------------------------------------------
// EV-02 (library-level resume): the algorithm now provides the resume path
// itself — `SimpleGA::checkpoint_run` / `resume` / `run_from_checkpoint` thread
// a SnapshotRng-capable ChaCha RNG through the incremental run loop — so a user
// no longer has to reimplement the generation loop (as `run_straight` /
// `run_with_checkpoint` above do). These tests drive that library API only.
// ---------------------------------------------------------------------------

type SphereGa = SimpleGA<
    RealVector,
    f64,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
    Sphere,
    MaxGenerations,
>;

fn build_ga(total: usize) -> SphereGa {
    SimpleGABuilder::real_valued()
        .population_size(POP_SIZE)
        .bounds(MultiBounds::symmetric(5.12, DIM))
        .fitness(Sphere::new(DIM))
        .max_generations(total)
        .build()
        .expect("build GA")
}

#[test]
fn library_resume_is_bit_identical() {
    let total = 12;
    let checkpoint_at = 6;
    let ga = build_ga(total);

    // Uninterrupted run driven purely through the library stepping API.
    let mut rng_a = ChaCha8Rng::seed_from_u64(SEED);
    let mut state_a = ga.init_run(&mut rng_a).expect("init_run");
    while ga.step_generation(&mut state_a, &mut rng_a).expect("step") {}
    let straight = ga.finish_run(state_a);

    // Interrupted run: step to `checkpoint_at`, snapshot state + RNG via the
    // library, then resume and run to termination — no hand-rolled loop.
    let mut rng_b = ChaCha8Rng::seed_from_u64(SEED);
    let mut state_b = ga.init_run(&mut rng_b).expect("init_run");
    for _ in 0..checkpoint_at {
        assert!(ga.step_generation(&mut state_b, &mut rng_b).expect("step"));
    }
    let checkpoint = ga.checkpoint_run(&state_b, &rng_b).expect("checkpoint_run");
    assert!(
        checkpoint.rng_state.is_some(),
        "checkpoint_run must capture the RNG state"
    );
    assert_eq!(checkpoint.generation, checkpoint_at);

    let resumed = ga
        .run_from_checkpoint::<ChaCha8Rng>(&checkpoint)
        .expect("run_from_checkpoint");

    assert_eq!(
        straight.best_fitness.to_bits(),
        resumed.best_fitness.to_bits(),
        "library resume diverged: straight={} resumed={}",
        straight.best_fitness,
        resumed.best_fitness
    );
    assert_eq!(straight.generations, resumed.generations);
    assert_eq!(
        straight.best_genome.as_vec(),
        resumed.best_genome.as_vec(),
        "resumed best genome must be identical"
    );
}

#[test]
fn library_resume_through_disk_and_rejects_missing_rng() {
    let dir = tempdir().unwrap();
    let total = 10;
    let checkpoint_at = 4;
    let ga = build_ga(total);

    // Baseline uninterrupted run.
    let mut rng_a = ChaCha8Rng::seed_from_u64(SEED);
    let mut state_a = ga.init_run(&mut rng_a).expect("init_run");
    while ga.step_generation(&mut state_a, &mut rng_a).expect("step") {}
    let straight = ga.finish_run(state_a);

    // Checkpoint to disk mid-run, reload, and resume through the library API.
    let mut rng_b = ChaCha8Rng::seed_from_u64(SEED);
    let mut state_b = ga.init_run(&mut rng_b).expect("init_run");
    for _ in 0..checkpoint_at {
        assert!(ga.step_generation(&mut state_b, &mut rng_b).expect("step"));
    }
    let checkpoint = ga.checkpoint_run(&state_b, &rng_b).expect("checkpoint_run");
    let path = dir.path().join("library.ckpt");
    save_checkpoint(&checkpoint, &path, CheckpointFormat::Binary).expect("save");

    let loaded: Checkpoint<RealVector> = load_checkpoint(&path).expect("load");
    let resumed = ga
        .run_from_checkpoint::<ChaCha8Rng>(&loaded)
        .expect("run_from_checkpoint");

    assert_eq!(
        straight.best_fitness.to_bits(),
        resumed.best_fitness.to_bits(),
        "disk resume diverged: straight={} resumed={}",
        straight.best_fitness,
        resumed.best_fitness
    );

    // A checkpoint that carries no captured RNG cannot be resumed
    // bit-identically, so `resume` must reject it rather than silently diverging.
    let no_rng = Checkpoint::new(checkpoint_at, loaded.population.clone());
    assert!(no_rng.rng_state.is_none());
    let result = ga.resume::<ChaCha8Rng>(&no_rng);
    assert!(
        result.is_err(),
        "resume must reject a checkpoint with no captured RNG state"
    );
}
