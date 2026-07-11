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
