//! Cross-cutting NaN-policy regression tests.
//!
//! regression: EV-106 — the crate documents a specific NaN policy: a fitness
//! function that returns `NaN` must panic at the point it is recorded
//! (`Individual::set_fitness`, see `src/population/individual.rs`), rather than
//! silently corrupting downstream best/worst/sort comparisons (EV-07, wave A).
//! That guard has direct unit coverage in `population/individual.rs`, but
//! nothing previously exercised it through the *public, end-to-end* algorithm
//! entry points a real caller would hit (`SimpleGA::run`, `EvolutionStrategy::run`,
//! `ContinuousUMDA::run`, and the underlying `Population::evaluate`). These
//! tests close that gap: each algorithm that routes fitness through
//! `Population::evaluate` / `Individual::set_fitness` must panic loudly and
//! immediately, with a message identifying the NaN, when given a fitness
//! function that returns `NaN` — never silently rank a `NaN` individual as
//! "best" or "worst" and continue.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A fitness function that always returns `NaN`, standing in for a
/// mis-implemented objective (e.g. `0.0 / 0.0`, `log` of a negative number).
struct NanFitness;

impl Fitness for NanFitness {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, _genome: &RealVector) -> f64 {
        f64::NAN
    }
}

// ==================== Population::evaluate (the guard itself) ====================

#[test]
#[should_panic(expected = "NaN")]
fn population_evaluate_panics_on_nan_fitness() {
    let mut rng = StdRng::seed_from_u64(1);
    let bounds = MultiBounds::symmetric(5.0, 4);
    let mut population: Population<RealVector, f64> = Population::random(6, &bounds, &mut rng);
    population.evaluate(&NanFitness);
}

// ==================== SimpleGA ====================

#[test]
#[should_panic(expected = "NaN")]
fn simple_ga_run_panics_on_nan_fitness() {
    let mut rng = StdRng::seed_from_u64(2);
    let ga = SimpleGABuilder::real_valued()
        .population_size(10)
        .bounds(MultiBounds::symmetric(5.0, 4))
        .fitness(NanFitness)
        .max_generations(5)
        .build()
        .expect("build");
    // Must panic during the initial-population evaluation, not silently
    // produce a "best" individual carrying a NaN fitness.
    let _ = ga.run(&mut rng);
}

// ==================== Evolution Strategy ====================

#[test]
#[should_panic(expected = "NaN")]
fn evolution_strategy_run_panics_on_nan_fitness() {
    let mut rng = StdRng::seed_from_u64(3);
    let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::mu_plus_lambda(5, 20)
        .bounds(MultiBounds::symmetric(5.0, 4))
        .fitness(NanFitness)
        .max_generations(5)
        .build()
        .expect("build");
    let _ = es.run(&mut rng);
}

// ==================== UMDA ====================

#[test]
#[should_panic(expected = "NaN")]
fn umda_run_panics_on_nan_fitness() {
    let mut rng = StdRng::seed_from_u64(4);
    let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
        .population_size(10)
        .bounds(MultiBounds::symmetric(5.0, 4))
        .fitness(NanFitness)
        .max_generations(5)
        .build()
        .expect("build");
    let _ = umda.run(&mut rng);
}
