//! Seeded-determinism regression suite.
//!
//! regression: EV-106 — fugue-evo had no test analogous to fugue's
//! `test_replay_and_score_handlers` (tests/model_execution.rs) proving that a
//! seeded RNG makes a full run byte-for-byte reproducible. Without such a test,
//! nothing would catch accidental hidden nondeterminism creeping into an
//! algorithm (e.g. `HashMap` iteration order, an internal `rand::thread_rng()`
//! call, or unordered parallel reduction) — the algorithm-level unit tests only
//! ever run *once* per seed and assert a stability bound on that single run, so
//! they cannot see this class of bug.
//!
//! For each of SimpleGA, CMA-ES, (μ+λ)-ES, UMDA, and NSGA-II this file:
//!  1. runs the algorithm twice with `StdRng::seed_from_u64(SAME_SEED)` and
//!     asserts the best genome and best fitness (or, for NSGA-II, the whole
//!     final population) come out byte-identical, and
//!  2. runs it once more with a different seed and asserts the result differs
//!     (almost surely true for continuous search spaces / floating point
//!     fitness landscapes; a hash collision here would itself be newsworthy).

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SEED_A: u64 = 20260710;
const SEED_B: u64 = 20260711;

// ==================== SimpleGA ====================

fn run_simple_ga(seed: u64) -> (RealVector, f64) {
    let dim = 6;
    let mut rng = StdRng::seed_from_u64(seed);
    let result = SimpleGABuilder::real_valued()
        .population_size(24)
        .bounds(MultiBounds::symmetric(5.12, dim))
        .fitness(Sphere::new(dim))
        .max_generations(15)
        .build()
        .expect("build")
        .run(&mut rng)
        .expect("run");
    (result.best_genome, result.best_fitness)
}

#[test]
fn simple_ga_same_seed_is_deterministic() {
    let (genome_a, fitness_a) = run_simple_ga(SEED_A);
    let (genome_b, fitness_b) = run_simple_ga(SEED_A);
    assert_eq!(
        genome_a, genome_b,
        "same seed must reproduce the same best genome"
    );
    assert_eq!(
        fitness_a, fitness_b,
        "same seed must reproduce the same best fitness bit-for-bit"
    );
}

#[test]
fn simple_ga_different_seed_differs() {
    let (genome_a, fitness_a) = run_simple_ga(SEED_A);
    let (genome_b, fitness_b) = run_simple_ga(SEED_B);
    assert!(
        genome_a != genome_b || fitness_a != fitness_b,
        "different seeds should (almost surely) not reproduce an identical run"
    );
}

// ==================== CMA-ES ====================

/// Sphere fitness (minimization, lower is better) for direct `CmaEsFitness` use
/// — a plain `fn` item rather than a closure, matching this module's own
/// `rosenbrock` test helper and sidestepping the need to name an anonymous
/// closure type in the `CmaEs<F>` annotation below.
fn sphere_cmaes_fitness(x: &RealVector) -> f64 {
    x.genes().iter().map(|v| v * v).sum()
}

fn run_cmaes(seed: u64) -> (Vec<f64>, f64) {
    let dim = 6;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cmaes: CmaEs<_> = CmaEsBuilder::new()
        .mean(vec![2.0; dim])
        .sigma(0.5)
        .build()
        .expect("build");
    let best = cmaes
        .run_generations(&sphere_cmaes_fitness, 20, &mut rng)
        .expect("run_generations");
    (best.genome().genes().to_vec(), best.fitness_f64())
}

#[test]
fn cmaes_same_seed_is_deterministic() {
    let (genome_a, fitness_a) = run_cmaes(SEED_A);
    let (genome_b, fitness_b) = run_cmaes(SEED_A);
    assert_eq!(
        genome_a, genome_b,
        "same seed must reproduce the same best solution vector"
    );
    assert_eq!(
        fitness_a, fitness_b,
        "same seed must reproduce the same best fitness bit-for-bit"
    );
}

#[test]
fn cmaes_different_seed_differs() {
    let (genome_a, fitness_a) = run_cmaes(SEED_A);
    let (genome_b, fitness_b) = run_cmaes(SEED_B);
    assert!(
        genome_a != genome_b || fitness_a != fitness_b,
        "different seeds should (almost surely) not reproduce an identical run"
    );
}

// ==================== (mu+lambda)-ES ====================

fn run_es(seed: u64) -> (RealVector, f64) {
    let dim = 6;
    let mut rng = StdRng::seed_from_u64(seed);
    let es: EvolutionStrategy<RealVector, f64, _, _> = ESBuilder::mu_plus_lambda(6, 24)
        .initial_sigma(1.0)
        .self_adaptive(true)
        .bounds(MultiBounds::symmetric(5.12, dim))
        .fitness(Sphere::new(dim))
        .max_generations(15)
        .build()
        .expect("build");
    let result = es.run(&mut rng).expect("run");
    (result.best_genome, result.best_fitness)
}

#[test]
fn es_same_seed_is_deterministic() {
    let (genome_a, fitness_a) = run_es(SEED_A);
    let (genome_b, fitness_b) = run_es(SEED_A);
    assert_eq!(
        genome_a, genome_b,
        "same seed must reproduce the same best genome"
    );
    assert_eq!(
        fitness_a, fitness_b,
        "same seed must reproduce the same best fitness bit-for-bit"
    );
}

#[test]
fn es_different_seed_differs() {
    let (genome_a, fitness_a) = run_es(SEED_A);
    let (genome_b, fitness_b) = run_es(SEED_B);
    assert!(
        genome_a != genome_b || fitness_a != fitness_b,
        "different seeds should (almost surely) not reproduce an identical run"
    );
}

// ==================== UMDA ====================

fn run_umda(seed: u64) -> (RealVector, f64) {
    let dim = 6;
    let mut rng = StdRng::seed_from_u64(seed);
    let umda: ContinuousUMDA<f64, _, _> = UMDABuilder::new()
        .population_size(40)
        .selection_ratio(0.3)
        .bounds(MultiBounds::symmetric(5.12, dim))
        .fitness(Sphere::new(dim))
        .max_generations(15)
        .build()
        .expect("build");
    let result = umda.run(&mut rng).expect("run");
    (result.best_genome, result.best_fitness)
}

#[test]
fn umda_same_seed_is_deterministic() {
    let (genome_a, fitness_a) = run_umda(SEED_A);
    let (genome_b, fitness_b) = run_umda(SEED_A);
    assert_eq!(
        genome_a, genome_b,
        "same seed must reproduce the same best genome"
    );
    assert_eq!(
        fitness_a, fitness_b,
        "same seed must reproduce the same best fitness bit-for-bit"
    );
}

#[test]
fn umda_different_seed_differs() {
    let (genome_a, fitness_a) = run_umda(SEED_A);
    let (genome_b, fitness_b) = run_umda(SEED_B);
    assert!(
        genome_a != genome_b || fitness_a != fitness_b,
        "different seeds should (almost surely) not reproduce an identical run"
    );
}

// ==================== NSGA-II ====================

/// Flatten an NSGA-II final population into a comparable form: each
/// individual's genome and objective vector, in return order. `Nsga2Individual`
/// does not derive `PartialEq` (its `crowding_distance` field is a `f64`
/// computed quantity, not identity-bearing), so we compare the fields that
/// define a solution rather than deriving equality on the whole struct.
fn nsga2_signature(population: &[Nsga2Individual<RealVector>]) -> Vec<(Vec<f64>, Vec<f64>)> {
    population
        .iter()
        .map(|ind| (ind.genome.genes().to_vec(), ind.objectives.clone()))
        .collect()
}

fn run_nsga2(seed: u64) -> Vec<(Vec<f64>, Vec<f64>)> {
    let dim = 8;
    let bounds = MultiBounds::new(vec![Bounds::new(0.0, 1.0); dim]);
    let zdt = Zdt1::new(dim);
    let fitness =
        ClosureMultiObjective::new(2, move |g: &RealVector| zdt.evaluate(g.genes()).to_vec());
    let crossover = SbxCrossover::new(15.0);
    let mutation = PolynomialMutation::new(20.0);
    let nsga2: Nsga2<RealVector, _, _, _> = Nsga2::new(20);
    let mut rng = StdRng::seed_from_u64(seed);
    // `run_bounded` (rather than `run`) clamps offspring back into [0, 1] after
    // crossover/mutation, matching ZDT1's domain. Without it, SBX/polynomial
    // operators can push a gene negative, sending ZDT1's `sqrt(f1 / g)` term
    // to `NaN` — which is an unrelated pre-existing sharp edge, not the
    // determinism property under test here (and `NaN != NaN` would make the
    // straightforward `assert_eq!` below spuriously fail even on identical
    // bit-for-bit reproductions).
    let population = nsga2
        .run_bounded(&fitness, &crossover, &mutation, &bounds, 12, &mut rng)
        .expect("run_bounded");
    nsga2_signature(&population)
}

#[test]
fn nsga2_same_seed_is_deterministic() {
    let population_a = run_nsga2(SEED_A);
    let population_b = run_nsga2(SEED_A);
    assert_eq!(
        population_a, population_b,
        "same seed must reproduce an identical final population (genomes and objectives)"
    );
}

#[test]
fn nsga2_different_seed_differs() {
    let population_a = run_nsga2(SEED_A);
    let population_b = run_nsga2(SEED_B);
    assert_ne!(
        population_a, population_b,
        "different seeds should (almost surely) not reproduce an identical final population"
    );
}
