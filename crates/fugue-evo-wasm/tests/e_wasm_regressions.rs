//! Host-side regression tests for the fugue-evo-wasm audit fixes.
//!
//! These are plain `#[test]`s (not `wasm_bindgen_test`, which only runs in a
//! browser) so they execute under `cargo test -p fugue-evo-wasm` on the host and
//! exercise the non-JS code paths (built-in fitness, stepping, island model,
//! NSGA-II eval counting, CMA-ES history). Each pins a specific audit finding.

// NOTE: Only success paths are exercised on the host. Constructing a `JsValue`
// (as every `Err(...)` branch does) aborts under wasm-bindgen's non-wasm stubs,
// so the structured-error paths (EV-32) are asserted in the browser
// `wasm_tests.rs` suite instead.
use fugue_evo_wasm::{
    Algorithm, IslandModelOptimizer, IslandTopology, Nsga2Optimizer, RealVectorOptimizer,
    SteppedRealOptimizer, ZdtProblem,
};

// ---------------------------------------------------------------------------
// EV-33: MultiObjectiveResult.evaluations must be the REAL evaluation count,
// not the fabricated `population_size * max_generations * 2` formula.
// ---------------------------------------------------------------------------
#[test]
fn ev33_nsga2_reports_real_evaluation_count() {
    // regression: EV-33
    // NSGA-II evaluates the initial population once (pop) plus `pop` offspring
    // per generation => pop * (gens + 1). The old code fabricated pop*gens*2.
    let pop = 12;
    let gens = 6;
    let mut o = Nsga2Optimizer::new(2, 2);
    o.set_population_size(pop);
    o.set_max_generations(gens);
    o.set_seed(123);

    let result = o
        .optimize_zdt(ZdtProblem::Zdt1)
        .expect("ZDT1 optimization should succeed");

    let real = pop * (gens + 1); // 84
    let fabricated = pop * gens * 2; // 144 (old, wrong)
    assert_eq!(
        result.evaluations(),
        real,
        "evaluations should be the real measured count pop*(gens+1)"
    );
    assert_ne!(
        result.evaluations(),
        fabricated,
        "evaluations must not be the old fabricated pop*gens*2 formula"
    );
}

// ---------------------------------------------------------------------------
// EV-76: CMA-ES fitness_history must be a per-generation trajectory, not a
// single point.
// ---------------------------------------------------------------------------
#[test]
fn ev76_cmaes_fitness_history_is_a_trajectory() {
    // regression: EV-76
    let gens = 30;
    let mut o = RealVectorOptimizer::new(4);
    o.set_algorithm(Algorithm::CmaES);
    o.set_cmaes_sigma(0.5);
    o.set_population_size(16);
    o.set_max_generations(gens);
    o.set_bounds(-5.0, 5.0);
    o.set_fitness("sphere");
    o.set_seed(7);

    let result = o.optimize().expect("CMA-ES should succeed");
    let history = result.fitness_history();

    // Pre-fix this was a hard-coded 1-element vector; now it is per-generation.
    assert!(
        history.len() > 1,
        "CMA-ES fitness history must have more than one point (got {})",
        history.len()
    );
    assert!(
        history.len() <= gens,
        "history should not exceed the generation budget"
    );
    // Best-so-far must be monotonically non-decreasing (higher = better here).
    for w in history.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-9,
            "best-so-far trajectory must be non-decreasing: {:?}",
            w
        );
    }
}

// ---------------------------------------------------------------------------
// EV-34: incremental / cancellable step API.
// ---------------------------------------------------------------------------
#[test]
fn ev34_stepped_optimizer_advances_and_reports_progress() {
    // regression: EV-34
    let mut s = SteppedRealOptimizer::new(3);
    s.set_population_size(20);
    s.set_max_generations(15);
    s.set_bounds(-5.0, 5.0);
    s.set_fitness("sphere");
    s.set_seed(99);

    assert!(!s.is_started());
    // Take 5 generations.
    let still_running = s.step(5).expect("step should succeed");
    assert!(still_running, "should still be running after 5/15 gens");
    assert!(s.is_started());
    assert_eq!(s.current_generation(), 5);
    let evals_after_5 = s.evaluations();
    assert!(evals_after_5 > 0);
    assert!(s.best_fitness().is_finite());
    assert_eq!(s.fitness_history().len(), 6); // gen 0..=5

    // Take 5 more; progress must advance monotonically.
    assert!(s.step(5).expect("step"));
    assert_eq!(s.current_generation(), 10);
    assert!(s.evaluations() > evals_after_5);
}

#[test]
fn ev34_stepped_optimizer_finishes_and_matches_one_shot() {
    // regression: EV-34 — driving the step loop to completion must produce the
    // exact same result as the one-shot RealVectorOptimizer.optimize() for the
    // same seed/config (the native run() is implemented via the same step API).
    let mut s = SteppedRealOptimizer::new(3);
    s.set_population_size(20);
    s.set_max_generations(15);
    s.set_bounds(-5.0, 5.0);
    s.set_fitness("sphere");
    s.set_seed(99);
    while s.step(4).expect("step") {}
    assert!(s.is_finished());
    assert_eq!(s.current_generation(), 15);

    let mut one = RealVectorOptimizer::new(3);
    one.set_population_size(20);
    one.set_max_generations(15);
    one.set_bounds(-5.0, 5.0);
    one.set_fitness("sphere");
    one.set_seed(99);
    let r = one.optimize().expect("optimize");

    assert_eq!(r.generations(), s.current_generation());
    assert!(
        (r.best_fitness() - s.best_fitness()).abs() < 1e-12,
        "stepped ({}) must equal one-shot ({})",
        s.best_fitness(),
        r.best_fitness()
    );
}

#[test]
fn ev34_stepped_optimizer_supports_early_cancel() {
    // regression: EV-34 — a caller can stop stepping at any point and read a
    // consistent partial result.
    let mut s = SteppedRealOptimizer::new(2);
    s.set_population_size(15);
    s.set_max_generations(50);
    s.set_bounds(-5.0, 5.0);
    s.set_fitness("sphere");
    s.set_seed(3);

    s.step(3).expect("step");
    // "Cancel" by simply not stepping further and snapshotting.
    let snapshot = s.get_result().expect("get_result");
    assert_eq!(snapshot.generations(), 3);
    assert!(!s.is_finished(), "cancelled before the generation budget");
    assert_eq!(snapshot.best_genome().len(), 2);
}

// ---------------------------------------------------------------------------
// EV-77: island-model WASM binding.
// ---------------------------------------------------------------------------
#[test]
fn ev77_island_model_runs_with_migration() {
    // regression: EV-77
    let dim = 3;
    let gens = 20;
    let mut o = IslandModelOptimizer::new(dim, 4);
    assert_eq!(o.num_islands(), 4);
    o.set_population_size(20);
    o.set_max_generations(gens);
    o.set_bounds(-5.0, 5.0);
    o.set_fitness("sphere");
    o.set_seed(1);
    o.set_migration_interval(5);
    o.set_migration_size(2);

    let result = o.optimize().expect("island model should succeed");
    assert_eq!(result.best_genome().len(), dim);
    assert!(result.best_fitness().is_finite());
    // 4 islands * 20 pop * (20 gens + 1) initial+offspring, plus migrants:
    // at minimum every island evaluated its initial population.
    assert!(result.evaluations() >= 4 * 20);
    assert_eq!(result.fitness_history().len(), gens);
    // Global best-so-far must be non-decreasing (higher = better).
    for w in result.fitness_history().windows(2) {
        assert!(w[1] >= w[0] - 1e-9);
    }
}

#[test]
fn ev77_island_model_all_topologies_smoke() {
    // regression: EV-77 — every topology must run without panicking and produce
    // a well-formed result.
    for topo in [
        IslandTopology::Ring,
        IslandTopology::FullyConnected,
        IslandTopology::Star,
    ] {
        let mut o = IslandModelOptimizer::new(2, 3);
        o.set_population_size(12);
        o.set_max_generations(12);
        o.set_bounds(-5.0, 5.0);
        o.set_fitness("sphere");
        o.set_seed(42);
        o.set_migration_interval(3);
        o.set_topology(topo);
        let result = o.optimize().expect("topology run should succeed");
        assert_eq!(result.best_genome().len(), 2);
        assert!(result.best_fitness().is_finite());
    }
}
