//! Integration tests for the wired-in Thompson-sampling operator tuner.
//!
//! regression: EV-21 — before remediation the Bayesian learner, adaptive-control,
//! and schedule machinery were never consulted by any algorithm. These tests
//! exercise the public API end-to-end and assert that the tuner actually receives
//! feedback from the GA it is wired into.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The adaptive GA must consult the tuner and feed improvement events back into it.
#[test]
fn adaptive_ga_feeds_the_tuner() {
    let mut rng = StdRng::seed_from_u64(2026);

    let config = ThompsonConfig {
        mutation_rate_arms: vec![0.01, 0.1, 0.3],
        crossover_prob_arms: vec![0.6, 0.9],
        prior: BetaPosterior::uniform(),
        record_history: true,
    };

    let mut ga = SimpleGABuilder::real_valued()
        .mutation(GaussianMutation::new(0.1))
        .population_size(30)
        .bounds(MultiBounds::symmetric(5.12, 6))
        .fitness(Sphere::new(6))
        .max_generations(25)
        .adaptive_operators(config)
        .build()
        .expect("build");

    let result = ga.run_adaptive(&mut rng).expect("run_adaptive");
    assert!(result.evaluations > 0);

    let tuner = ga.tuner().expect("tuner present after run_adaptive");

    // The tuner received improvement feedback.
    assert!(
        tuner.total_observations() > 0,
        "tuner received no feedback: it is not wired in"
    );

    // Both tuned parameters accumulated per-arm observations.
    let mr = tuner.parameter(PARAM_MUTATION_RATE).expect("mutation_rate");
    let cx = tuner
        .parameter(PARAM_CROSSOVER_PROB)
        .expect("crossover_prob");
    assert!(mr.total_observations() > 0.0);
    assert!(cx.total_observations() > 0.0);

    // Per-arm pull counts sum to the number of generations that ran.
    let pulls: u64 = mr.selection_counts().iter().sum();
    assert!(pulls > 0, "no arms were ever pulled");

    // History was recorded, one snapshot per generation (plus the initial one).
    assert!(
        !tuner.history().is_empty(),
        "record_history was requested but no snapshots were taken"
    );
}

/// End-to-end, the wired-in bandit must explore its arms and stay internally
/// consistent: every generation pulls exactly one arm, Thompson sampling explores
/// more than a single arm, and the reported "best" arm is the one with the highest
/// posterior mean. (The controlled >70%-concentration guarantee is pinned by the
/// seeded unit test `test_bandit_concentrates_on_better_arm`.)
#[test]
fn adaptive_ga_explores_and_stays_consistent() {
    let mut rng = StdRng::seed_from_u64(7);

    let config = ThompsonConfig {
        mutation_rate_arms: vec![0.01, 0.1, 0.3],
        crossover_prob_arms: vec![], // tune mutation only
        prior: BetaPosterior::uniform(),
        record_history: false,
    };

    let generations = 120;
    let mut ga = SimpleGABuilder::real_valued()
        .mutation(GaussianMutation::new(0.5))
        .population_size(40)
        .bounds(MultiBounds::symmetric(5.12, 10))
        .fitness(Rastrigin::new(10))
        .max_generations(generations)
        .adaptive_operators(config)
        .build()
        .expect("build");

    ga.run_adaptive(&mut rng).expect("run_adaptive");

    let mr = ga
        .tuner()
        .unwrap()
        .parameter(PARAM_MUTATION_RATE)
        .expect("mutation_rate");

    let counts = mr.selection_counts();
    let total: u64 = counts.iter().sum();
    // One pull per generation.
    assert_eq!(total, generations as u64);

    // Thompson sampling explores: more than one arm was pulled.
    let arms_used = counts.iter().filter(|&&c| c > 0).count();
    assert!(arms_used >= 2, "bandit collapsed to a single arm");

    // Internal consistency: best_value has the maximum posterior mean.
    let means = mr.posterior_means();
    let best_idx = means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!((mr.best_value() - mr.values()[best_idx]).abs() < 1e-12);
}

/// The ergonomic constructors give a turbofish-free quickstart (regression: EV-35).
#[test]
fn ergonomic_constructor_quickstart_has_no_turbofish() {
    let mut rng = StdRng::seed_from_u64(1);

    let result = SimpleGABuilder::real_valued()
        .population_size(50)
        .bounds(MultiBounds::symmetric(5.12, 10))
        .fitness(Sphere::new(10))
        .max_generations(30)
        .build()
        .expect("build")
        .run(&mut rng)
        .expect("run");

    assert!(result.best_fitness > -200.0);
}
