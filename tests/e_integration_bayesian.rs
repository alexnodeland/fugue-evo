//! Integration tests for the `fugue_integration` Bayesian-evolution pipeline.
//!
//! These exercise the same end-to-end path as `examples/bayesian_evolution.rs`
//! (tempered SMC over the Boltzmann posterior + the Bayesian adaptive GA) with
//! small budgets, through the public crate API only.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Quadratic fitness `f(x) = -0.5·Σ(x_i - center)²`, maximised at `center`.
#[derive(Clone)]
struct Quadratic {
    center: f64,
}

impl Fitness for Quadratic {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        -0.5 * genome
            .genes()
            .iter()
            .map(|x| (x - self.center).powi(2))
            .sum::<f64>()
    }
}

#[test]
fn e_integration_smc_targets_conjugate_posterior() {
    // regression: EV-16/EV-52/EV-90 — the SMC pipeline is a valid tempered
    // sampler that lands on the analytic Gaussian conjugate posterior.
    //
    // Prior N(0, 2²) ⇒ τ0 = 0.25; f(x) = -0.5(x-2)² ⇒ k = 1, c = 2.
    // Posterior at β=1: mean = c/(τ0+1) = 1.6, variance = 1/(τ0+1) = 0.8.
    let center = 2.0;
    let sigma0 = 2.0;
    let bounds = MultiBounds::symmetric(25.0, 1);
    let smc = EvolutionarySMC::new(Quadratic { center }, bounds, 1500)
        .with_prior(Prior::Gaussian {
            mean: 0.0,
            std: sigma0,
        })
        .with_beta_schedule(EvolutionarySMC::<RealVector, Quadratic>::linear_beta_schedule(12))
        .with_mcmc_steps(5)
        .with_mutation(0.9, 0.6)
        .with_crossover(true);

    let mut rng = StdRng::seed_from_u64(1234);
    let particles = smc.run(&mut rng);

    let tau = 1.0 / (sigma0 * sigma0) + 1.0;
    let analytic_mean = center / tau;
    let analytic_var = 1.0 / tau;

    let mean = EvolutionarySMC::<RealVector, Quadratic>::weighted_mean(&particles)[0];
    let var = EvolutionarySMC::<RealVector, Quadratic>::weighted_variance(&particles)[0];

    assert!(
        (mean - analytic_mean).abs() < 0.2,
        "SMC posterior mean {} vs analytic {}",
        mean,
        analytic_mean
    );
    assert!(
        (var - analytic_var).abs() < 0.25,
        "SMC posterior variance {} vs analytic {}",
        var,
        analytic_var
    );

    let total: f64 = particles.iter().map(|p| p.normalized_weight).sum();
    assert!((total - 1.0).abs() < 1e-6, "weights must self-normalise");
}

#[test]
fn e_integration_weighted_trace_is_boltzmann_weight() {
    // regression: EV-52 — total_log_weight() == β·f(x).
    let bounds = MultiBounds::symmetric(5.0, 3);
    let model = EvolutionModel::new(Quadratic { center: 0.0 }, bounds).with_beta(1.5);
    let g = RealVector::new(vec![1.0, 0.0, -2.0]);
    let f = model.fitness_value(&g);
    let trace = model.to_weighted_trace(&g);
    assert!((trace.total_log_weight() - 1.5 * f).abs() < 1e-9);
}

#[test]
fn e_integration_mh_stays_in_bounds() {
    // regression: EV-90 — the MH kernel never leaves the uniform-prior bounds.
    let bounds = MultiBounds::symmetric(3.0, 2);
    let model = EvolutionModel::new(Quadratic { center: 100.0 }, bounds).with_beta(1.0);
    // center far outside bounds pushes the chain toward the boundary.
    let config = EvolutionChainConfig::default()
        .mutation_rate(1.0)
        .mutation_sigma(1.0);
    let step = EvolutionStep::new(model, config);

    let mut rng = StdRng::seed_from_u64(77);
    let mut current = RealVector::new(vec![0.0, 0.0]);
    for _ in 0..5000 {
        current = step.step(&current, &mut rng);
        for &x in current.genes() {
            assert!((-3.0..=3.0).contains(&x), "escaped bounds: {}", x);
        }
    }
}

#[test]
fn e_integration_bayesian_ga_learns_and_optimises() {
    // regression: EV-53 — the adaptive GA performs genuine conjugate posterior
    // updates and makes optimisation progress on a small budget.
    let sphere = Sphere::new(4);
    let bounds = MultiBounds::symmetric(5.12, 4);
    let mut ga = BayesianAdaptiveGA::new(sphere, bounds, 40, 40);
    let mut rng = StdRng::seed_from_u64(9);
    let result = ga.run(&mut rng);

    // Posteriors accumulated the observed improvement trials.
    let evidence: f64 = result
        .operator_posteriors
        .iter()
        .map(|a| a.posterior.total() - 2.0)
        .sum();
    assert!(evidence >= (40 * 40) as f64 - 1.0, "evidence {}", evidence);
    assert!(result.improvement_rate.shape > 1.0);
    assert!(result.best_fitness > -5.0, "best {}", result.best_fitness);
    assert_eq!(result.fitness_history.len(), 40);
}
