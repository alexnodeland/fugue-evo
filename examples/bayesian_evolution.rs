//! Evolution as Bayesian inference — an end-to-end pipeline through the
//! `fugue_integration` layer.
//!
//! This flagship example makes the "evolution as Bayesian inference over
//! solution spaces" story concrete and checkable. Given a fitness `f` and a
//! prior `p`, it targets the **Boltzmann / Gibbs posterior**
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x))
//! ```
//!
//! and runs a genuine tempered Sequential Monte Carlo sampler (through Fugue's
//! `Model`/`Handler`/`factor` machinery) from the prior (`β = 0`) to the
//! posterior (`β = 1`), using trace-based mutation and crossover as
//! `π_β`-invariant rejuvenation moves. Because the prior is Gaussian and the
//! fitness quadratic, the posterior is a known conjugate Gaussian, so we can
//! print the SMC estimate next to the analytic truth.
//!
//! It then runs the single-level Bayesian adaptive GA, which learns which
//! mutation step size works via conjugate `Beta`/`Gamma` posteriors and
//! Thompson sampling.
//!
//! Run with: `cargo run --example bayesian_evolution`

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Quadratic fitness `f(x) = -0.5 · Σ (x_i - center)²` (maximised at `x_i = center`).
///
/// `exp(β·f)` is Gaussian, so with a Gaussian prior the Boltzmann posterior is a
/// conjugate Gaussian with a closed form we can check against.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Evolution as Bayesian Inference ===\n");

    // Fixed seed for reproducibility (audit date 2026-07-10).
    let mut rng = StdRng::seed_from_u64(20260710);

    // ---------------------------------------------------------------------
    // Part 1: Tempered SMC over the Boltzmann posterior.
    // ---------------------------------------------------------------------
    // Prior:   x_i ~ N(0, σ0²) with σ0 = 2 ⇒ prior precision τ0 = 1/σ0² = 0.25.
    // Fitness: f(x) = -0.5·Σ(x_i - c)² with c = 3 ⇒ likelihood precision k = 1.
    // Posterior at β = 1 (per coordinate):
    //     precision τ = τ0 + k = 1.25,
    //     mean     μ = (τ0·0 + k·c)/τ = c / 1.25 = 2.4,
    //     variance = 1/τ = 0.8.
    const DIM: usize = 2;
    let center = 3.0;
    let sigma0 = 2.0;

    let fitness = Quadratic { center };
    let bounds = MultiBounds::symmetric(30.0, DIM);

    let smc = EvolutionarySMC::new(fitness.clone(), bounds.clone(), 3000)
        .with_prior(Prior::Gaussian {
            mean: 0.0,
            std: sigma0,
        })
        .with_beta_schedule(EvolutionarySMC::<RealVector, Quadratic>::linear_beta_schedule(16))
        .with_mcmc_steps(6)
        .with_mutation(0.9, 0.6)
        .with_crossover(true); // trace-based crossover as a rejuvenation move

    println!(
        "Running tempered SMC: prior N(0, {:.1}²) ⇒ Boltzmann posterior over a",
        sigma0
    );
    println!(
        "quadratic fitness peaked at {}, along a β ladder 0 → 1.\n",
        center
    );

    let particles = smc.run(&mut rng);

    let est_mean = EvolutionarySMC::<RealVector, Quadratic>::weighted_mean(&particles);
    let est_var = EvolutionarySMC::<RealVector, Quadratic>::weighted_variance(&particles);
    let ess = EvolutionarySMC::<RealVector, Quadratic>::effective_sample_size(&particles);

    let tau0 = 1.0 / (sigma0 * sigma0);
    let tau = tau0 + 1.0;
    let analytic_mean = center / tau;
    let analytic_var = 1.0 / tau;

    println!(
        "  Effective sample size at β=1 : {:.1} / {}",
        ess,
        particles.len()
    );
    println!(
        "  Posterior mean     (SMC)     : [{:.3}, {:.3}]",
        est_mean[0], est_mean[1]
    );
    println!(
        "  Posterior mean     (analytic): [{:.3}, {:.3}]",
        analytic_mean, analytic_mean
    );
    println!(
        "  Posterior variance (SMC)     : [{:.3}, {:.3}]",
        est_var[0], est_var[1]
    );
    println!(
        "  Posterior variance (analytic): [{:.3}, {:.3}]",
        analytic_var, analytic_var
    );

    if let Some(best) = EvolutionarySMC::<RealVector, Quadratic>::best_particle(&particles) {
        println!(
            "  MAP-ish best particle        : {:?} (f = {:.4})\n",
            best.genome.genes(),
            best.fitness
        );
    }

    // ---------------------------------------------------------------------
    // Part 2: fitness as a literal likelihood in a Fugue trace.
    // ---------------------------------------------------------------------
    // `to_weighted_trace` runs the genuine Fugue model `factor(β·f(x))` through
    // a real Handler, so the trace's total log-weight IS β·f(x).
    let model = EvolutionModel::new(fitness.clone(), bounds.clone()).with_beta(1.0);
    let probe = RealVector::new(vec![center, center]); // the fitness peak
    let wt = model.to_weighted_trace(&probe);
    println!("Fitness as likelihood (β = 1):");
    println!(
        "  f(peak)                = {:.4}",
        model.fitness_value(&probe)
    );
    println!(
        "  trace.total_log_weight = {:.4}  (== β·f, in log_factors: {:.4})\n",
        wt.total_log_weight(),
        wt.log_factors
    );

    // ---------------------------------------------------------------------
    // Part 3: single-level Bayesian adaptive GA (Thompson sampling).
    // ---------------------------------------------------------------------
    println!("Running Bayesian adaptive GA (Thompson sampling over step sizes)...\n");
    let sphere = Sphere::new(5);
    let ga_bounds = MultiBounds::symmetric(5.12, 5);
    let mut ga = BayesianAdaptiveGA::new(sphere, ga_bounds, 60, 120)
        .with_step_sizes(vec![0.02, 0.1, 0.5, 2.0]);
    let result = ga.run(&mut rng);

    println!("  Best fitness found          : {:.5}", result.best_fitness);
    println!("  Learned operator posteriors (Beta success probability):");
    for arm in &result.operator_posteriors {
        println!(
            "    σ = {:<4}  E[θ] = {:.3}  (α={:.0}, β={:.0}, selected {} gens)",
            arm.sigma,
            arm.posterior.mean(),
            arm.posterior.alpha,
            arm.posterior.beta,
            arm.times_selected
        );
    }
    println!(
        "  Improvement-rate posterior  : Gamma(shape={:.1}, rate={:.1}), E[λ] = {:.2} improving/gen\n",
        result.improvement_rate.shape,
        result.improvement_rate.rate,
        result.improvement_rate.mean()
    );

    println!("Done. Evolution ran as genuine posterior inference through the Fugue layer.");
    Ok(())
}
