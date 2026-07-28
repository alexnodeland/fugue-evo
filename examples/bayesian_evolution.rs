//! Evolution as Bayesian inference — an end-to-end pipeline through the
//! `inference` layer.
//!
//! Given a fitness `f` and a prior *program* `p(x)` (a [`GenomePrior`] — any
//! fugue `Model<G>`), the Boltzmann / Gibbs posterior
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x))
//! ```
//!
//! is itself a fugue program: `prior.model().bind(|g| factor(β·f(g)))`. This
//! example runs fugue's tempered Sequential Monte Carlo against that program —
//! adaptive β ladder from the prior (β = 0) to the posterior (β = 1),
//! ESS-driven resampling, typed single-site MH rejuvenation, a
//! population-coupled crossover kernel, and an unbiased log-evidence estimate.
//! Because the prior is Gaussian and the fitness quadratic, the posterior is a
//! known conjugate Gaussian, so every estimate is printed next to the analytic
//! truth.
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
    let model = EvolutionModel::new(GaussianPrior::new(0.0, sigma0, DIM), fitness.clone());

    println!("-- Tempered SMC (fugue adaptive_smc_with_kernel) --");
    let result = EvolutionSMC::run(
        &mut rng,
        &model,
        EvoSmcConfig {
            num_particles: 3000,
            rejuvenation_steps: 5,
            crossover: Some(CrossoverConfig {
                n_pairs: 500,
                swap_probability: 0.5,
            }),
            ..Default::default()
        },
    );

    let tau = 1.0 / (sigma0 * sigma0) + 1.0;
    let analytic_mean = center / tau;
    let analytic_var = 1.0 / tau;

    for coord in 0..DIM {
        let mean = result.weighted_mean(coord);
        let var = result.weighted_variance(coord);
        println!(
            "  coord {coord}: posterior mean {mean:7.4}  (analytic {analytic_mean:7.4})   \
             variance {var:6.4}  (analytic {analytic_var:6.4})"
        );
    }
    println!(
        "  log evidence: {:.4} (Bayesian model score, free from the tempering ladder)",
        result.log_evidence
    );

    // Optimizer-mode readout: the MAP-ish best particle by raw fitness.
    let model_fn = model.smc_model();
    if let Some((best, best_f)) = result.best(&fitness, &model_fn) {
        println!(
            "  best genome (decode-replay): {:?} with fitness {best_f:.4}\n",
            best.genes()
        );
    }

    // ---------------------------------------------------------------------
    // Part 2: A fixed-β Metropolis-Hastings chain over the same target.
    // ---------------------------------------------------------------------
    println!("-- MH chain (fugue adaptive_single_site_mh) --");
    let chain_model = EvolutionModel::new(GaussianPrior::new(0.0, sigma0, 1), Quadratic { center })
        .with_beta(1.0);
    let mut chain = EvolutionChain::new(chain_model);
    let mut current = chain.init(&mut rng);
    let mut samples = Vec::new();
    for i in 0..20_000 {
        let (g, t) = chain.step(&mut rng, &current);
        current = t;
        if i >= 2_000 {
            samples.push(g.genes()[0]);
        }
    }
    let mh_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("  MH posterior mean {mh_mean:7.4}  (analytic {analytic_mean:7.4})\n");

    // ---------------------------------------------------------------------
    // Part 3: The Bayesian adaptive GA (Thompson sampling over operators).
    // ---------------------------------------------------------------------
    println!("-- Bayesian adaptive GA (conjugate posteriors + Thompson sampling) --");
    let sphere = Sphere::new(4);
    let bounds = MultiBounds::symmetric(5.12, 4);
    let mut ga = BayesianAdaptiveGA::new(UniformBoxPrior::new(bounds), sphere, 60, 80);
    let result = ga.run(&mut rng);

    println!("  best fitness: {:.6}", result.best_fitness);
    for (i, arm) in result.operator_posteriors.iter().enumerate() {
        println!(
            "  arm {i}: σ = {:5.2}  posterior mean success = {:.3}  (selected {}×)",
            arm.sigma,
            arm.posterior.mean(),
            arm.times_selected
        );
    }
    println!(
        "  improvement-rate posterior mean: {:.2} events/generation",
        result.improvement_rate.mean()
    );

    Ok(())
}
