//! Online Operator-Parameter Learning with a Thompson-Sampling Bandit
//!
//! This example demonstrates the *wired-in* hyperparameter learner: a
//! [`ThompsonSamplingTuner`] driven directly by [`SimpleGA::run_adaptive`].
//!
//! Each generation the GA Thompson-samples a per-gene mutation probability and a
//! whole-genome crossover probability from the tuner, applies those arm values to
//! its operators, and credits each arm with the observed parent-vs-offspring
//! improvement events. The arm's Beta draw is used *only* to pick the arm — it is
//! never returned as the parameter value itself (the bug the old learner had).
//!
//! Run with:
//! ```text
//! cargo run --example hyperparameter_learning
//! ```

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Thompson-Sampling Operator-Parameter Learning ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    const DIM: usize = 20;
    let bounds = MultiBounds::symmetric(5.12, DIM);

    println!("Problem: {DIM}-D Rastrigin");
    println!("Learner: Thompson-sampling bandit over mutation- and crossover-probability arms\n");

    // Configure the bandit: candidate values ("arms") for each tunable parameter.
    // Each arm holds a Beta posterior over P(offspring improves on parents | arm).
    let config = ThompsonConfig {
        mutation_rate_arms: vec![0.01, 0.05, 0.1, 0.2, 0.4],
        crossover_prob_arms: vec![0.5, 0.7, 0.9],
        prior: BetaPosterior::uniform(),
        record_history: true,
    };

    println!("Mutation-rate arms:  {:?}", config.mutation_rate_arms);
    println!("Crossover-prob arms: {:?}\n", config.crossover_prob_arms);

    // Build a GA that consults the tuner every generation. `real_valued()` fixes
    // the genome/fitness types (no turbofish); we override the mutation operator
    // with a tunable Gaussian mutation whose per-gene probability the bandit sets.
    let mut ga = SimpleGABuilder::real_valued()
        .mutation(GaussianMutation::new(0.1))
        .population_size(100)
        .bounds(bounds)
        .fitness(Rastrigin::new(DIM))
        .max_generations(200)
        .adaptive_operators(config)
        .build()?;

    let result = ga.run_adaptive(&mut rng)?;

    // --- Posterior evolution ---------------------------------------------------
    let tuner = ga.tuner().expect("tuner is present after run_adaptive");
    let mr = tuner
        .parameter(PARAM_MUTATION_RATE)
        .expect("mutation-rate parameter");
    let arm_values = mr.values();

    println!("Posterior evolution — P(improvement) mean per mutation-rate arm:\n");
    print!("  {:>4}", "gen");
    for v in &arm_values {
        print!("   p={v:<5.2}");
    }
    println!("   selected");

    for snap in tuner.history().iter().step_by(20) {
        // Find this parameter's entry in the snapshot.
        if let Some((_, selected, means)) = snap
            .parameters
            .iter()
            .find(|(name, _, _)| name == PARAM_MUTATION_RATE)
        {
            print!("  {:>4}", snap.generation);
            for m in means {
                print!("   {m:>6.3}");
            }
            match selected {
                Some(v) => println!("   {v:.2}"),
                None => println!("   -"),
            }
        }
    }

    // --- Learned parameters ----------------------------------------------------
    println!("\n=== Learned operator parameters ===");
    for param in tuner.parameters() {
        let best = param.best_value();
        println!("\nParameter '{}':", param.name);
        for arm in param.arms() {
            let flag = if (arm.value - best).abs() < 1e-12 {
                " <-- best"
            } else {
                ""
            };
            println!(
                "  value {:<5.2}  P(improve)~{:.3}  pulls={}{}",
                arm.value,
                arm.posterior.mean(),
                arm.selections,
                flag
            );
        }
        println!("  => favored value: {best:.2}");
    }
    println!(
        "\nTotal improvement events fed back to the tuner: {}",
        tuner.total_observations()
    );

    println!("\n=== Result ===");
    println!("Best fitness (adaptive): {:.6}", result.best_fitness);

    // --- Comparison with fixed mutation rates ---------------------------------
    println!("\n--- Comparison with fixed mutation rates ---\n");
    for fixed_rate in [0.05, 0.1, 0.2, 0.5] {
        let best = run_with_fixed_rate(fixed_rate, DIM)?;
        println!("Fixed rate {fixed_rate:.2}: best = {best:.6}");
    }
    println!(
        "\nAdaptive (favored {:.2}): best = {:.6}",
        mr.best_value(),
        result.best_fitness
    );

    Ok(())
}

/// Run a non-adaptive GA with a fixed per-gene mutation probability for comparison.
fn run_with_fixed_rate(rate: f64, dim: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42); // Same seed for a fair comparison.
    let bounds = MultiBounds::symmetric(5.12, dim);

    let result = SimpleGABuilder::real_valued()
        .mutation(GaussianMutation::new(0.1).with_probability(rate))
        .population_size(100)
        .bounds(bounds)
        .fitness(Rastrigin::new(dim))
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    Ok(result.best_fitness)
}
