//! Optimizer mode: annealed inference vs a classic GA on the same problem.
//!
//! `EvolutionSMC::anneal` keeps tempering past the posterior (β = 1) toward
//! `β_max`, so the particle population concentrates on the optima — a
//! principled, uncertainty-aware single-objective optimizer built entirely
//! from inference machinery. This example runs it head-to-head with the
//! classic `SimpleGA` on the sphere benchmark and prints what each layer
//! gives you.
//!
//! Run with: `cargo run --example optimize_by_inference`

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const DIM: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Single-objective optimization: annealed inference vs classic GA ===\n");
    let mut rng = StdRng::seed_from_u64(20260728);

    // Sphere: f(x) = -Σx², optimum 0 at the origin.
    let fitness = Sphere::new(DIM);
    let bounds = MultiBounds::symmetric(5.12, DIM);

    // ---------------- Classic GA ----------------
    let ga_result = SimpleGABuilder::real_valued()
        .population_size(100)
        .bounds(bounds.clone())
        .fitness(Sphere::new(DIM))
        .max_generations(200)
        .build()?
        .run(&mut rng)?;
    println!("-- SimpleGA (classic layer) --");
    println!(
        "  best fitness: {:.6}   (~{} fitness evaluations)",
        ga_result.best_fitness,
        100 * 200
    );

    // ---------------- Annealed inference ----------------
    let model = EvolutionModel::new(UniformBoxPrior::new(bounds), fitness.clone());
    let annealed = EvolutionSMC::anneal(
        &mut rng,
        &model,
        EvoSmcConfig {
            num_particles: 300,
            rejuvenation_steps: 4,
            crossover: Some(CrossoverConfig::default()),
            ..Default::default()
        },
        500.0, // β_max: how hard to anneal
        15,    // annealing rungs past β = 1
    );
    let model_fn = model.smc_model();
    let (best, best_f) = annealed.best(&fitness, &model_fn).unwrap();
    println!("\n-- EvolutionSMC::anneal (inference layer) --");
    println!("  best fitness: {:.6}", best_f);
    println!("  best genome:  {:?}", best.genes());
    println!(
        "  population spread at β=500: {:.4} (posterior-style uncertainty, not a point)",
        (0..DIM)
            .map(|i| annealed.weighted_variance(i))
            .sum::<f64>()
            .sqrt()
    );
    println!(
        "  log evidence (β ≤ 1 ladder): {:.3} — a model score no GA can report",
        annealed.log_evidence
    );

    Ok(())
}
