//! CMA-ES Optimization Example
//!
//! This example demonstrates the Covariance Matrix Adaptation Evolution
//! Strategy (CMA-ES), a state-of-the-art derivative-free optimization
//! algorithm for continuous domains.
//!
//! CMA-ES adapts both the mean and covariance of a multivariate normal
//! distribution to efficiently search the fitness landscape.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CMA-ES Optimization Example ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Rosenbrock function - a classic test for optimization algorithms
    // The global minimum is at (1, 1, ..., 1) with value 0
    const DIM: usize = 10;

    println!("Problem: {}-D Rosenbrock function", DIM);
    println!("Global optimum: 0.0 at (1, 1, ..., 1)\n");

    // Create CMA-ES optimizer
    let initial_mean = vec![0.0; DIM]; // Start at origin
    let initial_sigma = 0.5; // Initial step size
    let bounds = MultiBounds::symmetric(5.0, DIM);

    // Create fitness function that CMA-ES can use
    let fitness = RosenbrockCmaEs { dim: DIM };

    let mut cmaes = CmaEs::new(initial_mean, initial_sigma).with_bounds(bounds);

    // Run optimization
    let best = cmaes.run_generations(&fitness, 1000, &mut rng)?;

    println!("Results:");
    println!("  Best fitness (minimized): {:.10}", best.fitness_value());
    println!("  Generations: {}", cmaes.state.generation);
    println!("  Evaluations: {}", cmaes.state.evaluations);
    println!("  Final sigma: {:.6}", cmaes.state.sigma);

    println!("\nBest solution:");
    for (i, val) in best.genome.genes().iter().enumerate().take(5) {
        println!("  x[{}] = {:.6}", i, val);
    }
    if DIM > 5 {
        println!("  ... ({} more dimensions)", DIM - 5);
    }

    // Calculate distance from optimal solution
    let distance_from_opt: f64 = best
        .genome
        .genes()
        .iter()
        .map(|x| (x - 1.0).powi(2))
        .sum::<f64>()
        .sqrt();

    println!("\nDistance from optimum: {:.10}", distance_from_opt);

    Ok(())
}

/// Rosenbrock function wrapper for CMA-ES
struct RosenbrockCmaEs {
    dim: usize,
}

impl CmaEsFitness for RosenbrockCmaEs {
    fn evaluate(&self, x: &RealVector) -> f64 {
        let genes = x.genes();
        let mut sum = 0.0;
        for i in 0..self.dim - 1 {
            let term1 = genes[i + 1] - genes[i] * genes[i];
            let term2 = 1.0 - genes[i];
            sum += 100.0 * term1 * term1 + term2 * term2;
        }
        sum // CMA-ES minimizes, so return positive value
    }
}
