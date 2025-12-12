//! Hyperparameter Learning with Bayesian Adaptation
//!
//! This example demonstrates online Bayesian learning of GA hyperparameters.
//! The system learns optimal mutation rates based on observed fitness improvements.

use fugue_evo::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Bayesian Hyperparameter Learning ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    const DIM: usize = 20;
    let fitness = Rastrigin::new(DIM);
    let bounds = MultiBounds::symmetric(5.12, DIM);

    println!("Problem: {}-D Rastrigin", DIM);
    println!("Learning optimal mutation rate via Beta posterior\n");

    // Initialize Bayesian learner for mutation rate
    // Prior: Beta(2, 2) centered around 0.5
    let mut mutation_posterior = BetaPosterior::new(2.0, 2.0);

    // Track statistics
    let mut successful_mutations = 0;
    let mut total_mutations = 0;

    // Initialize population
    let mut population: Population<RealVector, f64> = Population::random(100, &bounds, &mut rng);
    population.evaluate(&fitness);

    let selection = TournamentSelection::new(3);
    let crossover = SbxCrossover::new(15.0);

    // Initial mutation rate from prior
    let mut current_mutation_rate = mutation_posterior.mean();

    println!("Initial mutation rate (prior mean): {:.4}", current_mutation_rate);
    println!();

    let max_generations = 200;
    let adaptation_interval = 20;

    for gen in 0..max_generations {
        // Sample mutation rate from current posterior every adaptation interval
        if gen > 0 && gen % adaptation_interval == 0 {
            current_mutation_rate = mutation_posterior.sample(&mut rng);
            println!(
                "Gen {:3}: Sampled mutation rate = {:.4} (posterior mean = {:.4})",
                gen,
                current_mutation_rate,
                mutation_posterior.mean()
            );
        }

        let selection_pool: Vec<_> = population.as_fitness_pairs();
        let mut new_pop: Population<RealVector, f64> = Population::with_capacity(100);

        // Elitism
        if let Some(best) = population.best() {
            new_pop.push(best.clone());
        }

        while new_pop.len() < 100 {
            let p1_idx = selection.select(&selection_pool, &mut rng);
            let p2_idx = selection.select(&selection_pool, &mut rng);

            let (mut c1, mut c2) = crossover
                .crossover(&selection_pool[p1_idx].0, &selection_pool[p2_idx].0, &mut rng)
                .genome()
                .unwrap_or_else(|| {
                    (selection_pool[p1_idx].0.clone(), selection_pool[p2_idx].0.clone())
                });

            let parent1_fitness = selection_pool[p1_idx].1;
            let parent2_fitness = selection_pool[p2_idx].1;

            // Apply mutation with learned rate
            let mutation = GaussianMutation::new(0.1).with_probability(current_mutation_rate);
            mutation.mutate(&mut c1, &mut rng);
            mutation.mutate(&mut c2, &mut rng);

            // Evaluate children
            let child1_fitness = fitness.evaluate(&c1);
            let child2_fitness = fitness.evaluate(&c2);

            // Update Bayesian posterior based on improvement
            let improved1 = child1_fitness > parent1_fitness;
            let improved2 = child2_fitness > parent2_fitness;

            mutation_posterior.observe(improved1);
            mutation_posterior.observe(improved2);

            total_mutations += 2;
            if improved1 { successful_mutations += 1; }
            if improved2 { successful_mutations += 1; }

            new_pop.push(Individual::with_fitness(c1, child1_fitness));
            if new_pop.len() < 100 {
                new_pop.push(Individual::with_fitness(c2, child2_fitness));
            }
        }

        new_pop.set_generation(gen + 1);
        population = new_pop;
    }

    // Results
    println!("\n=== Results ===");
    let best = population.best().unwrap();
    println!("Best fitness: {:.6}", best.fitness_value());
    println!();

    println!("Learned hyperparameters:");
    println!("  Final mutation rate (posterior mean): {:.4}", mutation_posterior.mean());
    let ci = mutation_posterior.credible_interval(0.95);
    println!("  95% credible interval: [{:.4}, {:.4}]", ci.0, ci.1);
    println!();

    println!("Mutation statistics:");
    println!("  Total mutations: {}", total_mutations);
    println!("  Successful mutations: {}", successful_mutations);
    println!(
        "  Observed success rate: {:.4}",
        successful_mutations as f64 / total_mutations as f64
    );

    // Compare with fixed rates
    println!("\n--- Comparison with fixed mutation rates ---\n");

    for fixed_rate in [0.05, 0.1, 0.2, 0.5] {
        let result = run_with_fixed_rate(fixed_rate, DIM)?;
        println!("Fixed rate {:.2}: Best = {:.6}", fixed_rate, result);
    }

    println!(
        "\nLearned rate {:.2}: Best = {:.6}",
        mutation_posterior.mean(),
        best.fitness_value()
    );

    Ok(())
}

fn run_with_fixed_rate(rate: f64, dim: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);  // Same seed for fair comparison

    let fitness = Rastrigin::new(dim);
    let bounds = MultiBounds::symmetric(5.12, dim);

    let mut population: Population<RealVector, f64> = Population::random(100, &bounds, &mut rng);
    population.evaluate(&fitness);

    let selection = TournamentSelection::new(3);
    let crossover = SbxCrossover::new(15.0);
    let mutation = GaussianMutation::new(0.1).with_probability(rate);

    for gen in 0..200 {
        let selection_pool: Vec<_> = population.as_fitness_pairs();
        let mut new_pop: Population<RealVector, f64> = Population::with_capacity(100);

        if let Some(best) = population.best() {
            new_pop.push(best.clone());
        }

        while new_pop.len() < 100 {
            let p1_idx = selection.select(&selection_pool, &mut rng);
            let p2_idx = selection.select(&selection_pool, &mut rng);

            let (mut c1, mut c2) = crossover
                .crossover(&selection_pool[p1_idx].0, &selection_pool[p2_idx].0, &mut rng)
                .genome()
                .unwrap_or_else(|| {
                    (selection_pool[p1_idx].0.clone(), selection_pool[p2_idx].0.clone())
                });

            mutation.mutate(&mut c1, &mut rng);
            mutation.mutate(&mut c2, &mut rng);

            new_pop.push(Individual::new(c1));
            if new_pop.len() < 100 {
                new_pop.push(Individual::new(c2));
            }
        }

        new_pop.evaluate(&fitness);
        new_pop.set_generation(gen + 1);
        population = new_pop;
    }

    Ok(*population.best().unwrap().fitness_value())
}
