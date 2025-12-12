//! Interactive Genetic Algorithm Example
//!
//! This example demonstrates how to use the Interactive GA for human-in-the-loop
//! evolutionary optimization. Instead of an automated fitness function, users
//! provide feedback by rating, comparing, or selecting candidates.
//!
//! In this example, we simulate user feedback with a simple automated scorer,
//! but in a real application, you would present candidates to users via a UI.

use fugue_evo::genome::bounds::Bounds;
use fugue_evo::interactive::prelude::*;
use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Simulates user preference for solutions close to a target
struct SimulatedUserPreference {
    target: Vec<f64>,
}

impl SimulatedUserPreference {
    fn new(dim: usize) -> Self {
        // User prefers solutions where values are around 0.5
        Self {
            target: vec![0.5; dim],
        }
    }

    /// Simulate a user rating (1-10 scale)
    fn rate(&self, genome: &RealVector) -> f64 {
        let distance: f64 = genome
            .genes()
            .iter()
            .zip(self.target.iter())
            .map(|(g, t)| (g - t).powi(2))
            .sum::<f64>()
            .sqrt();

        // Convert distance to rating (closer = higher rating)
        let rating = 10.0 - (distance * 5.0).min(9.0);
        rating.max(1.0)
    }

    /// Simulate pairwise comparison
    fn compare(&self, a: &RealVector, b: &RealVector) -> std::cmp::Ordering {
        let rating_a = self.rate(a);
        let rating_b = self.rate(b);
        rating_a.partial_cmp(&rating_b).unwrap()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Interactive Genetic Algorithm Demo ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    const DIM: usize = 5;
    let bounds = MultiBounds::uniform(Bounds::new(0.0, 1.0), DIM);

    // Create the simulated user preference
    let user = SimulatedUserPreference::new(DIM);

    // Build the Interactive GA
    let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
        .population_size(12)
        .elitism_count(2)
        .evaluation_mode(EvaluationMode::Rating)
        .batch_size(4)
        .min_coverage(0.8)
        .max_generations(5)
        .aggregation_model(AggregationModel::DirectRating {
            default_rating: 5.0,
        })
        .bounds(bounds)
        .selection(TournamentSelection::new(2))
        .crossover(SbxCrossover::new(15.0))
        .mutation(PolynomialMutation::new(20.0))
        .build()?;

    println!("Starting interactive evolution...\n");
    println!(
        "Configuration: {} individuals, {} mode, {} generations max",
        iga.config().population_size,
        match iga.config().evaluation_mode {
            EvaluationMode::Rating => "rating",
            EvaluationMode::Pairwise => "pairwise",
            EvaluationMode::BatchSelection => "batch selection",
            EvaluationMode::Adaptive => "adaptive",
        },
        iga.config().max_generations
    );
    println!();

    // Main evolution loop
    loop {
        match iga.step(&mut rng) {
            StepResult::NeedsEvaluation(request) => {
                // In a real app, you'd present this to a user via UI
                // Here we simulate user feedback
                let response = simulate_user_response(&user, &request);
                iga.provide_response(response);
            }

            StepResult::GenerationComplete {
                generation,
                best_fitness,
                coverage,
            } => {
                println!(
                    "Generation {} complete: best = {:.2}, coverage = {:.0}%",
                    generation,
                    best_fitness.unwrap_or(0.0),
                    coverage * 100.0
                );
            }

            StepResult::Complete(result) => {
                println!("\n=== Evolution Complete ===");
                println!("Reason: {}", result.termination_reason);
                println!("Generations: {}", result.generations);
                println!("Total evaluations: {}", result.total_evaluations);
                println!("\nTop 3 candidates:");

                for (i, candidate) in result.best_candidates.iter().take(3).enumerate() {
                    println!(
                        "  #{}: fitness = {:.2}, genes = {:?}",
                        i + 1,
                        candidate.fitness_estimate.unwrap_or(0.0),
                        candidate
                            .genome
                            .genes()
                            .iter()
                            .map(|g| format!("{:.3}", g))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                break;
            }
        }
    }

    // Demonstrate different evaluation modes
    println!("\n=== Batch Selection Mode Demo ===\n");

    let mut iga_batch = InteractiveGABuilder::<RealVector, (), (), ()>::new()
        .population_size(12)
        .evaluation_mode(EvaluationMode::BatchSelection)
        .batch_size(6)
        .select_count(2)
        .min_coverage(0.5)
        .max_generations(3)
        .aggregation_model(AggregationModel::ImplicitRanking {
            selected_bonus: 1.0,
            not_selected_penalty: 0.3,
            base_fitness: 5.0,
        })
        .bounds(MultiBounds::uniform(Bounds::new(0.0, 1.0), DIM))
        .selection(TournamentSelection::new(2))
        .crossover(SbxCrossover::new(15.0))
        .mutation(PolynomialMutation::new(20.0))
        .build()?;

    loop {
        match iga_batch.step(&mut rng) {
            StepResult::NeedsEvaluation(request) => {
                let response = simulate_user_response(&user, &request);
                iga_batch.provide_response(response);
            }

            StepResult::GenerationComplete {
                generation,
                best_fitness,
                ..
            } => {
                println!(
                    "Generation {}: best = {:.2}",
                    generation,
                    best_fitness.unwrap_or(0.0)
                );
            }

            StepResult::Complete(result) => {
                println!("\nBatch selection mode complete!");
                println!(
                    "Best fitness: {:.2}",
                    result.best_candidates[0].fitness_estimate.unwrap_or(0.0)
                );
                break;
            }
        }
    }

    Ok(())
}

/// Simulate user response to an evaluation request
fn simulate_user_response(
    user: &SimulatedUserPreference,
    request: &EvaluationRequest<RealVector>,
) -> EvaluationResponse {
    match request {
        EvaluationRequest::RateCandidates { candidates, .. } => {
            let ratings: Vec<_> = candidates
                .iter()
                .map(|c| (c.id, user.rate(&c.genome)))
                .collect();
            EvaluationResponse::ratings(ratings)
        }

        EvaluationRequest::PairwiseComparison {
            candidate_a,
            candidate_b,
            ..
        } => {
            use std::cmp::Ordering;
            match user.compare(&candidate_a.genome, &candidate_b.genome) {
                Ordering::Greater => EvaluationResponse::winner(candidate_a.id),
                Ordering::Less => EvaluationResponse::winner(candidate_b.id),
                Ordering::Equal => EvaluationResponse::tie(),
            }
        }

        EvaluationRequest::BatchSelection {
            candidates,
            select_count,
            ..
        } => {
            // Sort by rating and select top N
            let mut rated: Vec<_> = candidates
                .iter()
                .map(|c| (c.id, user.rate(&c.genome)))
                .collect();
            rated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let selected: Vec<_> = rated
                .iter()
                .take(*select_count)
                .map(|(id, _)| *id)
                .collect();
            EvaluationResponse::selected(selected)
        }
    }
}
