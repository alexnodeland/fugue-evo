# Interactive Evolution Tutorial

**Interactive Genetic Algorithms (IGA)** incorporate human feedback into the evolutionary process. Instead of an automated fitness function, users evaluate candidates through ratings, comparisons, or selections.

## When to Use Interactive Evolution

**Ideal for:**
- Aesthetic optimization (art, design, music)
- Subjective preferences (user interfaces)
- Hard-to-formalize objectives
- Creative exploration

**Challenges:**
- User fatigue limits evaluations
- Noisy, inconsistent feedback
- Slower convergence than automated GA

## Evaluation Modes

Fugue-evo supports multiple ways to gather user feedback:

| Mode | User Action | Best For |
|------|-------------|----------|
| **Rating** | Score each candidate 1-10 | Absolute quality assessment |
| **Pairwise** | Choose better of two | Relative comparisons |
| **Batch Selection** | Pick best N from batch | Quick approximate ranking |

## Complete Example

```rust,ignore
{{#include ../../../examples/interactive_evolution.rs}}
```

> **Source**: [`examples/interactive_evolution.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/interactive_evolution.rs)

## Running the Example

```bash
cargo run --example interactive_evolution
```

This example simulates user feedback. In a real application, you would replace the simulation with actual UI interaction.

## Key Components

### Building an Interactive GA

```rust,ignore
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
```

**Key parameters:**
- `evaluation_mode`: How users provide feedback
- `batch_size`: Candidates shown per evaluation round
- `min_coverage`: Fraction of population needing evaluation
- `aggregation_model`: How to combine multiple evaluations

### The Step Loop

```rust,ignore
loop {
    match iga.step(&mut rng) {
        StepResult::NeedsEvaluation(request) => {
            // Present to user, get feedback
            let response = get_user_response(&request);
            iga.provide_response(response);
        }

        StepResult::GenerationComplete { generation, best_fitness, coverage } => {
            println!("Generation {} complete", generation);
        }

        StepResult::Complete(result) => {
            println!("Evolution complete!");
            break;
        }
    }
}
```

### Handling Evaluation Requests

**Rating Mode:**
```rust,ignore
EvaluationRequest::RateCandidates { candidates, .. } => {
    // Show candidates to user
    for candidate in candidates {
        display_candidate(&candidate.genome);
    }
    // Collect ratings
    let ratings: Vec<(CandidateId, f64)> = /* user input */;
    EvaluationResponse::ratings(ratings)
}
```

**Pairwise Mode:**
```rust,ignore
EvaluationRequest::PairwiseComparison { candidate_a, candidate_b, .. } => {
    // Show both candidates
    display_comparison(&candidate_a.genome, &candidate_b.genome);
    // Get user's choice
    let winner = /* user choice */;
    EvaluationResponse::winner(winner)
}
```

**Batch Selection:**
```rust,ignore
EvaluationRequest::BatchSelection { candidates, select_count, .. } => {
    // Show all candidates
    for c in candidates { display_candidate(&c.genome); }
    // User selects best N
    let selected: Vec<CandidateId> = /* user picks */;
    EvaluationResponse::selected(selected)
}
```

## Aggregation Models

How to combine feedback into fitness estimates:

### Direct Rating

```rust,ignore
AggregationModel::DirectRating { default_rating: 5.0 }
```

Uses ratings directly as fitness. Unevaluated candidates get the default.

### Implicit Ranking

```rust,ignore
AggregationModel::ImplicitRanking {
    selected_bonus: 1.0,
    not_selected_penalty: 0.3,
    base_fitness: 5.0,
}
```

For batch selection mode:
- Selected candidates get bonus
- Non-selected get penalty
- Accumulates over evaluations

### Bradley-Terry Model

For pairwise comparisons, estimates latent "skill" from win/loss records using the Bradley-Terry statistical model.

## Reducing User Fatigue

### Smaller Population

```rust,ignore
.population_size(12)
```

Fewer candidates = fewer evaluations needed.

### Coverage Threshold

```rust,ignore
.min_coverage(0.5)  // Only evaluate 50% of population
```

Not every candidate needs evaluation each generation.

### Batch Size Tuning

```rust,ignore
.batch_size(4)  // Show 4 at a time
```

- Too small: Many rounds, tedious
- Too large: Overwhelming, poor decisions

### Evaluation Budget

```rust,ignore
.max_evaluations(100)  // Stop after 100 user interactions
```

## Simulating User Feedback

For testing, simulate user preferences:

```rust,ignore
struct SimulatedUser {
    target: Vec<f64>,
}

impl SimulatedUser {
    fn rate(&self, genome: &RealVector) -> f64 {
        let distance: f64 = genome.genes().iter()
            .zip(self.target.iter())
            .map(|(g, t)| (g - t).powi(2))
            .sum::<f64>()
            .sqrt();

        // Closer to target = higher rating
        10.0 - distance.min(9.0)
    }
}
```

This allows testing IGA logic without human interaction.

## Real-World Integration

### Web Application

```rust,ignore
// Pseudocode for web integration
async fn evolution_endpoint(state: &mut IgaState) -> Response {
    match state.iga.step(&mut state.rng) {
        StepResult::NeedsEvaluation(request) => {
            // Return candidates to frontend
            Json(CandidatesForEvaluation::from(request))
        }
        // ...
    }
}

async fn feedback_endpoint(feedback: UserFeedback, state: &mut IgaState) {
    let response = EvaluationResponse::from(feedback);
    state.iga.provide_response(response);
}
```

### GUI Application

```rust,ignore
fn update(&mut self, message: Message) {
    match message {
        Message::NextStep => {
            if let StepResult::NeedsEvaluation(req) = self.iga.step(&mut self.rng) {
                self.current_request = Some(req);
            }
        }
        Message::UserRated(ratings) => {
            let response = EvaluationResponse::ratings(ratings);
            self.iga.provide_response(response);
        }
    }
}
```

## Exercises

1. **Different modes**: Compare Rating vs Batch Selection modes
2. **Noisy preferences**: Add randomness to simulated user, observe robustness
3. **Visualization**: Display RealVector as colors/shapes for visual evaluation

## Next Steps

- [Hyperparameter Learning](./hyperparameter-learning.md) - Adaptive parameter tuning
- [Custom Fitness Functions](../how-to/custom-fitness.md) - Combine interactive with automated fitness
