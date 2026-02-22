# Hyperparameter Learning Tutorial

GA performance depends heavily on hyperparameters like mutation rate, crossover probability, and population size. This tutorial demonstrates **online Bayesian learning** of hyperparameters during evolution.

## The Problem

Traditional approach: Set parameters once, hope they work.

Better approach: **Learn** optimal parameters from feedback during evolution.

## Hyperparameter Control Methods

Fugue-evo supports several approaches (following Eiben et al.'s classification):

| Method | Description | Example |
|--------|-------------|---------|
| **Deterministic** | Pre-defined schedule | Decay mutation over time |
| **Adaptive** | Rule-based adjustment | Increase mutation if stagnant |
| **Self-Adaptive** | Encode in genome | Parameters evolve with solutions |
| **Bayesian** | Statistical learning | Update beliefs from observations |

This tutorial focuses on **Bayesian learning** with conjugate priors.

## Complete Example

```rust,ignore
{{#include ../../../examples/hyperparameter_learning.rs}}
```

> **Source**: [`examples/hyperparameter_learning.rs`](https://github.com/fugue-evo/fugue-evo/blob/main/examples/hyperparameter_learning.rs)

## Running the Example

```bash
cargo run --example hyperparameter_learning
```

## Key Components

### Beta Posterior for Mutation Rate

```rust,ignore
// Prior: Beta(2, 2) centered around 0.5
let mut mutation_posterior = BetaPosterior::new(2.0, 2.0);
```

The **Beta distribution** is perfect for learning probabilities:
- Domain: [0, 1] (valid probability range)
- Conjugate to Bernoulli outcomes (success/failure)
- Prior parameters encode initial beliefs

**Beta(2, 2):**
- Mean = 0.5 (start uncertain)
- Moderate confidence (equivalent to 4 observations)

### Observing Outcomes

```rust,ignore
// Check if mutation improved fitness
let improved = child_fitness > parent_fitness;

// Update posterior with observation
mutation_posterior.observe(improved);
```

Each observation updates the distribution:
- Success (improvement): Increases mean
- Failure: Decreases mean
- More observations → narrower distribution

### Sampling Parameters

```rust,ignore
// Sample mutation rate from current posterior
current_mutation_rate = mutation_posterior.sample(&mut rng);
```

**Thompson Sampling**: Sample from posterior, use as parameter.
- Balances exploration (uncertainty) and exploitation (best estimate)
- Naturally adapts as confidence grows

### Adaptation Interval

```rust,ignore
if gen % adaptation_interval == 0 {
    current_mutation_rate = mutation_posterior.sample(&mut rng);
}
```

Don't update every generation:
- Too frequent: Not enough signal
- Too rare: Slow adaptation
- Typical: Every 10-50 generations

## Understanding the Output

```text
Initial mutation rate (prior mean): 0.5000

Gen  20: Sampled mutation rate = 0.4123 (posterior mean = 0.3876)
Gen  40: Sampled mutation rate = 0.3456 (posterior mean = 0.3245)
...

=== Results ===
Learned hyperparameters:
  Final mutation rate (posterior mean): 0.2134
  95% credible interval: [0.1823, 0.2445]

Mutation statistics:
  Total mutations: 20000
  Successful mutations: 4268
  Observed success rate: 0.2134
```

The posterior converges toward the empirically optimal rate.

### Credible Intervals

```rust,ignore
let ci = mutation_posterior.credible_interval(0.95);
println!("95% CI: [{:.4}, {:.4}]", ci.0, ci.1);
```

Unlike frequentist confidence intervals, Bayesian credible intervals have a direct interpretation: "95% probability the true value is in this range (given our data)."

## Comparing with Fixed Rates

```rust,ignore
for fixed_rate in [0.05, 0.1, 0.2, 0.5] {
    let result = run_with_fixed_rate(fixed_rate)?;
    println!("Fixed rate {:.2}: Best = {:.6}", fixed_rate, result);
}
```

The learned rate often outperforms any single fixed rate because:
1. It adapts to the problem
2. It can change as evolution progresses
3. It handles different phases (exploration vs. exploitation)

## Other Learnable Parameters

### Crossover Probability

```rust,ignore
let mut crossover_posterior = BetaPosterior::new(2.0, 2.0);

// Observe: did crossover produce better offspring than parents?
let offspring_better = child_fitness > max(parent1_fitness, parent2_fitness);
crossover_posterior.observe(offspring_better);
```

### Tournament Size

Use a categorical posterior for discrete choices:

```rust,ignore
let tournament_sizes = [2, 3, 5, 7];
let mut size_weights = vec![1.0; tournament_sizes.len()];

// Update weights based on selection quality
// ... observe which sizes produce better offspring
```

### Multiple Parameters

Learn multiple parameters simultaneously:

```rust,ignore
struct AdaptiveGA {
    mutation_posterior: BetaPosterior,
    crossover_posterior: BetaPosterior,
    // ... other parameters
}

impl AdaptiveGA {
    fn adapt(&mut self, gen: usize, rng: &mut Rng) {
        if gen % interval == 0 {
            self.mutation_rate = self.mutation_posterior.sample(rng);
            self.crossover_prob = self.crossover_posterior.sample(rng);
        }
    }
}
```

## Deterministic Schedules

For simpler adaptation, use time-based schedules:

```rust,ignore
use fugue_evo::hyperparameter::schedules::*;

// Linear decay: 0.5 → 0.05 over 500 generations
let schedule = LinearSchedule::new(0.5, 0.05, 500);
let rate = schedule.value_at(gen);

// Exponential decay
let schedule = ExponentialSchedule::new(0.5, 0.99, 500);

// Sigmoid decay
let schedule = SigmoidSchedule::new(0.5, 0.05, 500);
```

## When to Use Which Method

| Scenario | Recommended |
|----------|-------------|
| Known good parameters | Fixed |
| Exploration→exploitation | Deterministic schedule |
| Problem-dependent optimal | Bayesian learning |
| No prior knowledge | Bayesian with weak prior |
| Fast prototyping | Adaptive rules |

## Exercises

1. **Prior sensitivity**: Try Beta(1,1), Beta(5,5), Beta(10,1) priors
2. **Learning speed**: Vary adaptation interval (5, 20, 50, 100)
3. **Multiple parameters**: Learn both mutation and crossover rates

## Next Steps

- [Custom Operators](../how-to/custom-operators.md) - Create learnable custom operators
- [Advanced Algorithms](./advanced-algorithms.md) - Built-in adaptive algorithms
