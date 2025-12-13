# Fugue Integration

Fugue-evo integrates deeply with [fugue-ppl](https://github.com/fugue-ppl/fugue), a probabilistic programming library. This enables novel trace-based genetic operators.

## Core Concept: Genomes as Traces

In fugue-ppl, a **trace** records all random choices made during program execution. Fugue-evo represents genomes as traces:

```rust,ignore
// A genome's genes become trace entries
genome.to_trace() → {
    addr!("gene", 0) → 1.234,
    addr!("gene", 1) → -0.567,
    addr!("gene", 2) → 3.890,
    // ...
}
```

## Why Traces?

### 1. Selective Resampling

Mutation becomes selective resampling of addresses:

```rust,ignore
// Traditional mutation: perturb random genes
genes[i] += noise;

// Trace mutation: resample selected addresses
let addresses_to_mutate = select_addresses(&trace, probability);
let mutated_trace = resample(trace, addresses_to_mutate);
```

### 2. Structured Crossover

Crossover can respect genome structure:

```rust,ignore
// Exchange subtrees of traces
let child_trace = merge_traces(
    parent1_trace,
    parent2_trace,
    &crossover_points,
);
```

### 3. Probabilistic Interpretation

Genetic operators have probabilistic semantics:

- **Selection** = Conditioning on high fitness
- **Mutation** = Partial resampling from prior
- **Crossover** = Trace merging

## The EvolutionaryGenome Trait

```rust,ignore
pub trait EvolutionaryGenome: Clone + Send + Sync {
    /// Convert genome to Fugue trace
    fn to_trace(&self) -> Trace;

    /// Reconstruct genome from trace
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;
}
```

### Example: RealVector Implementation

```rust,ignore
impl EvolutionaryGenome for RealVector {
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::new();
        for (i, &gene) in self.genes.iter().enumerate() {
            trace.insert(addr!("gene", i), gene);
        }
        trace
    }

    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let mut genes = Vec::new();
        let mut i = 0;
        while let Some(&value) = trace.get(&addr!("gene", i)) {
            genes.push(value);
            i += 1;
        }
        Ok(RealVector::new(genes))
    }
}
```

## Trace-Based Operators

### Trace Mutation

```rust,ignore
use fugue_evo::fugue_integration::trace_operators::TraceMutation;

let mutation = TraceMutation::new(mutation_probability);
let mutated_genome = mutation.mutate_via_trace(&genome, &bounds, &mut rng);
```

How it works:
1. Convert genome to trace
2. For each address, decide whether to resample
3. Resample selected addresses from prior (uniform within bounds)
4. Reconstruct genome from mutated trace

### Trace Crossover

```rust,ignore
use fugue_evo::fugue_integration::trace_operators::TraceCrossover;

let crossover = TraceCrossover::new(crossover_type);
let (child1, child2) = crossover.crossover_via_trace(&p1, &p2, &mut rng);
```

How it works:
1. Convert both parents to traces
2. For each address, select source parent
3. Merge into child traces
4. Reconstruct child genomes

## Effect Handlers

Fugue uses **effect handlers** (poutine-style) for program transformation. Fugue-evo provides evolution-specific handlers:

### Conditioning Handler

Interprets selection as conditioning:

```rust,ignore
use fugue_evo::fugue_integration::effect_handlers::ConditioningHandler;

// Selection biases toward high fitness
let handler = ConditioningHandler::new(fitness_function);
let selected_trace = handler.condition(trace, fitness_threshold);
```

### Resampling Handler

Implements mutation as partial resampling:

```rust,ignore
use fugue_evo::fugue_integration::effect_handlers::ResamplingHandler;

let handler = ResamplingHandler::new(resample_probability);
let mutated_trace = handler.resample(trace, &prior, &mut rng);
```

## Evolution Model

The full probabilistic evolution model:

```rust,ignore
use fugue_evo::fugue_integration::evolution_model::EvolutionModel;

let model = EvolutionModel::new()
    .with_prior(UniformPrior::new(&bounds))
    .with_likelihood(fitness_function)
    .with_mutation_kernel(GaussianKernel::new(sigma));

// One generation = one inference step
let posterior_population = model.step(prior_population, &mut rng);
```

## Advanced: Custom Trace Structures

For complex genomes, design meaningful trace structures:

```rust,ignore
// Neural network genome
impl EvolutionaryGenome for NeuralNetwork {
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::new();

        // Layer structure
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
                // Weight addresses
                for (weight_idx, &weight) in neuron.weights.iter().enumerate() {
                    trace.insert(
                        addr!("layer", layer_idx, "neuron", neuron_idx, "weight", weight_idx),
                        weight,
                    );
                }
                // Bias address
                trace.insert(
                    addr!("layer", layer_idx, "neuron", neuron_idx, "bias"),
                    neuron.bias,
                );
            }
        }

        trace
    }
}
```

This enables:
- Layer-aware mutation (mutate one layer at a time)
- Structural crossover (exchange layers between parents)
- Hierarchical analysis of evolved networks

## See Also

- [Custom Genome Types](../how-to/custom-genome.md)
- [Design Philosophy](./philosophy.md)
