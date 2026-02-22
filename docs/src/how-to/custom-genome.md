# Custom Genome Types

This guide shows how to create custom genome representations for domain-specific optimization problems.

## When to Create Custom Genomes

Create a custom genome when:

- Built-in types don't match your problem structure
- You need domain-specific constraints
- You want optimized operators for your representation
- Your solution has a unique structure

## Basic Pattern

Implement the `EvolutionaryGenome` trait:

```rust,ignore
use fugue_evo::prelude::*;
use fugue::Trace;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MyGenome {
    // Your genome data
}

impl EvolutionaryGenome for MyGenome {
    fn to_trace(&self) -> Trace {
        // Convert genome to Fugue trace
        let mut trace = Trace::new();
        // ... add values to trace
        trace
    }

    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        // Reconstruct genome from trace
        // ... extract values from trace
        Ok(MyGenome { /* ... */ })
    }
}
```

## Example: Schedule Genome

Let's create a genome for job scheduling problems:

```rust,ignore
use fugue_evo::prelude::*;
use fugue::{Trace, addr};
use serde::{Deserialize, Serialize};

/// A schedule assigns jobs to time slots on machines
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduleGenome {
    /// assignments[job] = (machine, start_time)
    pub assignments: Vec<(usize, f64)>,
    pub num_machines: usize,
}

impl ScheduleGenome {
    pub fn new(num_jobs: usize, num_machines: usize) -> Self {
        Self {
            assignments: vec![(0, 0.0); num_jobs],
            num_machines,
        }
    }

    pub fn random(num_jobs: usize, num_machines: usize, rng: &mut impl Rng) -> Self {
        let assignments = (0..num_jobs)
            .map(|_| {
                let machine = rng.gen_range(0..num_machines);
                let start = rng.gen_range(0.0..100.0);
                (machine, start)
            })
            .collect();

        Self { assignments, num_machines }
    }

    /// Get the makespan (total schedule length)
    pub fn makespan(&self) -> f64 {
        // Simplified: just max end time
        self.assignments.iter()
            .map(|(_, start)| start + 1.0) // Assume unit job duration
            .fold(0.0, f64::max)
    }
}

impl EvolutionaryGenome for ScheduleGenome {
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::new();

        for (i, (machine, start)) in self.assignments.iter().enumerate() {
            // Store machine assignment
            trace.insert(addr!("machine", i), *machine as f64);
            // Store start time
            trace.insert(addr!("start", i), *start);
        }

        // Store metadata
        trace.insert(addr!("num_machines"), self.num_machines as f64);

        trace
    }

    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        let num_machines = trace
            .get(&addr!("num_machines"))
            .ok_or(GenomeError::InvalidTrace("missing num_machines".into()))?
            .round() as usize;

        // Find number of jobs by counting machine entries
        let num_jobs = (0..)
            .take_while(|i| trace.get(&addr!("machine", *i)).is_some())
            .count();

        let assignments = (0..num_jobs)
            .map(|i| {
                let machine = trace
                    .get(&addr!("machine", i))
                    .unwrap()
                    .round() as usize;
                let start = *trace
                    .get(&addr!("start", i))
                    .unwrap();
                (machine, start)
            })
            .collect();

        Ok(Self { assignments, num_machines })
    }
}
```

## Custom Operators

Create operators tailored to your genome:

```rust,ignore
/// Mutation: randomly reassign one job
pub struct ScheduleMutation {
    pub probability: f64,
}

impl MutationOperator<ScheduleGenome> for ScheduleMutation {
    fn mutate(&self, genome: &mut ScheduleGenome, rng: &mut impl Rng) {
        if rng.gen::<f64>() < self.probability {
            // Pick random job
            let job = rng.gen_range(0..genome.assignments.len());

            // Reassign to random machine and time
            genome.assignments[job] = (
                rng.gen_range(0..genome.num_machines),
                rng.gen_range(0.0..100.0),
            );
        }
    }
}

/// Crossover: combine schedules by job subset
pub struct ScheduleCrossover;

impl CrossoverOperator<ScheduleGenome> for ScheduleCrossover {
    type Output = CrossoverResult<ScheduleGenome>;

    fn crossover(
        &self,
        parent1: &ScheduleGenome,
        parent2: &ScheduleGenome,
        rng: &mut impl Rng,
    ) -> Self::Output {
        let n = parent1.assignments.len();
        let crossover_point = rng.gen_range(0..n);

        // Child 1: first part from p1, second from p2
        let mut child1 = parent1.clone();
        for i in crossover_point..n {
            child1.assignments[i] = parent2.assignments[i];
        }

        // Child 2: first part from p2, second from p1
        let mut child2 = parent2.clone();
        for i in crossover_point..n {
            child2.assignments[i] = parent1.assignments[i];
        }

        CrossoverResult::new(child1, child2)
    }
}
```

## Validity Constraints

Enforce constraints in your genome:

```rust,ignore
impl ScheduleGenome {
    /// Ensure all assignments are valid
    pub fn repair(&mut self) {
        for (machine, start) in &mut self.assignments {
            // Clamp machine index
            *machine = (*machine).min(self.num_machines - 1);
            // Clamp start time
            *start = start.max(0.0);
        }
    }

    /// Check if schedule is valid
    pub fn is_valid(&self) -> bool {
        self.assignments.iter().all(|(m, s)| {
            *m < self.num_machines && *s >= 0.0
        })
    }
}

// Apply repair in mutation
impl MutationOperator<ScheduleGenome> for ScheduleMutation {
    fn mutate(&self, genome: &mut ScheduleGenome, rng: &mut impl Rng) {
        // ... mutation logic ...
        genome.repair(); // Ensure validity
    }
}
```

## Using Custom Genome with SimpleGA

```rust,ignore
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);

    let num_jobs = 20;
    let num_machines = 4;

    // Create initial population
    let initial_pop: Vec<ScheduleGenome> = (0..100)
        .map(|_| ScheduleGenome::random(num_jobs, num_machines, &mut rng))
        .collect();

    // Define fitness
    let fitness = ScheduleFitness::new(/* job durations, dependencies, etc. */);

    // Run GA with custom operators
    let result = SimpleGABuilder::<ScheduleGenome, f64, _, _, _, _, _>::new()
        .population_size(100)
        .initial_population(initial_pop)
        .selection(TournamentSelection::new(3))
        .crossover(ScheduleCrossover)
        .mutation(ScheduleMutation { probability: 0.1 })
        .fitness(fitness)
        .max_generations(200)
        .build()?
        .run(&mut rng)?;

    println!("Best makespan: {}", result.best_genome.makespan());
    Ok(())
}
```

## Composite Genomes

Combine multiple genome types:

```rust,ignore
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridGenome {
    pub continuous: RealVector,
    pub discrete: BitString,
}

impl EvolutionaryGenome for HybridGenome {
    fn to_trace(&self) -> Trace {
        let mut trace = Trace::new();

        // Namespace continuous genes
        for (i, val) in self.continuous.genes().iter().enumerate() {
            trace.insert(addr!("continuous", i), *val);
        }

        // Namespace discrete genes
        for (i, bit) in self.discrete.bits().iter().enumerate() {
            trace.insert(addr!("discrete", i), if *bit { 1.0 } else { 0.0 });
        }

        trace
    }

    fn from_trace(trace: &Trace) -> Result<Self, GenomeError> {
        // Extract continuous part
        let continuous_len = (0..)
            .take_while(|i| trace.get(&addr!("continuous", *i)).is_some())
            .count();

        let continuous_genes: Vec<f64> = (0..continuous_len)
            .map(|i| *trace.get(&addr!("continuous", i)).unwrap())
            .collect();

        // Extract discrete part
        let discrete_len = (0..)
            .take_while(|i| trace.get(&addr!("discrete", *i)).is_some())
            .count();

        let discrete_bits: Vec<bool> = (0..discrete_len)
            .map(|i| *trace.get(&addr!("discrete", i)).unwrap() > 0.5)
            .collect();

        Ok(Self {
            continuous: RealVector::new(continuous_genes),
            discrete: BitString::new(discrete_bits),
        })
    }
}
```

## Testing Custom Genomes

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_roundtrip() {
        let original = ScheduleGenome::random(10, 4, &mut StdRng::seed_from_u64(42));
        let trace = original.to_trace();
        let reconstructed = ScheduleGenome::from_trace(&trace).unwrap();

        assert_eq!(original.assignments, reconstructed.assignments);
        assert_eq!(original.num_machines, reconstructed.num_machines);
    }

    #[test]
    fn test_mutation_validity() {
        let mut genome = ScheduleGenome::random(10, 4, &mut StdRng::seed_from_u64(42));
        let mutation = ScheduleMutation { probability: 1.0 };

        for _ in 0..100 {
            mutation.mutate(&mut genome, &mut StdRng::seed_from_u64(42));
            assert!(genome.is_valid());
        }
    }
}
```

## Next Steps

- [Custom Operators](./custom-operators.md) - Create optimized operators for your genome
- [Custom Fitness Functions](./custom-fitness.md) - Evaluate your custom genomes
