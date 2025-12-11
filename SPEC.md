# Product Requirements Document: fugue-evo

**A Probabilistic Genetic Algorithm Library for Rust**

-----

**Version:** 0.2.0-draft  
**Author:** Alex Nodeland  
**Date:** December 2024  
**Status:** Draft (Validated)

-----

## Executive Summary

**fugue-evo** is a Rust library that implements genetic algorithms through the lens of probabilistic programming, built on top of the [Fugue PPL](https://github.com/alexnodeland/fugue). By treating evolution as Bayesian inference over solution spaces, fugue-evo provides principled approaches to selection, crossover, and mutation while enabling automatic hyperparameter learning through probabilistic inference.

The library supports customizable genome representations, hierarchical Bayesian genetic algorithms (HBGA), and learnable genetic operators—bridging the gap between evolutionary computation and modern probabilistic machine learning.

### Core Insight: Fitness as Likelihood

The foundational principle is that **selection pressure maps directly to Bayesian conditioning**. Selection probability proportional to `exp(f(x)/T)` (Boltzmann selection) is mathematically equivalent to importance sampling from a fitness-tempered posterior:

```
P(x | selection) ∝ P(x) · exp(f(x) / T)
```

This connection enables expressing EDAs as probabilistic models where Fugue’s existing inference machinery—MCMC, SMC, variational inference—becomes applicable to evolutionary search. Fugue’s `Model<T>` monad with `sample()`, `observe()`, and `addr!()` addressing provides the substrate for this integration.

-----

## Problem Statement

### Current Landscape

Traditional genetic algorithm libraries treat evolutionary operators as fixed, hand-tuned procedures. This creates several challenges:

1. **Hyperparameter Sensitivity**: Crossover rates, mutation rates, and selection pressure require extensive manual tuning per problem domain
1. **Lack of Uncertainty Quantification**: Standard GAs provide point estimates without confidence measures
1. **No Principled Adaptation**: Operator parameters don’t adapt based on fitness landscape structure
1. **Limited Composability**: Most GA libraries use imperative APIs that don’t compose well with other probabilistic systems

### Analysis of Existing Rust GA Libraries

|Library       |Pattern             |Strengths                   |Weaknesses                    |
|--------------|--------------------|----------------------------|------------------------------|
|**genevo**    |Separated traits    |Clean separation, extensible|Verbose, no PPL integration   |
|**oxigen**    |Monolithic genotype |Simple API, fast            |Inflexible, tight coupling    |
|**genetic-rs**|Reproduction-centric|Minimal boilerplate         |Limited operator customization|
|**rsgenetic** |Simulation-focused  |Good parallelism            |Dated API design              |

### Opportunity

Probabilistic programming provides a natural framework for evolutionary computation:

- **Selection as Conditioning**: High-fitness individuals are more probable under the posterior
- **Operators as Probabilistic Programs**: Crossover and mutation become learnable distributions
- **SMC as Evolution**: Sequential Monte Carlo maps directly onto generational dynamics
- **Hierarchical Modeling**: Population-level and individual-level parameters can be jointly inferred
- **Traces as Genomes**: Fugue’s trace infrastructure (address→value maps) naturally represents genetic material

-----

## Goals and Non-Goals

### Goals

1. **Probabilistic Foundation**: Implement GAs where fitness functions define likelihoods and selection emerges from Bayesian conditioning
1. **Learnable Operators**: Enable automatic inference of optimal crossover, mutation, and selection hyperparameters using all three adaptation paradigms (deterministic, adaptive, self-adaptive)
1. **Flexible Genomes**: Support arbitrary genome types through a trait-based abstraction with Fugue trace integration
1. **HBGA Support**: Implement hierarchical Bayesian genetic algorithms with population-level priors
1. **Production Quality**: Checkpointing, island model parallelism, convergence detection, multi-objective optimization
1. **Mathematical Rigor**: Exact implementations of CMA-ES, SBX, polynomial mutation with correct formulations
1. **Fugue Integration**: Leverage Fugue’s `Model<T>` monad, inference engines, and diagnostic tools

### Non-Goals

1. **GPU Acceleration**: Initial release targets CPU; GPU support is future work
1. **Distributed Evolution**: Multi-node parallelism is out of scope for v1.0
1. **Neuroevolution**: Neural architecture search requires specialized representations beyond initial scope
1. **Real-time Systems**: Hard real-time guarantees are not a design constraint

-----

## Core Concepts

### Genome Abstraction

The `EvolutionaryGenome` trait defines the interface for evolvable solution representations, with explicit Fugue trace integration:

```rust
use fugue::{Trace, Model, Address};
use serde::{Serialize, Deserialize};

/// Core genome abstraction with Fugue integration
pub trait EvolutionaryGenome: Clone + Send + Sync + Serialize + DeserializeOwned {
    /// The allele type for individual genes
    type Allele: Clone + Send;
    
    /// The phenotype or decoded solution type
    type Phenotype;
    
    /// Convert genome to Fugue trace for probabilistic operations
    fn to_trace(&self) -> Trace;
    
    /// Reconstruct genome from trace after evolutionary operations
    fn from_trace(trace: Trace) -> Result<Self, GenomeError>;
    
    /// Decode genome into phenotype for fitness evaluation
    fn decode(&self) -> Self::Phenotype;
    
    /// Compute dimensionality for adaptive operators
    fn dimension(&self) -> usize;
    
    /// Generate genome from Fugue model with importance weight
    fn generate<R: Rng>(model: &Model<Self>, rng: &mut R) -> (Self, f64);
}

/// Error type for genome operations
#[derive(Debug, thiserror::Error)]
pub enum GenomeError {
    #[error("Missing address in trace: {0}")]
    MissingAddress(Address),
    #[error("Type mismatch at address {0}: expected {1}, got {2}")]
    TypeMismatch(Address, String, String),
    #[error("Invalid genome structure: {0}")]
    InvalidStructure(String),
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}
```

### Built-in Genome Types

|Type                   |Description                 |Use Case                   |Operators                           |
|-----------------------|----------------------------|---------------------------|------------------------------------|
|`RealVector<N>`        |Fixed-length f64 vector     |Continuous optimization    |SBX, BLX-α, Gaussian mutation       |
|`DynamicRealVector`    |Variable-length f64 vector  |Variable-dimension problems|Segment crossover, length mutation  |
|`BitString<N>`         |Fixed-length boolean vector |Combinatorial optimization |Uniform crossover, bit-flip mutation|
|`Permutation<N>`       |Permutation of 0..N         |Ordering problems (TSP)    |PMX, OX, CX, swap/insert mutation   |
|`Tree<T>`              |S-expression tree           |Genetic programming        |Subtree crossover, point mutation   |
|`AdaptiveGenome<G>`    |Genome + strategy parameters|Self-adaptive ES           |Correlated mutation                 |
|`CompositeGenome<A, B>`|Product of two genomes      |Multi-representation       |Component-wise operators            |

### Fitness Abstraction

```rust
/// Fitness evaluation with probabilistic interpretation
pub trait Fitness {
    type Genome: EvolutionaryGenome;
    type Value: FitnessValue;
    
    /// Evaluate fitness (higher = better by convention)
    fn evaluate(&self, genome: &Self::Genome) -> Self::Value;
    
    /// Convert fitness to log-likelihood for probabilistic selection
    /// Default: Boltzmann distribution with temperature T
    fn as_log_likelihood(&self, genome: &Self::Genome, temperature: f64) -> f64 {
        let fitness = self.evaluate(genome).to_f64();
        fitness / temperature
    }
    
    /// Optional: Provide gradient for gradient-assisted mutation
    fn gradient(&self, _genome: &Self::Genome) -> Option<Vec<f64>> {
        None
    }
}

/// Trait bound for fitness values
pub trait FitnessValue: PartialOrd + Clone + Send + Debug {
    fn to_f64(&self) -> f64;
    fn is_better_than(&self, other: &Self) -> bool;
}

impl FitnessValue for f64 {
    fn to_f64(&self) -> f64 { *self }
    fn is_better_than(&self, other: &Self) -> bool { self > other }
}

/// Multi-objective fitness
#[derive(Clone, Debug)]
pub struct ParetoFitness {
    pub objectives: Vec<f64>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl FitnessValue for ParetoFitness {
    fn to_f64(&self) -> f64 {
        // Aggregated scalar for probabilistic interpretation
        -(self.rank as f64) + self.crowding_distance * 0.001
    }
    
    fn is_better_than(&self, other: &Self) -> bool {
        self.rank < other.rank || 
        (self.rank == other.rank && self.crowding_distance > other.crowding_distance)
    }
}
```

-----

## Mathematical Specifications

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

CMA-ES adapts a multivariate normal **N(m, σ²C)** through the following update equations:

#### Mean Update (Weighted Recombination)

```
m^(g+1) = Σᵢ₌₁^μ wᵢ · x_{i:λ}
```

where weights satisfy `w₁ ≥ ... ≥ wμ > 0` and `Σwᵢ = 1`, with effective selection mass:

```
μ_eff = 1 / Σwᵢ²
```

#### Evolution Path for Step-Size Control

```
p_σ^(g+1) = (1 - c_σ) · p_σ^(g) + √(c_σ(2-c_σ)μ_eff) · C^(-1/2) · (m^(g+1) - m^(g)) / σ
```

#### Cumulative Step-Size Adaptation (CSA)

```
σ^(g+1) = σ^(g) · exp((c_σ / d_σ) · (‖p_σ‖ / E[‖N(0,I)‖] - 1))
```

where `E[‖N(0,I)‖] ≈ √n · (1 - 1/(4n) + 1/(21n²))`.

#### Covariance Matrix Update (Rank-1 + Rank-μ)

```
C^(g+1) = (1 - c₁ - c_μ) · C^(g) + c₁ · p_c · p_cᵀ + c_μ · Σᵢ₌₁^μ wᵢ · y_{i:λ} · y_{i:λ}ᵀ
```

#### Default Parameters

|Parameter|Formula                                               |Description            |
|---------|------------------------------------------------------|-----------------------|
|λ        |`4 + ⌊3 ln n⌋`                                        |Population size        |
|μ        |`⌊λ/2⌋`                                               |Parent population size |
|c_σ      |`(μ_eff + 2) / (n + μ_eff + 5)`                       |Step-size learning rate|
|d_σ      |`1 + 2·max(0, √((μ_eff-1)/(n+1)) - 1) + c_σ`          |Step-size damping      |
|c_c      |`(4 + μ_eff/n) / (n + 4 + 2μ_eff/n)`                  |Evolution path decay   |
|c₁       |`2 / ((n+1.3)² + μ_eff)`                              |Rank-1 update weight   |
|c_μ      |`min(1-c₁, 2(μ_eff - 2 + 1/μ_eff) / ((n+2)² + μ_eff))`|Rank-μ update weight   |

```rust
/// CMA-ES state
#[derive(Clone, Serialize, Deserialize)]
pub struct CmaEsState {
    pub mean: DVector<f64>,
    pub sigma: f64,
    pub covariance: DMatrix<f64>,
    pub path_sigma: DVector<f64>,
    pub path_c: DVector<f64>,
    pub generation: usize,
    // Cached eigendecomposition for efficiency
    eigenvalues: DVector<f64>,
    eigenvectors: DMatrix<f64>,
    invsqrt_c: DMatrix<f64>,
}

impl CmaEsState {
    pub fn new(dimension: usize, initial_mean: DVector<f64>, initial_sigma: f64) -> Self {
        Self {
            mean: initial_mean,
            sigma: initial_sigma,
            covariance: DMatrix::identity(dimension, dimension),
            path_sigma: DVector::zeros(dimension),
            path_c: DVector::zeros(dimension),
            generation: 0,
            eigenvalues: DVector::from_element(dimension, 1.0),
            eigenvectors: DMatrix::identity(dimension, dimension),
            invsqrt_c: DMatrix::identity(dimension, dimension),
        }
    }
    
    pub fn sample_population<R: Rng>(&self, rng: &mut R, lambda: usize) -> Vec<DVector<f64>> {
        let n = self.mean.len();
        (0..lambda)
            .map(|_| {
                let z: DVector<f64> = DVector::from_fn(n, |_, _| rng.sample(StandardNormal));
                &self.mean + self.sigma * (&self.eigenvectors * self.eigenvalues.map(|e| e.sqrt()).component_mul(&z))
            })
            .collect()
    }
    
    pub fn update(&mut self, sorted_population: &[DVector<f64>], params: &CmaEsParams) {
        // Implementation of full CMA-ES update equations
        // ... (see detailed implementation in algorithms/cma_es.rs)
    }
}
```

### Simulated Binary Crossover (SBX)

SBX generates offspring from parents x₁, x₂ using spread factor β:

```
β = (2u)^(1/(η+1))           if u ≤ 0.5
β = (1/(2(1-u)))^(1/(η+1))   otherwise

c₁ = 0.5 · [(1+β)·x₁ + (1-β)·x₂]
c₂ = 0.5 · [(1-β)·x₁ + (1+β)·x₂]
```

The **distribution index η ∈ [2, 20]** controls exploitation/exploration:

- η = 20: Offspring stay near parents (exploitation)
- η = 2: Wider exploration

```rust
/// Simulated Binary Crossover operator
pub struct SbxCrossover {
    /// Distribution index (typically 2-20)
    pub eta: f64,
    /// Per-gene crossover probability
    pub crossover_probability: f64,
}

impl SbxCrossover {
    pub fn new(eta: f64) -> Self {
        Self { eta, crossover_probability: 0.9 }
    }
    
    /// Compute spread factor β from uniform random u
    fn spread_factor(&self, u: f64) -> f64 {
        if u <= 0.5 {
            (2.0 * u).powf(1.0 / (self.eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta + 1.0))
        }
    }
    
    /// Apply SBX to produce two offspring
    pub fn crossover<R: Rng>(
        &self,
        parent1: &[f64],
        parent2: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut R,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut child1 = parent1.to_vec();
        let mut child2 = parent2.to_vec();
        
        for i in 0..parent1.len() {
            if rng.gen::<f64>() < self.crossover_probability {
                let (x1, x2) = (parent1[i], parent2[i]);
                if (x1 - x2).abs() > 1e-14 {
                    let u = rng.gen::<f64>();
                    let beta = self.spread_factor(u);
                    
                    child1[i] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
                    child2[i] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2);
                    
                    // Clamp to bounds
                    let (lo, hi) = bounds[i];
                    child1[i] = child1[i].clamp(lo, hi);
                    child2[i] = child2[i].clamp(lo, hi);
                }
            }
        }
        
        (child1, child2)
    }
}
```

### Polynomial Mutation (Bounded)

For bounded domains, use NSGA-II’s formulation incorporating boundary distances:

```
δ₁ = (x - a) / (b - a)
δ₂ = (b - x) / (b - a)

δ_q = (2u + (1-2u)(1-δ₁)^(η_m+1))^(1/(η_m+1)) - 1      if u ≤ 0.5
δ_q = 1 - (2(1-u) + 2(u-0.5)(1-δ₂)^(η_m+1))^(1/(η_m+1)) otherwise

x' = x + δ_q · (b - a)
```

Default mutation probability: **1/n** (one gene on average).

```rust
/// Polynomial mutation operator
pub struct PolynomialMutation {
    /// Distribution index (typically 20-100)
    pub eta_m: f64,
    /// Per-gene mutation probability (default: 1/n)
    pub mutation_probability: Option<f64>,
}

impl PolynomialMutation {
    pub fn new(eta_m: f64) -> Self {
        Self { eta_m, mutation_probability: None }
    }
    
    pub fn mutate<R: Rng>(
        &self,
        genome: &mut [f64],
        bounds: &[(f64, f64)],
        rng: &mut R,
    ) {
        let prob = self.mutation_probability.unwrap_or(1.0 / genome.len() as f64);
        
        for i in 0..genome.len() {
            if rng.gen::<f64>() < prob {
                let (a, b) = bounds[i];
                let x = genome[i];
                let delta1 = (x - a) / (b - a);
                let delta2 = (b - x) / (b - a);
                
                let u = rng.gen::<f64>();
                let delta_q = if u <= 0.5 {
                    let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(self.eta_m + 1.0);
                    val.powf(1.0 / (self.eta_m + 1.0)) - 1.0
                } else {
                    let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(self.eta_m + 1.0);
                    1.0 - val.powf(1.0 / (self.eta_m + 1.0))
                };
                
                genome[i] = (x + delta_q * (b - a)).clamp(a, b);
            }
        }
    }
}
```

### Permutation Operators

#### Partially Mapped Crossover (PMX)

```rust
/// PMX crossover for permutation genomes
pub fn pmx_crossover<R: Rng>(
    parent1: &[usize],
    parent2: &[usize],
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let n = parent1.len();
    let (mut start, mut end) = (rng.gen_range(0..n), rng.gen_range(0..n));
    if start > end { std::mem::swap(&mut start, &mut end); }
    
    let mut child1 = vec![usize::MAX; n];
    let mut child2 = vec![usize::MAX; n];
    
    // Copy segment from opposite parent
    for i in start..=end {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    
    // Build mapping
    let mut map1: HashMap<usize, usize> = HashMap::new();
    let mut map2: HashMap<usize, usize> = HashMap::new();
    for i in start..=end {
        map1.insert(parent2[i], parent1[i]);
        map2.insert(parent1[i], parent2[i]);
    }
    
    // Fill remaining positions
    for i in (0..start).chain(end+1..n) {
        let mut val1 = parent1[i];
        while child1[start..=end].contains(&val1) {
            val1 = *map1.get(&val1).unwrap_or(&val1);
        }
        child1[i] = val1;
        
        let mut val2 = parent2[i];
        while child2[start..=end].contains(&val2) {
            val2 = *map2.get(&val2).unwrap_or(&val2);
        }
        child2[i] = val2;
    }
    
    (child1, child2)
}
```

#### Order Crossover (OX)

```rust
/// Order crossover for permutation genomes
pub fn order_crossover<R: Rng>(
    parent1: &[usize],
    parent2: &[usize],
    rng: &mut R,
) -> Vec<usize> {
    let n = parent1.len();
    let (mut start, mut end) = (rng.gen_range(0..n), rng.gen_range(0..n));
    if start > end { std::mem::swap(&mut start, &mut end); }
    
    let mut child = vec![usize::MAX; n];
    let segment: HashSet<_> = parent1[start..=end].iter().copied().collect();
    
    // Copy segment from parent1
    for i in start..=end {
        child[i] = parent1[i];
    }
    
    // Fill from parent2 in order, skipping segment elements
    let mut pos = (end + 1) % n;
    for &gene in parent2.iter().cycle().skip(end + 1).take(n) {
        if !segment.contains(&gene) {
            child[pos] = gene;
            pos = (pos + 1) % n;
            if pos == start { break; }
        }
    }
    
    child
}
```

-----

## Hyperparameter Adaptation Taxonomy

Following Eiben et al.’s classification, fugue-evo supports three adaptation paradigms:

### 1. Deterministic Control (Schedules)

```rust
/// Parameter schedule trait
pub trait ParameterSchedule: Clone + Send + Sync {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64;
}

/// Exponential decay: p(t) = p₀ · e^(-λt)
#[derive(Clone)]
pub struct ExponentialDecay {
    pub initial: f64,
    pub decay_rate: f64,
    pub minimum: f64,
}

impl ParameterSchedule for ExponentialDecay {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        (self.initial * (-self.decay_rate * generation as f64).exp()).max(self.minimum)
    }
}

/// Linear annealing: p(t) = p_start + (p_end - p_start) · t / T
#[derive(Clone)]
pub struct LinearAnnealing {
    pub start: f64,
    pub end: f64,
}

impl ParameterSchedule for LinearAnnealing {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64 {
        let t = generation as f64 / max_generations as f64;
        self.start + (self.end - self.start) * t
    }
}

/// Cosine annealing with warm restarts
#[derive(Clone)]
pub struct CosineAnnealing {
    pub max_value: f64,
    pub min_value: f64,
    pub period: usize,
}

impl ParameterSchedule for CosineAnnealing {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        let t = (generation % self.period) as f64 / self.period as f64;
        self.min_value + 0.5 * (self.max_value - self.min_value) * (1.0 + (std::f64::consts::PI * t).cos())
    }
}
```

### 2. Adaptive Control (Feedback-Based)

```rust
/// Rechenberg's 1/5 success rule for step-size adaptation
#[derive(Clone)]
pub struct OneFifthRule {
    pub increase_factor: f64,  // typically 1.22 ≈ e^(1/5)
    pub decrease_factor: f64,  // typically 0.82 ≈ e^(-1/5)
    pub window_size: usize,
    pub target_success_rate: f64,  // 0.2
    // Internal state
    success_history: VecDeque<bool>,
}

impl OneFifthRule {
    pub fn new() -> Self {
        Self {
            increase_factor: 1.22,
            decrease_factor: 0.82,
            window_size: 10,
            target_success_rate: 0.2,
            success_history: VecDeque::with_capacity(10),
        }
    }
    
    pub fn record(&mut self, success: bool) {
        self.success_history.push_back(success);
        if self.success_history.len() > self.window_size {
            self.success_history.pop_front();
        }
    }
    
    pub fn adapt(&self, sigma: f64) -> f64 {
        if self.success_history.len() < self.window_size {
            return sigma;
        }
        
        let success_rate = self.success_history.iter().filter(|&&s| s).count() as f64 
            / self.success_history.len() as f64;
        
        if success_rate > self.target_success_rate {
            sigma * self.increase_factor
        } else if success_rate < self.target_success_rate {
            sigma * self.decrease_factor
        } else {
            sigma
        }
    }
}

/// Fitness-based adaptive operator selection
#[derive(Clone)]
pub struct AdaptiveOperatorSelection<O: Clone> {
    pub operators: Vec<O>,
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub min_probability: f64,
}

impl<O: Clone> AdaptiveOperatorSelection<O> {
    pub fn select<R: Rng>(&self, rng: &mut R) -> (usize, &O) {
        let dist = WeightedIndex::new(&self.weights).unwrap();
        let idx = dist.sample(rng);
        (idx, &self.operators[idx])
    }
    
    pub fn update(&mut self, operator_idx: usize, fitness_improvement: f64) {
        // Credit assignment based on fitness improvement
        let reward = fitness_improvement.max(0.0);
        self.weights[operator_idx] += self.learning_rate * reward;
        
        // Normalize and enforce minimum probability
        let sum: f64 = self.weights.iter().sum();
        let n = self.weights.len() as f64;
        for w in &mut self.weights {
            *w = (*w / sum).max(self.min_probability / n);
        }
        let sum: f64 = self.weights.iter().sum();
        for w in &mut self.weights {
            *w /= sum;
        }
    }
}
```

### 3. Self-Adaptive Control (Encoded in Genome)

```rust
/// Genome with evolved strategy parameters
#[derive(Clone, Serialize, Deserialize)]
pub struct AdaptiveGenome<G: EvolutionaryGenome> {
    pub genome: G,
    /// Per-gene or global step sizes
    pub strategy_params: StrategyParams,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum StrategyParams {
    /// Single global step size (σ)
    Isotropic(f64),
    /// Per-gene step sizes (σ₁, ..., σₙ)
    NonIsotropic(Vec<f64>),
    /// Full covariance (σ₁, ..., σₙ, α₁, ..., αₖ) where k = n(n-1)/2
    Correlated { sigmas: Vec<f64>, rotations: Vec<f64> },
}

impl<G: EvolutionaryGenome<Allele = f64>> AdaptiveGenome<G> {
    /// Self-adaptive mutation with log-normal step size update
    /// τ = 1/√(2n), τ' = 1/√(2√n)
    pub fn self_adaptive_mutate<R: Rng>(&mut self, rng: &mut R) {
        let n = self.genome.dimension();
        let tau = 1.0 / (2.0 * n as f64).sqrt();
        let tau_prime = 1.0 / (2.0 * (n as f64).sqrt()).sqrt();
        
        let n0: f64 = rng.sample(StandardNormal);
        
        match &mut self.strategy_params {
            StrategyParams::Isotropic(sigma) => {
                *sigma *= (tau_prime * n0).exp();
                *sigma = sigma.max(1e-10);  // Lower bound
                
                // Mutate genome
                let trace = self.genome.to_trace();
                // ... apply mutation with updated sigma
            }
            StrategyParams::NonIsotropic(sigmas) => {
                for (i, sigma) in sigmas.iter_mut().enumerate() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(1e-10);
                }
                // ... apply per-gene mutation
            }
            StrategyParams::Correlated { sigmas, rotations } => {
                // Update sigmas
                for sigma in sigmas.iter_mut() {
                    let ni: f64 = rng.sample(StandardNormal);
                    *sigma *= (tau_prime * n0 + tau * ni).exp();
                    *sigma = sigma.max(1e-10);
                }
                // Update rotation angles
                let beta = 0.0873;  // ≈ 5°
                for alpha in rotations.iter_mut() {
                    *alpha += beta * rng.sample::<f64, _>(StandardNormal);
                }
                // ... apply correlated mutation
            }
        }
    }
}
```

### 4. Bayesian Hyperparameter Learning (via Fugue)

```rust
/// Online Bayesian learning of operator parameters
pub struct BayesianHyperparameterLearner {
    /// Sliding window of (params, fitness_improvement) observations
    history: VecDeque<(OperatorParams, f64)>,
    window_size: usize,
    /// Posterior approximations using conjugate priors
    posteriors: HyperparameterPosteriors,
}

#[derive(Clone)]
pub struct HyperparameterPosteriors {
    /// Mutation rate: Beta(α, β) posterior
    pub mutation_rate: BetaPosterior,
    /// Crossover probability: Beta posterior
    pub crossover_prob: BetaPosterior,
    /// Selection temperature: Gamma posterior
    pub temperature: GammaPosterior,
    /// SBX distribution index: Gamma posterior
    pub sbx_eta: GammaPosterior,
    /// Step sizes: LogNormal posteriors
    pub step_sizes: Vec<LogNormalPosterior>,
}

#[derive(Clone)]
pub struct BetaPosterior {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaPosterior {
    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }
    
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
    
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Beta::new(self.alpha, self.beta).unwrap().sample(rng)
    }
}

impl BayesianHyperparameterLearner {
    /// Update posteriors given observed fitness improvement
    pub fn observe(&mut self, params: OperatorParams, parent_fitness: f64, child_fitness: f64) {
        let improvement = child_fitness - parent_fitness;
        let success = improvement > 0.0;
        
        self.history.push_back((params.clone(), improvement));
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }
        
        // Update conjugate posteriors
        self.posteriors.mutation_rate.update(success);
        // ... update other posteriors based on which operators were used
    }
    
    /// Sample hyperparameters from current posterior as Fugue Model
    pub fn sample_params(&self) -> Model<OperatorParams> {
        prob! {
            let mutation_rate <- sample(
                addr!("hp", "mutation_rate"),
                Beta::new(self.posteriors.mutation_rate.alpha, 
                         self.posteriors.mutation_rate.beta).unwrap()
            );
            let crossover_prob <- sample(
                addr!("hp", "crossover_prob"),
                Beta::new(self.posteriors.crossover_prob.alpha,
                         self.posteriors.crossover_prob.beta).unwrap()
            );
            let temperature <- sample(
                addr!("hp", "temperature"),
                Gamma::new(self.posteriors.temperature.shape,
                          self.posteriors.temperature.rate).unwrap()
            );
            pure(OperatorParams { mutation_rate, crossover_prob, temperature, .. })
        }
    }
}
```

-----

## Architecture

### Module Structure

```
fugue-evo/
├── src/
│   ├── lib.rs                 # Public API re-exports
│   ├── prelude.rs             # Common imports
│   │
│   ├── genome/
│   │   ├── mod.rs
│   │   ├── traits.rs          # EvolutionaryGenome trait
│   │   ├── error.rs           # GenomeError
│   │   ├── real_vector.rs     # Continuous genomes
│   │   ├── bit_string.rs      # Binary genomes
│   │   ├── permutation.rs     # Ordering genomes
│   │   ├── tree.rs            # GP tree genomes
│   │   ├── adaptive.rs        # Self-adaptive genomes
│   │   └── composite.rs       # Product genomes
│   │
│   ├── fitness/
│   │   ├── mod.rs
│   │   ├── traits.rs          # Fitness, FitnessValue traits
│   │   ├── single.rs          # Single-objective
│   │   ├── multi.rs           # ParetoFitness, NSGA-II ranking
│   │   ├── benchmarks/
│   │   │   ├── continuous.rs  # Sphere, Rastrigin, Rosenbrock, etc.
│   │   │   ├── combinatorial.rs # OneMax, NK Landscape
│   │   │   └── dynamic.rs     # Moving peaks, changing fitness
│   │   └── caching.rs         # Memoized fitness evaluation
│   │
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── traits.rs          # GeneticOperator, CrossoverOp, MutationOp
│   │   ├── selection/
│   │   │   ├── tournament.rs
│   │   │   ├── roulette.rs
│   │   │   ├── truncation.rs
│   │   │   ├── boltzmann.rs   # Probabilistic selection
│   │   │   └── nsga2.rs       # Crowded tournament
│   │   ├── crossover/
│   │   │   ├── sbx.rs         # Simulated binary
│   │   │   ├── blx_alpha.rs   # Blend crossover
│   │   │   ├── uniform.rs     # Uniform crossover
│   │   │   ├── pmx.rs         # Partially mapped
│   │   │   ├── ox.rs          # Order crossover
│   │   │   ├── cx.rs          # Cycle crossover
│   │   │   └── subtree.rs     # GP subtree
│   │   └── mutation/
│   │       ├── gaussian.rs    # Gaussian perturbation
│   │       ├── polynomial.rs  # NSGA-II polynomial
│   │       ├── bit_flip.rs    # Bit string mutation
│   │       ├── swap.rs        # Permutation swap/insert/invert
│   │       ├── point.rs       # GP point mutation
│   │       └── self_adaptive.rs # Strategy parameter evolution
│   │
│   ├── algorithms/
│   │   ├── mod.rs
│   │   ├── simple_ga.rs       # Standard generational GA
│   │   ├── steady_state.rs    # Steady-state replacement
│   │   ├── mu_plus_lambda.rs  # (μ+λ) ES
│   │   ├── mu_comma_lambda.rs # (μ,λ) ES
│   │   ├── cma_es.rs          # Covariance Matrix Adaptation
│   │   ├── nsga2.rs           # NSGA-II multi-objective
│   │   ├── eda/
│   │   │   ├── umda.rs        # Univariate marginal
│   │   │   ├── pbil.rs        # Population-based incremental learning
│   │   │   └── boa.rs         # Bayesian optimization algorithm
│   │   ├── smc_evolution.rs   # SMC-based evolution
│   │   └── hbga.rs            # Hierarchical Bayesian GA
│   │
│   ├── hyperparameter/
│   │   ├── mod.rs
│   │   ├── schedules.rs       # Deterministic schedules
│   │   ├── adaptive.rs        # Feedback-based (1/5 rule, etc.)
│   │   ├── self_adaptive.rs   # Strategy parameter evolution
│   │   ├── bayesian.rs        # Online Bayesian learning
│   │   └── meta_learning.rs   # GP-based cross-problem learning
│   │
│   ├── population/
│   │   ├── mod.rs
│   │   ├── individual.rs      # Individual<G> wrapper
│   │   ├── population.rs      # Population<G> container
│   │   ├── island.rs          # Island model
│   │   ├── archive.rs         # Elite archive, Pareto front
│   │   └── niching.rs         # Fitness sharing, crowding
│   │
│   ├── termination/
│   │   ├── mod.rs
│   │   ├── generations.rs     # Max generations
│   │   ├── evaluations.rs     # Max fitness evaluations
│   │   ├── stagnation.rs      # Fitness stagnation detection
│   │   ├── diversity.rs       # Diversity threshold
│   │   ├── target.rs          # Target fitness reached
│   │   └── composite.rs       # And/Or/Any combinators
│   │
│   ├── checkpoint/
│   │   ├── mod.rs
│   │   ├── state.rs           # Checkpoint<G> struct
│   │   ├── serialization.rs   # Serde + versioning
│   │   └── recovery.rs        # Resume from checkpoint
│   │
│   ├── diagnostics/
│   │   ├── mod.rs
│   │   ├── statistics.rs      # GenerationStats
│   │   ├── diversity.rs       # Population diversity metrics
│   │   ├── convergence.rs     # Convergence detection
│   │   ├── operator_stats.rs  # Operator success tracking
│   │   └── trace_analysis.rs  # Fugue trace diagnostics
│   │
│   └── fugue_integration/
│       ├── mod.rs
│       ├── trace_genome.rs    # Trace ↔ Genome conversions
│       ├── effect_handlers.rs # Mutation/crossover as handlers
│       ├── models.rs          # Evolution as Fugue Model
│       └── inference.rs       # SMC/MCMC/VI for evolution
│
├── examples/
│   ├── sphere_optimization.rs
│   ├── rastrigin_benchmark.rs
│   ├── traveling_salesman.rs
│   ├── symbolic_regression.rs
│   ├── multi_objective_dtlz.rs
│   ├── cma_es_example.rs
│   ├── hyperparameter_learning.rs
│   ├── island_model.rs
│   └── checkpointing.rs
│
├── benches/
│   ├── genome_operations.rs
│   ├── fitness_evaluation.rs
│   ├── selection_algorithms.rs
│   └── full_evolution.rs
│
└── tests/
    ├── genome_tests.rs
    ├── operator_tests.rs
    ├── algorithm_tests.rs
    └── integration_tests.rs
```

### Dependency Graph

```
fugue-evo
    │
    ├── fugue-ppl (core probabilistic primitives)
    │   ├── Model<T> monad
    │   ├── Distributions
    │   ├── Inference engines (MCMC, SMC, VI)
    │   ├── Trace infrastructure
    │   └── Diagnostics (R-hat, ESS)
    │
    ├── nalgebra (linear algebra for CMA-ES)
    ├── rand / rand_distr (randomness)
    ├── rayon (parallelism)
    ├── serde / bincode (serialization)
    ├── thiserror (error handling)
    └── tracing (observability)
```

-----

## Trace-Based Genetic Operators

The key architectural insight is that **Fugue’s traces are genomes**. A trace contains a choice map (address→value pairs), log-probability, and return value—exactly what evolutionary algorithms need.

### Trace-Based Mutation

```rust
use fugue::{Trace, Model, Address, ChoiceMap};

/// Mutation via selective resampling of trace addresses
pub fn mutate_trace<G, R>(
    trace: &Trace,
    model: &Model<G>,
    mutation_selector: &MutationSelector,
    rng: &mut R,
) -> (Trace, f64)
where
    G: EvolutionaryGenome,
    R: Rng,
{
    // Determine which addresses to resample
    let mutation_sites = mutation_selector.select_sites(trace, rng);
    
    // Build constraints from non-mutated sites
    let mut constraints = ChoiceMap::new();
    for (addr, value) in trace.choices() {
        if !mutation_sites.contains(addr) {
            constraints.insert(addr.clone(), value.clone());
        }
    }
    
    // Generate new trace with constraints
    // Mutated sites get fresh samples from prior
    fugue::generate(model, &constraints, rng)
}

/// Selector for mutation sites
pub trait MutationSelector: Send + Sync {
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address>;
}

/// Uniform random mutation: each address mutates with probability p
pub struct UniformMutationSelector {
    pub mutation_probability: f64,
}

impl MutationSelector for UniformMutationSelector {
    fn select_sites<R: Rng>(&self, trace: &Trace, rng: &mut R) -> HashSet<Address> {
        trace.choices()
            .filter(|_| rng.gen::<f64>() < self.mutation_probability)
            .map(|(addr, _)| addr.clone())
            .collect()
    }
}

/// Gradient-guided mutation: prefer sites with high gradient magnitude
pub struct GradientGuidedSelector {
    pub base_probability: f64,
    pub gradient_weight: f64,
}
```

### Trace-Based Crossover

```rust
/// Crossover via constrained generation from merged parent traces
pub fn crossover_traces<G, R>(
    parent1: &Trace,
    parent2: &Trace,
    model: &Model<G>,
    crossover_mask: &CrossoverMask,
    rng: &mut R,
) -> (Trace, f64)
where
    G: EvolutionaryGenome,
    R: Rng,
{
    // Build merged constraints from both parents
    let mut constraints = ChoiceMap::new();
    
    let all_addresses: HashSet<_> = parent1.choices()
        .chain(parent2.choices())
        .map(|(addr, _)| addr.clone())
        .collect();
    
    for addr in all_addresses {
        let value = if crossover_mask.from_parent1(&addr) {
            parent1.get(&addr)
        } else {
            parent2.get(&addr)
        };
        
        if let Some(v) = value {
            constraints.insert(addr, v.clone());
        }
    }
    
    // Generate child trace consistent with constraints
    fugue::generate(model, &constraints, rng)
}

/// Determines which addresses come from which parent
pub trait CrossoverMask: Send + Sync {
    fn from_parent1(&self, addr: &Address) -> bool;
}

/// Uniform crossover: each address randomly from either parent
pub struct UniformCrossoverMask {
    pub bias: f64,  // Probability of choosing parent1
}

impl CrossoverMask for UniformCrossoverMask {
    fn from_parent1(&self, _addr: &Address) -> bool {
        rand::thread_rng().gen::<f64>() < self.bias
    }
}

/// Single-point crossover based on address ordering
pub struct SinglePointCrossoverMask {
    crossover_point: usize,
    address_order: Vec<Address>,
}

impl CrossoverMask for SinglePointCrossoverMask {
    fn from_parent1(&self, addr: &Address) -> bool {
        self.address_order.iter()
            .position(|a| a == addr)
            .map(|pos| pos < self.crossover_point)
            .unwrap_or(true)
    }
}
```

### Effect Handler Architecture

Following Pyro’s Poutine pattern:

```rust
/// Effect handler for evolutionary operations
pub trait EvolutionaryHandler {
    fn handle_sample(&mut self, addr: &Address, dist: &dyn Distribution) -> Value;
    fn handle_observe(&mut self, addr: &Address, dist: &dyn Distribution, obs: Value) -> f64;
}

/// Handler that replays a parent trace with selective resampling
pub struct MutationHandler<R: Rng> {
    parent_trace: Trace,
    mutation_sites: HashSet<Address>,
    rng: R,
    new_trace: Trace,
}

impl<R: Rng> EvolutionaryHandler for MutationHandler<R> {
    fn handle_sample(&mut self, addr: &Address, dist: &dyn Distribution) -> Value {
        if self.mutation_sites.contains(addr) {
            // Fresh sample (mutation)
            let value = dist.sample(&mut self.rng);
            self.new_trace.record(addr.clone(), value.clone(), dist.log_prob(&value));
            value
        } else {
            // Replay from parent (no mutation)
            let value = self.parent_trace.get(addr)
                .expect("Address not found in parent trace")
                .clone();
            self.new_trace.record(addr.clone(), value.clone(), dist.log_prob(&value));
            value
        }
    }
    
    fn handle_observe(&mut self, addr: &Address, dist: &dyn Distribution, obs: Value) -> f64 {
        let log_prob = dist.log_prob(&obs);
        self.new_trace.record_observe(addr.clone(), obs, log_prob);
        log_prob
    }
}

/// Handler that merges two parent traces for crossover
pub struct CrossoverHandler<R: Rng> {
    parent1: Trace,
    parent2: Trace,
    mask: Box<dyn CrossoverMask>,
    rng: R,
    child_trace: Trace,
}

impl<R: Rng> EvolutionaryHandler for CrossoverHandler<R> {
    fn handle_sample(&mut self, addr: &Address, dist: &dyn Distribution) -> Value {
        let value = if self.mask.from_parent1(addr) {
            self.parent1.get(addr).cloned()
        } else {
            self.parent2.get(addr).cloned()
        };
        
        let value = value.unwrap_or_else(|| dist.sample(&mut self.rng));
        self.child_trace.record(addr.clone(), value.clone(), dist.log_prob(&value));
        value
    }
    
    fn handle_observe(&mut self, addr: &Address, dist: &dyn Distribution, obs: Value) -> f64 {
        let log_prob = dist.log_prob(&obs);
        self.child_trace.record_observe(addr.clone(), obs, log_prob);
        log_prob
    }
}
```

-----

## Production Features

### Checkpointing

```rust
use serde::{Serialize, Deserialize};

/// Complete evolution state for checkpointing
#[derive(Serialize, Deserialize)]
pub struct Checkpoint<G: Serialize> {
    /// Schema version for forward compatibility
    pub version: u32,
    /// Current generation
    pub generation: usize,
    /// Total fitness evaluations
    pub evaluations: usize,
    /// Population with fitness values
    pub population: Vec<Individual<G>>,
    /// RNG state for reproducibility
    pub rng_state: SerializableRng,
    /// Best individual found
    pub best: Individual<G>,
    /// Algorithm-specific state
    pub algorithm_state: AlgorithmState,
    /// Hyperparameter learner state
    pub hyperparameter_state: Option<HyperparameterState>,
    /// Statistics history
    pub statistics: Vec<GenerationStats>,
}

/// Algorithm-specific state variants
#[derive(Serialize, Deserialize)]
pub enum AlgorithmState {
    SimpleGA,
    SteadyState { replacement_count: usize },
    CmaEs(CmaEsState),
    Nsga2 { pareto_front: Vec<usize> },
    Hbga { population_params: Vec<f64> },
}

impl<G: Serialize + DeserializeOwned> Checkpoint<G> {
    pub fn save(&self, path: &Path) -> Result<(), CheckpointError> {
        let file = File::create(path)?;
        let mut encoder = zstd::Encoder::new(file, 3)?;
        bincode::serialize_into(&mut encoder, self)?;
        encoder.finish()?;
        Ok(())
    }
    
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = File::open(path)?;
        let decoder = zstd::Decoder::new(file)?;
        let checkpoint: Self = bincode::deserialize_from(decoder)?;
        
        // Version compatibility check
        if checkpoint.version > CURRENT_VERSION {
            return Err(CheckpointError::VersionTooNew(checkpoint.version));
        }
        
        Ok(checkpoint)
    }
}

/// Automatic checkpointing wrapper
pub struct CheckpointingEvolution<A: Algorithm> {
    algorithm: A,
    checkpoint_interval: usize,
    checkpoint_path: PathBuf,
    keep_n_checkpoints: usize,
}

impl<A: Algorithm> CheckpointingEvolution<A> {
    pub fn run<R: Rng>(&mut self, rng: &mut R) -> EvolutionResult<A::Genome> {
        // Try to resume from checkpoint
        if let Ok(checkpoint) = Checkpoint::load(&self.checkpoint_path) {
            self.algorithm.restore_from_checkpoint(checkpoint);
        }
        
        loop {
            let result = self.algorithm.step(rng);
            
            if self.algorithm.generation() % self.checkpoint_interval == 0 {
                let checkpoint = self.algorithm.create_checkpoint(rng);
                checkpoint.save(&self.checkpoint_path)?;
                self.rotate_checkpoints();
            }
            
            if result.is_terminal() {
                return result;
            }
        }
    }
}
```

### Island Model Parallelism

```rust
use rayon::prelude::*;

/// Island model for parallel evolution with migration
pub struct IslandModel<G: EvolutionaryGenome, A: Algorithm<Genome = G>> {
    pub islands: Vec<Island<G, A>>,
    pub topology: MigrationTopology,
    pub migration_policy: MigrationPolicy,
    pub migration_interval: usize,
    pub migration_rate: f64,
}

pub struct Island<G: EvolutionaryGenome, A: Algorithm<Genome = G>> {
    pub algorithm: A,
    pub population: Population<G>,
    pub rng: SmallRng,
}

/// Migration topology determines which islands exchange individuals
#[derive(Clone)]
pub enum MigrationTopology {
    /// Ring: island i sends to island (i+1) mod n
    Ring,
    /// Toroidal grid
    Toroidal { rows: usize, cols: usize },
    /// Fully connected: every island can send to every other
    FullyConnected,
    /// Custom adjacency function
    Custom(Arc<dyn Fn(usize, usize) -> bool + Send + Sync>),
}

impl MigrationTopology {
    pub fn neighbors(&self, island_idx: usize, num_islands: usize) -> Vec<usize> {
        match self {
            Self::Ring => vec![(island_idx + 1) % num_islands],
            Self::Toroidal { rows, cols } => {
                let row = island_idx / cols;
                let col = island_idx % cols;
                vec![
                    ((row + rows - 1) % rows) * cols + col,  // up
                    ((row + 1) % rows) * cols + col,          // down
                    row * cols + (col + cols - 1) % cols,     // left
                    row * cols + (col + 1) % cols,            // right
                ]
            }
            Self::FullyConnected => (0..num_islands).filter(|&i| i != island_idx).collect(),
            Self::Custom(f) => (0..num_islands).filter(|&i| f(island_idx, i)).collect(),
        }
    }
}

/// Policy for selecting migrants
#[derive(Clone)]
pub enum MigrationPolicy {
    /// Send best individuals
    Best,
    /// Send random individuals
    Random,
    /// Tournament selection for migrants
    Tournament(usize),
}

impl<G, A> IslandModel<G, A>
where
    G: EvolutionaryGenome + Send + Sync,
    A: Algorithm<Genome = G> + Send + Sync,
{
    pub fn run(&mut self, generations: usize) -> EvolutionResult<G> {
        for gen in 0..generations {
            // Parallel evolution on each island
            self.islands.par_iter_mut().for_each(|island| {
                island.algorithm.step(&mut island.rng);
            });
            
            // Migration at intervals
            if gen > 0 && gen % self.migration_interval == 0 {
                self.migrate();
            }
        }
        
        // Return best across all islands
        self.best_result()
    }
    
    fn migrate(&mut self) {
        let num_islands = self.islands.len();
        let migration_count = (self.migration_rate * self.islands[0].population.len() as f64) as usize;
        
        // Collect emigrants from each island
        let emigrants: Vec<Vec<Individual<G>>> = self.islands.iter()
            .map(|island| {
                self.migration_policy.select(&island.population, migration_count, &mut rand::thread_rng())
            })
            .collect();
        
        // Distribute to neighbors
        for (src_idx, src_emigrants) in emigrants.into_iter().enumerate() {
            for dest_idx in self.topology.neighbors(src_idx, num_islands) {
                self.islands[dest_idx].population.integrate_migrants(src_emigrants.clone());
            }
        }
    }
}
```

### Convergence Detection

```rust
/// Termination criterion trait
pub trait TerminationCriterion<G: EvolutionaryGenome>: Send + Sync {
    fn should_terminate(&self, state: &EvolutionState<G>) -> bool;
    fn reason(&self) -> &'static str;
}

/// Evolution state for termination checking
pub struct EvolutionState<G> {
    pub generation: usize,
    pub evaluations: usize,
    pub best_fitness: f64,
    pub population: Population<G>,
    pub statistics_history: Vec<GenerationStats>,
}

/// Terminate after N generations
pub struct MaxGenerations(pub usize);

impl<G: EvolutionaryGenome> TerminationCriterion<G> for MaxGenerations {
    fn should_terminate(&self, state: &EvolutionState<G>) -> bool {
        state.generation >= self.0
    }
    fn reason(&self) -> &'static str { "Maximum generations reached" }
}

/// Terminate when fitness improvement stagnates
pub struct FitnessStagnation {
    pub window: usize,
    pub epsilon: f64,
}

impl<G: EvolutionaryGenome> TerminationCriterion<G> for FitnessStagnation {
    fn should_terminate(&self, state: &EvolutionState<G>) -> bool {
        if state.statistics_history.len() < self.window {
            return false;
        }
        
        let recent = &state.statistics_history[state.statistics_history.len() - self.window..];
        let improvement = recent.last().unwrap().best_fitness - recent.first().unwrap().best_fitness;
        
        improvement.abs() < self.epsilon
    }
    fn reason(&self) -> &'static str { "Fitness stagnation detected" }
}

/// Terminate when population diversity drops below threshold
pub struct DiversityThreshold {
    pub min_diversity: f64,
    pub metric: DiversityMetric,
}

#[derive(Clone)]
pub enum DiversityMetric {
    /// Average pairwise Euclidean distance
    AveragePairwiseDistance,
    /// Ratio of unique genomes
    UniqueRatio,
    /// Entropy of fitness distribution
    FitnessEntropy,
    /// Population spread (max - min for each dimension)
    PopulationSpread,
}

impl<G: EvolutionaryGenome> TerminationCriterion<G> for DiversityThreshold {
    fn should_terminate(&self, state: &EvolutionState<G>) -> bool {
        let diversity = self.metric.compute(&state.population);
        diversity < self.min_diversity
    }
    fn reason(&self) -> &'static str { "Diversity threshold reached" }
}

/// Combine multiple criteria
pub struct AnyOf<G>(pub Vec<Box<dyn TerminationCriterion<G>>>);
pub struct AllOf<G>(pub Vec<Box<dyn TerminationCriterion<G>>>);

impl<G: EvolutionaryGenome> TerminationCriterion<G> for AnyOf<G> {
    fn should_terminate(&self, state: &EvolutionState<G>) -> bool {
        self.0.iter().any(|c| c.should_terminate(state))
    }
    fn reason(&self) -> &'static str { "One of multiple criteria met" }
}
```

### Multi-Objective Optimization

```rust
/// NSGA-II non-dominated sorting
pub fn fast_non_dominated_sort<G>(
    population: &[(G, Vec<f64>)],
) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
    
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            
            if dominates(&population[i].1, &population[j].1) {
                dominated_solutions[i].push(j);
            } else if dominates(&population[j].1, &population[i].1) {
                domination_count[i] += 1;
            }
        }
        
        if domination_count[i] == 0 {
            fronts[0].push(i);
        }
    }
    
    let mut current_front = 0;
    while !fronts[current_front].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[current_front] {
            for &j in &dominated_solutions[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        current_front += 1;
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
    }
    
    fronts
}

/// Crowding distance assignment
pub fn crowding_distance<G>(
    population: &[(G, Vec<f64>)],
    front: &[usize],
) -> Vec<f64> {
    let num_objectives = population[0].1.len();
    let mut distances = vec![0.0; front.len()];
    
    for obj in 0..num_objectives {
        // Sort by objective
        let mut sorted: Vec<_> = front.iter().enumerate().collect();
        sorted.sort_by(|a, b| {
            population[*a.1].1[obj].partial_cmp(&population[*b.1].1[obj]).unwrap()
        });
        
        // Boundary points get infinite distance
        distances[sorted[0].0] = f64::INFINITY;
        distances[sorted[sorted.len() - 1].0] = f64::INFINITY;
        
        let obj_min = population[*sorted[0].1].1[obj];
        let obj_max = population[*sorted[sorted.len() - 1].1].1[obj];
        let range = obj_max - obj_min;
        
        if range > 0.0 {
            for i in 1..sorted.len() - 1 {
                let prev_obj = population[*sorted[i - 1].1].1[obj];
                let next_obj = population[*sorted[i + 1].1].1[obj];
                distances[sorted[i].0] += (next_obj - prev_obj) / range;
            }
        }
    }
    
    distances
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut dominated = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai < bi { return false; }
        if ai > bi { dominated = true; }
    }
    dominated
}

/// Hypervolume indicator
pub struct HypervolumeIndicator {
    pub reference_point: Vec<f64>,
}

impl HypervolumeIndicator {
    pub fn compute(&self, pareto_front: &[Vec<f64>]) -> f64 {
        // WFG algorithm for hypervolume computation
        // ... (see algorithms/nsga2.rs for full implementation)
        todo!()
    }
}
```

-----

## Diagnostics and Observability

### Generation Statistics

```rust
/// Statistics for a single generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub evaluations: usize,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub mean_fitness: f64,
    pub median_fitness: f64,
    pub fitness_std: f64,
    pub diversity: DiversityStats,
    pub operator_stats: OperatorStats,
    pub timing: TimingStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiversityStats {
    pub genotype_diversity: f64,
    pub phenotype_diversity: f64,
    pub fitness_entropy: f64,
    pub unique_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatorStats {
    pub crossover_applications: usize,
    pub crossover_improvements: usize,
    pub mutation_applications: usize,
    pub mutation_improvements: usize,
    pub selection_pressure: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimingStats {
    pub fitness_evaluation_ms: f64,
    pub selection_ms: f64,
    pub crossover_ms: f64,
    pub mutation_ms: f64,
    pub total_ms: f64,
}
```

### Fugue-Specific Diagnostics

```rust
/// ESS (Effective Sample Size) for evolutionary SMC
pub fn evolutionary_ess(particles: &[Particle]) -> f64 {
    let weights: Vec<f64> = particles.iter()
        .map(|p| p.log_weight.exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    
    if sum_sq == 0.0 {
        particles.len() as f64
    } else {
        (sum * sum) / sum_sq
    }
}

/// R-hat analog for evolutionary convergence
/// Compares fitness distributions across multiple runs
pub fn evolutionary_rhat(runs: &[Vec<f64>]) -> f64 {
    let m = runs.len() as f64;
    let n = runs[0].len() as f64;
    
    // Between-chain variance
    let chain_means: Vec<f64> = runs.iter()
        .map(|r| r.iter().sum::<f64>() / n)
        .collect();
    let grand_mean = chain_means.iter().sum::<f64>() / m;
    let b = n / (m - 1.0) * chain_means.iter()
        .map(|cm| (cm - grand_mean).powi(2))
        .sum::<f64>();
    
    // Within-chain variance
    let w: f64 = runs.iter()
        .map(|r| {
            let mean = r.iter().sum::<f64>() / n;
            r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        })
        .sum::<f64>() / m;
    
    // Pooled variance estimate
    let var_plus = ((n - 1.0) / n) * w + b / n;
    
    (var_plus / w).sqrt()
}

/// Trace divergence between parent and offspring
pub fn trace_divergence(parent: &Trace, offspring: &Trace) -> f64 {
    let parent_addresses: HashSet<_> = parent.choices().map(|(a, _)| a).collect();
    let offspring_addresses: HashSet<_> = offspring.choices().map(|(a, _)| a).collect();
    
    let common_addresses: Vec<_> = parent_addresses.intersection(&offspring_addresses).collect();
    
    let mut divergence = 0.0;
    for addr in &common_addresses {
        if let (Some(p_val), Some(o_val)) = (parent.get(addr), offspring.get(addr)) {
            if let (Some(p), Some(o)) = (p_val.as_f64(), o_val.as_f64()) {
                divergence += (p - o).abs();
            }
        }
    }
    
    divergence / common_addresses.len() as f64
}
```

-----

## Benchmark Suite

### Continuous Test Functions

|Function  |Formula                                             |Optimum        |Characteristics                 |
|----------|----------------------------------------------------|---------------|--------------------------------|
|Sphere    |`Σxᵢ²`                                              |0 at origin    |Unimodal, convex, separable     |
|Rastrigin |`10n + Σ(xᵢ² - 10cos(2πxᵢ))`                        |0 at origin    |Highly multimodal               |
|Rosenbrock|`Σ[100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²]`                       |0 at (1,…,1)   |Valley structure                |
|Ackley    |`-20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e`|0 at origin    |Nearly flat outer region        |
|Griewank  |`Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1`                       |0 at origin    |Many local minima               |
|Schwefel  |`418.9829n - Σxᵢsin(√|xᵢ|)`                         |0 at (420.97,…)|Deceptive, global far from local|
|Levy      |Complex                                             |0 at (1,…,1)   |Multimodal with valleys         |

### Combinatorial Benchmarks

|Problem     |Description              |Standard Instances            |
|------------|-------------------------|------------------------------|
|OneMax      |Count ones in bitstring  |n = 100, 500, 1000            |
|NK Landscape|Tunable epistasis        |K = 2, 4, 8                   |
|Royal Road  |Building block hypothesis|8×8, 16×16 schemas            |
|TSP         |Traveling salesman       |TSPLIB: eil51, kroA100, pr1002|
|MaxSat      |Maximum satisfiability   |SATLIB benchmarks             |

```rust
/// Benchmark function trait
pub trait BenchmarkFunction: Send + Sync {
    fn name(&self) -> &'static str;
    fn dimension(&self) -> usize;
    fn bounds(&self) -> (f64, f64);
    fn optimal_fitness(&self) -> f64;
    fn optimal_solution(&self) -> Option<Vec<f64>>;
    fn evaluate(&self, x: &[f64]) -> f64;
}

/// Sphere function
pub struct Sphere { dim: usize }

impl BenchmarkFunction for Sphere {
    fn name(&self) -> &'static str { "Sphere" }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> (f64, f64) { (-5.12, 5.12) }
    fn optimal_fitness(&self) -> f64 { 0.0 }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
    
    fn evaluate(&self, x: &[f64]) -> f64 {
        -x.iter().map(|xi| xi * xi).sum::<f64>()  // Negated for maximization
    }
}

/// Rastrigin function
pub struct Rastrigin { dim: usize }

impl BenchmarkFunction for Rastrigin {
    fn name(&self) -> &'static str { "Rastrigin" }
    fn dimension(&self) -> usize { self.dim }
    fn bounds(&self) -> (f64, f64) { (-5.12, 5.12) }
    fn optimal_fitness(&self) -> f64 { 0.0 }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
    
    fn evaluate(&self, x: &[f64]) -> f64 {
        let a = 10.0;
        let sum: f64 = x.iter()
            .map(|xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        -(a * self.dim as f64 + sum)  // Negated for maximization
    }
}
```

-----

## API Examples

### Basic Usage

```rust
use fugue_evo::prelude::*;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // Define fitness function
    let fitness = Rastrigin { dim: 10 };
    
    // Build and run GA with learned hyperparameters
    let result = SimpleGA::<RealVector<10>>::builder()
        .population_size(100)
        .max_generations(500)
        .crossover(SbxCrossover::new(20.0))
        .mutation(PolynomialMutation::new(20.0))
        .selection(TournamentSelection::new(3))
        .with_bayesian_hyperparameters()
        .termination(AnyOf(vec![
            Box::new(MaxGenerations(500)),
            Box::new(FitnessStagnation { window: 50, epsilon: 1e-8 }),
        ]))
        .build(fitness)?
        .run(&mut rng)?;
    
    println!("Best fitness: {:.6}", result.best_fitness);
    println!("Best solution: {:?}", result.best_genome.decode());
    println!("Generations: {}", result.generations);
    println!("Learned mutation rate: {:.4}", result.learned_params.mutation_rate);
    
    Ok(())
}
```

### CMA-ES

```rust
use fugue_evo::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    let result = CmaEs::builder()
        .dimension(20)
        .initial_mean(vec![0.0; 20])
        .initial_sigma(0.5)
        .termination(AnyOf(vec![
            Box::new(MaxGenerations(1000)),
            Box::new(FitnessStagnation { window: 100, epsilon: 1e-12 }),
        ]))
        .build(Rosenbrock { dim: 20 })?
        .run(&mut rng)?;
    
    println!("CMA-ES converged in {} generations", result.generations);
    println!("Final σ: {:.6}", result.final_sigma);
    println!("Best fitness: {:.6}", result.best_fitness);
    
    Ok(())
}
```

### HBGA with Fugue Integration

```rust
use fugue_evo::prelude::*;
use fugue::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let fitness = Sphere { dim: 5 };
    
    // Define evolution as a Fugue probabilistic model
    let evolution_model = hbga_model(
        &fitness,
        &SbxCrossover::new(20.0),
        &PolynomialMutation::new(20.0),
        100,   // population size
        50,    // generations
    );
    
    // Run SMC inference
    let smc_result = fugue::smc(
        &mut rng,
        || evolution_model.clone(),
        500,   // particles
        0.5,   // ESS threshold
    );
    
    // Analyze posterior over hyperparameters
    let sbx_etas: Vec<f64> = smc_result.particles.iter()
        .filter_map(|p| p.trace.get_f64(&addr!("hp", "sbx_eta")))
        .collect();
    
    println!("Posterior mean SBX η: {:.2}", mean(&sbx_etas));
    println!("Posterior std SBX η: {:.2}", std(&sbx_etas));
    
    // Extract best solutions
    let best = smc_result.particles.iter()
        .max_by(|a, b| a.log_weight.partial_cmp(&b.log_weight).unwrap())
        .unwrap();
    
    println!("Best log-weight: {:.4}", best.log_weight);
    
    Ok(())
}
```

### Island Model with Checkpointing

```rust
use fugue_evo::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fitness = Rastrigin { dim: 20 };
    
    let mut island_model = IslandModel::builder()
        .num_islands(8)
        .population_per_island(50)
        .topology(MigrationTopology::Toroidal { rows: 2, cols: 4 })
        .migration_interval(25)
        .migration_rate(0.1)
        .migration_policy(MigrationPolicy::Tournament(3))
        .algorithm(|_| {
            SimpleGA::<RealVector<20>>::builder()
                .crossover(SbxCrossover::new(15.0))
                .mutation(PolynomialMutation::new(20.0))
                .build_island()
        })
        .build(fitness)?;
    
    // Run with automatic checkpointing
    let result = CheckpointingEvolution::new(island_model)
        .checkpoint_interval(100)
        .checkpoint_path("evolution_checkpoint.bin")
        .keep_n_checkpoints(3)
        .run(1000)?;
    
    println!("Best across islands: {:.6}", result.best_fitness);
    
    Ok(())
}
```

-----

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)

1. Core `EvolutionaryGenome` trait with Fugue trace integration
1. Basic genome implementations (`RealVector`, `BitString`)
1. `Fitness` trait with likelihood conversion
1. Essential operators (SBX, polynomial mutation, tournament selection)
1. Simple GA algorithm
1. Basic diagnostics and statistics

**Deliverable:** Working GA that solves Sphere and Rastrigin benchmarks

### Phase 2: Mathematical Rigor (Weeks 5-8)

1. Complete CMA-ES with all adaptation equations
1. PMX, OX, CX for permutation genomes
1. Polynomial mutation with correct bounded formulation
1. NSGA-II multi-objective support
1. Comprehensive benchmark suite

**Deliverable:** Validated implementations matching published results

### Phase 3: Probabilistic Operators (Weeks 9-12)

1. Trace-based mutation and crossover
1. Effect handler architecture
1. Learnable operator parameter framework
1. Bayesian hyperparameter learning
1. Integration with Fugue’s `Model<T>`

**Deliverable:** HBGA working with Fugue SMC/MCMC

### Phase 4: Production Features (Weeks 13-16)

1. Checkpointing with version compatibility
1. Island model parallelism
1. Convergence detection suite
1. Tree genomes for genetic programming
1. Adaptive operator selection

**Deliverable:** Production-ready library with full feature set

### Phase 5: Polish (Weeks 17-20)

1. Documentation and examples
1. Performance optimization (SIMD, memory pooling)
1. API stabilization
1. Fuzzing and property-based testing
1. Release preparation

**Deliverable:** v1.0 release on crates.io

-----

## Success Metrics

1. **Correctness**: All benchmark problems solved to known optima (within 1% tolerance)
1. **Mathematical Accuracy**: CMA-ES convergence rates match Hansen’s reference implementation
1. **Hyperparameter Learning**: Learned parameters outperform defaults on 80%+ of problems
1. **Performance**: Within 2x of optimized C++ GA implementations on equivalent benchmarks
1. **Ergonomics**: API rated “easy to use” by 80%+ of beta testers
1. **Integration**: Seamless interop with Fugue inference engines (MCMC, SMC, VI)
1. **Reliability**: Zero panics in 1M random evolution runs (fuzz testing)

-----

## Error Handling Strategy

```rust
/// Top-level error type
#[derive(Debug, thiserror::Error)]
pub enum EvolutionError {
    #[error("Genome error: {0}")]
    Genome(#[from] GenomeError),
    
    #[error("Fitness evaluation failed: {0}")]
    FitnessEvaluation(String),
    
    #[error("Operator failed: {0}")]
    Operator(String),
    
    #[error("Invalid configuration: {0}")]
    Configuration(String),
    
    #[error("Checkpoint error: {0}")]
    Checkpoint(#[from] CheckpointError),
    
    #[error("Numerical instability: {0}")]
    Numerical(String),
    
    #[error("Fugue integration error: {0}")]
    Fugue(#[from] fugue::Error),
}

/// Operator result with optional repair
pub enum OperatorResult<G> {
    /// Successful operation
    Success(G),
    /// Operation failed, repaired genome returned
    Repaired(G, RepairInfo),
    /// Unrecoverable failure
    Failed(OperatorError),
}

/// Repair information for diagnostics
pub struct RepairInfo {
    pub constraint_violations: Vec<String>,
    pub repair_method: &'static str,
}
```

-----

## References

1. Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772
1. Deb, K., et al. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE TEC.
1. Eiben, A. E., et al. (1999). Parameter Control in Evolutionary Algorithms. IEEE TEC.
1. Pelikan, M., et al. (1999). BOA: The Bayesian Optimization Algorithm. GECCO.
1. Deb, K., & Agrawal, R. B. (1995). Simulated Binary Crossover for Continuous Search Space. Complex Systems.
1. Chopin, N., et al. (2020). An Introduction to Sequential Monte Carlo. Springer.
1. Goldberg, D. E., & Lingle, R. (1985). Alleles, Loci, and the Traveling Salesman Problem. ICGA.

-----

## Appendix: Numerical Stability

### Log-Sum-Exp for Fitness Aggregation

```rust
/// Numerically stable log-sum-exp
pub fn log_sum_exp(log_values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = log_values.collect();
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    
    let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
    max_val + sum.ln()
}

/// Normalize log-weights to probabilities
pub fn log_weights_to_probs(log_weights: &[f64]) -> Vec<f64> {
    let log_sum = log_sum_exp(log_weights.iter().cloned());
    log_weights.iter().map(|lw| (lw - log_sum).exp()).collect()
}
```

### Covariance Matrix Decomposition

```rust
/// Safe eigendecomposition with regularization
pub fn safe_eigen_decomposition(
    covariance: &DMatrix<f64>,
    min_eigenvalue: f64,
) -> (DVector<f64>, DMatrix<f64>) {
    let eigen = covariance.symmetric_eigen();
    
    // Regularize small/negative eigenvalues
    let eigenvalues = eigen.eigenvalues.map(|e| e.max(min_eigenvalue));
    
    (eigenvalues, eigen.eigenvectors)
}
```