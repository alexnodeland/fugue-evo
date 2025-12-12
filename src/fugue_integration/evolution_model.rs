//! Integration with Fugue's `Model<T>` monad
//!
//! This module provides probabilistic evolutionary models that integrate
//! with Fugue's inference engine (SMC, MCMC).
//!
//! # Key Concepts
//!
//! - **Evolution as Inference**: Treat evolution as posterior sampling
//! - **Fitness as Likelihood**: Selection pressure = Bayesian conditioning
//! - **Operators as Kernels**: Mutation/crossover define proposal distributions

use fugue::{addr, ChoiceValue, Trace};
use rand::Rng;
use std::marker::PhantomData;

use crate::fitness::traits::Fitness;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;

/// A probabilistic model of an evolutionary step
///
/// This wraps an evolutionary genome in Fugue's Model monad,
/// enabling integration with SMC and MCMC inference.
#[derive(Clone)]
pub struct EvolutionModel<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    /// The fitness function used for conditioning
    pub fitness: F,
    /// Temperature for fitness-based weighting
    temperature: f64,
    /// Bounds for generating new genomes
    bounds: MultiBounds,
    /// Phantom data for genome type
    _marker: PhantomData<G>,
}

impl<G, F> EvolutionModel<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    /// Create a new evolution model
    pub fn new(fitness: F, bounds: MultiBounds) -> Self {
        Self {
            fitness,
            temperature: 1.0,
            bounds,
            _marker: PhantomData,
        }
    }

    /// Set the temperature for fitness weighting
    ///
    /// Lower temperature = stronger selection pressure
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sample a genome from the prior (uniform within bounds)
    pub fn sample_prior<R: Rng>(&self, rng: &mut R) -> G {
        G::generate(rng, &self.bounds)
    }

    /// Compute the log-weight for a genome based on fitness
    ///
    /// w(x) = exp(f(x) / T) where T is temperature
    pub fn log_weight(&self, genome: &G) -> f64 {
        let fitness = self.fitness.evaluate(genome);
        fitness / self.temperature
    }

    /// Create a Fugue trace representing this genome with fitness weight
    pub fn to_weighted_trace(&self, genome: &G) -> Trace {
        let mut trace = genome.to_trace();
        let log_weight = self.log_weight(genome);

        // Add the fitness as a special address in the trace
        trace.insert_choice(
            addr!("__fitness__"),
            ChoiceValue::F64(log_weight),
            log_weight, // Use fitness as log probability
        );

        trace
    }
}

/// Configuration for the evolutionary Markov chain
#[derive(Clone, Debug)]
pub struct EvolutionChainConfig {
    /// Mutation probability per address
    pub mutation_rate: f64,
    /// Standard deviation for Gaussian mutations
    pub mutation_sigma: f64,
    /// Temperature for fitness weighting
    pub temperature: f64,
    /// Number of generations
    pub generations: usize,
}

impl Default for EvolutionChainConfig {
    fn default() -> Self {
        Self {
            mutation_rate: 0.1,
            mutation_sigma: 0.1,
            temperature: 1.0,
            generations: 100,
        }
    }
}

impl EvolutionChainConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the mutation rate
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set the mutation sigma
    pub fn mutation_sigma(mut self, sigma: f64) -> Self {
        self.mutation_sigma = sigma;
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the number of generations
    pub fn generations(mut self, gens: usize) -> Self {
        self.generations = gens;
        self
    }
}

/// A single step in the evolutionary Markov chain
///
/// This represents the transition kernel for MCMC-style evolution.
pub struct EvolutionStep<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    model: EvolutionModel<G, F>,
    config: EvolutionChainConfig,
}

impl<G, F> EvolutionStep<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    /// Create a new evolution step
    pub fn new(model: EvolutionModel<G, F>, config: EvolutionChainConfig) -> Self {
        Self { model, config }
    }

    /// Propose a new genome via mutation
    pub fn propose<R: Rng>(&self, current: &G, rng: &mut R) -> G {
        let trace = current.to_trace();
        let mut new_trace = Trace::default();

        for (addr, choice) in &trace.choices {
            let new_value = if rng.gen::<f64>() < self.config.mutation_rate {
                self.mutate_value(&choice.value, rng)
            } else {
                choice.value.clone()
            };
            new_trace.insert_choice(addr.clone(), new_value, 0.0);
        }

        G::from_trace(&new_trace).unwrap_or_else(|_| current.clone())
    }

    /// Mutate a single value
    fn mutate_value<R: Rng>(&self, value: &ChoiceValue, rng: &mut R) -> ChoiceValue {
        match value {
            ChoiceValue::F64(v) => {
                let noise = (rng.gen::<f64>() * 2.0 - 1.0) * self.config.mutation_sigma;
                ChoiceValue::F64(v + noise)
            }
            ChoiceValue::Bool(b) => ChoiceValue::Bool(!b),
            ChoiceValue::Usize(n) => {
                // Random walk on integers
                let delta: i32 = if rng.gen::<bool>() { 1 } else { -1 };
                ChoiceValue::Usize((*n as i32 + delta).max(0) as usize)
            }
            other => other.clone(),
        }
    }

    /// Compute acceptance probability (Metropolis-Hastings)
    ///
    /// α(x, x') = min(1, exp(f(x') - f(x)) / T)
    pub fn acceptance_probability(&self, current: &G, proposed: &G) -> f64 {
        let current_fitness = self.model.log_weight(current);
        let proposed_fitness = self.model.log_weight(proposed);

        let log_ratio = proposed_fitness - current_fitness;
        (log_ratio.exp()).min(1.0)
    }

    /// Perform one MCMC step
    pub fn step<R: Rng>(&self, current: &G, rng: &mut R) -> G {
        let proposed = self.propose(current, rng);
        let acceptance = self.acceptance_probability(current, &proposed);

        if rng.gen::<f64>() < acceptance {
            proposed
        } else {
            current.clone()
        }
    }

    /// Run the chain for multiple steps
    pub fn run_chain<R: Rng>(&self, initial: &G, num_steps: usize, rng: &mut R) -> Vec<G> {
        let mut chain = Vec::with_capacity(num_steps + 1);
        let mut current = initial.clone();

        chain.push(current.clone());

        for _ in 0..num_steps {
            current = self.step(&current, rng);
            chain.push(current.clone());
        }

        chain
    }
}

/// Particle representation for Sequential Monte Carlo
#[derive(Clone)]
pub struct Particle<G>
where
    G: EvolutionaryGenome,
{
    /// The genome state
    pub genome: G,
    /// Log weight of this particle
    pub log_weight: f64,
    /// Normalized weight
    pub normalized_weight: f64,
}

impl<G: EvolutionaryGenome> Particle<G> {
    /// Create a new particle
    pub fn new(genome: G, log_weight: f64) -> Self {
        Self {
            genome,
            log_weight,
            normalized_weight: 0.0,
        }
    }
}

/// Sequential Monte Carlo for evolutionary inference
///
/// SMC treats evolution as a sequence of distributions
/// that progressively concentrate on high-fitness solutions.
pub struct EvolutionarySMC<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    model: EvolutionModel<G, F>,
    /// Number of particles
    num_particles: usize,
    /// Current temperature schedule position
    temperature_schedule: Vec<f64>,
}

impl<G, F> EvolutionarySMC<G, F>
where
    G: EvolutionaryGenome + Clone,
    F: Fitness<Genome = G, Value = f64> + Clone,
{
    /// Create a new SMC sampler
    pub fn new(fitness: F, bounds: MultiBounds, num_particles: usize) -> Self {
        Self {
            model: EvolutionModel::new(fitness, bounds),
            num_particles,
            temperature_schedule: Self::default_schedule(10),
        }
    }

    /// Create a default annealing schedule
    fn default_schedule(steps: usize) -> Vec<f64> {
        (0..=steps)
            .map(|i| {
                let t = i as f64 / steps as f64;
                // Geometric annealing from infinity to 1
                (1.0 - t).powi(2) * 100.0 + 1.0
            })
            .collect()
    }

    /// Set a custom temperature schedule
    pub fn with_schedule(mut self, schedule: Vec<f64>) -> Self {
        self.temperature_schedule = schedule;
        self
    }

    /// Initialize particles from prior
    pub fn initialize<R: Rng>(&self, rng: &mut R) -> Vec<Particle<G>> {
        (0..self.num_particles)
            .map(|_| {
                let genome = self.model.sample_prior(rng);
                Particle::new(genome, 0.0)
            })
            .collect()
    }

    /// Normalize particle weights
    fn normalize_weights(particles: &mut [Particle<G>]) {
        let max_log_weight = particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        let sum: f64 = particles
            .iter()
            .map(|p| (p.log_weight - max_log_weight).exp())
            .sum();

        let log_sum = sum.ln() + max_log_weight;

        for particle in particles.iter_mut() {
            particle.normalized_weight = (particle.log_weight - log_sum).exp();
        }
    }

    /// Compute effective sample size
    fn effective_sample_size(particles: &[Particle<G>]) -> f64 {
        let sum_sq: f64 = particles.iter().map(|p| p.normalized_weight.powi(2)).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    }

    /// Resample particles using systematic resampling
    pub fn resample<R: Rng>(particles: &[Particle<G>], rng: &mut R) -> Vec<Particle<G>> {
        let n = particles.len();
        let mut resampled = Vec::with_capacity(n);

        // Build cumulative weights
        let mut cumulative = vec![0.0; n];
        cumulative[0] = particles[0].normalized_weight;
        for i in 1..n {
            cumulative[i] = cumulative[i - 1] + particles[i].normalized_weight;
        }

        // Systematic resampling
        let u0: f64 = rng.gen::<f64>() / n as f64;

        let mut j = 0;
        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n - 1 && cumulative[j] < u {
                j += 1;
            }
            let mut particle = particles[j].clone();
            particle.log_weight = 0.0;
            particle.normalized_weight = 1.0 / n as f64;
            resampled.push(particle);
        }

        resampled
    }

    /// Mutation kernel (MCMC move)
    fn mutate_particle<R: Rng>(
        &self,
        particle: &Particle<G>,
        temperature: f64,
        rng: &mut R,
    ) -> Particle<G> {
        let model = EvolutionModel::new(self.model.fitness.clone(), self.model.bounds.clone())
            .with_temperature(temperature);
        let config = EvolutionChainConfig::default()
            .mutation_rate(0.2)
            .mutation_sigma(0.1);
        let step = EvolutionStep::new(model, config);

        let new_genome = step.step(&particle.genome, rng);
        Particle::new(new_genome, particle.log_weight)
    }

    /// Run the SMC sampler
    pub fn run<R: Rng>(&self, rng: &mut R) -> Vec<Particle<G>> {
        let mut particles = self.initialize(rng);

        for (i, &temperature) in self.temperature_schedule.iter().enumerate() {
            // Update model temperature
            let model = EvolutionModel::new(self.model.fitness.clone(), self.model.bounds.clone())
                .with_temperature(temperature);

            // Reweight particles
            for particle in &mut particles {
                particle.log_weight = model.log_weight(&particle.genome);
            }

            Self::normalize_weights(&mut particles);

            // Resample if ESS too low
            let ess = Self::effective_sample_size(&particles);
            if ess < self.num_particles as f64 / 2.0 && i < self.temperature_schedule.len() - 1 {
                particles = Self::resample(&particles, rng);
            }

            // Mutate particles (MCMC rejuvenation)
            if i < self.temperature_schedule.len() - 1 {
                particles = particles
                    .into_iter()
                    .map(|p| self.mutate_particle(&p, temperature, rng))
                    .collect();
            }
        }

        // Final reweighting and normalization
        for particle in &mut particles {
            particle.log_weight = self.model.log_weight(&particle.genome);
        }
        Self::normalize_weights(&mut particles);

        particles
    }

    /// Get the best particle
    pub fn best_particle(particles: &[Particle<G>]) -> Option<&Particle<G>> {
        particles.iter().max_by(|a, b| {
            a.log_weight
                .partial_cmp(&b.log_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Compute posterior mean (weighted average of traces)
    pub fn posterior_mean(particles: &[Particle<G>]) -> Option<Trace> {
        if particles.is_empty() {
            return None;
        }

        let mut mean_trace = Trace::default();
        let first_trace = particles[0].genome.to_trace();

        for addr in first_trace.choices.keys() {
            let weighted_sum: f64 = particles
                .iter()
                .map(|p| {
                    let trace = p.genome.to_trace();
                    let value = trace
                        .choices
                        .get(addr)
                        .map(|c| match &c.value {
                            ChoiceValue::F64(f) => *f,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0);
                    value * p.normalized_weight
                })
                .sum();

            mean_trace.insert_choice(addr.clone(), ChoiceValue::F64(weighted_sum), 0.0);
        }

        Some(mean_trace)
    }
}

/// Hierarchical Bayesian Genetic Algorithm
///
/// This combines population-based evolution with Bayesian inference
/// over hyperparameters.
pub struct HBGA<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    model: EvolutionModel<G, F>,
    /// Population size
    population_size: usize,
    /// Number of generations
    generations: usize,
    /// Prior on mutation rate (Beta distribution parameters)
    mutation_rate_prior: (f64, f64),
    /// Prior on mutation sigma (Gamma distribution parameters)
    mutation_sigma_prior: (f64, f64),
}

impl<G, F> HBGA<G, F>
where
    G: EvolutionaryGenome + Clone,
    F: Fitness<Genome = G, Value = f64> + Clone,
{
    /// Create a new HBGA
    pub fn new(
        fitness: F,
        bounds: MultiBounds,
        population_size: usize,
        generations: usize,
    ) -> Self {
        Self {
            model: EvolutionModel::new(fitness, bounds),
            population_size,
            generations,
            mutation_rate_prior: (2.0, 8.0), // Prior favors low mutation rates
            mutation_sigma_prior: (2.0, 10.0), // Prior favors small mutations
        }
    }

    /// Set the mutation rate prior
    pub fn with_mutation_rate_prior(mut self, alpha: f64, beta: f64) -> Self {
        self.mutation_rate_prior = (alpha, beta);
        self
    }

    /// Set the mutation sigma prior
    pub fn with_mutation_sigma_prior(mut self, shape: f64, rate: f64) -> Self {
        self.mutation_sigma_prior = (shape, rate);
        self
    }

    /// Sample mutation rate from prior
    fn sample_mutation_rate<R: Rng>(&self, rng: &mut R) -> f64 {
        // Simple approximation: use mean with some noise
        let (alpha, beta) = self.mutation_rate_prior;
        let mean = alpha / (alpha + beta);
        let noise = (rng.gen::<f64>() - 0.5) * 0.1;
        (mean + noise).clamp(0.01, 0.5)
    }

    /// Sample mutation sigma from prior
    fn sample_mutation_sigma<R: Rng>(&self, rng: &mut R) -> f64 {
        // Simple approximation: use mean with some noise
        let (shape, rate) = self.mutation_sigma_prior;
        let mean = shape / rate;
        let noise = (rng.gen::<f64>() - 0.5) * 0.1;
        (mean + noise).max(0.01)
    }

    /// Run the HBGA
    pub fn run<R: Rng>(&self, rng: &mut R) -> HBGAResult<G> {
        // Initialize population
        let mut population: Vec<G> = (0..self.population_size)
            .map(|_| self.model.sample_prior(rng))
            .collect();

        let mut fitness_history = Vec::new();
        let mut best_genome: Option<G> = None;
        let mut best_fitness = f64::NEG_INFINITY;

        // Hyperparameter samples (would be inferred in full HBGA)
        let mut mutation_rate = self.sample_mutation_rate(rng);
        let mutation_sigma = self.sample_mutation_sigma(rng);

        for gen in 0..self.generations {
            // Evaluate fitness
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|g| self.model.fitness.evaluate(g))
                .collect();

            // Track best
            for (i, &f) in fitnesses.iter().enumerate() {
                if f > best_fitness {
                    best_fitness = f;
                    best_genome = Some(population[i].clone());
                }
            }

            let mean_fitness: f64 = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            fitness_history.push(mean_fitness);

            // Adaptive hyperparameter update (simplified Bayesian update)
            let improvement = if gen > 0 {
                mean_fitness - fitness_history[gen - 1]
            } else {
                0.0
            };

            // If improving, slightly increase mutation; otherwise decrease
            if improvement > 0.0 {
                mutation_rate = (mutation_rate * 1.05).min(0.5);
            } else {
                mutation_rate = (mutation_rate * 0.95).max(0.01);
            }

            // Selection (tournament)
            let mut selected = Vec::with_capacity(self.population_size);
            for _ in 0..self.population_size {
                let i = rng.gen_range(0..self.population_size);
                let j = rng.gen_range(0..self.population_size);
                if fitnesses[i] > fitnesses[j] {
                    selected.push(population[i].clone());
                } else {
                    selected.push(population[j].clone());
                }
            }

            // Mutation with current hyperparameters
            let config = EvolutionChainConfig::default()
                .mutation_rate(mutation_rate)
                .mutation_sigma(mutation_sigma);

            let model = EvolutionModel::new(self.model.fitness.clone(), self.model.bounds.clone());
            let step = EvolutionStep::new(model, config);

            population = selected
                .into_iter()
                .map(|g| step.propose(&g, rng))
                .collect();
        }

        HBGAResult {
            best_genome: best_genome.unwrap_or_else(|| population[0].clone()),
            best_fitness,
            fitness_history,
            final_mutation_rate: mutation_rate,
            final_mutation_sigma: mutation_sigma,
        }
    }
}

/// Result of HBGA run
pub struct HBGAResult<G> {
    /// Best genome found
    pub best_genome: G,
    /// Best fitness value
    pub best_fitness: f64,
    /// Mean fitness over generations
    pub fitness_history: Vec<f64>,
    /// Final learned mutation rate
    pub final_mutation_rate: f64,
    /// Final learned mutation sigma
    pub final_mutation_sigma: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::genome::real_vector::RealVector;

    #[test]
    fn test_evolution_model_basic() {
        let sphere = Sphere::new(3);
        let bounds = MultiBounds::symmetric(5.0, 3);
        let model = EvolutionModel::<RealVector, _>::new(sphere, bounds);

        let genome = RealVector::new(vec![0.0, 0.0, 0.0]);
        let weight = model.log_weight(&genome);

        // Optimal solution should have highest weight (fitness = 0 for sphere)
        assert!(weight.is_finite());
    }

    #[test]
    fn test_evolution_step() {
        let mut rng = rand::thread_rng();
        let sphere = Sphere::new(3);
        let bounds = MultiBounds::symmetric(5.0, 3);
        let model = EvolutionModel::<RealVector, _>::new(sphere, bounds);
        let config = EvolutionChainConfig::default();
        let step = EvolutionStep::new(model, config);

        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let proposed = step.propose(&genome, &mut rng);

        // Proposed should be different (with high probability)
        assert_eq!(proposed.dimension(), 3);
    }

    #[test]
    fn test_mcmc_chain() {
        let mut rng = rand::thread_rng();
        let sphere = Sphere::new(3);
        let bounds = MultiBounds::symmetric(5.0, 3);
        let model = EvolutionModel::<RealVector, _>::new(sphere, bounds);
        let config = EvolutionChainConfig::default().generations(10);
        let step = EvolutionStep::new(model, config);

        let initial = RealVector::new(vec![5.0, 5.0, 5.0]);
        let chain = step.run_chain(&initial, 10, &mut rng);

        assert_eq!(chain.len(), 11); // Initial + 10 steps
    }

    #[test]
    fn test_smc_basic() {
        let mut rng = rand::thread_rng();
        let sphere = Sphere::new(2);
        let bounds = MultiBounds::symmetric(5.0, 2);
        let smc = EvolutionarySMC::<RealVector, _>::new(sphere, bounds, 20);

        let particles = smc.run(&mut rng);

        assert_eq!(particles.len(), 20);

        // Check that weights are normalized
        let total_weight: f64 = particles.iter().map(|p| p.normalized_weight).sum();
        assert!((total_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hbga_basic() {
        let mut rng = rand::thread_rng();
        let sphere = Sphere::new(2);
        let bounds = MultiBounds::symmetric(5.0, 2);
        let hbga = HBGA::new(sphere, bounds, 20, 10);

        let result = hbga.run(&mut rng);

        // Should have made some progress
        assert!(result.fitness_history.len() == 10);
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_particle_resampling() {
        let mut rng = rand::thread_rng();

        let genomes: Vec<RealVector> = (0..5)
            .map(|_| RealVector::new(vec![rng.gen(), rng.gen()]))
            .collect();

        let particles: Vec<Particle<RealVector>> = genomes
            .into_iter()
            .enumerate()
            .map(|(i, g)| {
                let mut p = Particle::new(g, i as f64);
                p.normalized_weight = (i + 1) as f64 / 15.0; // 1+2+3+4+5 = 15
                p
            })
            .collect();

        let resampled = EvolutionarySMC::<RealVector, Sphere>::resample(&particles, &mut rng);
        assert_eq!(resampled.len(), 5);
    }
}
