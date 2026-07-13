//! Evolution as inference over the Boltzmann posterior on genomes
//!
//! # What this module actually computes
//!
//! Fix a fitness function `f: G → ℝ` (higher is better) and a prior `p(x)` over
//! genomes. For an inverse temperature `β ≥ 0` we define the **Boltzmann /
//! Gibbs posterior**
//!
//! ```text
//!     π_β(x) ∝ p(x) · exp(β · f(x)).
//! ```
//!
//! - `β = 0` gives the prior `p`.
//! - `β = 1` gives the Boltzmann posterior with fitness acting as an
//!   (unnormalised) log-likelihood — this is the precise sense in which
//!   "selection pressure = Bayesian conditioning": conditioning on the `factor`
//!   `β·f(x)` reweights the prior by `exp(β·f(x))`.
//! - `β → ∞` concentrates on the global optima of `f`.
//!
//! Two samplers target this object:
//!
//! - [`EvolutionStep`] is a **Metropolis–Hastings** kernel that is invariant for
//!   `π_β`. Its acceptance ratio uses the *full* target `p(x)·exp(β·f(x))`, so
//!   the prior's support is respected (out-of-support proposals are rejected).
//! - [`EvolutionarySMC`] is a **tempered Sequential Monte Carlo sampler** (Del
//!   Moral, Doucet & Jasra 2006; Neal 2001 annealed importance sampling). It
//!   moves a population of particles along a ladder `0 = β_0 < … < β_T = 1`,
//!   using incremental importance weights `w_t = exp((β_t − β_{t−1})·f(x))`,
//!   self-normalisation, ESS-triggered resampling (weights reset to uniform
//!   afterwards), and `π_{β_t}`-invariant MH mutation / crossover moves as
//!   rejuvenation.
//!
//! The fitness-as-likelihood correspondence is made literal in
//! [`EvolutionModel::to_weighted_trace`], which runs a genuine Fugue
//! `factor(β·f(x))` model through [`TraceScoringHandler`] so the returned
//! [`Trace`] has `total_log_weight() == β·f(x)`.

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::PriorHandler;
use fugue::{
    factor, sample, sequence_vec, ChoiceValue, Distribution, Model, Normal, Trace, Uniform,
};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;
use std::marker::PhantomData;

use super::effect_handlers::TraceScoringHandler;
use crate::fitness::traits::Fitness;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::{gene_address, EvolutionaryGenome};

/// Prior distribution over genomes, `p(x)`.
///
/// The prior enters both the initial SMC population (`β = 0`) and the MH
/// acceptance ratio, so its choice determines the support and shape of the
/// Boltzmann posterior `π_β ∝ p·exp(β·f)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prior {
    /// Uniform over the model's per-dimension bounds. The (improper for the
    /// purposes of the ratio) log-density is constant inside the box and
    /// `−∞` outside, so MH proposals that leave the box are rejected. This is
    /// the correct prior for a bounded-domain optimisation problem.
    UniformBounds,
    /// Independent Gaussian `N(mean, std²)` on every real coordinate. Combined
    /// with a quadratic fitness this yields the textbook conjugate Gaussian
    /// posterior, which the SMC sampler reproduces.
    Gaussian {
        /// Prior mean applied to every coordinate.
        mean: f64,
        /// Prior standard deviation applied to every coordinate.
        std: f64,
    },
}

/// A probabilistic model of an evolutionary population.
///
/// Bundles a fitness function, a [`Prior`], an inverse temperature `β`, and the
/// per-dimension bounds. It defines the Boltzmann posterior `π_β ∝ p·exp(β·f)`
/// and the Fugue machinery to sample the prior and score the fitness factor.
#[derive(Clone)]
pub struct EvolutionModel<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    /// The fitness function whose exponential defines the likelihood.
    pub fitness: F,
    /// Inverse temperature `β`. `target ∝ prior · exp(β · f)`.
    beta: f64,
    /// Prior over genomes.
    prior: Prior,
    /// Per-dimension bounds (support of the uniform prior; problem geometry).
    bounds: MultiBounds,
    /// Phantom data for genome type
    _marker: PhantomData<G>,
}

impl<G, F> EvolutionModel<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    /// Create a new evolution model at `β = 1` with a uniform-on-bounds prior.
    pub fn new(fitness: F, bounds: MultiBounds) -> Self {
        Self {
            fitness,
            beta: 1.0,
            prior: Prior::UniformBounds,
            bounds,
            _marker: PhantomData,
        }
    }

    /// Set the inverse temperature `β` directly (`β ≥ 0`).
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.max(0.0);
        self
    }

    /// Set the temperature `T`; equivalent to `β = 1/T`.
    ///
    /// Lower temperature = higher `β` = stronger selection pressure.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };
        self
    }

    /// Set the prior distribution over genomes.
    pub fn with_prior(mut self, prior: Prior) -> Self {
        self.prior = prior;
        self
    }

    /// Current inverse temperature `β`.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Current temperature `T = 1/β`.
    pub fn temperature(&self) -> f64 {
        1.0 / self.beta
    }

    /// The prior over genomes.
    pub fn prior(&self) -> Prior {
        self.prior
    }

    /// The per-dimension bounds.
    pub fn bounds(&self) -> &MultiBounds {
        &self.bounds
    }

    /// Raw fitness `f(x)` (higher is better).
    pub fn fitness_value(&self, genome: &G) -> f64 {
        self.fitness.evaluate(genome)
    }

    /// The tempered fitness log-factor `β · f(x)`.
    ///
    /// This is exactly the `factor` that [`Self::to_weighted_trace`] injects and
    /// equals the trace's `total_log_weight()`.
    pub fn log_weight(&self, genome: &G) -> f64 {
        self.beta * self.fitness_value(genome)
    }

    /// Extract the real-valued coordinates of a genome from its trace.
    fn coords(&self, genome: &G) -> Vec<f64> {
        coords_of(genome)
    }

    /// Log prior density `log p(x)` (up to an additive constant).
    ///
    /// For [`Prior::UniformBounds`] this is `0` inside the box and `−∞` outside.
    /// For [`Prior::Gaussian`] it is the sum of per-coordinate Gaussian
    /// log-densities, computed with Fugue's own [`Normal`] distribution.
    pub fn log_prior_density(&self, genome: &G) -> f64 {
        let coords = self.coords(genome);
        match self.prior {
            Prior::UniformBounds => {
                for (i, &x) in coords.iter().enumerate() {
                    if let Some(b) = self.bounds.get(i) {
                        if x < b.min || x > b.max {
                            return f64::NEG_INFINITY;
                        }
                    }
                }
                0.0
            }
            Prior::Gaussian { mean, std } => {
                let normal = Normal::new(mean, std).expect("Gaussian prior std must be > 0");
                coords.iter().map(|x| normal.log_prob(x)).sum()
            }
        }
    }

    /// Unnormalised log target `log π_β(x) = log p(x) + β · f(x)`.
    pub fn log_boltzmann_target(&self, genome: &G) -> f64 {
        let lp = self.log_prior_density(genome);
        if lp == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        lp + self.beta * self.fitness_value(genome)
    }

    /// Build a genuine Fugue model that samples every coordinate from the prior.
    ///
    /// Running this under a [`PriorHandler`] both draws a prior genome and
    /// accumulates its `log_prior` — real forward sampling, not a hand-rolled
    /// loop.
    fn prior_coordinate_model(&self) -> Model<Vec<f64>> {
        let dim = self.bounds.dimension().max(1);
        let prefix = G::trace_prefix();
        let mut models: Vec<Model<f64>> = Vec::with_capacity(dim);
        for i in 0..dim {
            let a = gene_address(prefix, i);
            let m = match self.prior {
                Prior::UniformBounds => {
                    let (lo, hi) = match self.bounds.get(i) {
                        Some(b) if b.max > b.min => (b.min, b.max),
                        Some(b) => (b.min - 1e-9, b.min + 1e-9),
                        None => (-1.0, 1.0),
                    };
                    sample(a, Uniform::new(lo, hi).expect("valid uniform prior bounds"))
                }
                Prior::Gaussian { mean, std } => sample(
                    a,
                    Normal::new(mean, std).expect("Gaussian prior std must be > 0"),
                ),
            };
            models.push(m);
        }
        sequence_vec(models)
    }

    /// Draw a genome from the prior `p(x)` by running a Fugue model.
    pub fn sample_prior<R: Rng>(&self, rng: &mut R) -> G {
        let model = self.prior_coordinate_model();
        let (_vals, trace) = run(
            PriorHandler {
                rng: &mut *rng,
                trace: Trace::default(),
            },
            model,
        );
        G::from_trace(&trace).unwrap_or_else(|_| G::generate(rng, &self.bounds))
    }

    /// Create a Fugue trace representing this genome whose probability mass
    /// carries the fitness likelihood.
    ///
    /// The returned trace's choices equal `genome.to_trace()` and its
    /// `total_log_weight()` equals `β · f(x)` — obtained by running the genuine
    /// Fugue model `factor(β·f(x))` through [`TraceScoringHandler`], so the
    /// factor lands in `log_factors` (not merely as an inert choice value).
    pub fn to_weighted_trace(&self, genome: &G) -> Trace {
        let logw = self.log_weight(genome);
        let base = genome.to_trace();
        let (_r, trace) = run(TraceScoringHandler::new(base), factor(logw));
        trace
    }
}

/// Extract the real-valued coordinates (`gene#0`, `gene#1`, …) of a genome.
fn coords_of<G: EvolutionaryGenome>(genome: &G) -> Vec<f64> {
    let trace = genome.to_trace();
    let prefix = G::trace_prefix();
    let mut coords = Vec::new();
    let mut i = 0;
    while let Some(x) = trace.get_f64(&gene_address(prefix, i)) {
        coords.push(x);
        i += 1;
    }
    coords
}

/// Configuration for the evolutionary MH kernel [`EvolutionStep`].
#[derive(Clone, Debug)]
pub struct EvolutionChainConfig {
    /// Probability of perturbing each coordinate in a proposal.
    pub mutation_rate: f64,
    /// Standard deviation of the (true Gaussian) coordinate perturbation.
    pub mutation_sigma: f64,
    /// Default number of steps for [`EvolutionStep::run_chain`].
    pub generations: usize,
}

impl Default for EvolutionChainConfig {
    fn default() -> Self {
        Self {
            mutation_rate: 0.5,
            mutation_sigma: 0.1,
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

    /// Set the number of generations
    pub fn generations(mut self, gens: usize) -> Self {
        self.generations = gens;
        self
    }
}

/// A Metropolis–Hastings transition kernel that is invariant for the Boltzmann
/// posterior `π_β ∝ p·exp(β·f)`.
///
/// The proposal is a symmetric Gaussian random walk on the genome's real
/// coordinates. The acceptance ratio uses the *full* log target
/// [`EvolutionModel::log_boltzmann_target`], so the prior's support is honoured:
/// with [`Prior::UniformBounds`] any proposal that leaves the box has
/// `log p = −∞` and is rejected, making the on-bounds Boltzmann posterior the
/// exact stationary law.
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

    /// The underlying model (carrying the current `β`).
    pub fn model(&self) -> &EvolutionModel<G, F> {
        &self.model
    }

    /// Propose a new genome via a symmetric Gaussian random-walk mutation.
    pub fn propose<R: Rng>(&self, current: &G, rng: &mut R) -> G {
        let trace = current.to_trace();
        let mut new_trace = Trace::default();
        let normal = Normal::new(0.0, self.config.mutation_sigma.max(1e-12))
            .expect("mutation sigma must be > 0");

        for (addr, choice) in &trace.choices {
            let new_value = match &choice.value {
                ChoiceValue::F64(v) if rng.gen::<f64>() < self.config.mutation_rate => {
                    ChoiceValue::F64(v + normal.sample(rng))
                }
                other => other.clone(),
            };
            new_trace.insert_choice(addr.clone(), new_value, 0.0);
        }

        G::from_trace(&new_trace).unwrap_or_else(|_| current.clone())
    }

    /// Log Metropolis acceptance ratio `log[π_β(x')/π_β(x)]` for the symmetric
    /// proposal. `−∞` (reject) when the proposal leaves the prior's support.
    pub fn log_acceptance_ratio(&self, current: &G, proposed: &G) -> f64 {
        self.model.log_boltzmann_target(proposed) - self.model.log_boltzmann_target(current)
    }

    /// Metropolis acceptance probability `min(1, π_β(x')/π_β(x))`.
    pub fn acceptance_probability(&self, current: &G, proposed: &G) -> f64 {
        self.log_acceptance_ratio(current, proposed).exp().min(1.0)
    }

    /// Perform one MH step (propose, then accept/reject against `π_β`).
    pub fn step<R: Rng>(&self, current: &G, rng: &mut R) -> G {
        let proposed = self.propose(current, rng);
        let log_ratio = self.log_acceptance_ratio(current, &proposed);
        // log(U) < log_ratio  ⇔  U < min(1, exp(log_ratio)).
        // Handles ±∞ correctly: log_ratio = −∞ ⇒ always reject.
        let u: f64 = rng.gen::<f64>();
        if u.ln() < log_ratio {
            proposed
        } else {
            current.clone()
        }
    }

    /// Run the chain for `num_steps` MH steps, returning every state.
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

/// A weighted particle in the [`EvolutionarySMC`] population.
#[derive(Clone)]
pub struct Particle<G>
where
    G: EvolutionaryGenome,
{
    /// The genome state.
    pub genome: G,
    /// Cached raw fitness `f(x)` of the genome.
    pub fitness: f64,
    /// Unnormalised log importance weight.
    pub log_weight: f64,
    /// Self-normalised weight (fills in after normalisation).
    pub normalized_weight: f64,
}

impl<G: EvolutionaryGenome> Particle<G> {
    /// Create a new particle with a cached fitness and log weight.
    pub fn new(genome: G, fitness: f64, log_weight: f64) -> Self {
        Self {
            genome,
            fitness,
            log_weight,
            normalized_weight: 0.0,
        }
    }
}

/// A tempered Sequential Monte Carlo sampler for the Boltzmann posterior.
///
/// Targets the ladder `γ_t(x) ∝ p(x) · exp(β_t · f(x))` for
/// `0 = β_0 < … < β_T = 1`. Each temperature step:
///
/// 1. **reweights** every particle by the incremental importance weight
///    `w_t = exp((β_t − β_{t−1}) · f(x))` using the *pre-move* state,
/// 2. **self-normalises** the weights and computes the effective sample size,
/// 3. **resamples** (systematic) when `ESS < ess_threshold · N`, resetting the
///    weights to uniform, and
/// 4. **rejuvenates** with `π_{β_t}`-invariant MH mutation and (optionally)
///    joint MH crossover moves.
///
/// This is a valid SMC sampler (Del Moral, Doucet & Jasra 2006): the weighted
/// particles approximate `π_{β_T} = π_1`, the Boltzmann posterior.
pub struct EvolutionarySMC<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    model: EvolutionModel<G, F>,
    /// Number of particles.
    num_particles: usize,
    /// Ascending inverse-temperature ladder `0 = β_0 < … < β_T = 1`.
    beta_schedule: Vec<f64>,
    /// Resample when `ESS < ess_threshold · N`.
    ess_threshold: f64,
    /// Number of MH rejuvenation sweeps per temperature.
    mcmc_steps: usize,
    /// Coordinate perturbation probability of the MH mutation kernel.
    mutation_rate: f64,
    /// Coordinate perturbation std of the MH mutation kernel.
    mutation_sigma: f64,
    /// Whether to apply a joint MH crossover sweep during rejuvenation.
    use_crossover: bool,
}

impl<G, F> EvolutionarySMC<G, F>
where
    G: EvolutionaryGenome + Clone,
    F: Fitness<Genome = G, Value = f64> + Clone,
{
    /// Create a new SMC sampler over a default 20-rung linear `β` ladder.
    pub fn new(fitness: F, bounds: MultiBounds, num_particles: usize) -> Self {
        Self {
            model: EvolutionModel::new(fitness, bounds),
            num_particles,
            beta_schedule: Self::linear_beta_schedule(20),
            ess_threshold: 0.5,
            mcmc_steps: 3,
            mutation_rate: 0.5,
            mutation_sigma: 0.5,
            use_crossover: true,
        }
    }

    /// A linear inverse-temperature ladder `β_i = i/steps` for `i = 0..=steps`.
    pub fn linear_beta_schedule(steps: usize) -> Vec<f64> {
        let steps = steps.max(1);
        (0..=steps).map(|i| i as f64 / steps as f64).collect()
    }

    /// Set the prior over genomes.
    pub fn with_prior(mut self, prior: Prior) -> Self {
        self.model = self.model.with_prior(prior);
        self
    }

    /// Set a custom inverse-temperature ladder (should be ascending, ending 1).
    pub fn with_beta_schedule(mut self, schedule: Vec<f64>) -> Self {
        self.beta_schedule = schedule;
        self
    }

    /// Set the resampling ESS threshold fraction (default 0.5).
    pub fn with_ess_threshold(mut self, frac: f64) -> Self {
        self.ess_threshold = frac.clamp(0.0, 1.0);
        self
    }

    /// Set the number of MH rejuvenation sweeps per temperature.
    pub fn with_mcmc_steps(mut self, steps: usize) -> Self {
        self.mcmc_steps = steps;
        self
    }

    /// Set the MH mutation kernel parameters.
    pub fn with_mutation(mut self, rate: f64, sigma: f64) -> Self {
        self.mutation_rate = rate;
        self.mutation_sigma = sigma;
        self
    }

    /// Enable or disable the joint MH crossover rejuvenation sweep.
    pub fn with_crossover(mut self, enabled: bool) -> Self {
        self.use_crossover = enabled;
        self
    }

    /// The underlying model.
    pub fn model(&self) -> &EvolutionModel<G, F> {
        &self.model
    }

    /// Initialise particles from the prior (`β = 0`) with uniform weights.
    pub fn initialize<R: Rng>(&self, rng: &mut R) -> Vec<Particle<G>> {
        let inv_n = 1.0 / self.num_particles as f64;
        (0..self.num_particles)
            .map(|_| {
                let genome = self.model.sample_prior(rng);
                let fitness = self.model.fitness_value(&genome);
                let mut p = Particle::new(genome, fitness, 0.0);
                p.normalized_weight = inv_n;
                p
            })
            .collect()
    }

    /// Self-normalise particle weights via the log-sum-exp trick.
    fn normalize_weights(particles: &mut [Particle<G>]) {
        let max_log_weight = particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        if !max_log_weight.is_finite() {
            let inv_n = 1.0 / particles.len() as f64;
            for p in particles.iter_mut() {
                p.normalized_weight = inv_n;
            }
            return;
        }

        let sum: f64 = particles
            .iter()
            .map(|p| (p.log_weight - max_log_weight).exp())
            .sum();
        let log_sum = sum.ln() + max_log_weight;

        for particle in particles.iter_mut() {
            particle.normalized_weight = (particle.log_weight - log_sum).exp();
        }
    }

    /// Effective sample size `1 / Σ w̃_i²`.
    pub fn effective_sample_size(particles: &[Particle<G>]) -> f64 {
        let sum_sq: f64 = particles.iter().map(|p| p.normalized_weight.powi(2)).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    }

    /// Systematic resampling. Resampled particles carry uniform weight.
    pub fn resample<R: Rng>(particles: &[Particle<G>], rng: &mut R) -> Vec<Particle<G>> {
        let n = particles.len();
        let inv_n = 1.0 / n as f64;
        let mut resampled = Vec::with_capacity(n);

        let mut cumulative = vec![0.0; n];
        cumulative[0] = particles[0].normalized_weight;
        for i in 1..n {
            cumulative[i] = cumulative[i - 1] + particles[i].normalized_weight;
        }

        let u0: f64 = rng.gen::<f64>() / n as f64;
        let mut j = 0;
        for i in 0..n {
            let u = u0 + i as f64 / n as f64;
            while j < n - 1 && cumulative[j] < u {
                j += 1;
            }
            let mut particle = particles[j].clone();
            particle.log_weight = 0.0;
            particle.normalized_weight = inv_n;
            resampled.push(particle);
        }

        resampled
    }

    /// One MH mutation sweep at inverse temperature `beta` (π_β-invariant).
    fn mutation_sweep<R: Rng>(&self, particles: &mut [Particle<G>], beta: f64, rng: &mut R) {
        let step_model = self.model.clone().with_beta(beta);
        let config = EvolutionChainConfig::default()
            .mutation_rate(self.mutation_rate)
            .mutation_sigma(self.mutation_sigma);
        let step = EvolutionStep::new(step_model, config);

        for p in particles.iter_mut() {
            let moved = step.step(&p.genome, rng);
            p.fitness = self.model.fitness_value(&moved);
            p.genome = moved;
        }
    }

    /// One joint MH crossover sweep at inverse temperature `beta`.
    ///
    /// Particles are shuffled and paired. For each pair a random subset `S` of
    /// coordinates is proposed to be swapped between the two genomes. Because
    /// the swap proposal is symmetric on the pair, accepting it with
    /// `min(1, [π_β(x')π_β(y')]/[π_β(x)π_β(y)])` leaves the product target
    /// `π_β ⊗ π_β` invariant, hence each particle's marginal stays `π_β`.
    fn crossover_sweep<R: Rng>(&self, particles: &mut [Particle<G>], beta: f64, rng: &mut R) {
        let n = particles.len();
        if n < 2 {
            return;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(rng);

        let target = |g: &G, f: f64| -> f64 {
            let lp = self.model.log_prior_density(g);
            if lp == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                lp + beta * f
            }
        };

        for pair in order.chunks(2) {
            if pair.len() < 2 {
                continue;
            }
            let (i, j) = (pair[0], pair[1]);

            // Random symmetric swap mask over the shared coordinate addresses.
            let trace_i = particles[i].genome.to_trace();
            let swap: HashSet<_> = trace_i
                .choices
                .keys()
                .filter(|_| rng.gen::<bool>())
                .cloned()
                .collect();
            if swap.is_empty() {
                continue;
            }

            let (child_i, child_j) =
                swap_coordinates(&particles[i].genome, &particles[j].genome, &swap);
            let fi = self.model.fitness_value(&child_i);
            let fj = self.model.fitness_value(&child_j);

            let log_ratio = target(&child_i, fi) + target(&child_j, fj)
                - target(&particles[i].genome, particles[i].fitness)
                - target(&particles[j].genome, particles[j].fitness);

            if rng.gen::<f64>().ln() < log_ratio {
                particles[i].genome = child_i;
                particles[i].fitness = fi;
                particles[j].genome = child_j;
                particles[j].fitness = fj;
            }
        }
    }

    /// Run the SMC sampler, returning the final weighted particle population
    /// approximating the Boltzmann posterior `π_1 ∝ p·exp(f)`.
    pub fn run<R: Rng>(&self, rng: &mut R) -> Vec<Particle<G>> {
        let mut particles = self.initialize(rng);
        if self.beta_schedule.is_empty() {
            return particles;
        }

        let n = self.num_particles as f64;
        let mut prev_beta = self.beta_schedule[0];

        for &beta in self.beta_schedule.iter().skip(1) {
            let dbeta = beta - prev_beta;

            // (1) Incremental reweight using the pre-move fitness.
            for p in &mut particles {
                p.log_weight += dbeta * p.fitness;
            }
            Self::normalize_weights(&mut particles);

            // (2) Resample when ESS drops below the threshold.
            let ess = Self::effective_sample_size(&particles);
            if ess < self.ess_threshold * n {
                particles = Self::resample(&particles, rng);
            }

            // (3) π_β-invariant rejuvenation (mutation + optional crossover).
            for _ in 0..self.mcmc_steps {
                self.mutation_sweep(&mut particles, beta, rng);
                if self.use_crossover {
                    self.crossover_sweep(&mut particles, beta, rng);
                }
            }

            prev_beta = beta;
        }

        Self::normalize_weights(&mut particles);
        particles
    }

    /// The particle with the highest fitness.
    pub fn best_particle(particles: &[Particle<G>]) -> Option<&Particle<G>> {
        particles.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Self-normalised weighted per-coordinate posterior mean.
    pub fn weighted_mean(particles: &[Particle<G>]) -> Vec<f64> {
        if particles.is_empty() {
            return Vec::new();
        }
        let dim = coords_of(&particles[0].genome).len();
        let total: f64 = particles.iter().map(|p| p.normalized_weight).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        let mut mean = vec![0.0; dim];
        for p in particles {
            let c = coords_of(&p.genome);
            for k in 0..dim.min(c.len()) {
                mean[k] += p.normalized_weight * c[k];
            }
        }
        for m in &mut mean {
            *m /= total;
        }
        mean
    }

    /// Self-normalised weighted per-coordinate posterior variance.
    pub fn weighted_variance(particles: &[Particle<G>]) -> Vec<f64> {
        if particles.is_empty() {
            return Vec::new();
        }
        let mean = Self::weighted_mean(particles);
        let dim = mean.len();
        let total: f64 = particles.iter().map(|p| p.normalized_weight).sum();
        let total = if total > 0.0 { total } else { 1.0 };
        let mut var = vec![0.0; dim];
        for p in particles {
            let c = coords_of(&p.genome);
            for k in 0..dim.min(c.len()) {
                let d = c[k] - mean[k];
                var[k] += p.normalized_weight * d * d;
            }
        }
        for v in &mut var {
            *v /= total;
        }
        var
    }
}

/// Build two children by swapping the `swap` coordinate addresses between two
/// genomes (in trace space). `child_i` equals `a` except on `swap` (where it
/// takes `b`); `child_j` equals `b` except on `swap` (where it takes `a`).
fn swap_coordinates<G: EvolutionaryGenome>(a: &G, b: &G, swap: &HashSet<fugue::Address>) -> (G, G) {
    let ta = a.to_trace();
    let tb = b.to_trace();
    let mut child_i = Trace::default();
    let mut child_j = Trace::default();

    for (addr, choice) in &ta.choices {
        if swap.contains(addr) {
            let from_b = tb
                .choices
                .get(addr)
                .map(|c| c.value.clone())
                .unwrap_or_else(|| choice.value.clone());
            child_i.insert_choice(addr.clone(), from_b, 0.0);
            child_j.insert_choice(addr.clone(), choice.value.clone(), 0.0);
        } else {
            let from_b = tb
                .choices
                .get(addr)
                .map(|c| c.value.clone())
                .unwrap_or_else(|| choice.value.clone());
            child_i.insert_choice(addr.clone(), choice.value.clone(), 0.0);
            child_j.insert_choice(addr.clone(), from_b, 0.0);
        }
    }

    let ci = G::from_trace(&child_i).unwrap_or_else(|_| a.clone());
    let cj = G::from_trace(&child_j).unwrap_or_else(|_| b.clone());
    (ci, cj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::traits::Fitness;
    use crate::genome::bounds::{Bounds, MultiBounds};
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A `Clone`-able fitness wrapping a function pointer, so it satisfies the
    /// `F: Clone` bound required by `EvolutionarySMC`.
    #[derive(Clone, Copy)]
    struct PtrFitness(fn(&RealVector) -> f64);

    impl Fitness for PtrFitness {
        type Genome = RealVector;
        type Value = f64;
        fn evaluate(&self, genome: &RealVector) -> f64 {
            (self.0)(genome)
        }
    }

    fn quad_k1_c3(g: &RealVector) -> f64 {
        -0.5 * g.genes().iter().map(|x| (x - 3.0).powi(2)).sum::<f64>()
    }
    fn quad_origin(g: &RealVector) -> f64 {
        -0.5 * g.genes().iter().map(|x| x * x).sum::<f64>()
    }
    fn linear_x0(g: &RealVector) -> f64 {
        g.genes()[0]
    }

    #[test]
    fn test_to_weighted_trace_carries_fitness_mass() {
        // regression: EV-52 — total_log_weight() must equal β·f(x), not 0.
        let bounds = MultiBounds::symmetric(5.0, 2);
        let model = EvolutionModel::new(PtrFitness(quad_origin), bounds).with_beta(2.0);
        let genome = RealVector::new(vec![1.0, 2.0]);
        let f = model.fitness_value(&genome); // -0.5*(1+4) = -2.5
        let trace = model.to_weighted_trace(&genome);
        assert!((trace.total_log_weight() - 2.0 * f).abs() < 1e-9);
        // And the mass is genuinely in log_factors, not an inert choice value.
        assert!((trace.log_factors - 2.0 * f).abs() < 1e-9);
        assert!(trace.total_log_weight().abs() > 1e-6);
    }

    #[test]
    fn test_mh_respects_bounds() {
        // regression: EV-90 — no MH sample may escape the uniform-prior bounds,
        // and the boundary is not over-weighted (mean matches the truncated
        // exponential analytic value).
        // Fitness f(x) = x pushes mass toward the upper bound; on [-2, 2] the
        // β=1 Boltzmann posterior is ∝ e^x truncated to [-2, 2], with analytic
        // mean (e^2 + 3 e^-2)/(e^2 - e^-2) ≈ 1.0746.
        let bounds = MultiBounds::new(vec![Bounds::new(-2.0, 2.0)]);
        let model = EvolutionModel::new(PtrFitness(linear_x0), bounds).with_beta(1.0);
        let config = EvolutionChainConfig::default()
            .mutation_rate(1.0)
            .mutation_sigma(0.7);
        let step = EvolutionStep::new(model, config);

        let mut rng = StdRng::seed_from_u64(20260710);
        let mut current = RealVector::new(vec![0.0]);
        let mut samples = Vec::new();
        // Burn-in then collect.
        for i in 0..40_000 {
            current = step.step(&current, &mut rng);
            let x = current.genes()[0];
            assert!((-2.0..=2.0).contains(&x), "MH sample escaped bounds: {}", x);
            if i >= 5_000 {
                samples.push(x);
            }
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let analytic = {
            let e2 = 2.0_f64.exp();
            let em2 = (-2.0_f64).exp();
            (e2 + 3.0 * em2) / (e2 - em2)
        };
        assert!(
            (mean - analytic).abs() < 0.1,
            "posterior mean {} deviates from truncated-exponential analytic {}",
            mean,
            analytic
        );
    }

    #[test]
    fn test_smc_matches_gaussian_conjugate_posterior() {
        // regression: EV-16 — a valid tempered SMC on a quadratic fitness with a
        // Gaussian prior reproduces the conjugate Boltzmann posterior.
        //
        // Prior N(0, σ0²=4) ⇒ precision τ0 = 0.25.
        // Fitness f(x) = -0.5*(x-3)^2 ⇒ likelihood precision k = 1, center c = 3.
        // Posterior at β=1: precision τ = τ0 + k = 1.25,
        //   mean = (τ0*0 + k*c)/τ = 3/1.25 = 2.4, variance = 1/τ = 0.8.
        let k = 1.0;
        let c = 3.0;
        let sigma0 = 2.0;
        let bounds = MultiBounds::new(vec![Bounds::new(-30.0, 30.0)]);
        let smc = EvolutionarySMC::new(PtrFitness(quad_k1_c3), bounds, 4000)
            .with_prior(Prior::Gaussian {
                mean: 0.0,
                std: sigma0,
            })
            .with_beta_schedule(EvolutionarySMC::<RealVector, PtrFitness>::linear_beta_schedule(16))
            .with_mcmc_steps(6)
            .with_mutation(1.0, 0.6)
            .with_crossover(false);

        let mut rng = StdRng::seed_from_u64(42);
        let particles = smc.run(&mut rng);

        let tau0 = 1.0 / (sigma0 * sigma0);
        let tau = tau0 + k;
        let post_mean = (k * c) / tau;
        let post_var = 1.0 / tau;

        let mean = EvolutionarySMC::<RealVector, PtrFitness>::weighted_mean(&particles)[0];
        let var = EvolutionarySMC::<RealVector, PtrFitness>::weighted_variance(&particles)[0];

        assert!(
            (mean - post_mean).abs() < 0.15,
            "posterior mean {} vs analytic {}",
            mean,
            post_mean
        );
        assert!(
            (var - post_var).abs() < 0.2,
            "posterior variance {} vs analytic {}",
            var,
            post_var
        );

        // Weights are self-normalised.
        let total: f64 = particles.iter().map(|p| p.normalized_weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evolution_model_basic() {
        let bounds = MultiBounds::symmetric(5.0, 3);
        let model = EvolutionModel::new(PtrFitness(quad_origin), bounds);

        let genome = RealVector::new(vec![0.0, 0.0, 0.0]);
        let weight = model.log_weight(&genome);
        assert!(weight.is_finite());
        assert_eq!(weight, 0.0); // fitness 0 at optimum
    }

    #[test]
    fn test_smc_basic_normalized() {
        let mut rng = StdRng::seed_from_u64(7);
        let bounds = MultiBounds::symmetric(5.0, 2);
        let smc = EvolutionarySMC::new(PtrFitness(quad_origin), bounds, 40)
            .with_beta_schedule(EvolutionarySMC::<RealVector, PtrFitness>::linear_beta_schedule(6))
            .with_mcmc_steps(2);

        let particles = smc.run(&mut rng);
        assert_eq!(particles.len(), 40);
        let total_weight: f64 = particles.iter().map(|p| p.normalized_weight).sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_particle_resampling() {
        let mut rng = StdRng::seed_from_u64(1);
        let genomes: Vec<RealVector> = (0..5)
            .map(|_| RealVector::new(vec![rng.gen(), rng.gen()]))
            .collect();

        let particles: Vec<Particle<RealVector>> = genomes
            .into_iter()
            .enumerate()
            .map(|(i, g)| {
                let mut p = Particle::new(g, i as f64, i as f64);
                p.normalized_weight = (i + 1) as f64 / 15.0; // 1+2+3+4+5 = 15
                p
            })
            .collect();

        let resampled = EvolutionarySMC::<RealVector, PtrFitness>::resample(&particles, &mut rng);
        assert_eq!(resampled.len(), 5);
        for p in &resampled {
            assert_eq!(p.log_weight, 0.0);
        }
    }

    #[test]
    fn test_sample_prior_gaussian_uses_fugue_model() {
        // The Gaussian prior draws are actually Gaussian (mean ≈ 0, std ≈ 2).
        let bounds = MultiBounds::symmetric(50.0, 1);
        let model =
            EvolutionModel::new(PtrFitness(quad_origin), bounds).with_prior(Prior::Gaussian {
                mean: 0.0,
                std: 2.0,
            });
        let mut rng = StdRng::seed_from_u64(99);
        let xs: Vec<f64> = (0..5000)
            .map(|_| model.sample_prior(&mut rng).genes()[0])
            .collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.2, "prior mean {}", mean);
        assert!((var.sqrt() - 2.0).abs() < 0.2, "prior std {}", var.sqrt());
    }
}
