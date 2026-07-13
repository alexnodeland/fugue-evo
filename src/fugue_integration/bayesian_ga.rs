//! A single-level Bayesian adaptive genetic algorithm
//!
//! This module replaces the former "HBGA" heuristic (which collapsed its priors
//! to their means and adapted with a fixed `×1.05 / ×0.95` rule) with a genuine
//! Bayesian treatment of the operator hyperparameters.
//!
//! # The model
//!
//! Each mutation operator `k` (a coordinate-wise Gaussian with its own step
//! size `σ_k`) has an unknown per-application **success probability** `θ_k` —
//! the probability that applying it to a parent yields a fitter child. We place
//! a conjugate `Beta(α_k, β_k)` prior on `θ_k` and treat each offspring as a
//! Bernoulli improvement trial, so the posterior is the exact conjugate update
//!
//! ```text
//!     α_k ← α_k + (# improving children),   β_k ← β_k + (# non-improving).
//! ```
//!
//! Operator selection is **Thompson sampling**: every generation we draw one
//! `θ̃_k ~ Beta(α_k, β_k)` from each *current* posterior and apply the operator
//! with the largest draw. This replaces the fixed heuristic with hyperparameters
//! sampled from the current posterior each generation.
//!
//! In addition, a `Gamma(shape, rate)` posterior tracks the rate `λ` of
//! improvement events per generation via the conjugate Gamma–Poisson update
//! (`shape ← shape + count`, `rate ← rate + 1` each generation). It is reported
//! as a learned diagnostic of how "improvable" the search currently is.
//!
//! This is a **single-level** Bayesian model (independent conjugate posteriors,
//! no hyperprior over the `Beta`/`Gamma` parameters), hence the honest name
//! `BayesianAdaptiveGA` rather than "hierarchical Bayesian".

use fugue::{Beta, Distribution, Gamma};
use rand::Rng;

use super::evolution_model::{EvolutionChainConfig, EvolutionModel, EvolutionStep};
use crate::fitness::traits::Fitness;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;

/// Conjugate `Beta(α, β)` posterior over a Bernoulli success probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaSuccessPosterior {
    /// Pseudo-count of successes (`α`).
    pub alpha: f64,
    /// Pseudo-count of failures (`β`).
    pub beta: f64,
}

impl BetaSuccessPosterior {
    /// Create a `Beta(α, β)` prior. Both parameters must be positive.
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(1e-6),
            beta: beta.max(1e-6),
        }
    }

    /// Conjugate update from observed Bernoulli trials.
    pub fn update(&mut self, successes: u64, failures: u64) {
        self.alpha += successes as f64;
        self.beta += failures as f64;
    }

    /// Posterior mean `α / (α + β)`.
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance `αβ / ((α+β)² (α+β+1))`.
    pub fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// Total observed evidence `α + β`.
    pub fn total(&self) -> f64 {
        self.alpha + self.beta
    }

    /// Draw `θ ~ Beta(α, β)` from the current posterior (Thompson sampling).
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Beta::new(self.alpha, self.beta)
            .expect("valid Beta parameters")
            .sample(rng)
    }
}

/// Conjugate `Gamma(shape, rate)` posterior over a Poisson rate `λ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaRatePosterior {
    /// Shape parameter.
    pub shape: f64,
    /// Rate parameter (inverse scale).
    pub rate: f64,
}

impl GammaRatePosterior {
    /// Create a `Gamma(shape, rate)` prior. Both parameters must be positive.
    pub fn new(shape: f64, rate: f64) -> Self {
        Self {
            shape: shape.max(1e-6),
            rate: rate.max(1e-6),
        }
    }

    /// Conjugate Gamma–Poisson update: observe `count` events over `exposure`
    /// units of exposure (`shape += count`, `rate += exposure`).
    pub fn observe(&mut self, count: u64, exposure: f64) {
        self.shape += count as f64;
        self.rate += exposure;
    }

    /// Posterior mean `shape / rate`.
    pub fn mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Draw `λ ~ Gamma(shape, rate)` from the current posterior.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        Gamma::new(self.shape, self.rate)
            .expect("valid Gamma parameters")
            .sample(rng)
    }
}

/// A mutation operator arm: a Gaussian step size with its own success posterior.
#[derive(Clone, Copy, Debug)]
pub struct OperatorArm {
    /// Gaussian mutation standard deviation.
    pub sigma: f64,
    /// Posterior over this operator's per-application success probability.
    pub posterior: BetaSuccessPosterior,
    /// Number of generations this arm was selected.
    pub times_selected: usize,
}

impl OperatorArm {
    /// Create an arm with the given step size and a `Beta(1, 1)` prior.
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            posterior: BetaSuccessPosterior::new(1.0, 1.0),
            times_selected: 0,
        }
    }
}

/// A single-level Bayesian adaptive genetic algorithm.
///
/// See the [module docs](self) for the model. Operator step sizes are selected
/// by Thompson sampling over per-operator `Beta` success posteriors, which are
/// updated by conjugate Bayesian updates from observed improvement events.
pub struct BayesianAdaptiveGA<G, F>
where
    G: EvolutionaryGenome,
    F: Fitness<Genome = G, Value = f64>,
{
    model: EvolutionModel<G, F>,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_rate: f64,
    arms: Vec<OperatorArm>,
    improvement_rate: GammaRatePosterior,
}

impl<G, F> BayesianAdaptiveGA<G, F>
where
    G: EvolutionaryGenome + Clone,
    F: Fitness<Genome = G, Value = f64> + Clone,
{
    /// Create a new adaptive GA with a default set of mutation step sizes.
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
            tournament_size: 3,
            mutation_rate: 0.5,
            arms: vec![
                OperatorArm::new(0.05),
                OperatorArm::new(0.2),
                OperatorArm::new(0.5),
                OperatorArm::new(1.0),
            ],
            improvement_rate: GammaRatePosterior::new(1.0, 1.0),
        }
    }

    /// Replace the operator step sizes (each gets a fresh `Beta(1, 1)` prior).
    pub fn with_step_sizes(mut self, sigmas: Vec<f64>) -> Self {
        self.arms = sigmas.into_iter().map(OperatorArm::new).collect();
        self
    }

    /// Set the coordinate mutation probability.
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set the tournament size for parent selection.
    pub fn with_tournament_size(mut self, size: usize) -> Self {
        self.tournament_size = size.max(1);
        self
    }

    /// Thompson-sample one `θ̃_k` from each arm's current posterior and return
    /// the index of the arm with the largest draw.
    fn thompson_select<R: Rng>(&self, rng: &mut R) -> usize {
        let mut best_idx = 0;
        let mut best_draw = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let draw = arm.posterior.sample(rng);
            if draw > best_draw {
                best_draw = draw;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Run the adaptive GA.
    pub fn run<R: Rng>(&mut self, rng: &mut R) -> BayesianAdaptiveGAResult<G> {
        let mut population: Vec<G> = (0..self.population_size)
            .map(|_| self.model.sample_prior(rng))
            .collect();
        let mut fitnesses: Vec<f64> = population
            .iter()
            .map(|g| self.model.fitness_value(g))
            .collect();

        let mut best_genome = population[0].clone();
        let mut best_fitness = fitnesses[0];
        for (g, &f) in population.iter().zip(fitnesses.iter()) {
            if f > best_fitness {
                best_fitness = f;
                best_genome = g.clone();
            }
        }

        let mut fitness_history = Vec::with_capacity(self.generations);
        let mut selected_arm_history = Vec::with_capacity(self.generations);

        for _ in 0..self.generations {
            // (1) Thompson-sample the operator to use this generation.
            let arm_idx = self.thompson_select(rng);
            self.arms[arm_idx].times_selected += 1;
            selected_arm_history.push(arm_idx);
            let sigma = self.arms[arm_idx].sigma;

            let step = EvolutionStep::new(
                self.model.clone(),
                EvolutionChainConfig::default()
                    .mutation_rate(self.mutation_rate)
                    .mutation_sigma(sigma),
            );

            // (2) Produce the next generation via tournament selection + the
            // chosen mutation operator, recording improvement events.
            let mut next_population = Vec::with_capacity(self.population_size);
            let mut next_fitness = Vec::with_capacity(self.population_size);
            let mut successes: u64 = 0;
            let mut failures: u64 = 0;

            for _ in 0..self.population_size {
                let parent_idx = self.tournament(&fitnesses, rng);
                let parent = &population[parent_idx];
                let parent_fitness = fitnesses[parent_idx];

                let child = step.propose(parent, rng);
                let child_fitness = self.model.fitness_value(&child);

                if child_fitness > parent_fitness {
                    successes += 1;
                } else {
                    failures += 1;
                }

                if child_fitness > best_fitness {
                    best_fitness = child_fitness;
                    best_genome = child.clone();
                }

                next_population.push(child);
                next_fitness.push(child_fitness);
            }

            // (3) Conjugate Bayesian updates from the observed events.
            self.arms[arm_idx].posterior.update(successes, failures);
            self.improvement_rate.observe(successes, 1.0);

            population = next_population;
            fitnesses = next_fitness;

            let mean_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
            fitness_history.push(mean_fitness);
        }

        BayesianAdaptiveGAResult {
            best_genome,
            best_fitness,
            fitness_history,
            selected_arm_history,
            operator_posteriors: self.arms.clone(),
            improvement_rate: self.improvement_rate,
        }
    }

    fn tournament<R: Rng>(&self, fitnesses: &[f64], rng: &mut R) -> usize {
        let mut best = rng.gen_range(0..fitnesses.len());
        for _ in 1..self.tournament_size {
            let challenger = rng.gen_range(0..fitnesses.len());
            if fitnesses[challenger] > fitnesses[best] {
                best = challenger;
            }
        }
        best
    }
}

/// Result of a [`BayesianAdaptiveGA`] run.
pub struct BayesianAdaptiveGAResult<G> {
    /// Best genome found.
    pub best_genome: G,
    /// Best fitness value.
    pub best_fitness: f64,
    /// Mean fitness over generations.
    pub fitness_history: Vec<f64>,
    /// Index of the operator arm selected in each generation.
    pub selected_arm_history: Vec<usize>,
    /// Final per-operator success posteriors.
    pub operator_posteriors: Vec<OperatorArm>,
    /// Final `Gamma` posterior over the improvement-event rate.
    pub improvement_rate: GammaRatePosterior,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::genome::bounds::MultiBounds;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_beta_posterior_conjugate_update() {
        // regression: EV-53 — the posterior is a genuine conjugate Beta update,
        // not a prior collapsed to its mean.
        let mut post = BetaSuccessPosterior::new(2.0, 8.0);
        assert!((post.mean() - 0.2).abs() < 1e-12);
        post.update(5, 3);
        assert_eq!(post.alpha, 7.0);
        assert_eq!(post.beta, 11.0);
        assert!((post.mean() - 7.0 / 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_beta_posterior_sampling_matches_beta_moments() {
        // regression: EV-53 — draws are true Beta(α,β) samples. The old HBGA
        // returned mean + U(-0.05, 0.05), whose std ≈ 0.029 would fail here;
        // Beta(2,8) has std ≈ 0.1206.
        let post = BetaSuccessPosterior::new(2.0, 8.0);
        let mut rng = StdRng::seed_from_u64(2024);
        let draws: Vec<f64> = (0..20000).map(|_| post.sample(&mut rng)).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!((mean - post.mean()).abs() < 0.01, "mean {}", mean);
        assert!(
            (var.sqrt() - post.variance().sqrt()).abs() < 0.02,
            "std {} vs analytic {}",
            var.sqrt(),
            post.variance().sqrt()
        );
        // Distinguishes a real Beta draw from the old collapsed sampler.
        assert!(
            var.sqrt() > 0.06,
            "std too small for a Beta draw: {}",
            var.sqrt()
        );
    }

    #[test]
    fn test_gamma_posterior_conjugate_update() {
        let mut post = GammaRatePosterior::new(2.0, 1.0);
        post.observe(5, 1.0);
        assert_eq!(post.shape, 7.0);
        assert_eq!(post.rate, 2.0);
        assert!((post.mean() - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_ga_updates_posteriors_and_improves() {
        // regression: EV-53 — running the GA performs real posterior updates
        // (total evidence grows) and Thompson sampling drives optimisation.
        // Sphere::evaluate returns -Σx² (higher is better, optimum 0 at origin).
        let fit = Sphere::new(3);
        let bounds = MultiBounds::symmetric(5.0, 3);
        let mut ga = BayesianAdaptiveGA::new(fit, bounds, 40, 60);
        let mut rng = StdRng::seed_from_u64(7);
        let result = ga.run(&mut rng);

        // Every generation contributes population_size trials to some arm.
        let total_evidence: f64 = result
            .operator_posteriors
            .iter()
            .map(|a| a.posterior.total() - 2.0) // subtract Beta(1,1) prior mass
            .sum();
        assert!(
            total_evidence >= (60 * 40) as f64 - 1.0,
            "posteriors did not accumulate the expected evidence: {}",
            total_evidence
        );

        // At least one arm was actually exercised (Thompson selection ran).
        assert!(result
            .operator_posteriors
            .iter()
            .any(|a| a.times_selected > 0));

        // The improvement-rate Gamma posterior was updated away from its prior.
        assert!(result.improvement_rate.shape > 1.0);

        // Optimisation made real progress toward the sphere optimum (0).
        assert!(
            result.best_fitness > -1.0,
            "best fitness {} did not converge",
            result.best_fitness
        );
    }

    #[test]
    fn test_thompson_prefers_better_operator() {
        // On a problem near the optimum, small steps improve far more often than
        // huge ones, so the small-σ arm should earn a higher posterior mean.
        let fit = Sphere::new(2);
        let bounds = MultiBounds::symmetric(0.5, 2);
        let mut ga = BayesianAdaptiveGA::new(fit, bounds, 50, 80).with_step_sizes(vec![0.02, 2.0]);
        let mut rng = StdRng::seed_from_u64(11);
        let result = ga.run(&mut rng);

        let small = result.operator_posteriors[0].posterior.mean();
        let large = result.operator_posteriors[1].posterior.mean();
        assert!(
            small > large,
            "small-step success posterior {} should exceed large-step {}",
            small,
            large
        );
    }
}
