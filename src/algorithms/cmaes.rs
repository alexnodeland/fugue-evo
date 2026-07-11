//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
//!
//! Implements the CMA-ES algorithm with full covariance matrix adaptation,
//! evolution path management, and step-size control.
//!
//! Reference: Hansen, N., & Ostermeier, A. (2001). Completely Derandomized
//! Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2).

use std::marker::PhantomData;

use nalgebra::{DMatrix, SymmetricEigen};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{EvoResult, EvolutionError};
use crate::genome::bounds::MultiBounds;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::RealValuedGenome;
use crate::population::individual::Individual;

/// Trait for fitness functions used with CMA-ES
///
/// CMA-ES is a minimization algorithm, so lower values are better.
#[cfg(feature = "parallel")]
pub trait CmaEsFitness: Send + Sync {
    /// Evaluate the fitness of a solution (lower is better)
    fn evaluate(&self, x: &RealVector) -> f64;
}

/// Trait for fitness functions used with CMA-ES
///
/// CMA-ES is a minimization algorithm, so lower values are better.
#[cfg(not(feature = "parallel"))]
pub trait CmaEsFitness {
    /// Evaluate the fitness of a solution (lower is better)
    fn evaluate(&self, x: &RealVector) -> f64;
}

/// CMA-ES state containing all adaptation parameters
#[derive(Clone, Debug)]
pub struct CmaEsState {
    /// Current mean of the search distribution
    pub mean: Vec<f64>,

    /// Global step size (σ)
    pub sigma: f64,

    /// Covariance matrix C (stored as upper triangular for efficiency)
    pub covariance: Vec<Vec<f64>>,

    /// Evolution path for σ adaptation (p_σ)
    pub path_sigma: Vec<f64>,

    /// Evolution path for C adaptation (p_c)
    pub path_c: Vec<f64>,

    /// Eigenvalues of C (D²)
    pub eigenvalues: Vec<f64>,

    /// Eigenvectors of C (B)
    pub eigenvectors: Vec<Vec<f64>>,

    /// Generation counter for eigendecomposition
    pub eigen_eval: usize,

    /// Problem dimension
    pub dimension: usize,

    /// Population size (λ)
    pub lambda: usize,

    /// Parent number (μ)
    pub mu: usize,

    /// Recombination weights
    pub weights: Vec<f64>,

    /// Variance effective selection mass (μ_eff)
    pub mu_eff: f64,

    /// Learning rate for rank-1 update
    pub c_1: f64,

    /// Learning rate for rank-μ update
    pub c_mu: f64,

    /// Learning rate for cumulation for σ control
    pub c_sigma: f64,

    /// Damping for σ
    pub d_sigma: f64,

    /// Learning rate for cumulation for C
    pub c_c: f64,

    /// Expected length of random vector ||N(0, I)||
    pub chi_n: f64,

    /// Current generation
    pub generation: usize,

    /// Total evaluations
    pub evaluations: usize,

    /// Best fitness found
    pub best_fitness: f64,

    /// Best solution found
    pub best_solution: Vec<f64>,
}

impl CmaEsState {
    /// Create a new CMA-ES state
    pub fn new(initial_mean: Vec<f64>, initial_sigma: f64, lambda: Option<usize>) -> Self {
        let n = initial_mean.len();

        // Default population size: 4 + floor(3 * ln(n))
        let lambda = lambda.unwrap_or((4.0 + (3.0 * (n as f64).ln()).floor()) as usize);
        let lambda = lambda.max(4); // Minimum 4

        // Number of parents
        let mu = lambda / 2;

        // Recombination weights (log-linear)
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= weight_sum;
        }

        // Variance effective selection mass
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Strategy parameter settings
        // Time constants for cumulation
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);

        // Learning rates for covariance matrix update
        let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let alpha_mu = 2.0;
        let c_mu = (alpha_mu * (mu_eff - 2.0 + 1.0 / mu_eff))
            / ((n as f64 + 2.0).powi(2) + alpha_mu * mu_eff / 2.0);
        let c_mu = c_mu.min(1.0 - c_1); // Ensure c_1 + c_mu <= 1

        // Damping for step-size
        let d_sigma =
            1.0 + 2.0 * (0.0_f64.max(((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0)) + c_sigma;

        // Expected length of a N(0,I) random vector
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // Initialize covariance matrix to identity
        let covariance: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        // Initialize eigenvalues and eigenvectors (identity)
        let eigenvalues = vec![1.0; n];
        let eigenvectors: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        Self {
            mean: initial_mean.clone(),
            sigma: initial_sigma,
            covariance,
            path_sigma: vec![0.0; n],
            path_c: vec![0.0; n],
            eigenvalues,
            eigenvectors,
            eigen_eval: 0,
            dimension: n,
            lambda,
            mu,
            weights,
            mu_eff,
            c_1,
            c_mu,
            c_sigma,
            d_sigma,
            c_c,
            chi_n,
            generation: 0,
            evaluations: 0,
            best_fitness: f64::INFINITY,
            best_solution: initial_mean,
        }
    }

    /// Sample lambda offspring from the current distribution
    pub fn sample_population<R: Rng>(&self, rng: &mut R) -> Vec<RealVector> {
        let n = self.dimension;
        let normal = StandardNormal;

        (0..self.lambda)
            .map(|_| {
                // Sample z ~ N(0, I)
                let z: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();

                // Transform: y = B * D * z
                let y: Vec<f64> = self.transform_sample(&z);

                // x = m + σ * y
                let genes: Vec<f64> = self
                    .mean
                    .iter()
                    .zip(y.iter())
                    .map(|(&m, &yi)| m + self.sigma * yi)
                    .collect();

                RealVector::new(genes)
            })
            .collect()
    }

    /// Transform a standard normal sample using the covariance structure
    fn transform_sample(&self, z: &[f64]) -> Vec<f64> {
        let n = self.dimension;
        let mut y = vec![0.0; n];

        // y = B * D * z where D = diag(sqrt(eigenvalues))
        for i in 0..n {
            for j in 0..n {
                y[i] += self.eigenvectors[i][j] * self.eigenvalues[j].sqrt() * z[j];
            }
        }

        y
    }

    /// Update the CMA-ES state based on evaluated offspring
    ///
    /// Offspring should be sorted by fitness (best first for minimization).
    pub fn update(&mut self, offspring: &[(RealVector, f64)]) {
        let n = self.dimension;

        // Extract the best μ individuals
        let selected: Vec<&(RealVector, f64)> = offspring.iter().take(self.mu).collect();

        // Calculate weighted mean of selected steps (y values)
        // y_w = Σ w_i * (x_i - m_old) / σ
        let mut y_w = vec![0.0; n];
        for (i, (genome, _fitness)) in selected.iter().enumerate() {
            let genes = genome.genes();
            for j in 0..n {
                y_w[j] += self.weights[i] * (genes[j] - self.mean[j]) / self.sigma;
            }
        }

        // Calculate B * D^-1 * y_w for path updates
        let mut bd_inv_yw = vec![0.0; n];
        {
            // First compute D^-1 * B^T * y_w
            let mut temp = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    temp[i] += self.eigenvectors[j][i] * y_w[j];
                }
                temp[i] /= self.eigenvalues[i].sqrt().max(1e-16);
            }
            // Then compute B * temp
            for i in 0..n {
                for j in 0..n {
                    bd_inv_yw[i] += self.eigenvectors[i][j] * temp[j];
                }
            }
        }

        // Update evolution path for sigma (p_σ)
        let c_sigma_factor = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..n {
            self.path_sigma[i] =
                (1.0 - self.c_sigma) * self.path_sigma[i] + c_sigma_factor * bd_inv_yw[i];
        }

        // Calculate ||p_σ||²
        let path_sigma_norm_sq: f64 = self.path_sigma.iter().map(|x| x * x).sum();
        let path_sigma_norm = path_sigma_norm_sq.sqrt();

        // Heaviside function for stall detection
        let h_sigma = if path_sigma_norm
            / (1.0 - (1.0 - self.c_sigma).powi((2 * (self.generation + 1)) as i32)).sqrt()
            / self.chi_n
            < 1.4 + 2.0 / (n as f64 + 1.0)
        {
            1.0
        } else {
            0.0
        };

        // Update evolution path for C (p_c)
        let c_c_factor = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..n {
            self.path_c[i] = (1.0 - self.c_c) * self.path_c[i] + h_sigma * c_c_factor * y_w[i];
        }

        // Update covariance matrix
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);

        for i in 0..n {
            for j in 0..=i {
                // Decay
                self.covariance[i][j] *= 1.0 - self.c_1 - self.c_mu + delta_h * self.c_1;

                // Rank-1 update
                self.covariance[i][j] += self.c_1 * self.path_c[i] * self.path_c[j];

                // Rank-μ update
                for k in 0..self.mu {
                    let y_k: Vec<f64> = selected[k]
                        .0
                        .genes()
                        .iter()
                        .zip(self.mean.iter())
                        .map(|(&x, &m)| (x - m) / self.sigma)
                        .collect();
                    self.covariance[i][j] += self.c_mu * self.weights[k] * y_k[i] * y_k[j];
                }

                // Symmetry
                if i != j {
                    self.covariance[j][i] = self.covariance[i][j];
                }
            }
        }

        // Update mean
        for i in 0..n {
            self.mean[i] += self.sigma * y_w[i];
        }

        // Update sigma (step-size control)
        self.sigma *= ((self.c_sigma / self.d_sigma) * (path_sigma_norm / self.chi_n - 1.0)).exp();

        // Update generation counter
        self.generation += 1;

        // Update eigendecomposition on the lazy purecmaes cadence (expensive, so
        // don't do it every generation). Best-solution tracking is handled by the
        // caller (`step`) so that it records the *feasible* (bound-repaired)
        // point rather than the raw sample used for the distribution update.
        if self.generation - self.eigen_eval >= self.eigen_update_interval() {
            self.update_eigensystem();
            self.eigen_eval = self.generation;
        }
    }

    /// Number of generations between eigendecomposition recomputes.
    ///
    /// purecmaes recomputes the eigensystem when
    /// `counteval - eigeneval > lambda/(c1+cmu)/N/10`, where `counteval` counts
    /// **evaluations** and advances by `lambda` each generation. In generation
    /// units this is `generation - eigen_eval > 1/(10·N·(c1+cmu))`, i.e. the
    /// interval is `max(1, floor(1/(10·N·(c1+cmu))))`. The previous code compared
    /// a *generation* difference against the *evaluation*-scaled right-hand side,
    /// making recomputes a factor of λ too infrequent (see AUDIT EV-37).
    pub fn eigen_update_interval(&self) -> usize {
        let n = self.dimension as f64;
        ((1.0 / (10.0 * n * (self.c_1 + self.c_mu))).floor() as usize).max(1)
    }

    /// Update eigendecomposition of C
    fn update_eigensystem(&mut self) {
        let n = self.dimension;

        // Force symmetry
        for i in 0..n {
            for j in 0..i {
                self.covariance[i][j] = (self.covariance[i][j] + self.covariance[j][i]) / 2.0;
                self.covariance[j][i] = self.covariance[i][j];
            }
        }

        // Symmetric eigendecomposition via nalgebra. `eigenvalues[j]` is paired
        // with column `j` of `eigenvectors` (matching the B·D·z sampling and
        // C^{-1/2} = B·D^{-1}·Bᵀ conventions used below).
        let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&self.covariance);

        self.eigenvalues = eigenvalues;
        self.eigenvectors = eigenvectors;

        // Ensure eigenvalues are positive (numerical stability)
        for ev in &mut self.eigenvalues {
            *ev = ev.max(1e-16);
        }
    }

    /// Check if algorithm has converged
    pub fn has_converged(&self) -> bool {
        // Check for various convergence criteria
        let max_eigenvalue = self
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_eigenvalue = self
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        // Condition number too large
        if max_eigenvalue / min_eigenvalue.max(1e-16) > 1e14 {
            return true;
        }

        // Sigma too small
        if self.sigma < 1e-16 {
            return true;
        }

        // Sigma times max standard deviation very small
        if self.sigma * max_eigenvalue.sqrt() < 1e-16 {
            return true;
        }

        false
    }
}

/// Symmetric eigendecomposition of a real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvalues[j]` is the
/// eigenvalue associated with **column `j`** of the returned `eigenvectors`
/// matrix (i.e. `eigenvectors[i][j]` is the `i`-th component of the `j`-th
/// eigenvector). Eigenvectors form an orthonormal set, and for the CMA-ES
/// covariance `C` the reconstruction `B · diag(λ) · Bᵀ = C` holds.
///
/// Backed by `nalgebra::SymmetricEigen`, which uses a numerically stable
/// symmetric tridiagonalization + implicit-shift QR iteration rather than the
/// previous hand-rolled cyclic-Jacobi routine (which never mutated its input
/// matrix and therefore returned incorrect — even negative — eigenvalues and
/// non-diagonalizing eigenvectors; see AUDIT EV-01).
fn symmetric_eigendecomposition(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();

    // nalgebra stores column-major; build from a row-major flat copy of `a`.
    let flat: Vec<f64> = (0..n).flat_map(|i| (0..n).map(move |j| a[i][j])).collect();
    let matrix = DMatrix::from_row_slice(n, n, &flat);

    let eig = SymmetricEigen::new(matrix);

    let eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| eig.eigenvectors[(i, j)]).collect())
        .collect();

    (eigenvalues, eigenvectors)
}

/// Implement CmaEsFitness for any Fn that matches the signature
#[cfg(feature = "parallel")]
impl<F> CmaEsFitness for F
where
    F: Fn(&RealVector) -> f64 + Send + Sync,
{
    fn evaluate(&self, x: &RealVector) -> f64 {
        self(x)
    }
}

/// Implement CmaEsFitness for any Fn that matches the signature
#[cfg(not(feature = "parallel"))]
impl<F> CmaEsFitness for F
where
    F: Fn(&RealVector) -> f64,
{
    fn evaluate(&self, x: &RealVector) -> f64 {
        self(x)
    }
}

/// CMA-ES optimizer
#[derive(Clone)]
pub struct CmaEs<F> {
    /// State of the optimizer
    pub state: CmaEsState,
    /// Problem bounds
    pub bounds: Option<MultiBounds>,
    /// Weight of the optional quadratic boundary penalty (0.0 = disabled).
    ///
    /// When positive, an infeasible sample `x` is scored as
    /// `f(clamp(x)) + boundary_penalty · ‖x − clamp(x)‖²`, following Hansen's
    /// reference boundary handling. The distribution (mean/covariance/paths) is
    /// always adapted from the *unrepaired* sample regardless of this weight.
    pub boundary_penalty: f64,
    /// Fitness function marker
    _phantom: PhantomData<F>,
}

impl<F: CmaEsFitness> CmaEs<F> {
    /// Create a new CMA-ES optimizer
    pub fn new(initial_mean: Vec<f64>, initial_sigma: f64) -> Self {
        Self {
            state: CmaEsState::new(initial_mean, initial_sigma, None),
            bounds: None,
            boundary_penalty: 0.0,
            _phantom: PhantomData,
        }
    }

    /// Create with custom population size
    pub fn with_lambda(initial_mean: Vec<f64>, initial_sigma: f64, lambda: usize) -> Self {
        Self {
            state: CmaEsState::new(initial_mean, initial_sigma, Some(lambda)),
            bounds: None,
            boundary_penalty: 0.0,
            _phantom: PhantomData,
        }
    }

    /// Set problem bounds
    pub fn with_bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the quadratic boundary-penalty weight (see [`CmaEs::boundary_penalty`]).
    pub fn with_boundary_penalty(mut self, weight: f64) -> Self {
        self.boundary_penalty = weight;
        self
    }

    /// Run a single generation
    ///
    /// Box constraints are handled per Hansen's recommendation (AUDIT EV-36):
    /// each sample is evaluated at its *repaired* (bound-clamped, and therefore
    /// feasible) position — optionally with a quadratic infeasibility penalty —
    /// but the search distribution (mean, covariance, evolution paths, step
    /// size) is adapted from the **unrepaired** sample `y = (x − m)/σ`. Feeding
    /// repaired points back into the distribution update biases `C` and `m`
    /// toward active bounds; keeping the raw sample avoids that bias.
    pub fn step<R: Rng>(
        &mut self,
        fitness: &F,
        rng: &mut R,
    ) -> EvoResult<Vec<Individual<RealVector>>> {
        // Sample offspring (unrepaired). These raw samples drive the update.
        let unrepaired = self.state.sample_population(rng);

        // For each sample: build the feasible (repaired) point, evaluate fitness
        // there, and add the optional quadratic boundary penalty. Keep the
        // unrepaired sample for the distribution update.
        let mut evaluated: Vec<(RealVector, RealVector, f64)> = unrepaired
            .into_iter()
            .map(|raw| {
                let feasible = match self.bounds {
                    Some(ref bounds) => {
                        let mut repaired = raw.clone();
                        repaired.apply_bounds(bounds);
                        repaired
                    }
                    None => raw.clone(),
                };

                let mut f = fitness.evaluate(&feasible);
                if self.boundary_penalty > 0.0 {
                    let penalty: f64 = raw
                        .genes()
                        .iter()
                        .zip(feasible.genes().iter())
                        .map(|(&x, &c)| {
                            let d = x - c;
                            d * d
                        })
                        .sum();
                    f += self.boundary_penalty * penalty;
                }

                (raw, feasible, f)
            })
            .collect();

        self.state.evaluations += evaluated.len();

        // Sort by fitness (minimization).
        evaluated.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Update the distribution from the UNREPAIRED samples.
        let update_input: Vec<(RealVector, f64)> = evaluated
            .iter()
            .map(|(raw, _feasible, f)| (raw.clone(), *f))
            .collect();
        self.state.update(&update_input);

        // Track the best *feasible* solution.
        if let Some((_, feasible, f)) = evaluated.first() {
            if *f < self.state.best_fitness {
                self.state.best_fitness = *f;
                self.state.best_solution = feasible.genes().to_vec();
            }
        }

        // Convert to individuals using the feasible points.
        let individuals: Vec<Individual<RealVector>> = evaluated
            .into_iter()
            .map(|(_raw, feasible, f)| Individual::with_fitness(feasible, f))
            .collect();

        Ok(individuals)
    }

    /// Run the optimizer for a fixed number of generations
    pub fn run_generations<R: Rng>(
        &mut self,
        fitness: &F,
        max_generations: usize,
        rng: &mut R,
    ) -> EvoResult<Individual<RealVector>> {
        let mut best: Option<Individual<RealVector>> = None;

        for _ in 0..max_generations {
            let population = self.step(fitness, rng)?;

            // Track best
            if let Some(current_best) = population.first() {
                match &best {
                    None => best = Some(current_best.clone()),
                    Some(existing) => {
                        if current_best.fitness_f64() < existing.fitness_f64() {
                            best = Some(current_best.clone());
                        }
                    }
                }
            }

            // Check internal convergence
            if self.state.has_converged() {
                break;
            }
        }

        best.ok_or(EvolutionError::EmptyPopulation)
    }

    /// Run the optimizer until a target fitness is reached or max generations
    pub fn run_until<R: Rng>(
        &mut self,
        fitness: &F,
        target_fitness: f64,
        max_generations: usize,
        rng: &mut R,
    ) -> EvoResult<Individual<RealVector>> {
        let mut best: Option<Individual<RealVector>> = None;

        for _ in 0..max_generations {
            let population = self.step(fitness, rng)?;

            // Track best
            if let Some(current_best) = population.first() {
                match &best {
                    None => best = Some(current_best.clone()),
                    Some(existing) => {
                        if current_best.fitness_f64() < existing.fitness_f64() {
                            best = Some(current_best.clone());
                        }
                    }
                }
            }

            // Check target fitness
            if let Some(ref b) = best {
                if b.fitness_f64() <= target_fitness {
                    break;
                }
            }

            // Check internal convergence
            if self.state.has_converged() {
                break;
            }
        }

        best.ok_or(EvolutionError::EmptyPopulation)
    }

    /// Get the current generation
    pub fn generation(&self) -> usize {
        self.state.generation
    }

    /// Get total evaluations
    pub fn evaluations(&self) -> usize {
        self.state.evaluations
    }

    /// Get current mean
    pub fn mean(&self) -> &[f64] {
        &self.state.mean
    }

    /// Get current sigma
    pub fn sigma(&self) -> f64 {
        self.state.sigma
    }

    /// Get the best solution found
    pub fn best_solution(&self) -> &[f64] {
        &self.state.best_solution
    }

    /// Get the best fitness found
    pub fn best_fitness(&self) -> f64 {
        self.state.best_fitness
    }
}

/// Builder for CMA-ES
pub struct CmaEsBuilder {
    initial_mean: Option<Vec<f64>>,
    initial_sigma: f64,
    lambda: Option<usize>,
    bounds: Option<MultiBounds>,
    boundary_penalty: f64,
}

impl CmaEsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            initial_mean: None,
            initial_sigma: 1.0,
            lambda: None,
            bounds: None,
            boundary_penalty: 0.0,
        }
    }

    /// Set the quadratic boundary-penalty weight (see [`CmaEs::boundary_penalty`]).
    pub fn boundary_penalty(mut self, weight: f64) -> Self {
        self.boundary_penalty = weight;
        self
    }

    /// Set the initial mean
    pub fn mean(mut self, mean: Vec<f64>) -> Self {
        self.initial_mean = Some(mean);
        self
    }

    /// Set the initial step size (sigma)
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.initial_sigma = sigma;
        self
    }

    /// Set the population size
    pub fn lambda(mut self, lambda: usize) -> Self {
        self.lambda = Some(lambda);
        self
    }

    /// Set the bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Build the CMA-ES optimizer
    pub fn build<F: CmaEsFitness>(self) -> EvoResult<CmaEs<F>> {
        let mean = self
            .initial_mean
            .ok_or_else(|| EvolutionError::Configuration("Initial mean not set".to_string()))?;

        let mut cmaes = match self.lambda {
            Some(l) => CmaEs::with_lambda(mean, self.initial_sigma, l),
            None => CmaEs::new(mean, self.initial_sigma),
        };

        if let Some(bounds) = self.bounds {
            cmaes = cmaes.with_bounds(bounds);
        }

        cmaes.boundary_penalty = self.boundary_penalty;

        Ok(cmaes)
    }
}

impl Default for CmaEsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::traits::EvolutionaryGenome;
    use approx::assert_relative_eq;

    // Simple sphere function for testing (minimization: lower is better)
    struct Sphere;

    impl CmaEsFitness for Sphere {
        fn evaluate(&self, genome: &RealVector) -> f64 {
            genome.genes().iter().map(|x| x * x).sum()
        }
    }

    #[test]
    fn test_cmaes_state_initialization() {
        let mean = vec![0.0, 0.0, 0.0];
        let state = CmaEsState::new(mean.clone(), 1.0, None);

        assert_eq!(state.dimension, 3);
        assert_eq!(state.mean, mean);
        assert_eq!(state.sigma, 1.0);
        assert!(state.lambda >= 4);
        assert!(state.mu > 0);
        assert_eq!(state.weights.len(), state.mu);
    }

    #[test]
    fn test_cmaes_weights_sum_to_one() {
        let state = CmaEsState::new(vec![0.0; 10], 1.0, None);
        let sum: f64 = state.weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cmaes_sampling() {
        let mut rng = rand::thread_rng();
        let state = CmaEsState::new(vec![0.0; 5], 1.0, Some(10));

        let samples = state.sample_population(&mut rng);

        assert_eq!(samples.len(), 10);
        for sample in &samples {
            assert_eq!(sample.dimension(), 5);
        }
    }

    #[test]
    fn test_cmaes_step() {
        let mut rng = rand::thread_rng();
        let fitness = Sphere;
        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![5.0, 5.0, 5.0], 2.0);

        let population = cmaes.step(&fitness, &mut rng).unwrap();

        assert_eq!(population.len(), cmaes.state.lambda);
        assert_eq!(cmaes.state.generation, 1);
    }

    #[test]
    fn test_cmaes_optimization() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let fitness = Sphere;
        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![5.0, 5.0], 2.0);

        // Run for more generations to allow convergence
        let result = cmaes.run_generations(&fitness, 150, &mut rng).unwrap();

        // CMA-ES should find solution close to origin
        // Starting from [5,5] (fitness=50), should improve significantly
        let final_fitness = result.fitness_f64();
        let initial_fitness = 50.0; // 5^2 + 5^2
        assert!(
            final_fitness < initial_fitness * 0.7,
            "Final fitness {} should be significantly better than initial {}",
            final_fitness,
            initial_fitness
        );
    }

    #[test]
    fn test_cmaes_with_bounds() {
        let mut rng = rand::thread_rng();
        let fitness = Sphere;
        let bounds = MultiBounds::symmetric(10.0, 3);

        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![5.0, 5.0, 5.0], 2.0).with_bounds(bounds);

        let result = cmaes.run_generations(&fitness, 30, &mut rng).unwrap();

        // Check all genes are within bounds
        for gene in result.genome().genes() {
            assert!(*gene >= -10.0 && *gene <= 10.0);
        }
    }

    #[test]
    fn test_cmaes_builder() {
        let cmaes: CmaEs<Sphere> = CmaEsBuilder::new()
            .mean(vec![0.0, 0.0, 0.0])
            .sigma(0.5)
            .lambda(20)
            .bounds(MultiBounds::symmetric(5.0, 3))
            .build()
            .unwrap();

        assert_eq!(cmaes.state.lambda, 20);
        assert_eq!(cmaes.state.sigma, 0.5);
        assert!(cmaes.bounds.is_some());
    }

    #[test]
    fn test_cmaes_convergence_detection() {
        let mut state = CmaEsState::new(vec![0.0, 0.0], 1e-20, None);
        assert!(state.has_converged());

        state.sigma = 1.0;
        state.eigenvalues = vec![1e15, 1.0];
        assert!(state.has_converged());
    }

    // Rosenbrock (minimization, optimum f=0 at all-ones) for CMA-ES behavioural
    // testing. Defined directly (not via the benchmark type, which negates for
    // maximization) so it matches CMA-ES's lower-is-better convention.
    fn rosenbrock(x: &RealVector) -> f64 {
        x.genes()
            .windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }

    // Build a random symmetric positive-definite matrix C = A·Aᵀ + n·I.
    fn random_spd(n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        let a: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect::<Vec<f64>>()
            })
            .collect();
        let mut c = vec![vec![0.0; n]; n];
        for (i, ci) in c.iter_mut().enumerate() {
            for (j, cij) in ci.iter_mut().enumerate() {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i][k] * a[j][k];
                }
                *cij = s;
            }
        }
        for (i, ci) in c.iter_mut().enumerate() {
            ci[i] += n as f64;
        }
        c
    }

    /// regression: EV-01 / EV-38 — the eigendecomposition must return the true
    /// eigenvalues (all positive for an SPD matrix), eigenvectors that actually
    /// diagonalize C (C·vᵢ = λᵢ·vᵢ), and reconstruct C via B·diag(λ)·Bᵀ. The
    /// pre-fix cyclic-Jacobi routine never mutated its input, so it returned
    /// wrong (even negative) eigenvalues while only preserving the trace — the
    /// old test asserted only Σλ = trace, which the broken routine passed.
    #[test]
    fn test_eigendecomposition_known_matrix() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&a);

        // Eigenvalues of [[4,1],[1,3]] are (7±√5)/2 ≈ 4.618 and 2.382.
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_relative_eq!(sorted[0], (7.0 - 5.0_f64.sqrt()) / 2.0, epsilon = 1e-9);
        assert_relative_eq!(sorted[1], (7.0 + 5.0_f64.sqrt()) / 2.0, epsilon = 1e-9);

        // All eigenvalues of this PD matrix are positive.
        assert!(eigenvalues.iter().all(|&l| l > 0.0));

        // Each (λ_j, column j) is an eigenpair: C·v = λ·v.
        for j in 0..2 {
            let v: Vec<f64> = (0..2).map(|i| eigenvectors[i][j]).collect();
            for i in 0..2 {
                let cv: f64 = (0..2).map(|k| a[i][k] * v[k]).sum();
                assert_relative_eq!(cv, eigenvalues[j] * v[i], epsilon = 1e-9);
            }
        }
    }

    /// regression: EV-01 — on random SPD matrices, every returned pair must
    /// satisfy C·vᵢ = λᵢ·vᵢ to 1e-9, all eigenvalues must be positive, and
    /// B·diag(λ)·Bᵀ must reconstruct C.
    #[test]
    fn test_eigendecomposition_random_spd() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);

        for &n in &[2usize, 3, 5, 8] {
            let c = random_spd(n, &mut rng);
            let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&c);

            // Positivity.
            assert!(
                eigenvalues.iter().all(|&l| l > 0.0),
                "SPD matrix must have all-positive eigenvalues, got {:?}",
                eigenvalues
            );

            // Eigenpair equation C·vⱼ = λⱼ·vⱼ.
            for j in 0..n {
                let v: Vec<f64> = (0..n).map(|i| eigenvectors[i][j]).collect();
                for i in 0..n {
                    let cv: f64 = (0..n).map(|k| c[i][k] * v[k]).sum();
                    assert!(
                        (cv - eigenvalues[j] * v[i]).abs() < 1e-9,
                        "n={}: C·v_{} component {} mismatch",
                        n,
                        j,
                        i
                    );
                }
            }

            // Reconstruction B·diag(λ)·Bᵀ = C.
            for i in 0..n {
                for j in 0..n {
                    let recon: f64 = (0..n)
                        .map(|k| eigenvectors[i][k] * eigenvalues[k] * eigenvectors[j][k])
                        .sum();
                    assert!(
                        (recon - c[i][j]).abs() < 1e-9,
                        "n={}: reconstruction mismatch at ({},{})",
                        n,
                        i,
                        j
                    );
                }
            }
        }
    }

    /// regression: EV-01 (behavioural) — with a correct eigendecomposition,
    /// CMA-ES solves the 5-D Rosenbrock function to f < 1e-6 within a generous
    /// seeded budget. The pre-fix garbage eigensystem corrupted sampling and
    /// C^{-1/2}, so it could not converge.
    #[test]
    fn test_cmaes_rosenbrock_convergence() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        // Slightly larger λ improves robustness on the Rosenbrock valley.
        let mut cmaes: CmaEs<_> = CmaEs::with_lambda(vec![0.0; 5], 0.5, 20);
        let best = cmaes.run_until(&rosenbrock, 1e-6, 4000, &mut rng).unwrap();

        assert!(
            best.fitness_f64() < 1e-6,
            "CMA-ES should reach f < 1e-6 on 5-D Rosenbrock, got {}",
            best.fitness_f64()
        );
    }

    /// regression: EV-37 — for large n the correct per-generation cadence
    /// recomputes the eigensystem (nearly) every generation, whereas the pre-fix
    /// λ-too-infrequent cadence recomputed only a handful of times.
    #[test]
    fn test_eigen_recompute_cadence() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let fitness = Sphere;
        let n = 100;
        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![1.0; n], 1.0);

        // For n=100 the correct interval is 1 generation.
        assert_eq!(cmaes.state.eigen_update_interval(), 1);

        let gens = 30;
        let mut recomputes = 0;
        let mut last = cmaes.state.eigen_eval;
        for _ in 0..gens {
            cmaes.step(&fitness, &mut rng).unwrap();
            if cmaes.state.eigen_eval != last {
                recomputes += 1;
                last = cmaes.state.eigen_eval;
            }
        }

        assert!(
            recomputes >= gens - 1,
            "expected ~every-generation eigen recompute for n=100, got {} in {} gens",
            recomputes,
            gens
        );
    }

    /// regression: EV-36 — the distribution update must be adapted from the
    /// UNREPAIRED samples. With bounds sitting far above the sampling region,
    /// every sample clamps up to the lower bound, so a clamped-update (pre-fix)
    /// would slam the mean straight onto the boundary (~2.9) in a single step,
    /// while an unrepaired update keeps the mean near the sampled region.
    #[test]
    fn test_cmaes_update_uses_unrepaired_samples() {
        use crate::genome::bounds::Bounds;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let fitness = Sphere;

        let bounds = MultiBounds::uniform(Bounds::new(2.9, 3.1), 2);
        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![0.0, 0.0], 1.0).with_bounds(bounds);

        cmaes.step(&fitness, &mut rng).unwrap();

        for &m in &cmaes.state.mean {
            assert!(
                m.abs() < 2.0,
                "mean component {} was pulled onto the clamped boundary (~2.9); \
                 the distribution update must use unrepaired samples",
                m
            );
        }

        // The best solution reported must still be feasible (within bounds).
        for &g in &cmaes.state.best_solution {
            assert!((2.9..=3.1).contains(&g));
        }
    }

    /// The optional quadratic boundary penalty adds a positive infeasibility
    /// cost while leaving the feasible-region behaviour unchanged when disabled.
    #[test]
    fn test_cmaes_boundary_penalty_option() {
        use crate::genome::bounds::Bounds;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let fitness = Sphere;

        let bounds = MultiBounds::uniform(Bounds::new(-1.0, 1.0), 3);
        let mut cmaes: CmaEs<Sphere> = CmaEs::new(vec![0.0; 3], 2.0)
            .with_bounds(bounds)
            .with_boundary_penalty(10.0);
        assert_eq!(cmaes.boundary_penalty, 10.0);

        // Should run without panicking and keep the reported best feasible.
        cmaes.step(&fitness, &mut rng).unwrap();
        for &g in &cmaes.state.best_solution {
            assert!((-1.0..=1.0).contains(&g));
        }
    }
}
