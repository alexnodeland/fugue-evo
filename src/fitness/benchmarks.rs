//! Benchmark fitness functions
//!
//! This module provides standard benchmark functions for testing evolutionary algorithms.

use std::f64::consts::PI;

use crate::fitness::traits::Fitness;
use crate::genome::bit_string::BitString;
use crate::genome::real_vector::RealVector;
use crate::genome::traits::{BinaryGenome, RealValuedGenome};

/// Trait for benchmark functions
pub trait BenchmarkFunction: Send + Sync {
    /// Name of the benchmark function
    fn name(&self) -> &'static str;

    /// Dimensionality of the problem
    fn dimension(&self) -> usize;

    /// Search space bounds (min, max)
    fn bounds(&self) -> (f64, f64);

    /// Optimal (minimum) fitness value
    fn optimal_fitness(&self) -> f64;

    /// Optimal solution (if known)
    fn optimal_solution(&self) -> Option<Vec<f64>>;

    /// Evaluate the function (returns value to be MINIMIZED)
    fn evaluate_raw(&self, x: &[f64]) -> f64;
}

/// Sphere function: f(x) = Σxᵢ²
///
/// Unimodal, convex, separable. Optimum at origin.
#[derive(Clone, Debug)]
pub struct Sphere {
    dimension: usize,
}

impl Sphere {
    /// Create a new Sphere function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for Sphere {
    fn name(&self) -> &'static str {
        "Sphere"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-5.12, 5.12)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
}

impl Fitness for Sphere {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        // Negate for maximization (GA convention)
        -self.evaluate_raw(genome.genes())
    }
}

/// Rastrigin function: f(x) = 10n + Σ(xᵢ² - 10cos(2πxᵢ))
///
/// Highly multimodal with many local minima. Optimum at origin.
#[derive(Clone, Debug)]
pub struct Rastrigin {
    dimension: usize,
}

impl Rastrigin {
    /// Create a new Rastrigin function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for Rastrigin {
    fn name(&self) -> &'static str {
        "Rastrigin"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-5.12, 5.12)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|xi| xi * xi - a * (2.0 * PI * xi).cos())
                .sum::<f64>()
    }
}

impl Fitness for Rastrigin {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Rosenbrock function: f(x) = Σ[100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²]
///
/// Valley structure, non-separable. Optimum at (1,1,...,1).
#[derive(Clone, Debug)]
pub struct Rosenbrock {
    dimension: usize,
}

impl Rosenbrock {
    /// Create a new Rosenbrock function
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 2, "Rosenbrock requires at least 2 dimensions");
        Self { dimension }
    }
}

impl BenchmarkFunction for Rosenbrock {
    fn name(&self) -> &'static str {
        "Rosenbrock"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-5.0, 10.0)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| {
                let xi = w[0];
                let xi1 = w[1];
                100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2)
            })
            .sum()
    }
}

impl Fitness for Rosenbrock {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Ackley function
///
/// Nearly flat outer region with many local minima. Optimum at origin.
#[derive(Clone, Debug)]
pub struct Ackley {
    dimension: usize,
    a: f64,
    b: f64,
    c: f64,
}

impl Ackley {
    /// Create a new Ackley function with default parameters
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            a: 20.0,
            b: 0.2,
            c: 2.0 * PI,
        }
    }

    /// Create with custom parameters
    pub fn with_params(dimension: usize, a: f64, b: f64, c: f64) -> Self {
        Self { dimension, a, b, c }
    }
}

impl BenchmarkFunction for Ackley {
    fn name(&self) -> &'static str {
        "Ackley"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-32.768, 32.768)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (self.c * xi).cos()).sum::<f64>();

        -self.a * (-self.b * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + self.a
            + std::f64::consts::E
    }
}

impl Fitness for Ackley {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Griewank function: f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1
///
/// Many local minima. Optimum at origin.
#[derive(Clone, Debug)]
pub struct Griewank {
    dimension: usize,
}

impl Griewank {
    /// Create a new Griewank function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for Griewank {
    fn name(&self) -> &'static str {
        "Griewank"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-600.0, 600.0)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
        let prod_cos: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product();
        sum_sq - prod_cos + 1.0
    }
}

impl Fitness for Griewank {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Schwefel function
///
/// Deceptive - global optimum far from local optima.
#[derive(Clone, Debug)]
pub struct Schwefel {
    dimension: usize,
}

impl Schwefel {
    /// Create a new Schwefel function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for Schwefel {
    fn name(&self) -> &'static str {
        "Schwefel"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-500.0, 500.0)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![420.9687; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        418.9829 * n - x.iter().map(|xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
    }
}

impl Fitness for Schwefel {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// OneMax function for bit strings
///
/// Counts the number of 1s in the bit string. Optimum when all bits are 1.
#[derive(Clone, Debug)]
pub struct OneMax {
    length: usize,
}

impl OneMax {
    /// Create a new OneMax function
    pub fn new(length: usize) -> Self {
        Self { length }
    }

    /// Get the length
    pub fn length(&self) -> usize {
        self.length
    }
}

impl Fitness for OneMax {
    type Genome = BitString;
    type Value = usize;

    fn evaluate(&self, genome: &Self::Genome) -> usize {
        genome.count_ones()
    }
}

/// LeadingOnes function for bit strings
///
/// Counts the number of leading 1s before the first 0.
#[derive(Clone, Debug)]
pub struct LeadingOnes {
    #[allow(dead_code)]
    length: usize,
}

impl LeadingOnes {
    /// Create a new LeadingOnes function
    pub fn new(length: usize) -> Self {
        Self { length }
    }
}

impl Fitness for LeadingOnes {
    type Genome = BitString;
    type Value = usize;

    fn evaluate(&self, genome: &Self::Genome) -> usize {
        genome.bits().iter().take_while(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::traits::Fitness;
    use approx::assert_relative_eq;

    // Sphere function tests
    #[test]
    fn test_sphere_at_optimum() {
        let sphere = Sphere::new(3);
        let optimum = RealVector::new(vec![0.0, 0.0, 0.0]);
        assert_relative_eq!(sphere.evaluate(&optimum), 0.0);
    }

    #[test]
    fn test_sphere_non_optimum() {
        let sphere = Sphere::new(3);
        let point = RealVector::new(vec![1.0, 2.0, 3.0]);
        // 1 + 4 + 9 = 14, negated = -14
        assert_relative_eq!(sphere.evaluate(&point), -14.0);
    }

    #[test]
    fn test_sphere_metadata() {
        let sphere = Sphere::new(5);
        assert_eq!(sphere.name(), "Sphere");
        assert_eq!(sphere.dimension(), 5);
        assert_eq!(sphere.bounds(), (-5.12, 5.12));
        assert_relative_eq!(sphere.optimal_fitness(), 0.0);
        assert_eq!(sphere.optimal_solution(), Some(vec![0.0; 5]));
    }

    // Rastrigin function tests
    #[test]
    fn test_rastrigin_at_optimum() {
        let rastrigin = Rastrigin::new(3);
        let optimum = RealVector::new(vec![0.0, 0.0, 0.0]);
        assert_relative_eq!(rastrigin.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rastrigin_non_optimum() {
        let rastrigin = Rastrigin::new(2);
        let point = RealVector::new(vec![1.0, 1.0]);
        // At x=1, cos(2π*1) = 1, so each term is 1 - 10*1 = -9
        // Total = 10*2 + 2*(-9) = 20 - 18 = 2, but we need to calculate more precisely
        let expected = 10.0 * 2.0
            + (1.0 * 1.0 - 10.0 * (2.0 * PI * 1.0).cos())
            + (1.0 * 1.0 - 10.0 * (2.0 * PI * 1.0).cos());
        assert_relative_eq!(rastrigin.evaluate(&point), -expected, epsilon = 1e-10);
    }

    #[test]
    fn test_rastrigin_metadata() {
        let rastrigin = Rastrigin::new(10);
        assert_eq!(rastrigin.name(), "Rastrigin");
        assert_eq!(rastrigin.dimension(), 10);
        assert_eq!(rastrigin.bounds(), (-5.12, 5.12));
    }

    // Rosenbrock function tests
    #[test]
    fn test_rosenbrock_at_optimum() {
        let rosenbrock = Rosenbrock::new(3);
        let optimum = RealVector::new(vec![1.0, 1.0, 1.0]);
        assert_relative_eq!(rosenbrock.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_non_optimum() {
        let rosenbrock = Rosenbrock::new(2);
        let point = RealVector::new(vec![0.0, 0.0]);
        // 100*(0 - 0)^2 + (1 - 0)^2 = 1
        assert_relative_eq!(rosenbrock.evaluate(&point), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rosenbrock_metadata() {
        let rosenbrock = Rosenbrock::new(5);
        assert_eq!(rosenbrock.name(), "Rosenbrock");
        assert_eq!(rosenbrock.dimension(), 5);
        assert_eq!(rosenbrock.bounds(), (-5.0, 10.0));
        assert_eq!(rosenbrock.optimal_solution(), Some(vec![1.0; 5]));
    }

    // Ackley function tests
    #[test]
    fn test_ackley_at_optimum() {
        let ackley = Ackley::new(3);
        let optimum = RealVector::new(vec![0.0, 0.0, 0.0]);
        assert_relative_eq!(ackley.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ackley_metadata() {
        let ackley = Ackley::new(10);
        assert_eq!(ackley.name(), "Ackley");
        assert_eq!(ackley.dimension(), 10);
        assert_eq!(ackley.bounds(), (-32.768, 32.768));
    }

    // Griewank function tests
    #[test]
    fn test_griewank_at_optimum() {
        let griewank = Griewank::new(3);
        let optimum = RealVector::new(vec![0.0, 0.0, 0.0]);
        assert_relative_eq!(griewank.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_griewank_metadata() {
        let griewank = Griewank::new(10);
        assert_eq!(griewank.name(), "Griewank");
        assert_eq!(griewank.dimension(), 10);
        assert_eq!(griewank.bounds(), (-600.0, 600.0));
    }

    // Schwefel function tests
    #[test]
    fn test_schwefel_metadata() {
        let schwefel = Schwefel::new(10);
        assert_eq!(schwefel.name(), "Schwefel");
        assert_eq!(schwefel.dimension(), 10);
        assert_eq!(schwefel.bounds(), (-500.0, 500.0));
    }

    // OneMax function tests
    #[test]
    fn test_onemax_all_ones() {
        let onemax = OneMax::new(10);
        let genome = BitString::ones(10);
        assert_eq!(onemax.evaluate(&genome), 10);
    }

    #[test]
    fn test_onemax_all_zeros() {
        let onemax = OneMax::new(10);
        let genome = BitString::zeros(10);
        assert_eq!(onemax.evaluate(&genome), 0);
    }

    #[test]
    fn test_onemax_mixed() {
        let onemax = OneMax::new(5);
        let genome = BitString::new(vec![true, false, true, false, true]);
        assert_eq!(onemax.evaluate(&genome), 3);
    }

    // LeadingOnes function tests
    #[test]
    fn test_leadingones_all_ones() {
        let lo = LeadingOnes::new(10);
        let genome = BitString::ones(10);
        assert_eq!(lo.evaluate(&genome), 10);
    }

    #[test]
    fn test_leadingones_all_zeros() {
        let lo = LeadingOnes::new(10);
        let genome = BitString::zeros(10);
        assert_eq!(lo.evaluate(&genome), 0);
    }

    #[test]
    fn test_leadingones_mixed() {
        let lo = LeadingOnes::new(5);
        let genome = BitString::new(vec![true, true, false, true, true]);
        assert_eq!(lo.evaluate(&genome), 2); // First 2 are 1s, then a 0
    }

    #[test]
    fn test_leadingones_starts_with_zero() {
        let lo = LeadingOnes::new(5);
        let genome = BitString::new(vec![false, true, true, true, true]);
        assert_eq!(lo.evaluate(&genome), 0);
    }
}
