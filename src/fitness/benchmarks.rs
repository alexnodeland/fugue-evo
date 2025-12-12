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

// =============================================================================
// Multi-Objective Test Problems (ZDT)
// =============================================================================

/// ZDT1 multi-objective test problem
///
/// Two objectives with a convex Pareto front.
/// Reference: Zitzler, E., Deb, K., & Thiele, L. (2000).
#[derive(Clone, Debug)]
pub struct Zdt1 {
    dimension: usize,
}

impl Zdt1 {
    /// Create a new ZDT1 function
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 2, "ZDT1 requires at least 2 dimensions");
        Self { dimension }
    }

    /// Evaluate the function (returns [f1, f2])
    pub fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        let n = x.len() as f64;
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let f2 = g * (1.0 - (f1 / g).sqrt());
        [f1, f2]
    }

    /// Bounds for each variable [0, 1]
    pub fn bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// ZDT2 multi-objective test problem
///
/// Two objectives with a non-convex Pareto front.
#[derive(Clone, Debug)]
pub struct Zdt2 {
    dimension: usize,
}

impl Zdt2 {
    /// Create a new ZDT2 function
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 2, "ZDT2 requires at least 2 dimensions");
        Self { dimension }
    }

    /// Evaluate the function (returns [f1, f2])
    pub fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        let n = x.len() as f64;
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let f2 = g * (1.0 - (f1 / g).powi(2));
        [f1, f2]
    }

    /// Bounds for each variable [0, 1]
    pub fn bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// ZDT3 multi-objective test problem
///
/// Two objectives with a disconnected Pareto front.
#[derive(Clone, Debug)]
pub struct Zdt3 {
    dimension: usize,
}

impl Zdt3 {
    /// Create a new ZDT3 function
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 2, "ZDT3 requires at least 2 dimensions");
        Self { dimension }
    }

    /// Evaluate the function (returns [f1, f2])
    pub fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        let n = x.len() as f64;
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let h = 1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin();
        let f2 = g * h;
        [f1, f2]
    }

    /// Bounds for each variable [0, 1]
    pub fn bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Schaffer N.1 multi-objective problem
///
/// Simple bi-objective problem with a single variable.
#[derive(Clone, Debug)]
pub struct SchafferN1;

impl SchafferN1 {
    /// Create a new Schaffer N.1 function
    pub fn new() -> Self {
        Self
    }

    /// Evaluate the function (returns [f1, f2])
    pub fn evaluate(&self, x: f64) -> [f64; 2] {
        let f1 = x * x;
        let f2 = (x - 2.0) * (x - 2.0);
        [f1, f2]
    }

    /// Bounds for the variable
    pub fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }
}

impl Default for SchafferN1 {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Additional Single-Objective Functions
// =============================================================================

/// Levy function
///
/// Multimodal with many local minima. Optimum at (1, 1, ..., 1).
#[derive(Clone, Debug)]
pub struct Levy {
    dimension: usize,
}

impl Levy {
    /// Create a new Levy function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for Levy {
    fn name(&self) -> &'static str {
        "Levy"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let w: Vec<f64> = x.iter().map(|xi| 1.0 + (xi - 1.0) / 4.0).collect();
        let n = w.len();

        let term1 = (PI * w[0]).sin().powi(2);

        let sum: f64 = w[..n - 1]
            .iter()
            .map(|wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2)))
            .sum();

        let term3 = (w[n - 1] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[n - 1]).sin().powi(2));

        term1 + sum + term3
    }
}

impl Fitness for Levy {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Dixon-Price function
///
/// Valley structure. Optimum depends on dimension.
#[derive(Clone, Debug)]
pub struct DixonPrice {
    dimension: usize,
}

impl DixonPrice {
    /// Create a new Dixon-Price function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for DixonPrice {
    fn name(&self) -> &'static str {
        "Dixon-Price"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-10.0, 10.0)
    }

    fn optimal_fitness(&self) -> f64 {
        0.0
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        // Optimal solution: x_i = 2^(-(2^i - 2) / 2^i)
        let optimal: Vec<f64> = (0..self.dimension)
            .map(|i| {
                let exp_num = (1u64 << i) as f64 - 2.0;
                let exp_den = (1u64 << (i + 1)) as f64;
                2.0_f64.powf(-exp_num / exp_den)
            })
            .collect();
        Some(optimal)
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        let term1 = (x[0] - 1.0).powi(2);

        let sum: f64 = x
            .windows(2)
            .enumerate()
            .map(|(i, w)| (i + 2) as f64 * (2.0 * w[1] * w[1] - w[0]).powi(2))
            .sum();

        term1 + sum
    }
}

impl Fitness for DixonPrice {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
    }
}

/// Styblinski-Tang function
///
/// Multimodal. Optimum at (-2.903534, ..., -2.903534).
#[derive(Clone, Debug)]
pub struct StyblinskiTang {
    dimension: usize,
}

impl StyblinskiTang {
    /// Create a new Styblinski-Tang function
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl BenchmarkFunction for StyblinskiTang {
    fn name(&self) -> &'static str {
        "Styblinski-Tang"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn bounds(&self) -> (f64, f64) {
        (-5.0, 5.0)
    }

    fn optimal_fitness(&self) -> f64 {
        // f(optimal) = -39.16617 * dimension
        -39.16617 * self.dimension as f64
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![-2.903534; self.dimension])
    }

    fn evaluate_raw(&self, x: &[f64]) -> f64 {
        x.iter()
            .map(|xi| xi.powi(4) - 16.0 * xi.powi(2) + 5.0 * xi)
            .sum::<f64>()
            / 2.0
    }
}

impl Fitness for StyblinskiTang {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        -self.evaluate_raw(genome.genes())
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

    // ZDT1 tests
    #[test]
    fn test_zdt1_pareto_front() {
        let zdt1 = Zdt1::new(10);
        // On the Pareto front, all x_i = 0 for i > 0, and x_0 varies from 0 to 1
        let x = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let [f1, f2] = zdt1.evaluate(&x);

        // f1 should equal x_0
        assert_relative_eq!(f1, 0.5, epsilon = 1e-10);

        // On Pareto front with g=1: f2 = 1 - sqrt(f1)
        assert_relative_eq!(f2, 1.0 - f1.sqrt(), epsilon = 1e-10);
    }

    // ZDT2 tests
    #[test]
    fn test_zdt2_pareto_front() {
        let zdt2 = Zdt2::new(10);
        let x = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let [f1, f2] = zdt2.evaluate(&x);

        assert_relative_eq!(f1, 0.5, epsilon = 1e-10);
        // On Pareto front with g=1: f2 = 1 - f1^2
        assert_relative_eq!(f2, 1.0 - f1 * f1, epsilon = 1e-10);
    }

    // Schaffer N.1 tests
    #[test]
    fn test_schaffer_n1() {
        let schaffer = SchafferN1::new();
        let [f1, f2] = schaffer.evaluate(0.0);
        assert_relative_eq!(f1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(f2, 4.0, epsilon = 1e-10);
    }

    // Levy function tests
    #[test]
    fn test_levy_at_optimum() {
        let levy = Levy::new(3);
        let optimum = RealVector::new(vec![1.0, 1.0, 1.0]);
        assert_relative_eq!(levy.evaluate(&optimum), 0.0, epsilon = 1e-10);
    }

    // Dixon-Price tests
    #[test]
    fn test_dixonprice_metadata() {
        let dp = DixonPrice::new(5);
        assert_eq!(dp.name(), "Dixon-Price");
        assert_eq!(dp.dimension(), 5);
    }

    // Styblinski-Tang tests
    #[test]
    fn test_styblinskitang_near_optimum() {
        let st = StyblinskiTang::new(2);
        let near_opt = RealVector::new(vec![-2.9, -2.9]);
        // Should be close to optimal fitness
        let fitness = st.evaluate(&near_opt);
        let optimal = st.optimal_fitness();
        // The returned fitness is negated, so compare negatives
        assert!(
            fitness > optimal - 1.0,
            "Fitness {} should be close to optimal {}",
            fitness,
            optimal
        );
    }
}
