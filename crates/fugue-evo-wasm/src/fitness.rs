//! Fitness function support for WASM

use fugue_evo::algorithms::cmaes::CmaEsFitness;
use fugue_evo::fitness::benchmarks::{Rastrigin, Rosenbrock, Sphere};
use fugue_evo::fitness::traits::Fitness;
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::RealValuedGenome;
use wasm_bindgen::prelude::*;

/// Built-in fitness function types
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum FitnessType {
    /// Sphere function (unimodal, easy)
    Sphere,
    /// Rastrigin function (highly multimodal)
    Rastrigin,
    /// Rosenbrock function (narrow valley)
    Rosenbrock,
}

/// Fitness function wrapper that can hold different function types
pub enum FitnessWrapper {
    Sphere(Sphere),
    Rastrigin(Rastrigin),
    Rosenbrock(Rosenbrock),
    Custom(Box<dyn Fn(&[f64]) -> f64 + Send + Sync>),
}

impl FitnessWrapper {
    /// Create a fitness wrapper from a type and dimension
    pub fn from_type(fitness_type: FitnessType, dimension: usize) -> Self {
        match fitness_type {
            FitnessType::Sphere => Self::Sphere(Sphere::new(dimension)),
            FitnessType::Rastrigin => Self::Rastrigin(Rastrigin::new(dimension)),
            FitnessType::Rosenbrock => Self::Rosenbrock(Rosenbrock::new(dimension)),
        }
    }

    /// Create a fitness wrapper from a function name
    pub fn from_name(name: &str, dimension: usize) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sphere" => Some(Self::Sphere(Sphere::new(dimension))),
            "rastrigin" => Some(Self::Rastrigin(Rastrigin::new(dimension))),
            "rosenbrock" => Some(Self::Rosenbrock(Rosenbrock::new(dimension))),
            _ => None,
        }
    }

    /// Evaluate a genome
    pub fn evaluate(&self, genome: &RealVector) -> f64 {
        match self {
            Self::Sphere(f) => {
                // Sphere is minimization, negate for maximization
                -f.evaluate(genome)
            }
            Self::Rastrigin(f) => {
                // Rastrigin is minimization, negate for maximization
                -f.evaluate(genome)
            }
            Self::Rosenbrock(f) => {
                // Rosenbrock is minimization, negate for maximization
                -f.evaluate(genome)
            }
            Self::Custom(f) => f(genome.genes()),
        }
    }
}

/// A wrapper that implements Fitness trait for use with the GA
pub struct FitnessEvaluator {
    wrapper: FitnessWrapper,
}

impl FitnessEvaluator {
    pub fn new(wrapper: FitnessWrapper) -> Self {
        Self { wrapper }
    }
}

impl Fitness for FitnessEvaluator {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        self.wrapper.evaluate(genome)
    }
}

/// Get a list of available built-in fitness functions
#[wasm_bindgen(js_name = getAvailableFitnessFunctions)]
pub fn get_available_fitness_functions() -> Vec<String> {
    vec![
        "sphere".to_string(),
        "rastrigin".to_string(),
        "rosenbrock".to_string(),
    ]
}

/// CMA-ES fitness wrapper (CMA-ES minimizes, so we negate)
pub struct CmaEsFitnessWrapper {
    wrapper: FitnessWrapper,
}

impl CmaEsFitnessWrapper {
    pub fn new(wrapper: FitnessWrapper) -> Self {
        Self { wrapper }
    }
}

impl CmaEsFitness for CmaEsFitnessWrapper {
    fn evaluate(&self, x: &RealVector) -> f64 {
        // CMA-ES minimizes, our FitnessWrapper negates for maximization
        // So we negate back to get minimization values
        -self.wrapper.evaluate(x)
    }
}
