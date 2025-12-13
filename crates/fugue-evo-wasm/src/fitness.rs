//! Fitness function support for WASM

use fugue_evo::algorithms::cmaes::CmaEsFitness;
use fugue_evo::fitness::benchmarks::{
    Ackley, DixonPrice, Griewank, Levy, Rastrigin, Rosenbrock, Schwefel, Sphere, StyblinskiTang,
};
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
    /// Ackley function (nearly flat outer region)
    Ackley,
    /// Griewank function (many local minima)
    Griewank,
    /// Schwefel function (deceptive)
    Schwefel,
    /// Levy function (multimodal)
    Levy,
    /// Dixon-Price function (valley structure)
    DixonPrice,
    /// Styblinski-Tang function (multimodal)
    StyblinskiTang,
}

/// Fitness function wrapper that can hold different function types
#[allow(clippy::type_complexity)]
pub enum FitnessWrapper {
    Sphere(Sphere),
    Rastrigin(Rastrigin),
    Rosenbrock(Rosenbrock),
    Ackley(Ackley),
    Griewank(Griewank),
    Schwefel(Schwefel),
    Levy(Levy),
    DixonPrice(DixonPrice),
    StyblinskiTang(StyblinskiTang),
    Custom(Box<dyn Fn(&[f64]) -> f64 + Send + Sync>),
}

impl FitnessWrapper {
    /// Create a fitness wrapper from a type and dimension
    pub fn from_type(fitness_type: FitnessType, dimension: usize) -> Self {
        match fitness_type {
            FitnessType::Sphere => Self::Sphere(Sphere::new(dimension)),
            FitnessType::Rastrigin => Self::Rastrigin(Rastrigin::new(dimension)),
            FitnessType::Rosenbrock => Self::Rosenbrock(Rosenbrock::new(dimension.max(2))),
            FitnessType::Ackley => Self::Ackley(Ackley::new(dimension)),
            FitnessType::Griewank => Self::Griewank(Griewank::new(dimension)),
            FitnessType::Schwefel => Self::Schwefel(Schwefel::new(dimension)),
            FitnessType::Levy => Self::Levy(Levy::new(dimension)),
            FitnessType::DixonPrice => Self::DixonPrice(DixonPrice::new(dimension)),
            FitnessType::StyblinskiTang => Self::StyblinskiTang(StyblinskiTang::new(dimension)),
        }
    }

    /// Create a fitness wrapper from a function name
    pub fn from_name(name: &str, dimension: usize) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sphere" => Some(Self::Sphere(Sphere::new(dimension))),
            "rastrigin" => Some(Self::Rastrigin(Rastrigin::new(dimension))),
            "rosenbrock" => Some(Self::Rosenbrock(Rosenbrock::new(dimension.max(2)))),
            "ackley" => Some(Self::Ackley(Ackley::new(dimension))),
            "griewank" => Some(Self::Griewank(Griewank::new(dimension))),
            "schwefel" => Some(Self::Schwefel(Schwefel::new(dimension))),
            "levy" => Some(Self::Levy(Levy::new(dimension))),
            "dixon-price" | "dixonprice" | "dixon_price" => {
                Some(Self::DixonPrice(DixonPrice::new(dimension)))
            }
            "styblinski-tang" | "styblinskitang" | "styblinski_tang" => {
                Some(Self::StyblinskiTang(StyblinskiTang::new(dimension)))
            }
            _ => None,
        }
    }

    /// Evaluate a genome
    pub fn evaluate(&self, genome: &RealVector) -> f64 {
        match self {
            Self::Sphere(f) => -f.evaluate(genome),
            Self::Rastrigin(f) => -f.evaluate(genome),
            Self::Rosenbrock(f) => -f.evaluate(genome),
            Self::Ackley(f) => -f.evaluate(genome),
            Self::Griewank(f) => -f.evaluate(genome),
            Self::Schwefel(f) => -f.evaluate(genome),
            Self::Levy(f) => -f.evaluate(genome),
            Self::DixonPrice(f) => -f.evaluate(genome),
            Self::StyblinskiTang(f) => -f.evaluate(genome),
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
        "ackley".to_string(),
        "griewank".to_string(),
        "schwefel".to_string(),
        "levy".to_string(),
        "dixon-price".to_string(),
        "styblinski-tang".to_string(),
    ]
}

/// Get information about a fitness function
#[wasm_bindgen(js_name = getFitnessInfo)]
pub fn get_fitness_info(name: &str) -> Result<String, JsValue> {
    let info = match name.to_lowercase().as_str() {
        "sphere" => {
            r#"{"name":"Sphere","bounds":[-5.12,5.12],"optimum":0,"description":"Unimodal, convex, separable. Optimum at origin."}"#
        }
        "rastrigin" => {
            r#"{"name":"Rastrigin","bounds":[-5.12,5.12],"optimum":0,"description":"Highly multimodal with many local minima. Optimum at origin."}"#
        }
        "rosenbrock" => {
            r#"{"name":"Rosenbrock","bounds":[-5,10],"optimum":0,"description":"Valley structure, non-separable. Optimum at (1,1,...,1)."}"#
        }
        "ackley" => {
            r#"{"name":"Ackley","bounds":[-32.768,32.768],"optimum":0,"description":"Nearly flat outer region with many local minima. Optimum at origin."}"#
        }
        "griewank" => {
            r#"{"name":"Griewank","bounds":[-600,600],"optimum":0,"description":"Many local minima. Optimum at origin."}"#
        }
        "schwefel" => {
            r#"{"name":"Schwefel","bounds":[-500,500],"optimum":0,"description":"Deceptive - global optimum far from local optima."}"#
        }
        "levy" => {
            r#"{"name":"Levy","bounds":[-10,10],"optimum":0,"description":"Multimodal with many local minima. Optimum at (1,1,...,1)."}"#
        }
        "dixon-price" => {
            r#"{"name":"Dixon-Price","bounds":[-10,10],"optimum":0,"description":"Valley structure."}"#
        }
        "styblinski-tang" => {
            r#"{"name":"Styblinski-Tang","bounds":[-5,5],"optimum":"varies","description":"Multimodal. Optimum at (-2.903534,...)."}"#
        }
        _ => {
            return Err(JsValue::from_str(&format!(
                "Unknown fitness function: {}",
                name
            )))
        }
    };
    Ok(info.to_string())
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
