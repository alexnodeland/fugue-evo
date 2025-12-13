//! Result types for WASM optimizers

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Result of an optimization run
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct OptimizationResult {
    /// Best genome found
    best_genome: Vec<f64>,
    /// Fitness of the best genome
    best_fitness: f64,
    /// Number of generations completed
    generations: usize,
    /// Total fitness evaluations
    evaluations: usize,
    /// Fitness history (best per generation)
    fitness_history: Vec<f64>,
}

#[wasm_bindgen]
impl OptimizationResult {
    /// Get the best genome as a Float64Array
    #[wasm_bindgen(js_name = bestGenome)]
    pub fn best_genome(&self) -> Vec<f64> {
        self.best_genome.clone()
    }

    /// Get the best fitness value
    #[wasm_bindgen(js_name = bestFitness, getter)]
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Get the number of generations completed
    #[wasm_bindgen(getter)]
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// Get the total number of fitness evaluations
    #[wasm_bindgen(getter)]
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Get the fitness history
    #[wasm_bindgen(js_name = fitnessHistory)]
    pub fn fitness_history(&self) -> Vec<f64> {
        self.fitness_history.clone()
    }

    /// Serialize to JSON
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl OptimizationResult {
    /// Create a new optimization result
    pub fn new(
        best_genome: Vec<f64>,
        best_fitness: f64,
        generations: usize,
        evaluations: usize,
        fitness_history: Vec<f64>,
    ) -> Self {
        Self {
            best_genome,
            best_fitness,
            generations,
            evaluations,
            fitness_history,
        }
    }
}
