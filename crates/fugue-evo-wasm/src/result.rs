//! Result types for WASM optimizers

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Macro to implement common result methods
macro_rules! impl_result_common {
    ($name:ident) => {
        #[wasm_bindgen]
        impl $name {
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
    };
}

// ============================================================================
// OptimizationResult (RealVector)
// ============================================================================

/// Result of an optimization run
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct OptimizationResult {
    best_genome: Vec<f64>,
    best_fitness: f64,
    generations: usize,
    evaluations: usize,
    fitness_history: Vec<f64>,
}

impl_result_common!(OptimizationResult);

#[wasm_bindgen]
impl OptimizationResult {
    /// Get the best genome as a Float64Array
    #[wasm_bindgen(js_name = bestGenome)]
    pub fn best_genome(&self) -> Vec<f64> {
        self.best_genome.clone()
    }
}

impl OptimizationResult {
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

// ============================================================================
// BitStringResult
// ============================================================================

/// Result of a BitString optimization run
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct BitStringResult {
    #[wasm_bindgen(skip)]
    pub best_genome: Vec<bool>,
    best_fitness: f64,
    generations: usize,
    evaluations: usize,
    fitness_history: Vec<f64>,
}

impl_result_common!(BitStringResult);

#[wasm_bindgen]
impl BitStringResult {
    /// Get the best genome as an array of 0s and 1s
    #[wasm_bindgen(js_name = bestGenome)]
    pub fn best_genome(&self) -> Vec<u8> {
        self.best_genome.iter().map(|&b| u8::from(b)).collect()
    }

    /// Get the best genome as a string of 0s and 1s
    #[wasm_bindgen(js_name = bestGenomeString)]
    pub fn best_genome_string(&self) -> String {
        self.best_genome
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    /// Get the number of 1-bits in the best genome
    #[wasm_bindgen(js_name = countOnes)]
    pub fn count_ones(&self) -> usize {
        self.best_genome.iter().filter(|&&b| b).count()
    }
}

impl BitStringResult {
    pub fn new(
        best_genome: Vec<bool>,
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

// ============================================================================
// PermutationResult
// ============================================================================

/// Result of a Permutation optimization run
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct PermutationResult {
    best_genome: Vec<usize>,
    best_fitness: f64,
    generations: usize,
    evaluations: usize,
    fitness_history: Vec<f64>,
}

impl_result_common!(PermutationResult);

#[wasm_bindgen]
impl PermutationResult {
    /// Get the best genome as an array of indices
    #[wasm_bindgen(js_name = bestGenome)]
    pub fn best_genome(&self) -> Vec<usize> {
        self.best_genome.clone()
    }
}

impl PermutationResult {
    pub fn new(
        best_genome: Vec<usize>,
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

// ============================================================================
// Multi-Objective Results
// ============================================================================

/// A solution on the Pareto front
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct ParetoSolution {
    #[wasm_bindgen(skip)]
    pub genome: Vec<f64>,
    #[wasm_bindgen(skip)]
    pub objectives: Vec<f64>,
}

#[wasm_bindgen]
impl ParetoSolution {
    #[wasm_bindgen(getter)]
    pub fn genome(&self) -> Vec<f64> {
        self.genome.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn objectives(&self) -> Vec<f64> {
        self.objectives.clone()
    }
}

impl ParetoSolution {
    pub fn new(genome: Vec<f64>, objectives: Vec<f64>) -> Self {
        Self { genome, objectives }
    }
}

/// Result of a multi-objective optimization run
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct MultiObjectiveResult {
    #[wasm_bindgen(skip)]
    pub pareto_front: Vec<ParetoSolution>,
    generations: usize,
    evaluations: usize,
}

#[wasm_bindgen]
impl MultiObjectiveResult {
    /// Get the number of solutions on the Pareto front
    #[wasm_bindgen(js_name = frontSize, getter)]
    pub fn front_size(&self) -> usize {
        self.pareto_front.len()
    }

    /// Get a solution from the Pareto front by index
    #[wasm_bindgen(js_name = getSolution)]
    pub fn get_solution(&self, index: usize) -> Option<ParetoSolution> {
        self.pareto_front.get(index).cloned()
    }

    /// Get all genomes as a flat array (for visualization)
    #[wasm_bindgen(js_name = allGenomes)]
    pub fn all_genomes(&self) -> Vec<f64> {
        self.pareto_front
            .iter()
            .flat_map(|s| s.genome.clone())
            .collect()
    }

    /// Get all objectives as a flat array (for visualization)
    #[wasm_bindgen(js_name = allObjectives)]
    pub fn all_objectives(&self) -> Vec<f64> {
        self.pareto_front
            .iter()
            .flat_map(|s| s.objectives.clone())
            .collect()
    }

    #[wasm_bindgen(getter)]
    pub fn generations(&self) -> usize {
        self.generations
    }

    #[wasm_bindgen(getter)]
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl MultiObjectiveResult {
    pub fn new(pareto_front: Vec<ParetoSolution>, generations: usize, evaluations: usize) -> Self {
        Self {
            pareto_front,
            generations,
            evaluations,
        }
    }
}
