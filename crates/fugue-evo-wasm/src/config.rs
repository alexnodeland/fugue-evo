//! Configuration types for WASM optimizers

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Configuration for real-vector optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct OptimizationConfig {
    /// Number of dimensions
    pub dimension: usize,
    /// Population size
    pub population_size: usize,
    /// Maximum generations
    pub max_generations: usize,
    /// Lower bound for all dimensions
    pub lower_bound: f64,
    /// Upper bound for all dimensions
    pub upper_bound: f64,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover distribution index (SBX)
    pub crossover_eta: f64,
    /// Mutation distribution index (polynomial)
    pub mutation_eta: f64,
    /// Random seed (0 for random)
    pub seed: u64,
}

#[wasm_bindgen]
impl OptimizationConfig {
    /// Create a new configuration with default values
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            population_size: 100,
            max_generations: 200,
            lower_bound: -5.0,
            upper_bound: 5.0,
            tournament_size: 3,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 0,
        }
    }

    /// Set population size
    #[wasm_bindgen(js_name = setPopulationSize)]
    pub fn set_population_size(&mut self, size: usize) {
        self.population_size = size;
    }

    /// Set maximum generations
    #[wasm_bindgen(js_name = setMaxGenerations)]
    pub fn set_max_generations(&mut self, gens: usize) {
        self.max_generations = gens;
    }

    /// Set bounds for all dimensions
    #[wasm_bindgen(js_name = setBounds)]
    pub fn set_bounds(&mut self, lower: f64, upper: f64) {
        self.lower_bound = lower;
        self.upper_bound = upper;
    }

    /// Set tournament size
    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, size: usize) {
        self.tournament_size = size;
    }

    /// Set crossover eta (distribution index)
    #[wasm_bindgen(js_name = setCrossoverEta)]
    pub fn set_crossover_eta(&mut self, eta: f64) {
        self.crossover_eta = eta;
    }

    /// Set mutation eta (distribution index)
    #[wasm_bindgen(js_name = setMutationEta)]
    pub fn set_mutation_eta(&mut self, eta: f64) {
        self.mutation_eta = eta;
    }

    /// Set random seed (0 for random)
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::new(10)
    }
}
