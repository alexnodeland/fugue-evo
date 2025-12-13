//! WASM optimizer wrappers

use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use fugue_evo::algorithms::simple_ga::SimpleGABuilder;
use fugue_evo::genome::bounds::{Bounds, MultiBounds};
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::RealValuedGenome;
use fugue_evo::operators::crossover::SbxCrossover;
use fugue_evo::operators::mutation::PolynomialMutation;
use fugue_evo::operators::selection::TournamentSelection;

use crate::config::OptimizationConfig;
use crate::fitness::{FitnessEvaluator, FitnessWrapper};
use crate::result::OptimizationResult;

/// Real-vector optimizer using genetic algorithms
#[wasm_bindgen]
pub struct RealVectorOptimizer {
    config: OptimizationConfig,
    fitness_name: String,
}

#[wasm_bindgen]
impl RealVectorOptimizer {
    /// Create a new optimizer for the given dimension
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            fitness_name: "sphere".to_string(),
        }
    }

    /// Set the population size
    #[wasm_bindgen(js_name = setPopulationSize)]
    pub fn set_population_size(&mut self, size: usize) {
        self.config.set_population_size(size);
    }

    /// Set the maximum generations
    #[wasm_bindgen(js_name = setMaxGenerations)]
    pub fn set_max_generations(&mut self, gens: usize) {
        self.config.set_max_generations(gens);
    }

    /// Set bounds for all dimensions
    #[wasm_bindgen(js_name = setBounds)]
    pub fn set_bounds(&mut self, lower: f64, upper: f64) {
        self.config.set_bounds(lower, upper);
    }

    /// Set the fitness function by name
    #[wasm_bindgen(js_name = setFitness)]
    pub fn set_fitness(&mut self, name: &str) {
        self.fitness_name = name.to_string();
    }

    /// Set the random seed (0 for random)
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.config.set_seed(seed);
    }

    /// Set tournament size for selection
    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, size: usize) {
        self.config.set_tournament_size(size);
    }

    /// Set crossover eta (distribution index)
    #[wasm_bindgen(js_name = setCrossoverEta)]
    pub fn set_crossover_eta(&mut self, eta: f64) {
        self.config.set_crossover_eta(eta);
    }

    /// Set mutation eta (distribution index)
    #[wasm_bindgen(js_name = setMutationEta)]
    pub fn set_mutation_eta(&mut self, eta: f64) {
        self.config.set_mutation_eta(eta);
    }

    /// Run the optimization
    #[wasm_bindgen]
    pub fn optimize(&self) -> Result<OptimizationResult, JsValue> {
        // Create RNG
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        // Create fitness function
        let fitness_wrapper = FitnessWrapper::from_name(&self.fitness_name, self.config.dimension)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown fitness function: {}", self.fitness_name)))?;
        let fitness = FitnessEvaluator::new(fitness_wrapper);

        // Create bounds
        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        // Build GA
        let ga = SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
            .population_size(self.config.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.config.tournament_size))
            .crossover(SbxCrossover::new(self.config.crossover_eta))
            .mutation(PolynomialMutation::new(self.config.mutation_eta))
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Run evolution
        let result = ga.run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Extract fitness history from stats
        let fitness_history = result.stats.best_fitness_history();

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            fitness_history,
        ))
    }

    /// Get the current configuration as JSON
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
