//! WASM optimizer wrappers

use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use fugue_evo::algorithms::cmaes::CmaEs;
use fugue_evo::algorithms::nsga2::{MultiObjectiveFitness, Nsga2};
use fugue_evo::algorithms::simple_ga::SimpleGABuilder;
use fugue_evo::genome::bit_string::BitString;
use fugue_evo::genome::bounds::{Bounds, MultiBounds};
use fugue_evo::genome::permutation::Permutation;
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::{BinaryGenome, PermutationGenome, RealValuedGenome};
use fugue_evo::operators::crossover::{OxCrossover, SbxCrossover, UniformCrossover};
use fugue_evo::operators::mutation::{
    BitFlipMutation, PermutationSwapMutation, PolynomialMutation,
};
use fugue_evo::operators::selection::TournamentSelection;

use crate::config::OptimizationConfig;
use crate::fitness::{FitnessEvaluator, FitnessWrapper};
use crate::result::{
    BitStringResult, MultiObjectiveResult, OptimizationResult, ParetoSolution, PermutationResult,
};

/// Algorithm type for optimization
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub enum Algorithm {
    /// Simple Genetic Algorithm (default)
    #[default]
    SimpleGA,
    /// CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    CmaES,
}

/// Real-vector optimizer using genetic algorithms
#[wasm_bindgen]
pub struct RealVectorOptimizer {
    config: OptimizationConfig,
    fitness_name: String,
    algorithm: Algorithm,
    cmaes_sigma: f64,
}

#[wasm_bindgen]
impl RealVectorOptimizer {
    /// Create a new optimizer for the given dimension
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            fitness_name: "sphere".to_string(),
            algorithm: Algorithm::default(),
            cmaes_sigma: 1.0,
        }
    }

    /// Set the algorithm to use
    #[wasm_bindgen(js_name = setAlgorithm)]
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.algorithm = algorithm;
    }

    /// Set the initial sigma for CMA-ES
    #[wasm_bindgen(js_name = setCmaEsSigma)]
    pub fn set_cmaes_sigma(&mut self, sigma: f64) {
        self.cmaes_sigma = sigma;
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

    /// Run the optimization with built-in fitness function
    #[wasm_bindgen]
    pub fn optimize(&self) -> Result<OptimizationResult, JsValue> {
        match self.algorithm {
            Algorithm::SimpleGA => self.run_simple_ga(),
            Algorithm::CmaES => self.run_cmaes(),
        }
    }

    /// Run the optimization with a custom JS fitness function
    /// The fitness function receives a Float64Array and should return a number (higher is better)
    #[wasm_bindgen(js_name = optimizeCustom)]
    pub fn optimize_custom(
        &self,
        fitness_fn: &js_sys::Function,
    ) -> Result<OptimizationResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let fitness_fn_clone = fitness_fn.clone();
        let fitness = RealVectorFitnessEvaluator::new(move |genes: &[f64]| {
            let arr = js_sys::Float64Array::from(genes);
            match fitness_fn_clone.call1(&JsValue::NULL, &arr) {
                Ok(result) => result.as_f64().unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        });

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

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    fn run_simple_ga(&self) -> Result<OptimizationResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let fitness_wrapper = FitnessWrapper::from_name(&self.fitness_name, self.config.dimension)
            .ok_or_else(|| {
                JsValue::from_str(&format!("Unknown fitness function: {}", self.fitness_name))
            })?;
        let fitness = FitnessEvaluator::new(fitness_wrapper);

        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

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

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    fn run_cmaes(&self) -> Result<OptimizationResult, JsValue> {
        use crate::fitness::CmaEsFitnessWrapper;

        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let fitness_wrapper = FitnessWrapper::from_name(&self.fitness_name, self.config.dimension)
            .ok_or_else(|| {
                JsValue::from_str(&format!("Unknown fitness function: {}", self.fitness_name))
            })?;
        let fitness = CmaEsFitnessWrapper::new(fitness_wrapper);

        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        // Initialize at center of bounds
        let initial_mean: Vec<f64> = (0..self.config.dimension)
            .map(|_| (self.config.lower_bound + self.config.upper_bound) / 2.0)
            .collect();

        let mut cmaes = if self.config.population_size > 0 {
            CmaEs::with_lambda(initial_mean, self.cmaes_sigma, self.config.population_size)
        } else {
            CmaEs::new(initial_mean, self.cmaes_sigma)
        };
        cmaes = cmaes.with_bounds(bounds);

        let result = cmaes
            .run_generations(&fitness, self.config.max_generations, &mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // CMA-ES minimizes, negate back for consistency
        Ok(OptimizationResult::new(
            result.genome.genes().to_vec(),
            -cmaes.state.best_fitness,
            cmaes.state.generation,
            cmaes.state.evaluations,
            vec![-cmaes.state.best_fitness],
        ))
    }

    /// Get the current configuration as JSON
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// BitString optimizer using genetic algorithms
#[wasm_bindgen]
pub struct BitStringOptimizer {
    config: OptimizationConfig,
    flip_probability: f64,
}

#[wasm_bindgen]
impl BitStringOptimizer {
    /// Create a new optimizer for the given length
    #[wasm_bindgen(constructor)]
    pub fn new(length: usize) -> Self {
        Self {
            config: OptimizationConfig::new(length),
            flip_probability: 0.01,
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

    /// Set the bit flip probability for mutation
    #[wasm_bindgen(js_name = setFlipProbability)]
    pub fn set_flip_probability(&mut self, prob: f64) {
        self.flip_probability = prob.clamp(0.0, 1.0);
    }

    /// Run the optimization with a custom fitness function
    /// The fitness function receives a Uint8Array of bits (0 or 1) and returns a number
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<BitStringResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        // Dummy bounds for BitString (not used for values, just dimension)
        let bounds = MultiBounds::symmetric(1.0, self.config.dimension);

        // Create a fitness evaluator that calls the JS function
        let fitness_fn_clone = fitness_fn.clone();
        let fitness = BitStringFitnessEvaluator::new(move |bits: &[bool]| {
            let arr = js_sys::Array::new();
            for &b in bits {
                arr.push(&JsValue::from(b));
            }
            match fitness_fn_clone.call1(&JsValue::NULL, &arr) {
                Ok(result) => result.as_f64().unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        });

        let ga = SimpleGABuilder::<BitString, f64, _, _, _, _, _>::new()
            .population_size(self.config.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.config.tournament_size))
            .crossover(UniformCrossover::new())
            .mutation(BitFlipMutation::new().with_probability(self.flip_probability))
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(BitStringResult::new(
            result.best_genome.bits().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    /// Solve OneMax problem (maximize number of 1s)
    #[wasm_bindgen(js_name = solveOneMax)]
    pub fn solve_one_max(&self) -> Result<BitStringResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let bounds = MultiBounds::symmetric(1.0, self.config.dimension);

        let fitness = BitStringFitnessEvaluator::new(|bits: &[bool]| {
            bits.iter().filter(|&&b| b).count() as f64
        });

        let ga = SimpleGABuilder::<BitString, f64, _, _, _, _, _>::new()
            .population_size(self.config.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.config.tournament_size))
            .crossover(UniformCrossover::new())
            .mutation(BitFlipMutation::new().with_probability(self.flip_probability))
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(BitStringResult::new(
            result.best_genome.bits().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }
}

/// Permutation optimizer using genetic algorithms
#[wasm_bindgen]
pub struct PermutationOptimizer {
    config: OptimizationConfig,
}

#[wasm_bindgen]
impl PermutationOptimizer {
    /// Create a new optimizer for permutations of size n
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            config: OptimizationConfig::new(size),
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

    /// Run the optimization with a custom fitness function
    /// The fitness function receives an array of indices and returns a number
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<PermutationResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let bounds = MultiBounds::symmetric(1.0, self.config.dimension);

        let fitness_fn_clone = fitness_fn.clone();
        let fitness = PermutationFitnessEvaluator::new(move |perm: &[usize]| {
            let arr = js_sys::Array::new();
            for &idx in perm {
                arr.push(&JsValue::from(idx as u32));
            }
            match fitness_fn_clone.call1(&JsValue::NULL, &arr) {
                Ok(result) => result.as_f64().unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        });

        let ga = SimpleGABuilder::<Permutation, f64, _, _, _, _, _>::new()
            .population_size(self.config.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.config.tournament_size))
            .crossover(OxCrossover)
            .mutation(PermutationSwapMutation::new())
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(PermutationResult::new(
            result.best_genome.permutation().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }
}

// Internal fitness evaluators for BitString and Permutation

use fugue_evo::fitness::traits::Fitness;

struct BitStringFitnessEvaluator<F: Fn(&[bool]) -> f64> {
    func: F,
}

impl<F: Fn(&[bool]) -> f64> BitStringFitnessEvaluator<F> {
    fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F: Fn(&[bool]) -> f64> Fitness for BitStringFitnessEvaluator<F> {
    type Genome = BitString;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        (self.func)(genome.bits())
    }
}

struct PermutationFitnessEvaluator<F: Fn(&[usize]) -> f64> {
    func: F,
}

impl<F: Fn(&[usize]) -> f64> PermutationFitnessEvaluator<F> {
    fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F: Fn(&[usize]) -> f64> Fitness for PermutationFitnessEvaluator<F> {
    type Genome = Permutation;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        (self.func)(genome.permutation())
    }
}

/// Multi-objective optimizer using NSGA-II
#[wasm_bindgen]
pub struct Nsga2Optimizer {
    config: OptimizationConfig,
    num_objectives: usize,
    crossover_eta: f64,
    mutation_eta: f64,
}

#[wasm_bindgen]
impl Nsga2Optimizer {
    /// Create a new multi-objective optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, num_objectives: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            num_objectives,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
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

    /// Set the random seed (0 for random)
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.config.set_seed(seed);
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

    /// Run the optimization with a custom multi-objective fitness function
    /// The fitness function receives an array of floats and returns an array of objective values
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<MultiObjectiveResult, JsValue> {
        let mut rng = if self.config.seed == 0 {
            rand::rngs::StdRng::from_entropy()
        } else {
            rand::rngs::StdRng::seed_from_u64(self.config.seed)
        };

        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let num_objectives = self.num_objectives;
        let fitness_fn_clone = fitness_fn.clone();
        let fitness = MultiObjectiveFitnessEvaluator::new(
            move |genes: &[f64]| {
                let arr = js_sys::Float64Array::from(genes);
                match fitness_fn_clone.call1(&JsValue::NULL, &arr) {
                    Ok(result) => {
                        if let Some(arr) = result.dyn_ref::<js_sys::Array>() {
                            (0..arr.length())
                                .map(|i| arr.get(i).as_f64().unwrap_or(f64::INFINITY))
                                .collect()
                        } else if let Some(arr) = result.dyn_ref::<js_sys::Float64Array>() {
                            arr.to_vec()
                        } else {
                            vec![f64::INFINITY; num_objectives]
                        }
                    }
                    Err(_) => vec![f64::INFINITY; num_objectives],
                }
            },
            num_objectives,
        );

        let crossover = SbxCrossover::new(self.crossover_eta);
        let mutation = PolynomialMutation::new(self.mutation_eta);

        let nsga2 = Nsga2::<RealVector, _, _, _>::new(self.config.population_size)
            .with_bounds(bounds.clone());

        let result = nsga2
            .run(
                &fitness,
                &crossover,
                &mutation,
                &bounds,
                self.config.max_generations,
                &mut rng,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Extract Pareto front (rank 0 solutions)
        let pareto_front: Vec<ParetoSolution> = result
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ParetoSolution::new(ind.genome.genes().to_vec(), ind.objectives.clone()))
            .collect();

        let evaluations = self.config.population_size * self.config.max_generations * 2;

        Ok(MultiObjectiveResult::new(
            pareto_front,
            self.config.max_generations,
            evaluations,
        ))
    }
}

struct MultiObjectiveFitnessEvaluator<F: Fn(&[f64]) -> Vec<f64>> {
    func: F,
    num_objectives: usize,
}

impl<F: Fn(&[f64]) -> Vec<f64>> MultiObjectiveFitnessEvaluator<F> {
    fn new(func: F, num_objectives: usize) -> Self {
        Self {
            func,
            num_objectives,
        }
    }
}

impl<F: Fn(&[f64]) -> Vec<f64>> MultiObjectiveFitness<RealVector>
    for MultiObjectiveFitnessEvaluator<F>
{
    fn num_objectives(&self) -> usize {
        self.num_objectives
    }

    fn evaluate(&self, genome: &RealVector) -> Vec<f64> {
        (self.func)(genome.genes())
    }
}

struct RealVectorFitnessEvaluator<F: Fn(&[f64]) -> f64> {
    func: F,
}

impl<F: Fn(&[f64]) -> f64> RealVectorFitnessEvaluator<F> {
    fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F: Fn(&[f64]) -> f64> Fitness for RealVectorFitnessEvaluator<F> {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        (self.func)(genome.genes())
    }
}
