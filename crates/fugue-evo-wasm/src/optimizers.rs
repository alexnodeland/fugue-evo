//! WASM optimizer wrappers

use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use fugue_evo::algorithms::cmaes::CmaEs;
use fugue_evo::algorithms::nsga2::{MultiObjectiveFitness, Nsga2};
use fugue_evo::algorithms::simple_ga::SimpleGABuilder;
use fugue_evo::fitness::traits::Fitness;
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

// ============================================================================
// Macros for DRY code
// ============================================================================

/// Macro to implement common optimizer configuration methods
macro_rules! impl_optimizer_config {
    ($name:ident) => {
        #[wasm_bindgen]
        impl $name {
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
        }
    };
    // Variant with bounds support
    ($name:ident, with_bounds) => {
        impl_optimizer_config!($name);

        #[wasm_bindgen]
        impl $name {
            /// Set bounds for all dimensions
            #[wasm_bindgen(js_name = setBounds)]
            pub fn set_bounds(&mut self, lower: f64, upper: f64) {
                self.config.set_bounds(lower, upper);
            }
        }
    };
}

/// Helper to create RNG from config
fn create_rng(seed: u64) -> rand::rngs::StdRng {
    if seed == 0 {
        rand::rngs::StdRng::from_entropy()
    } else {
        rand::rngs::StdRng::seed_from_u64(seed)
    }
}

// ============================================================================
// Fitness Evaluators
// ============================================================================

struct GenericFitness<G, F> {
    func: F,
    _phantom: std::marker::PhantomData<G>,
}

impl<G, F: Fn(&G) -> f64> GenericFitness<G, F> {
    fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<G: fugue_evo::genome::traits::EvolutionaryGenome, F: Fn(&G) -> f64> Fitness
    for GenericFitness<G, F>
{
    type Genome = G;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> Self::Value {
        (self.func)(genome)
    }
}

type RealVectorFitness<F> = GenericFitness<RealVector, F>;
type BitStringFitness<F> = GenericFitness<BitString, F>;
type PermutationFitness<F> = GenericFitness<Permutation, F>;

// ============================================================================
// Algorithm Enum
// ============================================================================

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

// ============================================================================
// RealVectorOptimizer
// ============================================================================

/// Real-vector optimizer using genetic algorithms
#[wasm_bindgen]
pub struct RealVectorOptimizer {
    config: OptimizationConfig,
    fitness_name: String,
    algorithm: Algorithm,
    cmaes_sigma: f64,
}

impl_optimizer_config!(RealVectorOptimizer, with_bounds);

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

    /// Set the fitness function by name
    #[wasm_bindgen(js_name = setFitness)]
    pub fn set_fitness(&mut self, name: &str) {
        self.fitness_name = name.to_string();
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

    /// Run with a custom JS fitness function (higher is better)
    #[wasm_bindgen(js_name = optimizeCustom)]
    pub fn optimize_custom(
        &self,
        fitness_fn: &js_sys::Function,
    ) -> Result<OptimizationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = self.create_bounds();

        let fitness_fn = fitness_fn.clone();
        let fitness = RealVectorFitness::new(move |genome: &RealVector| {
            call_js_fitness_f64(&fitness_fn, genome.genes())
        });

        self.run_ga_with_fitness(fitness, bounds, &mut rng)
    }

    /// Get the current configuration as JSON
    #[wasm_bindgen(js_name = getConfig)]
    pub fn get_config(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.config).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    fn create_bounds(&self) -> MultiBounds {
        MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        )
    }

    fn run_ga_with_fitness<F: Fitness<Genome = RealVector, Value = f64>>(
        &self,
        fitness: F,
        bounds: MultiBounds,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<OptimizationResult, JsValue> {
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

        let result = ga.run(rng).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    fn run_simple_ga(&self) -> Result<OptimizationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = self.create_bounds();

        let fitness_wrapper = FitnessWrapper::from_name(&self.fitness_name, self.config.dimension)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown fitness: {}", self.fitness_name)))?;

        self.run_ga_with_fitness(FitnessEvaluator::new(fitness_wrapper), bounds, &mut rng)
    }

    fn run_cmaes(&self) -> Result<OptimizationResult, JsValue> {
        use crate::fitness::CmaEsFitnessWrapper;

        let mut rng = create_rng(self.config.seed);
        let bounds = self.create_bounds();

        let fitness_wrapper = FitnessWrapper::from_name(&self.fitness_name, self.config.dimension)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown fitness: {}", self.fitness_name)))?;
        let fitness = CmaEsFitnessWrapper::new(fitness_wrapper);

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

        Ok(OptimizationResult::new(
            result.genome.genes().to_vec(),
            -cmaes.state.best_fitness,
            cmaes.state.generation,
            cmaes.state.evaluations,
            vec![-cmaes.state.best_fitness],
        ))
    }
}

// ============================================================================
// BitStringOptimizer
// ============================================================================

/// BitString optimizer using genetic algorithms
#[wasm_bindgen]
pub struct BitStringOptimizer {
    config: OptimizationConfig,
    flip_probability: f64,
}

impl_optimizer_config!(BitStringOptimizer);

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

    /// Set the bit flip probability for mutation
    #[wasm_bindgen(js_name = setFlipProbability)]
    pub fn set_flip_probability(&mut self, prob: f64) {
        self.flip_probability = prob.clamp(0.0, 1.0);
    }

    /// Run with a custom fitness function
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<BitStringResult, JsValue> {
        let fitness_fn = fitness_fn.clone();
        let fitness = BitStringFitness::new(move |genome: &BitString| {
            call_js_fitness_bool(&fitness_fn, genome.bits())
        });
        self.run_with_fitness(fitness)
    }

    /// Solve OneMax problem (maximize number of 1s)
    #[wasm_bindgen(js_name = solveOneMax)]
    pub fn solve_one_max(&self) -> Result<BitStringResult, JsValue> {
        let fitness = BitStringFitness::new(|genome: &BitString| {
            genome.bits().iter().filter(|&&b| b).count() as f64
        });
        self.run_with_fitness(fitness)
    }

    fn run_with_fitness<F: Fitness<Genome = BitString, Value = f64>>(
        &self,
        fitness: F,
    ) -> Result<BitStringResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::symmetric(1.0, self.config.dimension);

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

// ============================================================================
// PermutationOptimizer
// ============================================================================

/// Permutation optimizer using genetic algorithms
#[wasm_bindgen]
pub struct PermutationOptimizer {
    config: OptimizationConfig,
}

impl_optimizer_config!(PermutationOptimizer);

#[wasm_bindgen]
impl PermutationOptimizer {
    /// Create a new optimizer for permutations of size n
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            config: OptimizationConfig::new(size),
        }
    }

    /// Run with a custom fitness function
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<PermutationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::symmetric(1.0, self.config.dimension);

        let fitness_fn = fitness_fn.clone();
        let fitness = PermutationFitness::new(move |genome: &Permutation| {
            call_js_fitness_usize(&fitness_fn, genome.permutation())
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

// ============================================================================
// Nsga2Optimizer
// ============================================================================

/// Multi-objective optimizer using NSGA-II
#[wasm_bindgen]
pub struct Nsga2Optimizer {
    config: OptimizationConfig,
    num_objectives: usize,
    crossover_eta: f64,
    mutation_eta: f64,
}

impl_optimizer_config!(Nsga2Optimizer, with_bounds);

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

    /// Run with a multi-objective fitness function
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<MultiObjectiveResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let num_objectives = self.num_objectives;
        let fitness_fn = fitness_fn.clone();
        let fitness = MultiObjectiveFitnessEvaluator::new(
            move |genes: &[f64]| call_js_multi_objective(&fitness_fn, genes, num_objectives),
            num_objectives,
        );

        let nsga2 = Nsga2::<RealVector, _, _, _>::new(self.config.population_size)
            .with_bounds(bounds.clone());

        let result = nsga2
            .run(
                &fitness,
                &SbxCrossover::new(self.crossover_eta),
                &PolynomialMutation::new(self.mutation_eta),
                &bounds,
                self.config.max_generations,
                &mut rng,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let pareto_front: Vec<ParetoSolution> = result
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ParetoSolution::new(ind.genome.genes().to_vec(), ind.objectives.clone()))
            .collect();

        Ok(MultiObjectiveResult::new(
            pareto_front,
            self.config.max_generations,
            self.config.population_size * self.config.max_generations * 2,
        ))
    }
}

// ============================================================================
// Multi-objective fitness (special case - needs num_objectives)
// ============================================================================

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

// ============================================================================
// JS interop helpers
// ============================================================================

fn call_js_fitness_f64(func: &js_sys::Function, genes: &[f64]) -> f64 {
    let arr = js_sys::Float64Array::from(genes);
    func.call1(&JsValue::NULL, &arr)
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::NEG_INFINITY)
}

fn call_js_fitness_bool(func: &js_sys::Function, bits: &[bool]) -> f64 {
    let arr = js_sys::Array::new();
    for &b in bits {
        arr.push(&JsValue::from(b));
    }
    func.call1(&JsValue::NULL, &arr)
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::NEG_INFINITY)
}

fn call_js_fitness_usize(func: &js_sys::Function, perm: &[usize]) -> f64 {
    let arr = js_sys::Array::new();
    for &idx in perm {
        arr.push(&JsValue::from(idx as u32));
    }
    func.call1(&JsValue::NULL, &arr)
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::NEG_INFINITY)
}

fn call_js_multi_objective(
    func: &js_sys::Function,
    genes: &[f64],
    num_objectives: usize,
) -> Vec<f64> {
    let arr = js_sys::Float64Array::from(genes);
    match func.call1(&JsValue::NULL, &arr) {
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
}

// ============================================================================
// Evolution Strategy Optimizer
// ============================================================================

use fugue_evo::algorithms::evolution_strategy::{ESBuilder, ESSelectionStrategy, RecombinationType};

/// Selection strategy for Evolution Strategy
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub enum ESSelection {
    /// (μ+λ): Parents compete with offspring
    #[default]
    MuPlusLambda,
    /// (μ,λ): Only offspring compete
    MuCommaLambda,
}

/// Evolution Strategy optimizer (μ+λ)-ES or (μ,λ)-ES
#[wasm_bindgen]
pub struct EvolutionStrategyOptimizer {
    config: OptimizationConfig,
    mu: usize,
    lambda: usize,
    selection_strategy: ESSelection,
    initial_sigma: f64,
    self_adaptive: bool,
}

impl_optimizer_config!(EvolutionStrategyOptimizer, with_bounds);

#[wasm_bindgen]
impl EvolutionStrategyOptimizer {
    /// Create a new (μ+λ)-ES optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            mu: 15,
            lambda: 100,
            selection_strategy: ESSelection::MuPlusLambda,
            initial_sigma: 1.0,
            self_adaptive: true,
        }
    }

    /// Set μ (number of parents)
    #[wasm_bindgen(js_name = setMu)]
    pub fn set_mu(&mut self, mu: usize) {
        self.mu = mu;
    }

    /// Set λ (number of offspring)
    #[wasm_bindgen(js_name = setLambda)]
    pub fn set_lambda(&mut self, lambda: usize) {
        self.lambda = lambda;
    }

    /// Set the selection strategy
    #[wasm_bindgen(js_name = setSelectionStrategy)]
    pub fn set_selection_strategy(&mut self, strategy: ESSelection) {
        self.selection_strategy = strategy;
    }

    /// Set the initial step size (σ)
    #[wasm_bindgen(js_name = setInitialSigma)]
    pub fn set_initial_sigma(&mut self, sigma: f64) {
        self.initial_sigma = sigma;
    }

    /// Enable or disable self-adaptive mutation
    #[wasm_bindgen(js_name = setSelfAdaptive)]
    pub fn set_self_adaptive(&mut self, enabled: bool) {
        self.self_adaptive = enabled;
    }

    /// Run the optimization with a custom fitness function
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_fn: &js_sys::Function) -> Result<OptimizationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let fitness_fn = fitness_fn.clone();
        let fitness = RealVectorFitness::new(move |genome: &RealVector| {
            call_js_fitness_f64(&fitness_fn, genome.genes())
        });

        let selection = match self.selection_strategy {
            ESSelection::MuPlusLambda => ESSelectionStrategy::MuPlusLambda,
            ESSelection::MuCommaLambda => ESSelectionStrategy::MuCommaLambda,
        };

        let es = ESBuilder::<RealVector, f64, _, _>::new()
            .mu(self.mu)
            .lambda(self.lambda)
            .selection_strategy(selection)
            .initial_sigma(self.initial_sigma)
            .self_adaptive(self.self_adaptive)
            .recombination(RecombinationType::Intermediate)
            .bounds(bounds)
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = es
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
}

// Note: SteadyStateGA requires Sync for fitness functions, which JS functions
// don't support. Use SimpleGA or CmaEs as alternatives in WASM.

// ============================================================================
// Symbolic Regression (Tree-Based Genetic Programming)
// ============================================================================

use fugue_evo::genome::tree::{ArithmeticFunction, ArithmeticTerminal, TreeGenome};
use serde::{Deserialize, Serialize};

/// Type alias for arithmetic trees
pub type ArithmeticTree = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;

/// Result of a symbolic regression optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct SymbolicRegressionResult {
    /// The best tree expression as S-expression string
    expression: String,
    /// Best fitness value
    best_fitness: f64,
    /// Generations completed
    generations: usize,
    /// Total evaluations
    evaluations: usize,
    /// Tree depth
    tree_depth: usize,
    /// Tree size (number of nodes)
    tree_size: usize,
    /// Fitness history
    #[wasm_bindgen(skip)]
    pub fitness_history: Vec<f64>,
}

#[wasm_bindgen]
impl SymbolicRegressionResult {
    /// Get the best expression as S-expression
    #[wasm_bindgen(getter)]
    pub fn expression(&self) -> String {
        self.expression.clone()
    }

    /// Get the best fitness
    #[wasm_bindgen(js_name = bestFitness, getter)]
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Get the number of generations
    #[wasm_bindgen(getter)]
    pub fn generations(&self) -> usize {
        self.generations
    }

    /// Get the number of evaluations
    #[wasm_bindgen(getter)]
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// Get the tree depth
    #[wasm_bindgen(js_name = treeDepth, getter)]
    pub fn tree_depth(&self) -> usize {
        self.tree_depth
    }

    /// Get the tree size (number of nodes)
    #[wasm_bindgen(js_name = treeSize, getter)]
    pub fn tree_size(&self) -> usize {
        self.tree_size
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

/// Symbolic Regression optimizer using tree-based genetic programming
///
/// Evolves mathematical expressions to fit data points.
#[wasm_bindgen]
pub struct SymbolicRegressionOptimizer {
    population_size: usize,
    max_generations: usize,
    tournament_size: usize,
    max_tree_depth: usize,
    seed: u64,
}

#[wasm_bindgen]
impl SymbolicRegressionOptimizer {
    /// Create a new symbolic regression optimizer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 100,
            tournament_size: 3,
            max_tree_depth: 6,
            seed: 0,
        }
    }

    /// Set the population size
    #[wasm_bindgen(js_name = setPopulationSize)]
    pub fn set_population_size(&mut self, size: usize) {
        self.population_size = size;
    }

    /// Set the maximum generations
    #[wasm_bindgen(js_name = setMaxGenerations)]
    pub fn set_max_generations(&mut self, gens: usize) {
        self.max_generations = gens;
    }

    /// Set the tournament size
    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, size: usize) {
        self.tournament_size = size;
    }

    /// Set the maximum tree depth
    #[wasm_bindgen(js_name = setMaxTreeDepth)]
    pub fn set_max_tree_depth(&mut self, depth: usize) {
        self.max_tree_depth = depth;
    }

    /// Set the random seed
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Run symbolic regression with provided data points
    /// x_data: flat array of input values (num_points * num_vars)
    /// y_data: array of target output values
    /// num_vars: number of input variables
    #[wasm_bindgen(js_name = optimizeData)]
    pub fn optimize_data(
        &self,
        x_data: &[f64],
        y_data: &[f64],
        num_vars: usize,
    ) -> Result<SymbolicRegressionResult, JsValue> {
        use fugue_evo::operators::crossover::SubtreeCrossover;
        use fugue_evo::operators::mutation::SubtreeMutation;

        let num_points = y_data.len();
        if x_data.len() != num_points * num_vars {
            return Err(JsValue::from_str(&format!(
                "x_data length ({}) must be num_points ({}) * num_vars ({})",
                x_data.len(),
                num_points,
                num_vars
            )));
        }

        // Convert flat array to 2D data
        let data_points: Vec<Vec<f64>> = (0..num_points)
            .map(|i| x_data[i * num_vars..(i + 1) * num_vars].to_vec())
            .collect();
        let targets: Vec<f64> = y_data.to_vec();

        let mut rng = create_rng(self.seed);
        let bounds = MultiBounds::symmetric(1.0, self.max_tree_depth);

        // Create fitness function: minimize mean squared error
        let fitness = GenericFitness::<ArithmeticTree, _>::new(move |tree: &ArithmeticTree| {
            let mse: f64 = data_points
                .iter()
                .zip(targets.iter())
                .map(|(x, &y)| {
                    let pred = tree.evaluate(x);
                    let diff = pred - y;
                    diff * diff
                })
                .sum::<f64>()
                / num_points as f64;

            // Return negative MSE (we're maximizing)
            -mse
        });

        let ga = SimpleGABuilder::<ArithmeticTree, f64, _, _, _, _, _>::new()
            .population_size(self.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.tournament_size))
            .crossover(SubtreeCrossover::new().with_max_depth(self.max_tree_depth))
            .mutation(SubtreeMutation::new().with_max_depth(self.max_tree_depth))
            .fitness(fitness)
            .max_generations(self.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(SymbolicRegressionResult {
            expression: result.best_genome.to_sexpr(),
            best_fitness: result.best_fitness,
            generations: result.generations,
            evaluations: result.evaluations,
            tree_depth: result.best_genome.depth(),
            tree_size: result.best_genome.size(),
            fitness_history: result.stats.best_fitness_history(),
        })
    }

    /// Run symbolic regression with a custom fitness function
    /// The JS function receives the tree expression as a string and should return fitness
    #[wasm_bindgen(js_name = optimizeCustom)]
    pub fn optimize_custom(
        &self,
        fitness_fn: &js_sys::Function,
    ) -> Result<SymbolicRegressionResult, JsValue> {
        use fugue_evo::operators::crossover::SubtreeCrossover;
        use fugue_evo::operators::mutation::SubtreeMutation;

        let mut rng = create_rng(self.seed);
        let bounds = MultiBounds::symmetric(1.0, self.max_tree_depth);

        let fitness_fn = fitness_fn.clone();
        let fitness = GenericFitness::<ArithmeticTree, _>::new(move |tree: &ArithmeticTree| {
            let expr = tree.to_sexpr();
            let this = JsValue::null();
            let js_expr = JsValue::from_str(&expr);
            match fitness_fn.call1(&this, &js_expr) {
                Ok(result) => result.as_f64().unwrap_or(f64::NEG_INFINITY),
                Err(_) => f64::NEG_INFINITY,
            }
        });

        let ga = SimpleGABuilder::<ArithmeticTree, f64, _, _, _, _, _>::new()
            .population_size(self.population_size)
            .bounds(bounds)
            .selection(TournamentSelection::new(self.tournament_size))
            .crossover(SubtreeCrossover::new().with_max_depth(self.max_tree_depth))
            .mutation(SubtreeMutation::new().with_max_depth(self.max_tree_depth))
            .fitness(fitness)
            .max_generations(self.max_generations)
            .build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = ga
            .run(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(SymbolicRegressionResult {
            expression: result.best_genome.to_sexpr(),
            best_fitness: result.best_fitness,
            generations: result.generations,
            evaluations: result.evaluations,
            tree_depth: result.best_genome.depth(),
            tree_size: result.best_genome.size(),
            fitness_history: result.stats.best_fitness_history(),
        })
    }
}
