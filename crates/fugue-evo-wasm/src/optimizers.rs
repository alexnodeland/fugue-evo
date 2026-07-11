//! WASM optimizer wrappers

use rand::SeedableRng;
use wasm_bindgen::prelude::*;

use fugue_evo::algorithms::cmaes::CmaEs;
use fugue_evo::algorithms::nsga2::{
    fast_non_dominated_sort, recompute_crowding_distance_per_front, MultiObjectiveFitness, Nsga2,
};
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
use crate::error::evolution_error_to_js;
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
            .map_err(evolution_error_to_js)?;

        let result = ga.run(rng).map_err(evolution_error_to_js)?;

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

        // Drive CMA-ES one generation at a time so we can record a genuine
        // per-generation best-so-far trajectory, matching the fitness_history
        // that SimpleGA/UMDA/ES return (AUDIT EV-76). CMA-ES minimizes on the
        // negated objective, so we flip the sign for reporting (higher = better).
        let mut fitness_history: Vec<f64> = Vec::with_capacity(self.config.max_generations);
        let mut best_individual: Option<fugue_evo::population::individual::Individual<RealVector>> =
            None;
        for _ in 0..self.config.max_generations {
            let population = cmaes
                .step(&fitness, &mut rng)
                .map_err(evolution_error_to_js)?;
            if let Some(current_best) = population.first() {
                let better = match &best_individual {
                    None => true,
                    Some(existing) => current_best.fitness_f64() < existing.fitness_f64(),
                };
                if better {
                    best_individual = Some(current_best.clone());
                }
            }
            // Record the best-so-far for this generation (sign-flipped).
            fitness_history.push(-cmaes.state.best_fitness);

            if cmaes.state.has_converged() {
                break;
            }
        }

        let best_genome = best_individual
            .map(|ind| ind.genome.genes().to_vec())
            .unwrap_or_else(|| cmaes.state.best_solution.clone());

        Ok(OptimizationResult::new(
            best_genome,
            -cmaes.state.best_fitness,
            cmaes.state.generation,
            cmaes.state.evaluations,
            fitness_history,
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

    /// Run with a custom fitness function, reporting progress per generation and
    /// allowing cancellation (AUDIT EV-34).
    ///
    /// `progressFn(generation, bestFitness)` is invoked once per generation
    /// before it is evolved; return `false` to cancel and receive the best result
    /// found so far. Drive this from a Web Worker to `postMessage` a progress bar
    /// or honor a cancel button without blocking on one opaque `optimize()` call.
    #[wasm_bindgen(js_name = optimizeWithProgress)]
    pub fn optimize_with_progress(
        &self,
        fitness_fn: &js_sys::Function,
        progress_fn: &js_sys::Function,
    ) -> Result<BitStringResult, JsValue> {
        let fitness_fn = fitness_fn.clone();
        let fitness = BitStringFitness::new(move |genome: &BitString| {
            call_js_fitness_bool(&fitness_fn, genome.bits())
        });

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
            .map_err(evolution_error_to_js)?;

        let mut state = ga.init_run(&mut rng).map_err(evolution_error_to_js)?;
        loop {
            if !report_progress(progress_fn, state.generation(), state.best_fitness())? {
                break;
            }
            if !ga
                .step_generation(&mut state, &mut rng)
                .map_err(evolution_error_to_js)?
            {
                break;
            }
        }

        Ok(BitStringResult::new(
            state.best_genome().bits().to_vec(),
            state.best_fitness(),
            state.generation(),
            state.evaluations(),
            state.fitness_history().to_vec(),
        ))
    }

    /// Solve OneMax problem (maximize number of 1s)
    #[wasm_bindgen(js_name = solveOneMax)]
    pub fn solve_one_max(&self) -> Result<BitStringResult, JsValue> {
        let fitness = BitStringFitness::new(|genome: &BitString| {
            genome.bits().iter().filter(|&&b| b).count() as f64
        });
        self.run_with_fitness(fitness)
    }

    /// Solve LeadingOnes problem (maximize leading 1s before first 0)
    #[wasm_bindgen(js_name = solveLeadingOnes)]
    pub fn solve_leading_ones(&self) -> Result<BitStringResult, JsValue> {
        let fitness = BitStringFitness::new(|genome: &BitString| {
            genome.bits().iter().take_while(|&&b| b).count() as f64
        });
        self.run_with_fitness(fitness)
    }

    /// Solve Royal Road problem (maximize complete schema blocks)
    /// schema_size: size of each block (default 8)
    #[wasm_bindgen(js_name = solveRoyalRoad)]
    pub fn solve_royal_road(&self, schema_size: usize) -> Result<BitStringResult, JsValue> {
        let length = self.config.dimension;
        let num_schemas = length / schema_size.max(1);
        let schema_size = schema_size.max(1);

        let fitness = BitStringFitness::new(move |genome: &BitString| {
            let bits = genome.bits();
            let mut complete = 0;
            for schema_idx in 0..num_schemas {
                let start = schema_idx * schema_size;
                let end = (start + schema_size).min(bits.len());
                if bits[start..end].iter().all(|&b| b) {
                    complete += 1;
                }
            }
            complete as f64
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
            .map_err(evolution_error_to_js)?;

        let result = ga.run(&mut rng).map_err(evolution_error_to_js)?;

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
            .map_err(evolution_error_to_js)?;

        let result = ga.run(&mut rng).map_err(evolution_error_to_js)?;

        Ok(PermutationResult::new(
            result.best_genome.permutation().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    /// Run with a custom fitness function, reporting progress per generation and
    /// allowing cancellation (AUDIT EV-34).
    ///
    /// `progressFn(generation, bestFitness)` is invoked once per generation before
    /// it is evolved; return `false` to cancel and receive the best-so-far result.
    #[wasm_bindgen(js_name = optimizeWithProgress)]
    pub fn optimize_with_progress(
        &self,
        fitness_fn: &js_sys::Function,
        progress_fn: &js_sys::Function,
    ) -> Result<PermutationResult, JsValue> {
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
            .map_err(evolution_error_to_js)?;

        let mut state = ga.init_run(&mut rng).map_err(evolution_error_to_js)?;
        loop {
            if !report_progress(progress_fn, state.generation(), state.best_fitness())? {
                break;
            }
            if !ga
                .step_generation(&mut state, &mut rng)
                .map_err(evolution_error_to_js)?
            {
                break;
            }
        }

        Ok(PermutationResult::new(
            state.best_genome().permutation().to_vec(),
            state.best_fitness(),
            state.generation(),
            state.evaluations(),
            state.fitness_history().to_vec(),
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
        // Count real evaluations performed by NSGA-II instead of fabricating a
        // formula (AUDIT EV-33).
        let fitness = CountingMoFitness::new(fitness);

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
            .map_err(evolution_error_to_js)?;

        let pareto_front: Vec<ParetoSolution> = result
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ParetoSolution::new(ind.genome.genes().to_vec(), ind.objectives.clone()))
            .collect();

        Ok(MultiObjectiveResult::new(
            pareto_front,
            self.config.max_generations,
            fitness.evaluations(),
        ))
    }

    /// Run NSGA-II reporting progress per generation and allowing cancellation
    /// (AUDIT EV-34).
    ///
    /// `progressFn(generation, paretoFrontSize)` is invoked once per generation
    /// before it is evolved; the second argument is the number of non-dominated
    /// (rank-0) solutions found so far, a natural multi-objective progress signal
    /// in place of a single scalar fitness. Return `false` to cancel and receive
    /// the best-so-far Pareto front.
    #[wasm_bindgen(js_name = optimizeWithProgress)]
    pub fn optimize_with_progress(
        &self,
        fitness_fn: &js_sys::Function,
        progress_fn: &js_sys::Function,
    ) -> Result<MultiObjectiveResult, JsValue> {
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
        let fitness = CountingMoFitness::new(fitness);

        let nsga2 = Nsga2::<RealVector, _, _, _>::new(self.config.population_size)
            .with_bounds(bounds.clone());
        let crossover = SbxCrossover::new(self.crossover_eta);
        let mutation = PolynomialMutation::new(self.mutation_eta);

        // Mirror Nsga2::run's initialization, then drive the generation loop
        // manually so we can report progress and cancel between generations.
        let mut population = nsga2.initialize_population(&fitness, &bounds, &mut rng);
        fast_non_dominated_sort(&mut population);
        recompute_crowding_distance_per_front(&mut population);

        let mut generation = 0usize;
        while generation < self.config.max_generations {
            let pareto_size = population.iter().filter(|ind| ind.rank == 0).count();
            if !report_progress(progress_fn, generation, pareto_size as f64)? {
                break;
            }
            nsga2.step(&mut population, &fitness, &crossover, &mutation, &mut rng);
            generation += 1;
        }

        let pareto_front: Vec<ParetoSolution> = population
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ParetoSolution::new(ind.genome.genes().to_vec(), ind.objectives.clone()))
            .collect();

        Ok(MultiObjectiveResult::new(
            pareto_front,
            generation,
            fitness.evaluations(),
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

/// A [`MultiObjectiveFitness`] decorator that counts the actual number of
/// objective evaluations performed.
///
/// `Nsga2::run` does not return an evaluation counter, and the previous code
/// fabricated one as `population_size * max_generations * 2` — a value that
/// overstated the true count by ~90% for typical configs and had no basis in
/// the algorithm's behavior (AUDIT EV-33). Wrapping the fitness lets us report
/// the real, measured evaluation count (one per `evaluate` call). Uses an atomic
/// so the wrapper stays `Send + Sync` when the inner fitness is.
struct CountingMoFitness<Fit> {
    inner: Fit,
    count: std::sync::atomic::AtomicUsize,
}

impl<Fit: MultiObjectiveFitness<RealVector>> CountingMoFitness<Fit> {
    fn new(inner: Fit) -> Self {
        Self {
            inner,
            count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The number of objective evaluations performed so far.
    fn evaluations(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<Fit: MultiObjectiveFitness<RealVector>> MultiObjectiveFitness<RealVector>
    for CountingMoFitness<Fit>
{
    fn num_objectives(&self) -> usize {
        self.inner.num_objectives()
    }

    fn evaluate(&self, genome: &RealVector) -> Vec<f64> {
        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.evaluate(genome)
    }
}

// ============================================================================
// JS interop helpers
// ============================================================================

/// Invoke a JS per-generation progress callback (AUDIT EV-34).
///
/// The callback receives `(generation, bestFitness)`. It may update a progress
/// bar, `postMessage` from a Web Worker, etc. Returning boolean `false` cancels
/// the run; any other return value (including `undefined`) continues, so a
/// callback that just reports progress needs no explicit `return true`. A thrown
/// exception is propagated to the caller as an `Err`.
fn report_progress(
    progress_fn: &js_sys::Function,
    generation: usize,
    best_fitness: f64,
) -> Result<bool, JsValue> {
    let ret = progress_fn.call2(
        &JsValue::NULL,
        &JsValue::from_f64(generation as f64),
        &JsValue::from_f64(best_fitness),
    )?;
    // Continue unless the callback explicitly returned boolean `false`.
    Ok(ret.as_bool() != Some(false))
}

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

use fugue_evo::algorithms::evolution_strategy::{
    ESBuilder, ESSelectionStrategy, RecombinationType,
};

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
            .map_err(evolution_error_to_js)?;

        let result = es.run(&mut rng).map_err(evolution_error_to_js)?;

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    /// Run the evolution strategy reporting progress per generation and allowing
    /// cancellation (AUDIT EV-34).
    ///
    /// `progressFn(generation, bestFitness)` is invoked once per generation before
    /// it is evolved; return `false` to cancel and receive the best-so-far result.
    /// Backed by the native `EvolutionStrategy::run_with_callback` hook.
    #[wasm_bindgen(js_name = optimizeWithProgress)]
    pub fn optimize_with_progress(
        &self,
        fitness_fn: &js_sys::Function,
        progress_fn: &js_sys::Function,
    ) -> Result<OptimizationResult, JsValue> {
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
            .map_err(evolution_error_to_js)?;

        // A thrown exception in the JS callback can't cross the `FnMut -> bool`
        // boundary, so capture it and cancel, then surface it after the run.
        let mut progress_err: Option<JsValue> = None;
        let result = es
            .run_with_callback(&mut rng, |generation, best_fitness| {
                match report_progress(progress_fn, generation, best_fitness) {
                    Ok(cont) => cont,
                    Err(e) => {
                        progress_err = Some(e);
                        false
                    }
                }
            })
            .map_err(evolution_error_to_js)?;
        if let Some(e) = progress_err {
            return Err(e);
        }

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

impl Default for SymbolicRegressionOptimizer {
    fn default() -> Self {
        Self::new()
    }
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
            .map_err(evolution_error_to_js)?;

        let result = ga.run(&mut rng).map_err(evolution_error_to_js)?;

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
            .map_err(evolution_error_to_js)?;

        let result = ga.run(&mut rng).map_err(evolution_error_to_js)?;

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

    /// Run symbolic regression (custom JS fitness over the expression string)
    /// reporting progress per generation and allowing cancellation (AUDIT EV-34).
    ///
    /// `fitnessFn(expr)` returns the fitness of a candidate expression;
    /// `progressFn(generation, bestFitness)` is invoked once per generation before
    /// it is evolved. Return `false` from `progressFn` to cancel and receive the
    /// best-so-far expression.
    #[wasm_bindgen(js_name = optimizeCustomWithProgress)]
    pub fn optimize_custom_with_progress(
        &self,
        fitness_fn: &js_sys::Function,
        progress_fn: &js_sys::Function,
    ) -> Result<SymbolicRegressionResult, JsValue> {
        use fugue_evo::operators::crossover::SubtreeCrossover;
        use fugue_evo::operators::mutation::SubtreeMutation;

        let mut rng = create_rng(self.seed);
        let bounds = MultiBounds::symmetric(1.0, self.max_tree_depth);

        let fitness_fn = fitness_fn.clone();
        let fitness = GenericFitness::<ArithmeticTree, _>::new(move |tree: &ArithmeticTree| {
            let expr = tree.to_sexpr();
            let js_expr = JsValue::from_str(&expr);
            match fitness_fn.call1(&JsValue::null(), &js_expr) {
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
            .map_err(evolution_error_to_js)?;

        let mut state = ga.init_run(&mut rng).map_err(evolution_error_to_js)?;
        loop {
            if !report_progress(progress_fn, state.generation(), state.best_fitness())? {
                break;
            }
            if !ga
                .step_generation(&mut state, &mut rng)
                .map_err(evolution_error_to_js)?
            {
                break;
            }
        }

        Ok(SymbolicRegressionResult {
            expression: state.best_genome().to_sexpr(),
            best_fitness: state.best_fitness(),
            generations: state.generation(),
            evaluations: state.evaluations(),
            tree_depth: state.best_genome().depth(),
            tree_size: state.best_genome().size(),
            fitness_history: state.fitness_history().to_vec(),
        })
    }
}

// ============================================================================
// UMDA (Estimation of Distribution Algorithm)
// ============================================================================

use fugue_evo::algorithms::eda::umda::UMDABuilder;

/// UMDA optimizer for continuous optimization
/// Uses probability distribution estimation instead of crossover/mutation
#[wasm_bindgen]
pub struct UmdaOptimizer {
    config: OptimizationConfig,
    selection_ratio: f64,
    min_variance: f64,
    learning_rate: f64,
}

impl_optimizer_config!(UmdaOptimizer, with_bounds);

#[wasm_bindgen]
impl UmdaOptimizer {
    /// Create a new UMDA optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            selection_ratio: 0.5,
            min_variance: 0.01,
            learning_rate: 1.0,
        }
    }

    /// Set the selection ratio (top proportion selected for model learning)
    #[wasm_bindgen(js_name = setSelectionRatio)]
    pub fn set_selection_ratio(&mut self, ratio: f64) {
        self.selection_ratio = ratio.clamp(0.1, 0.9);
    }

    /// Set the minimum variance to prevent collapse
    #[wasm_bindgen(js_name = setMinVariance)]
    pub fn set_min_variance(&mut self, variance: f64) {
        self.min_variance = variance.max(0.001);
    }

    /// Set the learning rate for model update (1.0 = replace, <1.0 = blend)
    #[wasm_bindgen(js_name = setLearningRate)]
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Run optimization with a built-in fitness function
    #[wasm_bindgen]
    pub fn optimize(&self, fitness_name: &str) -> Result<OptimizationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let fitness_wrapper = FitnessWrapper::from_name(fitness_name, self.config.dimension)
            .ok_or_else(|| {
                JsValue::from_str(&format!("Unknown fitness function: {}", fitness_name))
            })?;

        let fitness = FitnessEvaluator::new(fitness_wrapper);

        let umda = UMDABuilder::<RealVector, f64, _, _>::new()
            .population_size(self.config.population_size)
            .selection_ratio(self.selection_ratio)
            .min_variance(self.min_variance)
            .learning_rate(self.learning_rate)
            .bounds(bounds)
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(evolution_error_to_js)?;

        let result = umda.run(&mut rng).map_err(evolution_error_to_js)?;

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    /// Run UMDA (built-in fitness) reporting progress per generation and allowing
    /// cancellation (AUDIT EV-34).
    ///
    /// `progressFn(generation, bestFitness)` is invoked once per generation before
    /// it is sampled/evaluated; return `false` to cancel and receive the
    /// best-so-far result. Backed by the native `UMDA::run_with_callback` hook.
    #[wasm_bindgen(js_name = optimizeWithProgress)]
    pub fn optimize_with_progress(
        &self,
        fitness_name: &str,
        progress_fn: &js_sys::Function,
    ) -> Result<OptimizationResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::uniform(
            Bounds::new(self.config.lower_bound, self.config.upper_bound),
            self.config.dimension,
        );

        let fitness_wrapper = FitnessWrapper::from_name(fitness_name, self.config.dimension)
            .ok_or_else(|| {
                JsValue::from_str(&format!("Unknown fitness function: {}", fitness_name))
            })?;

        let fitness = FitnessEvaluator::new(fitness_wrapper);

        let umda = UMDABuilder::<RealVector, f64, _, _>::new()
            .population_size(self.config.population_size)
            .selection_ratio(self.selection_ratio)
            .min_variance(self.min_variance)
            .learning_rate(self.learning_rate)
            .bounds(bounds)
            .fitness(fitness)
            .max_generations(self.config.max_generations)
            .build()
            .map_err(evolution_error_to_js)?;

        // Capture a thrown JS-callback error (can't cross the FnMut->bool
        // boundary) and surface it after the run.
        let mut progress_err: Option<JsValue> = None;
        let result = umda
            .run_with_callback(&mut rng, |generation, best_fitness| {
                match report_progress(progress_fn, generation, best_fitness) {
                    Ok(cont) => cont,
                    Err(e) => {
                        progress_err = Some(e);
                        false
                    }
                }
            })
            .map_err(evolution_error_to_js)?;
        if let Some(e) = progress_err {
            return Err(e);
        }

        Ok(OptimizationResult::new(
            result.best_genome.genes().to_vec(),
            result.best_fitness,
            result.generations,
            result.evaluations,
            result.stats.best_fitness_history(),
        ))
    }

    // Note: Custom fitness functions are not supported for UMDA because
    // UMDA requires Sync which JS functions don't provide.
    // Use the built-in fitness functions via the `optimize` method.
}

// ============================================================================
// ZDT Multi-Objective Test Problems
// ============================================================================

use fugue_evo::fitness::benchmarks::{Zdt1, Zdt2, Zdt3};

/// Multi-objective test problem type
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum ZdtProblem {
    /// ZDT1: Convex Pareto front
    Zdt1,
    /// ZDT2: Non-convex Pareto front
    Zdt2,
    /// ZDT3: Disconnected Pareto front
    Zdt3,
}

/// NSGA-II optimizer with built-in ZDT test problems
#[wasm_bindgen]
impl Nsga2Optimizer {
    /// Optimize a ZDT test problem
    #[wasm_bindgen(js_name = optimizeZdt)]
    pub fn optimize_zdt(&self, problem: ZdtProblem) -> Result<MultiObjectiveResult, JsValue> {
        let mut rng = create_rng(self.config.seed);
        let bounds = MultiBounds::uniform(Bounds::new(0.0, 1.0), self.config.dimension);

        match problem {
            ZdtProblem::Zdt1 => {
                let zdt = Zdt1::new(self.config.dimension.max(2));
                self.run_zdt(zdt, bounds, &mut rng)
            }
            ZdtProblem::Zdt2 => {
                let zdt = Zdt2::new(self.config.dimension.max(2));
                self.run_zdt(zdt, bounds, &mut rng)
            }
            ZdtProblem::Zdt3 => {
                let zdt = Zdt3::new(self.config.dimension.max(2));
                self.run_zdt(zdt, bounds, &mut rng)
            }
        }
    }

    fn run_zdt<Z: ZdtEvaluator>(
        &self,
        zdt: Z,
        bounds: MultiBounds,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<MultiObjectiveResult, JsValue> {
        // Count real evaluations rather than fabricating a formula (AUDIT EV-33).
        let fitness = CountingMoFitness::new(ZdtFitness::new(zdt));

        let nsga2 = Nsga2::<RealVector, _, _, _>::new(self.config.population_size)
            .with_bounds(bounds.clone());

        let result = nsga2
            .run(
                &fitness,
                &SbxCrossover::new(self.crossover_eta),
                &PolynomialMutation::new(self.mutation_eta),
                &bounds,
                self.config.max_generations,
                rng,
            )
            .map_err(evolution_error_to_js)?;

        let pareto_front: Vec<ParetoSolution> = result
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ParetoSolution::new(ind.genome.genes().to_vec(), ind.objectives.clone()))
            .collect();

        Ok(MultiObjectiveResult::new(
            pareto_front,
            self.config.max_generations,
            fitness.evaluations(),
        ))
    }
}

/// Trait for ZDT evaluators
trait ZdtEvaluator: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> [f64; 2];
}

impl ZdtEvaluator for Zdt1 {
    fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        Zdt1::evaluate(self, x)
    }
}

impl ZdtEvaluator for Zdt2 {
    fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        Zdt2::evaluate(self, x)
    }
}

impl ZdtEvaluator for Zdt3 {
    fn evaluate(&self, x: &[f64]) -> [f64; 2] {
        Zdt3::evaluate(self, x)
    }
}

/// Multi-objective fitness for ZDT problems
struct ZdtFitness<Z> {
    zdt: Z,
}

impl<Z> ZdtFitness<Z> {
    fn new(zdt: Z) -> Self {
        Self { zdt }
    }
}

impl<Z: ZdtEvaluator> MultiObjectiveFitness<RealVector> for ZdtFitness<Z> {
    fn num_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, genome: &RealVector) -> Vec<f64> {
        let [f1, f2] = self.zdt.evaluate(genome.genes());
        vec![f1, f2]
    }
}

// ============================================================================
// SteppedRealOptimizer — incremental / cancellable GA (AUDIT EV-34)
// ============================================================================

use fugue_evo::algorithms::simple_ga::{SimpleGA, SimpleGaRun};
use fugue_evo::termination::MaxGenerations;

/// Concrete `SimpleGA` type driven by the incremental step API.
type SteppedGa = SimpleGA<
    RealVector,
    f64,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
    FitnessEvaluator,
    MaxGenerations,
>;

/// One in-progress GA run (algorithm + its mutable stepping state + RNG).
struct SteppedRun {
    ga: SteppedGa,
    state: SimpleGaRun<RealVector, f64>,
    rng: rand::rngs::StdRng,
}

/// Build a `SteppedGa` for a built-in fitness function from a config snapshot.
fn build_stepped_ga(config: &OptimizationConfig, fitness_name: &str) -> Result<SteppedGa, JsValue> {
    let bounds = MultiBounds::uniform(
        Bounds::new(config.lower_bound, config.upper_bound),
        config.dimension,
    );
    let wrapper = FitnessWrapper::from_name(fitness_name, config.dimension)
        .ok_or_else(|| JsValue::from_str(&format!("Unknown fitness: {}", fitness_name)))?;

    SimpleGABuilder::<RealVector, f64, _, _, _, _, _>::new()
        .population_size(config.population_size)
        .bounds(bounds)
        .selection(TournamentSelection::new(config.tournament_size))
        .crossover(SbxCrossover::new(config.crossover_eta))
        .mutation(PolynomialMutation::new(config.mutation_eta))
        .fitness(FitnessEvaluator::new(wrapper))
        .max_generations(config.max_generations)
        .build()
        .map_err(evolution_error_to_js)
}

/// A real-vector GA that runs **incrementally**.
///
/// Instead of a single blocking `optimize()` call, JavaScript drives the
/// generation loop in chunks via [`SteppedRealOptimizer::step`], reads progress
/// between chunks (best fitness/genome, generation, evaluations, history), and
/// simply stops calling to cancel (AUDIT EV-34). This keeps a UI thread — or a
/// Web Worker that wants to `postMessage` progress — responsive.
///
/// Uses the same configuration surface as [`RealVectorOptimizer`] with a
/// built-in fitness function, driven through the native incremental stepping API
/// (`SimpleGA::init_run` / `step_generation` / `finish_run`).
#[wasm_bindgen]
pub struct SteppedRealOptimizer {
    config: OptimizationConfig,
    fitness_name: String,
    run: Option<SteppedRun>,
}

#[wasm_bindgen]
impl SteppedRealOptimizer {
    /// Create a new stepped optimizer for the given dimension.
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            fitness_name: "sphere".to_string(),
            run: None,
        }
    }

    /// Set the population size (resets any in-progress run).
    #[wasm_bindgen(js_name = setPopulationSize)]
    pub fn set_population_size(&mut self, size: usize) {
        self.config.set_population_size(size);
        self.run = None;
    }

    /// Set the maximum number of generations (resets any in-progress run).
    #[wasm_bindgen(js_name = setMaxGenerations)]
    pub fn set_max_generations(&mut self, gens: usize) {
        self.config.set_max_generations(gens);
        self.run = None;
    }

    /// Set the bounds for all dimensions (resets any in-progress run).
    #[wasm_bindgen(js_name = setBounds)]
    pub fn set_bounds(&mut self, lower: f64, upper: f64) {
        self.config.set_bounds(lower, upper);
        self.run = None;
    }

    /// Set the random seed, 0 for entropy (resets any in-progress run).
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.config.set_seed(seed);
        self.run = None;
    }

    /// Set the tournament size for selection (resets any in-progress run).
    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, size: usize) {
        self.config.set_tournament_size(size);
        self.run = None;
    }

    /// Set the built-in fitness function by name (resets any in-progress run).
    #[wasm_bindgen(js_name = setFitness)]
    pub fn set_fitness(&mut self, name: &str) {
        self.fitness_name = name.to_string();
        self.run = None;
    }

    /// Build the GA and evaluate the initial population, if not already started.
    fn ensure_started(&mut self) -> Result<(), JsValue> {
        if self.run.is_some() {
            return Ok(());
        }
        let ga = build_stepped_ga(&self.config, &self.fitness_name)?;
        let mut rng = create_rng(self.config.seed);
        let state = ga.init_run(&mut rng).map_err(evolution_error_to_js)?;
        self.run = Some(SteppedRun { ga, state, rng });
        Ok(())
    }

    /// Advance the optimization by up to `n_generations`.
    ///
    /// Returns `true` if the optimizer is still running (more generations remain
    /// before `maxGenerations`), or `false` once it has reached the generation
    /// budget or converged. Call repeatedly (e.g. `step(1)` per animation frame)
    /// and stop whenever you like to cancel.
    #[wasm_bindgen]
    pub fn step(&mut self, n_generations: usize) -> Result<bool, JsValue> {
        self.ensure_started()?;
        let run = self
            .run
            .as_mut()
            .ok_or_else(|| JsValue::from_str("optimizer not started"))?;
        for _ in 0..n_generations {
            let advanced = run
                .ga
                .step_generation(&mut run.state, &mut run.rng)
                .map_err(evolution_error_to_js)?;
            if !advanced {
                return Ok(false);
            }
        }
        Ok(!run.state.is_terminated())
    }

    /// The number of generations completed so far.
    #[wasm_bindgen(js_name = currentGeneration, getter)]
    pub fn current_generation(&self) -> usize {
        self.run.as_ref().map(|r| r.state.generation()).unwrap_or(0)
    }

    /// The best fitness found so far (higher is better).
    #[wasm_bindgen(js_name = bestFitness, getter)]
    pub fn best_fitness(&self) -> f64 {
        self.run
            .as_ref()
            .map(|r| r.state.best_fitness())
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// The best genome found so far.
    #[wasm_bindgen(js_name = bestGenome)]
    pub fn best_genome(&self) -> Vec<f64> {
        self.run
            .as_ref()
            .map(|r| r.state.best_genome().genes().to_vec())
            .unwrap_or_default()
    }

    /// Total fitness evaluations performed so far.
    #[wasm_bindgen(getter)]
    pub fn evaluations(&self) -> usize {
        self.run
            .as_ref()
            .map(|r| r.state.evaluations())
            .unwrap_or(0)
    }

    /// Per-generation best-so-far fitness trajectory.
    #[wasm_bindgen(js_name = fitnessHistory)]
    pub fn fitness_history(&self) -> Vec<f64> {
        self.run
            .as_ref()
            .map(|r| r.state.fitness_history().to_vec())
            .unwrap_or_default()
    }

    /// `true` once the generation budget is exhausted or the run converged.
    #[wasm_bindgen(js_name = isFinished, getter)]
    pub fn is_finished(&self) -> bool {
        self.run
            .as_ref()
            .map(|r| r.state.is_terminated())
            .unwrap_or(false)
    }

    /// `true` once the initial population has been built and evaluated.
    #[wasm_bindgen(js_name = isStarted, getter)]
    pub fn is_started(&self) -> bool {
        self.run.is_some()
    }

    /// Snapshot the current progress as an [`OptimizationResult`].
    ///
    /// Can be called at any point (starts the run if needed). Does not consume
    /// the optimizer, so stepping can continue afterward.
    #[wasm_bindgen(js_name = getResult)]
    pub fn get_result(&mut self) -> Result<OptimizationResult, JsValue> {
        self.ensure_started()?;
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| JsValue::from_str("optimizer not started"))?;
        Ok(OptimizationResult::new(
            run.state.best_genome().genes().to_vec(),
            run.state.best_fitness(),
            run.state.generation(),
            run.state.evaluations(),
            run.state.fitness_history().to_vec(),
        ))
    }
}

// ============================================================================
// IslandModelOptimizer — single-threaded island GA (AUDIT EV-77)
// ============================================================================
//
// The native `algorithms::island` module requires the `parallel` feature
// (rayon), which is disabled for the WASM build (`fugue-evo` is pulled in with
// `default-features = false, features = ["std"]`, and rayon threads are not
// available in wasm32). Rather than pull an unusable dependency into the wasm
// binary, this is a self-contained single-threaded island model built on top of
// the incremental stepping API (`SimpleGA::step_generation`) plus the migration
// helpers (`SimpleGaRun::best_genomes` / `SimpleGA::inject_migrants`). It mirrors
// the native model's Ring / FullyConnected / Star topologies with best-k
// migration.

/// Migration topology for the island model.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub enum IslandTopology {
    /// Each island sends its emigrants to the next island (index + 1, wrapping).
    #[default]
    Ring,
    /// Each island sends its emigrants to every other island.
    FullyConnected,
    /// All spokes send to the hub (island 0); the hub sends to every spoke.
    Star,
}

impl IslandTopology {
    /// Target island indices that `source` migrates to.
    fn targets(&self, source: usize, num_islands: usize) -> Vec<usize> {
        match self {
            IslandTopology::Ring => vec![(source + 1) % num_islands],
            IslandTopology::FullyConnected => (0..num_islands).filter(|&i| i != source).collect(),
            IslandTopology::Star => {
                if source == 0 {
                    (1..num_islands).collect()
                } else {
                    vec![0]
                }
            }
        }
    }
}

/// A single-threaded island-model GA over a built-in fitness function.
///
/// Runs `numIslands` independent populations that evolve with the same
/// incremental GA and periodically exchange their best individuals according to
/// the chosen [`IslandTopology`] (AUDIT EV-77).
#[wasm_bindgen]
pub struct IslandModelOptimizer {
    config: OptimizationConfig,
    fitness_name: String,
    num_islands: usize,
    migration_interval: usize,
    migration_size: usize,
    topology: IslandTopology,
}

#[wasm_bindgen]
impl IslandModelOptimizer {
    /// Create an island model with `num_islands` islands of the given dimension.
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize, num_islands: usize) -> Self {
        Self {
            config: OptimizationConfig::new(dimension),
            fitness_name: "sphere".to_string(),
            num_islands: num_islands.max(1),
            migration_interval: 5,
            migration_size: 1,
            topology: IslandTopology::Ring,
        }
    }

    /// Set the population size per island.
    #[wasm_bindgen(js_name = setPopulationSize)]
    pub fn set_population_size(&mut self, size: usize) {
        self.config.set_population_size(size);
    }

    /// Set the maximum number of generations.
    #[wasm_bindgen(js_name = setMaxGenerations)]
    pub fn set_max_generations(&mut self, gens: usize) {
        self.config.set_max_generations(gens);
    }

    /// Set the bounds for all dimensions.
    #[wasm_bindgen(js_name = setBounds)]
    pub fn set_bounds(&mut self, lower: f64, upper: f64) {
        self.config.set_bounds(lower, upper);
    }

    /// Set the random seed (0 for entropy).
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.config.set_seed(seed);
    }

    /// Set the tournament size for selection.
    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, size: usize) {
        self.config.set_tournament_size(size);
    }

    /// Set the built-in fitness function by name.
    #[wasm_bindgen(js_name = setFitness)]
    pub fn set_fitness(&mut self, name: &str) {
        self.fitness_name = name.to_string();
    }

    /// Set the number of generations between migration events.
    #[wasm_bindgen(js_name = setMigrationInterval)]
    pub fn set_migration_interval(&mut self, interval: usize) {
        self.migration_interval = interval.max(1);
    }

    /// Set the number of (best) individuals each island sends per migration.
    #[wasm_bindgen(js_name = setMigrationSize)]
    pub fn set_migration_size(&mut self, size: usize) {
        self.migration_size = size.max(1);
    }

    /// Set the migration topology.
    #[wasm_bindgen(js_name = setTopology)]
    pub fn set_topology(&mut self, topology: IslandTopology) {
        self.topology = topology;
    }

    /// The number of islands.
    #[wasm_bindgen(js_name = numIslands, getter)]
    pub fn num_islands(&self) -> usize {
        self.num_islands
    }

    /// Run the island model to completion and return the global-best result.
    ///
    /// The returned [`OptimizationResult`] holds the best genome/fitness across
    /// all islands, the summed evaluation count, and a per-generation trajectory
    /// of the global best-so-far fitness.
    #[wasm_bindgen]
    pub fn optimize(&self) -> Result<OptimizationResult, JsValue> {
        let n = self.num_islands;

        // Build the islands. Each gets a distinct but deterministic RNG seed so
        // the populations differ yet the whole run is reproducible.
        let mut islands: Vec<SteppedRun> = Vec::with_capacity(n);
        for i in 0..n {
            let ga = build_stepped_ga(&self.config, &self.fitness_name)?;
            let island_seed = if self.config.seed == 0 {
                0 // entropy for every island
            } else {
                self.config.seed.wrapping_add(i as u64 + 1)
            };
            let mut rng = create_rng(island_seed);
            let state = ga.init_run(&mut rng).map_err(evolution_error_to_js)?;
            islands.push(SteppedRun { ga, state, rng });
        }

        let mut history: Vec<f64> = Vec::with_capacity(self.config.max_generations);

        for generation in 0..self.config.max_generations {
            // Advance every island one generation.
            for island in islands.iter_mut() {
                island
                    .ga
                    .step_generation(&mut island.state, &mut island.rng)
                    .map_err(evolution_error_to_js)?;
            }

            // Migrate on the configured interval (never on generation 0).
            if n > 1 && generation > 0 && generation % self.migration_interval == 0 {
                // Snapshot emigrants from every source first (immutable reads),
                // then inject into targets (mutable writes).
                let mut deliveries: Vec<(usize, Vec<RealVector>)> = Vec::new();
                for source in 0..n {
                    let migrants = islands[source].state.best_genomes(self.migration_size);
                    for target in self.topology.targets(source, n) {
                        deliveries.push((target, migrants.clone()));
                    }
                }
                for (target, migrants) in deliveries {
                    let island = &mut islands[target];
                    island.ga.inject_migrants(&mut island.state, migrants);
                }
            }

            // Track the global best-so-far across islands.
            let global_best = islands
                .iter()
                .map(|isl| isl.state.best_fitness())
                .fold(f64::NEG_INFINITY, f64::max);
            history.push(global_best);
        }

        // Collect the global-best solution and summed evaluations.
        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_genome: Vec<f64> = Vec::new();
        let mut total_evaluations = 0usize;
        for island in &islands {
            total_evaluations += island.state.evaluations();
            let f = island.state.best_fitness();
            if f > best_fitness {
                best_fitness = f;
                best_genome = island.state.best_genome().genes().to_vec();
            }
        }

        Ok(OptimizationResult::new(
            best_genome,
            best_fitness,
            self.config.max_generations,
            total_evaluations,
            history,
        ))
    }
}
