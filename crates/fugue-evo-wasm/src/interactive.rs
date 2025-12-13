//! Interactive Evolution WASM bindings
//!
//! Provides human-in-the-loop optimization where fitness is derived from user feedback.

use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use fugue_evo::genome::bounds::MultiBounds;
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::RealValuedGenome;
use fugue_evo::interactive::prelude::*;
use fugue_evo::operators::crossover::SbxCrossover;
use fugue_evo::operators::mutation::PolynomialMutation;
use fugue_evo::operators::selection::TournamentSelection;

// ============================================================================
// Evaluation Mode
// ============================================================================

/// Evaluation mode for interactive evolution
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Default)]
pub enum WasmEvaluationMode {
    /// Rate each candidate on a numeric scale
    #[default]
    Rating,
    /// Compare pairs and select the better one
    Pairwise,
    /// Select favorites from a batch
    BatchSelection,
}

impl From<WasmEvaluationMode> for EvaluationMode {
    fn from(mode: WasmEvaluationMode) -> Self {
        match mode {
            WasmEvaluationMode::Rating => EvaluationMode::Rating,
            WasmEvaluationMode::Pairwise => EvaluationMode::Pairwise,
            WasmEvaluationMode::BatchSelection => EvaluationMode::BatchSelection,
        }
    }
}

// ============================================================================
// Candidate (JS-friendly wrapper)
// ============================================================================

/// A candidate for evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmCandidate {
    /// Unique identifier
    id: usize,
    /// Genome values
    #[wasm_bindgen(skip)]
    pub genome: Vec<f64>,
    /// Current fitness estimate (if any)
    fitness: Option<f64>,
    /// Number of evaluations
    evaluation_count: usize,
}

#[wasm_bindgen]
impl WasmCandidate {
    /// Get the candidate ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the genome as a Float64Array
    #[wasm_bindgen(getter)]
    pub fn genome(&self) -> Vec<f64> {
        self.genome.clone()
    }

    /// Get the current fitness estimate
    #[wasm_bindgen(getter)]
    pub fn fitness(&self) -> Option<f64> {
        self.fitness
    }

    /// Get the number of times this candidate has been evaluated
    #[wasm_bindgen(js_name = evaluationCount, getter)]
    pub fn evaluation_count(&self) -> usize {
        self.evaluation_count
    }

    /// Check if this candidate has been evaluated
    #[wasm_bindgen(js_name = isEvaluated)]
    pub fn is_evaluated(&self) -> bool {
        self.evaluation_count > 0
    }

    /// Serialize to JSON
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl From<&Candidate<RealVector>> for WasmCandidate {
    fn from(c: &Candidate<RealVector>) -> Self {
        Self {
            id: c.id.0,
            genome: c.genome.genes().to_vec(),
            fitness: c.fitness_estimate,
            evaluation_count: c.evaluation_count,
        }
    }
}

// ============================================================================
// Evaluation Request (JS-friendly)
// ============================================================================

/// Type of evaluation request
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RequestType {
    /// Rate candidates on a scale
    Rating,
    /// Compare two candidates
    Pairwise,
    /// Select favorites from batch
    BatchSelection,
}

/// Evaluation request sent to the user
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmEvaluationRequest {
    request_type: RequestType,
    #[wasm_bindgen(skip)]
    pub candidates: Vec<WasmCandidate>,
    /// For batch selection: how many to select
    select_count: usize,
    /// Rating scale min
    scale_min: f64,
    /// Rating scale max
    scale_max: f64,
}

#[wasm_bindgen]
impl WasmEvaluationRequest {
    /// Get the request type
    #[wasm_bindgen(js_name = requestType, getter)]
    pub fn request_type(&self) -> RequestType {
        self.request_type
    }

    /// Get the number of candidates
    #[wasm_bindgen(js_name = candidateCount, getter)]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Get a candidate by index
    #[wasm_bindgen(js_name = getCandidate)]
    pub fn get_candidate(&self, index: usize) -> Option<WasmCandidate> {
        self.candidates.get(index).cloned()
    }

    /// Get all candidate IDs
    #[wasm_bindgen(js_name = getCandidateIds)]
    pub fn get_candidate_ids(&self) -> Vec<usize> {
        self.candidates.iter().map(|c| c.id).collect()
    }

    /// Get all candidates as JSON
    #[wasm_bindgen(js_name = getCandidatesJson)]
    pub fn get_candidates_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.candidates).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// For batch selection: how many to select
    #[wasm_bindgen(js_name = selectCount, getter)]
    pub fn select_count(&self) -> usize {
        self.select_count
    }

    /// Rating scale minimum
    #[wasm_bindgen(js_name = scaleMin, getter)]
    pub fn scale_min(&self) -> f64 {
        self.scale_min
    }

    /// Rating scale maximum
    #[wasm_bindgen(js_name = scaleMax, getter)]
    pub fn scale_max(&self) -> f64 {
        self.scale_max
    }

    /// Serialize to JSON
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ============================================================================
// Step Result
// ============================================================================

/// Result type from stepping the algorithm
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum StepResultType {
    /// Algorithm needs user evaluation
    NeedsEvaluation,
    /// Generation complete
    GenerationComplete,
    /// Evolution finished
    Complete,
}

/// Result of stepping the interactive algorithm
#[derive(Clone, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmStepResult {
    result_type: StepResultType,
    /// Evaluation request (if NeedsEvaluation)
    #[wasm_bindgen(skip)]
    pub request: Option<WasmEvaluationRequest>,
    /// Generation number (if GenerationComplete)
    generation: Option<usize>,
    /// Best fitness (if GenerationComplete)
    best_fitness: Option<f64>,
    /// Coverage (if GenerationComplete)
    coverage: Option<f64>,
    /// Best candidates (if Complete)
    #[wasm_bindgen(skip)]
    pub best_candidates: Vec<WasmCandidate>,
    /// Total evaluations (if Complete)
    total_evaluations: Option<usize>,
    /// Termination reason (if Complete)
    termination_reason: Option<String>,
}

#[wasm_bindgen]
impl WasmStepResult {
    /// Get the result type
    #[wasm_bindgen(js_name = resultType, getter)]
    pub fn result_type(&self) -> StepResultType {
        self.result_type
    }

    /// Get the evaluation request (if NeedsEvaluation)
    #[wasm_bindgen(js_name = getRequest)]
    pub fn get_request(&self) -> Option<WasmEvaluationRequest> {
        self.request.clone()
    }

    /// Get the generation number (if GenerationComplete)
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> Option<usize> {
        self.generation
    }

    /// Get the best fitness (if GenerationComplete)
    #[wasm_bindgen(js_name = bestFitness, getter)]
    pub fn best_fitness(&self) -> Option<f64> {
        self.best_fitness
    }

    /// Get the coverage (if GenerationComplete)
    #[wasm_bindgen(getter)]
    pub fn coverage(&self) -> Option<f64> {
        self.coverage
    }

    /// Get the number of best candidates (if Complete)
    #[wasm_bindgen(js_name = bestCandidateCount, getter)]
    pub fn best_candidate_count(&self) -> usize {
        self.best_candidates.len()
    }

    /// Get a best candidate by index (if Complete)
    #[wasm_bindgen(js_name = getBestCandidate)]
    pub fn get_best_candidate(&self, index: usize) -> Option<WasmCandidate> {
        self.best_candidates.get(index).cloned()
    }

    /// Get total evaluations (if Complete)
    #[wasm_bindgen(js_name = totalEvaluations, getter)]
    pub fn total_evaluations(&self) -> Option<usize> {
        self.total_evaluations
    }

    /// Get termination reason (if Complete)
    #[wasm_bindgen(js_name = terminationReason, getter)]
    pub fn termination_reason(&self) -> Option<String> {
        self.termination_reason.clone()
    }

    /// Check if needs evaluation
    #[wasm_bindgen(js_name = needsEvaluation)]
    pub fn needs_evaluation(&self) -> bool {
        matches!(self.result_type, StepResultType::NeedsEvaluation)
    }

    /// Check if generation complete
    #[wasm_bindgen(js_name = isGenerationComplete)]
    pub fn is_generation_complete(&self) -> bool {
        matches!(self.result_type, StepResultType::GenerationComplete)
    }

    /// Check if complete
    #[wasm_bindgen(js_name = isComplete)]
    pub fn is_complete(&self) -> bool {
        matches!(self.result_type, StepResultType::Complete)
    }
}

// ============================================================================
// Interactive Optimizer
// ============================================================================

/// Interactive evolution optimizer for human-in-the-loop optimization
#[wasm_bindgen]
pub struct InteractiveOptimizer {
    iga: InteractiveGA<
        RealVector,
        TournamentSelection,
        SbxCrossover,
        PolynomialMutation,
    >,
    rng: rand::rngs::StdRng,
}

#[wasm_bindgen]
impl InteractiveOptimizer {
    /// Create a new interactive optimizer
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Self {
        let config = InteractiveGAConfig::default();
        let bounds = MultiBounds::symmetric(5.0, dimension);
        let selection = TournamentSelection::new(2);
        let crossover = SbxCrossover::new(15.0);
        let mutation = PolynomialMutation::new(20.0);

        let iga = InteractiveGA::new(config, Some(bounds), selection, crossover, mutation);
        let rng = rand::rngs::StdRng::from_entropy();

        Self { iga, rng }
    }

    /// Create with custom configuration
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(
        dimension: usize,
        population_size: usize,
        evaluation_mode: WasmEvaluationMode,
        batch_size: usize,
        max_generations: usize,
    ) -> Self {
        let mut config = InteractiveGAConfig::default();
        config.population_size = population_size;
        config.evaluation_mode = evaluation_mode.into();
        config.batch_size = batch_size;
        config.max_generations = max_generations;

        let bounds = MultiBounds::symmetric(5.0, dimension);
        let selection = TournamentSelection::new(2);
        let crossover = SbxCrossover::new(15.0);
        let mutation = PolynomialMutation::new(20.0);

        let iga = InteractiveGA::new(config, Some(bounds), selection, crossover, mutation);
        let rng = rand::rngs::StdRng::from_entropy();

        Self { iga, rng }
    }

    /// Set the random seed
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = rand::rngs::StdRng::seed_from_u64(seed);
    }

    /// Set bounds for all dimensions
    /// Note: Must be called before initialization via withConfig
    #[wasm_bindgen(js_name = setBounds)]
    pub fn set_bounds(&mut self, _lower: f64, _upper: f64) {
        // Note: Bounds are set at construction time in current API
        // Use withConfig for custom bounds
    }

    /// Step the algorithm forward
    #[wasm_bindgen]
    pub fn step(&mut self) -> WasmStepResult {
        match self.iga.step(&mut self.rng) {
            StepResult::NeedsEvaluation(request) => {
                let wasm_request = convert_request(&request);
                WasmStepResult {
                    result_type: StepResultType::NeedsEvaluation,
                    request: Some(wasm_request),
                    generation: None,
                    best_fitness: None,
                    coverage: None,
                    best_candidates: vec![],
                    total_evaluations: None,
                    termination_reason: None,
                }
            }
            StepResult::GenerationComplete {
                generation,
                best_fitness,
                coverage,
            } => WasmStepResult {
                result_type: StepResultType::GenerationComplete,
                request: None,
                generation: Some(generation),
                best_fitness,
                coverage: Some(coverage),
                best_candidates: vec![],
                total_evaluations: None,
                termination_reason: None,
            },
            StepResult::Complete(result) => {
                let best_candidates: Vec<WasmCandidate> =
                    result.best_candidates.iter().map(WasmCandidate::from).collect();
                WasmStepResult {
                    result_type: StepResultType::Complete,
                    request: None,
                    generation: Some(result.generations),
                    best_fitness: result.best_candidates.first().and_then(|c| c.fitness()),
                    coverage: None,
                    best_candidates,
                    total_evaluations: Some(result.total_evaluations),
                    termination_reason: Some(result.termination_reason),
                }
            }
        }
    }

    /// Provide ratings for candidates
    /// ratings: array of [candidate_id, rating] pairs
    #[wasm_bindgen(js_name = provideRatings)]
    pub fn provide_ratings(&mut self, ratings: &[f64]) -> Result<(), JsValue> {
        if ratings.len() % 2 != 0 {
            return Err(JsValue::from_str("Ratings array must have even length (id, rating pairs)"));
        }

        let ratings_vec: Vec<(CandidateId, f64)> = ratings
            .chunks(2)
            .map(|chunk| (CandidateId(chunk[0] as usize), chunk[1]))
            .collect();

        let response = EvaluationResponse::Ratings(ratings_vec);
        self.iga.provide_response(response);
        Ok(())
    }

    /// Provide pairwise comparison result
    /// winner_id: ID of the winning candidate, or -1 for a tie
    #[wasm_bindgen(js_name = provideComparison)]
    pub fn provide_comparison(&mut self, winner_id: i32) {
        let response = if winner_id < 0 {
            EvaluationResponse::PairwiseWinner(None)
        } else {
            EvaluationResponse::PairwiseWinner(Some(CandidateId(winner_id as usize)))
        };
        self.iga.provide_response(response);
    }

    /// Provide batch selection result
    /// selected_ids: array of selected candidate IDs
    #[wasm_bindgen(js_name = provideSelection)]
    pub fn provide_selection(&mut self, selected_ids: &[usize]) {
        let ids: Vec<CandidateId> = selected_ids.iter().map(|&id| CandidateId(id)).collect();
        let response = EvaluationResponse::BatchSelected(ids);
        self.iga.provide_response(response);
    }

    /// Skip the current evaluation
    #[wasm_bindgen]
    pub fn skip(&mut self) {
        self.iga.provide_response(EvaluationResponse::Skip);
    }

    /// Get the current generation
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.iga.session().generation
    }

    /// Get the population size
    #[wasm_bindgen(js_name = populationSize, getter)]
    pub fn population_size(&self) -> usize {
        self.iga.session().population.len()
    }

    /// Get coverage statistics
    #[wasm_bindgen(js_name = getCoverage)]
    pub fn get_coverage(&self) -> f64 {
        self.iga.coverage_stats().coverage
    }

    /// Get the current best candidate
    #[wasm_bindgen(js_name = getBestCandidate)]
    pub fn get_best_candidate(&self) -> Option<WasmCandidate> {
        self.iga
            .session()
            .population
            .iter()
            .filter(|c| c.fitness_estimate.is_some())
            .max_by(|a, b| {
                a.fitness_estimate
                    .partial_cmp(&b.fitness_estimate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(WasmCandidate::from)
    }

    /// Get all candidates as JSON
    #[wasm_bindgen(js_name = getPopulationJson)]
    pub fn get_population_json(&self) -> Result<String, JsValue> {
        let candidates: Vec<WasmCandidate> = self
            .iga
            .session()
            .population
            .iter()
            .map(WasmCandidate::from)
            .collect();
        serde_json::to_string(&candidates).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Export session state to JSON (for saving)
    #[wasm_bindgen(js_name = exportSession)]
    pub fn export_session(&self) -> Result<String, JsValue> {
        self.iga
            .session()
            .to_json()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn convert_request(request: &EvaluationRequest<RealVector>) -> WasmEvaluationRequest {
    match request {
        EvaluationRequest::RateCandidates { candidates, scale } => WasmEvaluationRequest {
            request_type: RequestType::Rating,
            candidates: candidates.iter().map(WasmCandidate::from).collect(),
            select_count: 0,
            scale_min: scale.min,
            scale_max: scale.max,
        },
        EvaluationRequest::PairwiseComparison {
            candidate_a,
            candidate_b,
            ..
        } => WasmEvaluationRequest {
            request_type: RequestType::Pairwise,
            candidates: vec![WasmCandidate::from(candidate_a), WasmCandidate::from(candidate_b)],
            select_count: 1,
            scale_min: 0.0,
            scale_max: 1.0,
        },
        EvaluationRequest::BatchSelection {
            candidates,
            select_count,
            ..
        } => WasmEvaluationRequest {
            request_type: RequestType::BatchSelection,
            candidates: candidates.iter().map(WasmCandidate::from).collect(),
            select_count: *select_count,
            scale_min: 0.0,
            scale_max: 1.0,
        },
    }
}
