//! WASM integration tests for fugue-evo-wasm
//!
//! Run with: wasm-pack test --headless --chrome

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use fugue_evo_wasm::{
    version, Algorithm, BitStringOptimizer, EvolutionStrategyOptimizer, Nsga2Optimizer,
    PermutationOptimizer, RealVectorOptimizer, SymbolicRegressionOptimizer,
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_version() {
    let v = version();
    assert!(!v.is_empty());
    assert!(v.starts_with("0."));
}

// ============================================================================
// RealVectorOptimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_real_vector_optimizer_creation() {
    let optimizer = RealVectorOptimizer::new(10);
    let config = optimizer.get_config().expect("Should get config");
    assert!(config.contains("\"dimension\":10"));
}

#[wasm_bindgen_test]
fn test_real_vector_optimizer_configuration() {
    let mut optimizer = RealVectorOptimizer::new(5);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_seed(42);
    optimizer.set_tournament_size(5);
    optimizer.set_crossover_eta(15.0);
    optimizer.set_mutation_eta(15.0);

    let config = optimizer.get_config().expect("Should get config");
    assert!(config.contains("\"population_size\":50"));
    assert!(config.contains("\"max_generations\":100"));
}

#[wasm_bindgen_test]
fn test_real_vector_sphere_optimization() {
    let mut optimizer = RealVectorOptimizer::new(3);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_fitness("sphere");
    optimizer.set_seed(12345);

    let result = optimizer.optimize().expect("Optimization should succeed");

    // Sphere optimum is at origin with fitness 0
    assert!(result.best_fitness() >= 0.0);
    assert_eq!(result.generations(), 50);
    assert!(result.evaluations() > 0);
    assert_eq!(result.best_genome().len(), 3);
}

#[wasm_bindgen_test]
fn test_real_vector_rastrigin_optimization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_bounds(-5.12, 5.12);
    optimizer.set_fitness("rastrigin");
    optimizer.set_seed(42);

    let result = optimizer.optimize().expect("Optimization should succeed");

    // Rastrigin optimum is at origin with fitness 0
    assert!(result.best_fitness() >= 0.0);
}

#[wasm_bindgen_test]
fn test_real_vector_rosenbrock_optimization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_fitness("rosenbrock");
    optimizer.set_seed(42);

    let result = optimizer.optimize().expect("Optimization should succeed");
    assert!(result.best_fitness() >= 0.0);
}

// ============================================================================
// CMA-ES Algorithm Tests (via RealVectorOptimizer)
// ============================================================================

#[wasm_bindgen_test]
fn test_cmaes_algorithm() {
    let mut optimizer = RealVectorOptimizer::new(5);
    optimizer.set_algorithm(Algorithm::CmaES);
    optimizer.set_cmaes_sigma(0.5);
    optimizer.set_population_size(20);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_fitness("sphere");
    optimizer.set_seed(42);

    let result = optimizer.optimize().expect("CMA-ES should succeed");
    assert!(result.best_fitness() >= 0.0);
}

// ============================================================================
// BitStringOptimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_bitstring_optimizer_creation() {
    let _optimizer = BitStringOptimizer::new(20);
}

#[wasm_bindgen_test]
fn test_bitstring_onemax() {
    let mut optimizer = BitStringOptimizer::new(20);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_seed(42);

    let result = optimizer.solve_one_max().expect("OneMax should succeed");

    // OneMax optimum is all 1s with fitness = length
    assert!(result.best_fitness() <= 20.0);
    assert!(result.best_fitness() > 10.0); // Should find a decent solution
    assert_eq!(result.best_genome().len(), 20);
}

#[wasm_bindgen_test]
fn test_bitstring_configuration() {
    let mut optimizer = BitStringOptimizer::new(30);
    optimizer.set_population_size(100);
    optimizer.set_max_generations(50);
    optimizer.set_flip_probability(0.05);
    optimizer.set_tournament_size(3);
    optimizer.set_seed(123);

    let result = optimizer.solve_one_max().expect("OneMax should succeed");
    assert!(result.generations() == 50);
}

// ============================================================================
// PermutationOptimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_permutation_optimizer_creation() {
    let _optimizer = PermutationOptimizer::new(10);
}

#[wasm_bindgen_test]
fn test_permutation_configuration() {
    let mut optimizer = PermutationOptimizer::new(8);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(50);
    optimizer.set_tournament_size(3);
    optimizer.set_seed(42);
    // Configuration verified by not panicking
}

// ============================================================================
// Nsga2Optimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_nsga2_optimizer_creation() {
    let _optimizer = Nsga2Optimizer::new(5, 2);
}

#[wasm_bindgen_test]
fn test_nsga2_configuration() {
    let mut optimizer = Nsga2Optimizer::new(3, 2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_bounds(0.0, 1.0);
    optimizer.set_crossover_eta(15.0);
    optimizer.set_mutation_eta(20.0);
    optimizer.set_seed(42);
    // Configuration verified by not panicking
}

// ============================================================================
// EvolutionStrategyOptimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_evolution_strategy_creation() {
    let _optimizer = EvolutionStrategyOptimizer::new(5);
}

#[wasm_bindgen_test]
fn test_evolution_strategy_configuration() {
    let mut optimizer = EvolutionStrategyOptimizer::new(5);
    optimizer.set_mu(10);
    optimizer.set_lambda(50);
    optimizer.set_initial_sigma(0.5);
    optimizer.set_self_adaptive(true);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_seed(42);
    // Configuration verified by not panicking
}

// ============================================================================
// SymbolicRegressionOptimizer Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_symbolic_regression_creation() {
    let _optimizer = SymbolicRegressionOptimizer::new();
}

#[wasm_bindgen_test]
fn test_symbolic_regression_default() {
    let _optimizer = SymbolicRegressionOptimizer::default();
}

#[wasm_bindgen_test]
fn test_symbolic_regression_configuration() {
    let mut optimizer = SymbolicRegressionOptimizer::new();
    optimizer.set_population_size(50);
    optimizer.set_max_generations(20);
    optimizer.set_tournament_size(3);
    optimizer.set_max_tree_depth(5);
    optimizer.set_seed(42);
    // Configuration verified by not panicking
}

#[wasm_bindgen_test]
fn test_symbolic_regression_simple_data() {
    let mut optimizer = SymbolicRegressionOptimizer::new();
    optimizer.set_population_size(50);
    optimizer.set_max_generations(20);
    optimizer.set_max_tree_depth(4);
    optimizer.set_seed(42);

    // Simple linear data: y = 2*x
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![0.0, 2.0, 4.0, 6.0, 8.0];

    let result = optimizer
        .optimize_data(&x_data, &y_data, 1)
        .expect("Symbolic regression should succeed");

    assert!(!result.expression().is_empty());
    assert!(result.tree_depth() > 0);
    assert!(result.tree_size() > 0);
    assert_eq!(result.generations(), 20);
}

#[wasm_bindgen_test]
fn test_symbolic_regression_result_json() {
    let mut optimizer = SymbolicRegressionOptimizer::new();
    optimizer.set_population_size(20);
    optimizer.set_max_generations(10);
    optimizer.set_seed(42);

    let x_data = vec![1.0, 2.0, 3.0];
    let y_data = vec![1.0, 4.0, 9.0]; // y = x^2

    let result = optimizer
        .optimize_data(&x_data, &y_data, 1)
        .expect("Should succeed");

    let json = result.to_json().expect("Should serialize to JSON");
    assert!(json.contains("expression"));
    assert!(json.contains("best_fitness"));
}

// ============================================================================
// Result Serialization Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_optimization_result_serialization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(20);
    optimizer.set_max_generations(10);
    optimizer.set_fitness("sphere");
    optimizer.set_seed(42);

    let result = optimizer.optimize().expect("Should succeed");
    let json = result.to_json().expect("Should serialize to JSON");

    assert!(json.contains("best_fitness"));
    assert!(json.contains("best_genome"));
    assert!(json.contains("generations"));
}

#[wasm_bindgen_test]
fn test_bitstring_result_serialization() {
    let mut optimizer = BitStringOptimizer::new(10);
    optimizer.set_population_size(20);
    optimizer.set_max_generations(10);
    optimizer.set_seed(42);

    let result = optimizer.solve_one_max().expect("Should succeed");
    let json = result.to_json().expect("Should serialize to JSON");

    assert!(json.contains("best_fitness"));
    assert!(json.contains("best_genome"));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[wasm_bindgen_test]
fn test_invalid_fitness_name() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_fitness("invalid_fitness_name");

    let result = optimizer.optimize();
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_symbolic_regression_invalid_data() {
    let optimizer = SymbolicRegressionOptimizer::new();

    // Mismatched data sizes
    let x_data = vec![1.0, 2.0, 3.0];
    let y_data = vec![1.0, 2.0]; // Wrong length

    let result = optimizer.optimize_data(&x_data, &y_data, 1);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_single_dimension() {
    let mut optimizer = RealVectorOptimizer::new(1);
    optimizer.set_population_size(10);
    optimizer.set_max_generations(10);
    optimizer.set_fitness("sphere");
    optimizer.set_seed(42);

    let result = optimizer.optimize().expect("Should succeed with 1D");
    assert_eq!(result.best_genome().len(), 1);
}

#[wasm_bindgen_test]
fn test_zero_seed_uses_entropy() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(10);
    optimizer.set_max_generations(5);
    optimizer.set_fitness("sphere");
    optimizer.set_seed(0); // 0 means use entropy

    let result = optimizer
        .optimize()
        .expect("Should succeed with entropy seed");
    assert!(result.evaluations() > 0);
}

// ============================================================================
// New Fitness Functions Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_ackley_optimization() {
    let mut optimizer = RealVectorOptimizer::new(3);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_fitness("ackley");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Ackley optimization should succeed");
    assert!(result.best_fitness() >= 0.0);
}

#[wasm_bindgen_test]
fn test_griewank_optimization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-10.0, 10.0);
    optimizer.set_fitness("griewank");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Griewank optimization should succeed");
    assert!(result.best_fitness() >= 0.0);
}

#[wasm_bindgen_test]
fn test_schwefel_optimization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-500.0, 500.0);
    optimizer.set_fitness("schwefel");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Schwefel optimization should succeed");
    assert!(result.evaluations() > 0);
}

#[wasm_bindgen_test]
fn test_levy_optimization() {
    let mut optimizer = RealVectorOptimizer::new(2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-10.0, 10.0);
    optimizer.set_fitness("levy");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Levy optimization should succeed");
    assert!(result.best_fitness() >= 0.0);
}

// ============================================================================
// BitString Extended Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_bitstring_leading_ones() {
    let mut optimizer = BitStringOptimizer::new(20);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_seed(42);

    let result = optimizer
        .solve_leading_ones()
        .expect("LeadingOnes should succeed");
    assert!(result.best_fitness() <= 20.0);
    assert!(result.best_fitness() >= 0.0);
}

#[wasm_bindgen_test]
fn test_bitstring_royal_road() {
    let mut optimizer = BitStringOptimizer::new(32);
    optimizer.set_population_size(100);
    optimizer.set_max_generations(100);
    optimizer.set_seed(42);

    let result = optimizer
        .solve_royal_road(8)
        .expect("RoyalRoad should succeed");
    // With 32 bits and schema_size=8, max fitness is 4
    assert!(result.best_fitness() <= 4.0);
    assert!(result.best_fitness() >= 0.0);
}

// ============================================================================
// UMDA Tests
// ============================================================================

use fugue_evo_wasm::UmdaOptimizer;

#[wasm_bindgen_test]
fn test_umda_creation() {
    let _optimizer = UmdaOptimizer::new(5);
}

#[wasm_bindgen_test]
fn test_umda_configuration() {
    let mut optimizer = UmdaOptimizer::new(5);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(50);
    optimizer.set_selection_ratio(0.3);
    optimizer.set_min_variance(0.001);
    optimizer.set_learning_rate(0.8);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_seed(42);
    // Configuration verified by not panicking
}

#[wasm_bindgen_test]
fn test_umda_sphere_optimization() {
    let mut optimizer = UmdaOptimizer::new(3);
    optimizer.set_population_size(50);
    optimizer.set_max_generations(100);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_seed(42);

    let result = optimizer.optimize("sphere").expect("UMDA should succeed");
    assert!(result.best_fitness() >= 0.0);
    assert_eq!(result.best_genome().len(), 3);
}

// ============================================================================
// ZDT Multi-Objective Tests
// ============================================================================

use fugue_evo_wasm::ZdtProblem;

#[wasm_bindgen_test]
fn test_nsga2_zdt1() {
    let mut optimizer = Nsga2Optimizer::new(10, 2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_seed(42);

    let result = optimizer
        .optimize_zdt(ZdtProblem::Zdt1)
        .expect("ZDT1 should succeed");
    assert!(result.front_size() > 0);
}

#[wasm_bindgen_test]
fn test_nsga2_zdt2() {
    let mut optimizer = Nsga2Optimizer::new(10, 2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_seed(42);

    let result = optimizer
        .optimize_zdt(ZdtProblem::Zdt2)
        .expect("ZDT2 should succeed");
    assert!(result.front_size() > 0);
}

#[wasm_bindgen_test]
fn test_nsga2_zdt3() {
    let mut optimizer = Nsga2Optimizer::new(10, 2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_seed(42);

    let result = optimizer
        .optimize_zdt(ZdtProblem::Zdt3)
        .expect("ZDT3 should succeed");
    assert!(result.front_size() > 0);
}

// ============================================================================
// Fitness Info Tests
// ============================================================================

use fugue_evo_wasm::{get_available_fitness_functions, get_fitness_info};

#[wasm_bindgen_test]
fn test_available_fitness_functions() {
    let functions = get_available_fitness_functions();
    assert!(functions.contains(&"sphere".to_string()));
    assert!(functions.contains(&"rastrigin".to_string()));
    assert!(functions.contains(&"ackley".to_string()));
    assert!(functions.contains(&"griewank".to_string()));
    assert!(functions.len() >= 9); // We have at least 9 functions
}

#[wasm_bindgen_test]
fn test_fitness_info() {
    let info = get_fitness_info("sphere").expect("Should get sphere info");
    assert!(info.contains("Sphere"));
    assert!(info.contains("bounds"));

    let info = get_fitness_info("ackley").expect("Should get ackley info");
    assert!(info.contains("Ackley"));

    let err = get_fitness_info("unknown_function");
    assert!(err.is_err());
}

// ============================================================================
// Evolution Strategy Extended Tests
// ============================================================================

use fugue_evo_wasm::ESSelection;

#[wasm_bindgen_test]
fn test_evolution_strategy_mu_plus_lambda() {
    let mut optimizer = EvolutionStrategyOptimizer::new(3);
    optimizer.set_mu(5);
    optimizer.set_lambda(20);
    optimizer.set_selection_strategy(ESSelection::MuPlusLambda);
    optimizer.set_initial_sigma(0.5);
    optimizer.set_self_adaptive(true);
    optimizer.set_population_size(20);
    optimizer.set_max_generations(30);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_seed(42);
    // Can't actually call optimize without a JS fitness function
    // Just verify configuration doesn't panic
}

#[wasm_bindgen_test]
fn test_evolution_strategy_mu_comma_lambda() {
    let mut optimizer = EvolutionStrategyOptimizer::new(3);
    optimizer.set_mu(5);
    optimizer.set_lambda(30);
    optimizer.set_selection_strategy(ESSelection::MuCommaLambda);
    optimizer.set_initial_sigma(1.0);
    optimizer.set_self_adaptive(false);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_bounds(-10.0, 10.0);
    optimizer.set_seed(123);
    // Configuration verified by not panicking
}

// ============================================================================
// BitStringResult Extended Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_bitstring_result_methods() {
    let mut optimizer = BitStringOptimizer::new(16);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_seed(42);

    let result = optimizer.solve_one_max().expect("Should succeed");

    // Test count_ones
    let ones_count = result.count_ones();
    assert!(ones_count <= 16);
    assert!(ones_count > 0);

    // Test best_genome_string
    let genome_str = result.best_genome_string();
    assert_eq!(genome_str.len(), 16);
    assert!(genome_str.chars().all(|c| c == '0' || c == '1'));

    // Test fitness_history
    let history = result.fitness_history();
    assert!(!history.is_empty());
}

// ============================================================================
// MultiObjectiveResult Extended Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_multi_objective_result_methods() {
    let mut optimizer = Nsga2Optimizer::new(5, 2);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(20);
    optimizer.set_seed(42);

    let result = optimizer
        .optimize_zdt(ZdtProblem::Zdt1)
        .expect("Should succeed");

    // Test front_size
    let front_size = result.front_size();
    assert!(front_size > 0);

    // Test generations and evaluations
    assert_eq!(result.generations(), 20);
    assert!(result.evaluations() > 0);

    // Test get_solution
    if front_size > 0 {
        let solution = result.get_solution(0).expect("Should get first solution");
        assert!(!solution.genome().is_empty());
        assert!(!solution.objectives().is_empty());
    }

    // Test all_genomes and all_objectives
    let all_genomes = result.all_genomes();
    let all_objectives = result.all_objectives();
    assert!(!all_genomes.is_empty());
    assert!(!all_objectives.is_empty());

    // Test to_json
    let json = result.to_json().expect("Should serialize to JSON");
    assert!(json.contains("pareto_front"));
}

// ============================================================================
// Additional Fitness Function Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_dixon_price_optimization() {
    let mut optimizer = RealVectorOptimizer::new(3);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-10.0, 10.0);
    optimizer.set_fitness("dixon_price");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Dixon-Price optimization should succeed");
    assert!(result.best_fitness() >= 0.0);
}

#[wasm_bindgen_test]
fn test_styblinski_tang_optimization() {
    let mut optimizer = RealVectorOptimizer::new(3);
    optimizer.set_population_size(30);
    optimizer.set_max_generations(50);
    optimizer.set_bounds(-5.0, 5.0);
    optimizer.set_fitness("styblinski_tang");
    optimizer.set_seed(42);

    let result = optimizer
        .optimize()
        .expect("Styblinski-Tang optimization should succeed");
    // Result can be negative for Styblinski-Tang
    assert!(result.evaluations() > 0);
}

// ============================================================================
// Interactive Evolution Tests
// ============================================================================

use fugue_evo_wasm::{InteractiveOptimizer, StepResultType, WasmEvaluationMode};

#[wasm_bindgen_test]
fn test_interactive_optimizer_creation() {
    let _optimizer = InteractiveOptimizer::new(5);
}

#[wasm_bindgen_test]
fn test_interactive_optimizer_with_config() {
    let _optimizer = InteractiveOptimizer::with_config(
        5,                          // dimension
        20,                         // population_size
        WasmEvaluationMode::Rating, // evaluation_mode
        5,                          // batch_size
        10,                         // max_generations
    );
}

#[wasm_bindgen_test]
fn test_interactive_optimizer_with_extended_config() {
    let _optimizer = InteractiveOptimizer::with_extended_config(
        5,                            // dimension
        20,                           // population_size
        WasmEvaluationMode::Pairwise, // evaluation_mode
        5,                            // batch_size
        3,                            // select_count
        10,                           // max_generations
        2,                            // elitism_count
        0.8,                          // min_coverage
        0.9,                          // crossover_probability
        0.1,                          // mutation_probability
        -5.0,                         // lower_bound
        5.0,                          // upper_bound
    );
}

#[wasm_bindgen_test]
fn test_interactive_optimizer_step() {
    let mut optimizer = InteractiveOptimizer::with_config(3, 10, WasmEvaluationMode::Rating, 3, 5);
    optimizer.set_seed(42);

    // First step should request evaluation
    let step_result = optimizer.step();
    assert!(step_result.needs_evaluation());
    assert!(matches!(
        step_result.result_type(),
        StepResultType::NeedsEvaluation
    ));

    // Verify we can get the request
    let request = step_result.get_request().expect("Should have request");
    assert!(request.candidate_count() > 0);
}

#[wasm_bindgen_test]
fn test_interactive_optimizer_getters() {
    let mut optimizer = InteractiveOptimizer::with_config(3, 10, WasmEvaluationMode::Rating, 3, 5);
    optimizer.set_seed(42);

    // Test initial state getters
    assert_eq!(optimizer.generation(), 0);
    assert_eq!(optimizer.population_size(), 10);
    assert!(optimizer.get_coverage() >= 0.0);

    // Get population JSON
    let pop_json = optimizer
        .get_population_json()
        .expect("Should get population JSON");
    assert!(pop_json.contains('['));
}

#[wasm_bindgen_test]
fn test_interactive_optimizer_batch_selection_mode() {
    let optimizer =
        InteractiveOptimizer::with_config(3, 10, WasmEvaluationMode::BatchSelection, 5, 5);

    assert_eq!(optimizer.population_size(), 10);
}

// ============================================================================
// Evaluation Request Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_evaluation_request_methods() {
    let mut optimizer = InteractiveOptimizer::with_config(3, 10, WasmEvaluationMode::Rating, 3, 5);
    optimizer.set_seed(42);

    let step_result = optimizer.step();
    let request = step_result.get_request().expect("Should have request");

    // Test request type
    use fugue_evo_wasm::RequestType;
    assert!(matches!(request.request_type(), RequestType::Rating));

    // Test candidate access
    let candidate_count = request.candidate_count();
    assert!(candidate_count > 0);

    // Test get candidate
    if candidate_count > 0 {
        let candidate = request
            .get_candidate(0)
            .expect("Should get first candidate");
        assert!(candidate.genome().len() == 3);
        assert!(!candidate.is_evaluated());
        assert_eq!(candidate.evaluation_count(), 0);
    }

    // Test get candidate IDs
    let ids = request.get_candidate_ids();
    assert_eq!(ids.len(), candidate_count);

    // Test get candidates JSON
    let json = request
        .get_candidates_json()
        .expect("Should serialize candidates");
    assert!(json.contains('['));

    // Test scale bounds
    assert!(request.scale_min() <= request.scale_max());

    // Test to_json
    let request_json = request.to_json().expect("Should serialize request");
    assert!(request_json.contains("request_type"));
}

// ============================================================================
// WasmCandidate Tests
// ============================================================================

#[wasm_bindgen_test]
fn test_wasm_candidate_methods() {
    let mut optimizer = InteractiveOptimizer::with_config(3, 10, WasmEvaluationMode::Rating, 3, 5);
    optimizer.set_seed(42);

    let step_result = optimizer.step();
    let request = step_result.get_request().expect("Should have request");
    let candidate = request.get_candidate(0).expect("Should get candidate");

    // Test id
    let id = candidate.id();
    assert!(id < 1000); // Reasonable ID

    // Test genome
    let genome = candidate.genome();
    assert_eq!(genome.len(), 3);

    // Test fitness (should be None initially)
    assert!(candidate.fitness().is_none());

    // Test evaluation count
    assert_eq!(candidate.evaluation_count(), 0);

    // Test is_evaluated
    assert!(!candidate.is_evaluated());

    // Test to_json
    let json = candidate.to_json().expect("Should serialize candidate");
    assert!(json.contains("id"));
    assert!(json.contains("genome"));
}
