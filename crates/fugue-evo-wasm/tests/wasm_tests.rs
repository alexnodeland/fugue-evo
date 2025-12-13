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
