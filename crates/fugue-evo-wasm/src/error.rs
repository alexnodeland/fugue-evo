//! Error handling for WASM bindings

use wasm_bindgen::prelude::*;

/// Convert an evolution error to a JsValue
pub fn evolution_error_to_js(err: fugue_evo::error::EvolutionError) -> JsValue {
    let error_type = match &err {
        fugue_evo::error::EvolutionError::Genome(_) => "GenomeError",
        fugue_evo::error::EvolutionError::Operator(_) => "OperatorError",
        fugue_evo::error::EvolutionError::FitnessEvaluation(_) => "FitnessError",
        fugue_evo::error::EvolutionError::Configuration(_) => "ConfigError",
        fugue_evo::error::EvolutionError::Checkpoint(_) => "CheckpointError",
        fugue_evo::error::EvolutionError::Numerical(_) => "NumericalError",
        fugue_evo::error::EvolutionError::EmptyPopulation => "EmptyPopulation",
        fugue_evo::error::EvolutionError::InteractiveEvaluation(_) => "InteractiveError",
        fugue_evo::error::EvolutionError::InsufficientCoverage { .. } => "CoverageError",
    };

    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &"type".into(), &JsValue::from_str(error_type));
    let _ = js_sys::Reflect::set(
        &obj,
        &"message".into(),
        &JsValue::from_str(&err.to_string()),
    );
    obj.into()
}

/// Create a JS error from a string message
pub fn string_error(msg: &str) -> JsValue {
    JsValue::from_str(msg)
}
