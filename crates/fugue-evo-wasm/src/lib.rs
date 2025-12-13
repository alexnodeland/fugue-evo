//! WebAssembly bindings for fugue-evo
//!
//! This crate provides JavaScript-friendly wrappers for the fugue-evo
//! genetic algorithm library, enabling browser-based optimization.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { RealVectorOptimizer } from 'fugue-evo-wasm';
//!
//! await init();
//!
//! const optimizer = new RealVectorOptimizer(10); // 10 dimensions
//! optimizer.setPopulationSize(100);
//! optimizer.setBounds(-5.12, 5.12);
//! optimizer.setFitness("rastrigin");
//!
//! const result = optimizer.optimize(200); // 200 generations
//! console.log("Best fitness:", result.best_fitness);
//! console.log("Best solution:", result.best_genome);
//! ```

use wasm_bindgen::prelude::*;

mod config;
mod error;
mod fitness;
mod interactive;
mod optimizers;
mod result;

pub use config::*;
pub use error::*;
pub use fitness::*;
pub use interactive::*;
pub use optimizers::*;
pub use result::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // WASM initialization complete
}

/// Get the library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
