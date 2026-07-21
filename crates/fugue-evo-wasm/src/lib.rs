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
mod explore;
mod fitness;
mod interactive;
mod optimizers;
mod result;

pub use config::*;
pub use error::*;
pub use explore::*;
pub use fitness::*;
pub use interactive::*;
pub use optimizers::*;
pub use result::*;

/// Initialize the WASM module.
///
/// Installs [`console_error_panic_hook`] so that any Rust panic in the
/// `fugue-evo` call graph is logged to the JavaScript console with its real
/// Rust file/line and message, instead of surfacing as an opaque
/// `RuntimeError: unreachable executed` (AUDIT EV-09).
///
/// Note: a Rust panic still aborts the *current* call with a wasm trap — this
/// hook does not make panics recoverable, it makes them *diagnosable*. All
/// fallible entry points in this crate additionally return `Result<_, JsValue>`
/// with structured error information (see `error::evolution_error_to_js`) rather
/// than panicking, and the JS-boundary fitness helpers use saturating fallbacks
/// (`unwrap_or(...)`) rather than panicking unwraps.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Get the library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
