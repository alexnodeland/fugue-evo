//! Checkpointing support for evolution state persistence
//!
//! This module provides serialization and recovery of evolution state,
//! enabling long-running experiments to be paused and resumed.
//!
//! The `state` submodule (data structures) is always available.
//! The `recovery` submodule (file I/O) requires the `checkpoint` feature.

#[cfg(feature = "checkpoint")]
mod recovery;
mod rng;
mod state;

#[cfg(feature = "checkpoint")]
pub use recovery::*;
pub use rng::*;
pub use state::*;

/// Prelude for checkpoint module
pub mod prelude {
    #[cfg(feature = "checkpoint")]
    pub use super::recovery::*;
    pub use super::rng::*;
    pub use super::state::*;
}
