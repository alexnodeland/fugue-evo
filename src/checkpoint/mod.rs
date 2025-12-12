//! Checkpointing support for evolution state persistence
//!
//! This module provides serialization and recovery of evolution state,
//! enabling long-running experiments to be paused and resumed.

mod recovery;
mod state;

pub use recovery::*;
pub use state::*;

/// Prelude for checkpoint module
pub mod prelude {
    pub use super::recovery::*;
    pub use super::state::*;
}
