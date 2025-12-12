//! Checkpointing support for evolution state persistence
//!
//! This module provides serialization and recovery of evolution state,
//! enabling long-running experiments to be paused and resumed.

mod state;
mod recovery;

pub use state::*;
pub use recovery::*;

/// Prelude for checkpoint module
pub mod prelude {
    pub use super::state::*;
    pub use super::recovery::*;
}
