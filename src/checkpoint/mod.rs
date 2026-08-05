//! Checkpointing support for evolution state persistence
//!
//! This module provides serialization and recovery of evolution state,
//! enabling long-running experiments to be paused and resumed.
//!
//! The `state` submodule (data structures) is always available.
//! The `recovery` submodule (file I/O) requires the `checkpoint` feature **and
//! a target with a filesystem**.
//!
//! The second half of that condition is not a new restriction, it is the one
//! [`CheckpointError`](crate::error::CheckpointError) already encodes: its
//! `Io` variant is `#[cfg(not(target_arch = "wasm32"))]` and its docs read
//! "native only", with `Storage(String)` offered as the wasm alternative. The
//! module gate had not been kept in step, so on `wasm32` the file-I/O code was
//! still compiled while the error variant its `?` operators desugar into was
//! not — `cargo check --target wasm32-unknown-unknown` failed on the default
//! feature set with a dozen "`?` couldn't convert the error to
//! `CheckpointError`".
//!
//! Gating the module rather than adding a wasm `From<io::Error>` is the fix
//! that matches the intent: `wasm32-unknown-unknown` has no filesystem, so
//! `File::create` there could only ever be a compile-time promise of a
//! runtime failure.
#[cfg(all(feature = "checkpoint", not(target_arch = "wasm32")))]
mod recovery;
mod rng;
mod state;

#[cfg(all(feature = "checkpoint", not(target_arch = "wasm32")))]
pub use recovery::*;
pub use rng::*;
pub use state::*;

/// Prelude for checkpoint module
pub mod prelude {
    #[cfg(all(feature = "checkpoint", not(target_arch = "wasm32")))]
    pub use super::recovery::*;
    pub use super::rng::*;
    pub use super::state::*;
}
