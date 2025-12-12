//! Population management
//!
//! This module provides the Individual and Population types.

pub mod individual;
#[allow(clippy::module_inception)]
pub mod population;

pub mod prelude {
    pub use super::individual::*;
    pub use super::population::*;
}
