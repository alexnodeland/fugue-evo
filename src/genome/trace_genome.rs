//! The fugue trace encoding of genomes (`ppl` feature).
//!
//! This module is the boundary between the classic evolutionary-computation
//! layer and the PPL-native inference layer: a genome that implements
//! [`TraceGenome`] can be round-tripped through a [`fugue::Trace`] and
//! therefore driven by the trace-space machinery in
//! [`crate::inference`] (MH rejuvenation, tempered SMC, block regeneration).
//!
//! The encoding produced by [`TraceGenome::to_trace`] is *canonical* — an
//! address→value map at the genome's site addresses (`gene#i`, `bit#i`,
//! `perm#i`, …) with zero stored log-probabilities. Probability mass is never
//! carried by this encoding; it is recovered by scoring the trace under a
//! genuine prior model (see [`crate::inference::prior::GenomePrior`]), which is also
//! how the inference layer decodes a genome from a particle
//! (`decode`-by-replay).

use fugue::{addr, Address, Trace};

// Re-export ChoiceValue for use in genome trace implementations (relocated
// from `genome::traits` when the trait was split).
pub use fugue::ChoiceValue;

use crate::error::GenomeError;
use crate::genome::traits::EvolutionaryGenome;

/// Extension trait: genomes that can be encoded as fugue traces.
///
/// Implementing this trait is what admits a genome to the `ppl` inference
/// layer. The classic algorithms never require it.
pub trait TraceGenome: EvolutionaryGenome {
    /// Convert genome to a fugue trace.
    ///
    /// Each gene is stored at an indexed address (e.g., `gene#0`, `gene#1`, …)
    /// as a pure value; stored log-probabilities are zero. Score the trace
    /// under a prior model to obtain real probability mass.
    fn to_trace(&self) -> Trace;

    /// Reconstruct genome from a fugue trace.
    ///
    /// This is the inverse of [`Self::to_trace`], extracting gene values from
    /// the trace's choice map.
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;

    /// Get the address prefix used for trace storage (default: `"gene"`).
    fn trace_prefix() -> &'static str {
        "gene"
    }
}

/// Helper function to create a gene address for trace storage
pub fn gene_address(prefix: &str, index: usize) -> Address {
    addr!(prefix, index)
}
