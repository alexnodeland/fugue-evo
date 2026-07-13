//! Serializable RNG snapshots for reproducible checkpoint resume (EV-02)
//!
//! Reproducible resume requires an RNG whose entire internal state can be
//! captured and restored byte-for-byte. `rand`'s generic `Rng`/`RngCore`
//! traits provide no serialization hook, so a resumed run seeded from a fresh
//! (or differently-seeded) generator silently diverges from the trajectory a
//! continuous run would have taken.
//!
//! The [`SnapshotRng`] trait fills that gap for the ChaCha family of counter
//! based RNGs (`ChaCha8Rng`, `ChaCha12Rng`, `ChaCha20Rng`), whose state is
//! fully serializable via `serde`. Capturing the RNG alongside the population
//! in a [`Checkpoint`](crate::checkpoint::Checkpoint) makes resume
//! bit-identical to a run that was never interrupted.
//!
//! # Reproducibility contract
//!
//! Bit-identical resume is only guaranteed when the algorithm draws all of its
//! randomness from a `SnapshotRng` and that RNG is captured into the checkpoint
//! (via [`Checkpoint::with_rng`](crate::checkpoint::Checkpoint::with_rng)) and
//! restored on resume (via
//! [`Checkpoint::restore_rng`](crate::checkpoint::Checkpoint::restore_rng)).
//! Non-ChaCha generators (e.g. `StdRng`, `ThreadRng`) cannot be snapshotted and
//! therefore cannot provide reproducible resume.

use rand_chacha::{ChaCha12Rng, ChaCha20Rng, ChaCha8Rng};

use crate::error::CheckpointError;

/// An RNG whose complete internal state can be captured to bytes and restored,
/// enabling bit-identical checkpoint resume.
///
/// Implemented for the ChaCha family (`ChaCha8Rng`, `ChaCha12Rng`,
/// `ChaCha20Rng`), all of which expose fully serializable state.
pub trait SnapshotRng: Sized {
    /// Serialize the full RNG state to a byte buffer.
    fn capture(&self) -> Result<Vec<u8>, CheckpointError>;

    /// Reconstruct an RNG from bytes previously produced by [`capture`].
    ///
    /// [`capture`]: SnapshotRng::capture
    fn restore(bytes: &[u8]) -> Result<Self, CheckpointError>;
}

macro_rules! impl_snapshot_rng {
    ($rng:ty) => {
        impl SnapshotRng for $rng {
            fn capture(&self) -> Result<Vec<u8>, CheckpointError> {
                bincode::serialize(self).map_err(|e| CheckpointError::SerializeError(Box::new(e)))
            }

            fn restore(bytes: &[u8]) -> Result<Self, CheckpointError> {
                bincode::deserialize(bytes)
                    .map_err(|e| CheckpointError::DeserializeError(Box::new(e)))
            }
        }
    };
}

impl_snapshot_rng!(ChaCha8Rng);
impl_snapshot_rng!(ChaCha12Rng);
impl_snapshot_rng!(ChaCha20Rng);

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_snapshot_rng_round_trip_is_bit_identical() {
        // regression: EV-02 - a captured RNG must restore to the exact same
        // state, producing an identical sequence of draws afterwards.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Advance the stream so we are not just testing the seeded state.
        for _ in 0..37 {
            let _: u64 = rng.gen();
        }

        let bytes = rng.capture().unwrap();
        let mut restored = ChaCha8Rng::restore(&bytes).unwrap();

        // Both generators must now produce an identical sequence.
        for _ in 0..1000 {
            let a: u64 = rng.gen();
            let b: u64 = restored.gen();
            assert_eq!(a, b, "restored RNG diverged from the original");
        }
    }

    #[test]
    fn test_snapshot_rng_variants() {
        let mut c12 = ChaCha12Rng::seed_from_u64(7);
        let _: u32 = c12.gen();
        let bytes = c12.capture().unwrap();
        let mut r12 = ChaCha12Rng::restore(&bytes).unwrap();
        assert_eq!(c12.gen::<u64>(), r12.gen::<u64>());

        let mut c20 = ChaCha20Rng::seed_from_u64(9);
        let _: u32 = c20.gen();
        let bytes = c20.capture().unwrap();
        let mut r20 = ChaCha20Rng::restore(&bytes).unwrap();
        assert_eq!(c20.gen::<u64>(), r20.gen::<u64>());
    }
}
