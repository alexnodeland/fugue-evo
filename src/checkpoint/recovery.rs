//! Checkpoint recovery and persistence
//!
//! Provides serialization to/from files with compression and versioning.

use serde::{de::DeserializeOwned, Serialize};

#[cfg(feature = "checkpoint")]
use bincode::Options;
#[cfg(feature = "checkpoint")]
use std::fs::{self, File};
#[cfg(feature = "checkpoint")]
use std::io::{BufReader, BufWriter, Read, Write};
#[cfg(feature = "checkpoint")]
use std::path::{Path, PathBuf};

use super::state::{Checkpoint, CHECKPOINT_VERSION};
use crate::error::CheckpointError;

/// Default cap on the size (in bytes) of a checkpoint file that
/// [`load_checkpoint`] will deserialize (EV-48). Files larger than this, and
/// individual length-prefixed fields larger than this, are rejected with a
/// typed error rather than triggering an unbounded allocation.
pub const DEFAULT_MAX_CHECKPOINT_BYTES: u64 = 256 * 1024 * 1024;

/// Oldest checkpoint schema version this build can still load (EV-45).
///
/// Checkpoints older than this are rejected with
/// [`CheckpointError::VersionTooOld`] rather than being silently
/// misinterpreted.
pub const MIN_SUPPORTED_CHECKPOINT_VERSION: u32 = 1;

/// Validate a checkpoint schema version against the supported range (EV-45).
///
/// Applied uniformly to every format (JSON, binary, compressed binary) so that
/// a JSON checkpoint written by an incompatible library version is gated
/// exactly like its binary siblings.
#[cfg(feature = "checkpoint")]
fn check_version(version: u32) -> Result<(), CheckpointError> {
    if version > CHECKPOINT_VERSION {
        return Err(CheckpointError::VersionMismatch {
            expected: CHECKPOINT_VERSION,
            found: version,
        });
    }
    if version < MIN_SUPPORTED_CHECKPOINT_VERSION {
        return Err(CheckpointError::VersionTooOld(version));
    }
    Ok(())
}

/// bincode options for reading checkpoint data with a total-size limit (EV-48).
///
/// Uses fixint encoding + trailing-byte tolerance so the produced format is
/// byte-compatible with `bincode::serialize`/`serialize_into` (used on the
/// write path), while adding a byte limit that turns a corrupted length prefix
/// into a clean error instead of a huge allocation.
#[cfg(feature = "checkpoint")]
fn bincode_read_options(limit: u64) -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(limit)
}

/// Compute the temporary sibling path used for atomic writes (EV-47).
#[cfg(feature = "checkpoint")]
fn temp_path_for(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".tmp");
    PathBuf::from(os)
}

/// Parse the numeric index out of a checkpoint filename such as
/// `evolution_00000042.ckpt` (EV-46 / EV-87).
///
/// Width-agnostic: it accepts both the current zero-padded 8-digit names and
/// legacy 4-digit (`{:04}`) names, so ordering and restart-resume keep working
/// across a format change.
#[cfg(feature = "checkpoint")]
fn parse_checkpoint_index(file_name: &str, base_name: &str) -> Option<usize> {
    let rest = file_name.strip_prefix(base_name)?.strip_prefix('_')?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    // Whatever follows the digits must be an extension separator (or nothing).
    let tail = &rest[digits.len()..];
    if !tail.is_empty() && !tail.starts_with('.') {
        return None;
    }
    digits.parse::<usize>().ok()
}

/// Scan a directory for the highest existing checkpoint index for `base_name`
/// (EV-46). Returns `None` if the directory is unreadable or has no matching
/// files.
#[cfg(feature = "checkpoint")]
fn scan_max_index(directory: &Path, base_name: &str) -> Option<usize> {
    std::fs::read_dir(directory)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            parse_checkpoint_index(&name, base_name)
        })
        .max()
}

/// Format for checkpoint serialization
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// JSON format (human-readable, larger)
    Json,
    /// Binary format (compact, fast)
    Binary,
    /// Compressed binary (smallest, slower)
    CompressedBinary,
}

impl Default for CheckpointFormat {
    fn default() -> Self {
        Self::Binary
    }
}

/// Save a checkpoint to a file
#[cfg(feature = "checkpoint")]
pub fn save_checkpoint<G>(
    checkpoint: &Checkpoint<G>,
    path: impl AsRef<Path>,
    format: CheckpointFormat,
) -> Result<(), CheckpointError>
where
    G: Clone + Serialize + crate::genome::traits::EvolutionaryGenome,
{
    let path = path.as_ref();

    // EV-47: write to a temporary sibling file, fsync it, then atomically
    // rename into place. A crash mid-write can only corrupt the throwaway
    // `.tmp` file, never the destination the reader loads from.
    let tmp_path = temp_path_for(path);

    let write_result = (|| -> Result<(), CheckpointError> {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        match format {
            CheckpointFormat::Json => {
                serde_json::to_writer_pretty(&mut writer, checkpoint)
                    .map_err(|e| CheckpointError::SerializeError(Box::new(e)))?;
            }
            CheckpointFormat::Binary => {
                // Write version header first
                writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
                // Write magic bytes for format identification
                writer.write_all(b"FEVO")?;
                // Serialize with bincode
                bincode::serialize_into(&mut writer, checkpoint)
                    .map_err(|e| CheckpointError::SerializeError(Box::new(e)))?;
            }
            CheckpointFormat::CompressedBinary => {
                // Write version and magic
                writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
                writer.write_all(b"FEVC")?; // C for compressed
                                            // Serialize to bytes first
                let bytes = bincode::serialize(checkpoint)
                    .map_err(|e| CheckpointError::SerializeError(Box::new(e)))?;
                // Compress with simple RLE-like compression
                let compressed = compress_data(&bytes);
                // Write length and data
                writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
                writer.write_all(&compressed)?;
            }
        }

        writer.flush()?;
        // Recover the underlying File and fsync data + metadata *before* the
        // rename, so a successful rename implies fully-durable contents.
        let file = writer.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        // Best-effort cleanup of the partial temp file.
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Load a checkpoint from a file using the default size limit
/// ([`DEFAULT_MAX_CHECKPOINT_BYTES`]).
#[cfg(feature = "checkpoint")]
pub fn load_checkpoint<G>(path: impl AsRef<Path>) -> Result<Checkpoint<G>, CheckpointError>
where
    G: Clone + Serialize + DeserializeOwned + crate::genome::traits::EvolutionaryGenome,
{
    load_checkpoint_with_limit(path, DEFAULT_MAX_CHECKPOINT_BYTES)
}

/// Load a checkpoint from a file, rejecting inputs larger than `max_bytes`.
///
/// The limit guards against unbounded allocation from a corrupted or hostile
/// checkpoint (EV-48): both the on-disk file size and every length-prefixed
/// field decoded by bincode are bounded by `max_bytes`. The embedded schema
/// version is validated for every format, including JSON (EV-45).
#[cfg(feature = "checkpoint")]
pub fn load_checkpoint_with_limit<G>(
    path: impl AsRef<Path>,
    max_bytes: u64,
) -> Result<Checkpoint<G>, CheckpointError>
where
    G: Clone + Serialize + DeserializeOwned + crate::genome::traits::EvolutionaryGenome,
{
    let path = path.as_ref();
    if !path.exists() {
        return Err(CheckpointError::NotFound(path.display().to_string()));
    }

    // EV-48: reject oversized files up front, before any large read/allocation.
    let file_len = fs::metadata(path)?.len();
    if file_len > max_bytes {
        return Err(CheckpointError::TooLarge {
            size: file_len,
            limit: max_bytes,
        });
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Try to detect format by reading first bytes
    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;

    // Check for binary format magic
    if &header[4..8] == b"FEVO" {
        // Binary format
        let version = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        check_version(version)?;

        let checkpoint: Checkpoint<G> = bincode_read_options(max_bytes)
            .deserialize_from(&mut reader)
            .map_err(|e| CheckpointError::DeserializeError(e))?;
        check_version(checkpoint.version)?;
        Ok(checkpoint)
    } else if &header[4..8] == b"FEVC" {
        // Compressed binary format
        let version = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        check_version(version)?;

        // Read length
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let compressed_len = u64::from_le_bytes(len_bytes);
        // EV-48: a corrupted length prefix must not drive a huge allocation.
        if compressed_len > max_bytes {
            return Err(CheckpointError::TooLarge {
                size: compressed_len,
                limit: max_bytes,
            });
        }

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len as usize];
        reader.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = decompress_data(&compressed).map_err(CheckpointError::Corrupted)?;

        let checkpoint: Checkpoint<G> = bincode_read_options(max_bytes)
            .deserialize(&decompressed)
            .map_err(|e| CheckpointError::DeserializeError(e))?;
        check_version(checkpoint.version)?;
        Ok(checkpoint)
    } else {
        // Try JSON format - need to re-read from start
        drop(reader);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let checkpoint: Checkpoint<G> = serde_json::from_reader(reader)
            .map_err(|e| CheckpointError::DeserializeError(Box::new(e)))?;
        // EV-45: JSON was previously accepted with no version gate at all.
        check_version(checkpoint.version)?;
        Ok(checkpoint)
    }
}

/// Simple compression using run-length encoding for repeated bytes
#[cfg(feature = "checkpoint")]
fn compress_data(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut compressed = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut count = 1u8;

        // Count consecutive identical bytes (max 255)
        while i + (count as usize) < data.len() && data[i + (count as usize)] == byte && count < 255
        {
            count += 1;
        }

        if count >= 4 || byte == 0xFF {
            // Use RLE: 0xFF, count, byte
            compressed.push(0xFF);
            compressed.push(count);
            compressed.push(byte);
        } else {
            // Store literally
            for _ in 0..count {
                if byte == 0xFF {
                    compressed.push(0xFF);
                    compressed.push(1);
                    compressed.push(0xFF);
                } else {
                    compressed.push(byte);
                }
            }
        }

        i += count as usize;
    }

    compressed
}

/// Decompress RLE-encoded data
#[cfg(feature = "checkpoint")]
fn decompress_data(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut decompressed = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0xFF {
            if i + 2 >= data.len() {
                return Err("Truncated RLE sequence".to_string());
            }
            let count = data[i + 1] as usize;
            let byte = data[i + 2];
            for _ in 0..count {
                decompressed.push(byte);
            }
            i += 3;
        } else {
            decompressed.push(data[i]);
            i += 1;
        }
    }

    Ok(decompressed)
}

/// Checkpoint manager for automatic saving
#[cfg(feature = "checkpoint")]
pub struct CheckpointManager {
    /// Directory for checkpoint files
    pub directory: std::path::PathBuf,
    /// Base filename for checkpoints
    pub base_name: String,
    /// Serialization format
    pub format: CheckpointFormat,
    /// How many checkpoints to keep
    pub keep_n: usize,
    /// Save interval (generations)
    pub interval: usize,
    /// Maximum checkpoint size accepted when loading (EV-48)
    pub max_bytes: u64,
    /// Current checkpoint index
    current_index: usize,
}

#[cfg(feature = "checkpoint")]
impl CheckpointManager {
    /// Create a new checkpoint manager.
    ///
    /// The manager is restart-safe (EV-46): it scans `directory` for existing
    /// checkpoints named after `base_name` and continues the index *after* the
    /// highest one found, so a fresh manager constructed after a crash never
    /// overwrites or shadows pre-crash checkpoints.
    pub fn new(directory: impl Into<std::path::PathBuf>, base_name: impl Into<String>) -> Self {
        let directory = directory.into();
        let base_name = base_name.into();
        let current_index = scan_max_index(&directory, &base_name)
            .map(|max| max + 1)
            .unwrap_or(0);
        Self {
            directory,
            base_name,
            format: CheckpointFormat::Binary,
            keep_n: 3,
            interval: 100,
            max_bytes: DEFAULT_MAX_CHECKPOINT_BYTES,
            current_index,
        }
    }

    /// Set the serialization format
    pub fn with_format(mut self, format: CheckpointFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the maximum checkpoint size accepted by [`load_latest`] (EV-48).
    ///
    /// [`load_latest`]: CheckpointManager::load_latest
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// The index the next [`save`](CheckpointManager::save) will write to.
    ///
    /// Exposed primarily so restart-resume behavior (EV-46) is observable/testable.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Set how many checkpoints to keep
    pub fn keep(mut self, n: usize) -> Self {
        self.keep_n = n;
        self
    }

    /// Set the save interval
    pub fn every(mut self, generations: usize) -> Self {
        self.interval = generations;
        self
    }

    /// Check if a checkpoint should be saved at this generation
    pub fn should_save(&self, generation: usize) -> bool {
        generation > 0 && generation.is_multiple_of(self.interval)
    }

    /// Get the path for the current checkpoint
    pub fn current_path(&self) -> std::path::PathBuf {
        let extension = match self.format {
            CheckpointFormat::Json => "json",
            CheckpointFormat::Binary | CheckpointFormat::CompressedBinary => "ckpt",
        };
        // EV-87: zero-pad to a fixed 8-digit width so lexicographic and numeric
        // ordering agree well beyond the 4-digit (`{:04}`) overflow at 10000.
        self.directory.join(format!(
            "{}_{:08}.{}",
            self.base_name, self.current_index, extension
        ))
    }

    /// Save a checkpoint and rotate old ones
    pub fn save<G>(&mut self, checkpoint: &Checkpoint<G>) -> Result<(), CheckpointError>
    where
        G: Clone + Serialize + crate::genome::traits::EvolutionaryGenome,
    {
        // Ensure directory exists
        std::fs::create_dir_all(&self.directory)?;

        // Save current checkpoint
        let path = self.current_path();
        save_checkpoint(checkpoint, &path, self.format)?;

        // Rotate: remove old checkpoints
        self.current_index += 1;
        if self.current_index > self.keep_n {
            let old_index = self.current_index - self.keep_n - 1;
            let extension = match self.format {
                CheckpointFormat::Json => "json",
                CheckpointFormat::Binary | CheckpointFormat::CompressedBinary => "ckpt",
            };
            let old_path = self
                .directory
                .join(format!("{}_{:08}.{}", self.base_name, old_index, extension));
            let _ = std::fs::remove_file(old_path); // Ignore errors
        }

        Ok(())
    }

    /// Find and load the latest checkpoint
    pub fn load_latest<G>(&self) -> Result<Option<Checkpoint<G>>, CheckpointError>
    where
        G: Clone + Serialize + DeserializeOwned + crate::genome::traits::EvolutionaryGenome,
    {
        let extension = match self.format {
            CheckpointFormat::Json => "json",
            CheckpointFormat::Binary | CheckpointFormat::CompressedBinary => "ckpt",
        };

        // Find all checkpoint files
        let _pattern = format!("{}_*.{}", self.base_name, extension);
        let mut checkpoints: Vec<_> = std::fs::read_dir(&self.directory)?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(&self.base_name))
            .collect();

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Sort by parsed numeric index (newest/highest first). EV-87: comparing
        // filenames as strings breaks once the index widens (e.g. "10000" <
        // "9999" lexically), so we parse the index and compare numerically.
        // Names that do not parse (e.g. legacy/foreign files) sort last.
        checkpoints.sort_by(|a, b| {
            let ia = parse_checkpoint_index(&a.file_name().to_string_lossy(), &self.base_name);
            let ib = parse_checkpoint_index(&b.file_name().to_string_lossy(), &self.base_name);
            ib.cmp(&ia)
        });

        // Try to load the newest checkpoint
        for entry in checkpoints {
            match load_checkpoint_with_limit(entry.path(), self.max_bytes) {
                Ok(checkpoint) => return Ok(Some(checkpoint)),
                Err(_) => continue, // Try next if corrupted
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::population::individual::Individual;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.json");

        let population: Vec<Individual<RealVector>> = vec![
            Individual::new(RealVector::new(vec![1.0, 2.0])),
            Individual::new(RealVector::new(vec![3.0, 4.0])),
        ];
        let checkpoint = Checkpoint::new(10, population);

        save_checkpoint(&checkpoint, &path, CheckpointFormat::Json).unwrap();
        let loaded: Checkpoint<RealVector> = load_checkpoint(&path).unwrap();

        assert_eq!(loaded.generation, 10);
        assert_eq!(loaded.population.len(), 2);
    }

    #[test]
    fn test_save_load_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ckpt");

        let population: Vec<Individual<RealVector>> =
            vec![Individual::new(RealVector::new(vec![1.0, 2.0, 3.0]))];
        let checkpoint = Checkpoint::new(5, population)
            .with_evaluations(500)
            .with_metadata("test", "value");

        save_checkpoint(&checkpoint, &path, CheckpointFormat::Binary).unwrap();
        let loaded: Checkpoint<RealVector> = load_checkpoint(&path).unwrap();

        assert_eq!(loaded.generation, 5);
        assert_eq!(loaded.evaluations, 500);
        assert_eq!(loaded.metadata.get("test"), Some(&"value".to_string()));
    }

    #[test]
    fn test_save_load_compressed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_compressed.ckpt");

        // Create a larger checkpoint with repetitive data
        let population: Vec<Individual<RealVector>> = (0..100)
            .map(|i| Individual::new(RealVector::new(vec![i as f64; 10])))
            .collect();
        let checkpoint = Checkpoint::new(100, population);

        save_checkpoint(&checkpoint, &path, CheckpointFormat::CompressedBinary).unwrap();
        let loaded: Checkpoint<RealVector> = load_checkpoint(&path).unwrap();

        assert_eq!(loaded.generation, 100);
        assert_eq!(loaded.population.len(), 100);
    }

    #[test]
    fn test_compression_decompression() {
        let original = vec![0u8, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5];
        let compressed = compress_data(&original);
        let decompressed = decompress_data(&compressed).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_checkpoint_manager() {
        let dir = tempdir().unwrap();
        let mut manager = CheckpointManager::new(dir.path(), "evolution")
            .with_format(CheckpointFormat::Binary)
            .keep(2)
            .every(10);

        // Save multiple checkpoints
        for gen in [10, 20, 30, 40] {
            let population: Vec<Individual<RealVector>> =
                vec![Individual::new(RealVector::new(vec![gen as f64]))];
            let checkpoint = Checkpoint::new(gen, population);
            manager.save(&checkpoint).unwrap();
        }

        // Load latest
        let loaded: Option<Checkpoint<RealVector>> = manager.load_latest().unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().generation, 40);
    }

    #[test]
    fn test_version_check() {
        let population: Vec<Individual<RealVector>> = vec![];
        let checkpoint = Checkpoint::new(0, population);
        assert!(checkpoint.is_compatible());
    }

    #[test]
    fn test_json_version_gate_too_new() {
        // regression: EV-45 - a too-new JSON checkpoint was silently accepted;
        // it must now be rejected exactly like its binary siblings.
        let dir = tempdir().unwrap();
        let path = dir.path().join("future.json");

        let population: Vec<Individual<RealVector>> = vec![];
        let mut checkpoint = Checkpoint::new(3, population);
        checkpoint.version = CHECKPOINT_VERSION + 1; // pretend a newer library wrote it

        save_checkpoint(&checkpoint, &path, CheckpointFormat::Json).unwrap();

        let err = load_checkpoint::<RealVector>(&path).unwrap_err();
        assert!(
            matches!(err, CheckpointError::VersionMismatch { .. }),
            "expected VersionMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_json_version_gate_too_old() {
        // regression: EV-45 - VersionTooOld must actually be returned (it was
        // previously dead code), for JSON as for any other format.
        let dir = tempdir().unwrap();
        let path = dir.path().join("ancient.json");

        let population: Vec<Individual<RealVector>> = vec![];
        let mut checkpoint = Checkpoint::new(3, population);
        checkpoint.version = MIN_SUPPORTED_CHECKPOINT_VERSION - 1;

        save_checkpoint(&checkpoint, &path, CheckpointFormat::Json).unwrap();

        let err = load_checkpoint::<RealVector>(&path).unwrap_err();
        assert!(
            matches!(err, CheckpointError::VersionTooOld(v) if v == MIN_SUPPORTED_CHECKPOINT_VERSION - 1),
            "expected VersionTooOld, got {err:?}"
        );
    }

    #[test]
    fn test_manager_is_restart_safe() {
        // regression: EV-46 - a manager reconstructed after a restart must
        // continue the index sequence, not reset to 0 and shadow/overwrite
        // pre-crash checkpoints.
        let dir = tempdir().unwrap();

        {
            let mut manager = CheckpointManager::new(dir.path(), "evolution")
                .with_format(CheckpointFormat::Binary)
                .keep(10);
            for gen in [10usize, 20, 30] {
                let population: Vec<Individual<RealVector>> =
                    vec![Individual::new(RealVector::new(vec![gen as f64]))];
                manager.save(&Checkpoint::new(gen, population)).unwrap();
            }
            assert_eq!(manager.current_index(), 3);
        }

        // Simulate a restart: brand-new manager over the same directory.
        let manager2 = CheckpointManager::new(dir.path(), "evolution")
            .with_format(CheckpointFormat::Binary)
            .keep(10);
        assert_eq!(
            manager2.current_index(),
            3,
            "restarted manager must continue after the highest existing index"
        );

        // The genuinely-newest pre-restart checkpoint must still load.
        let loaded: Option<Checkpoint<RealVector>> = manager2.load_latest().unwrap();
        assert_eq!(loaded.unwrap().generation, 30);
    }

    #[test]
    fn test_atomic_save_preserves_destination_on_failure() {
        // regression: EV-47 - a failed save must not truncate/destroy the
        // existing good checkpoint. Pre-fix, save_checkpoint called
        // File::create(path) directly and truncated the destination before
        // writing; here we block the temp file and assert the old file survives.
        let dir = tempdir().unwrap();
        let path = dir.path().join("evolution.ckpt");

        // Write a good checkpoint (generation 111).
        let good: Vec<Individual<RealVector>> = vec![Individual::new(RealVector::new(vec![1.0]))];
        save_checkpoint(&Checkpoint::new(111, good), &path, CheckpointFormat::Binary).unwrap();

        // Block the atomic temp path so the next write fails before rename.
        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        std::fs::create_dir(&tmp).unwrap();

        let newer: Vec<Individual<RealVector>> = vec![Individual::new(RealVector::new(vec![2.0]))];
        let result = save_checkpoint(
            &Checkpoint::new(222, newer),
            &path,
            CheckpointFormat::Binary,
        );
        assert!(
            result.is_err(),
            "save should fail when temp file is blocked"
        );

        // The original good checkpoint must be intact and loadable.
        let loaded: Checkpoint<RealVector> = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.generation, 111);

        // No leftover .tmp *file* (we created a dir there deliberately).
        std::fs::remove_dir(&tmp).unwrap();
    }

    #[test]
    fn test_atomic_save_leaves_no_temp_file() {
        // EV-47: after a successful atomic save, the throwaway temp file must be
        // gone (renamed into place).
        let dir = tempdir().unwrap();
        let path = dir.path().join("evolution.ckpt");
        let population: Vec<Individual<RealVector>> =
            vec![Individual::new(RealVector::new(vec![1.0]))];
        save_checkpoint(
            &Checkpoint::new(5, population),
            &path,
            CheckpointFormat::Binary,
        )
        .unwrap();

        let mut tmp = path.clone().into_os_string();
        tmp.push(".tmp");
        assert!(!PathBuf::from(tmp).exists(), "temp file must not remain");
    }

    #[test]
    fn test_load_rejects_oversized_file() {
        // regression: EV-48 - loading must reject a file larger than the limit
        // with a typed error instead of an unbounded read/allocation.
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.ckpt");
        let population: Vec<Individual<RealVector>> = (0..50)
            .map(|i| Individual::new(RealVector::new(vec![i as f64; 8])))
            .collect();
        save_checkpoint(
            &Checkpoint::new(1, population),
            &path,
            CheckpointFormat::Binary,
        )
        .unwrap();

        let err = load_checkpoint_with_limit::<RealVector>(&path, 16).unwrap_err();
        assert!(
            matches!(err, CheckpointError::TooLarge { limit: 16, .. }),
            "expected TooLarge, got {err:?}"
        );
    }

    #[test]
    fn test_load_rejects_corrupt_length_prefix() {
        // regression: EV-48 - a corrupted length prefix must not drive a huge
        // allocation; it is caught by the limit and returned as a typed error.
        let dir = tempdir().unwrap();
        let path = dir.path().join("corrupt.ckpt");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(b"FEVC");
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // absurd length prefix
        std::fs::write(&path, &bytes).unwrap();

        let err = load_checkpoint::<RealVector>(&path).unwrap_err();
        assert!(
            matches!(err, CheckpointError::TooLarge { .. }),
            "expected TooLarge, got {err:?}"
        );
    }

    #[test]
    fn test_load_latest_orders_across_digit_boundary() {
        // regression: EV-87 - lexicographic filename ordering picks the wrong
        // "latest" once the index widens ("100000" < "99999" as strings). The
        // manager must order by parsed numeric index instead, and still parse
        // legacy unpadded names.
        let dir = tempdir().unwrap();

        // Emulate legacy unpadded filenames straddling the 5->6 digit boundary.
        for (idx, gen) in [(99999usize, 99999usize), (100000, 100000)] {
            let path = dir.path().join(format!("evolution_{idx}.ckpt"));
            let population: Vec<Individual<RealVector>> =
                vec![Individual::new(RealVector::new(vec![gen as f64]))];
            save_checkpoint(
                &Checkpoint::new(gen, population),
                &path,
                CheckpointFormat::Binary,
            )
            .unwrap();
        }

        let manager = CheckpointManager::new(dir.path(), "evolution");
        let loaded: Option<Checkpoint<RealVector>> = manager.load_latest().unwrap();
        assert_eq!(
            loaded.unwrap().generation,
            100000,
            "index 100000 is newer than 99999 despite lexicographic order"
        );
    }

    #[test]
    fn test_current_path_is_zero_padded_to_8() {
        // EV-87: names must be fixed-width 8-digit so string and numeric order agree.
        let dir = tempdir().unwrap();
        let manager = CheckpointManager::new(dir.path(), "evolution");
        let name = manager.current_path();
        assert!(
            name.file_name().unwrap().to_string_lossy() == "evolution_00000000.ckpt",
            "got {name:?}"
        );
    }
}
