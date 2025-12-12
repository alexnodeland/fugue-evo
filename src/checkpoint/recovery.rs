//! Checkpoint recovery and persistence
//!
//! Provides serialization to/from files with compression and versioning.

use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::state::{Checkpoint, CHECKPOINT_VERSION};
use crate::error::CheckpointError;

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
pub fn save_checkpoint<G>(
    checkpoint: &Checkpoint<G>,
    path: impl AsRef<Path>,
    format: CheckpointFormat,
) -> Result<(), CheckpointError>
where
    G: Clone + Serialize + crate::genome::traits::EvolutionaryGenome,
{
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    match format {
        CheckpointFormat::Json => {
            serde_json::to_writer_pretty(&mut writer, checkpoint)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        }
        CheckpointFormat::Binary => {
            // Write version header first
            writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
            // Write magic bytes for format identification
            writer.write_all(b"FEVO")?;
            // Serialize with bincode
            bincode::serialize_into(&mut writer, checkpoint)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        }
        CheckpointFormat::CompressedBinary => {
            // Write version and magic
            writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
            writer.write_all(b"FEVC")?; // C for compressed
            // Serialize to bytes first
            let bytes = bincode::serialize(checkpoint)
                .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
            // Compress with simple RLE-like compression
            let compressed = compress_data(&bytes);
            // Write length and data
            writer.write_all(&(compressed.len() as u64).to_le_bytes())?;
            writer.write_all(&compressed)?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Load a checkpoint from a file
pub fn load_checkpoint<G>(path: impl AsRef<Path>) -> Result<Checkpoint<G>, CheckpointError>
where
    G: Clone + Serialize + DeserializeOwned + crate::genome::traits::EvolutionaryGenome,
{
    let path = path.as_ref();
    if !path.exists() {
        return Err(CheckpointError::NotFound(path.display().to_string()));
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
        if version > CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                found: version,
            });
        }

        bincode::deserialize_from(&mut reader)
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))
    } else if &header[4..8] == b"FEVC" {
        // Compressed binary format
        let version = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if version > CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                found: version,
            });
        }

        // Read length
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let compressed_len = u64::from_le_bytes(len_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        reader.read_exact(&mut compressed)?;

        // Decompress
        let decompressed = decompress_data(&compressed)
            .map_err(|e| CheckpointError::Corrupted(e))?;

        bincode::deserialize(&decompressed)
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))
    } else {
        // Try JSON format - need to re-read from start
        drop(reader);
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader)
            .map_err(|e| CheckpointError::Deserialization(e.to_string()))
    }
}

/// Simple compression using run-length encoding for repeated bytes
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
        while i + (count as usize) < data.len()
            && data[i + (count as usize)] == byte
            && count < 255
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
    /// Current checkpoint index
    current_index: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(directory: impl Into<std::path::PathBuf>, base_name: impl Into<String>) -> Self {
        Self {
            directory: directory.into(),
            base_name: base_name.into(),
            format: CheckpointFormat::Binary,
            keep_n: 3,
            interval: 100,
            current_index: 0,
        }
    }

    /// Set the serialization format
    pub fn with_format(mut self, format: CheckpointFormat) -> Self {
        self.format = format;
        self
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
        generation > 0 && generation % self.interval == 0
    }

    /// Get the path for the current checkpoint
    pub fn current_path(&self) -> std::path::PathBuf {
        let extension = match self.format {
            CheckpointFormat::Json => "json",
            CheckpointFormat::Binary | CheckpointFormat::CompressedBinary => "ckpt",
        };
        self.directory.join(format!(
            "{}_{:04}.{}",
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
            let old_path = self.directory.join(format!(
                "{}_{:04}.{}",
                self.base_name, old_index, extension
            ));
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
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&self.base_name)
            })
            .collect();

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Sort by modification time (newest first)
        checkpoints.sort_by(|a, b| {
            b.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .cmp(
                    &a.metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                )
        });

        // Try to load the newest checkpoint
        for entry in checkpoints {
            match load_checkpoint(entry.path()) {
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

        let population: Vec<Individual<RealVector>> = vec![
            Individual::new(RealVector::new(vec![1.0, 2.0, 3.0])),
        ];
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
            let population: Vec<Individual<RealVector>> = vec![Individual::new(RealVector::new(vec![gen as f64]))];
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
}
