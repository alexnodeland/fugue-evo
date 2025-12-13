# Genome Types Reference

This section documents all built-in genome types in fugue-evo.

## Overview

| Genome | Module | Use Case |
|--------|--------|----------|
| [RealVector](./genomes/real-vector.md) | `genome::real_vector` | Continuous optimization |
| [BitString](./genomes/bit-string.md) | `genome::bit_string` | Binary/discrete optimization |
| [Permutation](./genomes/permutation.md) | `genome::permutation` | Ordering problems |
| [TreeGenome](./genomes/tree-genome.md) | `genome::tree` | Genetic programming |

## Core Trait

All genomes implement `EvolutionaryGenome`:

```rust,ignore
pub trait EvolutionaryGenome: Clone + Send + Sync + Serialize + DeserializeOwned {
    /// Convert to Fugue trace representation
    fn to_trace(&self) -> Trace;

    /// Reconstruct from trace
    fn from_trace(trace: &Trace) -> Result<Self, GenomeError>;
}
```

## Specialized Traits

Additional traits for specific genome types:

```rust,ignore
/// Real-valued genomes
pub trait RealValuedGenome: EvolutionaryGenome {
    fn genes(&self) -> &[f64];
    fn genes_mut(&mut self) -> &mut [f64];
    fn dimension(&self) -> usize;
}

/// Binary genomes
pub trait BinaryGenome: EvolutionaryGenome {
    fn bits(&self) -> &[bool];
    fn bits_mut(&mut self) -> &mut [bool];
    fn length(&self) -> usize;
}

/// Permutation genomes
pub trait PermutationGenome: EvolutionaryGenome {
    fn as_slice(&self) -> &[usize];
    fn len(&self) -> usize;
}
```

## Genome Selection Guide

```
What are you optimizing?
        │
        ├── Continuous variables → RealVector
        │
        ├── Yes/No choices → BitString
        │
        ├── Ordering/arrangement → Permutation
        │
        └── Programs/expressions → TreeGenome
```

## See Also

- [Custom Genome Types](../how-to/custom-genome.md) - Create your own
- [API Documentation](../api-docs.md) - Full rustdoc
