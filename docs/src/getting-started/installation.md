# Installation

Add fugue-evo to your Rust project using Cargo.

## Basic Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
fugue-evo = "0.4"
rand = "0.8"
```

The `rand` crate is required for random number generation in evolutionary algorithms.

## Feature Flags

Fugue-evo is two layers behind two features, both on by default:

```toml
[dependencies]
fugue-evo = { version = "0.4", features = ["std", "ppl", "classic", "parallel", "checkpoint"] }
```

### Available Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `ppl` | Yes | The inference layer: priors as programs, `EvolutionModel`, MH / tempered SMC, grammar GP (adds the `fugue-ppl` dependency) |
| `classic` | Yes | The classic EC toolkit: SimpleGA, CMA-ES, NSGA-II, Island Model, operators, interactive GA, checkpointing (owns the `nalgebra`, `rand_chacha`, `serde_json` dependencies) |
| `parallel` | Yes | Rayon-based parallel fitness evaluation (classic code paths) |
| `checkpoint` | Yes | Save/restore evolution state to files (classic code paths) |

The minimum supported Rust version is 1.87 (inherited from `fugue-ppl`).

### Inference layer only

The configuration a downstream probabilistic-programming user wants — no
classic EC code and none of its dependencies (this is also the configuration
to build for `wasm32-unknown-unknown`):

```toml
[dependencies]
fugue-evo = { version = "0.4", default-features = false, features = ["std", "ppl"] }
```

### Classic toolkit only

The standalone EC toolkit with no probabilistic-programming dependency at all:

```toml
[dependencies]
fugue-evo = { version = "0.4", default-features = false, features = ["std", "parallel", "checkpoint", "classic"] }
```

Both of these configurations are built and tested in CI alongside
`--all-features`.

## WASM Support

For browser-based optimization, use the WASM package:

```toml
[dependencies]
fugue-evo-wasm = "0.1"
```

Or install via npm for JavaScript projects:

```bash
npm install fugue-evo-wasm
```

See [WASM & Browser Usage](../how-to/wasm.md) for detailed setup instructions.

## Verifying Installation

Create a simple test program to verify your installation:

```rust,ignore
use fugue_evo::prelude::*;

fn main() {
    // Create a simple real vector genome
    let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
    println!("Genome: {:?}", genome.genes());

    // Create bounds for optimization
    let bounds = MultiBounds::symmetric(5.0, 3);
    println!("Bounds: {:?}", bounds);

    println!("fugue-evo is working!");
}
```

Run with:

```bash
cargo run
```

## Development Installation

To work with the latest development version:

```toml
[dependencies]
fugue-evo = { git = "https://github.com/fugue-evo/fugue-evo" }
```

## Next Steps

Now that fugue-evo is installed, continue to:

- [Core Concepts](./concepts.md) - Understand the fundamental abstractions
- [Quick Start](./quickstart.md) - Run your first optimization
