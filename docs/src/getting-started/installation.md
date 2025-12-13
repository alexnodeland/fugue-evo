# Installation

Add fugue-evo to your Rust project using Cargo.

## Basic Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
fugue-evo = "0.1"
rand = "0.8"
```

The `rand` crate is required for random number generation in evolutionary algorithms.

## Feature Flags

Fugue-evo provides several optional features:

```toml
[dependencies]
fugue-evo = { version = "0.1", features = ["parallel", "checkpoint"] }
```

### Available Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `parallel` | Yes | Rayon-based parallel fitness evaluation |
| `checkpoint` | Yes | Save/restore evolution state to files |

### Minimal Installation

For embedded or no-std environments:

```toml
[dependencies]
fugue-evo = { version = "0.1", default-features = false }
```

### Full Installation

To enable all features:

```toml
[dependencies]
fugue-evo = { version = "0.1", features = ["std", "parallel", "checkpoint"] }
```

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
