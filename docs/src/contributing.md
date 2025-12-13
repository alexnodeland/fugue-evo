# Contributing

Thank you for your interest in contributing to fugue-evo! This guide will help you get started.

## Getting Started

### Prerequisites

- Rust stable (1.70+)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/fugue-evo/fugue-evo
cd fugue-evo

# Build
cargo build

# Run tests
cargo test

# Run examples
cargo run --example sphere_optimization
```

## Development Workflow

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_name

# With output
cargo test -- --nocapture

# Property tests
cargo test --test property_tests
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint
cargo clippy

# Check all features
cargo check --all-features
```

### Documentation

```bash
# Generate docs
cargo doc --open

# Build mdbook
cd docs && mdbook build
```

## Code Style

### Formatting

We use `rustfmt` with default settings:

```bash
cargo fmt
```

### Linting

We use `clippy`:

```bash
cargo clippy -- -D warnings
```

### Documentation

- All public items should have doc comments
- Include examples where helpful
- Use `# Panics`, `# Errors`, `# Safety` sections as appropriate

```rust,ignore
/// Brief description.
///
/// Longer description if needed.
///
/// # Arguments
///
/// * `param` - Description
///
/// # Returns
///
/// Description of return value.
///
/// # Example
///
/// ```
/// // Example code
/// ```
pub fn function(param: Type) -> ReturnType {
    // ...
}
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-fix
```

### 2. Make Changes

- Write tests for new functionality
- Update documentation
- Follow existing code patterns

### 3. Test Thoroughly

```bash
cargo test
cargo clippy
cargo fmt -- --check
```

### 4. Commit

Use clear, descriptive commit messages:

```
feat: add new crossover operator for permutations

- Implement cycle crossover (CX)
- Add tests for edge cases
- Update operator reference docs
```

### 5. Submit PR

- Describe what the PR does
- Reference any related issues
- Ensure CI passes

## Types of Contributions

### Bug Fixes

1. Create an issue describing the bug
2. Write a failing test
3. Fix the bug
4. Ensure test passes

### New Features

1. Discuss in an issue first
2. Design API carefully
3. Implement with tests
4. Add documentation
5. Add to mdbook if appropriate

### Documentation

- Fix typos and unclear explanations
- Add examples
- Improve API docs
- Expand tutorials

### Benchmarks

- Add new benchmark functions
- Performance comparisons
- Algorithm benchmarks

## Architecture Guidelines

### Adding a New Algorithm

1. Create module in `src/algorithms/`
2. Implement algorithm struct
3. Create builder with type-safe API
4. Add to prelude
5. Write tests
6. Document thoroughly
7. Add tutorial if complex

### Adding a New Genome Type

1. Create module in `src/genome/`
2. Implement `EvolutionaryGenome` trait
3. Implement appropriate operators
4. Add serialization support
5. Add to prelude
6. Write tests including trace roundtrip

### Adding a New Operator

1. Implement appropriate trait(s)
2. Make `Send + Sync`
3. Document parameters and behavior
4. Add tests
5. Add to reference docs

## Testing Guidelines

### Unit Tests

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let input = setup();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

### Property Tests

We use `proptest` for property-based testing:

```rust,ignore
proptest! {
    #[test]
    fn test_trace_roundtrip(genome in any_genome()) {
        let trace = genome.to_trace();
        let reconstructed = Genome::from_trace(&trace).unwrap();
        prop_assert_eq!(genome, reconstructed);
    }
}
```

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the project's license.
