.PHONY: all build check test lint fmt clippy doc clean ci help

# Default target
all: check

# Build the project
build:
	cargo build

# Build in release mode
release:
	cargo build --release

# Run cargo check (type checking)
check:
	cargo check --all-targets --all-features

# Run all tests
test:
	cargo test --all-features

# Run tests with output
test-verbose:
	cargo test --all-features -- --nocapture

# Run formatting check
fmt:
	cargo fmt --all -- --check

# Run formatting and apply fixes
fmt-fix:
	cargo fmt --all

# Run clippy linter
clippy:
	cargo clippy --all-targets --all-features -- -D warnings

# Run clippy and apply suggestions
clippy-fix:
	cargo clippy --all-targets --all-features --fix --allow-dirty

# Generate documentation
doc:
	cargo doc --no-deps --all-features

# Open documentation in browser
doc-open:
	cargo doc --no-deps --all-features --open

# Clean build artifacts
clean:
	cargo clean

# Run the full CI pipeline (same as GitHub Actions)
ci: fmt clippy check test doc
	@echo "CI pipeline completed successfully!"

# Run quick checks (for development)
quick: fmt-fix clippy check test
	@echo "Quick checks completed!"

# Watch for changes and run tests
watch:
	cargo watch -x test

# Run benchmarks (if any)
bench:
	cargo bench

# Check for security vulnerabilities
audit:
	cargo audit

# Update dependencies
update:
	cargo update

# Show outdated dependencies
outdated:
	cargo outdated

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Default target (runs check)"
	@echo "  build        - Build the project"
	@echo "  release      - Build in release mode"
	@echo "  check        - Run cargo check (type checking)"
	@echo "  test         - Run all tests"
	@echo "  test-verbose - Run tests with output"
	@echo "  fmt          - Check code formatting"
	@echo "  fmt-fix      - Fix code formatting"
	@echo "  clippy       - Run clippy linter"
	@echo "  clippy-fix   - Run clippy and apply fixes"
	@echo "  doc          - Generate documentation"
	@echo "  doc-open     - Generate and open documentation"
	@echo "  clean        - Clean build artifacts"
	@echo "  ci           - Run full CI pipeline"
	@echo "  quick        - Run quick development checks"
	@echo "  watch        - Watch for changes and run tests"
	@echo "  bench        - Run benchmarks"
	@echo "  audit        - Check for security vulnerabilities"
	@echo "  update       - Update dependencies"
	@echo "  outdated     - Show outdated dependencies"
	@echo "  help         - Show this help message"
