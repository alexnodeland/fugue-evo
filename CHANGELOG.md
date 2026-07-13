# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Remediation of the full 2026-07 audit (`AUDIT-2026-07.md`, findings EV-01
through EV-106: correctness, math, completeness, usability, elegance, and
docs issues across CMA-ES, hyperparameter learning, interactive/Bradley-Terry
ranking, genome traces, population/operators, algorithms, Fugue integration,
checkpointing, the WASM bindings, and package metadata/dependencies).

### Fixed

- `Individual::set_fitness` now panics on a NaN fitness value, and `Population::best`/`worst`/`sort_by_fitness` treat NaN as strictly worst, so a NaN-fitness individual can no longer be silently reported as the best/worst (EV-07).
- `Population::best`/`worst`/`sort_by_fitness` now rank via `FitnessValue::is_better_than` (new `cmp_by_quality` total order) instead of a `to_f64()` scalar, returning the correct result for `ParetoFitness` with infinite crowding distances (EV-08).
- NSGA-II now recomputes crowding distance per non-dominated front (Deb 2002) rather than over the whole mixed-rank population, correcting binary-tournament parent-selection diversity pressure and the reported `crowding_distance` (EV-13).
- `Individual::genome_mut` now clears the cached fitness, and a new `Individual::set_genome` does the same, so a mutated genome is always re-evaluated (EV-28).
- NSGA-II binary tournament now draws two distinct competitors (sampling without replacement) (EV-84).
- SubtreeMutation (GP) no longer violates a genome's max_depth; the replacement subtree is generated within the depth budget max_depth - depth(mutation point), preventing bloat-control overruns (EV-27).
- Bounded Simulated Binary Crossover now uses Deb & Agrawal's bounds-aware spread factor so offspring fall inside [min,max] by construction, eliminating the probability mass previously piled onto the bounds by clamping (EV-71).
- SwapMutation, PermutationSwapMutation, and InsertMutation `Default::default()` now perform one operation instead of being a silent no-op (EV-101).
- Composite genome trace round-trip now delegates to each component's own to_trace/from_trace under a 'first/'/'second/' namespace, fixing silent data loss for Permutation and Tree components (EV-03).
- TreeGenome trace encode/decode is now lossless: function nodes serialize as their index in the stable ArithmeticFunction::functions() ordering and terminals as a (discriminant, payload) pair, so from_trace(to_trace(g)) reproduces g exactly (EV-04).
- DynamicRealVector::generate no longer panics on empty bounds; added try_generate -> Result for the degenerate case (EV-58).
- from_trace on RealVector/BitString/Permutation now returns GenomeError::TypeMismatch for a present-but-wrong-typed choice instead of silently truncating, distinguishing it from a genuinely missing address (EV-59).
- Deep GP trees: eval/depth/size are now iterative (explicit stack) and an iterative teardown (TreeGenome::dismantle / drop_node_iteratively) is provided so pathologically deep trees no longer overflow the stack (EV-60).
- DynamicRealVector trace I/O now derives its gene address from trace_prefix(), so the advertised and actual prefixes can no longer diverge (EV-91).
- Interactive/Bradley-Terry: the MLE is now re-fit inside the live pairwise loop (via process_pairwise), so pairwise user feedback actually orders candidates (EV-06).
- Interactive/Bradley-Terry: Newton-Raphson uses a Gaussian log-strength prior and MM a Gamma pseudo-count prior, keeping all-win/all-loss candidates finite; the `regularization` field is renamed `prior_lambda` (default 0.1, serde alias retained) (EV-67).
- Interactive/Bradley-Terry: uncertainty is reported on the strength scale for both optimizers via a delta-method, sum-to-zero-constrained Fisher pseudo-inverse (the ridge-inflation bug is gone) (EV-25, EV-66).
- Interactive/Bradley-Terry: the backtracking line search now enforces the correct Armijo sufficient-increase condition (EV-65).
- Interactive/Aggregation: Elo and ImplicitRanking uncertainties are now on the same scale as their means (Elo gains a steady-state floor; ImplicitRanking uses the score-scale binomial variance) (EV-98).
- Interactive/Selection: CoverageAware pairing never returns a self-pair; exploration/coverage bonuses are normalized to a model-agnostic scale by mean population variance, and zero-variance (already-known) pairs score ~0 instead of the max-uncertainty sentinel (EV-68, EV-69, EV-70).
- License metadata reconciled: `fugue-evo` and `fugue-evo-wasm` now declare a single `license = "MIT"` (matching `fugue-ppl`'s `license = "MIT"`), and a root `LICENSE` file (MIT text, copyright Alex Nodeland 2025-2026) is now shipped; README/crate docs no longer claim a dual MIT-OR-Apache-2.0 license with no accompanying license texts (EV-29).
- `fugue-ppl` now resolves to the co-developed sibling crate via `fugue-ppl = { path = "../fugue", version = "0.1.0" }` instead of the published `fugue-ppl = "0.1.0"` crates.io release, so `fugue-evo`'s Fugue integration is finally built and tested against the actual co-developed `../fugue` source rather than a registry release the two crates were never exercised against together — the gap this finding was originally about. This became safe once `fugue`'s own 2026-07 audit remediation landed with a green full-test gate; adapting to that post-remediation API required migrating `genome::composite`'s trace namespacing to the new `Address` struct (`Address::new(..)`/`addr.as_str()` in place of the former tuple-struct `Address(..)` constructor and `.0` field), with no behavior change. The `version = "0.1.0"` field is retained so the dependency still resolves from crates.io if the sibling checkout is absent, and the README "Development" section documents how to pin back to the published release (EV-30).
- `crates/fugue-evo-wasm/Cargo.toml`'s `[profile.release]` (opt-level "s", LTO) has moved to the workspace-root `Cargo.toml`, where Cargo actually honors it; the member manifest previously declared it in a location Cargo silently ignores, leaving the WASM release build unshrunk and non-LTO (EV-31).
- Checkpoint resume is now a first-class library API (EV-02): `SimpleGA::checkpoint_run` snapshots an in-progress incremental run — population, best, evaluations, statistics, and a captured `SnapshotRng` (ChaCha family) — into a `Checkpoint`, and `SimpleGA::resume`/`SimpleGA::run_from_checkpoint` restore it (RNG included) so a resumed run is bit-identical to an uninterrupted one, instead of forcing users to re-implement the generation loop. `resume` rejects a checkpoint with no captured RNG rather than silently diverging. The incremental stepping API (`SimpleGaRun` + `init_run`/`step_generation`/`finish_run`) is now available in all builds (previously `parallel`-gated), and `examples/checkpointing.rs` was rewritten to drive the resume purely through this API.
- Every remaining WASM optimizer now exposes a per-generation progress/cancel callback (EV-34), extending the incremental support beyond the RealVector `SteppedRealOptimizer`: `BitStringOptimizer`, `PermutationOptimizer`, `Nsga2Optimizer`, and `SymbolicRegressionOptimizer` gain `optimizeWithProgress`/`optimizeCustomWithProgress` methods (driven through `SimpleGA::init_run`/`step_generation` and `Nsga2::step`), and `EvolutionStrategyOptimizer`/`UmdaOptimizer` gain `optimizeWithProgress` backed by new native `EvolutionStrategy::run_with_callback` and `UMDA::run_with_callback` hooks. The callback receives `(generation, bestFitness)` (NSGA-II reports the Pareto-front size) and returning `false` cancels the run, so a Web Worker can `postMessage` progress or honor a cancel button instead of blocking on one opaque call.
- Reworded the misleading "we negate because fugue-evo maximizes" comment in `examples/sphere_optimization.rs` (the built-in `Sphere` fitness already negates internally; no user negation is needed) and fixed the printed "Best fitness" to report the un-negated sum-of-squares objective so it reads as the expected near-zero, non-negative value at the optimum (EV-78).

### Changed

- The closure `MultiObjectiveFitness` blanket impl (hardcoded 2 objectives) is replaced by `ClosureMultiObjective::new(num_objectives, closure)`, which reports the true objective count (EV-85).
- SbxCrossover now exposes two separate probabilities — per-pair `crossover_probability` (default 0.9) and per-gene `exchange_probability` (default 0.5, canonical) — via distinct fields and builders (`with_probability`, `with_exchange_probability`) (EV-72).
- Unbounded PolynomialMutation now applies a local Gaussian perturbation (sigma default 0.1*(1+|x|), configurable via `with_unbounded_sigma`) instead of fabricating +/-1e10 bounds (EV-102).
- MutationOperator::mutation_probability now returns Option<f64>, reporting None for the length-dependent 1/n default instead of an untruthful 1.0 (EV-103) **(breaking)**.
- TournamentSelection samples with replacement by default (canonical selection pressure; no longer deterministic when tournament_size >= population size); use `TournamentSelection::without_replacement` for the distinct-competitor variant (EV-104).
- Added length-aware variation operators for DynamicRealVector (cut_and_splice crossover and DynamicGaussianMutation) in the new genome::dynamic_ops module (EV-57).
- Documented the MultiBounds-as-length/depth convention on EvolutionaryGenome::generate and added honest per-type constructors: BitString/Permutation/DynamicRealVector::generate_with_len and TreeGenome::generate_with_depth (EV-94).
- Bounds gained a fallible try_new constructor (rejects min > max); normalize()/denormalize() now handle degenerate min==max bounds (0.5 / min) instead of producing NaN via divide-by-zero (EV-56).
- README/SPEC updated to precisely describe post-remediation behavior: the Bayesian hyperparameter learner is a wired, opt-in `ThompsonSamplingTuner` (`SimpleGABuilder::adaptive_operators` + `run_adaptive`); the Fugue integration runs a genuine tempered-SMC/Boltzmann pipeline with a flagship `examples/bayesian_evolution.rs`; and checkpointing supports bit-identical resume for the ChaCha RNG family (EV-29 through EV-78 doc sweep).
- The duplicate `rand` major in dev/test builds is eliminated by pinning `proptest = ">=1.5, <1.7"`. proptest migrated its internal RNG stack to rand 0.9 in 1.7.0; the 1.5.x/1.6.x line still uses rand 0.8, so pinning below 1.7 collapses the graph back to a single rand major (0.8.5). Verified empirically (`cargo update -p proptest --precise 1.6.0` drops rand 0.9.2/rand_chacha 0.9.0/rand_core 0.9.3, after which `cargo tree -d` shows one rand major and `cargo check --all-targets` + the property-test suite pass). A `make deps-check` target (`cargo tree -d` guard, wired into `make ci`) now fails the build if a duplicate rand major reappears (EV-74).

### Breaking

- `EvolutionaryGenome::distance` is now a required method (no silent 0.0 default) and panics on structural mismatch; a new required `try_distance -> Result` provides the fallible path. RealVector/BitString/Permutation distance no longer silently truncate or report 0.0 on length mismatch (EV-19, EV-20, EV-55, EV-93).
- `Permutation::new_unchecked` renamed to `from_vec_unchecked`, with documented invariants and a debug-build validity assertion (EV-92).

## [0.1.0] - 2025-12-12

### Added

- **Core Genetic Algorithm Framework**
  - `SimpleGA` builder pattern for easy algorithm configuration
  - Generational evolution with configurable operators
  - Elitism support for preserving best individuals

- **Genome Types**
  - `RealVector` for continuous optimization
  - `BitString` for binary/combinatorial problems
  - `Permutation` for ordering problems (TSP, scheduling)
  - `TreeGenome` for genetic programming
  - Unified `EvolutionaryGenome` trait abstraction

- **Selection Operators**
  - `TournamentSelection` with configurable tournament size
  - `RouletteWheelSelection` (fitness-proportionate)
  - `TruncationSelection` for steady-state evolution
  - `RankSelection` for rank-based selection
  - `BoltzmannSelection` with temperature parameter

- **Crossover Operators**
  - `SbxCrossover` (Simulated Binary Crossover) for real-valued genomes
  - `UniformCrossover` for bit strings
  - `SinglePointCrossover` and `TwoPointCrossover`
  - `OrderCrossover` (OX) for permutations
  - `SubtreeCrossover` for tree genomes

- **Mutation Operators**
  - `PolynomialMutation` for real-valued genomes
  - `GaussianMutation` with adaptive step sizes
  - `BitFlipMutation` for bit strings
  - `SwapMutation` and `InsertMutation` for permutations
  - `PointMutation` and `SubtreeMutation` for trees

- **Advanced Algorithms**
  - `CmaEs` (Covariance Matrix Adaptation Evolution Strategy)
  - `NSGA2` for multi-objective optimization with Pareto fronts
  - `IslandModel` for parallel evolution with migration

- **Fugue PPL Integration**
  - `to_trace()` and `from_trace()` for probabilistic programming interop
  - Trace-based evolutionary operators
  - Bayesian hyperparameter learning with `BetaPosterior`

- **Production Features**
  - Checkpointing with `CheckpointManager` (JSON, Binary, Compressed)
  - Convergence detection with configurable criteria
  - Evolution statistics tracking
  - Termination conditions (max generations, target fitness, stagnation)

- **Benchmark Functions**
  - `Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`, `Griewank`
  - `OneMax`, `LeadingOnes` for bit strings
  - `SymbolicRegression` for GP benchmarks

- **Examples**
  - `sphere_optimization.rs` - Basic continuous optimization
  - `rastrigin_benchmark.rs` - Multimodal function optimization
  - `cma_es_example.rs` - CMA-ES for Rosenbrock
  - `island_model.rs` - Parallel island model
  - `checkpointing.rs` - Save/restore evolution state
  - `symbolic_regression.rs` - Genetic programming
  - `hyperparameter_learning.rs` - Bayesian adaptation

- **Testing**
  - Comprehensive unit tests (370+ tests)
  - Property-based tests with proptest (21 tests)
