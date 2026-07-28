# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-28

**Inference-first.** fugue-evo's identity is now "an implementation of fugue
for running evolutionary algorithms as Bayesian inference"; the classic EC
toolkit is a standalone, feature-gated companion. This release closes the
three caveats left by 0.2.0: black-box-only fitness, the classic layer's
monopoly on optimization/multi-objective, and the tree-encoding seam.

### Added

- **Likelihoods as programs** (`inference::likelihood`): the new
  `GenomeLikelihood<G>` trait — an observation program `p(data|x)` that may
  contain per-datum `observe` statements, `factor`s, and **latent nuisance
  parameters jointly inferred with the genome**. `tempered_observe` helper;
  `FactorFitness` adapter keeps the classical black-box mode as an explicit
  Gibbs / generalized-Bayes posterior; `NoLikelihood` for prior-only runs.
  `GaussianRegression` (`inference::grammar`) demonstrates the payoff: the
  observation noise is a latent site (`NoiseSpec::Infer`), and its posterior
  is read off the particle traces — pinned by
  `test_symreg_infers_noise_jointly` (recovers a true sigma of 0.3).
- **Optimizer mode** (`EvolutionSMC::anneal`): continue the tempering ladder
  past beta = 1 toward `beta_max` (incremental reweight + resample +
  pi_beta-invariant rejuvenation + optional crossover sweeps, all fugue
  primitives), concentrating the population on the optima — a principled,
  uncertainty-carrying single-objective optimizer. Pinned by
  `test_anneal_concentrates_on_optimum`; head-to-head with SimpleGA in
  `examples/optimize_by_inference.rs`.
- **Multi-objective as inference** (`inference::pareto`):
  `ParetoScalarization` puts the scalarization weight *inside the model*
  (uniform-simplex stick-breaking Beta sites), so the joint posterior's
  marginal traces the Pareto front and `particle_weights` reads each
  particle's front position off its trace. Pinned analytically by
  `test_pareto_posterior_traces_the_front` (biobjective with Pareto set
  [0,2]: mass on the set, both ends covered, particles near their weight's
  scalarized optimum x* = 2(1-w)).
- **Prior-owned encodings** (`GenomePrior::trace_of`): encode a genome under
  *the prior's* address scheme (default: the canonical `TraceGenome`
  encoding; `ArithmeticGrammarPrior` overrides with the exact inverse of its
  generative walk — pinned by `test_trace_of_inverts_generative_run` and a
  hand-computed PCFG score). `EvolutionModel::score`/`to_weighted_trace` now
  work for grammar trees, and the new `EvolutionChain::init_from(genome)`
  warm-starts a chain from any in-support genome — including a classic GA/GP
  result.
- **`MemoizedFitness`**: exact-key (bincode) shared-cache fitness wrapper,
  removing repeated evaluations under replay-heavy inference.

### Changed (breaking)

- `EvolutionModel<P, F>` is now `EvolutionModel<P, L: GenomeLikelihood>`.
  `EvolutionModel::new(prior, fitness)` still works (it now returns
  `EvolutionModel<P, FactorFitness<F>>`); explicit type annotations need the
  `FactorFitness` wrapper. `from_likelihood(prior, likelihood)` accepts any
  observation program. `fitness_value`/`log_weight`/`to_weighted_trace` are
  specific to the `FactorFitness` mode (EV-52 unchanged and green).
- **`classic` feature (default on)**: `algorithms`, `operators`,
  `population`, `hyperparameter`, `interactive`, `checkpoint`,
  `diagnostics`, `termination` are now gated. `--no-default-features
  --features std,ppl` builds the inference layer with no classic EC code;
  `--features std,parallel,checkpoint,classic` builds classic with no fugue.
  `MultiObjectiveFitness`/`ClosureMultiObjective` moved to the core
  `fitness::multi_objective` (re-exported from `algorithms::nsga2`).
- Crate description and README lead with the inference identity.


## [0.2.0] - 2026-07-28

**"Evolutionary algorithms as probabilistic programs"** — the two-layer
refactor (cross-repo plan, tracking issue
[#18](https://github.com/alexnodeland/fugue-evo/issues/18); upstream
primitives in fugue-ppl 0.2.1 / fugue#45). fugue-evo is now explicitly two
layers: a standalone classic EC layer with **no** fugue dependency, and a
fugue-native inference layer where the Boltzmann target is literally a fugue
program and every sampler is fugue's own inference machinery.

### Changed (breaking)

- **Trait split**: `EvolutionaryGenome` no longer has `to_trace` /
  `from_trace` / `trace_prefix`. They moved to the new `TraceGenome`
  extension trait (`genome::trace_genome`, behind the `ppl` feature); bring
  it into scope with `use fugue_evo::genome::trace_genome::TraceGenome`.
  The `ChoiceValue` re-export moved there too.
- **`ppl` feature (default on)**: `fugue-ppl` is now optional. With
  `--no-default-features --features std,parallel,checkpoint` the entire
  classic layer (all 8 algorithms, operators, wasm crate) compiles with no
  fugue dependency.
- **`fugue_integration` renamed to `inference`** (deprecated alias kept for
  one release).
- **`Prior` enum removed** — priors are programs now. `GenomePrior::model()
  -> fugue::Model<G>` returns the decoded genome; built-in constructors:
  `UniformBoxPrior`, `GaussianPrior`, `BitStringPrior`, `PermutationPrior`,
  and the PCFG `ArithmeticGrammarPrior`. All hand-written density code
  (`log_prior_density`, `log_boltzmann_target` internals) is deleted;
  scoring is `ScoreGivenTrace` replay of the target program.
- **`EvolutionModel<G, F>` is now `EvolutionModel<P: GenomePrior, F>`**:
  `EvolutionModel::new(prior, fitness)`. `target_model()` builds the fixed-β
  Boltzmann program for MH; `smc_model()` builds the untempered program for
  SMC (β applied exactly once by fugue's adaptive tempering — fixing the old
  hand-rolled SMC's β double-counting).
- **`EvolutionStep` removed** → `EvolutionChain`, a thin wrapper over
  `fugue::adaptive_single_site_mh`. Typed proposals move **every** site kind;
  the old proposal only perturbed `F64` choices, so BitString/Permutation
  chains silently never moved (new regressions:
  `test_bitstring_chain_moves`, `test_permutation_chain_moves`).
- **`Permutation`'s trace encoding is now the Lehmer code** (ranks against
  the shrinking available-value list) instead of raw values, matching the
  sequential-categorical `PermutationPrior` so single-site MH moves decode to
  valid, distinct permutations.
- **`EvolutionarySMC` removed** → `EvolutionSMC::run` /
  `run_with_kernel` over `fugue::adaptive_smc_with_kernel`: adaptive
  ESS-driven β ladder, systematic resampling, per-particle rejuvenation, the
  population-coupled `CrossoverKernel`, and an unbiased **log-evidence**
  estimate. Results are `EvolutionPosterior` (fugue particles); genomes are
  recovered by decode-replay (`best`, `genomes`, `weighted_mean/variance`).
- **`BayesianAdaptiveGA::new(prior, fitness, pop, gens)`** (was
  `(fitness, bounds, ..)`); its conjugate `Beta`/`Gamma` machinery now uses
  `rand_distr` instead of fugue distributions.

### Added

- **`ArithmeticGrammarPrior`** (`inference::grammar`): expression trees as a
  probabilistic context-free grammar program with tree-path addresses
  (`node/0/1#leaf`, `#func`, `#const`, …). Structure lives in the choices, so
  fugue's generic machinery becomes genetic programming: single-site MH on a
  `#leaf`/`#func` site births/kills subtrees with automatic reversible-jump
  corrections (subtree regeneration), and `subtree_crossover_mask()` +
  `fugue::CrossoverKernel` grafts subtrees between particles (subtree
  crossover). Parsimony is the grammar prior itself.
- **Flagship example** `examples/symbolic_regression_inference.rs`: symbolic
  regression posed as exact Bayesian inference — PCFG prior, Gaussian
  likelihood factor, tempered SMC with both genetic moves, MAP program by
  decode-replay, posterior-predictive readout, and grammar comparison by
  Bayes factor. Pinned by `test_symreg_recovers_known_expression` (recovers
  `x² + 1`).
- Analytic regression anchors kept green through the rewrite: EV-16
  (conjugate SMC posterior, now with an added analytic *evidence* check),
  EV-52 (weighted trace = β·f), EV-90 (MH truncated-exponential mean),
  EV-53 (conjugate updates / Thompson preference).


## [0.1.1] - 2026-07-21

### Added

- Package metadata now includes `documentation` (docs.rs/fugue-evo) and `homepage` (evo.fugue.run), so crates.io shows the documentation link.
- The island model (`algorithms::island`) is now available without the `parallel` feature: islands evolve sequentially when rayon is absent (e.g. wasm32 builds), with identical seeded results thanks to the per-island RNGs (EV-12).
- `fugue-evo-wasm` gains incremental explorable engines (`ExploreGa`, `ExploreCma`, `ExploreNsga2`, `ExploreIsland`, `ExploreUmda`, plus `explore_landscape_grid`/`explore_landscape_info`): seeded, generation-by-generation `step()` APIs streaming population, fitness, CMA-ES covariance/eigenstructure, Pareto ranks, and migration events as JSON — the compute layer behind the interactive docs at evo.fugue.run.
- Interactive explorable documentation (evo.fugue.run): the fugue-viz foundation (navy theme, seeded canvas widgets, lazy init) with evo-specific explorables — a live CMA-ES covariance ellipse, NSGA-II Pareto front formation, island-model migration, GA operator anatomy, UMDA model contraction — plus a WASM playground page running the real crate in the browser.

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
