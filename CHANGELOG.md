# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-05

**Audit follow-up (AUDIT-2026-09: EV-N1 … EV-N5, X-3, X-5).** Requires
fugue-ppl **0.2.3** (its FG-N1 … FG-N9 / X-5 fixes): this crate now calls
`adaptive_single_site_mh_cached`, `score_given_trace_reconciled`,
`PopulationKernel::is_identity`, `NoKernel`, and relies on the
support-based `f64` proposal selection. The dependency's `version`
requirement is `0.2.3` accordingly; a published `fugue-ppl 0.2.2` does
**not** have these APIs, which is why the first publish attempt of 0.4.0
(made while the requirement still read `0.2.1`) failed its dry run.

### Breaking

- **`CrossoverMaskFn` is `Box<dyn Fn(..) -> Vec<Address> + Send>`**,
  matching fugue's `CrossoverKernel::mask`. Closures that capture only
  `Send` state — every mask in this crate — already satisfy it.
- **`EvolutionModel::score`, `log_boltzmann_target`, `to_weighted_trace`
  and `smc::score_genome` return `Result<_, GenomeError>`** (EV-N3, below).
- **`EvolutionPosterior::weighted_mean` / `weighted_variance` return
  `Option<f64>`**: `None` when no particle carries the real coordinate
  `<prefix>#coord` — a tree genome, a coordinate past the genome's
  dimension, an empty population — where they used to return a silent
  `0.0` (EV-N5).
- **`nalgebra`, `rand_chacha` and `serde_json` are optional, gated behind
  `classic`** (`classic = ["dep:nalgebra", "dep:rand_chacha",
  "dep:serde_json"]`). A `--no-default-features --features std,ppl` build —
  auracle's configuration, and its WASM bundle — no longer compiles
  `nalgebra` or `serde_json` at all (`rand_chacha` remains in that graph only
  as `rand`'s own `StdRng` backend, which is not this crate's to gate).
  Nothing changes for default-feature users; a downstream crate that
  used one of these through fugue-evo's dependency graph without depending
  on it must add it (EV-N4 / X-3).
- **`EvolutionModel::with_beta` panics on a non-finite `β` and
  `with_temperature` on `T ≤ 0` or non-finite `T`.** `T = 0` used to set
  `β = ∞`, building `factor(∞·f(x))` — `NaN` wherever `f = 0` — a target no
  sampler can move on. Negative `β` still clamps to `0` (the prior).
  Optimizer mode is `EvolutionSMC::anneal` with a large finite `beta_max`
  (EV-N5).
- **`rust-version = "1.87"`** is declared (fugue-ppl requires it); the
  crate never built on older toolchains, it just did not say so (X-2).

### Fixed

- **`EvoSmcConfig::default()` no longer panics on variable-structure priors,
  and grammar priors can be annealed (EV-N1).** The generic crossover mask
  was a random subset of the *first* parent's addresses; on the grammar
  prior `swap_block` could move a site the partner lacks out of a child,
  and fugue's `ScoreGivenTrace` re-score panicked on it — so every grammar
  test and example passed `crossover: None`, and `anneal` (which had no
  `_with_kernel` variant) could not be used on trees at all.

  Address-set intersection is not enough: exchanging a structural site
  whose value differs between the parents (`node#leaf` on a leaf-rooted vs
  a function-rooted tree) opens a branch the child has no choices for. The
  new `SharedSiteCrossover` kernel (a `fugue::PopulationKernel`, what
  `EvoSmcConfig::crossover` now builds) swaps a coin-flipped subset of the
  addresses present in **both** parents with the same value type —
  value-independent and pair-symmetric — and re-scores each child with
  `score_given_trace_reconciled`, accepting the pair only when neither
  report lists a fresh or a vanished site. An accepted pair therefore has
  exactly the parents' address sets, the swap is an involution, and the
  rejected proposals are self-loops: the move is symmetric and leaves the
  product of tempered targets invariant. On a fixed-structure prior it is
  exactly fugue's `CrossoverKernel`; on a variable-structure prior it
  exchanges constants, variable indices and same-arity function choices,
  never structure — documented on `CrossoverConfig`. Subtree grafts remain
  the model-aware `subtree_crossover_mask` through `run_with_kernel` and
  the new **`EvolutionSMC::anneal_with_kernel`**. `shared_site_crossover_mask`
  is exported for callers building fugue's kernel on a fixed-structure model.

  The reconciling scorer is used instead of the strict one deliberately:
  `StrictScoreGivenTrace` hands the program `Default::default()` after
  recording a missing site, and the grammar reads a missing `#leaf` as
  `false` ("function node") and recurses without bound — a stack overflow
  reproduced while building this fix. Pinned by
  `test_default_config_runs_on_grammar_prior`, `test_anneal_runs_on_grammar_prior`
  and `test_shared_site_crossover_rejects_structural_mismatch`; the
  fixed-structure crossover anchor still reproduces the conjugate posterior.
- **`EvolutionChain::step` honours `override_site` and costs one model
  execution (EV-N2 / X-5).** `step` ran `adaptive_single_site_mh`, which
  ignored the chain's override table (only `run_chain` passed it through)
  and re-scored `current` on every call — two program executions, hence two
  fitness evaluations, per transition. It now delegates to the new
  **`step_scored`** — one call into fugue's
  `adaptive_single_site_mh_cached` with the overrides; `None` on rejection —
  and on rejection decodes the genome by replaying the **prior program
  only** (**`decode`**), so the fitness is evaluated exactly once per step.
  `step`'s `(genome, trace)` signature is unchanged. The contract is now
  stated on `step`: `current` must come from `init`, `init_from` or a
  previous `step`/`step_scored`; a `to_trace`/`trace_of` trace (per-site
  `logp = 0`) fed straight to `step` over-accepts structure-shrinking moves
  until the first acceptance. Bounded `Uniform` sites need no override:
  fugue selects the reflected walk from `Distribution::support()`, so
  `UniformBoxPrior` mixes through `step()` out of the box — pinned by the
  EV-90 anchor re-run on `[-0.5, 0.5]` over four seeds
  (`test_mh_bounded_prior_containing_negatives_mixes_across_zero`: analytic
  mean `a·coth(a) − 1 = 0.0820`, `P(x > 0) = 0.6225`; the pre-fix fugue
  selector confined this chain to the sign of its first state). Also
  pinned by `test_step_honours_override_site` and
  `test_step_costs_one_fitness_evaluation`.
- **`score` / `init_from` / `to_weighted_trace` return errors instead of
  panicking on a structural mismatch (EV-N3).** A `RealVector` shorter than
  the prior's dimension panicked, a longer one was silently truncated, and
  any likelihood with a latent nuisance site (`NoiseSpec::Infer`, a Pareto
  weight) made `init_from` panic on every genome. `GenomePrior` gains
  `validate(&genome) -> Result<(), GenomeError>` (the vector priors return
  `DimensionMismatch { expected, actual }`); `score` reports
  `MissingAddress(site)` for a site the program visits that the encoding
  lacks (a latent site) and `InvalidStructure` for sites the program never
  visits; out-of-support still scores `−∞`. New: `score_with_latents(rng,
  genome)` draws the latent sites from their priors and returns a complete
  scored state; `EvolutionChain::try_init_from` gives the reason
  `init_from` (still `Option`) returns `None`; `init_from_with_latents(rng,
  genome)` warm-starts a chain whose likelihood has latent sites. Pinned by
  `test_score_rejects_dimension_mismatch`,
  `test_latent_site_likelihood_is_an_error_not_a_panic`,
  `test_vector_priors_validate_dimension`.
- **CI builds the two feature configurations `CLAUDE.md` calls CI-relevant
  (EV-N4 / X-3).** A `features` matrix job checks and tests
  `--no-default-features --features std,ppl` natively, checks it on
  `wasm32-unknown-unknown`, and checks and tests
  `std,parallel,checkpoint,classic`; `make feature-matrix` runs the same
  locally and is part of `make ci`. Nothing built either before; an ungated
  reference across the classic/ppl boundary would have broken downstream
  builds without failing `--all-features`.
- **`EvolutionSMC::anneal` keeps one proposal-scale adaptation across all
  annealing rungs (EV-N5).** fugue's `rejuvenate_particles` starts a fresh
  `DiminishingAdaptation` on every call, so each rung re-learned its scales
  from the default. Rejuvenation at rung `β` now runs
  `adaptive_single_site_mh_cached` against the model's own fixed-`β`
  program (`target_model()` at `β` — exactly fugue's tempered density
  whenever the likelihood tempers linearly, which `FactorFitness` and
  `tempered_observe` do) from a single adaptation shared by every rung and
  particle, re-scoring under the `β = 1` program afterwards for the next
  reweight; per particle and rung this is `steps + 2` executions instead of
  `2·steps`. `anneal` with `rejuvenation_steps == 0` and no kernel only
  reweights and resamples past `β = 1` (duplicates of the fittest
  particles); the docs now say so.
- **`BayesianAdaptiveGA` children never leave the prior's support
  (EV-N5).** A mutant outside the prior's support (a `UniformBoxPrior`'s
  box, a grammar's depth limit) is discarded before evaluation and the
  parent keeps its slot, counted as a failed trial. The docs now state what
  the prior is used for: initial population and feasible region — its
  density does not enter selection; this is a GA with Bayesian
  operator-selection, not a posterior sampler. Pinned by
  `test_children_stay_inside_prior_support`.
- **Doctests are compiled again (EV-N5).** The `interactive` module's six
  examples and the two crate-level quick starts were `ignore`d; they now
  compile (the interactive loops as `no_run`, the quick starts run under
  their feature via `cfg_attr`), so the inference-API example in `lib.rs`
  is checked against the real API on every `cargo test`.
- **`test_subtree_crossover_swaps_prefix_range` asserts the swap happened
  (EV-N5)**: grafting two subtrees rooted at the same path conserves the
  pair's prior mass, so under the prior-only target every non-trivial
  proposal is accepted and at least two particles must differ from their
  prior draws — previously the test ended in `let _ = (before, after)`.
- **The `Fitness` / `MultiObjectiveFitness` `Send + Sync` split is
  documented as a deliberate, known non-additive feature (EV-N5).** With
  `parallel` the traits have `Send + Sync` supertraits (rayon), without it
  they do not, so a crate built without `parallel` can implement `Fitness`
  for a `!Send` type and break when another crate enables the feature.
  Requiring the bound unconditionally was implemented and then reverted:
  the crate's own WASM bindings (`fugue-evo-wasm`, built with `std,classic,ppl`
  and no `parallel`) implement `Fitness` over closures that capture a
  `js_sys::Function`, which is `!Send`, and any browser consumer is in the
  same position; the unconditional bound would have forced them into
  `unsafe impl Send` wrappers. The trait docs now state the split, the
  rationale, and the advice (make the fitness `Send + Sync` if the crate must
  build both ways; the inference layer requires that independently).
- **Docs**: `SPEC.md` and `docs/src/api-docs.md` no longer describe
  `fugue_integration::{EvolutionarySMC, EvolutionStep}` (deleted in
  0.2.0); they name `inference::{EvolutionSMC, EvolutionChain}` and the
  real `src/inference/` layout. The installation page documents the `ppl`
  / `classic` features, the two supported reduced configurations and the
  MSRV; the README's feature paragraph states what CI actually builds.

### Added

- `EvolutionSMC::anneal_with_kernel`, `SharedSiteCrossover`,
  `shared_site_crossover_mask` (EV-N1).
- `EvolutionChain::{step_scored, decode, overrides, try_init_from,
  init_from_with_latents}` (EV-N2, EV-N3).
- `EvolutionModel::score_with_latents`, `GenomePrior::validate` (EV-N3).
- `make feature-matrix` / `test-ppl` / `check-ppl-wasm` / `test-classic`
  (EV-N4).

## [0.3.1] - 2026-07-28

### Added

- **Chebyshev scalarization for non-convex Pareto fronts**
  (`inference::pareto::ChebyshevScalarization`): the weighted-max norm
  `max_i w_i·(f_i − z_i)` over an ideal point `z`, reaching every weakly
  Pareto-optimal point — including the non-convex front regions where every
  weighted-sum optimum provably collapses onto the endpoints. Supports a
  latent weight (like `ParetoScalarization`) or a **fixed** weight
  (`with_weight`) for uniform front sweeps. Pinned at the theorem level on
  the concave front `f1 = x, f2 = 1 − x²`: fixed-w weighted-sum mass avoids
  the interior (< 0.1) while fixed-w Chebyshev concentrates on the interior
  front point `x* = (√5−1)/2`, and a weight sweep traces the whole front.
- **Honest marginal-tilt documentation** for the latent-weight Pareto models:
  the `w`-marginal is tilted by `exp(−s·m(w))` (the scalarized optimum's
  value), so the *conditional* `x | w` is what tracks the front; uniform
  coverage comes from fixed-weight sweeps. New regression
  `test_chebyshev_latent_weight_conditional_tracks_front` pins the
  conditional property.
- **"Evolution as inference" explorable** (evo.fugue.run, Architecture →
  Evolution as Inference): `ExploreSmcInference` in `fugue-evo-wasm` steps
  the crate's real inference layer one tempering rung at a time — Gaussian
  prior program, twin-peaks Boltzmann target, fugue's SMC primitives
  (reweight / ESS-triggered systematic resampling / typed-MH rejuvenation /
  crossover kernel), exact tempered-density heat recomputed each rung,
  β-ladder up to annealed-optimizer territory, and live ESS / log-evidence /
  swap readouts. The wasm crate now enables the `ppl` feature and depends on
  `fugue-ppl` directly. Engine pinned by determinism, ladder-shape,
  analytic-posterior-mean, analytic-evidence, and annealing-concentration
  tests; the page verified live in a browser against the built wasm.


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
