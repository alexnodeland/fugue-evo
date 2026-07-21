# Playground

Everything on this page runs the **real fugue-evo crate**, compiled to
WebAssembly. Pick an algorithm and a landscape, press **Run**, and the crate's
actual implementations — `SimpleGA`'s tournament/SBX/polynomial operators,
`CmaEs::step` with its adapting covariance, `Nsga2`'s non-dominated sorting,
the island model's migration machinery, continuous UMDA's model refitting —
advance one generation per tick and stream their state onto the canvas. The
populations you watch are fugue-evo's populations, not a JavaScript imitation
of them.

<div class="fugue-explorable" data-viz="evo-playground" data-seed="11"></div>

## Things to try

1. **Watch CMA-ES learn a valley.** Pick CMA-ES on Rosenbrock. The blue
   ellipse is the search distribution (σ²C): it elongates along the curved
   valley, travels down it, and contracts onto the optimum — covariance
   adaptation in one picture.
2. **Trade exploration for exploitation.** SimpleGA on Rastrigin: raise the
   tournament size and the population commits to a basin fast (sometimes the
   wrong one); lower it and diversity survives longer. The convergence strip
   below the canvas is the receipt.
3. **Ask for a front, not a point.** NSGA-II on ZDT3: the population spreads
   along a *disconnected* Pareto front — five separate green arcs pushing onto
   the analytic curve. No single-objective run can give you that shape.
4. **Time the migrations.** Island model with interval 16, then 4: on a
   multimodal landscape, sparse migration lets islands diverge (good
   exploration, slow sharing); frequent migration behaves like one big
   population. Watch stuck islands drop right after each violet flash.
5. **Replay everything.** Every run is seeded — scrub the seed and the whole
   evolution replays deterministically. Same seed, same history, every time.

## How this works

`crates/fugue-evo-wasm` (in the fugue-evo repository) exposes seeded,
incremental engines over the crate's algorithms: construct one with a
configuration, then drive `step()` per animation frame and read the
generation's state back as JSON — population positions and fitness, the
CMA-ES mean/σ/covariance/eigenstructure, Pareto ranks and crowding distances,
migration events. Algorithm state lives in Rust; JavaScript only draws. The
interactive figures throughout these docs use the same package.

## Go deeper

- **Tutorials:** [CMA-ES](./tutorials/cmaes.md),
  [Island Model](./tutorials/island-model.md),
  [Multimodal Optimization](./tutorials/multimodal-optimization.md),
  [Multi-Objective Optimization](./how-to/multi-objective.md).
- **The real thing:** [Installation](./getting-started/installation.md) —
  `cargo add fugue-evo` for the full library: custom genomes and fitness,
  checkpointing, parallel evaluation, hyperparameter learning.
- **API:** [docs.rs/fugue-evo](https://docs.rs/fugue-evo).
