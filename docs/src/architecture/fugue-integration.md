# Evolution as inference

*This page describes the `ppl` feature (on by default). Without it, fugue-evo
is a standalone evolutionary-computation library with no
probabilistic-programming dependency at all.*

fugue-evo's inference layer makes "evolutionary algorithms as probabilistic
programs" literal. Given a fitness `f` and a prior over genomes, the
Boltzmann/Gibbs posterior

```text
pi_beta(x) ∝ p(x) · exp(beta · f(x))
```

is not merely a mathematical analogy — it is a [fugue](https://fugue.run)
program, and every sampler in the layer is fugue's own inference machinery run
against that program.

<div class="fugue-explorable" data-viz="smc-inference" data-seed="11" data-pop="240" data-rungs="18" data-betamax="1"></div>

*This is the whole idea, live: fugue-evo's real inference layer (compiled to
WASM) runs tempered SMC against a twin-peaks Boltzmann target. The yellow heat
is the exact tempered density `π_β ∝ p(x)·exp(β·f(x))`, recomputed each rung
as β climbs from the prior (β = 0, the blue rings) to the posterior (β = 1);
the green particles are the SMC population — reweighted, resampled when the
ESS drops (watch the ESS readout flash), rejuvenated by typed MH, and mixed
across the two modes by the crossover kernel (a product-target Metropolis
swap of x-coordinates). Push β MAX past 1 to watch inference become an
annealed optimizer, and drag the seed to replay a different history. The
log Z readout is the running evidence estimate — a model score no classic GA
can report.*

## Priors are programs

The `GenomePrior` trait replaces any notion of a built-in prior enum:

```rust,ignore
pub trait GenomePrior: Clone + Send + Sync + 'static {
    type Genome: TraceGenome;
    fn model(&self) -> fugue::Model<Self::Genome>;
}
```

The model samples the genome's canonical trace sites (`gene#i`, `bit#i`,
`perm#i`, …) and returns the assembled genome — the model's return value *is*
the decode. Anything expressible as a fugue model is a valid prior:
correlated coordinates, hierarchical scales, variable-length genomes.
Built-ins: `UniformBoxPrior`, `GaussianPrior`, `BitStringPrior`,
`PermutationPrior` (a Fisher–Yates/Lehmer-code program whose single-site
moves always decode to valid permutations), and `ArithmeticGrammarPrior`
(below).

There is deliberately **no hand-written density code** anywhere in the layer:
prior mass, tempered joints, and MH acceptance ratios are all obtained by
running or replaying the target program through fugue's handlers.

## The target as a program

`EvolutionModel::new(prior, fitness)` assembles

```rust,ignore
prior.model().bind(move |g| {
    let fit = fitness.evaluate(&g);
    factor(beta * fit).map(move |_| g)
})
```

Two builders exist because MH wants a fixed-beta target (`target_model()`)
while tempered SMC must receive the *untempered* factor (`smc_model()`):
fugue's `adaptive_smc` supplies beta by likelihood-tempering, applying it
exactly once.

## Samplers

- **`EvolutionChain`** — Metropolis–Hastings via fugue's
  `adaptive_single_site_mh`. Typed proposals move every site kind: Gaussian /
  log-space walks for reals, flips for bits, prior-resample for categorical
  sites (permutation ranks), with reversible-jump corrections when a proposal
  changes the model's structure.
- **`EvolutionSMC`** — tempered SMC via fugue's `adaptive_smc_with_kernel`:
  an adaptive ESS-driven beta ladder from the prior to the posterior,
  systematic resampling, per-particle MH rejuvenation, an optional
  population-coupled **crossover kernel** (a product-target Metropolis move
  that swaps an address block between two particles), and an unbiased
  **log-evidence** estimate — a genuine Bayesian model score. Genomes are
  recovered from bare particle traces by decode-replay. The kernel that
  `EvoSmcConfig::crossover` builds (`SharedSiteCrossover`) swaps only sites
  present in *both* parents and rejects any child that does not re-score as
  a complete execution over its own address set, so `run` and `anneal` with
  the default config are safe on every prior — on a variable-structure prior
  it exchanges constants, indices and same-arity function choices, never
  structure. Subtree grafts use the model-aware `subtree_crossover_mask`
  through `run_with_kernel` / `anneal_with_kernel`.

## Genetic programming as exact inference

`ArithmeticGrammarPrior` is a probabilistic context-free grammar over
expression trees, written as a fugue program with tree-path addresses
(`node#leaf`, `node/0#func`, `node/0/1#const`, …). Because an execution's
structure is encoded in its own choices:

- **subtree regeneration mutation** is fugue's ordinary single-site MH — a
  flip of one `#leaf` bit births or kills the subtree below it, fresh
  structure drawn from the grammar, reversible-jump corrections applied
  automatically;
- **subtree crossover** is the crossover kernel with a mask that grafts the
  subtrees under one shared node path between two particles;
- **parsimony** is the grammar prior itself — deeper trees pay more mass, no
  ad-hoc penalty needed.

The flagship example, `examples/symbolic_regression_inference.rs`, fits
`x² + 1` by sampling the posterior over programs and compares function-set
grammars by Bayes factor. The regression test
`test_symreg_recovers_known_expression` pins the recovery.

## Typed sequences as exact inference

`SequenceGrammarPrior` does for variable-length **sequences** what the
arithmetic grammar does for trees, for a language you describe with a
`SequenceGrammar`: a state type, the initial state, the finite set of choices
available in a state (grouped into kinds, each with its arguments, optionally
weighted), a transition, and a stop probability. Position `i` emits
`seq/<i>#stop`, `seq/<i>#kind` and `seq/<i>#arg`; a choice with a single
outcome emits no site. The supports depend on the prefix, so a symbol is only
ever proposed where it type-checks.

- The **log-density is a description length**: the sum of the
  log-probabilities of the choices that derive the sequence
  (`description_length`), obtained by replay like every other density here.
- **Edit, extend and truncate** are single-site MH: flipping a `#stop` bit
  births or kills the tail, with fugue's reversible-jump correction (the
  prior densities of born and dead sites as proposal terms, plus the change
  in the number of sites). `validate` rejects a sequence outside the language
  before any replay, and `trace_of` encodes one inside it.
- **Suffix crossover** is `PrefixKeyedSuffixCrossover`: it exchanges the
  tails of two particles at cut points whose prefix states have equal
  user-supplied keys, and corrects the acceptance ratio by the number of
  admissible cut pairs, so it is exact for any key.

The tests compare all three with exact enumeration of a small grammar,
including an exact (sampling-free) detailed-balance check of the crossover.

For the likelihood side of such a search, `MemoizedFactor` wraps a
`DeterministicFactor` (decode a genome to a keyed object or declare it
`Invalid`; score an object) so that each distinct object is scored once
across particles, rungs and replays. `EvolutionPosterior` reads the result
out with `expectation`, `probability`, `distinct` / `distinct_by` and
`marginal_inclusion` over decoded genomes.

## The layer boundary

The `TraceGenome` extension trait (`genome::trace_genome`) is the boundary:
classic algorithms require only `EvolutionaryGenome`; genomes that also
implement `TraceGenome` can be driven by the inference layer. The canonical
trace encoding is pure data (zero stored log-probabilities); probability mass
always comes from scoring under a prior program.

A genome that only exists as the return value of a prior program (it is
generated by the prior, decoded by replay, and encoded by the prior's
`trace_of`) implements the `ReplayGenome` marker instead: blanket impls give
it `EvolutionaryGenome` and `TraceGenome` without writing the classical
methods. `SequenceGenome` is one.
