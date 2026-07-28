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
  recovered from bare particle traces by decode-replay.

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

## The layer boundary

The `TraceGenome` extension trait (`genome::trace_genome`) is the boundary:
classic algorithms require only `EvolutionaryGenome`; genomes that also
implement `TraceGenome` can be driven by the inference layer. The canonical
trace encoding is pure data (zero stored log-probabilities); probability mass
always comes from scoring under a prior program.
