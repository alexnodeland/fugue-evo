//! Likelihoods as programs
//!
//! The inference layer's conditioning side. A [`GenomeLikelihood`] is an
//! *observation program* `p(data | genome)` — not merely a scalar score. It
//! may contain:
//!
//! - `observe` statements over real data (per-datum log-likelihoods land in
//!   the trace's `log_likelihood` accumulator with genuine structure),
//! - **latent nuisance parameters** (`sample` sites — e.g. an unknown
//!   observation noise `σ` — which are then *jointly inferred* with the
//!   genome; their posteriors are read straight off the particle traces),
//! - `factor` statements for soft constraints or black-box scores.
//!
//! The black-box case — an arbitrary fitness `f` entering as `factor(β·f)` —
//! is the [`FactorFitness`] adapter. That target is a **generalized-Bayes /
//! Gibbs posterior** (Bissiri, Holmes & Walker 2016): perfectly legitimate,
//! but now one mode among many rather than the only one.
//!
//! [`MemoizedFitness`] caches a scalar fitness by the genome's exact
//! serialization. [`MemoizedFactor`] caches a [`DeterministicFactor`] by a
//! **user-supplied key** of the decoded object, so genomes that decode to
//! the same object share one evaluation, and lets a genome be declared
//! [`FactorInput::Invalid`] without scoring it.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use fugue::{factor, observe, pure, Address, Distribution, Model, SampleType};

use crate::fitness::traits::Fitness;

/// An observation program `p(data | genome)`, possibly tempered.
///
/// `beta` is the likelihood temperature. Implementations decide how it
/// enters: [`FactorFitness`] scales its factor by `β`; observation models
/// can use [`tempered_observe`] per datum (which reduces to a plain `observe`
/// at `β = 1`). Callers on the SMC path always pass `β = 1` — fugue's
/// adaptive tempering supplies β there, exactly once.
pub trait GenomeLikelihood<G>: Clone + Send + Sync + 'static {
    /// The observation program conditioned on `genome`, at likelihood
    /// temperature `beta`.
    fn model(&self, genome: &G, beta: f64) -> Model<()>;
}

/// A tempered observation: at `β = 1` this is exactly `observe(addr, dist,
/// value)` (the log-density lands in `log_likelihood`); at other `β` it is
/// `factor(β · log p(value))` (landing in `log_factors`). Both accumulators
/// are tempered together by fugue's SMC, so the two forms are interchangeable
/// under tempering — the `β = 1` form is preferred because it keeps the
/// likelihood/prior decomposition visible in the trace.
pub fn tempered_observe<T: SampleType>(
    addr: Address,
    dist: impl Distribution<T> + 'static,
    value: T,
    beta: f64,
) -> Model<()> {
    if beta == 1.0 {
        observe(addr, dist, value)
    } else {
        factor(beta * dist.log_prob(&value))
    }
}

/// The black-box adapter: a scalar [`Fitness`] entering as `factor(β·f(g))`.
///
/// The resulting target `π_β(x) ∝ p(x)·exp(β·f(x))` is the Gibbs /
/// generalized-Bayes posterior — the classical "fitness as likelihood"
/// correspondence, now explicitly one [`GenomeLikelihood`] among many.
#[derive(Clone, Debug)]
pub struct FactorFitness<F> {
    /// The wrapped scalar fitness.
    pub fitness: F,
}

impl<F> FactorFitness<F> {
    /// Wrap a scalar fitness as a factor likelihood.
    pub fn new(fitness: F) -> Self {
        Self { fitness }
    }
}

impl<G, F> GenomeLikelihood<G> for FactorFitness<F>
where
    F: Fitness<Genome = G, Value = f64> + Clone + Send + Sync + 'static,
    G: 'static,
{
    fn model(&self, genome: &G, beta: f64) -> Model<()> {
        factor(beta * self.fitness.evaluate(genome))
    }
}

/// A memoizing wrapper around an expensive [`Fitness`].
///
/// The inference layer re-evaluates fitness whenever a trace is replayed
/// (scoring, decode, rejuvenation); when fitness evaluation dominates — the
/// usual case in evolutionary computation — memoization removes almost all of
/// that overhead. Keys are the exact `bincode` serialization of the genome
/// (no hash collisions); the cache is shared across clones (`Arc`) so the
/// closures a model constructor spawns all hit the same table.
///
/// The cache grows without bound; for long runs over continuous genomes
/// (where exact repeats are rare outside replay) wrap only genuinely
/// expensive fitness functions.
#[derive(Clone)]
pub struct MemoizedFitness<F> {
    inner: F,
    cache: Arc<Mutex<HashMap<Vec<u8>, f64>>>,
}

impl<F> MemoizedFitness<F> {
    /// Wrap `fitness` with a shared memo table.
    pub fn new(fitness: F) -> Self {
        Self {
            inner: fitness,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Number of distinct genomes evaluated so far.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl<F> Fitness for MemoizedFitness<F>
where
    F: Fitness<Value = f64>,
    F::Genome: serde::Serialize,
{
    type Genome = F::Genome;
    type Value = f64;

    fn evaluate(&self, genome: &Self::Genome) -> f64 {
        let key = match bincode::serialize(genome) {
            Ok(k) => k,
            Err(_) => return self.inner.evaluate(genome), // unkeyable: pass through
        };
        if let Ok(cache) = self.cache.lock() {
            if let Some(&v) = cache.get(&key) {
                return v;
            }
        }
        let v = self.inner.evaluate(genome);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, v);
        }
        v
    }
}

/// What a [`DeterministicFactor`] makes of a genome before scoring it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorInput<K, D> {
    /// The genome decodes to `decoded`, which `key` identifies: two genomes
    /// with equal keys must have equal log-factors.
    Valid {
        /// The memoisation key, typically a hash of a canonical form of
        /// the decoded object.
        key: K,
        /// The decoded object handed to
        /// [`DeterministicFactor::log_factor`] on a cache miss.
        decoded: D,
    },
    /// The genome has no valid decoding (it does not compile, violates a
    /// hard constraint, …). Its likelihood is zero: it is never scored and
    /// never cached.
    Invalid,
}

/// A likelihood that is a **deterministic factor of the decoded genome**:
/// `p(data | g) ∝ exp(log_factor(decode(g)))`, with no latent sites and no
/// randomness. A closed-form marginal likelihood (log evidence) of the model
/// a genome describes is the typical case. Wrap it in [`MemoizedFactor`] to
/// use it as a [`GenomeLikelihood`].
///
/// The work is split in two so that the cheap part runs on every replay and
/// the expensive part once per distinct decoded object:
///
/// 1. [`decode`](Self::decode) turns the genome into the object the score
///    depends on, with a key for it, or declares the genome
///    [`FactorInput::Invalid`];
/// 2. [`log_factor`](Self::log_factor) scores a decoded object. It is called
///    at most once per key (per cache).
pub trait DeterministicFactor: Send + Sync + 'static {
    /// The genome type scored.
    type Genome;
    /// The memoisation key.
    type Key: Eq + Hash + Send + 'static;
    /// The decoded object the score is a function of.
    type Decoded;

    /// Decode `genome` and key the result, or declare it invalid.
    fn decode(&self, genome: &Self::Genome) -> FactorInput<Self::Key, Self::Decoded>;

    /// The untempered log-factor of a decoded object. `NaN` is treated as
    /// `−∞`.
    fn log_factor(&self, decoded: &Self::Decoded) -> f64;
}

/// Memoisation for likelihood programs: a [`GenomeLikelihood`] that enters
/// a [`DeterministicFactor`] as `factor(β · log_factor)`, evaluating
/// `log_factor` **once per key**.
///
/// It is to likelihoods what [`MemoizedFitness`] is to a scalar
/// [`Fitness`], with two differences. The key is supplied by the user, so
/// genomes that decode to the same object (two orderings of commuting
/// edits, say) share one evaluation, where `MemoizedFitness` keys on the
/// genome's exact serialization. And a genome can be declared
/// [`FactorInput::Invalid`], which enters as `factor(−∞)` at every `β`
/// without being scored.
///
/// The cache stores the untempered value, so one cache serves every
/// temperature; it is shared across clones and grows without bound. A
/// cached `−∞` (a valid object with zero likelihood) also enters as
/// `factor(−∞)` at every `β`, including `β = 0`.
///
/// ```rust
/// use fugue_evo::genome::bit_string::BitString;
/// use fugue_evo::genome::traits::BinaryGenome;
/// use fugue_evo::inference::likelihood::{DeterministicFactor, FactorInput, MemoizedFactor};
/// use fugue_evo::inference::model::EvolutionModel;
/// use fugue_evo::inference::prior::BitStringPrior;
///
/// /// Scores the *set size* of a bit string; the all-zero string is invalid.
/// struct BySize;
/// impl DeterministicFactor for BySize {
///     type Genome = BitString;
///     type Key = usize;
///     type Decoded = usize;
///     fn decode(&self, g: &BitString) -> FactorInput<usize, usize> {
///         match g.count_ones() {
///             0 => FactorInput::Invalid,
///             n => FactorInput::Valid { key: n, decoded: n },
///         }
///     }
///     fn log_factor(&self, size: &usize) -> f64 {
///         -(*size as f64) // stands in for an expensive evaluation
///     }
/// }
///
/// let likelihood = MemoizedFactor::new(BySize);
/// let model = EvolutionModel::from_likelihood(BitStringPrior::uniform(3), likelihood.clone());
/// let score = |bits: [bool; 3]| {
///     let (_g, trace) = model.score(&BitString::new(bits.to_vec())).unwrap();
///     trace.log_factors
/// };
///
/// assert_eq!(score([true, false, false]), -1.0);
/// assert_eq!(score([false, false, true]), -1.0); // same key: not re-scored
/// assert_eq!(score([true, true, false]), -2.0);
/// assert_eq!(score([false, false, false]), f64::NEG_INFINITY); // invalid
///
/// assert_eq!(likelihood.evaluations(), 2);
/// assert_eq!(likelihood.hits(), 1);
/// assert_eq!(likelihood.invalid(), 1);
/// assert_eq!(likelihood.cache_len(), 2);
/// ```
pub struct MemoizedFactor<S: DeterministicFactor> {
    inner: Arc<S>,
    cache: Arc<Mutex<HashMap<S::Key, f64>>>,
    counters: Arc<FactorCounters>,
}

#[derive(Default)]
struct FactorCounters {
    evaluations: AtomicUsize,
    hits: AtomicUsize,
    invalid: AtomicUsize,
}

impl<S: DeterministicFactor> Clone for MemoizedFactor<S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            cache: Arc::clone(&self.cache),
            counters: Arc::clone(&self.counters),
        }
    }
}

impl<S: DeterministicFactor> MemoizedFactor<S> {
    /// Wrap `factor` with a shared memo table.
    pub fn new(factor: S) -> Self {
        Self {
            inner: Arc::new(factor),
            cache: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(FactorCounters::default()),
        }
    }

    /// The wrapped factor.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Number of distinct keys scored so far.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Number of calls to [`DeterministicFactor::log_factor`] so far.
    pub fn evaluations(&self) -> usize {
        self.counters.evaluations.load(Ordering::Relaxed)
    }

    /// Number of lookups answered from the cache so far.
    pub fn hits(&self) -> usize {
        self.counters.hits.load(Ordering::Relaxed)
    }

    /// Number of genomes declared [`FactorInput::Invalid`] so far (counted
    /// per lookup, not per distinct genome).
    pub fn invalid(&self) -> usize {
        self.counters.invalid.load(Ordering::Relaxed)
    }

    /// The untempered log-factor of `genome`: `None` if it is invalid,
    /// otherwise the cached value, computing and caching it on a miss.
    pub fn log_factor(&self, genome: &S::Genome) -> Option<f64> {
        let (key, decoded) = match self.inner.decode(genome) {
            FactorInput::Invalid => {
                self.counters.invalid.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            FactorInput::Valid { key, decoded } => (key, decoded),
        };
        if let Ok(cache) = self.cache.lock() {
            if let Some(&v) = cache.get(&key) {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
                return Some(v);
            }
        }
        // Scored outside the lock: concurrent misses on one key may each
        // evaluate, and they agree because the factor is deterministic.
        let v = self.inner.log_factor(&decoded);
        let v = if v.is_nan() { f64::NEG_INFINITY } else { v };
        self.counters.evaluations.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, v);
        }
        Some(v)
    }
}

impl<S: DeterministicFactor> GenomeLikelihood<S::Genome> for MemoizedFactor<S>
where
    S::Genome: 'static,
{
    fn model(&self, genome: &S::Genome, beta: f64) -> Model<()> {
        match self.log_factor(genome) {
            // `β · (−∞)` is NaN at β = 0; a zero likelihood stays zero.
            None | Some(f64::NEG_INFINITY) => factor(f64::NEG_INFINITY),
            Some(v) => factor(beta * v),
        }
    }
}

/// Convenience: a `()`-like likelihood that conditions on nothing (the
/// posterior is the prior). Useful for testing priors through the inference
/// drivers.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoLikelihood;

impl<G: 'static> GenomeLikelihood<G> for NoLikelihood {
    fn model(&self, _genome: &G, _beta: f64) -> Model<()> {
        pure(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_memoized_fitness_evaluates_once_per_genome() {
        static CALLS: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        struct Counting;
        impl Fitness for Counting {
            type Genome = RealVector;
            type Value = f64;
            fn evaluate(&self, g: &RealVector) -> f64 {
                CALLS.fetch_add(1, Ordering::SeqCst);
                -g.genes().iter().map(|x| x * x).sum::<f64>()
            }
        }

        let memo = MemoizedFitness::new(Counting);
        let a = RealVector::new(vec![1.0, 2.0]);
        let b = RealVector::new(vec![3.0, 4.0]);
        let fa = memo.evaluate(&a);
        for _ in 0..10 {
            assert_eq!(memo.evaluate(&a), fa);
        }
        memo.evaluate(&b);
        memo.evaluate(&b);
        assert_eq!(
            CALLS.load(Ordering::SeqCst),
            2,
            "each genome evaluated once"
        );
        assert_eq!(memo.cache_len(), 2);

        // Clones share the cache.
        let clone = memo.clone();
        clone.evaluate(&a);
        assert_eq!(CALLS.load(Ordering::SeqCst), 2);
    }

    #[derive(Default)]
    struct Parity {
        decodes: AtomicUsize,
        scores: AtomicUsize,
    }

    /// Keys a real vector by its rounded first gene; negative first genes
    /// are invalid; a first gene of 7 scores NaN and 8 scores −∞.
    impl DeterministicFactor for Parity {
        type Genome = RealVector;
        type Key = i64;
        type Decoded = i64;

        fn decode(&self, g: &RealVector) -> FactorInput<i64, i64> {
            self.decodes.fetch_add(1, Ordering::SeqCst);
            let k = g.genes()[0].round() as i64;
            if k < 0 {
                FactorInput::Invalid
            } else {
                FactorInput::Valid { key: k, decoded: k }
            }
        }

        fn log_factor(&self, k: &i64) -> f64 {
            self.scores.fetch_add(1, Ordering::SeqCst);
            match k {
                7 => f64::NAN,
                8 => f64::NEG_INFINITY,
                k => -(*k as f64),
            }
        }
    }

    fn log_factors_of(model: Model<()>) -> f64 {
        use fugue::runtime::handler::run;
        use fugue::runtime::interpreters::PriorHandler;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (_, t) = run(
            PriorHandler {
                rng: &mut rng,
                trace: fugue::Trace::default(),
            },
            model,
        );
        t.log_factors
    }

    #[test]
    fn test_memoized_factor_scores_once_per_key() {
        let memo = MemoizedFactor::new(Parity::default());
        let a = RealVector::new(vec![3.1, 0.0]);
        let b = RealVector::new(vec![2.9, 5.0]); // a different genome, same key
        assert_eq!(memo.log_factor(&a), Some(-3.0));
        assert_eq!(memo.log_factor(&b), Some(-3.0));
        let clone = memo.clone();
        assert_eq!(clone.log_factor(&a), Some(-3.0)); // clones share the cache
        assert_eq!(memo.inner().scores.load(Ordering::SeqCst), 1);
        assert_eq!(memo.inner().decodes.load(Ordering::SeqCst), 3);
        assert_eq!(
            (memo.evaluations(), memo.hits(), memo.cache_len()),
            (1, 2, 1)
        );
    }

    #[test]
    fn test_memoized_factor_invalid_is_never_scored_or_cached() {
        let memo = MemoizedFactor::new(Parity::default());
        let bad = RealVector::new(vec![-1.0]);
        for _ in 0..3 {
            assert_eq!(memo.log_factor(&bad), None);
        }
        assert_eq!(memo.invalid(), 3);
        assert_eq!((memo.evaluations(), memo.cache_len()), (0, 0));
        // Zero likelihood at every temperature, including β = 0.
        for beta in [0.0, 0.5, 1.0, 3.0] {
            assert_eq!(log_factors_of(memo.model(&bad, beta)), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_memoized_factor_tempers_the_cached_value() {
        let memo = MemoizedFactor::new(Parity::default());
        let g = RealVector::new(vec![4.0]);
        for beta in [0.0, 0.25, 1.0, 2.0] {
            assert_eq!(log_factors_of(memo.model(&g, beta)), beta * -4.0);
        }
        assert_eq!(memo.evaluations(), 1, "one cache serves every β");
        // NaN and −∞ scores are a zero likelihood at every β, cached.
        for first in [7.0, 8.0] {
            let g = RealVector::new(vec![first]);
            for beta in [0.0, 1.0] {
                assert_eq!(log_factors_of(memo.model(&g, beta)), f64::NEG_INFINITY);
            }
        }
        assert_eq!(memo.evaluations(), 3);
    }

    #[test]
    fn test_tempered_observe_matches_observe_at_beta_one() {
        use fugue::runtime::handler::run;
        use fugue::runtime::interpreters::PriorHandler;
        use fugue::{addr, Normal, Trace};
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(1);
        let dist = Normal::new(0.0, 1.0).unwrap();
        let (_, t1) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            tempered_observe(addr!("y"), dist, 0.7, 1.0),
        );
        let (_, t2) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            tempered_observe(addr!("y"), dist, 0.7, 0.5),
        );
        let lp = fugue::Distribution::log_prob(&dist, &0.7);
        assert!((t1.log_likelihood - lp).abs() < 1e-12);
        assert_eq!(t1.log_factors, 0.0);
        assert!((t2.log_factors - 0.5 * lp).abs() < 1e-12);
        assert_eq!(t2.log_likelihood, 0.0);
        // Under tempering both contribute identically at any β:
        // β·(log_likelihood + log_factors) is the same either way.
    }
}
