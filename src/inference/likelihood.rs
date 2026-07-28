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

use std::collections::HashMap;
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
