//! Priors over genomes as probabilistic programs
//!
//! A [`GenomePrior`] is *the* load-bearing abstraction of the inference layer:
//! instead of a closed enum of built-in priors, the prior over genomes is an
//! arbitrary fugue [`Model`] written by the user (or one of the constructors
//! below). Running it under a `PriorHandler` both draws `p(x)` and accumulates
//! `log_prior`; scoring an existing genome's trace against it recovers the
//! genuine prior density — there is no hand-written density code anywhere in
//! this layer.
//!
//! The model returns the **decoded genome** `G`, not a bare vector: the model's
//! return value *is* the decode, which is what lets the SMC layer recover a
//! genome from a bare particle trace by replay (see
//! [`crate::inference::smc::EvolutionPosterior`]).

use fugue::{addr, plate, sample, Bernoulli, Categorical, Model, ModelExt, Normal, Uniform};

use crate::genome::bit_string::BitString;
use crate::genome::bounds::MultiBounds;
use crate::genome::permutation::Permutation;
use crate::genome::real_vector::RealVector;
use crate::genome::trace_genome::TraceGenome;
use crate::genome::traits::{BinaryGenome, PermutationGenome, RealValuedGenome};

/// A prior distribution over genomes, expressed as a probabilistic program.
///
/// The program must sample the genome's canonical trace sites (the same
/// addresses [`TraceGenome::to_trace`] writes — `gene#i`, `bit#i`, `perm#i`,
/// …) and return the assembled genome. Anything expressible as a fugue
/// `Model` is a valid prior: correlated coordinates, hierarchical scales,
/// variable-length genomes (a length site followed by that many coordinate
/// sites — fugue's MH treats the resulting births/deaths as reversible-jump
/// moves with no extra code here).
pub trait GenomePrior: Clone + Send + Sync + 'static {
    /// The genome type this prior generates.
    type Genome: TraceGenome;

    /// The generative program `p(x)`.
    fn model(&self) -> Model<Self::Genome>;

    /// Encode a genome as a trace **under this prior's address scheme** — the
    /// inverse direction of running [`Self::model`]. The default delegates to
    /// the genome's canonical [`TraceGenome::to_trace`] encoding, which is
    /// correct whenever the prior samples exactly the canonical sites (all the
    /// vector priors here). Priors with their own generative scheme — e.g.
    /// [`ArithmeticGrammarPrior`](super::grammar::ArithmeticGrammarPrior)'s
    /// tree-path grammar — override this so that scoring, weighted traces, and
    /// chain warm-starts work for any genome the prior can express.
    fn trace_of(&self, genome: &Self::Genome) -> fugue::Trace {
        genome.to_trace()
    }
}

/// Independent uniform prior over a bounded box (per-dimension `[min, max]`).
///
/// The support behavior formerly hand-coded in the old `Prior::UniformBounds`
/// match (`−∞` outside the box) now falls out of scoring `Uniform` sites under
/// replay: an out-of-bounds value has `log_prob = −∞` and any MH move onto it
/// is rejected.
#[derive(Clone, Debug)]
pub struct UniformBoxPrior {
    bounds: MultiBounds,
}

impl UniformBoxPrior {
    /// Uniform prior over the given per-dimension bounds.
    pub fn new(bounds: MultiBounds) -> Self {
        Self { bounds }
    }

    /// The bounds of the box.
    pub fn bounds(&self) -> &MultiBounds {
        &self.bounds
    }
}

impl GenomePrior for UniformBoxPrior {
    type Genome = RealVector;

    fn model(&self) -> Model<RealVector> {
        let bounds = self.bounds.clone();
        plate!(i in 0..bounds.dimension().max(1) => {
            let (lo, hi) = match bounds.get(i) {
                Some(b) if b.max > b.min => (b.min, b.max),
                Some(b) => (b.min - 1e-9, b.min + 1e-9),
                None => (-1.0, 1.0),
            };
            sample(addr!("gene", i), Uniform::new(lo, hi).expect("valid uniform prior bounds"))
        })
        .map(|genes| RealVector::from_genes(genes).expect("plate produced genes"))
    }
}

/// Independent Gaussian `N(mean, std²)` prior on every real coordinate.
#[derive(Clone, Debug)]
pub struct GaussianPrior {
    mean: f64,
    std: f64,
    dim: usize,
}

impl GaussianPrior {
    /// I.i.d. Gaussian prior with the given per-coordinate mean/std over `dim`
    /// coordinates. `std` must be positive.
    pub fn new(mean: f64, std: f64, dim: usize) -> Self {
        assert!(std > 0.0, "Gaussian prior std must be > 0");
        Self { mean, std, dim }
    }
}

impl GenomePrior for GaussianPrior {
    type Genome = RealVector;

    fn model(&self) -> Model<RealVector> {
        let (mean, std, dim) = (self.mean, self.std, self.dim.max(1));
        plate!(i in 0..dim => {
            sample(addr!("gene", i), Normal::new(mean, std).expect("valid Gaussian prior"))
        })
        .map(|genes| RealVector::from_genes(genes).expect("plate produced genes"))
    }
}

/// Independent `Bernoulli(p)` prior on every bit of a [`BitString`].
#[derive(Clone, Debug)]
pub struct BitStringPrior {
    p_one: f64,
    len: usize,
}

impl BitStringPrior {
    /// Prior over `len`-bit strings with per-bit probability `p_one` of a set
    /// bit. `p_one` must lie in `(0, 1)` so every string has support.
    pub fn new(p_one: f64, len: usize) -> Self {
        assert!(
            p_one > 0.0 && p_one < 1.0,
            "BitStringPrior p_one must be in (0, 1)"
        );
        Self { p_one, len }
    }

    /// Uniform prior over `len`-bit strings (`p = 1/2` per bit).
    pub fn uniform(len: usize) -> Self {
        Self::new(0.5, len)
    }
}

impl GenomePrior for BitStringPrior {
    type Genome = BitString;

    fn model(&self) -> Model<BitString> {
        let (p, len) = (self.p_one, self.len.max(1));
        plate!(i in 0..len => {
            sample(addr!("bit", i), Bernoulli::new(p).expect("valid Bernoulli prior"))
        })
        .map(|bits| BitString::from_bits(bits).expect("plate produced bits"))
    }
}

/// Fisher–Yates / Lehmer-code uniform prior over permutations of `0..n`.
///
/// Position `i` samples a **rank** at `perm#i`, uniform over the `n−i` values
/// not yet used (a `Usize` in `0..n−i`); the rank sequence decodes to a
/// permutation against the shrinking available-value list. This coincides
/// site-for-site with [`Permutation::to_trace`]'s Lehmer encoding, so scoring
/// an existing genome's trace under replay finds every site with matching
/// semantics, and:
///
/// - every model execution decodes to a valid permutation (density exactly
///   `1/n!`), and
/// - under single-site MH, resampling one rank always decodes to a *different
///   valid* permutation — the rank encoding is what makes single-site moves
///   live (a raw value encoding would turn every single-site change into a
///   duplicate and freeze the chain).
#[derive(Clone, Debug)]
pub struct PermutationPrior {
    n: usize,
}

impl PermutationPrior {
    /// Uniform prior over permutations of `0..n`.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "PermutationPrior needs n > 0");
        Self { n }
    }
}

impl GenomePrior for PermutationPrior {
    type Genome = Permutation;

    fn model(&self) -> Model<Permutation> {
        let n = self.n;
        fn rank_model(n: usize, i: usize, ranks: Vec<usize>) -> Model<Vec<usize>> {
            if i == n {
                return fugue::pure(ranks);
            }
            let k = n - i;
            let probs = vec![1.0 / k as f64; k];
            sample(
                addr!("perm", i),
                Categorical::new(probs).expect("valid categorical prior"),
            )
            .bind(move |r| {
                let mut ranks = ranks;
                ranks.push(r);
                rank_model(n, i + 1, ranks)
            })
        }
        rank_model(n, 0, Vec::with_capacity(n)).map(move |ranks| {
            let mut available: Vec<usize> = (0..n).collect();
            let perm: Vec<usize> = ranks.into_iter().map(|r| available.remove(r)).collect();
            Permutation::from_permutation(perm).expect("Lehmer decode produced a permutation")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fugue::runtime::handler::run;
    use fugue::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
    use fugue::Trace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gaussian_prior_draws_match_moments() {
        let prior = GaussianPrior::new(0.0, 2.0, 1);
        let mut rng = StdRng::seed_from_u64(99);
        let xs: Vec<f64> = (0..5000)
            .map(|_| {
                let (g, _) = run(
                    PriorHandler {
                        rng: &mut rng,
                        trace: Trace::default(),
                    },
                    prior.model(),
                );
                g.genes()[0]
            })
            .collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.2, "prior mean {}", mean);
        assert!((var.sqrt() - 2.0).abs() < 0.2, "prior std {}", var.sqrt());
    }

    /// Replacement anchor for the deleted hand-written `log_prior_density`:
    /// scoring a genome's trace under the prior model reproduces the analytic
    /// Gaussian log-density.
    #[test]
    fn test_prior_model_log_prior_matches_analytic() {
        let prior = GaussianPrior::new(1.0, 2.0, 3);
        let g = RealVector::new(vec![0.5, 1.5, -2.0]);
        let (_, scored) = run(
            ScoreGivenTrace {
                base: g.to_trace(),
                trace: Trace::default(),
            },
            prior.model(),
        );
        let normal = Normal::new(1.0, 2.0).unwrap();
        let analytic: f64 = g
            .genes()
            .iter()
            .map(|x| fugue::Distribution::log_prob(&normal, x))
            .sum();
        assert!((scored.log_prior - analytic).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_prior_out_of_box_scores_neg_inf() {
        let prior = UniformBoxPrior::new(MultiBounds::symmetric(1.0, 2));
        let g = RealVector::new(vec![0.5, 5.0]); // second coordinate outside
        let (_, scored) = run(
            ScoreGivenTrace {
                base: g.to_trace(),
                trace: Trace::default(),
            },
            prior.model(),
        );
        assert_eq!(scored.log_prior, f64::NEG_INFINITY);
    }

    #[test]
    fn test_permutation_prior_generates_valid_permutations() {
        let prior = PermutationPrior::new(6);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..50 {
            let (p, trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            assert!(p.is_valid_permutation());
            // Density of any permutation is 1/n!.
            let expected = -(720.0f64).ln(); // ln(1/6!)
            assert!((trace.log_prior - expected).abs() < 1e-9);
            // The trace encoding coincides with Permutation::to_trace.
            let canonical = p.to_trace();
            for (addr, choice) in &canonical.choices {
                assert_eq!(trace.choices[addr].value, choice.value);
            }
        }
    }

    #[test]
    fn test_bitstring_prior_scores_canonical_trace() {
        let prior = BitStringPrior::uniform(4);
        let g = BitString::from_bits(vec![true, false, true, true]).unwrap();
        let (decoded, scored) = run(
            ScoreGivenTrace {
                base: g.to_trace(),
                trace: Trace::default(),
            },
            prior.model(),
        );
        assert_eq!(decoded.bits(), g.bits());
        assert!((scored.log_prior - 4.0 * (0.5f64).ln()).abs() < 1e-12);
    }
}
