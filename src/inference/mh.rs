//! Metropolis–Hastings over the Boltzmann target, delegated to fugue
//!
//! The old `EvolutionStep` hand-rolled its proposal (and only ever perturbed
//! `F64` choices, so BitString/Permutation chains silently never moved). This
//! wrapper deletes all of that: one transition is one call into
//! [`fugue::adaptive_single_site_mh_cached`], which picks the target site
//! uniformly over **all** sites and dispatches the proposal by value type
//! (for `F64` from the site's [`fugue::Support`]: Gaussian walk on the reals,
//! log-space walk on the positives, reflected walk on a bounded interval;
//! flip for `Bool`, reflected discrete walk for `U64`, prior-resample for
//! `Usize`, integer walk for `I64`), including the reversible-jump
//! corrections for structure-changing models. Per-address
//! [`SiteProposal`] overrides registered with
//! [`EvolutionChain::override_site`] are honoured by every entry point —
//! [`EvolutionChain::step`], [`EvolutionChain::step_scored`] and
//! [`EvolutionChain::run_chain`] alike.
//!
//! A transition costs **one** model execution (the proposal): the current
//! state's log-density and per-site densities are read from the scored trace
//! the caller holds, which is why `current` must be a trace produced by
//! [`EvolutionChain::init`], [`EvolutionChain::init_from`] or a previous
//! `step` (see [`EvolutionChain::step`]).

use std::collections::HashMap;

use fugue::inference::mcmc_utils::DiminishingAdaptation;
use fugue::runtime::handler::run;
use fugue::runtime::interpreters::{PriorHandler, ScoreGivenTrace};
use fugue::{
    adaptive_mcmc_chain_with_overrides, adaptive_single_site_mh_cached, Address, SiteProposal,
    Trace,
};
use rand::Rng;

use super::likelihood::GenomeLikelihood;
use super::model::EvolutionModel;
use super::prior::GenomePrior;
use crate::error::GenomeError;

fn finite_state(trace: Trace) -> Result<Trace, GenomeError> {
    if trace.total_log_weight().is_finite() {
        Ok(trace)
    } else {
        Err(GenomeError::ConstraintViolation(
            "genome is outside the prior's support (target log-density is not finite)".to_string(),
        ))
    }
}

/// An MH chain over the fixed-β Boltzmann target `π_β ∝ p(x)·exp(β·f(x))`.
pub struct EvolutionChain<P, L>
where
    P: GenomePrior,
    L: GenomeLikelihood<P::Genome>,
{
    model: EvolutionModel<P, L>,
    adaptation: DiminishingAdaptation,
    overrides: HashMap<Address, SiteProposal>,
}

impl<P, L> EvolutionChain<P, L>
where
    P: GenomePrior,
    L: GenomeLikelihood<P::Genome>,
{
    /// Create a chain over the model's fixed-β target.
    pub fn new(model: EvolutionModel<P, L>) -> Self {
        Self {
            model,
            adaptation: DiminishingAdaptation::new(0.44, 0.7),
            overrides: HashMap::new(),
        }
    }

    /// Set the adaptation's target acceptance rate (default 0.44).
    pub fn target_rate(mut self, rate: f64) -> Self {
        self.adaptation = DiminishingAdaptation::new(rate, 0.7);
        self
    }

    /// Force a specific `f64` proposal for one address (e.g.
    /// `SiteProposal::Reflect { lower, upper }` for a bounded coordinate, or
    /// `SiteProposal::PriorResample` for an independence move).
    ///
    /// Honoured by [`Self::step`], [`Self::step_scored`] and
    /// [`Self::run_chain`]. Since fugue selects the default `f64` proposal
    /// from the site's declared [`fugue::Support`] — a `Uniform` site already
    /// gets a reflected walk at its own bounds — an override is only needed
    /// to *change* that default (a narrower reflection interval, a log-space
    /// walk on a `Normal` site known to be positive, …).
    pub fn override_site(mut self, addr: Address, proposal: SiteProposal) -> Self {
        self.overrides.insert(addr, proposal);
        self
    }

    /// The registered per-address proposal overrides.
    pub fn overrides(&self) -> &HashMap<Address, SiteProposal> {
        &self.overrides
    }

    /// The underlying model.
    pub fn model(&self) -> &EvolutionModel<P, L> {
        &self.model
    }

    /// Draw an initial state: a prior sample's fully-scored trace (latent
    /// likelihood sites included).
    pub fn init<R: Rng>(&self, rng: &mut R) -> Trace {
        let (_g, trace) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            (self.model.target_model())(),
        );
        trace
    }

    /// Warm-start the chain from a given genome: encode it under the model's
    /// prior ([`GenomePrior::trace_of`]) and score it through the target
    /// program. Works for any prior — including grammar priors over trees —
    /// so a classic GA/GP result can seed an inference chain. Returns `None`
    /// if the genome is outside the prior's support (its target density is
    /// `−∞`, which can never be left by an MH chain) **or** cannot be scored
    /// from its encoding alone: wrong dimension for the prior, or a likelihood
    /// with latent nuisance sites (see [`Self::try_init_from`] for the reason
    /// and [`Self::init_from_with_latents`] to draw them). Never panics
    /// (EV-N3).
    pub fn init_from(&self, genome: &P::Genome) -> Option<Trace> {
        self.try_init_from(genome).ok()
    }

    /// [`Self::init_from`] with the reason on failure:
    /// [`EvolutionModel::score`]'s errors for a structural mismatch, or
    /// [`GenomeError::ConstraintViolation`] for a genome outside the prior's
    /// support.
    pub fn try_init_from(&self, genome: &P::Genome) -> Result<Trace, GenomeError> {
        let (_g, trace) = self.model.score(genome)?;
        finite_state(trace)
    }

    /// Warm-start from a genome when the likelihood has **latent nuisance
    /// sites** (an inferred noise scale, a Pareto weight): the genome's sites
    /// come from its encoding, the latent ones are drawn from their priors
    /// with `rng`, and the result is a complete, fully scored state. Same
    /// errors as [`Self::try_init_from`].
    pub fn init_from_with_latents<R: Rng>(
        &self,
        rng: &mut R,
        genome: &P::Genome,
    ) -> Result<Trace, GenomeError> {
        let (_g, trace) = self.model.score_with_latents(rng, genome)?;
        finite_state(trace)
    }

    /// One π_β-invariant transition. Moves ANY site type; honours
    /// [`Self::override_site`]. Returns the decoded genome and the new state
    /// (the freshly scored proposal on acceptance, a copy of `current` on
    /// rejection).
    ///
    /// Costs exactly one model execution — the proposal — plus, on rejection,
    /// a replay of the **prior program only** (no likelihood / fitness
    /// evaluation) to decode the genome of the unchanged state. Callers who
    /// keep their own decoded genome can use [`Self::step_scored`] and skip
    /// even that.
    ///
    /// # Contract on `current`
    ///
    /// `current` must be a fully scored trace of this chain's target: one
    /// returned by [`Self::init`], [`Self::init_from`] /
    /// [`Self::init_from_with_latents`], or a previous `step` /
    /// `step_scored`. Its accumulators and per-site densities are trusted as
    /// the current state's log-density and as the reverse-move densities of
    /// sites a proposal makes vanish. A trace assembled by hand —
    /// [`TraceGenome::to_trace`](crate::genome::trace_genome::TraceGenome::to_trace)
    /// or [`GenomePrior::trace_of`], whose per-site `logp` is 0 — violates
    /// this and over-accepts structure-shrinking moves until the first
    /// acceptance; route it through `init_from` first.
    pub fn step<R: Rng>(&mut self, rng: &mut R, current: &Trace) -> (P::Genome, Trace) {
        match self.step_scored(rng, current) {
            Some((g, t, _log_weight)) => (g, t),
            None => (self.decode(current), current.clone()),
        }
    }

    /// One π_β-invariant transition from a scored state, at the cost of a
    /// single model execution: `Some((genome, scored_trace, log_weight))` on
    /// acceptance — `log_weight == scored_trace.total_log_weight()` — or
    /// `None` on rejection, in which case the caller keeps `current`. Same
    /// contract on `current` as [`Self::step`].
    pub fn step_scored<R: Rng>(
        &mut self,
        rng: &mut R,
        current: &Trace,
    ) -> Option<(P::Genome, Trace, f64)> {
        adaptive_single_site_mh_cached(
            rng,
            self.model.target_model(),
            current,
            &mut self.adaptation,
            &self.overrides,
            true,
        )
    }

    /// Decode the genome of a chain state by replaying the **prior program**
    /// over it (the prior's return value *is* the decoded genome). No
    /// likelihood or fitness is evaluated. `state` must be a complete
    /// assignment for the prior — every trace this chain hands out is.
    pub fn decode(&self, state: &Trace) -> P::Genome {
        let (g, _) = run(
            ScoreGivenTrace {
                base: state.clone(),
                trace: Trace::default(),
            },
            self.model.prior().model(),
        );
        g
    }

    /// Full warmup-then-frozen chain: `warmup` adaptive iterations are
    /// discarded, then `n` samples are collected from the frozen kernel.
    /// Returns decoded genomes with their traces.
    pub fn run_chain<R: Rng>(&self, rng: &mut R, n: usize, warmup: usize) -> Vec<(P::Genome, Trace)>
    where
        P::Genome: Clone,
    {
        adaptive_mcmc_chain_with_overrides(
            rng,
            self.model.target_model(),
            n,
            warmup,
            &self.overrides,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::traits::Fitness;
    use crate::genome::bounds::{Bounds, MultiBounds};
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::{BinaryGenome, PermutationGenome, RealValuedGenome};
    use crate::inference::model::tests::PtrFitness;
    use crate::inference::prior::{BitStringPrior, PermutationPrior, UniformBoxPrior};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn linear_x0(g: &RealVector) -> f64 {
        g.genes()[0]
    }

    /// Regression: EV-90 — no MH sample may escape the uniform-prior bounds,
    /// and the boundary is not over-weighted: on [-2, 2] with f(x) = x the
    /// β=1 Boltzmann posterior is ∝ e^x truncated to [-2, 2], with analytic
    /// mean (e² + 3e⁻²)/(e² − e⁻²) ≈ 1.0746. Re-driven through the fugue
    /// kernel instead of the deleted hand-rolled one.
    #[test]
    fn test_mh_respects_bounds() {
        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-2.0, 2.0)]));
        let model = EvolutionModel::new(prior, PtrFitness(linear_x0)).with_beta(1.0);
        let mut chain = EvolutionChain::new(model);

        let mut rng = StdRng::seed_from_u64(20260710);
        let mut current = chain.init(&mut rng);
        let mut samples = Vec::new();
        for i in 0..40_000 {
            let (g, t) = chain.step(&mut rng, &current);
            current = t;
            let x = g.genes()[0];
            assert!((-2.0..=2.0).contains(&x), "MH sample escaped bounds: {}", x);
            if i >= 5_000 {
                samples.push(x);
            }
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let analytic = {
            let e2 = 2.0_f64.exp();
            let em2 = (-2.0_f64).exp();
            (e2 + 3.0 * em2) / (e2 - em2)
        };
        assert!(
            (mean - analytic).abs() < 0.1,
            "posterior mean {} deviates from truncated-exponential analytic {}",
            mean,
            analytic
        );
    }

    /// Regression: FG-N1 downstream — the same truncated-exponential anchor on
    /// `[-0.5, 0.5]`. Under the pre-fix fugue proposal selector a `Uniform`
    /// site whose support excludes `-1` but contains negatives was put on a
    /// log-space walk whenever its first draw was positive, so the chain
    /// inherited the sign of its initial state and could never cross zero
    /// (posterior mean ≈ +0.27 or ≈ −0.23 instead of the analytic
    /// `a·coth(a) − 1 = 0.0820` for `ρ ∝ eˣ` on `[−a, a]`, `a = 1/2`). fugue
    /// now selects the proposal from `Distribution::support()` (Reflect for a
    /// bounded site), so the chain driven through `EvolutionChain::step`
    /// visits both signs with the analytic mass `P(x > 0) = 0.6225`.
    #[test]
    fn test_mh_bounded_prior_containing_negatives_mixes_across_zero() {
        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-0.5, 0.5)]));
        let model = EvolutionModel::new(prior, PtrFitness(linear_x0)).with_beta(1.0);
        let analytic_mean = 0.5 / (0.5f64).tanh() - 1.0;
        let analytic_p_pos = (0.5f64.exp() - 1.0) / (0.5f64.exp() - (-0.5f64).exp());
        for seed in [1u64, 2, 3, 20260710] {
            let mut chain = EvolutionChain::new(model.clone());
            let mut rng = StdRng::seed_from_u64(seed);
            let mut current = chain.init(&mut rng);
            let mut samples = Vec::new();
            for i in 0..40_000 {
                let (g, t) = chain.step(&mut rng, &current);
                current = t;
                let x = g.genes()[0];
                assert!((-0.5..=0.5).contains(&x), "MH sample escaped bounds: {}", x);
                if i >= 5_000 {
                    samples.push(x);
                }
            }
            let n = samples.len() as f64;
            let mean = samples.iter().sum::<f64>() / n;
            let p_pos = samples.iter().filter(|&&x| x > 0.0).count() as f64 / n;
            assert!(
                (mean - analytic_mean).abs() < 0.04,
                "seed {seed}: posterior mean {mean} deviates from analytic {analytic_mean}"
            );
            assert!(
                (p_pos - analytic_p_pos).abs() < 0.08,
                "seed {seed}: P(x > 0) = {p_pos} vs analytic {analytic_p_pos} — chain stuck on one sign"
            );
        }
    }

    /// EV-N2 / X-5(a): `step` honours `override_site`. A `Reflect` override
    /// narrower than the prior box confines a chain started inside it — the
    /// reflected walk can never propose outside `[lower, upper]` — whereas
    /// the same chain without the override wanders over the whole box.
    #[test]
    fn test_step_honours_override_site() {
        let prior = || UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-2.0, 2.0)]));
        let start = RealVector::new(vec![0.0]);

        let mut confined = EvolutionChain::new(EvolutionModel::new(prior(), PtrFitness(linear_x0)))
            .override_site(
                fugue::addr!("gene", 0),
                SiteProposal::Reflect {
                    lower: -0.5,
                    upper: 0.5,
                },
            );
        let mut rng = StdRng::seed_from_u64(3);
        let mut current = confined.init_from(&start).expect("in support");
        let mut accepted = 0;
        for _ in 0..5_000 {
            if let Some((g, t, _)) = confined.step_scored(&mut rng, &current) {
                accepted += 1;
                current = t;
                let x = g.genes()[0];
                assert!(
                    (-0.5..=0.5).contains(&x),
                    "override ignored: reflected chain left [-0.5, 0.5] at {x}"
                );
            }
        }
        assert!(
            accepted > 500,
            "confined chain barely moved ({accepted} acceptances)"
        );

        let mut free = EvolutionChain::new(EvolutionModel::new(prior(), PtrFitness(linear_x0)));
        let mut rng = StdRng::seed_from_u64(3);
        let mut current = free.init_from(&start).expect("in support");
        let mut escaped = false;
        for _ in 0..5_000 {
            let (g, t) = free.step(&mut rng, &current);
            current = t;
            if g.genes()[0].abs() > 0.5 {
                escaped = true;
                break;
            }
        }
        assert!(
            escaped,
            "without the override the chain must explore the whole box"
        );
    }

    /// EV-N2 / X-5(b): a transition costs one model execution. `init_from`
    /// evaluates the fitness once (the scoring replay); each `step` evaluates
    /// it exactly once more (the proposal), whether accepted or rejected —
    /// the rejected path decodes the genome from the prior program alone.
    #[test]
    fn test_step_costs_one_fitness_evaluation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Clone)]
        struct Counting(Arc<AtomicUsize>);
        impl Fitness for Counting {
            type Genome = RealVector;
            type Value = f64;
            fn evaluate(&self, g: &RealVector) -> f64 {
                self.0.fetch_add(1, Ordering::SeqCst);
                -0.5 * g.genes()[0].powi(2)
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-3.0, 3.0)]));
        let mut chain = EvolutionChain::new(EvolutionModel::new(prior, Counting(counter.clone())));
        let mut rng = StdRng::seed_from_u64(5);
        let mut current = chain
            .init_from(&RealVector::new(vec![0.3]))
            .expect("in support");
        assert_eq!(counter.load(Ordering::SeqCst), 1, "init_from scores once");

        let n = 2_000;
        let mut rejections = 0;
        for _ in 0..n {
            let before = counter.load(Ordering::SeqCst);
            let (g, t) = chain.step(&mut rng, &current);
            assert_eq!(
                counter.load(Ordering::SeqCst) - before,
                1,
                "a step must evaluate the fitness exactly once"
            );
            let x = t.get_f64(&fugue::addr!("gene", 0)).unwrap();
            if x == current.get_f64(&fugue::addr!("gene", 0)).unwrap() {
                rejections += 1;
            }
            assert_eq!(g.genes()[0], x);
            current = t;
        }
        assert!(
            rejections > 0,
            "some proposals must be rejected for the test to bite"
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1 + n);
    }

    /// New regression (dead-chain fix): a BitString chain must actually move.
    /// The old `EvolutionStep::propose` cloned every non-F64 choice unchanged,
    /// making this exact scenario a silent no-op forever.
    #[test]
    fn test_bitstring_chain_moves() {
        #[derive(Clone, Copy)]
        struct OnesCount;
        impl Fitness for OnesCount {
            type Genome = crate::genome::bit_string::BitString;
            type Value = f64;
            fn evaluate(&self, g: &Self::Genome) -> f64 {
                g.bits().iter().filter(|&&b| b).count() as f64
            }
        }

        let model = EvolutionModel::new(BitStringPrior::uniform(8), OnesCount).with_beta(1.0);
        let mut chain = EvolutionChain::new(model);
        let mut rng = StdRng::seed_from_u64(11);
        let init = chain.init(&mut rng);
        let init_bits: Vec<Option<bool>> = (0..8)
            .map(|i| init.get_bool(&fugue::addr!("bit", i)))
            .collect();

        let mut current = init.clone();
        let mut moved = false;
        for _ in 0..200 {
            let (_g, t) = chain.step(&mut rng, &current);
            current = t;
            let bits: Vec<Option<bool>> = (0..8)
                .map(|i| current.get_bool(&fugue::addr!("bit", i)))
                .collect();
            if bits != init_bits {
                moved = true;
                break;
            }
        }
        assert!(moved, "BitString chain never moved (dead-chain regression)");
    }

    /// New regression (dead-chain fix): a Permutation chain must move AND stay
    /// inside the permutation support (the sequential categorical prior gives
    /// colliding proposals probability zero).
    #[test]
    fn test_permutation_chain_moves() {
        #[derive(Clone, Copy)]
        struct SortedNess;
        impl Fitness for SortedNess {
            type Genome = crate::genome::permutation::Permutation;
            type Value = f64;
            fn evaluate(&self, g: &Self::Genome) -> f64 {
                // Rewards ascending order.
                g.permutation().windows(2).filter(|w| w[0] < w[1]).count() as f64
            }
        }

        let model = EvolutionModel::new(PermutationPrior::new(5), SortedNess).with_beta(1.0);
        let mut chain = EvolutionChain::new(model);
        let mut rng = StdRng::seed_from_u64(17);
        let init = chain.init(&mut rng);
        let read_perm = |t: &Trace| -> Vec<usize> {
            (0..5)
                .map(|i| t.get_usize(&fugue::addr!("perm", i)).unwrap())
                .collect()
        };
        let init_perm = read_perm(&init);

        let mut current = init;
        let mut moved = false;
        for _ in 0..500 {
            let (g, t) = chain.step(&mut rng, &current);
            current = t;
            assert!(
                g.is_valid_permutation(),
                "chain left the permutation support: {:?}",
                g.permutation()
            );
            if read_perm(&current) != init_perm {
                moved = true;
            }
        }
        assert!(
            moved,
            "Permutation chain never moved (dead-chain regression)"
        );
    }
}
