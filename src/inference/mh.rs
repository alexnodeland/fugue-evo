//! Metropolis–Hastings over the Boltzmann target, delegated to fugue
//!
//! The old `EvolutionStep` hand-rolled its proposal (and only ever perturbed
//! `F64` choices, so BitString/Permutation chains silently never moved). This
//! wrapper deletes all of that: one transition is one call into
//! [`fugue::adaptive_single_site_mh`], which picks the target site uniformly
//! over **all** sites and dispatches the proposal by value type (Gaussian /
//! log-space walk for `F64`, flip for `Bool`, reflected discrete walk for
//! `U64`, prior-resample for `Usize`, integer walk for `I64`), including the
//! reversible-jump corrections for structure-changing models.

use std::collections::HashMap;

use fugue::inference::mcmc_utils::DiminishingAdaptation;
use fugue::{
    adaptive_mcmc_chain_with_overrides, adaptive_single_site_mh, Address, SiteProposal, Trace,
};
use rand::Rng;

use super::model::EvolutionModel;
use super::prior::GenomePrior;
use crate::fitness::traits::Fitness;

/// An MH chain over the fixed-β Boltzmann target `π_β ∝ p(x)·exp(β·f(x))`.
pub struct EvolutionChain<P, F>
where
    P: GenomePrior,
    F: Fitness<Genome = P::Genome, Value = f64> + Clone + Send + Sync + 'static,
{
    model: EvolutionModel<P, F>,
    adaptation: DiminishingAdaptation,
    overrides: HashMap<Address, SiteProposal>,
}

impl<P, F> EvolutionChain<P, F>
where
    P: GenomePrior,
    F: Fitness<Genome = P::Genome, Value = f64> + Clone + Send + Sync + 'static,
{
    /// Create a chain over the model's fixed-β target.
    pub fn new(model: EvolutionModel<P, F>) -> Self {
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
    /// `SiteProposal::Reflect { lower, upper }` for a bounded coordinate).
    pub fn override_site(mut self, addr: Address, proposal: SiteProposal) -> Self {
        self.overrides.insert(addr, proposal);
        self
    }

    /// The underlying model.
    pub fn model(&self) -> &EvolutionModel<P, F> {
        &self.model
    }

    /// Draw an initial state: a prior sample's fully-scored trace.
    pub fn init<R: Rng>(&self, rng: &mut R) -> Trace {
        use fugue::runtime::handler::run;
        use fugue::runtime::interpreters::PriorHandler;
        let (_g, trace) = run(
            PriorHandler {
                rng,
                trace: Trace::default(),
            },
            (self.model.target_model())(),
        );
        trace
    }

    /// One π_β-invariant transition. Moves ANY site type.
    pub fn step<R: Rng>(&mut self, rng: &mut R, current: &Trace) -> (P::Genome, Trace) {
        adaptive_single_site_mh(
            rng,
            self.model.target_model(),
            current,
            &mut self.adaptation,
        )
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
