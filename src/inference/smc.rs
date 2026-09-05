//! Tempered SMC over the Boltzmann posterior, built on fugue's SMC engine
//!
//! The old `EvolutionarySMC` hand-rolled the whole tempering loop (linear β
//! ladder, weight normalization, ESS, systematic resampling, MH sweeps) and
//! carried a weight-model bug: it reweighted by `dβ · f(x)` on top of a
//! `β·f(x)` factor, double-counting β. This rebuild deletes all of it. The
//! driver is [`fugue::adaptive_smc_with_kernel`] run against the model's
//! **untempered** target (`factor(f)`, see
//! [`EvolutionModel::smc_model`](super::model::EvolutionModel::smc_model)):
//! fugue supplies β by likelihood-tempering with an adaptive ESS-driven
//! ladder, applies it exactly once, and returns an unbiased log-evidence
//! estimate for free.
//!
//! Crossover is fugue's [`CrossoverKernel`] — a population-coupled Metropolis
//! move on the product target — driven by an address mask supplied here
//! (genome knowledge stays downstream; trace-space mechanics live upstream).

use std::marker::PhantomData;

use fugue::runtime::handler::run;
use fugue::runtime::interpreters::ScoreGivenTrace;
use fugue::{
    adaptive_smc_with_kernel, decode_particle, score_given_trace_reconciled, Address, Model,
    NoKernel, Particle, PopulationKernel, ResamplingMethod, SMCConfig, Trace,
};
use rand::Rng;

use super::grammar::CrossoverMaskFn;
use super::likelihood::GenomeLikelihood;
use super::model::EvolutionModel;
use super::prior::GenomePrior;
use crate::fitness::traits::Fitness;
use crate::genome::trace_genome::{gene_address, TraceGenome};

/// Configuration of the generic crossover population kernel
/// ([`SharedSiteCrossover`]) that [`EvolutionSMC::run`] and
/// [`EvolutionSMC::anneal`] build from [`EvoSmcConfig::crossover`].
///
/// The kernel swaps a random subset of the addresses **shared by both
/// parents** (same address, same value type; each joins the swap
/// independently with probability `swap_probability`) and re-scores both
/// children strictly, rejecting any proposal whose children are not complete
/// executions over their own pre-swap address sets. That makes it safe on
/// every prior, including variable-structure ones (grammar trees,
/// variable-length genomes): it cannot panic and it never accepts a
/// structurally inconsistent child. The price is that on a variable-structure
/// prior only *structure-preserving* swaps are ever accepted — constants,
/// variable indices, same-arity function choices — because exchanging a
/// structural site (a `#leaf` flag, an arity-changing `#func`) opens a branch
/// the child has no choices for. Subtree grafts need the model-aware
/// [`subtree_crossover_mask`](super::grammar::subtree_crossover_mask) driven
/// through [`EvolutionSMC::run_with_kernel`] /
/// [`EvolutionSMC::anneal_with_kernel`]. On a fixed-structure prior (every
/// vector prior in [`super::prior`]) the shared set is the whole address set
/// and the kernel is exactly fugue's [`fugue::CrossoverKernel`].
#[derive(Clone, Debug)]
pub struct CrossoverConfig {
    /// Number of (pair, swap) proposals per sweep.
    pub n_pairs: usize,
    /// Per-address probability that a shared site joins the swap mask.
    pub swap_probability: f64,
}

/// The address set a [`SharedSiteCrossover`] proposal may exchange: every
/// address present in **both** traces with the same value type, each kept
/// independently with probability `p_swap`.
///
/// Value-independent (it reads only addresses and value *types*) and
/// symmetric in its two arguments (the shared set is a set intersection,
/// iterated in `BTreeMap` order, so the coin sequence is identical for
/// `(a, b)` and `(b, a)`), which is the mask contract of
/// [`fugue::CrossoverKernel`]. Exposed for callers who build fugue's kernel
/// directly for a fixed-structure model; [`SharedSiteCrossover`] adds the
/// strict re-score that variable-structure models need on top of it.
pub fn shared_site_crossover_mask(p_swap: f64) -> CrossoverMaskFn {
    let p_swap = p_swap.clamp(0.0, 1.0);
    Box::new(move |a: &Trace, b: &Trace, rng: &mut dyn rand::RngCore| {
        shared_site_mask(a, b, p_swap, rng)
    })
}

fn shared_site_mask(
    a: &Trace,
    b: &Trace,
    p_swap: f64,
    rng: &mut dyn rand::RngCore,
) -> Vec<Address> {
    a.choices
        .iter()
        .filter(|(addr, ca)| {
            b.choices
                .get(*addr)
                .is_some_and(|cb| cb.value.type_name() == ca.value.type_name())
        })
        .filter(|_| rand::Rng::gen::<f64>(rng) < p_swap)
        .map(|(addr, _)| addr.clone())
        .collect()
}

/// Exchange the choices at `swap` between `a` and `b` (pure choice surgery;
/// the children's accumulators are not valid until re-scored).
fn swap_block(a: &Trace, b: &Trace, swap: &[Address]) -> (Trace, Trace) {
    let mut ca = a.clone();
    let mut cb = b.clone();
    for addr in swap {
        let from_a = ca.choices.remove(addr);
        let from_b = cb.choices.remove(addr);
        if let Some(c) = from_b {
            ca.choices.insert(addr.clone(), c);
        }
        if let Some(c) = from_a {
            cb.choices.insert(addr.clone(), c);
        }
    }
    (ca, cb)
}

/// Re-score `base` and accept it only as a **complete execution over exactly
/// its own address set**: the model must visit every address of `base` (with
/// the base's value types) and no others. `None` otherwise.
///
/// Uses fugue's reconciling scorer rather than the strict one: the strict
/// handler hands the model a `Default::default()` value after recording a
/// missing address, and a model whose *structure* depends on that value (a
/// grammar reading a missing `#leaf` as `false` = "function node") recurses
/// without bound. The reconciling scorer draws the missing site from its
/// prior instead, so the replay always terminates, and its report tells us
/// exactly whether the child was complete: no fresh and no vanished sites.
/// The draws consume `rng` only on proposals that are rejected anyway.
fn rescore_complete<A>(
    base: &Trace,
    rng: &mut dyn rand::RngCore,
    model_fn: &dyn Fn() -> Model<A>,
) -> Option<Trace> {
    let mut rng = &mut *rng;
    let (_a, scored, report) =
        score_given_trace_reconciled(base.clone(), &mut rng, model_fn()).ok()?;
    (report.fresh_addresses.is_empty()
        && report.vanished_addresses.is_empty()
        && scored.choices.len() == base.choices.len())
    .then_some(scored)
}

/// Structure-safe crossover population kernel: fugue's [`fugue::CrossoverKernel`]
/// move (pick two distinct particles, exchange the block of choices at a
/// masked address set, accept the pair with the product-target Metropolis
/// ratio) with the mask of [`shared_site_crossover_mask`] and a
/// **completeness-checked** re-score (fugue's reconciling scorer plus its
/// fresh/vanished report) in place of fugue's panicking one.
///
/// A proposal is rejected outright when either child fails to re-score as a
/// complete execution over its own pre-swap address set — the model visited
/// an address the child does not hold (a structural site whose new value
/// opened a branch), or left some of the child's choices unvisited (a branch
/// closed). Every accepted pair therefore has exactly the parents' address
/// sets, so the swap is an involution on the state space and the mask
/// distribution is the same in both directions: the move is symmetric and
/// leaves the product of tempered targets invariant; the rejected proposals
/// are self-loops, which detailed balance ignores. This is what
/// [`EvoSmcConfig::crossover`] builds, so `EvolutionSMC::run` / `anneal`
/// with `EvoSmcConfig::default()` are safe on any [`GenomePrior`].
#[derive(Clone, Debug)]
pub struct SharedSiteCrossover {
    /// Number of (pair, swap) proposals per sweep.
    pub n_pairs: usize,
    /// Per-address probability that a shared site joins the swap mask.
    pub swap_probability: f64,
}

impl From<&CrossoverConfig> for SharedSiteCrossover {
    fn from(cfg: &CrossoverConfig) -> Self {
        Self {
            n_pairs: cfg.n_pairs,
            swap_probability: cfg.swap_probability.clamp(0.0, 1.0),
        }
    }
}

impl<A> PopulationKernel<A> for SharedSiteCrossover {
    fn sweep(
        &mut self,
        rng: &mut dyn rand::RngCore,
        particles: &mut [Particle],
        model_fn: &dyn Fn() -> Model<A>,
        beta: f64,
    ) {
        let n = particles.len();
        if n < 2 {
            return;
        }
        for _ in 0..self.n_pairs {
            let i = rng.gen_range(0..n);
            let mut j = rng.gen_range(0..n - 1);
            if j >= i {
                j += 1; // distinct partner
            }
            let s = shared_site_mask(
                &particles[i].trace,
                &particles[j].trace,
                self.swap_probability,
                rng,
            );
            if s.is_empty() {
                continue;
            }
            let (ti, tj) = swap_block(&particles[i].trace, &particles[j].trace, &s);
            let (Some(ci), Some(cj)) = (
                rescore_complete(&ti, rng, model_fn),
                rescore_complete(&tj, rng, model_fn),
            ) else {
                continue; // structurally inconsistent child: self-loop
            };
            // Tempered log-density of one execution:
            //   log π_β(θ) = log_prior + β·(log_likelihood + log_factors).
            let logd = |t: &Trace| t.log_prior + beta * (t.log_likelihood + t.log_factors);
            let log_alpha =
                (logd(&ci) + logd(&cj)) - (logd(&particles[i].trace) + logd(&particles[j].trace));
            if log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp() {
                particles[i].trace = ci; // only traces move;
                particles[j].trace = cj; // weights untouched (kernel contract).
            }
        }
    }
}

impl Default for CrossoverConfig {
    fn default() -> Self {
        Self {
            n_pairs: 32,
            swap_probability: 0.5,
        }
    }
}

/// Configuration for [`EvolutionSMC::run`].
pub struct EvoSmcConfig {
    /// Number of particles.
    pub num_particles: usize,
    /// ESS threshold fraction driving both the adaptive β ladder and
    /// resampling (fugue `SMCConfig::ess_threshold`).
    pub ess_threshold: f64,
    /// Resampling algorithm.
    pub resampling: ResamplingMethod,
    /// Per-particle MH rejuvenation sweeps per tempering step.
    pub rejuvenation_steps: usize,
    /// Population crossover kernel; `None` = per-particle rejuvenation only.
    pub crossover: Option<CrossoverConfig>,
}

impl Default for EvoSmcConfig {
    fn default() -> Self {
        Self {
            num_particles: 500,
            ess_threshold: 0.5,
            resampling: ResamplingMethod::Systematic,
            rejuvenation_steps: 3,
            crossover: Some(CrossoverConfig::default()),
        }
    }
}

/// The result of a tempered-SMC evolution run: fugue particles (traces +
/// normalized weights) approximating the Boltzmann posterior `π ∝ p·exp(f)`,
/// plus the log-evidence estimate.
///
/// Genomes are not cached on particles; they are recovered by **decode-replay**
/// (replaying the particle's trace through the prior/target program, whose
/// return value *is* the decoded genome).
pub struct EvolutionPosterior<G: TraceGenome> {
    /// Final weighted particle population (fugue particles).
    pub particles: Vec<Particle>,
    /// Unbiased estimate of the log normalizing constant
    /// `log Σ_x p(x)·exp(f(x))` — the Bayesian model score.
    pub log_evidence: f64,
    _g: PhantomData<G>,
}

impl<G: TraceGenome> EvolutionPosterior<G> {
    /// Recover the genome of one particle by replaying its trace.
    pub fn genome(&self, particle: &Particle, model_fn: &impl Fn() -> Model<G>) -> G {
        decode_particle(particle, model_fn)
    }

    /// Decode the whole population as `(genome, normalized_weight)` pairs.
    pub fn genomes(&self, model_fn: &impl Fn() -> Model<G>) -> Vec<(G, f64)> {
        self.particles
            .iter()
            .map(|p| (decode_particle(p, model_fn), p.weight))
            .collect()
    }

    /// Self-normalised weighted posterior mean of coordinate `gene#coord`.
    pub fn weighted_mean(&self, coord: usize) -> f64 {
        let addr = gene_address(G::trace_prefix(), coord);
        let mut total_w = 0.0;
        let mut mean = 0.0;
        for p in &self.particles {
            if let Some(x) = p.trace.get_f64(&addr) {
                mean += p.weight * x;
                total_w += p.weight;
            }
        }
        if total_w > 0.0 {
            mean / total_w
        } else {
            0.0
        }
    }

    /// Self-normalised weighted posterior variance of coordinate `gene#coord`.
    pub fn weighted_variance(&self, coord: usize) -> f64 {
        let addr = gene_address(G::trace_prefix(), coord);
        let mean = self.weighted_mean(coord);
        let mut total_w = 0.0;
        let mut var = 0.0;
        for p in &self.particles {
            if let Some(x) = p.trace.get_f64(&addr) {
                var += p.weight * (x - mean).powi(2);
                total_w += p.weight;
            }
        }
        if total_w > 0.0 {
            var / total_w
        } else {
            0.0
        }
    }

    /// The decoded genome with the highest fitness, and that fitness —
    /// the optimizer-mode readout for benchmarking against the classic layer.
    pub fn best<F>(&self, fitness: &F, model_fn: &impl Fn() -> Model<G>) -> Option<(G, f64)>
    where
        F: Fitness<Genome = G, Value = f64>,
    {
        self.particles
            .iter()
            .map(|p| {
                let g = decode_particle(p, model_fn);
                let f = fitness.evaluate(&g);
                (g, f)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// The tempered-SMC evolution driver.
pub struct EvolutionSMC;

impl EvolutionSMC {
    /// Run tempered SMC targeting the Boltzmann posterior
    /// `π ∝ p(x)·exp(f(x))` of `model` (β is supplied by fugue's adaptive
    /// tempering; `model`'s own β setting is ignored here by construction).
    ///
    /// `cfg.crossover` builds a structure-safe [`SharedSiteCrossover`] kernel
    /// (see [`CrossoverConfig`] for what it can and cannot exchange on a
    /// variable-structure prior); `None` runs per-particle rejuvenation only.
    pub fn run<P, L, R>(
        rng: &mut R,
        model: &EvolutionModel<P, L>,
        cfg: EvoSmcConfig,
    ) -> EvolutionPosterior<P::Genome>
    where
        P: GenomePrior,
        L: GenomeLikelihood<P::Genome>,
        R: Rng,
    {
        match cfg.crossover.as_ref().map(SharedSiteCrossover::from) {
            None => Self::run_with_kernel(rng, model, cfg, &mut NoKernel),
            Some(mut kernel) => Self::run_with_kernel(rng, model, cfg, &mut kernel),
        }
    }
}

impl EvolutionSMC {
    /// Like [`EvolutionSMC::run`], but with an explicit population kernel
    /// (e.g. a [`CrossoverKernel`] with a
    /// [`subtree_crossover_mask`](super::grammar::subtree_crossover_mask) for
    /// grammar-driven tree genomes). `cfg.crossover` is ignored.
    pub fn run_with_kernel<P, L, R, K>(
        rng: &mut R,
        model: &EvolutionModel<P, L>,
        cfg: EvoSmcConfig,
        kernel: &mut K,
    ) -> EvolutionPosterior<P::Genome>
    where
        P: GenomePrior,
        L: GenomeLikelihood<P::Genome>,
        R: Rng,
        K: fugue::PopulationKernel<P::Genome>,
    {
        let model_fn = model.smc_model();
        let smc_cfg = SMCConfig {
            resampling_method: cfg.resampling,
            ess_threshold: cfg.ess_threshold,
            rejuvenation_steps: cfg.rejuvenation_steps,
        };
        let result = adaptive_smc_with_kernel(rng, cfg.num_particles, &model_fn, smc_cfg, kernel);
        EvolutionPosterior {
            particles: result.particles,
            log_evidence: result.log_evidence,
            _g: PhantomData,
        }
    }
}

impl EvolutionSMC {
    /// **Optimizer mode**: run tempered SMC to the posterior (β = 1), then
    /// keep annealing the ladder toward `beta_max`, concentrating the
    /// population on the maximizers of the likelihood/fitness.
    ///
    /// The continuation is built from fugue's exported primitives and keeps
    /// every invariant of the tempering loop: at each rung the particles are
    /// incrementally reweighted by `Δβ·(log_likelihood + log_factors)`,
    /// normalized, systematically resampled to uniform weights, and
    /// rejuvenated with π_β-invariant MH (plus the structure-safe
    /// [`SharedSiteCrossover`] sweep when `cfg.crossover` is set — see
    /// [`CrossoverConfig`]; for a model-aware kernel such as
    /// [`subtree_crossover_mask`](super::grammar::subtree_crossover_mask) use
    /// [`EvolutionSMC::anneal_with_kernel`]). The rung schedule is geometric
    /// from 1 to `beta_max` over `anneal_steps` rungs.
    ///
    /// The returned population approximates `π_{β_max} ∝ p(x)·L(x)^{β_max}`,
    /// which for large `beta_max` concentrates on the optima — a principled,
    /// uncertainty-aware replacement for a classic GA on single-objective
    /// problems. `log_evidence` reflects only the β ≤ 1 ladder (evidence is
    /// defined at the posterior).
    ///
    /// With `rejuvenation_steps == 0` and no kernel nothing moves a particle
    /// past β = 1: each rung only reweights and resamples, so the population
    /// collapses onto duplicates of the fittest posterior particles. Keep at
    /// least one rejuvenation step (or a kernel) when annealing.
    pub fn anneal<P, L, R>(
        rng: &mut R,
        model: &EvolutionModel<P, L>,
        cfg: EvoSmcConfig,
        beta_max: f64,
        anneal_steps: usize,
    ) -> EvolutionPosterior<P::Genome>
    where
        P: GenomePrior,
        L: GenomeLikelihood<P::Genome>,
        R: Rng,
    {
        match cfg.crossover.as_ref().map(SharedSiteCrossover::from) {
            None => {
                Self::anneal_with_kernel(rng, model, cfg, beta_max, anneal_steps, &mut NoKernel)
            }
            Some(mut kernel) => {
                Self::anneal_with_kernel(rng, model, cfg, beta_max, anneal_steps, &mut kernel)
            }
        }
    }

    /// Like [`EvolutionSMC::anneal`], but with an explicit population kernel
    /// applied both inside the β ≤ 1 ladder and at every annealing rung —
    /// the optimizer-mode counterpart of [`EvolutionSMC::run_with_kernel`]
    /// (e.g. a [`fugue::CrossoverKernel`] with a
    /// [`subtree_crossover_mask`](super::grammar::subtree_crossover_mask) to
    /// anneal a grammar prior with subtree grafts). `cfg.crossover` is
    /// ignored; pass [`fugue::NoKernel`] for rejuvenation only.
    pub fn anneal_with_kernel<P, L, R, K>(
        rng: &mut R,
        model: &EvolutionModel<P, L>,
        cfg: EvoSmcConfig,
        beta_max: f64,
        anneal_steps: usize,
        kernel: &mut K,
    ) -> EvolutionPosterior<P::Genome>
    where
        P: GenomePrior,
        L: GenomeLikelihood<P::Genome>,
        R: Rng,
        K: PopulationKernel<P::Genome>,
    {
        use fugue::{normalize_particles, rejuvenate_particles, resample_particles};

        let rejuvenation_steps = cfg.rejuvenation_steps;
        let resampling = cfg.resampling;
        let mut result = Self::run_with_kernel(rng, model, cfg, kernel);
        if beta_max <= 1.0 || anneal_steps == 0 {
            return result;
        }

        let model_fn = model.smc_model();
        let loglik = |t: &Trace| t.log_likelihood + t.log_factors;

        let ln_bmax = beta_max.ln();
        let mut prev_beta = 1.0;
        for i in 1..=anneal_steps {
            let beta = (ln_bmax * i as f64 / anneal_steps as f64).exp();
            let d_beta = beta - prev_beta;

            // (1) incremental reweight by the tempered increment.
            for p in &mut result.particles {
                p.log_weight += d_beta * loglik(&p.trace);
            }
            normalize_particles(&mut result.particles);

            // (2) resample to uniform weights.
            result.particles = resample_particles(rng, &result.particles, resampling);

            // (3) π_β-invariant rejuvenation (+ optional population kernel).
            rejuvenate_particles(
                rng,
                &mut result.particles,
                &model_fn,
                beta,
                rejuvenation_steps,
            );
            if !kernel.is_identity() {
                kernel.sweep(
                    rng as &mut dyn rand::RngCore,
                    &mut result.particles,
                    &model_fn,
                    beta,
                );
            }
            prev_beta = beta;
        }
        normalize_particles(&mut result.particles);
        result
    }
}

/// Score a genome's canonical trace under an arbitrary model — convenience
/// used by readouts and tests.
pub fn score_genome<G: TraceGenome, A>(genome: &G, model: Model<A>) -> (A, Trace) {
    run(
        ScoreGivenTrace {
            base: genome.to_trace(),
            trace: Trace::default(),
        },
        model,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::bounds::{Bounds, MultiBounds};
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use crate::inference::model::tests::PtrFitness;
    use crate::inference::prior::GaussianPrior;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn quad_k1_c3(g: &RealVector) -> f64 {
        -0.5 * g.genes().iter().map(|x| (x - 3.0).powi(2)).sum::<f64>()
    }

    /// Regression: EV-16 — tempered SMC on a quadratic fitness with a Gaussian
    /// prior reproduces the conjugate Boltzmann posterior.
    ///
    /// Prior N(0, σ0²=4) ⇒ τ0 = 0.25; fitness −0.5(x−3)² ⇒ k = 1, c = 3.
    /// Posterior at β=1: τ = 1.25, mean = 3/1.25 = 2.4, variance = 0.8.
    /// Re-driven through the fugue-backed rebuild — this directly exercises
    /// the β-single-counting fix (fitness enters as `factor(f)`; β only from
    /// tempering).
    #[test]
    fn test_smc_matches_gaussian_conjugate_posterior() {
        let prior = GaussianPrior::new(0.0, 2.0, 1);
        let model = EvolutionModel::new(prior, PtrFitness(quad_k1_c3));
        let mut rng = StdRng::seed_from_u64(42);
        let result = EvolutionSMC::run(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 4000,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 6,
                crossover: None,
            },
        );

        let mean = result.weighted_mean(0);
        let var = result.weighted_variance(0);
        assert!(
            (mean - 2.4).abs() < 0.15,
            "posterior mean {} vs analytic 2.4",
            mean
        );
        assert!(
            (var - 0.8).abs() < 0.2,
            "posterior variance {} vs analytic 0.8",
            var
        );

        // Weights are self-normalised.
        let total: f64 = result.particles.iter().map(|p| p.weight).sum();
        assert!((total - 1.0).abs() < 1e-6);

        // Analytic evidence check comes for free from the rebuild:
        // Z = ∫ N(x; 0, 4)·e^{-(x-3)²/2} dx = √(2π·0.8)/√(2π·4) · e^{-9/(2·5)}
        let analytic_log_z = 0.5 * ((0.8f64).ln() - (4.0f64).ln()) - 9.0 / (2.0 * 5.0);
        assert!(
            (result.log_evidence - analytic_log_z).abs() < 0.25,
            "log evidence {} vs analytic {}",
            result.log_evidence,
            analytic_log_z
        );
    }

    /// Same conjugate target, with the crossover population kernel enabled —
    /// the kernel must not bias the posterior (product-target invariance) nor
    /// the evidence (FG-58).
    #[test]
    fn test_smc_with_crossover_matches_conjugate_posterior() {
        let prior = GaussianPrior::new(0.0, 2.0, 2);
        // Independent per-coordinate quadratic pull toward 3.
        let model = EvolutionModel::new(prior, PtrFitness(quad_k1_c3));
        let mut rng = StdRng::seed_from_u64(1234);
        let result = EvolutionSMC::run(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 3000,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 4,
                crossover: Some(CrossoverConfig {
                    n_pairs: 500,
                    swap_probability: 0.5,
                }),
            },
        );
        for coord in 0..2 {
            let mean = result.weighted_mean(coord);
            let var = result.weighted_variance(coord);
            assert!(
                (mean - 2.4).abs() < 0.15,
                "coord {} posterior mean {} vs 2.4",
                coord,
                mean
            );
            assert!(
                (var - 0.8).abs() < 0.25,
                "coord {} posterior variance {} vs 0.8",
                coord,
                var
            );
        }
    }

    /// The rebuilt result exposes decode-replay: recover genomes from bare
    /// particle traces via the prior program's return value.
    #[test]
    fn test_decode_replay_recovers_genomes() {
        let prior = GaussianPrior::new(0.0, 2.0, 1);
        let model = EvolutionModel::new(prior, PtrFitness(quad_k1_c3));
        let mut rng = StdRng::seed_from_u64(5);
        let result = EvolutionSMC::run(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 100,
                rejuvenation_steps: 2,
                crossover: None,
                ..Default::default()
            },
        );
        let model_fn = model.smc_model();
        let decoded = result.genomes(&model_fn);
        assert_eq!(decoded.len(), 100);
        for (g, _w) in &decoded {
            assert_eq!(g.genes().len(), 1);
        }
        let (best, best_f) = result.best(&PtrFitness(quad_k1_c3), &model_fn).unwrap();
        assert!(best_f.is_finite());
        assert!((quad_k1_c3(&best) - best_f).abs() < 1e-12);
    }

    /// Optimizer mode: annealing past β = 1 concentrates the population on
    /// the fitness optimum far beyond the posterior's spread.
    #[test]
    fn test_anneal_concentrates_on_optimum() {
        // Fitness -0.5·Σx², prior N(0, 2²): posterior sd ≈ 0.89; at β = 200
        // the tempered target's sd ≈ 0.07.
        let prior = GaussianPrior::new(0.0, 2.0, 2);
        let model = EvolutionModel::new(prior, PtrFitness(super::tests::quad_origin_local));
        let mut rng = StdRng::seed_from_u64(31);
        let cfg = || EvoSmcConfig {
            num_particles: 400,
            ess_threshold: 0.5,
            resampling: ResamplingMethod::Systematic,
            rejuvenation_steps: 4,
            crossover: Some(CrossoverConfig::default()),
        };
        let posterior = EvolutionSMC::run(&mut rng, &model, cfg());
        let annealed = EvolutionSMC::anneal(&mut rng, &model, cfg(), 200.0, 12);

        let spread = |r: &EvolutionPosterior<RealVector>| {
            (r.weighted_variance(0) + r.weighted_variance(1)).sqrt()
        };
        assert!(
            spread(&annealed) < 0.35 * spread(&posterior),
            "annealed spread {} should be far below posterior spread {}",
            spread(&annealed),
            spread(&posterior)
        );

        let model_fn = model.smc_model();
        let (best, best_f) = annealed
            .best(&PtrFitness(super::tests::quad_origin_local), &model_fn)
            .unwrap();
        assert!(
            best_f > -0.02,
            "annealed best fitness {} (genome {:?}) not near optimum 0",
            best_f,
            best.genes()
        );
    }

    /// EV-N1: the generic crossover of `EvoSmcConfig::default()` used to
    /// panic on any variable-structure prior (the mask was a random subset of
    /// the first parent's addresses; `swap_block` could move a site absent
    /// from the partner out of a child, and fugue's `ScoreGivenTrace` re-score
    /// panics on a missing site). The default config must now run on the
    /// grammar prior and produce a sane posterior: normalized weights, finite
    /// evidence, every particle decodes to a valid tree with finite prior
    /// mass, and the posterior predictive tracks the data.
    #[test]
    fn test_default_config_runs_on_grammar_prior() {
        use crate::inference::grammar::{ArithmeticGrammarPrior, GaussianRegression, NoiseSpec};
        let xs: Vec<f64> = (-10..=10).map(|i| i as f64 / 5.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x + 1.0).collect();
        let prior = ArithmeticGrammarPrior {
            terminal_prob: 0.45,
            max_depth: 3,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 1, // {Add}
        };
        let likelihood = GaussianRegression {
            xs: xs.clone(),
            ys,
            noise: NoiseSpec::Fixed(0.3),
        };
        let model = EvolutionModel::from_likelihood(prior, likelihood);
        let mut rng = StdRng::seed_from_u64(2026);
        let cfg = EvoSmcConfig::default();
        assert!(
            cfg.crossover.is_some(),
            "the default config must exercise the kernel"
        );
        let result = EvolutionSMC::run(&mut rng, &model, cfg);

        let total: f64 = result.particles.iter().map(|p| p.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "weights not normalized: {total}"
        );
        assert!(
            result.log_evidence.is_finite(),
            "log evidence {}",
            result.log_evidence
        );
        let model_fn = model.smc_model();
        let decoded = fugue::decode_particles(&result.particles, &model_fn);
        for (p, (tree, _w)) in result.particles.iter().zip(&decoded) {
            assert!(p.trace.log_prior.is_finite());
            assert!(tree.size() >= 1);
        }
        for &x in &[-1.0, 0.0, 1.5] {
            let pred: f64 = decoded
                .iter()
                .map(|(tree, w)| {
                    let v = tree.evaluate(&[x]);
                    if v.is_finite() {
                        w * v
                    } else {
                        0.0
                    }
                })
                .sum();
            assert!(
                (pred - (x + 1.0)).abs() < 0.35,
                "posterior predictive at {x} was {pred} vs truth {}",
                x + 1.0
            );
        }
    }

    /// EV-N1: `anneal` (the advertised optimizer mode) with the default
    /// config on a grammar prior — previously impossible (no `_with_kernel`
    /// variant, and the default kernel panicked). Annealing must concentrate
    /// the population on higher-fitness programs than the posterior does.
    #[test]
    fn test_anneal_runs_on_grammar_prior() {
        use crate::genome::tree::{ArithmeticFunction, ArithmeticTerminal, TreeGenome};
        use crate::inference::grammar::ArithmeticGrammarPrior;

        #[derive(Clone)]
        struct Fit {
            xs: Vec<f64>,
            ys: Vec<f64>,
        }
        impl Fitness for Fit {
            type Genome = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;
            type Value = f64;
            fn evaluate(&self, tree: &Self::Genome) -> f64 {
                let sse: f64 = self
                    .xs
                    .iter()
                    .zip(&self.ys)
                    .map(|(&x, &y)| {
                        let p = tree.evaluate(&[x]);
                        if p.is_finite() {
                            (p - y).powi(2)
                        } else {
                            1e6
                        }
                    })
                    .sum();
                -0.5 * sse / (0.3 * 0.3)
            }
        }

        let xs: Vec<f64> = (-8..=8).map(|i| i as f64 / 4.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let fitness = Fit { xs, ys };
        let prior = ArithmeticGrammarPrior {
            terminal_prob: 0.4,
            max_depth: 3,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 3, // Add, Sub, Mul
        };
        let model = EvolutionModel::new(prior, fitness.clone());
        let cfg = || EvoSmcConfig {
            num_particles: 300,
            rejuvenation_steps: 3,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(99);
        let posterior = EvolutionSMC::run(&mut rng, &model, cfg());
        let annealed = EvolutionSMC::anneal(&mut rng, &model, cfg(), 10.0, 5);

        let model_fn = model.smc_model();
        let mean_fitness = |r: &EvolutionPosterior<_>| -> f64 {
            fugue::decode_particles(&r.particles, &model_fn)
                .iter()
                .map(|(t, w)| w * fitness.evaluate(t))
                .sum()
        };
        let (post_f, ann_f) = (mean_fitness(&posterior), mean_fitness(&annealed));
        assert!(ann_f.is_finite() && post_f.is_finite());
        assert!(
            ann_f > post_f,
            "annealed mean fitness {ann_f} should exceed posterior mean fitness {post_f}"
        );
        for p in &annealed.particles {
            assert!(p.trace.log_prior.is_finite());
        }
    }

    /// EV-N1 mechanism test: a population whose parents disagree on a
    /// structural site (`node#leaf`: leaf root vs function root) is swept with
    /// `swap_probability = 1` many times. Every structural swap must be
    /// rejected as a self-loop (no panic, address sets unchanged), while the
    /// kernel still moves the shared, structure-preserving sites.
    #[test]
    fn test_shared_site_crossover_rejects_structural_mismatch() {
        use crate::genome::tree::{ArithmeticFunction, ArithmeticTerminal, TreeGenome, TreeNode};
        use crate::inference::grammar::ArithmeticGrammarPrior;

        let prior = ArithmeticGrammarPrior::default();
        let model_fn = || prior.model();
        // Two leaf roots (different constants) and two function roots.
        let leaf = |c: f64| TreeGenome::new(TreeNode::terminal(ArithmeticTerminal::Constant(c)), 6);
        let func = |c: f64| {
            TreeGenome::new(
                TreeNode::function(
                    ArithmeticFunction::Add,
                    vec![
                        TreeNode::terminal(ArithmeticTerminal::Variable(0)),
                        TreeNode::terminal(ArithmeticTerminal::Constant(c)),
                    ],
                ),
                6,
            )
        };
        let trees = [leaf(0.5), leaf(-0.7), func(1.0), func(2.0)];
        let mut particles: Vec<Particle> = trees
            .iter()
            .map(|t| {
                let (_g, scored) = run(
                    ScoreGivenTrace {
                        base: prior.trace_of(t),
                        trace: Trace::default(),
                    },
                    model_fn(),
                );
                Particle {
                    trace: scored,
                    log_weight: 0.0,
                    weight: 0.25,
                }
            })
            .collect();
        let before: Vec<Vec<Address>> = particles
            .iter()
            .map(|p| p.trace.choices.keys().cloned().collect())
            .collect();

        let mut kernel = SharedSiteCrossover {
            n_pairs: 400,
            swap_probability: 1.0,
        };
        let mut rng = StdRng::seed_from_u64(8);
        PopulationKernel::<TreeGenome<ArithmeticTerminal, ArithmeticFunction>>::sweep(
            &mut kernel,
            &mut rng,
            &mut particles,
            &model_fn,
            1.0,
        );

        for (p, addrs) in particles.iter().zip(&before) {
            let after: Vec<Address> = p.trace.choices.keys().cloned().collect();
            assert_eq!(&after, addrs, "address set changed across a swap");
            assert!(p.trace.log_prior.is_finite());
            let tree = decode_particle(p, model_fn);
            assert!(tree.size() >= 1);
        }
        // Structure-preserving swaps did happen: the two leaves' constants
        // (or the two function roots' constants) were exchanged at least once
        // — with p_swap = 1 and 400 pair draws this is certain up to
        // acceptance, and same-structure swaps are accepted with ratio 1.
        let consts: Vec<f64> = particles
            .iter()
            .filter_map(|p| p.trace.get_f64(&fugue::addr!("node", "const")))
            .collect();
        assert_eq!(consts.len(), 2);
        let moved = particles.iter().any(|p| {
            p.trace
                .get_f64(&fugue::addr!("node/1", "const"))
                .is_some_and(|c| c != 1.0)
        }) || consts[0] != 0.5;
        assert!(moved, "no structure-preserving swap was ever accepted");
    }

    pub(super) fn quad_origin_local(g: &RealVector) -> f64 {
        -0.5 * g.genes().iter().map(|x| x * x).sum::<f64>()
    }

    /// Bounds are respected end-to-end: with a uniform-box prior every
    /// particle stays inside the box (out-of-box scores −∞ and can never
    /// survive).
    #[test]
    fn test_smc_respects_bounds() {
        use crate::inference::prior::UniformBoxPrior;
        let prior = UniformBoxPrior::new(MultiBounds::new(vec![Bounds::new(-2.0, 2.0)]));
        let model = EvolutionModel::new(prior, PtrFitness(|g: &RealVector| g.genes()[0]));
        let mut rng = StdRng::seed_from_u64(9);
        let result = EvolutionSMC::run(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 300,
                rejuvenation_steps: 3,
                crossover: Some(CrossoverConfig::default()),
                ..Default::default()
            },
        );
        for p in &result.particles {
            let x = p.trace.get_f64(&fugue::addr!("gene", 0)).unwrap();
            assert!((-2.0..=2.0).contains(&x), "particle escaped bounds: {}", x);
        }
    }
}
