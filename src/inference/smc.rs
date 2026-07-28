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
    adaptive_smc, adaptive_smc_with_kernel, decode_particle, CrossoverKernel, Model, Particle,
    ResamplingMethod, SMCConfig, Trace,
};
use rand::Rng;

use super::likelihood::GenomeLikelihood;
use super::model::EvolutionModel;
use super::prior::GenomePrior;
use crate::fitness::traits::Fitness;
use crate::genome::trace_genome::{gene_address, TraceGenome};

/// Configuration of the crossover population kernel.
#[derive(Clone, Debug)]
pub struct CrossoverConfig {
    /// Number of (pair, swap) proposals per sweep.
    pub n_pairs: usize,
    /// Per-address probability that a site joins the swap mask.
    pub swap_probability: f64,
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
        let model_fn = model.smc_model();
        let smc_cfg = SMCConfig {
            resampling_method: cfg.resampling,
            ess_threshold: cfg.ess_threshold,
            rejuvenation_steps: cfg.rejuvenation_steps,
        };
        let result = match cfg.crossover {
            None => adaptive_smc(rng, cfg.num_particles, &model_fn, smc_cfg),
            Some(xcfg) => {
                let p_swap = xcfg.swap_probability.clamp(0.0, 1.0);
                let mut kernel = CrossoverKernel {
                    n_pairs: xcfg.n_pairs,
                    // Value-independent, pair-symmetric mask: each address of
                    // the first parent joins the swap independently.
                    mask: Box::new(move |a: &Trace, _b: &Trace, rng: &mut dyn rand::RngCore| {
                        a.choices
                            .keys()
                            .filter(|_| rand::Rng::gen::<f64>(rng) < p_swap)
                            .cloned()
                            .collect()
                    }),
                };
                adaptive_smc_with_kernel(rng, cfg.num_particles, &model_fn, smc_cfg, &mut kernel)
            }
        };
        EvolutionPosterior {
            particles: result.particles,
            log_evidence: result.log_evidence,
            _g: PhantomData,
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
    /// rejuvenated with π_β-invariant MH (plus the crossover kernel when
    /// `cfg.crossover` is set). The rung schedule is geometric from 1 to
    /// `beta_max` over `anneal_steps` rungs.
    ///
    /// The returned population approximates `π_{β_max} ∝ p(x)·L(x)^{β_max}`,
    /// which for large `beta_max` concentrates on the optima — a principled,
    /// uncertainty-aware replacement for a classic GA on single-objective
    /// problems. `log_evidence` reflects only the β ≤ 1 ladder (evidence is
    /// defined at the posterior).
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
        use fugue::{normalize_particles, rejuvenate_particles, resample_particles};

        let crossover = cfg.crossover.clone();
        let rejuvenation_steps = cfg.rejuvenation_steps;
        let resampling = cfg.resampling;
        let mut result = Self::run(rng, model, cfg);
        if beta_max <= 1.0 || anneal_steps == 0 {
            return result;
        }

        let model_fn = model.smc_model();
        let loglik = |t: &Trace| t.log_likelihood + t.log_factors;
        let mut kernel = crossover.map(|xcfg| {
            let p_swap = xcfg.swap_probability.clamp(0.0, 1.0);
            CrossoverKernel {
                n_pairs: xcfg.n_pairs,
                mask: Box::new(move |a: &Trace, _b: &Trace, rng: &mut dyn rand::RngCore| {
                    a.choices
                        .keys()
                        .filter(|_| rand::Rng::gen::<f64>(rng) < p_swap)
                        .cloned()
                        .collect()
                }),
            }
        });

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

            // (3) π_β-invariant rejuvenation (+ optional crossover sweep).
            rejuvenate_particles(
                rng,
                &mut result.particles,
                &model_fn,
                beta,
                rejuvenation_steps,
            );
            if let Some(k) = kernel.as_mut() {
                fugue::PopulationKernel::<P::Genome>::sweep(
                    k,
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
