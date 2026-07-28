//! Incremental tempered-SMC explorable: evolution as inference, live.
//!
//! [`ExploreSmcInference`] drives the crate's *real* inference layer — a
//! [`GaussianPrior`] program, a bimodal fitness entering as a
//! [`FactorFitness`] likelihood, and fugue's SMC primitives
//! (`smc_prior_particles` / `normalize_particles` / `resample_particles` /
//! `rejuvenate_particles` / `CrossoverKernel`) — one tempering rung per
//! `step()`, streaming the particle population, ESS, resampling events,
//! accepted crossover swaps, and the running log-evidence as JSON.
//!
//! Module conventions match `explore.rs`: explicit `u64` seeds (a seed is a
//! replayable recording), every JS-supplied parameter clamped, and 2-D
//! genomes so positions map straight to the canvas. `density_grid` returns
//! the analytic **negative** log-target (lower = better, like the landscape
//! grids) so the same heat conventions apply and the particles can be seen
//! matching the exact tempered density at every β.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use wasm_bindgen::prelude::*;

use fugue::{
    addr, effective_sample_size, normalize_particles, rejuvenate_particles, resample_particles,
    smc_prior_particles, CrossoverKernel, Particle, PopulationKernel, ResamplingMethod, Trace,
};
use fugue_evo::fitness::traits::Fitness;
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::RealValuedGenome;
use fugue_evo::inference::likelihood::FactorFitness;
use fugue_evo::inference::model::EvolutionModel;
use fugue_evo::inference::prior::GaussianPrior;

/// Plot domain (matches the prior's ±2.25σ window).
const DOMAIN: (f64, f64) = (-4.5, 4.5);
/// Prior standard deviation per coordinate.
const PRIOR_STD: f64 = 2.0;

/// The two modes of the twin-peaks fitness: (center, std, mixture weight).
const MODES: [([f64; 2], f64, f64); 2] = [([-1.6, -0.9], 0.55, 0.65), ([1.7, 1.2], 0.8, 0.35)];

/// Log-mixture-of-Gaussians fitness (higher is better): two unequal peaks,
/// so the Boltzmann posterior is visibly bimodal at β = 1 and annealing
/// (β > 1) shifts mass toward the sharper peak — "selection pressure is
/// conditioning; temperature is selection strength", drawable.
#[derive(Clone, Copy)]
struct TwinPeaks;

fn twin_peaks_f(x: f64, y: f64) -> f64 {
    let mut s = 0.0;
    for (m, sd, w) in MODES {
        let r2 = (x - m[0]).powi(2) + (y - m[1]).powi(2);
        s += w * (-r2 / (2.0 * sd * sd)).exp();
    }
    if s > 0.0 {
        s.ln()
    } else {
        -1e12 // far outside the domain: crushed, never accepted
    }
}

impl Fitness for TwinPeaks {
    type Genome = RealVector;
    type Value = f64;
    fn evaluate(&self, g: &RealVector) -> f64 {
        let genes = g.genes();
        twin_peaks_f(genes[0], genes[1])
    }
}

/// Prior log-density (up to a constant): i.i.d. `N(0, PRIOR_STD²)`.
fn log_prior(x: f64, y: f64) -> f64 {
    -(x * x + y * y) / (2.0 * PRIOR_STD * PRIOR_STD)
}

fn loglik(t: &Trace) -> f64 {
    t.log_likelihood + t.log_factors
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// Incremental likelihood-tempered SMC over the Boltzmann posterior
/// `π_β(x) ∝ p(x)·exp(β·f(x))`, one rung per `step()`.
#[wasm_bindgen]
pub struct ExploreSmcInference {
    model: EvolutionModel<GaussianPrior, FactorFitness<TwinPeaks>>,
    particles: Vec<Particle>,
    rng: StdRng,
    rung: usize,
    n_rungs: usize,
    beta: f64,
    beta_max: f64,
    log_evidence: f64,
    rejuv_steps: usize,
    crossover: bool,
    resampled_last: bool,
    swaps_last: usize,
}

#[wasm_bindgen]
impl ExploreSmcInference {
    /// `pop_size` particles, a linear β ladder of `n_rungs` rungs from 0 to
    /// `beta_max` (1.0 = the posterior; > 1 keeps annealing into optimizer
    /// territory), optional population crossover, explicit seed.
    #[wasm_bindgen(constructor)]
    pub fn new(
        pop_size: usize,
        n_rungs: usize,
        beta_max: f64,
        crossover: bool,
        seed: u64,
    ) -> Result<ExploreSmcInference, JsValue> {
        let pop_size = pop_size.clamp(32, 600);
        let n_rungs = n_rungs.clamp(4, 60);
        let beta_max = if beta_max.is_finite() {
            beta_max.clamp(1.0, 8.0)
        } else {
            1.0
        };
        let model = EvolutionModel::new(GaussianPrior::new(0.0, PRIOR_STD, 2), TwinPeaks);
        let mut rng = StdRng::seed_from_u64(seed);
        let model_fn = model.smc_model();
        let mut particles = smc_prior_particles(&mut rng, pop_size, &model_fn);
        // β = 0 semantics: the ladder starts at the prior with uniform
        // weights (smc_prior_particles sets β = 1 importance weights).
        let inv_n = 1.0 / pop_size as f64;
        for p in &mut particles {
            p.weight = inv_n;
            p.log_weight = inv_n.ln();
        }
        drop(model_fn);
        Ok(ExploreSmcInference {
            model,
            particles,
            rng,
            rung: 0,
            n_rungs,
            beta: 0.0,
            beta_max,
            log_evidence: 0.0,
            rejuv_steps: 3,
            crossover,
            resampled_last: false,
            swaps_last: 0,
        })
    }

    /// Current state without stepping (for the initial paint).
    pub fn snapshot(&self) -> String {
        self.state_json()
    }

    /// Advance one tempering rung: incremental reweight by `Δβ·loglik`,
    /// evidence accumulation (β ≤ 1 portion only), normalize, ESS-triggered
    /// systematic resampling, π_β-invariant MH rejuvenation, and (optionally)
    /// a crossover sweep — all through fugue's real primitives.
    pub fn step(&mut self) -> String {
        if self.rung >= self.n_rungs {
            return self.state_json();
        }
        self.rung += 1;
        let new_beta = self.beta_max * self.rung as f64 / self.n_rungs as f64;
        let d_beta = new_beta - self.beta;
        let n = self.particles.len() as f64;

        // Evidence increment covers only the β ≤ 1 portion of this rung
        // (log Z is defined at the posterior; annealing past 1 adds nothing).
        let d_cap = new_beta.min(1.0) - self.beta.min(1.0);
        if d_cap > 0.0 {
            let lw: Vec<f64> = self
                .particles
                .iter()
                .map(|p| p.log_weight + d_cap * loglik(&p.trace))
                .collect();
            self.log_evidence += log_sum_exp(&lw);
        }

        // (1) incremental reweight + normalize (keeping log_weight in sync —
        // fugue's normalize_particles updates only the linear weights).
        for p in &mut self.particles {
            p.log_weight += d_beta * loglik(&p.trace);
        }
        normalize_particles(&mut self.particles);
        for p in &mut self.particles {
            p.log_weight = if p.weight > 0.0 {
                p.weight.ln()
            } else {
                f64::NEG_INFINITY
            };
        }

        // (2) ESS-triggered systematic resampling.
        let ess = effective_sample_size(&self.particles);
        self.resampled_last = ess < 0.5 * n;
        if self.resampled_last {
            self.particles =
                resample_particles(&mut self.rng, &self.particles, ResamplingMethod::Systematic);
        }

        // (3) π_β-invariant rejuvenation + optional crossover sweep.
        let model_fn = self.model.smc_model();
        rejuvenate_particles(
            &mut self.rng,
            &mut self.particles,
            &model_fn,
            new_beta,
            self.rejuv_steps,
        );
        self.swaps_last = 0;
        if self.crossover && self.particles.len() >= 2 {
            let before: Vec<Option<f64>> = self
                .particles
                .iter()
                .map(|p| p.trace.get_f64(&addr!("gene", 0usize)))
                .collect();
            let mut kernel = CrossoverKernel {
                n_pairs: self.particles.len() / 2,
                // Swap the x-coordinate block between the pair: a
                // product-target Metropolis move that exchanges mode
                // membership horizontally.
                mask: Box::new(|_: &Trace, _: &Trace, _: &mut dyn rand::RngCore| {
                    vec![addr!("gene", 0usize)]
                }),
            };
            PopulationKernel::<RealVector>::sweep(
                &mut kernel,
                &mut self.rng as &mut dyn rand::RngCore,
                &mut self.particles,
                &model_fn,
                new_beta,
            );
            self.swaps_last = self
                .particles
                .iter()
                .zip(&before)
                .filter(|(p, b)| p.trace.get_f64(&addr!("gene", 0usize)) != **b)
                .count();
        }

        self.beta = new_beta;
        self.state_json()
    }

    /// The analytic tempered target on an `nx × ny` grid over the plot
    /// domain, as **negative** log-density `−(log p + β·f)` (lower = better,
    /// matching the landscape-heat convention), row-major with `j` indexing
    /// `y` upward and cell-centered sampling.
    pub fn density_grid(&self, nx: usize, ny: usize) -> Vec<f64> {
        let nx = nx.clamp(2, 400);
        let ny = ny.clamp(2, 400);
        let (lo, hi) = DOMAIN;
        let mut out = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let y = lo + (hi - lo) * (j as f64 + 0.5) / ny as f64;
            for i in 0..nx {
                let x = lo + (hi - lo) * (i as f64 + 0.5) / nx as f64;
                out.push(-(log_prior(x, y) + self.beta * twin_peaks_f(x, y)));
            }
        }
        out
    }

    /// Plot metadata: `{lo, hi, prior_std, modes: [[x, y]…]}`.
    pub fn info(&self) -> String {
        json!({
            "lo": DOMAIN.0,
            "hi": DOMAIN.1,
            "prior_std": PRIOR_STD,
            "modes": MODES.iter().map(|(m, _, _)| vec![m[0], m[1]]).collect::<Vec<_>>(),
        })
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.rung
    }

    fn state_json(&self) -> String {
        let pts: Vec<serde_json::Value> = self
            .particles
            .iter()
            .map(|p| {
                let x = p.trace.get_f64(&addr!("gene", 0usize)).unwrap_or(0.0);
                let y = p.trace.get_f64(&addr!("gene", 1usize)).unwrap_or(0.0);
                json!([x, y, p.weight])
            })
            .collect();
        json!({
            "rung": self.rung,
            "n_rungs": self.n_rungs,
            "beta": self.beta,
            "done": self.rung >= self.n_rungs,
            "ess": effective_sample_size(&self.particles),
            "resampled": self.resampled_last,
            "swaps": self.swaps_last,
            "log_evidence": self.log_evidence,
            "particles": pts,
        })
        .to_string()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn transcript(seed: u64) -> String {
        let mut e = ExploreSmcInference::new(200, 10, 1.0, true, seed).unwrap();
        let mut out = e.snapshot();
        for _ in 0..10 {
            out.push_str(&e.step());
        }
        out
    }

    #[test]
    fn smc_inference_is_deterministic() {
        assert_eq!(transcript(7), transcript(7));
        assert_ne!(transcript(7), transcript(8));
    }

    #[test]
    fn smc_inference_ladder_completes_and_shapes_hold() {
        let mut e = ExploreSmcInference::new(150, 8, 1.0, false, 11).unwrap();
        for r in 1..=8 {
            let v: serde_json::Value = serde_json::from_str(&e.step()).unwrap();
            assert_eq!(v["rung"].as_u64().unwrap(), r);
            assert_eq!(v["particles"].as_array().unwrap().len(), 150);
            let ess = v["ess"].as_f64().unwrap();
            assert!((1.0..=150.0 + 1e-9).contains(&ess));
            let beta = v["beta"].as_f64().unwrap();
            assert!((beta - r as f64 / 8.0).abs() < 1e-12);
            for p in v["particles"].as_array().unwrap() {
                assert!(p[0].as_f64().unwrap().is_finite());
                assert!(p[1].as_f64().unwrap().is_finite());
            }
        }
        let v: serde_json::Value = serde_json::from_str(&e.step()).unwrap();
        assert!(v["done"].as_bool().unwrap());
        assert_eq!(v["rung"].as_u64().unwrap(), 8);
    }

    /// The widget shows real inference: at β = 1 the particle population's
    /// weighted mean must match the analytic posterior mean of the twin-peaks
    /// Boltzmann target (computed by grid quadrature of the exact density).
    #[test]
    fn smc_inference_matches_analytic_posterior_mean() {
        // Grid quadrature of π_1 ∝ exp(log_prior + f).
        let n = 400;
        let (lo, hi) = DOMAIN;
        let (mut z, mut ex, mut ey) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..n {
            let y = lo + (hi - lo) * (j as f64 + 0.5) / n as f64;
            for i in 0..n {
                let x = lo + (hi - lo) * (i as f64 + 0.5) / n as f64;
                let d = (log_prior(x, y) + twin_peaks_f(x, y)).exp();
                z += d;
                ex += d * x;
                ey += d * y;
            }
        }
        let (ex, ey) = (ex / z, ey / z);

        let mut e = ExploreSmcInference::new(500, 16, 1.0, true, 42).unwrap();
        let mut last = String::new();
        for _ in 0..16 {
            last = e.step();
        }
        let v: serde_json::Value = serde_json::from_str(&last).unwrap();
        let (mut mx, mut my, mut tw) = (0.0f64, 0.0f64, 0.0f64);
        for p in v["particles"].as_array().unwrap() {
            let w = p[2].as_f64().unwrap();
            mx += w * p[0].as_f64().unwrap();
            my += w * p[1].as_f64().unwrap();
            tw += w;
        }
        assert!((tw - 1.0).abs() < 1e-6, "weights self-normalise");
        let (mx, my) = (mx / tw, my / tw);
        assert!(
            (mx - ex).abs() < 0.3 && (my - ey).abs() < 0.3,
            "particle mean ({mx:.3}, {my:.3}) vs analytic posterior mean ({ex:.3}, {ey:.3})"
        );

        // Log-evidence agrees with the quadrature estimate of
        // log ∫ p(x)·e^f dx / ∫ p(x) dx  (both densities unnormalised the
        // same way, so compare against the ratio).
        let mut zp = 0.0f64;
        for j in 0..n {
            let y = lo + (hi - lo) * (j as f64 + 0.5) / n as f64;
            for i in 0..n {
                let x = lo + (hi - lo) * (i as f64 + 0.5) / n as f64;
                zp += log_prior(x, y).exp();
            }
        }
        let analytic_log_z = (z / zp).ln();
        let got = v["log_evidence"].as_f64().unwrap();
        assert!(
            (got - analytic_log_z).abs() < 0.25,
            "log evidence {got:.3} vs analytic {analytic_log_z:.3}"
        );
    }

    #[test]
    fn smc_inference_annealing_concentrates() {
        // β_max = 6 tightens the population onto the peaks: for a multimodal
        // target the honest concentration measure is the mean distance to the
        // NEAREST mode (total spread stays dominated by the inter-mode
        // distance as long as both peaks keep any mass).
        let run_mode_dist = |beta_max: f64| -> f64 {
            let mut e = ExploreSmcInference::new(300, 24, beta_max, true, 5).unwrap();
            let mut last = String::new();
            for _ in 0..24 {
                last = e.step();
            }
            let v: serde_json::Value = serde_json::from_str(&last).unwrap();
            let pts = v["particles"].as_array().unwrap();
            pts.iter()
                .map(|p| {
                    let (x, y) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
                    MODES
                        .iter()
                        .map(|(m, _, _)| ((x - m[0]).powi(2) + (y - m[1]).powi(2)).sqrt())
                        .fold(f64::INFINITY, f64::min)
                })
                .sum::<f64>()
                / pts.len() as f64
        };
        let posterior = run_mode_dist(1.0);
        let annealed = run_mode_dist(6.0);
        assert!(
            annealed < 0.6 * posterior,
            "annealed mean mode-distance {annealed:.3} should be well below posterior {posterior:.3}"
        );
    }
}
