//! Symbolic regression as **exact Bayesian inference** — the flagship of the
//! "evolutionary algorithms as probabilistic programs" story.
//!
//! The prior over expression trees is a probabilistic context-free grammar
//! written as a fugue program (`ArithmeticGrammarPrior`); the data enter as a
//! Gaussian likelihood factor `f(tree) = −0.5/σ²·Σ(tree(xₖ) − yₖ)²`; and the
//! posterior over programs
//!
//! ```text
//!     π(tree) ∝ p_grammar(tree) · exp(f(tree))
//! ```
//!
//! is sampled by fugue's tempered SMC with two genetic moves that are both
//! *generic trace machinery*:
//!
//! - **subtree regeneration** = fugue's single-site MH: flipping one node's
//!   `#leaf` bit births/kills the subtree below it, fresh structure drawn from
//!   the grammar, reversible-jump corrections applied automatically;
//! - **subtree crossover** = fugue's `CrossoverKernel` with a mask that grafts
//!   the subtrees under one shared node path between two particles.
//!
//! Parsimony comes from the grammar prior (deeper trees pay more mass), the
//! log-evidence is a genuine Bayesian model score (compare function sets!),
//! and the MAP tree is recovered from a bare particle trace by decode-replay.
//!
//! Run with: `cargo run --example symbolic_regression_inference`

use fugue_evo::genome::tree::{ArithmeticFunction, ArithmeticTerminal, TreeGenome};
use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Gaussian log-likelihood of the dataset under a candidate tree.
#[derive(Clone)]
struct SymRegFit {
    xs: Vec<f64>,
    ys: Vec<f64>,
    noise: f64,
}

impl Fitness for SymRegFit {
    type Genome = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;
    type Value = f64;

    fn evaluate(&self, tree: &Self::Genome) -> f64 {
        let sse: f64 = self
            .xs
            .iter()
            .zip(&self.ys)
            .map(|(&x, &y)| {
                let pred = tree.evaluate(&[x]);
                if pred.is_finite() {
                    (pred - y).powi(2)
                } else {
                    1e6 // off-support programs are crushed by the likelihood
                }
            })
            .sum();
        -0.5 * sse / (self.noise * self.noise)
    }
}

type SymRegTree = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;
type SymRegResult = (
    EvolutionPosterior<SymRegTree>,
    EvolutionModel<ArithmeticGrammarPrior, FactorFitness<SymRegFit>>,
);

fn run_inference(rng: &mut StdRng, fitness: &SymRegFit, n_functions: usize) -> SymRegResult {
    let prior = ArithmeticGrammarPrior {
        terminal_prob: 0.35,
        max_depth: 5,
        n_vars: 1,
        p_var: 0.6,
        const_std: 2.0,
        n_functions,
    };
    let model = EvolutionModel::new(prior, fitness.clone());
    let mut kernel = fugue::CrossoverKernel {
        n_pairs: 300,
        mask: subtree_crossover_mask(),
    };
    let posterior = EvolutionSMC::run_with_kernel(
        rng,
        &model,
        EvoSmcConfig {
            num_particles: 800,
            ess_threshold: 0.5,
            resampling: fugue::ResamplingMethod::Systematic,
            rejuvenation_steps: 6,
            crossover: None, // replaced by the explicit subtree kernel
        },
        &mut kernel,
    );
    (posterior, model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Symbolic regression as exact Bayesian inference ===\n");

    // Ground truth: y = x² + 1, observed noiselessly on a grid.
    let xs: Vec<f64> = (-8..=8).map(|i| i as f64 / 4.0).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x * x + 1.0).collect();
    let fitness = SymRegFit {
        xs: xs.clone(),
        ys: ys.clone(),
        noise: 0.25,
    };

    let mut rng = StdRng::seed_from_u64(20260728);

    // -----------------------------------------------------------------
    // Inference with the {Add, Sub, Mul} grammar — sufficient for x²+1.
    // -----------------------------------------------------------------
    let (posterior, model) = run_inference(&mut rng, &fitness, 3);
    let model_fn = model.smc_model();

    let (map_tree, map_fit) = posterior.best(&fitness, &model_fn).unwrap();
    println!("-- Grammar {{Add, Sub, Mul}} --");
    println!("  MAP program:   {}", map_tree.to_sexpr());
    println!("  MAP fitness:   {map_fit:.3}");
    println!("  log evidence:  {:.3}", posterior.log_evidence);

    // Posterior-predictive check at a few held-out points.
    println!("  posterior-weighted predictions vs truth:");
    let decoded = posterior.genomes(&model_fn);
    for &x in &[-1.5, 0.0, 0.5, 1.75] {
        let pred: f64 = decoded
            .iter()
            .map(|(tree, w)| {
                let p = tree.evaluate(&[x]);
                if p.is_finite() {
                    w * p
                } else {
                    0.0
                }
            })
            .sum();
        println!(
            "    x = {x:5.2}:  E[f(x)|data] = {pred:7.3}   truth = {:7.3}",
            x * x + 1.0
        );
    }

    // -----------------------------------------------------------------
    // Bayesian model comparison: a needlessly rich grammar pays for its
    // extra flexibility in evidence.
    // -----------------------------------------------------------------
    let (posterior_rich, _model_rich) = run_inference(&mut rng, &fitness, 12);
    println!("\n-- Model comparison via log-evidence (grammar as hypothesis) --");
    println!(
        "  {{Add, Sub, Mul}}      log Z = {:8.3}",
        posterior.log_evidence
    );
    println!(
        "  full 12-function set log Z = {:8.3}",
        posterior_rich.log_evidence
    );
    println!(
        "  Bayes factor (small vs rich): exp({:.2})",
        posterior.log_evidence - posterior_rich.log_evidence
    );

    Ok(())
}
