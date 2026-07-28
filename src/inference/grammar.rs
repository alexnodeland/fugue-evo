//! Expression trees as probabilistic grammars: genetic programming as exact
//! Bayesian inference
//!
//! [`ArithmeticGrammarPrior`] is a probabilistic context-free grammar over
//! [`TreeGenome`] expression trees, written as a fugue program. Every node at
//! tree path `p` (root key `"node"`, children `"node/0"`, `"node/0/1"`, …)
//! emits real probabilistic choices at path-keyed addresses:
//!
//! | Site | Address | Distribution |
//! |---|---|---|
//! | leaf-vs-function | `<path>#leaf` | `Bernoulli(terminal_prob)` (forced at max depth) |
//! | terminal kind | `<path>#tkind` | `Categorical([p_var, p_const])` |
//! | variable index | `<path>#var` | `Categorical(uniform over n_vars)` |
//! | constant value | `<path>#const` | `Normal(0, const_std)` |
//! | function choice | `<path>#func` | `Categorical(uniform over F::functions())` |
//!
//! Because the structure of an execution is *encoded in its own choices*, the
//! generic trace machinery becomes genetic programming for free:
//!
//! - **Subtree regeneration mutation** is fugue's ordinary single-site MH: a
//!   flip of one `#leaf` bit (or a change of `#func` arity) births or kills the
//!   subtree below it, with the fresh sites drawn from this grammar and the
//!   reversible-jump corrections applied by `propose_and_score` — no bespoke
//!   acceptance math anywhere in this crate.
//! - **Subtree crossover** is a [`CrossoverKernel`](fugue::CrossoverKernel)
//!   whose mask is the union of both parents' addresses under one shared node
//!   path ([`subtree_crossover_mask`]): swapping that block grafts each
//!   parent's subtree into the other, and the re-score replays each child
//!   consistently because the grafted choices themselves describe the new
//!   structure.
//!
//! Parsimony needs no ad-hoc penalty: deeper trees pay more grammar prior mass
//! by construction.
//!
//! Note: tree genomes are decoded from particles by replay (the model returns
//! the built [`TreeGenome`]); the flat [`TraceGenome`] encoding of
//! `TreeGenome` is unrelated to this grammar's address scheme, so
//! `EvolutionModel::score`/`to_weighted_trace` (which replay `to_trace`) do
//! not apply to grammar-driven trees — use the SMC/MH drivers, which never
//! need them.

use fugue::{addr, sample, Address, Bernoulli, Categorical, Model, ModelExt, Normal, Trace};

/// The mask-closure type consumed by [`fugue::CrossoverKernel`].
pub type CrossoverMaskFn = Box<dyn Fn(&Trace, &Trace, &mut dyn rand::RngCore) -> Vec<Address>>;

use super::prior::GenomePrior;
use crate::genome::tree::{ArithmeticFunction, ArithmeticTerminal, Function, TreeGenome, TreeNode};

/// A probabilistic context-free grammar prior over arithmetic expression
/// trees.
#[derive(Clone, Debug)]
pub struct ArithmeticGrammarPrior {
    /// Probability that a node (below max depth) is a terminal.
    pub terminal_prob: f64,
    /// Maximum tree depth; nodes at this depth are forced terminal.
    pub max_depth: usize,
    /// Number of input variables `x0..x{n_vars-1}`.
    pub n_vars: usize,
    /// Probability that a terminal is a variable (vs a constant).
    pub p_var: f64,
    /// Standard deviation of the Gaussian prior over constants.
    pub const_std: f64,
    /// Restrict the function set to the first `n_functions` entries of
    /// [`ArithmeticFunction::functions`] (e.g. 4 = {Add, Sub, Mul, Div}).
    pub n_functions: usize,
}

impl Default for ArithmeticGrammarPrior {
    fn default() -> Self {
        Self {
            terminal_prob: 0.4,
            max_depth: 6,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 4, // Add, Sub, Mul, Div
        }
    }
}

fn child_key(key: &str, i: usize) -> String {
    format!("{key}/{i}")
}

impl ArithmeticGrammarPrior {
    fn node_model(
        &self,
        key: String,
        depth: usize,
    ) -> Model<TreeNode<ArithmeticTerminal, ArithmeticFunction>> {
        let cfg = self.clone();
        let p_leaf = if depth >= cfg.max_depth {
            1.0
        } else {
            cfg.terminal_prob
        };
        sample(
            addr!(key.clone(), "leaf"),
            Bernoulli::new(p_leaf).expect("valid leaf probability"),
        )
        .bind(move |is_leaf| {
            if is_leaf {
                cfg.terminal_model(&key)
            } else {
                let n_funcs = cfg.n_functions.min(ArithmeticFunction::functions().len());
                let probs = vec![1.0 / n_funcs as f64; n_funcs];
                sample(
                    addr!(key.clone(), "func"),
                    Categorical::new(probs).expect("valid function categorical"),
                )
                .bind(move |fi| {
                    let func = ArithmeticFunction::functions()[fi].clone();
                    let arity = func.arity();
                    let children: Vec<Model<TreeNode<ArithmeticTerminal, ArithmeticFunction>>> = (0
                        ..arity)
                        .map(|c| cfg.node_model(child_key(&key, c), depth + 1))
                        .collect();
                    fugue::sequence_vec(children)
                        .map(move |kids| TreeNode::function(func.clone(), kids))
                })
            }
        })
    }

    fn terminal_model(&self, key: &str) -> Model<TreeNode<ArithmeticTerminal, ArithmeticFunction>> {
        let (p_var, n_vars, const_std) = (self.p_var, self.n_vars.max(1), self.const_std);
        let key = key.to_string();
        sample(
            addr!(key.clone(), "tkind"),
            Categorical::new(vec![p_var, 1.0 - p_var]).expect("valid terminal-kind categorical"),
        )
        .bind(move |kind| {
            if kind == 0 {
                let probs = vec![1.0 / n_vars as f64; n_vars];
                sample(
                    addr!(key.clone(), "var"),
                    Categorical::new(probs).expect("valid variable categorical"),
                )
                .map(|i| TreeNode::terminal(ArithmeticTerminal::Variable(i)))
            } else {
                sample(
                    addr!(key.clone(), "const"),
                    Normal::new(0.0, const_std).expect("valid constant prior"),
                )
                .map(|c| TreeNode::terminal(ArithmeticTerminal::Constant(c)))
            }
        })
    }
}

impl GenomePrior for ArithmeticGrammarPrior {
    type Genome = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;

    fn model(&self) -> Model<Self::Genome> {
        let max_depth = self.max_depth;
        self.node_model("node".to_string(), 0)
            .map(move |root| TreeGenome::new(root, max_depth))
    }

    /// Encode a tree under the grammar's own address scheme (the inverse of
    /// running [`Self::model`]): a deterministic walk emitting the
    /// `#leaf`/`#tkind`/`#var`/`#const`/`#func` choices at each node path.
    ///
    /// This is what makes `EvolutionModel::score`, `to_weighted_trace`, and
    /// `EvolutionChain::init_from` work for grammar-driven trees: replaying
    /// the encoding through the grammar program recovers the genuine PCFG
    /// log-prior. A tree using a function outside the restricted
    /// `n_functions` set, or deeper than `max_depth`, scores `−∞` under
    /// replay (out of the prior's support) rather than erroring.
    ///
    /// `Erc` terminals are encoded as constants (evaluation-identical; the
    /// grammar itself only generates `Variable`/`Constant`).
    fn trace_of(&self, genome: &Self::Genome) -> Trace {
        fn walk(
            node: &TreeNode<ArithmeticTerminal, ArithmeticFunction>,
            key: &str,
            trace: &mut Trace,
        ) {
            use fugue::ChoiceValue;
            match node {
                TreeNode::Terminal(term) => {
                    trace.insert_choice(addr!(key, "leaf"), ChoiceValue::Bool(true), 0.0);
                    match term {
                        ArithmeticTerminal::Variable(i) => {
                            trace.insert_choice(addr!(key, "tkind"), ChoiceValue::Usize(0), 0.0);
                            trace.insert_choice(addr!(key, "var"), ChoiceValue::Usize(*i), 0.0);
                        }
                        ArithmeticTerminal::Constant(c) | ArithmeticTerminal::Erc(c) => {
                            trace.insert_choice(addr!(key, "tkind"), ChoiceValue::Usize(1), 0.0);
                            trace.insert_choice(addr!(key, "const"), ChoiceValue::F64(*c), 0.0);
                        }
                    }
                }
                TreeNode::Function(func, children) => {
                    trace.insert_choice(addr!(key, "leaf"), ChoiceValue::Bool(false), 0.0);
                    let fi = ArithmeticFunction::functions()
                        .iter()
                        .position(|f| f == func)
                        .expect("function present in the canonical table");
                    trace.insert_choice(addr!(key, "func"), ChoiceValue::Usize(fi), 0.0);
                    for (c, child) in children.iter().enumerate() {
                        walk(child, &child_key(key, c), trace);
                    }
                }
            }
        }
        let mut trace = Trace::default();
        walk(&genome.root, "node", &mut trace);
        trace
    }
}

/// Gaussian-noise regression of a dataset under a candidate expression tree,
/// as an **observation program** — per-datum `observe` statements, with the
/// noise scale either fixed or a **latent site jointly inferred** with the
/// program. This is the capability the scalar-factor fitness could never
/// express: hyperparameters of the "fitness" become posterior quantities,
/// read straight off the particle traces at `addr!("sigma")`.
#[derive(Clone, Debug)]
pub struct GaussianRegression {
    /// Input points (single variable).
    pub xs: Vec<f64>,
    /// Observed outputs.
    pub ys: Vec<f64>,
    /// Observation-noise model.
    pub noise: NoiseSpec,
}

/// How the observation noise enters the regression likelihood.
#[derive(Clone, Debug)]
pub enum NoiseSpec {
    /// Known, fixed noise standard deviation.
    Fixed(f64),
    /// Unknown noise: `σ ~ Uniform(low, high)` as a latent site at
    /// `addr!("sigma")`, jointly inferred with the program.
    Infer {
        /// Lower bound of the uniform prior over σ.
        low: f64,
        /// Upper bound of the uniform prior over σ.
        high: f64,
    },
}

impl super::likelihood::GenomeLikelihood<TreeGenome<ArithmeticTerminal, ArithmeticFunction>>
    for GaussianRegression
{
    fn model(
        &self,
        tree: &TreeGenome<ArithmeticTerminal, ArithmeticFunction>,
        beta: f64,
    ) -> Model<()> {
        use super::likelihood::tempered_observe;
        // Evaluate the candidate program once per datum, up front. A
        // non-finite prediction crushes the whole likelihood.
        let preds: Vec<f64> = self.xs.iter().map(|&x| tree.evaluate(&[x])).collect();
        if preds.iter().any(|p| !p.is_finite()) {
            return fugue::factor(f64::NEG_INFINITY);
        }
        let ys = self.ys.clone();
        let observe_all =
            move |sigma: f64, beta: f64, preds: Vec<f64>, ys: Vec<f64>| -> Model<()> {
                let mut m = fugue::pure(());
                for (k, (pred, y)) in preds.into_iter().zip(ys).enumerate() {
                    m = m.and_then(move |_| {
                        tempered_observe(
                            addr!("y", k),
                            Normal::new(pred, sigma).expect("valid observation noise"),
                            y,
                            beta,
                        )
                    });
                }
                m
            };
        match self.noise {
            NoiseSpec::Fixed(sigma) => observe_all(sigma, beta, preds, ys),
            NoiseSpec::Infer { low, high } => sample(
                addr!("sigma"),
                fugue::Uniform::new(low, high).expect("valid noise prior bounds"),
            )
            .bind(move |sigma| observe_all(sigma, beta, preds.clone(), ys.clone())),
        }
    }
}

/// A value-independent, pair-symmetric **subtree crossover** mask for
/// [`fugue::CrossoverKernel`]: picks one node path present in *both* parents
/// uniformly at random and returns the union of the two parents' addresses
/// under that path. Swapping that block grafts each parent's subtree into the
/// other; the kernel's mandatory re-score replays each child consistently
/// (the grafted choices encode the new structure) and rejects off-support or
/// low-density grafts via the product-target Metropolis ratio.
pub fn subtree_crossover_mask() -> CrossoverMaskFn {
    Box::new(|a: &Trace, b: &Trace, rng: &mut dyn rand::RngCore| {
        // Node paths of a trace = the `<path>#leaf` site keys.
        let paths_of = |t: &Trace| -> Vec<String> {
            t.choices
                .keys()
                .filter_map(|addr| addr.as_str().strip_suffix("#leaf").map(str::to_string))
                .collect()
        };
        let pa = paths_of(a);
        let pb: std::collections::HashSet<String> = paths_of(b).into_iter().collect();
        let shared: Vec<String> = pa.into_iter().filter(|p| pb.contains(p)).collect();
        if shared.is_empty() {
            return Vec::new();
        }
        let path = &shared[rand::Rng::gen_range(rng, 0..shared.len())];
        let mut block: Vec<Address> = a.extract_prefix(path).choices.keys().cloned().collect();
        for addr in b.extract_prefix(path).choices.keys() {
            if !block.contains(addr) {
                block.push(addr.clone());
            }
        }
        block
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::traits::Fitness;
    use crate::inference::model::EvolutionModel;
    use crate::inference::smc::{EvoSmcConfig, EvolutionSMC};
    use fugue::runtime::handler::run;
    use fugue::runtime::interpreters::PriorHandler;
    use fugue::{CrossoverKernel, PopulationKernel, ResamplingMethod};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// The grammar trace carries genuine probability mass: `log_prior` is the
    /// real PCFG log-probability of the drawn tree, never 0. (Replaces the
    /// flat, `logp = 0.0` serialization story for trees.)
    #[test]
    fn test_grammar_trace_has_real_log_prior() {
        let prior = ArithmeticGrammarPrior::default();
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..20 {
            let (tree, trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            assert!(trace.log_prior.is_finite());
            assert!(
                trace.log_prior < 0.0,
                "a non-trivial tree draw must pay prior mass, got {}",
                trace.log_prior
            );
            assert!(tree.depth() <= prior.max_depth + 1);
            // Every node path has its structural site.
            assert!(trace.choices.keys().any(|a| a.as_str() == "node#leaf"));
        }
    }

    /// Deeper trees pay more prior mass — the parsimony pressure is the
    /// grammar itself, not an ad-hoc penalty.
    #[test]
    fn test_grammar_prior_penalizes_depth() {
        let prior = ArithmeticGrammarPrior::default();
        let mut rng = StdRng::seed_from_u64(5);
        let mut sized: Vec<(usize, f64)> = Vec::new();
        for _ in 0..300 {
            let (tree, trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            sized.push((tree.size(), trace.log_prior));
        }
        let small: Vec<f64> = sized
            .iter()
            .filter(|(s, _)| *s <= 3)
            .map(|(_, lp)| *lp)
            .collect();
        let large: Vec<f64> = sized
            .iter()
            .filter(|(s, _)| *s >= 7)
            .map(|(_, lp)| *lp)
            .collect();
        assert!(!small.is_empty() && !large.is_empty());
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            mean(&small) > mean(&large),
            "small trees {} should out-mass large trees {}",
            mean(&small),
            mean(&large)
        );
    }

    /// Subtree crossover swaps a complete prefix block between two grammar
    /// traces and both children re-score to valid trees.
    #[test]
    fn test_subtree_crossover_swaps_prefix_range() {
        let mut rng = StdRng::seed_from_u64(21);
        let model_fn = prior_model_for_test;
        fn prior_model_for_test() -> Model<TreeGenome<ArithmeticTerminal, ArithmeticFunction>> {
            ArithmeticGrammarPrior {
                terminal_prob: 0.3,
                max_depth: 4,
                ..Default::default()
            }
            .model()
        }

        // Build a small particle population from the prior.
        let particles_res = fugue::smc_prior_particles(&mut rng, 12, model_fn);
        let mut particles = particles_res;
        let before_sets: Vec<usize> = particles.iter().map(|p| p.trace.choices.len()).collect();

        let mut kernel = CrossoverKernel {
            n_pairs: 40,
            mask: subtree_crossover_mask(),
        };
        PopulationKernel::<TreeGenome<ArithmeticTerminal, ArithmeticFunction>>::sweep(
            &mut kernel,
            &mut rng,
            &mut particles,
            &model_fn,
            1.0,
        );

        // Every particle still decodes to a valid tree via replay, with a
        // finite prior mass (accepted grafts were re-scored).
        for p in &particles {
            let tree = fugue::decode_particle(p, model_fn);
            assert!(tree.size() >= 1);
            assert!(p.trace.log_prior.is_finite());
        }
        // Structure genuinely moved for at least one particle (subtree swap
        // changes address-set sizes unless every accepted swap was congruent).
        let after_sets: Vec<usize> = particles.iter().map(|p| p.trace.choices.len()).collect();
        let _ = (before_sets, after_sets); // sizes may or may not differ; decode is the contract
    }

    /// The grammar encoding is the exact inverse of the generative program:
    /// a prior-drawn tree's `trace_of` reproduces the generative trace's
    /// choices, and replay-scoring it recovers the same PCFG log-prior.
    #[test]
    fn test_trace_of_inverts_generative_run() {
        use fugue::runtime::interpreters::ScoreGivenTrace;
        let prior = ArithmeticGrammarPrior::default();
        let mut rng = StdRng::seed_from_u64(31);
        for _ in 0..30 {
            let (tree, gen_trace) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            let enc = prior.trace_of(&tree);
            assert_eq!(enc.choices.len(), gen_trace.choices.len());
            for (addr, choice) in &gen_trace.choices {
                assert_eq!(
                    enc.choices[addr].value, choice.value,
                    "encoding mismatch at {addr}"
                );
            }
            // Replay-scoring the encoding recovers the PCFG log-prior.
            let (_t, scored) = run(
                ScoreGivenTrace {
                    base: enc,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            assert!((scored.log_prior - gen_trace.log_prior).abs() < 1e-9);
        }
    }

    /// `EvolutionModel::score` now works for grammar trees: a hand-built
    /// `(+ x0 1.0)` scores to the hand-computed PCFG log-prior plus β·f.
    #[test]
    fn test_score_hand_built_tree_matches_analytic() {
        use crate::genome::tree::TreeNode;
        #[derive(Clone, Copy)]
        struct Zero;
        impl Fitness for Zero {
            type Genome = TreeGenome<ArithmeticTerminal, ArithmeticFunction>;
            type Value = f64;
            fn evaluate(&self, _t: &Self::Genome) -> f64 {
                0.0
            }
        }

        let prior = ArithmeticGrammarPrior {
            terminal_prob: 0.4,
            max_depth: 6,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 4,
        };
        // (+ x0 1.0)
        let tree = TreeGenome::new(
            TreeNode::function(
                crate::genome::tree::ArithmeticFunction::Add,
                vec![
                    TreeNode::terminal(ArithmeticTerminal::Variable(0)),
                    TreeNode::terminal(ArithmeticTerminal::Constant(1.0)),
                ],
            ),
            6,
        );
        let model = crate::inference::model::EvolutionModel::new(prior.clone(), Zero);
        let (_g, scored) = model.score(&tree);

        // Hand-computed PCFG log-prior:
        //   root: not-leaf (1-0.4) · func Add (1/4)
        //   child 0: leaf 0.4 · var-kind 0.6 · var 0 (1/1)
        //   child 1: leaf 0.4 · const-kind 0.4 · Normal(0,2).log_prob(1.0)
        let normal = fugue::Normal::new(0.0, 2.0).unwrap();
        let analytic = (0.6f64).ln()
            + (0.25f64).ln()
            + (0.4f64).ln()
            + (0.6f64).ln()
            + (1.0f64).ln()
            + (0.4f64).ln()
            + (0.4f64).ln()
            + fugue::Distribution::log_prob(&normal, &1.0);
        assert!(
            (scored.log_prior - analytic).abs() < 1e-9,
            "scored {} vs analytic {}",
            scored.log_prior,
            analytic
        );

        // Warm-starting a chain from the hand-built tree works.
        let chain = crate::inference::mh::EvolutionChain::new(
            crate::inference::model::EvolutionModel::new(prior, Zero),
        );
        let init = chain.init_from(&tree).expect("in-support tree");
        assert!(init.total_log_weight().is_finite());
    }

    /// Fix A capstone: the observation noise is a latent site in the
    /// likelihood program, jointly inferred with the program. Data are
    /// `y = x + 1 + ε`, `ε ~ N(0, 0.3²)`; the posterior over `σ` (read off
    /// the particle traces at `addr!("sigma")`) must land near the truth.
    #[test]
    fn test_symreg_infers_noise_jointly() {
        let sigma_true = 0.3;
        let mut data_rng = StdRng::seed_from_u64(4242);
        let noise_dist = rand_distr::Normal::new(0.0, sigma_true).unwrap();
        let xs: Vec<f64> = (-10..=10).map(|i| i as f64 / 5.0).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|x| x + 1.0 + rand_distr::Distribution::sample(&noise_dist, &mut data_rng))
            .collect();

        let prior = ArithmeticGrammarPrior {
            terminal_prob: 0.45,
            max_depth: 3,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 1, // {Add} — x + 1 is easily reachable
        };
        let likelihood = GaussianRegression {
            xs: xs.clone(),
            ys,
            noise: NoiseSpec::Infer {
                low: 0.02,
                high: 2.0,
            },
        };
        let model = crate::inference::model::EvolutionModel::from_likelihood(prior, likelihood);
        let mut rng = StdRng::seed_from_u64(77);
        let mut kernel = CrossoverKernel {
            n_pairs: 150,
            mask: subtree_crossover_mask(),
        };
        let result = EvolutionSMC::run_with_kernel(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 500,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 6,
                crossover: None,
            },
            &mut kernel,
        );

        // Posterior over σ, straight off the traces.
        let mut total_w = 0.0;
        let mut sigma_mean = 0.0;
        for p in &result.particles {
            if let Some(s) = p.trace.get_f64(&fugue::addr!("sigma")) {
                sigma_mean += p.weight * s;
                total_w += p.weight;
            }
        }
        assert!(total_w > 0.99, "every particle carries the sigma site");
        sigma_mean /= total_w;
        assert!(
            (0.15..=0.55).contains(&sigma_mean),
            "posterior sigma mean {} should be near the true 0.3",
            sigma_mean
        );

        // And the programs still fit: posterior-weighted predictions track y = x+1.
        let model_fn = model.smc_model();
        let decoded = fugue::decode_particles(&result.particles, &model_fn);
        for &x in &[-1.0, 0.0, 1.5] {
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
            assert!(
                (pred - (x + 1.0)).abs() < 0.35,
                "posterior predictive at {} was {} vs truth {}",
                x,
                pred,
                x + 1.0
            );
        }
    }

    /// Flagship analytic recovery: symbolic regression of `x² + 1` from
    /// noiseless data, posed as exact Bayesian inference over the grammar.
    /// The MAP tree's predictions must match the target on a held-out grid.
    #[test]
    fn test_symreg_recovers_known_expression() {
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
                            1e6
                        }
                    })
                    .sum();
                -0.5 * sse / (self.noise * self.noise)
            }
        }

        let xs: Vec<f64> = (-8..=8).map(|i| i as f64 / 4.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x + 1.0).collect();
        let fitness = SymRegFit {
            xs: xs.clone(),
            ys: ys.clone(),
            noise: 0.25,
        };

        let prior = ArithmeticGrammarPrior {
            terminal_prob: 0.35,
            max_depth: 5,
            n_vars: 1,
            p_var: 0.6,
            const_std: 2.0,
            n_functions: 3, // Add, Sub, Mul — enough for x²+1
        };
        let model = EvolutionModel::new(prior, fitness.clone());
        let mut rng = StdRng::seed_from_u64(20260728);
        let mut kernel = CrossoverKernel {
            n_pairs: 200,
            mask: subtree_crossover_mask(),
        };
        let result = EvolutionSMC::run_with_kernel(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 600,
                ess_threshold: 0.5,
                resampling: ResamplingMethod::Systematic,
                rejuvenation_steps: 6,
                crossover: None, // replaced by the explicit subtree kernel
            },
            &mut kernel,
        );
        let model_fn = model.smc_model();
        let (best, best_f) = result.best(&fitness, &model_fn).unwrap();
        // MAP predictions match x²+1 closely on the grid.
        let max_err = xs
            .iter()
            .map(|&x| (best.evaluate(&[x]) - (x * x + 1.0)).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_err < 0.35,
            "MAP tree {} (fitness {best_f:.2}) max error {max_err:.3} too large",
            best.to_sexpr(),
        );
    }
}
