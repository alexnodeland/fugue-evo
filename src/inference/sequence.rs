//! Typed sequences as probabilistic grammars: variable-length structure
//! search as exact Bayesian inference
//!
//! [`SequenceGrammarPrior`] is the sequence counterpart of
//! [`ArithmeticGrammarPrior`](super::grammar::ArithmeticGrammarPrior): a prior
//! over **variable-length sequences whose set of legal next symbols depends
//! on the prefix**, written as a fugue program. The user describes the
//! language with a [`SequenceGrammar`]: a state type, an initial state, the
//! finite set of typed choices available in a state (grouped into *kinds*,
//! each with its *arguments*, optionally weighted), a transition, and a stop
//! rule. Typical uses are sequences of typed edits or moves, where a move is
//! only available when it type-checks against what the earlier moves built.
//!
//! Position `i` of a sequence emits up to three choices at systematic
//! addresses (prefix `seq` by default, see
//! [`SequenceGrammarPrior::with_prefix`]):
//!
//! | Site | Address | Distribution |
//! |---|---|---|
//! | stop here | `seq/<i>#stop` | `Bernoulli(stop_probability(state))` |
//! | kind of the symbol | `seq/<i>#kind` | `Categorical` over the kinds available in `state` |
//! | argument of that kind | `seq/<i>#arg` | `Categorical` over that kind's arguments |
//!
//! A choice with **one** outcome emits no site: a stop probability of `0` or
//! `1`, a state with a single kind, a kind with a single argument. The stop
//! is also forced, with no site, when the state offers no choice at all and
//! at `max_len`. Such a site would carry no probability mass and would only
//! dilute the single-site sampler's proposals.
//!
//! # The prior is a description length
//!
//! The log-density of a sequence is the sum of the log-probabilities of the
//! choices that derive it. With uniform weights that is
//! `−Σ ln(number of alternatives)`: the length, in nats, of the shortest
//! code for the sequence under the grammar
//! ([`SequenceGrammarPrior::description_length`]). Longer and more specific
//! sequences pay more mass by construction, with no separate complexity
//! penalty. As everywhere in this layer the density is obtained by replaying
//! the program, never from hand-written density code.
//!
//! # Inference moves
//!
//! Because an execution's structure is encoded in its own choices, the
//! generic trace machinery searches the language directly:
//!
//! - **Edit, extend, truncate** are fugue's ordinary single-site MH
//!   ([`EvolutionChain`](super::mh::EvolutionChain) and the per-particle
//!   rejuvenation of [`EvolutionSMC`](super::smc::EvolutionSMC)). Flipping a
//!   `#stop` bit to `false` extends the sequence with fresh sites drawn from
//!   this grammar; flipping it to `true` truncates. Resampling a `#kind` or
//!   `#arg` changes one symbol and replays the later choices against the new
//!   prefix. This is the same mechanism the tree grammar relies on, and the
//!   reversible-jump correction is fugue's, applied by its
//!   `propose_and_score` with no acceptance code in this crate. The
//!   acceptance ratio is
//!
//!   ```text
//!   ln α = Δ ln π  +  ln q_rev − ln q_fwd  +  ln |sites(x)| − ln |sites(x′)|
//!   ```
//!
//!   where `q_fwd` is the proposal density of the chosen site times the prior
//!   density of every site **born** by the move (sites the new structure
//!   visits that the current trace lacks, drawn from the prior), `q_rev` is
//!   the reverse proposal density times the stored prior density of every
//!   site that **died** (present in the current trace, not visited any
//!   more), and the last term is the change in the probability of selecting
//!   that site uniformly among all sites. The Jacobian is 1. A replayed
//!   choice that is out of range for its new support scores `−∞` and the
//!   move is rejected.
//! - **Suffix crossover** is [`PrefixKeyedSuffixCrossover`], a
//!   [`fugue::PopulationKernel`]: it exchanges the tails of two particles at
//!   cut points whose prefix states have equal user-supplied keys, so the
//!   offspring are well typed.
//!
//! `tests::test_mh_with_flat_likelihood_recovers_the_prior` and its siblings
//! check these claims against exact enumeration of a small grammar.
//!
//! # Example
//!
//! ```rust
//! use fugue_evo::inference::likelihood::NoLikelihood;
//! use fugue_evo::inference::model::EvolutionModel;
//! use fugue_evo::inference::prior::GenomePrior;
//! use fugue_evo::inference::sequence::{
//!     ChoiceKind, SequenceGenome, SequenceGrammar, SequenceGrammarPrior,
//! };
//! use serde::{Deserialize, Serialize};
//!
//! /// Balanced-so-far bracket strings: `Close` needs an open bracket.
//! #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
//! enum Tok {
//!     Open,
//!     Close,
//! }
//!
//! struct Brackets;
//!
//! impl SequenceGrammar for Brackets {
//!     type State = usize; // open brackets so far
//!     type Symbol = Tok;
//!
//!     fn initial_state(&self) -> usize {
//!         0
//!     }
//!     fn choices(&self, depth: &usize) -> Vec<ChoiceKind<Tok>> {
//!         let mut toks = vec![Tok::Open];
//!         if *depth > 0 {
//!             toks.push(Tok::Close);
//!         }
//!         vec![ChoiceKind::uniform(toks)]
//!     }
//!     fn transition(&self, depth: &usize, tok: &Tok) -> usize {
//!         match tok {
//!             Tok::Open => depth + 1,
//!             Tok::Close => depth - 1,
//!         }
//!     }
//!     fn stop_probability(&self, _depth: &usize) -> f64 {
//!         0.5
//!     }
//! }
//!
//! let prior = SequenceGrammarPrior::new(Brackets, 8);
//!
//! // `( )`: continue, Open (the only choice, no site), continue, Close
//! // (one of two), stop. Three fair coins and one of two symbols.
//! let seq = SequenceGenome::new(vec![Tok::Open, Tok::Close]);
//! let dl = prior.description_length(&seq).unwrap();
//! assert!((dl - 4.0 * 2.0f64.ln()).abs() < 1e-12);
//!
//! // `)` is not in the language: rejected before any scoring.
//! assert!(prior.validate(&SequenceGenome::new(vec![Tok::Close])).is_err());
//!
//! // The prior drives the existing samplers like any other.
//! let model = EvolutionModel::from_likelihood(prior, NoLikelihood);
//! let mut rng = rand::thread_rng();
//! let draw: SequenceGenome<Tok> = model.sample_prior(&mut rng);
//! assert!(model.prior().validate(&draw).is_ok());
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use fugue::{
    pure, sample, Address, Bernoulli, Categorical, ChoiceValue, Model, ModelExt, Particle,
    PopulationKernel, Trace,
};
use rand::Rng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::model::score_complete;
use super::prior::GenomePrior;
use super::smc::rescore_complete;
use crate::error::GenomeError;
use crate::genome::trace_genome::ReplayGenome;

/// A variable-length sequence of symbols: the genome of a
/// [`SequenceGrammarPrior`].
///
/// It is a [`ReplayGenome`]: generated by the grammar program, decoded from
/// particles by replay, encoded by the prior's
/// [`trace_of`](GenomePrior::trace_of). It carries none of the classical
/// operator surface.
///
/// ```rust
/// use fugue_evo::inference::sequence::SequenceGenome;
///
/// let s = SequenceGenome::new(vec!['a', 'b']);
/// assert_eq!(s.len(), 2);
/// assert_eq!(s.symbols(), &['a', 'b']);
/// assert_eq!(s.into_symbols(), vec!['a', 'b']);
/// assert!(SequenceGenome::<char>::default().is_empty());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceGenome<S>(Vec<S>);

impl<S> SequenceGenome<S> {
    /// Wrap a vector of symbols.
    pub fn new(symbols: Vec<S>) -> Self {
        Self(symbols)
    }

    /// The symbols, in order.
    pub fn symbols(&self) -> &[S] {
        &self.0
    }

    /// Unwrap into the vector of symbols.
    pub fn into_symbols(self) -> Vec<S> {
        self.0
    }

    /// Number of symbols.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` for the empty sequence.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<S> Default for SequenceGenome<S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<S> From<Vec<S>> for SequenceGenome<S> {
    fn from(symbols: Vec<S>) -> Self {
        Self(symbols)
    }
}

impl<S> ReplayGenome for SequenceGenome<S>
where
    S: Clone + Send + Sync + Serialize + DeserializeOwned + 'static,
{
    fn replay_dimension(&self) -> usize {
        self.0.len()
    }
}

/// One *kind* of next symbol together with its admissible *arguments*: the
/// two-level choice a [`SequenceGrammar`] offers in a state.
///
/// The kind is drawn first (`#kind`, by [`Self::weight`]), then one of its
/// arguments (`#arg`, by the per-argument weights); each argument is a
/// complete symbol. Weights are relative: they are normalised over the kinds
/// of a state and over the arguments of a kind, and must be finite and
/// strictly positive. A grammar with a flat choice returns a single kind
/// (then only `#arg` is emitted).
///
/// ```rust
/// use fugue_evo::inference::sequence::ChoiceKind;
///
/// // Two equally likely arguments, kind weight 1.
/// let k = ChoiceKind::uniform(vec!["x", "y"]);
/// assert_eq!(k.weight, 1.0);
/// assert_eq!(k.args, vec![("x", 1.0), ("y", 1.0)]);
///
/// // A kind three times as likely as its siblings, with weighted arguments.
/// let k = ChoiceKind::weighted(vec![("x", 0.9), ("y", 0.1)]).with_weight(3.0);
/// assert_eq!(k.weight, 3.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceKind<S> {
    /// Relative weight of this kind among the kinds of the state.
    pub weight: f64,
    /// The symbols of this kind, each with its relative weight.
    pub args: Vec<(S, f64)>,
}

impl<S> ChoiceKind<S> {
    /// A kind of weight 1 whose arguments are equally likely.
    pub fn uniform(symbols: impl IntoIterator<Item = S>) -> Self {
        Self {
            weight: 1.0,
            args: symbols.into_iter().map(|s| (s, 1.0)).collect(),
        }
    }

    /// A kind of weight 1 with explicitly weighted arguments.
    pub fn weighted(args: impl IntoIterator<Item = (S, f64)>) -> Self {
        Self {
            weight: 1.0,
            args: args.into_iter().collect(),
        }
    }

    /// Set the relative weight of the kind.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// A language of typed sequences, described by a state machine with
/// prefix-dependent supports. See the [module documentation](self) for an
/// example.
///
/// All four functions must be **deterministic**: the prior program is
/// re-executed on every replay, and a trace only decodes to the same
/// sequence if the same state offers the same choices in the same order.
///
/// A state should offer each symbol at most once. If a symbol appears under
/// two kinds (or twice in one), the program's density is that of one
/// derivation, and [`GenomePrior::trace_of`] encodes the first.
///
/// # Rejected proposals and the likelihood
///
/// When a sampler replays a choice that is out of range for its new support
/// (a proposal that changed an earlier symbol), the site scores `−∞` and the
/// program ends there, returning the symbols read so far. The likelihood
/// program is still run on that return value before the proposal is
/// rejected. Every symbol of it was legal after its prefix, but the
/// sequence may end where the grammar would not stop, so a likelihood must
/// tolerate any such prefix of a sentence (score it, or declare it invalid
/// with [`FactorInput::Invalid`](super::likelihood::FactorInput::Invalid)).
pub trait SequenceGrammar: Send + Sync + 'static {
    /// What a prefix is remembered by: whatever determines which symbols may
    /// follow (a type environment, a stack depth, a partially built object).
    type State: Clone + Send + 'static;

    /// One element of the sequence.
    type Symbol: Clone + PartialEq + Send + Sync + Serialize + DeserializeOwned + 'static;

    /// The state of the empty prefix.
    fn initial_state(&self) -> Self::State;

    /// The symbols that may come next in `state`, grouped by kind. An empty
    /// vector (or only kinds with no arguments, which are ignored) ends the
    /// sequence there.
    fn choices(&self, state: &Self::State) -> Vec<ChoiceKind<Self::Symbol>>;

    /// The state after appending `symbol` in `state`. Only called with a
    /// symbol that [`Self::choices`] offered for `state`.
    fn transition(&self, state: &Self::State, symbol: &Self::Symbol) -> Self::State;

    /// Probability, in `[0, 1]`, that the sequence ends in `state`. `0`
    /// means it must continue and `1` that it must end; neither emits a
    /// site.
    fn stop_probability(&self, state: &Self::State) -> f64;
}

/// Which of a position's three choices an address names. See
/// [`SequenceGrammarPrior::site_of`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SequenceSite {
    /// `<prefix>/<i>#stop`
    Stop,
    /// `<prefix>/<i>#kind`
    Kind,
    /// `<prefix>/<i>#arg`
    Arg,
}

impl SequenceSite {
    fn suffix(self) -> &'static str {
        match self {
            SequenceSite::Stop => "stop",
            SequenceSite::Kind => "kind",
            SequenceSite::Arg => "arg",
        }
    }
}

/// The normalised kinds of one state: `(probability, [(symbol, probability)])`.
type Kinds<S> = Vec<(f64, Vec<(S, f64)>)>;

/// How a position decides whether the sequence ends.
enum StopRule {
    /// Ends here; no site.
    Forced,
    /// Cannot end here; no site.
    Never,
    /// A `Bernoulli(p)` site with `0 < p < 1`.
    Site(f64),
}

/// Why a sequence has no derivation, and at which position.
struct Underivable {
    position: usize,
    reason: String,
}

/// A grammar prior over variable-length typed sequences, as a fugue
/// program. See the [module documentation](self).
///
/// `max_len` bounds every sequence: at that length the stop is forced. It
/// also bounds every replay, including the replay of an incompatible trace,
/// so keep it at the scale of the sequences you expect.
pub struct SequenceGrammarPrior<Gr> {
    grammar: Arc<Gr>,
    max_len: usize,
    prefix: Arc<str>,
}

impl<Gr> Clone for SequenceGrammarPrior<Gr> {
    fn clone(&self) -> Self {
        Self {
            grammar: Arc::clone(&self.grammar),
            max_len: self.max_len,
            prefix: Arc::clone(&self.prefix),
        }
    }
}

impl<Gr> std::fmt::Debug for SequenceGrammarPrior<Gr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceGrammarPrior")
            .field("max_len", &self.max_len)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

fn normalise(weights: impl Iterator<Item = f64> + Clone, what: &str) -> Vec<f64> {
    let total: f64 = weights.clone().sum();
    weights
        .map(|w| {
            assert!(
                w.is_finite() && w > 0.0,
                "SequenceGrammar: {what} weights must be finite and > 0 (got {w})"
            );
            w / total
        })
        .collect()
}

impl<Gr: SequenceGrammar> SequenceGrammarPrior<Gr> {
    /// A prior over the sequences of `grammar` of length at most `max_len`,
    /// with sites under the address prefix `seq`.
    pub fn new(grammar: Gr, max_len: usize) -> Self {
        Self {
            grammar: Arc::new(grammar),
            max_len,
            prefix: Arc::from("seq"),
        }
    }

    /// Use `prefix` instead of `seq` for the site addresses
    /// (`<prefix>/<i>#stop`, …), for a model that holds more than one
    /// sequence.
    ///
    /// # Panics
    ///
    /// If `prefix` contains `#`, which separates an address from its site
    /// name.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        assert!(
            !prefix.contains('#'),
            "SequenceGrammarPrior::with_prefix: the prefix must not contain '#'"
        );
        self.prefix = Arc::from(prefix);
        self
    }

    /// The grammar.
    pub fn grammar(&self) -> &Gr {
        &self.grammar
    }

    /// The maximum sequence length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// The address prefix of the sites (`seq` by default).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The address of one of position `position`'s sites.
    ///
    /// ```rust
    /// # use fugue_evo::inference::sequence::*;
    /// # struct G;
    /// # impl SequenceGrammar for G {
    /// #     type State = ();
    /// #     type Symbol = u8;
    /// #     fn initial_state(&self) {}
    /// #     fn choices(&self, _: &()) -> Vec<ChoiceKind<u8>> { vec![] }
    /// #     fn transition(&self, _: &(), _: &u8) {}
    /// #     fn stop_probability(&self, _: &()) -> f64 { 1.0 }
    /// # }
    /// let prior = SequenceGrammarPrior::new(G, 4);
    /// let addr = prior.address(2, SequenceSite::Kind);
    /// assert_eq!(addr.as_str(), "seq/2#kind");
    /// assert_eq!(prior.site_of(&addr), Some((2, SequenceSite::Kind)));
    /// assert_eq!(prior.site_of(&fugue::addr!("sigma")), None);
    /// ```
    pub fn address(&self, position: usize, site: SequenceSite) -> Address {
        Address::new(format!("{}/{}#{}", self.prefix, position, site.suffix()))
    }

    /// The position and site an address names, or `None` if it is not one of
    /// this prior's addresses (a latent site of the likelihood, say).
    pub fn site_of(&self, addr: &Address) -> Option<(usize, SequenceSite)> {
        let rest = addr
            .as_str()
            .strip_prefix(&*self.prefix)?
            .strip_prefix('/')?;
        let (position, site) = rest.split_once('#')?;
        let site = match site {
            "stop" => SequenceSite::Stop,
            "kind" => SequenceSite::Kind,
            "arg" => SequenceSite::Arg,
            _ => return None,
        };
        // Reject non-canonical spellings ("+3", "03") of a position.
        let parsed: usize = position.parse().ok()?;
        (parsed.to_string() == position).then_some((parsed, site))
    }

    /// The kinds of `state`, without empty ones, with normalised weights.
    fn kinds(&self, state: &Gr::State) -> Kinds<Gr::Symbol> {
        let raw: Vec<ChoiceKind<Gr::Symbol>> = self
            .grammar
            .choices(state)
            .into_iter()
            .filter(|k| !k.args.is_empty())
            .collect();
        let kind_probs = normalise(raw.iter().map(|k| k.weight), "kind");
        raw.into_iter()
            .zip(kind_probs)
            .map(|(kind, p)| {
                let arg_probs = normalise(kind.args.iter().map(|(_, w)| *w), "argument");
                let args = kind
                    .args
                    .into_iter()
                    .zip(arg_probs)
                    .map(|((s, _), q)| (s, q))
                    .collect();
                (p, args)
            })
            .collect()
    }

    fn stop_rule(&self, position: usize, state: &Gr::State, n_kinds: usize) -> StopRule {
        if position >= self.max_len || n_kinds == 0 {
            return StopRule::Forced;
        }
        let p = self.grammar.stop_probability(state);
        assert!(
            (0.0..=1.0).contains(&p),
            "SequenceGrammar::stop_probability must lie in [0, 1] (got {p})"
        );
        if p >= 1.0 {
            StopRule::Forced
        } else if p <= 0.0 {
            StopRule::Never
        } else {
            StopRule::Site(p)
        }
    }

    /// The program from position `position` on, given the prefix's state and
    /// symbols.
    fn position_model(
        &self,
        position: usize,
        state: Gr::State,
        acc: Vec<Gr::Symbol>,
    ) -> Model<Vec<Gr::Symbol>> {
        let kinds = self.kinds(&state);
        match self.stop_rule(position, &state, kinds.len()) {
            StopRule::Forced => pure(acc),
            StopRule::Never => self.symbol_model(position, state, acc, kinds),
            StopRule::Site(p) => {
                let this = self.clone();
                sample(
                    self.address(position, SequenceSite::Stop),
                    Bernoulli::new(p).expect("stop probability in (0, 1)"),
                )
                .bind(move |stop| {
                    if stop {
                        pure(acc)
                    } else {
                        this.symbol_model(position, state, acc, kinds)
                    }
                })
            }
        }
    }

    /// Draw the symbol at `position` (kind, then argument) and continue.
    fn symbol_model(
        &self,
        position: usize,
        state: Gr::State,
        acc: Vec<Gr::Symbol>,
        kinds: Kinds<Gr::Symbol>,
    ) -> Model<Vec<Gr::Symbol>> {
        let this = self.clone();
        let kind_index: Model<usize> = if kinds.len() == 1 {
            pure(0)
        } else {
            sample(
                self.address(position, SequenceSite::Kind),
                Categorical::new(kinds.iter().map(|(p, _)| *p).collect())
                    .expect("normalised kind weights"),
            )
        };
        kind_index.bind(move |k| {
            // A replayed index outside the support has already scored −∞ at
            // its site; end the execution there.
            let Some((_, args)) = kinds.into_iter().nth(k) else {
                return pure(acc);
            };
            let arg_index: Model<usize> = if args.len() == 1 {
                pure(0)
            } else {
                sample(
                    this.address(position, SequenceSite::Arg),
                    Categorical::new(args.iter().map(|(_, q)| *q).collect())
                        .expect("normalised argument weights"),
                )
            };
            arg_index.bind(move |a| {
                let Some((symbol, _)) = args.into_iter().nth(a) else {
                    return pure(acc);
                };
                let next = this.grammar.transition(&state, &symbol);
                let mut acc = acc;
                acc.push(symbol);
                this.position_model(position + 1, next, acc)
            })
        })
    }

    /// Walk `symbols` through the grammar, writing the choices that derive
    /// them into `trace` (when given). `Err` at the first position that has
    /// no derivation.
    fn derive(
        &self,
        symbols: &[Gr::Symbol],
        mut trace: Option<&mut Trace>,
    ) -> Result<(), Underivable> {
        let mut put = |addr: Address, value: ChoiceValue| {
            if let Some(t) = trace.as_deref_mut() {
                t.insert_choice(addr, value, 0.0);
            }
        };
        let mut state = self.grammar.initial_state();
        for position in 0..=symbols.len() {
            let kinds = self.kinds(&state);
            let rule = self.stop_rule(position, &state, kinds.len());
            let Some(symbol) = symbols.get(position) else {
                // The sequence ends here.
                return match rule {
                    StopRule::Forced => Ok(()),
                    StopRule::Site(_) => {
                        put(
                            self.address(position, SequenceSite::Stop),
                            ChoiceValue::Bool(true),
                        );
                        Ok(())
                    }
                    StopRule::Never => Err(Underivable {
                        position,
                        reason: "the sequence ends where the grammar's stop probability is 0"
                            .to_string(),
                    }),
                };
            };
            match rule {
                StopRule::Forced => {
                    return Err(Underivable {
                        position,
                        reason: if position >= self.max_len {
                            format!("the sequence is longer than max_len = {}", self.max_len)
                        } else {
                            "the sequence continues where the grammar must stop".to_string()
                        },
                    })
                }
                StopRule::Site(_) => put(
                    self.address(position, SequenceSite::Stop),
                    ChoiceValue::Bool(false),
                ),
                StopRule::Never => {}
            }
            let found = kinds.iter().enumerate().find_map(|(k, (_, args))| {
                args.iter()
                    .position(|(s, _)| s == symbol)
                    .map(|a| (k, a, args.len()))
            });
            let Some((k, a, n_args)) = found else {
                return Err(Underivable {
                    position,
                    reason: "the symbol is not available after this prefix".to_string(),
                });
            };
            if kinds.len() > 1 {
                put(
                    self.address(position, SequenceSite::Kind),
                    ChoiceValue::Usize(k),
                );
            }
            if n_args > 1 {
                put(
                    self.address(position, SequenceSite::Arg),
                    ChoiceValue::Usize(a),
                );
            }
            state = self.grammar.transition(&state, symbol);
        }
        Ok(())
    }

    /// The states of every prefix of `genome`: `states[k]` is the state
    /// after its first `k` symbols, so the vector has `len + 1` entries.
    ///
    /// # Errors
    ///
    /// As [`GenomePrior::validate`], for a sequence outside the language.
    pub fn prefix_states(
        &self,
        genome: &SequenceGenome<Gr::Symbol>,
    ) -> Result<Vec<Gr::State>, GenomeError> {
        self.validate(genome)?;
        Ok(self.fold_states(genome.symbols()))
    }

    fn fold_states(&self, symbols: &[Gr::Symbol]) -> Vec<Gr::State> {
        let mut states = Vec::with_capacity(symbols.len() + 1);
        states.push(self.grammar.initial_state());
        for symbol in symbols {
            let next = self
                .grammar
                .transition(states.last().expect("non-empty"), symbol);
            states.push(next);
        }
        states
    }

    /// Log prior probability of `genome`, obtained by replaying its encoding
    /// through the prior program.
    ///
    /// # Errors
    ///
    /// As [`GenomePrior::validate`], for a sequence outside the language.
    pub fn log_prior(&self, genome: &SequenceGenome<Gr::Symbol>) -> Result<f64, GenomeError> {
        self.validate(genome)?;
        let (_g, scored) = score_complete(self.trace_of(genome), self.model())?;
        Ok(scored.log_prior)
    }

    /// The description length of `genome` under the grammar, in nats:
    /// `−log_prior`. With uniform weights it is the sum, over the choices
    /// that derive the sequence, of the log of the number of alternatives.
    ///
    /// # Errors
    ///
    /// As [`GenomePrior::validate`], for a sequence outside the language.
    pub fn description_length(
        &self,
        genome: &SequenceGenome<Gr::Symbol>,
    ) -> Result<f64, GenomeError> {
        Ok(-self.log_prior(genome)?)
    }

    /// Decode the sequence held by a complete trace of this prior (a chain
    /// state, a particle trace) by replaying the prior program over it.
    /// Sites of other programs in the trace are ignored.
    ///
    /// # Errors
    ///
    /// [`GenomeError::MissingAddress`] if the program visits a site the
    /// trace lacks.
    pub fn decode(&self, trace: &Trace) -> Result<SequenceGenome<Gr::Symbol>, GenomeError> {
        let (genome, _scored, report) = fugue::score_given_trace_reconciled(
            trace.clone(),
            &mut rand::rngs::mock::StepRng::new(0, 1),
            self.model(),
        )
        .map_err(|e| GenomeError::InvalidStructure(e.to_string()))?;
        match report.fresh_addresses.first() {
            Some(addr) => Err(GenomeError::MissingAddress(addr.to_string())),
            None => Ok(genome),
        }
    }
}

impl<Gr: SequenceGrammar> GenomePrior for SequenceGrammarPrior<Gr> {
    type Genome = SequenceGenome<Gr::Symbol>;

    fn model(&self) -> Model<Self::Genome> {
        self.position_model(0, self.grammar.initial_state(), Vec::new())
            .map(SequenceGenome::new)
    }

    /// Encode a sequence under the grammar's address scheme (the inverse of
    /// running [`Self::model`]): the `#stop` / `#kind` / `#arg` choices that
    /// derive it, omitting the choices that have a single outcome.
    ///
    /// A sequence outside the language has no encoding. Call
    /// [`Self::validate`] first, as `EvolutionModel::score` and the chain
    /// initialisers do. For such a sequence this returns the choices of its
    /// longest derivable prefix followed by an out-of-range index at the
    /// offending position's `#arg` address, which replay never scores as a
    /// finite density: it is reported as `−∞` or as a structural error.
    fn trace_of(&self, genome: &Self::Genome) -> Trace {
        let mut trace = Trace::default();
        if let Err(bad) = self.derive(genome.symbols(), Some(&mut trace)) {
            trace.insert_choice(
                self.address(bad.position, SequenceSite::Arg),
                ChoiceValue::Usize(usize::MAX),
                0.0,
            );
        }
        trace
    }

    /// `Ok` exactly for the sequences the grammar derives:
    /// [`GenomeError::InvalidStructure`], naming the first offending
    /// position, when a symbol is not available after its prefix, when the
    /// sequence continues past a forced stop or past `max_len`, or when it
    /// ends where the stop probability is 0.
    fn validate(&self, genome: &Self::Genome) -> Result<(), GenomeError> {
        self.derive(genome.symbols(), None).map_err(|bad| {
            GenomeError::InvalidStructure(format!(
                "sequence has no derivation at position {}: {}",
                bad.position, bad.reason
            ))
        })
    }
}

/// The key of a prefix state, for [`PrefixKeyedSuffixCrossover`].
type StateKeyFn<St, K> = Box<dyn Fn(&St) -> K + Send>;

/// **Suffix crossover** for [`SequenceGrammarPrior`] particles: a
/// [`fugue::PopulationKernel`] that exchanges the tails of two sequences at
/// cut points whose prefix states have **equal keys**, so that each tail is
/// grafted onto a prefix it type-checks against.
///
/// One proposal picks two distinct particles `a`, `b`, decodes their
/// sequences, and lists every pair of cut points `(i, j)`, with `i` a
/// position of `a` and `j` a position of `b` (each from 0 to the sequence's
/// length), such that `key(state_a[i]) == key(state_b[j])`. The pairs
/// `(0, 0)` (a plain exchange of the two sequences) and
/// `(len_a, len_b)` (two empty tails) are left out. One pair is drawn
/// uniformly, and the children are
///
/// ```text
/// a′ = a[..i] ++ b[j..]        b′ = b[..j] ++ a[i..]
/// ```
///
/// Each child's choices are the encoding of its sequence
/// ([`GenomePrior::trace_of`]) plus every other site its particle held, such
/// as a latent of the likelihood, which stays where it is. A proposal with
/// a child outside the language is rejected; otherwise both children are
/// re-scored through the model, and the proposal is rejected if either is
/// not a complete execution over exactly its own choices.
///
/// # Acceptance
///
/// The pair is accepted with probability `min(1, α)`,
///
/// ```text
/// α = π_β(a′)·π_β(b′) / (π_β(a)·π_β(b))  ×  n(a, b) / n(a′, b′)
/// ```
///
/// where `π_β = prior · likelihood^β` is the tempered target of one
/// execution and `n(·, ·)` is the number of admissible cut pairs. The same
/// cut pair maps the children back to the parents (the prefixes, hence
/// their keys, are unchanged), so `1/n(a, b)` and `1/n(a′, b′)` are the
/// forward and reverse proposal probabilities and the move leaves the
/// product of tempered targets invariant **for any key function**. The key
/// only decides how often a proposal is well typed: when equal keys imply
/// equal futures (the key determines the state's choices and is preserved
/// by transitions), the only children outside the language are those longer
/// than `max_len`. A coarser key is still exact; it proposes more children
/// that are rejected.
///
/// Particle weights are not touched ([`fugue::PopulationKernel`]'s
/// contract).
///
/// ```rust
/// use fugue_evo::inference::likelihood::NoLikelihood;
/// use fugue_evo::inference::model::EvolutionModel;
/// use fugue_evo::inference::sequence::*;
/// use fugue_evo::inference::smc::{EvoSmcConfig, EvolutionSMC};
/// use rand::{rngs::StdRng, SeedableRng};
///
/// /// Bit strings with no two consecutive ones.
/// struct NoTwoOnes;
/// impl SequenceGrammar for NoTwoOnes {
///     type State = bool; // was the last symbol a one?
///     type Symbol = u8;
///     fn initial_state(&self) -> bool {
///         false
///     }
///     fn choices(&self, last_one: &bool) -> Vec<ChoiceKind<u8>> {
///         vec![ChoiceKind::uniform(if *last_one { vec![0] } else { vec![0, 1] })]
///     }
///     fn transition(&self, _: &bool, bit: &u8) -> bool {
///         *bit == 1
///     }
///     fn stop_probability(&self, _: &bool) -> f64 {
///         0.3
///     }
/// }
///
/// let prior = SequenceGrammarPrior::new(NoTwoOnes, 6);
/// let model = EvolutionModel::from_likelihood(prior.clone(), NoLikelihood);
/// // Tails are exchanged only between prefixes that end the same way.
/// let mut kernel = PrefixKeyedSuffixCrossover::by_state(prior.clone(), 16);
/// let mut rng = StdRng::seed_from_u64(7);
/// let cfg = EvoSmcConfig {
///     num_particles: 64,
///     crossover: None, // replaced by the explicit kernel
///     ..EvoSmcConfig::default()
/// };
/// let posterior = EvolutionSMC::run_with_kernel(&mut rng, &model, cfg, &mut kernel);
/// for (seq, _w) in posterior.genomes(&model.smc_model()) {
///     assert!(prior.log_prior(&seq).unwrap().is_finite());
/// }
/// ```
pub struct PrefixKeyedSuffixCrossover<Gr: SequenceGrammar, K> {
    prior: SequenceGrammarPrior<Gr>,
    key: StateKeyFn<Gr::State, K>,
    /// Number of (pair, cut) proposals per sweep.
    pub n_pairs: usize,
    proposed: Arc<AtomicUsize>,
    accepted: Arc<AtomicUsize>,
}

impl<Gr: SequenceGrammar> PrefixKeyedSuffixCrossover<Gr, Gr::State>
where
    Gr::State: PartialEq,
{
    /// Cut only where the two prefix states are equal.
    pub fn by_state(prior: SequenceGrammarPrior<Gr>, n_pairs: usize) -> Self {
        Self::new(prior, n_pairs, Gr::State::clone)
    }
}

impl<Gr: SequenceGrammar, K: PartialEq> PrefixKeyedSuffixCrossover<Gr, K> {
    /// Cut only where `key` agrees on the two prefix states (a hash of a
    /// canonical form, a type environment, …). `n_pairs` proposals are made
    /// per sweep.
    pub fn new(
        prior: SequenceGrammarPrior<Gr>,
        n_pairs: usize,
        key: impl Fn(&Gr::State) -> K + Send + 'static,
    ) -> Self {
        Self {
            prior,
            key: Box::new(key),
            n_pairs,
            proposed: Arc::new(AtomicUsize::new(0)),
            accepted: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Proposals that had at least one admissible cut pair, over the life of
    /// the kernel.
    pub fn proposed(&self) -> usize {
        self.proposed.load(Ordering::Relaxed)
    }

    /// Accepted proposals over the life of the kernel.
    pub fn accepted(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Decode the sequence of a particle trace and the keys of its prefix
    /// states. `None` if the trace is not a complete execution of the prior,
    /// or holds a derivation other than the one [`GenomePrior::trace_of`]
    /// writes (possible only in a grammar that offers a symbol twice in one
    /// state); the kernel leaves such a particle alone, because it rebuilds
    /// children from their symbols.
    fn read(&self, trace: &Trace) -> Option<(Vec<Gr::Symbol>, Vec<K>)> {
        let genome = self.prior.decode(trace).ok()?;
        let canonical = self.prior.trace_of(&genome);
        let same_derivation = self.prior.position_sites(trace).count() == canonical.choices.len()
            && canonical
                .choices
                .iter()
                .all(|(addr, c)| trace.choices.get(addr).is_some_and(|t| t.value == c.value));
        if !same_derivation {
            return None;
        }
        let keys = self
            .prior
            .fold_states(genome.symbols())
            .iter()
            .map(|s| (self.key)(s))
            .collect();
        Some((genome.into_symbols(), keys))
    }

    /// Every admissible cut pair of two key sequences.
    fn cuts(a: &[K], b: &[K]) -> Vec<(usize, usize)> {
        let (la, lb) = (a.len() - 1, b.len() - 1);
        let mut out = Vec::new();
        for (i, ka) in a.iter().enumerate() {
            for (j, kb) in b.iter().enumerate() {
                let exchange = i == 0 && j == 0;
                let empty_tails = i == la && j == lb;
                if !exchange && !empty_tails && ka == kb {
                    out.push((i, j));
                }
            }
        }
        out
    }

    /// The particle `owner` with its sequence replaced by `symbols`: the
    /// encoding of the new sequence plus every site of `owner` that is not
    /// one of the prior's. `None` if `symbols` is outside the language.
    fn with_sequence(&self, owner: &Trace, symbols: Vec<Gr::Symbol>) -> Option<Trace> {
        let genome = SequenceGenome::new(symbols);
        self.prior.validate(&genome).ok()?;
        let mut child = self.prior.trace_of(&genome);
        for (addr, choice) in &owner.choices {
            if self.prior.site_of(addr).is_none() {
                child.choices.insert(addr.clone(), choice.clone());
            }
        }
        Some(child)
    }
}

impl<Gr: SequenceGrammar, K: PartialEq> PrefixKeyedSuffixCrossover<Gr, K> {
    /// The children of the particles `(ta, tb)`, holding the sequences
    /// `(a, b)`, cut at `(i, j)`, one of `n_forward` admissible cut pairs:
    /// re-scored under `model_fn`, with the log acceptance ratio of the
    /// exchange. `None` when a child is outside the language or is not a
    /// complete execution over its own choices.
    #[allow(clippy::too_many_arguments)]
    fn propose<A>(
        &self,
        (ta, a): (&Trace, &[Gr::Symbol]),
        (tb, b): (&Trace, &[Gr::Symbol]),
        (i, j): (usize, usize),
        n_forward: usize,
        rng: &mut dyn rand::RngCore,
        model_fn: &dyn Fn() -> Model<A>,
        beta: f64,
    ) -> Option<(Trace, Trace, f64)> {
        let join = |head: &[Gr::Symbol], tail: &[Gr::Symbol]| -> Vec<Gr::Symbol> {
            head.iter().chain(tail).cloned().collect()
        };
        let ca = self.with_sequence(ta, join(&a[..i], &b[j..]))?;
        let cb = self.with_sequence(tb, join(&b[..j], &a[i..]))?;
        let ca = rescore_complete(&ca, rng, model_fn)?;
        let cb = rescore_complete(&cb, rng, model_fn)?;
        let reverse = Self::cuts(&self.read(&ca)?.1, &self.read(&cb)?.1);
        // The prefixes, hence their keys, are unchanged, so the same cut
        // pair is admissible for the children and maps them back.
        debug_assert!(reverse.contains(&(i, j)));
        // Tempered log-density of one execution:
        //   log π_β(θ) = log_prior + β·(log_likelihood + log_factors).
        let logd = |t: &Trace| t.log_prior + beta * (t.log_likelihood + t.log_factors);
        let log_alpha = (logd(&ca) + logd(&cb)) - (logd(ta) + logd(tb)) + (n_forward as f64).ln()
            - (reverse.len() as f64).ln();
        Some((ca, cb, log_alpha))
    }
}

impl<Gr: SequenceGrammar> SequenceGrammarPrior<Gr> {
    /// The sequence sites of a trace, with their positions.
    fn position_sites<'a>(
        &'a self,
        trace: &'a Trace,
    ) -> impl Iterator<Item = (&'a Address, (usize, SequenceSite))> + 'a {
        trace
            .choices
            .keys()
            .filter_map(move |addr| self.site_of(addr).map(|ps| (addr, ps)))
    }
}

impl<A, Gr, K> PopulationKernel<A> for PrefixKeyedSuffixCrossover<Gr, K>
where
    Gr: SequenceGrammar,
    K: PartialEq,
{
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
            let a = rng.gen_range(0..n);
            let mut b = rng.gen_range(0..n - 1);
            if b >= a {
                b += 1; // distinct partner
            }
            let (ta, tb) = (&particles[a].trace, &particles[b].trace);
            let (Some((sa, ka)), Some((sb, kb))) = (self.read(ta), self.read(tb)) else {
                continue;
            };
            let forward = Self::cuts(&ka, &kb);
            if forward.is_empty() {
                continue;
            }
            self.proposed.fetch_add(1, Ordering::Relaxed);
            let cut = forward[rng.gen_range(0..forward.len())];
            let Some((ca, cb, log_alpha)) = self.propose(
                (ta, &sa),
                (tb, &sb),
                cut,
                forward.len(),
                rng,
                model_fn,
                beta,
            ) else {
                continue; // a tail does not fit its new prefix: self-loop
            };
            if log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp() {
                particles[a].trace = ca; // only traces move;
                particles[b].trace = cb; // weights untouched (kernel contract).
                self.accepted.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::traits::Fitness;
    use crate::inference::likelihood::NoLikelihood;
    use crate::inference::mh::EvolutionChain;
    use crate::inference::model::EvolutionModel;
    use crate::inference::smc::{EvoSmcConfig, EvolutionSMC};
    use fugue::runtime::handler::run;
    use fugue::runtime::interpreters::PriorHandler;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    enum Sym {
        Push,
        Pop,
        A,
        B,
        C,
    }

    /// A small typed language over a bounded stack. It exercises every
    /// shape of site: both kinds present (depth 1), one kind only (depth 2,
    /// no `#kind`), a kind with one argument (no `#arg`), a state that
    /// cannot stop (depth 1, no `#stop`), and non-uniform weights.
    #[derive(Clone)]
    struct Stack;

    impl SequenceGrammar for Stack {
        type State = u8;
        type Symbol = Sym;

        fn initial_state(&self) -> u8 {
            0
        }

        fn choices(&self, depth: &u8) -> Vec<ChoiceKind<Sym>> {
            let mut stack = Vec::new();
            if *depth < 2 {
                stack.push((Sym::Push, 2.0));
            }
            if *depth > 0 {
                stack.push((Sym::Pop, 1.0));
            }
            let lits = match depth {
                0 => vec![(Sym::A, 1.0), (Sym::B, 1.0), (Sym::C, 2.0)],
                1 => vec![(Sym::A, 1.0)],
                _ => vec![],
            };
            vec![
                ChoiceKind::weighted(stack),
                ChoiceKind::weighted(lits).with_weight(2.0),
            ]
        }

        fn transition(&self, depth: &u8, symbol: &Sym) -> u8 {
            match symbol {
                Sym::Push => depth + 1,
                Sym::Pop => depth - 1,
                _ => *depth,
            }
        }

        fn stop_probability(&self, depth: &u8) -> f64 {
            match depth {
                0 => 0.4,
                1 => 0.0,
                _ => 0.25,
            }
        }
    }

    const MAX_LEN: usize = 3;

    /// Chain length of the exact-recovery tests.
    const STEPS: usize = 120_000;

    fn prior() -> SequenceGrammarPrior<Stack> {
        SequenceGrammarPrior::new(Stack, MAX_LEN)
    }

    /// Exact enumeration of the language with its probabilities, computed
    /// from the grammar's raw weights independently of the program.
    fn enumerate() -> HashMap<Vec<Sym>, f64> {
        fn go(state: u8, prefix: Vec<Sym>, p: f64, out: &mut HashMap<Vec<Sym>, f64>) {
            let g = Stack;
            let kinds: Vec<ChoiceKind<Sym>> = g
                .choices(&state)
                .into_iter()
                .filter(|k| !k.args.is_empty())
                .collect();
            let p_stop = if prefix.len() == MAX_LEN || kinds.is_empty() {
                1.0
            } else {
                g.stop_probability(&state)
            };
            if p_stop > 0.0 {
                *out.entry(prefix.clone()).or_insert(0.0) += p * p_stop;
            }
            if p_stop >= 1.0 {
                return;
            }
            let kind_total: f64 = kinds.iter().map(|k| k.weight).sum();
            for kind in &kinds {
                let arg_total: f64 = kind.args.iter().map(|(_, w)| w).sum();
                for (symbol, w) in &kind.args {
                    let mut next = prefix.clone();
                    next.push(*symbol);
                    go(
                        g.transition(&state, symbol),
                        next,
                        p * (1.0 - p_stop) * (kind.weight / kind_total) * (w / arg_total),
                        out,
                    );
                }
            }
        }
        let mut out = HashMap::new();
        go(0, Vec::new(), 1.0, &mut out);
        out
    }

    /// A likelihood that prefers longer sequences and dislikes `A`, with a
    /// sharpness factor.
    #[derive(Clone, Copy)]
    struct Taste(f64);

    fn taste(symbols: &[Sym]) -> f64 {
        0.6 * symbols.len() as f64 - 1.1 * symbols.iter().filter(|s| **s == Sym::A).count() as f64
    }

    impl Fitness for Taste {
        type Genome = SequenceGenome<Sym>;
        type Value = f64;
        fn evaluate(&self, g: &Self::Genome) -> f64 {
            self.0 * taste(g.symbols())
        }
    }

    /// The exact posterior `∝ prior · exp(sharpness · taste)` and its log
    /// normaliser.
    fn exact_posterior(sharpness: f64) -> (HashMap<Vec<Sym>, f64>, f64) {
        let mut post: HashMap<Vec<Sym>, f64> = enumerate()
            .into_iter()
            .map(|(s, p)| {
                let w = p * (sharpness * taste(&s)).exp();
                (s, w)
            })
            .collect();
        let z: f64 = post.values().sum();
        post.values_mut().for_each(|w| *w /= z);
        (post, z.ln())
    }

    /// Total variation distance between weighted samples and an exact pmf.
    /// A sample outside the pmf's support counts in full.
    fn total_variation(samples: &HashMap<Vec<Sym>, f64>, exact: &HashMap<Vec<Sym>, f64>) -> f64 {
        let total: f64 = samples.values().sum();
        let mut tv = 0.0;
        for (s, p) in exact {
            tv += (samples.get(s).copied().unwrap_or(0.0) / total - p).abs();
        }
        for (s, w) in samples {
            if !exact.contains_key(s) {
                tv += w / total;
            }
        }
        tv / 2.0
    }

    #[test]
    fn test_enumeration_is_a_distribution() {
        let exact = enumerate();
        let total: f64 = exact.values().sum();
        assert!((total - 1.0).abs() < 1e-12, "total mass {total}");
        assert!(exact.len() > 30, "{} sequences", exact.len());
        // The language is prefix dependent: Pop never comes first, and a
        // sequence cannot end at depth 1 before max_len.
        assert!(!exact.contains_key(&vec![Sym::Pop]));
        assert!(!exact.contains_key(&vec![Sym::Push]));
        assert!(exact.contains_key(&vec![Sym::Push, Sym::A, Sym::A]));
    }

    /// The log-density of every sequence of the language, obtained by
    /// replaying its encoding through the program, is its exact probability;
    /// with the hand-computed description length of one of them.
    #[test]
    fn test_log_prior_matches_enumeration() {
        let prior = prior();
        for (symbols, p) in enumerate() {
            let genome = SequenceGenome::new(symbols.clone());
            prior.validate(&genome).expect("in the language");
            let lp = prior.log_prior(&genome).unwrap();
            assert!(
                (lp - p.ln()).abs() < 1e-12,
                "{symbols:?}: {lp} vs {}",
                p.ln()
            );
            assert_eq!(prior.description_length(&genome).unwrap(), -lp);
        }
        // [Push, Pop]: continue (0.6), kind stack (1/3), Push (the only
        // argument at depth 0); no stop site at depth 1, kind stack (1/3),
        // Pop (1/3); stop (0.4).
        let dl = prior
            .description_length(&SequenceGenome::new(vec![Sym::Push, Sym::Pop]))
            .unwrap();
        let by_hand = -(0.6f64 * (1.0 / 3.0) * (1.0 / 3.0) * (1.0 / 3.0) * 0.4).ln();
        assert!((dl - by_hand).abs() < 1e-12);
    }

    /// `trace_of` is the exact inverse of the generative program: it
    /// reproduces a prior draw's choices site for site, and `decode` brings
    /// the sequence back.
    #[test]
    fn test_trace_of_inverts_generative_run() {
        let prior = prior();
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..300 {
            let (genome, generated) = run(
                PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                },
                prior.model(),
            );
            let encoded = prior.trace_of(&genome);
            assert_eq!(encoded.choices.len(), generated.choices.len());
            for (addr, choice) in &generated.choices {
                assert_eq!(encoded.choices[addr].value, choice.value, "at {addr}");
                assert!(prior.site_of(addr).is_some());
            }
            assert_eq!(prior.decode(&generated).unwrap(), genome);
            assert_eq!(
                prior.prefix_states(&genome).unwrap().len(),
                genome.len() + 1
            );
            assert!(genome.len() <= MAX_LEN);
        }
    }

    /// Singleton choices emit no site.
    #[test]
    fn test_degenerate_choices_emit_no_site() {
        let prior = prior();
        // Push (one stack argument at depth 0), Push (stack kind has two
        // arguments at depth 1, no stop site), Pop (the only symbol at
        // depth 2), forced stop at max_len.
        let trace = prior.trace_of(&SequenceGenome::new(vec![Sym::Push, Sym::Push, Sym::Pop]));
        let mut addrs: Vec<&str> = trace.choices.keys().map(|a| a.as_str()).collect();
        addrs.sort_unstable();
        assert_eq!(
            addrs,
            vec![
                "seq/0#kind",
                "seq/0#stop",
                "seq/1#arg",
                "seq/1#kind",
                "seq/2#stop"
            ]
        );
    }

    #[test]
    fn test_validate_names_the_offending_position() {
        let prior = prior();
        let reason = |symbols: Vec<Sym>| match prior.validate(&SequenceGenome::new(symbols)) {
            Err(GenomeError::InvalidStructure(msg)) => msg,
            other => panic!("expected InvalidStructure, got {other:?}"),
        };
        assert!(reason(vec![Sym::Pop]).contains("position 0: the symbol is not available"));
        assert!(reason(vec![Sym::Push, Sym::B]).contains("position 1"));
        assert!(reason(vec![Sym::Push]).contains("stop probability is 0"));
        assert!(reason(vec![Sym::A; 4]).contains("longer than max_len = 3"));

        // Scoring and warm starts report the same error, before any replay.
        let model = EvolutionModel::from_likelihood(prior.clone(), NoLikelihood);
        let bad = SequenceGenome::new(vec![Sym::Pop]);
        assert!(matches!(
            model.score(&bad),
            Err(GenomeError::InvalidStructure(_))
        ));
        assert!(EvolutionChain::new(model).init_from(&bad).is_none());
    }

    /// Replaying the encoding of a sequence outside the language never
    /// yields a finite density, even when `validate` is skipped.
    #[test]
    fn test_trace_of_underivable_never_scores_finite() {
        let prior = prior();
        for symbols in [
            vec![Sym::Pop],
            vec![Sym::Push, Sym::B],
            vec![Sym::Push],
            vec![Sym::A; 4],
            vec![Sym::Push, Sym::Push, Sym::A],
        ] {
            let genome = SequenceGenome::new(symbols.clone());
            assert!(prior.validate(&genome).is_err(), "{symbols:?}");
            match score_complete(prior.trace_of(&genome), prior.model()) {
                Err(_) => {}
                Ok((_, scored)) => assert_eq!(
                    scored.log_prior,
                    f64::NEG_INFINITY,
                    "{symbols:?} scored finite"
                ),
            }
        }
    }

    #[test]
    fn test_addresses_round_trip_and_prefix() {
        let prior = prior().with_prefix("mv");
        assert_eq!(prior.prefix(), "mv");
        assert_eq!(prior.max_len(), MAX_LEN);
        for site in [SequenceSite::Stop, SequenceSite::Kind, SequenceSite::Arg] {
            let addr = prior.address(12, site);
            assert_eq!(prior.site_of(&addr), Some((12, site)));
        }
        assert_eq!(prior.address(0, SequenceSite::Stop).as_str(), "mv/0#stop");
        for foreign in [
            "seq/0#stop",
            "mv/0#leaf",
            "mv/x#stop",
            "mv/01#stop",
            "mvx/0#stop",
        ] {
            assert_eq!(prior.site_of(&Address::new(foreign)), None, "{foreign}");
        }
        let trace = prior.trace_of(&SequenceGenome::new(vec![Sym::C]));
        assert!(trace.choices.keys().all(|a| a.as_str().starts_with("mv/")));
    }

    #[test]
    #[should_panic(expected = "weights must be finite and > 0")]
    fn test_non_positive_weight_panics() {
        struct Bad;
        impl SequenceGrammar for Bad {
            type State = ();
            type Symbol = u8;
            fn initial_state(&self) {}
            fn choices(&self, _: &()) -> Vec<ChoiceKind<u8>> {
                vec![ChoiceKind::weighted(vec![(0, 1.0), (1, 0.0)])]
            }
            fn transition(&self, _: &(), _: &u8) {}
            fn stop_probability(&self, _: &()) -> f64 {
                0.5
            }
        }
        let _ = SequenceGrammarPrior::new(Bad, 2).model();
    }

    fn count(samples: &mut HashMap<Vec<Sym>, f64>, genome: &SequenceGenome<Sym>, w: f64) {
        *samples.entry(genome.symbols().to_vec()).or_insert(0.0) += w;
    }

    /// Run one MH chain and return the visit counts of its states.
    fn chain_counts<L>(
        model: EvolutionModel<SequenceGrammarPrior<Stack>, L>,
        seed: u64,
        steps: usize,
    ) -> HashMap<Vec<Sym>, f64>
    where
        L: crate::inference::likelihood::GenomeLikelihood<SequenceGenome<Sym>>,
    {
        let prior = model.prior().clone();
        let mut chain = EvolutionChain::new(model);
        let mut rng = StdRng::seed_from_u64(seed);
        // Start far from typical: the longest, deepest sequence.
        let start = SequenceGenome::new(vec![Sym::Push, Sym::Push, Sym::Pop]);
        let mut state = chain.init_from(&start).expect("in the language");
        let mut current = start;
        let mut samples = HashMap::new();
        for _ in 0..steps {
            if let Some((genome, trace, _lw)) = chain.step_scored(&mut rng, &state) {
                state = trace;
                current = genome;
            }
            debug_assert_eq!(prior.decode(&state).unwrap(), current);
            count(&mut samples, &current, 1.0);
        }
        samples
    }

    /// The prior-recovery property for the structure-changing MH moves:
    /// with a flat likelihood, fugue's single-site MH (with its
    /// reversible-jump correction for births, deaths and the change in the
    /// number of sites) samples exactly the grammar prior over sequences.
    #[test]
    fn test_mh_with_flat_likelihood_recovers_the_prior() {
        let model = EvolutionModel::from_likelihood(prior(), NoLikelihood);
        let samples = chain_counts(model, 2026, STEPS);
        let tv = total_variation(&samples, &enumerate());

        assert!(tv < 0.02, "MH prior recovery: total variation {tv}");
    }

    /// And with a likelihood it samples the exact posterior.
    #[test]
    fn test_mh_recovers_the_exact_posterior() {
        let model = EvolutionModel::new(prior(), Taste(1.0));
        let samples = chain_counts(model, 99, STEPS);
        let tv = total_variation(&samples, &exact_posterior(1.0).0);

        assert!(tv < 0.02, "MH posterior: total variation {tv}");
    }

    /// Prior draws of the program have the enumerated frequencies.
    #[test]
    fn test_prior_draws_match_enumeration() {
        let model = EvolutionModel::from_likelihood(prior(), NoLikelihood);
        let mut rng = StdRng::seed_from_u64(5);
        let mut samples = HashMap::new();
        for _ in 0..STEPS {
            count(&mut samples, &model.sample_prior(&mut rng), 1.0);
        }
        let tv = total_variation(&samples, &enumerate());
        assert!(tv < 0.02, "prior draws: total variation {tv}");
    }

    /// The likelihood sharpness of the SMC tests: large enough that the
    /// tempering ladder has several rungs, each with a resample, a
    /// rejuvenation sweep and a kernel sweep.
    const SHARP: f64 = 4.0;

    /// Compare an SMC posterior with the exact one on two projections with
    /// few cells (sequence length; first symbol), where 4000 particles
    /// resolve a total variation of a few percent, and on the evidence.
    fn smc_check(
        posterior: crate::inference::smc::EvolutionPosterior<SequenceGenome<Sym>>,
        what: &str,
    ) {
        let model = EvolutionModel::new(prior(), Taste(SHARP));
        let (exact, log_z) = exact_posterior(SHARP);
        let mut samples = HashMap::new();
        for (genome, w) in posterior.genomes(&model.smc_model()) {
            count(&mut samples, &genome, w);
        }
        assert!(samples.keys().all(|s| exact.contains_key(s)));
        type Projection = fn(&Vec<Sym>) -> Vec<Sym>;
        let by_length: Projection = |s| vec![Sym::A; s.len()];
        let by_first: Projection = |s| s.iter().take(1).copied().collect();
        for (name, project) in [("length", by_length), ("first symbol", by_first)] {
            let marginal = |m: &HashMap<Vec<Sym>, f64>| {
                let mut out: HashMap<Vec<Sym>, f64> = HashMap::new();
                for (s, w) in m {
                    *out.entry(project(s)).or_insert(0.0) += w;
                }
                out
            };
            let tv = total_variation(&marginal(&samples), &marginal(&exact));
            assert!(tv < 0.04, "{what}, {name}: total variation {tv}");
        }
        assert!(
            (posterior.log_evidence - log_z).abs() < 0.1,
            "{what}: log evidence {} vs exact {log_z}",
            posterior.log_evidence
        );
    }

    fn smc_cfg() -> EvoSmcConfig {
        EvoSmcConfig {
            num_particles: 4000,
            rejuvenation_steps: 4,
            ..EvoSmcConfig::default()
        }
    }

    /// Tempered SMC with the default configuration (single-site
    /// rejuvenation plus the structure-safe shared-site crossover) recovers
    /// the exact posterior and evidence.
    #[test]
    fn test_smc_default_config_recovers_the_exact_posterior() {
        let model = EvolutionModel::new(prior(), Taste(SHARP));
        let mut rng = StdRng::seed_from_u64(31);
        smc_check(
            EvolutionSMC::run(&mut rng, &model, smc_cfg()),
            "SMC default",
        );
    }

    /// And so does SMC with the suffix crossover kernel.
    #[test]
    fn test_smc_with_suffix_crossover_recovers_the_exact_posterior() {
        let model = EvolutionModel::new(prior(), Taste(SHARP));
        let mut rng = StdRng::seed_from_u64(32);
        let mut kernel = PrefixKeyedSuffixCrossover::by_state(prior(), 2000);
        let posterior = EvolutionSMC::run_with_kernel(
            &mut rng,
            &model,
            EvoSmcConfig {
                crossover: None,
                ..smc_cfg()
            },
            &mut kernel,
        );
        assert!(
            kernel.accepted() > 100,
            "{} of {} suffix swaps accepted",
            kernel.accepted(),
            kernel.proposed()
        );
        smc_check(posterior, "SMC suffix crossover");
    }

    /// The whole pipeline a structure search uses: grammar prior, a
    /// memoised deterministic factor keyed by a canonical form (here the
    /// sorted symbols, so reorderings share one evaluation) that declares
    /// some sequences invalid, tempered SMC, and the decoded-genome
    /// summaries, against exact enumeration.
    #[test]
    fn test_memoized_factor_posterior_matches_enumeration() {
        use crate::inference::likelihood::{DeterministicFactor, FactorInput, MemoizedFactor};

        /// Sequences containing `B` are invalid; the rest score by `taste`.
        struct NoB;
        impl DeterministicFactor for NoB {
            type Genome = SequenceGenome<Sym>;
            type Key = Vec<u8>;
            type Decoded = Vec<Sym>;
            fn decode(&self, g: &Self::Genome) -> FactorInput<Vec<u8>, Vec<Sym>> {
                if g.symbols().contains(&Sym::B) {
                    return FactorInput::Invalid;
                }
                let mut key: Vec<u8> = g.symbols().iter().map(|s| *s as u8).collect();
                key.sort_unstable();
                FactorInput::Valid {
                    key,
                    decoded: g.symbols().to_vec(),
                }
            }
            fn log_factor(&self, symbols: &Vec<Sym>) -> f64 {
                SHARP * taste(symbols)
            }
        }

        let likelihood = MemoizedFactor::new(NoB);
        let model = EvolutionModel::from_likelihood(prior(), likelihood.clone());
        let mut rng = StdRng::seed_from_u64(57);
        let posterior = EvolutionSMC::run(&mut rng, &model, smc_cfg());
        let model_fn = || prior().model();

        let mut exact: HashMap<Vec<Sym>, f64> = enumerate()
            .into_iter()
            .filter(|(s, _)| !s.contains(&Sym::B))
            .map(|(s, p)| {
                let w = p * (SHARP * taste(&s)).exp();
                (s, w)
            })
            .collect();
        let z: f64 = exact.values().sum();
        exact.values_mut().for_each(|w| *w /= z);

        // No invalid sequence carries posterior mass.
        let distinct = posterior.distinct(&model_fn);
        assert!(distinct
            .iter()
            .all(|(g, _)| exact.contains_key(g.symbols())));
        assert!(
            (posterior.log_evidence - z.ln()).abs() < 0.1,
            "log evidence {} vs exact {}",
            posterior.log_evidence,
            z.ln()
        );

        let mean_len = posterior
            .expectation(&model_fn, |g| g.len() as f64)
            .unwrap();
        let exact_len: f64 = exact.iter().map(|(s, p)| s.len() as f64 * p).sum();
        assert!(
            (mean_len - exact_len).abs() < 0.05,
            "{mean_len} vs {exact_len}"
        );

        let inclusion = posterior.marginal_inclusion(&model_fn, |g| g.symbols().to_vec());
        assert!(inclusion.iter().all(|(symbol, _)| *symbol != Sym::B));
        for (symbol, q) in inclusion {
            let exact_q: f64 = exact
                .iter()
                .filter(|(s, _)| s.contains(&symbol))
                .map(|(_, p)| p)
                .sum();
            assert!((q - exact_q).abs() < 0.04, "{symbol:?}: {q} vs {exact_q}");
        }

        // One evaluation per canonical form, however many particles, rungs
        // and replays asked for it. The forms are those of the sentences
        // and of their prefixes: a rejected off-support proposal ends its
        // execution early and scores the prefix it reached.
        let mut keys: Vec<Vec<u8>> = enumerate()
            .keys()
            .filter(|s| !s.contains(&Sym::B))
            .flat_map(|s| (0..=s.len()).map(|n| s[..n].to_vec()))
            .map(|s| {
                let mut k: Vec<u8> = s.iter().map(|x| *x as u8).collect();
                k.sort_unstable();
                k
            })
            .collect();
        keys.sort();
        keys.dedup();
        assert_eq!(likelihood.evaluations(), likelihood.cache_len());
        assert!(likelihood.cache_len() <= keys.len());
        assert!(keys.len() < exact.len(), "reorderings share a key");
        assert!(likelihood.hits() > 10 * likelihood.evaluations());
        assert!(likelihood.invalid() > 0);
    }

    /// Annealing past β = 1 with the suffix kernel concentrates on the
    /// maximiser of the likelihood.
    #[test]
    fn test_anneal_with_suffix_crossover_finds_the_optimum() {
        let model = EvolutionModel::new(prior(), Taste(1.0));
        let mut rng = StdRng::seed_from_u64(33);
        let mut kernel = PrefixKeyedSuffixCrossover::by_state(prior(), 100);
        let posterior = EvolutionSMC::anneal_with_kernel(
            &mut rng,
            &model,
            EvoSmcConfig {
                num_particles: 200,
                crossover: None,
                ..EvoSmcConfig::default()
            },
            40.0,
            8,
            &mut kernel,
        );
        let best = enumerate()
            .keys()
            .map(|s| taste(s))
            .fold(f64::NEG_INFINITY, f64::max);
        let (_g, f) = posterior.best(&Taste(1.0), &model.smc_model()).unwrap();
        assert!((f - best).abs() < 1e-12, "best found {f}, optimum {best}");
    }

    /// Admissible cut pairs of two sequences, counted from the grammar
    /// independently of the kernel.
    fn count_cuts<K: PartialEq>(a: &[Sym], b: &[Sym], key: &dyn Fn(&u8) -> K) -> usize {
        let keys = |s: &[Sym]| -> Vec<K> {
            let mut depth = 0u8;
            let mut out = vec![key(&depth)];
            for symbol in s {
                depth = Stack.transition(&depth, symbol);
                out.push(key(&depth));
            }
            out
        };
        let (ka, kb) = (keys(a), keys(b));
        let mut n = 0;
        for i in 0..ka.len() {
            for j in 0..kb.len() {
                let trivial = (i == 0 && j == 0) || (i == a.len() && j == b.len());
                if !trivial && ka[i] == kb[j] {
                    n += 1;
                }
            }
        }
        n
    }

    /// **Exact detailed balance** of the suffix crossover, with no sampling:
    /// for every ordered pair of sequences of the language and every
    /// admissible cut pair, the kernel's log acceptance ratio equals
    ///
    /// ```text
    /// ln π(a′)π(b′) − ln π(a)π(b) + ln n(a, b) − ln n(a′, b′)
    /// ```
    ///
    /// computed from the enumerated probabilities, the likelihood and an
    /// independent count of the cut pairs; the children are exactly
    /// `a[..i] ++ b[j..]` and `b[..j] ++ a[i..]`; and a proposal is refused
    /// exactly when a child is outside the language. With proposal
    /// probability `1/n` and acceptance `min(1, α)` that is detailed balance
    /// with respect to `π ⊗ π`. Returns (accepted-shape, refused) counts.
    fn exact_detailed_balance<K: PartialEq + 'static>(
        sharpness: f64,
        beta: f64,
        key: fn(&u8) -> K,
    ) -> (usize, usize) {
        let prior = prior();
        let model = EvolutionModel::new(prior.clone(), Taste(sharpness));
        let model_fn = model.smc_model();
        let kernel = PrefixKeyedSuffixCrossover::new(prior.clone(), 1, key);
        let exact = enumerate();
        let log_target = |s: &[Sym]| exact[s].ln() + beta * sharpness * taste(s);
        let chain = EvolutionChain::new(model.clone());
        let states: Vec<(Vec<Sym>, Trace)> = exact
            .keys()
            .map(|s| {
                let trace = chain
                    .init_from(&SequenceGenome::new(s.clone()))
                    .expect("in the language");
                (s.clone(), trace)
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(0);
        let (mut valid, mut refused) = (0, 0);
        for (a, ta) in &states {
            for (b, tb) in &states {
                let keys_a = kernel.read(ta).unwrap().1;
                let keys_b = kernel.read(tb).unwrap().1;
                let cuts = PrefixKeyedSuffixCrossover::<Stack, K>::cuts(&keys_a, &keys_b);
                assert_eq!(cuts.len(), count_cuts(a, b, &key), "{a:?} {b:?}");
                for &(i, j) in &cuts {
                    let child_a: Vec<Sym> = a[..i].iter().chain(&b[j..]).copied().collect();
                    let child_b: Vec<Sym> = b[..j].iter().chain(&a[i..]).copied().collect();
                    let proposal = kernel.propose::<SequenceGenome<Sym>>(
                        (ta, a),
                        (tb, b),
                        (i, j),
                        cuts.len(),
                        &mut rng,
                        &model_fn,
                        beta,
                    );
                    let in_language = exact.contains_key(&child_a) && exact.contains_key(&child_b);
                    let Some((ca, cb, log_alpha)) = proposal else {
                        assert!(!in_language, "refused {a:?} x {b:?} at {i},{j}");
                        refused += 1;
                        continue;
                    };
                    assert!(in_language, "accepted {child_a:?} / {child_b:?}");
                    assert_eq!(prior.decode(&ca).unwrap().symbols(), &child_a[..]);
                    assert_eq!(prior.decode(&cb).unwrap().symbols(), &child_b[..]);
                    let expected =
                        log_target(&child_a) + log_target(&child_b) - log_target(a) - log_target(b)
                            + (cuts.len() as f64).ln()
                            - (count_cuts(&child_a, &child_b, &key) as f64).ln();
                    assert!(
                        (log_alpha - expected).abs() < 1e-9,
                        "{a:?} x {b:?} at {i},{j}: ln α = {log_alpha}, expected {expected}"
                    );
                    valid += 1;
                }
            }
        }
        (valid, refused)
    }

    /// State-keyed cuts: the only refused grafts are those `max_len` cuts
    /// short. Checked under the prior, the posterior and a tempered target.
    #[test]
    fn test_suffix_crossover_detailed_balance_is_exact() {
        for (sharpness, beta) in [(0.0, 1.0), (1.0, 1.0), (SHARP, 0.3)] {
            let (valid, refused) = exact_detailed_balance(sharpness, beta, |depth: &u8| *depth);
            assert!(
                valid > 1000 && refused > 0,
                "{valid} valid, {refused} refused"
            );
        }
    }

    /// A key that does not determine the state (here: no information at
    /// all) proposes ill-typed grafts and makes the number of cut pairs
    /// differ between parents and children. The ill-typed grafts are
    /// refused and the acceptance ratio carries the cut-count correction,
    /// so the move is still exact.
    #[test]
    fn test_suffix_crossover_is_exact_for_a_coarse_key() {
        let (valid, refused) = exact_detailed_balance(1.0, 1.0, |_depth: &u8| ());
        assert!(
            valid > 1000 && refused > 1000,
            "{valid} valid, {refused} refused"
        );
    }

    /// Two particles, alternating one suffix-crossover proposal with one MH
    /// step each (the crossover alone is not ergodic): each particle's
    /// marginal is the exact posterior.
    #[test]
    fn test_suffix_crossover_with_mh_recovers_the_exact_posterior() {
        let model = EvolutionModel::new(prior(), Taste(1.0));
        let prior = prior();
        let mut kernel = PrefixKeyedSuffixCrossover::by_state(prior.clone(), 1);
        let mut chain = EvolutionChain::new(model.clone());
        let mut rng = StdRng::seed_from_u64(42);
        let mut particles: Vec<Particle> = (0..2)
            .map(|_| Particle {
                trace: chain.init(&mut rng),
                weight: 0.5,
                log_weight: (0.5f64).ln(),
            })
            .collect();
        // β = 1, so the kernel's target is the chain's.
        let model_fn = model.smc_model();
        let mut samples = HashMap::new();
        for _ in 0..STEPS / 2 {
            PopulationKernel::<SequenceGenome<Sym>>::sweep(
                &mut kernel,
                &mut rng,
                &mut particles,
                &model_fn,
                1.0,
            );
            for p in &mut particles {
                if let Some((_g, trace, _lw)) = chain.step_scored(&mut rng, &p.trace) {
                    p.trace = trace;
                }
                count(&mut samples, &prior.decode(&p.trace).unwrap(), 1.0);
            }
        }
        assert!(kernel.accepted() > 1_000 && kernel.accepted() < kernel.proposed());
        let tv = total_variation(&samples, &exact_posterior(1.0).0);
        assert!(tv < 0.02, "suffix crossover with MH: total variation {tv}");
    }

    /// A child keeps the sites of other programs that its particle held
    /// (a latent of the likelihood), and only those.
    #[test]
    fn test_children_keep_their_particles_foreign_sites() {
        let prior = prior();
        let kernel = PrefixKeyedSuffixCrossover::by_state(prior.clone(), 1);
        let mut owner = prior.trace_of(&SequenceGenome::new(vec![Sym::C, Sym::B]));
        owner.insert_choice(fugue::addr!("sigma"), ChoiceValue::F64(0.5), 0.0);
        let child = kernel
            .with_sequence(&owner, vec![Sym::C, Sym::A])
            .expect("in the language");
        assert_eq!(
            prior.decode(&child).unwrap(),
            SequenceGenome::new(vec![Sym::C, Sym::A])
        );
        assert_eq!(child.get_f64(&fugue::addr!("sigma")), Some(0.5));
        assert_eq!(
            child.choices.len(),
            prior
                .trace_of(&SequenceGenome::new(vec![Sym::C, Sym::A]))
                .choices
                .len()
                + 1
        );
        assert!(kernel.with_sequence(&owner, vec![Sym::Pop]).is_none());
    }
}
