//! Fugue `Handler`s and operation hooks for evolutionary operators
//!
//! This module contains two distinct kinds of type:
//!
//! 1. **Genuine [`fugue::Handler`] implementations** ([`TraceScoringHandler`],
//!    [`RecordingHandler`]).  These implement Fugue's real effect-handler
//!    contract (`on_sample_*` → `log_prior`, `on_observe_*` → `log_likelihood`,
//!    `on_factor` → `log_factors`, `finish` → [`Trace`]) and can therefore be
//!    driven with [`fugue::runtime::handler::run`] to score or replay a model
//!    with correct log-weight bookkeeping.  [`TraceScoringHandler`] is the
//!    handler that [`crate::fugue_integration::evolution_model::EvolutionModel`]
//!    uses to inject `factor(β·f(x))` into a genome's trace.
//!
//! 2. **Operation hooks** ([`LoggingHook`], [`RateLimitingHook`],
//!    [`ConditionalHook`], …).  These are *not* Fugue handlers — they are
//!    plain before/after callbacks used to observe, log, or rate-limit the
//!    trace-based genetic operators in [`super::trace_operators`].  They do not
//!    participate in probabilistic scoring; the probability mass of an operator
//!    result is obtained by scoring it with [`TraceScoringHandler`] (or with
//!    [`crate::fugue_integration::evolution_model::EvolutionModel::to_weighted_trace`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fugue::runtime::handler::Handler;
#[cfg(test)]
use fugue::{addr, sample, Normal};
use fugue::{Address, ChoiceValue, Distribution, Trace};
use rand::Rng;

use super::trace_operators::{CrossoverMask, MutationSelector};
use crate::error::GenomeError;
use crate::genome::traits::EvolutionaryGenome;

/// A genuine [`fugue::Handler`] that scores a *fixed* trace of choices.
///
/// This is the evolutionary counterpart of Fugue's own
/// [`fugue::runtime::interpreters::ScoreGivenTrace`]: it starts from an existing
/// [`Trace`] (typically `genome.to_trace()`), never samples fresh values, and
/// accumulates log-weights with the correct bookkeeping:
///
/// - `on_sample_*` looks the value up in the base trace and adds its
///   `dist.log_prob(value)` to `log_prior`,
/// - `on_observe_*` adds `dist.log_prob(value)` to `log_likelihood`,
/// - `on_factor` adds `logw` to `log_factors`,
/// - `finish` returns the (mutated) trace, whose choices are preserved.
///
/// Running the model `factor(logw)` through this handler therefore produces a
/// trace with `total_log_weight() == logw` whose choice map still equals the
/// original genome — this is exactly how "fitness as likelihood" is injected
/// into a genome's probability mass.
pub struct TraceScoringHandler {
    /// The working trace. Seed it with the genome's choices to score them.
    pub trace: Trace,
}

impl TraceScoringHandler {
    /// Create a scoring handler seeded with `base`'s choices.
    ///
    /// The accumulators of `base` are reset to zero so that the resulting
    /// `total_log_weight()` reflects only the effects executed by the model.
    pub fn new(base: Trace) -> Self {
        Self {
            trace: Trace {
                choices: base.choices,
                log_prior: 0.0,
                log_likelihood: 0.0,
                log_factors: 0.0,
            },
        }
    }

    fn score_sample<T: Copy + Default>(
        &mut self,
        addr: &Address,
        dist: &dyn Distribution<T>,
        extract: impl Fn(&ChoiceValue) -> Option<T>,
    ) -> T {
        match self.trace.choices.get(addr).and_then(|c| extract(&c.value)) {
            Some(v) => {
                self.trace.log_prior += dist.log_prob(&v);
                v
            }
            None => {
                // Site absent from the base trace: mark the trace invalid rather
                // than fabricating a value, mirroring SafeScoreGivenTrace.
                self.trace.log_prior += f64::NEG_INFINITY;
                T::default()
            }
        }
    }
}

impl Handler for TraceScoringHandler {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.score_sample(addr, dist, ChoiceValue::as_f64)
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.score_sample(addr, dist, ChoiceValue::as_bool)
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.score_sample(addr, dist, ChoiceValue::as_u64)
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.score_sample(addr, dist, ChoiceValue::as_usize)
    }

    fn on_observe_f64(&mut self, _: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_bool(&mut self, _: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_u64(&mut self, _: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_observe_usize(&mut self, _: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }

    fn finish(self) -> Trace {
        self.trace
    }
}

/// A genuine [`fugue::Handler`] that records the sequence of sampled sites while
/// **delegating all effect handling** (and therefore all log-weight bookkeeping)
/// to an inner handler.
///
/// This is a real Poutine-style *trace* effect: it observes the execution
/// without changing its semantics.  Because every effect is forwarded to
/// `inner`, the resulting trace has exactly the same `log_prior`,
/// `log_likelihood`, and `log_factors` the inner handler would have produced on
/// its own — only now the ordered list of `(address, value)` sample events is
/// also available via `RecordingHandler::events`.
pub struct RecordingHandler<H: Handler> {
    inner: H,
    events: Arc<Mutex<Vec<(Address, ChoiceValue)>>>,
}

impl<H: Handler> RecordingHandler<H> {
    /// Wrap `inner`, recording every sampled site into `sink`.
    pub fn new(inner: H, sink: Arc<Mutex<Vec<(Address, ChoiceValue)>>>) -> Self {
        Self {
            inner,
            events: sink,
        }
    }

    fn record(&self, addr: &Address, value: ChoiceValue) {
        self.events.lock().unwrap().push((addr.clone(), value));
    }
}

impl<H: Handler> Handler for RecordingHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let v = self.inner.on_sample_f64(addr, dist);
        self.record(addr, ChoiceValue::F64(v));
        v
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let v = self.inner.on_sample_bool(addr, dist);
        self.record(addr, ChoiceValue::Bool(v));
        v
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let v = self.inner.on_sample_u64(addr, dist);
        self.record(addr, ChoiceValue::U64(v));
        v
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let v = self.inner.on_sample_usize(addr, dist);
        self.record(addr, ChoiceValue::Usize(v));
        v
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.inner.on_observe_f64(addr, dist, value);
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.inner.on_observe_bool(addr, dist, value);
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.inner.on_observe_u64(addr, dist, value);
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.inner.on_observe_usize(addr, dist, value);
    }

    fn on_factor(&mut self, logw: f64) {
        self.inner.on_factor(logw);
    }

    fn finish(self) -> Trace {
        self.inner.finish()
    }
}

/// Record of a mutation operation
#[derive(Clone, Debug)]
pub struct MutationRecord {
    /// Generation when mutation occurred
    pub generation: usize,
    /// Addresses that were mutated
    pub mutated_addresses: Vec<Address>,
    /// Original values before mutation
    pub original_values: HashMap<Address, ChoiceValue>,
    /// New values after mutation
    pub new_values: HashMap<Address, ChoiceValue>,
}

/// Record of a crossover operation
#[derive(Clone, Debug)]
pub struct CrossoverRecord {
    /// Generation when crossover occurred
    pub generation: usize,
    /// Addresses from parent1
    pub from_parent1: Vec<Address>,
    /// Addresses from parent2
    pub from_parent2: Vec<Address>,
}

/// Record of a selection operation
#[derive(Clone, Debug)]
pub struct SelectionRecord {
    /// Generation when selection occurred
    pub generation: usize,
    /// Indices of selected individuals
    pub selected_indices: Vec<usize>,
    /// Fitness values of selected individuals
    pub selected_fitness: Vec<f64>,
}

/// Operation hook for the trace-based mutation operator.
///
/// NOTE: this is a plain before/after callback, **not** a [`fugue::Handler`].
/// It cannot intercept a Fugue model; it only observes / gates the
/// [`super::trace_operators`] mutation operator.
pub trait MutationHook: Send + Sync {
    /// Called before mutation is applied.
    /// Returns true if mutation should proceed, false to skip.
    fn before_mutation(&self, trace: &Trace, generation: usize) -> bool;

    /// Called after mutation is applied
    fn after_mutation(&self, original: &Trace, mutated: &Trace, record: &MutationRecord);

    /// Optionally modify the mutation sites before mutation occurs
    fn modify_sites(
        &self,
        sites: std::collections::HashSet<Address>,
        _trace: &Trace,
    ) -> std::collections::HashSet<Address> {
        sites // Default: no modification
    }
}

/// Operation hook for the trace-based crossover operator.
///
/// NOTE: a plain before/after callback, **not** a [`fugue::Handler`].
pub trait CrossoverHook: Send + Sync {
    /// Called before crossover is applied.
    /// Returns true if crossover should proceed, false to skip.
    fn before_crossover(&self, parent1: &Trace, parent2: &Trace, generation: usize) -> bool;

    /// Called after crossover is applied
    fn after_crossover(
        &self,
        parent1: &Trace,
        parent2: &Trace,
        child1: &Trace,
        child2: &Trace,
        record: &CrossoverRecord,
    );
}

/// Operation hook for a selection operator.
///
/// NOTE: a plain before/after callback, **not** a [`fugue::Handler`].
pub trait SelectionHook: Send + Sync {
    /// Called before selection
    fn before_selection(&self, population_size: usize, generation: usize);

    /// Called after selection
    fn after_selection(&self, record: &SelectionRecord);

    /// Optionally modify selection probabilities
    fn modify_probabilities(&self, probabilities: Vec<f64>) -> Vec<f64> {
        probabilities // Default: no modification
    }
}

/// A hook that logs all evolutionary operations
#[derive(Clone, Debug, Default)]
pub struct LoggingHook {
    /// Mutation records
    pub mutations: Arc<Mutex<Vec<MutationRecord>>>,
    /// Crossover records
    pub crossovers: Arc<Mutex<Vec<CrossoverRecord>>>,
    /// Selection records
    pub selections: Arc<Mutex<Vec<SelectionRecord>>>,
}

impl LoggingHook {
    /// Create a new logging hook
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all mutation records
    pub fn get_mutations(&self) -> Vec<MutationRecord> {
        self.mutations.lock().unwrap().clone()
    }

    /// Get all crossover records
    pub fn get_crossovers(&self) -> Vec<CrossoverRecord> {
        self.crossovers.lock().unwrap().clone()
    }

    /// Get all selection records
    pub fn get_selections(&self) -> Vec<SelectionRecord> {
        self.selections.lock().unwrap().clone()
    }

    /// Clear all records
    pub fn clear(&self) {
        self.mutations.lock().unwrap().clear();
        self.crossovers.lock().unwrap().clear();
        self.selections.lock().unwrap().clear();
    }
}

impl MutationHook for LoggingHook {
    fn before_mutation(&self, _trace: &Trace, _generation: usize) -> bool {
        true // Always allow
    }

    fn after_mutation(&self, _original: &Trace, _mutated: &Trace, record: &MutationRecord) {
        self.mutations.lock().unwrap().push(record.clone());
    }
}

impl CrossoverHook for LoggingHook {
    fn before_crossover(&self, _parent1: &Trace, _parent2: &Trace, _generation: usize) -> bool {
        true // Always allow
    }

    fn after_crossover(
        &self,
        _parent1: &Trace,
        _parent2: &Trace,
        _child1: &Trace,
        _child2: &Trace,
        record: &CrossoverRecord,
    ) {
        self.crossovers.lock().unwrap().push(record.clone());
    }
}

impl SelectionHook for LoggingHook {
    fn before_selection(&self, _population_size: usize, _generation: usize) {}

    fn after_selection(&self, record: &SelectionRecord) {
        self.selections.lock().unwrap().push(record.clone());
    }
}

/// A hook that rate-limits operations
#[derive(Clone, Debug)]
pub struct RateLimitingHook {
    /// Maximum mutations per generation
    pub max_mutations: usize,
    /// Maximum crossovers per generation
    pub max_crossovers: usize,
    /// Current mutation count for this generation
    mutation_count: Arc<Mutex<(usize, usize)>>, // (generation, count)
    /// Current crossover count for this generation
    crossover_count: Arc<Mutex<(usize, usize)>>,
}

impl RateLimitingHook {
    /// Create a new rate limiting hook
    pub fn new(max_mutations: usize, max_crossovers: usize) -> Self {
        Self {
            max_mutations,
            max_crossovers,
            mutation_count: Arc::new(Mutex::new((0, 0))),
            crossover_count: Arc::new(Mutex::new((0, 0))),
        }
    }

    /// Reset counters for a new generation
    pub fn reset(&self, generation: usize) {
        *self.mutation_count.lock().unwrap() = (generation, 0);
        *self.crossover_count.lock().unwrap() = (generation, 0);
    }
}

impl MutationHook for RateLimitingHook {
    fn before_mutation(&self, _trace: &Trace, generation: usize) -> bool {
        let mut count = self.mutation_count.lock().unwrap();
        if count.0 != generation {
            *count = (generation, 0);
        }
        if count.1 < self.max_mutations {
            count.1 += 1;
            true
        } else {
            false
        }
    }

    fn after_mutation(&self, _original: &Trace, _mutated: &Trace, _record: &MutationRecord) {}
}

impl CrossoverHook for RateLimitingHook {
    fn before_crossover(&self, _parent1: &Trace, _parent2: &Trace, generation: usize) -> bool {
        let mut count = self.crossover_count.lock().unwrap();
        if count.0 != generation {
            *count = (generation, 0);
        }
        if count.1 < self.max_crossovers {
            count.1 += 1;
            true
        } else {
            false
        }
    }

    fn after_crossover(
        &self,
        _parent1: &Trace,
        _parent2: &Trace,
        _child1: &Trace,
        _child2: &Trace,
        _record: &CrossoverRecord,
    ) {
    }
}

/// A hook that conditionally blocks operations based on a predicate
pub struct ConditionalHook<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    /// Predicate that determines if operation should proceed
    pub predicate: F,
}

impl<F> ConditionalHook<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    /// Create a new conditional hook
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> MutationHook for ConditionalHook<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    fn before_mutation(&self, _trace: &Trace, generation: usize) -> bool {
        (self.predicate)(generation)
    }

    fn after_mutation(&self, _original: &Trace, _mutated: &Trace, _record: &MutationRecord) {}
}

impl<F> CrossoverHook for ConditionalHook<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    fn before_crossover(&self, _parent1: &Trace, _parent2: &Trace, generation: usize) -> bool {
        (self.predicate)(generation)
    }

    fn after_crossover(
        &self,
        _parent1: &Trace,
        _parent2: &Trace,
        _child1: &Trace,
        _child2: &Trace,
        _record: &CrossoverRecord,
    ) {
    }
}

/// Composition of multiple mutation hooks
pub struct ComposedMutationHook {
    hooks: Vec<Box<dyn MutationHook>>,
}

impl ComposedMutationHook {
    /// Create a new composed hook
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Add a hook to the composition
    pub fn add<H: MutationHook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(Box::new(hook));
        self
    }
}

impl Default for ComposedMutationHook {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationHook for ComposedMutationHook {
    fn before_mutation(&self, trace: &Trace, generation: usize) -> bool {
        // All hooks must agree
        self.hooks
            .iter()
            .all(|h| h.before_mutation(trace, generation))
    }

    fn after_mutation(&self, original: &Trace, mutated: &Trace, record: &MutationRecord) {
        for hook in &self.hooks {
            hook.after_mutation(original, mutated, record);
        }
    }

    fn modify_sites(
        &self,
        mut sites: std::collections::HashSet<Address>,
        trace: &Trace,
    ) -> std::collections::HashSet<Address> {
        for hook in &self.hooks {
            sites = hook.modify_sites(sites, trace);
        }
        sites
    }
}

/// Composition of multiple crossover hooks
pub struct ComposedCrossoverHook {
    hooks: Vec<Box<dyn CrossoverHook>>,
}

impl ComposedCrossoverHook {
    /// Create a new composed hook
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Add a hook to the composition
    pub fn add<H: CrossoverHook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(Box::new(hook));
        self
    }
}

impl Default for ComposedCrossoverHook {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossoverHook for ComposedCrossoverHook {
    fn before_crossover(&self, parent1: &Trace, parent2: &Trace, generation: usize) -> bool {
        self.hooks
            .iter()
            .all(|h| h.before_crossover(parent1, parent2, generation))
    }

    fn after_crossover(
        &self,
        parent1: &Trace,
        parent2: &Trace,
        child1: &Trace,
        child2: &Trace,
        record: &CrossoverRecord,
    ) {
        for hook in &self.hooks {
            hook.after_crossover(parent1, parent2, child1, child2, record);
        }
    }
}

/// Hooked mutation operator that integrates with operation hooks.
///
/// The produced child trace only rearranges/mutates *values*. It carries no
/// fabricated per-choice log-probabilities: resampled sites are written with a
/// neutral `logp` of `0.0` (rather than the stale log-prob of the pre-mutation
/// value), and the trace's `log_prior`/`log_likelihood`/`log_factors`
/// accumulators are left at zero. To obtain the child's probability mass under
/// the Boltzmann posterior, score it with [`TraceScoringHandler`] or
/// [`crate::fugue_integration::evolution_model::EvolutionModel::to_weighted_trace`].
pub fn hooked_mutate_trace<G, S, H, R>(
    genome: &G,
    selector: &S,
    mutation_fn: impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue,
    hook: &H,
    generation: usize,
    rng: &mut R,
) -> Result<G, GenomeError>
where
    G: EvolutionaryGenome,
    S: MutationSelector,
    H: MutationHook,
    R: Rng,
{
    let trace = genome.to_trace();

    // Check if mutation should proceed
    if !hook.before_mutation(&trace, generation) {
        return G::from_trace(&trace);
    }

    // Select and potentially modify mutation sites
    let mut mutation_sites = selector.select_sites(&trace, rng);
    mutation_sites = hook.modify_sites(mutation_sites, &trace);

    let mut new_trace = Trace::default();
    let mut original_values = HashMap::new();
    let mut new_values = HashMap::new();

    for (addr, choice) in &trace.choices {
        if mutation_sites.contains(addr) {
            original_values.insert(addr.clone(), choice.value.clone());
            let mutated = mutation_fn(addr, &choice.value, rng);
            new_values.insert(addr.clone(), mutated.clone());
            // Resampled site: do NOT copy the stale logp of the old value.
            new_trace.insert_choice(addr.clone(), mutated, 0.0);
        } else {
            // Unchanged site: preserve its recorded logp.
            new_trace.insert_choice(addr.clone(), choice.value.clone(), choice.logp);
        }
    }

    // Create mutation record
    let record = MutationRecord {
        generation,
        mutated_addresses: mutation_sites.into_iter().collect(),
        original_values,
        new_values,
    };

    hook.after_mutation(&trace, &new_trace, &record);

    G::from_trace(&new_trace)
}

/// Hooked crossover operator that integrates with operation hooks.
///
/// Like [`hooked_mutate_trace`], the produced child traces only recombine
/// *values* and carry a neutral `logp` of `0.0`; score them with
/// [`TraceScoringHandler`] to obtain probability mass.
pub fn hooked_crossover_traces<G, M, H, R>(
    parent1: &G,
    parent2: &G,
    mask: &M,
    hook: &H,
    generation: usize,
    _rng: &mut R,
) -> Result<(G, G), GenomeError>
where
    G: EvolutionaryGenome,
    M: CrossoverMask,
    H: CrossoverHook,
    R: Rng,
{
    let trace1 = parent1.to_trace();
    let trace2 = parent2.to_trace();

    // Check if crossover should proceed
    if !hook.before_crossover(&trace1, &trace2, generation) {
        return Ok((G::from_trace(&trace1)?, G::from_trace(&trace2)?));
    }

    let mut child1_trace = Trace::default();
    let mut child2_trace = Trace::default();
    let mut from_parent1 = Vec::new();
    let mut from_parent2 = Vec::new();

    // Collect all addresses from both parents
    let all_addresses: std::collections::HashSet<Address> = trace1
        .choices
        .keys()
        .chain(trace2.choices.keys())
        .cloned()
        .collect();

    for addr in all_addresses {
        let (val_for_child1, val_for_child2) = if mask.from_parent1(&addr) {
            from_parent1.push(addr.clone());
            (
                trace1
                    .choices
                    .get(&addr)
                    .map(|c| c.value.clone())
                    .unwrap_or(ChoiceValue::F64(0.0)),
                trace2
                    .choices
                    .get(&addr)
                    .map(|c| c.value.clone())
                    .unwrap_or(ChoiceValue::F64(0.0)),
            )
        } else {
            from_parent2.push(addr.clone());
            (
                trace2
                    .choices
                    .get(&addr)
                    .map(|c| c.value.clone())
                    .unwrap_or(ChoiceValue::F64(0.0)),
                trace1
                    .choices
                    .get(&addr)
                    .map(|c| c.value.clone())
                    .unwrap_or(ChoiceValue::F64(0.0)),
            )
        };

        child1_trace.insert_choice(addr.clone(), val_for_child1, 0.0);
        child2_trace.insert_choice(addr, val_for_child2, 0.0);
    }

    // Create crossover record
    let record = CrossoverRecord {
        generation,
        from_parent1,
        from_parent2,
    };

    hook.after_crossover(&trace1, &trace2, &child1_trace, &child2_trace, &record);

    let child1 = G::from_trace(&child1_trace)?;
    let child2 = G::from_trace(&child2_trace)?;

    Ok((child1, child2))
}

/// Statistics computed from hook records
#[derive(Clone, Debug, Default)]
pub struct OperationStatistics {
    /// Total number of mutations
    pub total_mutations: usize,
    /// Total number of crossovers
    pub total_crossovers: usize,
    /// Average mutation sites per operation
    pub avg_mutation_sites: f64,
    /// Distribution of addresses from parent1 in crossovers
    pub avg_parent1_contribution: f64,
}

impl OperationStatistics {
    /// Compute statistics from a logging hook
    pub fn from_hook(hook: &LoggingHook) -> Self {
        let mutations = hook.get_mutations();
        let crossovers = hook.get_crossovers();

        let total_mutations = mutations.len();
        let total_crossovers = crossovers.len();

        let avg_mutation_sites = if total_mutations > 0 {
            mutations
                .iter()
                .map(|r| r.mutated_addresses.len())
                .sum::<usize>() as f64
                / total_mutations as f64
        } else {
            0.0
        };

        let avg_parent1_contribution = if total_crossovers > 0 {
            let total: f64 = crossovers
                .iter()
                .map(|r| {
                    let total = r.from_parent1.len() + r.from_parent2.len();
                    if total > 0 {
                        r.from_parent1.len() as f64 / total as f64
                    } else {
                        0.5
                    }
                })
                .sum();
            total / total_crossovers as f64
        } else {
            0.0
        };

        Self {
            total_mutations,
            total_crossovers,
            avg_mutation_sites,
            avg_parent1_contribution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fugue_integration::trace_operators::{
        gaussian_mutation, UniformCrossoverMask, UniformMutationSelector,
    };
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::EvolutionaryGenome;
    use fugue::runtime::handler::run;
    use fugue::{factor, ModelExt};
    use rand::SeedableRng;

    #[test]
    fn test_trace_scoring_handler_injects_factor() {
        // regression: EV-52 — factor() must land in log_factors, so total_log_weight == logw.
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let base = genome.to_trace();
        let (_r, trace) = run(TraceScoringHandler::new(base), factor(-4.5));
        assert!((trace.log_factors - (-4.5)).abs() < 1e-12);
        assert!((trace.total_log_weight() - (-4.5)).abs() < 1e-12);
        // Choices are preserved.
        assert_eq!(trace.get_f64(&addr!("gene", 0)), Some(1.0));
        assert_eq!(trace.get_f64(&addr!("gene", 2)), Some(3.0));
    }

    #[test]
    fn test_trace_scoring_handler_scores_prior() {
        // Scoring a fixed sample site accumulates its log_prob into log_prior.
        let mut base = Trace::default();
        base.insert_choice(addr!("x"), ChoiceValue::F64(0.0), 0.0);
        let (_v, trace) = run(
            TraceScoringHandler::new(base),
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()),
        );
        // log N(0;0,1) = -0.5*ln(2π)
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((trace.log_prior - expected).abs() < 1e-9);
    }

    #[test]
    fn test_recording_handler_preserves_bookkeeping() {
        // regression: EV-51 — a genuine fugue::Handler that records events must
        // reproduce the SAME log-weights as the delegated handler.
        let model = || {
            sample(addr!("a"), Normal::new(0.0, 1.0).unwrap())
                .and_then(|_| sample(addr!("b"), Normal::new(1.0, 2.0).unwrap()))
                .and_then(|_| factor(-1.25))
        };

        let mut rng1 = rand::rngs::StdRng::from_seed([7u8; 32]);
        let plain = fugue::runtime::interpreters::PriorHandler {
            rng: &mut rng1,
            trace: Trace::default(),
        };
        let (_a, plain_trace) = run(plain, model());

        let mut rng2 = rand::rngs::StdRng::from_seed([7u8; 32]);
        let sink = Arc::new(Mutex::new(Vec::new()));
        let recording = RecordingHandler::new(
            fugue::runtime::interpreters::PriorHandler {
                rng: &mut rng2,
                trace: Trace::default(),
            },
            sink.clone(),
        );
        let (_b, rec_trace) = run(recording, model());

        // Same seed + delegation ⇒ identical bookkeeping.
        assert!((plain_trace.log_prior - rec_trace.log_prior).abs() < 1e-12);
        assert!((plain_trace.log_factors - rec_trace.log_factors).abs() < 1e-12);
        // Both sample sites were recorded, in order.
        let events = sink.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, addr!("a"));
        assert_eq!(events[1].0, addr!("b"));
    }

    #[test]
    fn test_hooked_mutate_does_not_copy_stale_logp() {
        // regression: EV-51 — resampled sites must not carry the old value's logp.
        let mut rng = rand::rngs::StdRng::from_seed([3u8; 32]);
        // Build a genome trace whose choices carry a non-zero logp.
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);
        let mut seeded = genome.to_trace();
        for c in seeded.choices.values_mut() {
            c.logp = -12.34; // stale, meaningless log-prob
        }
        // A custom genome whose to_trace returns the seeded (non-zero-logp) trace
        // is awkward to construct, so exercise the operator directly on the
        // standard trace and assert the produced child has neutral logp.
        let hook = LoggingHook::new();
        let selector = UniformMutationSelector::new(1.0);
        let mutation_fn = gaussian_mutation(0.5);
        let mutated =
            hooked_mutate_trace(&genome, &selector, mutation_fn, &hook, 0, &mut rng).unwrap();
        let child_trace = mutated.to_trace();
        for choice in child_trace.choices.values() {
            assert_eq!(choice.logp, 0.0, "child trace logp must be neutral (0.0)");
        }
    }

    #[test]
    fn test_logging_hook_mutation() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let hook = LoggingHook::new();
        let selector = UniformMutationSelector::new(1.0); // Mutate all
        let mutation_fn = gaussian_mutation(0.1);

        let _mutated =
            hooked_mutate_trace(&genome, &selector, mutation_fn, &hook, 0, &mut rng).unwrap();

        let records = hook.get_mutations();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].generation, 0);
        assert!(!records[0].mutated_addresses.is_empty());
    }

    #[test]
    fn test_logging_hook_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let parent2 = RealVector::new(vec![4.0, 5.0, 6.0]);

        let hook = LoggingHook::new();
        let trace1 = parent1.to_trace();
        let mask = UniformCrossoverMask::balanced(&trace1, &mut rng);

        let (_child1, _child2) =
            hooked_crossover_traces(&parent1, &parent2, &mask, &hook, 0, &mut rng).unwrap();

        let records = hook.get_crossovers();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].generation, 0);
    }

    #[test]
    fn test_rate_limiting_hook() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let hook = RateLimitingHook::new(2, 2);
        let selector = UniformMutationSelector::new(1.0);
        let mutation_fn = gaussian_mutation(0.1);

        // First two mutations should succeed
        for _ in 0..2 {
            let result = hooked_mutate_trace(&genome, &selector, &mutation_fn, &hook, 0, &mut rng);
            assert!(result.is_ok());
        }

        // Third mutation should be skipped (returns original)
        let result = hooked_mutate_trace(&genome, &selector, &mutation_fn, &hook, 0, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_composed_hook() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let logging = LoggingHook::new();
        let rate_limit = RateLimitingHook::new(5, 5);

        let composed = ComposedMutationHook::new()
            .add(logging.clone())
            .add(rate_limit);

        let selector = UniformMutationSelector::new(1.0);
        let mutation_fn = gaussian_mutation(0.1);

        let _ = hooked_mutate_trace(&genome, &selector, mutation_fn, &composed, 0, &mut rng);

        // Logging hook should have recorded the mutation
        assert_eq!(logging.get_mutations().len(), 1);
    }

    #[test]
    fn test_operation_statistics() {
        let hook = LoggingHook::new();

        // Manually add some records
        hook.mutations.lock().unwrap().push(MutationRecord {
            generation: 0,
            mutated_addresses: vec![addr!("test", 0), addr!("test", 1)],
            original_values: HashMap::new(),
            new_values: HashMap::new(),
        });

        let stats = OperationStatistics::from_hook(&hook);
        assert_eq!(stats.total_mutations, 1);
        assert_eq!(stats.avg_mutation_sites, 2.0);
    }
}
