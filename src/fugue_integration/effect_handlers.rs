//! Poutine-style effect handlers for evolutionary operations
//!
//! Effect handlers intercept and transform evolutionary operations, enabling:
//! - Logging and tracing of all genetic operations
//! - Conditional modification of operator behavior
//! - Composition of effects (e.g., logging + rate limiting)
//! - Replay of evolutionary traces for debugging

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fugue::{Address, ChoiceValue, Trace};
#[cfg(test)]
use fugue::addr;
use rand::Rng;

use crate::error::GenomeError;
use crate::genome::traits::EvolutionaryGenome;
use super::trace_operators::{MutationSelector, CrossoverMask};

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

/// Effect handler trait for mutation operations
pub trait MutationHandler: Send + Sync {
    /// Called before mutation is applied
    /// Returns true if mutation should proceed, false to skip
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

/// Effect handler trait for crossover operations
pub trait CrossoverHandler: Send + Sync {
    /// Called before crossover is applied
    /// Returns true if crossover should proceed, false to skip
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

/// Effect handler trait for selection operations
pub trait SelectionHandler: Send + Sync {
    /// Called before selection
    fn before_selection(&self, population_size: usize, generation: usize);

    /// Called after selection
    fn after_selection(&self, record: &SelectionRecord);

    /// Optionally modify selection probabilities
    fn modify_probabilities(&self, probabilities: Vec<f64>) -> Vec<f64> {
        probabilities // Default: no modification
    }
}

/// A handler that logs all evolutionary operations
#[derive(Clone, Debug, Default)]
pub struct LoggingHandler {
    /// Mutation records
    pub mutations: Arc<Mutex<Vec<MutationRecord>>>,
    /// Crossover records
    pub crossovers: Arc<Mutex<Vec<CrossoverRecord>>>,
    /// Selection records
    pub selections: Arc<Mutex<Vec<SelectionRecord>>>,
}

impl LoggingHandler {
    /// Create a new logging handler
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

impl MutationHandler for LoggingHandler {
    fn before_mutation(&self, _trace: &Trace, _generation: usize) -> bool {
        true // Always allow
    }

    fn after_mutation(&self, _original: &Trace, _mutated: &Trace, record: &MutationRecord) {
        self.mutations.lock().unwrap().push(record.clone());
    }
}

impl CrossoverHandler for LoggingHandler {
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

impl SelectionHandler for LoggingHandler {
    fn before_selection(&self, _population_size: usize, _generation: usize) {}

    fn after_selection(&self, record: &SelectionRecord) {
        self.selections.lock().unwrap().push(record.clone());
    }
}

/// A handler that rate-limits operations
#[derive(Clone, Debug)]
pub struct RateLimitingHandler {
    /// Maximum mutations per generation
    pub max_mutations: usize,
    /// Maximum crossovers per generation
    pub max_crossovers: usize,
    /// Current mutation count for this generation
    mutation_count: Arc<Mutex<(usize, usize)>>, // (generation, count)
    /// Current crossover count for this generation
    crossover_count: Arc<Mutex<(usize, usize)>>,
}

impl RateLimitingHandler {
    /// Create a new rate limiting handler
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

impl MutationHandler for RateLimitingHandler {
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

impl CrossoverHandler for RateLimitingHandler {
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

/// A handler that conditionally blocks operations based on a predicate
pub struct ConditionalHandler<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    /// Predicate that determines if operation should proceed
    pub predicate: F,
}

impl<F> ConditionalHandler<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    /// Create a new conditional handler
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> MutationHandler for ConditionalHandler<F>
where
    F: Fn(usize) -> bool + Send + Sync,
{
    fn before_mutation(&self, _trace: &Trace, generation: usize) -> bool {
        (self.predicate)(generation)
    }

    fn after_mutation(&self, _original: &Trace, _mutated: &Trace, _record: &MutationRecord) {}
}

impl<F> CrossoverHandler for ConditionalHandler<F>
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

/// Composition of multiple mutation handlers
pub struct ComposedMutationHandler {
    handlers: Vec<Box<dyn MutationHandler>>,
}

impl ComposedMutationHandler {
    /// Create a new composed handler
    pub fn new() -> Self {
        Self { handlers: Vec::new() }
    }

    /// Add a handler to the composition
    pub fn add<H: MutationHandler + 'static>(mut self, handler: H) -> Self {
        self.handlers.push(Box::new(handler));
        self
    }
}

impl Default for ComposedMutationHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl MutationHandler for ComposedMutationHandler {
    fn before_mutation(&self, trace: &Trace, generation: usize) -> bool {
        // All handlers must agree
        self.handlers.iter().all(|h| h.before_mutation(trace, generation))
    }

    fn after_mutation(&self, original: &Trace, mutated: &Trace, record: &MutationRecord) {
        for handler in &self.handlers {
            handler.after_mutation(original, mutated, record);
        }
    }

    fn modify_sites(
        &self,
        mut sites: std::collections::HashSet<Address>,
        trace: &Trace,
    ) -> std::collections::HashSet<Address> {
        for handler in &self.handlers {
            sites = handler.modify_sites(sites, trace);
        }
        sites
    }
}

/// Composition of multiple crossover handlers
pub struct ComposedCrossoverHandler {
    handlers: Vec<Box<dyn CrossoverHandler>>,
}

impl ComposedCrossoverHandler {
    /// Create a new composed handler
    pub fn new() -> Self {
        Self { handlers: Vec::new() }
    }

    /// Add a handler to the composition
    pub fn add<H: CrossoverHandler + 'static>(mut self, handler: H) -> Self {
        self.handlers.push(Box::new(handler));
        self
    }
}

impl Default for ComposedCrossoverHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossoverHandler for ComposedCrossoverHandler {
    fn before_crossover(&self, parent1: &Trace, parent2: &Trace, generation: usize) -> bool {
        self.handlers
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
        for handler in &self.handlers {
            handler.after_crossover(parent1, parent2, child1, child2, record);
        }
    }
}

/// Handled mutation operator that integrates with effect handlers
pub fn handled_mutate_trace<G, S, H, R>(
    genome: &G,
    selector: &S,
    mutation_fn: impl Fn(&Address, &ChoiceValue, &mut R) -> ChoiceValue,
    handler: &H,
    generation: usize,
    rng: &mut R,
) -> Result<G, GenomeError>
where
    G: EvolutionaryGenome,
    S: MutationSelector,
    H: MutationHandler,
    R: Rng,
{
    let trace = genome.to_trace();

    // Check if mutation should proceed
    if !handler.before_mutation(&trace, generation) {
        return Ok(G::from_trace(&trace)?);
    }

    // Select and potentially modify mutation sites
    let mut mutation_sites = selector.select_sites(&trace, rng);
    mutation_sites = handler.modify_sites(mutation_sites, &trace);

    let mut new_trace = Trace::default();
    let mut original_values = HashMap::new();
    let mut new_values = HashMap::new();

    for (addr, choice) in &trace.choices {
        let new_value = if mutation_sites.contains(addr) {
            original_values.insert(addr.clone(), choice.value.clone());
            let mutated = mutation_fn(addr, &choice.value, rng);
            new_values.insert(addr.clone(), mutated.clone());
            mutated
        } else {
            choice.value.clone()
        };
        new_trace.insert_choice(addr.clone(), new_value, choice.logp);
    }

    // Create mutation record
    let record = MutationRecord {
        generation,
        mutated_addresses: mutation_sites.into_iter().collect(),
        original_values,
        new_values,
    };

    handler.after_mutation(&trace, &new_trace, &record);

    G::from_trace(&new_trace)
}

/// Handled crossover operator that integrates with effect handlers
pub fn handled_crossover_traces<G, M, H, R>(
    parent1: &G,
    parent2: &G,
    mask: &M,
    handler: &H,
    generation: usize,
    _rng: &mut R,
) -> Result<(G, G), GenomeError>
where
    G: EvolutionaryGenome,
    M: CrossoverMask,
    H: CrossoverHandler,
    R: Rng,
{
    let trace1 = parent1.to_trace();
    let trace2 = parent2.to_trace();

    // Check if crossover should proceed
    if !handler.before_crossover(&trace1, &trace2, generation) {
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
                trace1.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
                trace2.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
            )
        } else {
            from_parent2.push(addr.clone());
            (
                trace2.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
                trace1.choices.get(&addr).map(|c| c.value.clone()).unwrap_or(ChoiceValue::F64(0.0)),
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

    handler.after_crossover(&trace1, &trace2, &child1_trace, &child2_trace, &record);

    let child1 = G::from_trace(&child1_trace)?;
    let child2 = G::from_trace(&child2_trace)?;

    Ok((child1, child2))
}

/// Statistics computed from handler records
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
    /// Compute statistics from a logging handler
    pub fn from_handler(handler: &LoggingHandler) -> Self {
        let mutations = handler.get_mutations();
        let crossovers = handler.get_crossovers();

        let total_mutations = mutations.len();
        let total_crossovers = crossovers.len();

        let avg_mutation_sites = if total_mutations > 0 {
            mutations.iter().map(|r| r.mutated_addresses.len()).sum::<usize>() as f64
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
    use crate::genome::real_vector::RealVector;
    use crate::fugue_integration::trace_operators::{
        UniformMutationSelector, UniformCrossoverMask, gaussian_mutation,
    };

    #[test]
    fn test_logging_handler_mutation() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let handler = LoggingHandler::new();
        let selector = UniformMutationSelector::new(1.0); // Mutate all
        let mutation_fn = gaussian_mutation(0.1);

        let _mutated = handled_mutate_trace(
            &genome,
            &selector,
            mutation_fn,
            &handler,
            0,
            &mut rng,
        )
        .unwrap();

        let records = handler.get_mutations();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].generation, 0);
        assert!(!records[0].mutated_addresses.is_empty());
    }

    #[test]
    fn test_logging_handler_crossover() {
        let mut rng = rand::thread_rng();
        let parent1 = RealVector::new(vec![1.0, 2.0, 3.0]);
        let parent2 = RealVector::new(vec![4.0, 5.0, 6.0]);

        let handler = LoggingHandler::new();
        let trace1 = parent1.to_trace();
        let mask = UniformCrossoverMask::balanced(&trace1, &mut rng);

        let (_child1, _child2) = handled_crossover_traces(
            &parent1,
            &parent2,
            &mask,
            &handler,
            0,
            &mut rng,
        )
        .unwrap();

        let records = handler.get_crossovers();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].generation, 0);
    }

    #[test]
    fn test_rate_limiting_handler() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let handler = RateLimitingHandler::new(2, 2);
        let selector = UniformMutationSelector::new(1.0);
        let mutation_fn = gaussian_mutation(0.1);

        // First two mutations should succeed
        for _ in 0..2 {
            let result = handled_mutate_trace(
                &genome,
                &selector,
                &mutation_fn,
                &handler,
                0,
                &mut rng,
            );
            assert!(result.is_ok());
        }

        // Third mutation should be skipped (returns original)
        let result = handled_mutate_trace(
            &genome,
            &selector,
            &mutation_fn,
            &handler,
            0,
            &mut rng,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_composed_handler() {
        let mut rng = rand::thread_rng();
        let genome = RealVector::new(vec![1.0, 2.0, 3.0]);

        let logging = LoggingHandler::new();
        let rate_limit = RateLimitingHandler::new(5, 5);

        let composed = ComposedMutationHandler::new()
            .add(logging.clone())
            .add(rate_limit);

        let selector = UniformMutationSelector::new(1.0);
        let mutation_fn = gaussian_mutation(0.1);

        let _ = handled_mutate_trace(
            &genome,
            &selector,
            mutation_fn,
            &composed,
            0,
            &mut rng,
        );

        // Logging handler should have recorded the mutation
        assert_eq!(logging.get_mutations().len(), 1);
    }

    #[test]
    fn test_operation_statistics() {
        let handler = LoggingHandler::new();

        // Manually add some records
        handler.mutations.lock().unwrap().push(MutationRecord {
            generation: 0,
            mutated_addresses: vec![addr!("test", 0), addr!("test", 1)],
            original_values: HashMap::new(),
            new_values: HashMap::new(),
        });

        let stats = OperationStatistics::from_handler(&handler);
        assert_eq!(stats.total_mutations, 1);
        assert_eq!(stats.avg_mutation_sites, 2.0);
    }
}
