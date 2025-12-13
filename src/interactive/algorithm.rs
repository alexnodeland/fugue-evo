//! Interactive Genetic Algorithm implementation
//!
//! This module provides the `InteractiveGA` algorithm which uses a step-based
//! iterator pattern to allow human-in-the-loop fitness evaluation.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::aggregation::{AggregationModel, FitnessAggregator};
use super::evaluator::{Candidate, CandidateId, EvaluationRequest, EvaluationResponse};
use super::selection_strategy::SelectionStrategy;
use super::session::{CoverageStats, InteractiveSession};
use super::traits::EvaluationMode;
use crate::error::EvolutionError;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::{CrossoverOperator, MutationOperator, SelectionOperator};

/// Configuration for Interactive GA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractiveGAConfig {
    /// Population size (smaller than standard GA for human evaluation)
    pub population_size: usize,
    /// Number of elite individuals to preserve
    pub elitism_count: usize,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Mutation probability
    pub mutation_probability: f64,
    /// Evaluation mode
    pub evaluation_mode: EvaluationMode,
    /// Number of candidates per evaluation batch
    pub batch_size: usize,
    /// Number to select in batch selection mode
    pub select_count: usize,
    /// Minimum coverage fraction before proceeding to next generation
    pub min_coverage: f64,
    /// Number of comparisons per candidate per generation (for pairwise mode)
    pub comparisons_per_candidate: usize,
    /// Maximum generations (0 = unlimited)
    pub max_generations: usize,
    /// Aggregation model for fitness computation
    pub aggregation_model: AggregationModel,
    /// Active learning strategy for candidate selection
    #[serde(default)]
    pub selection_strategy: SelectionStrategy,
}

impl Default for InteractiveGAConfig {
    fn default() -> Self {
        Self {
            population_size: 20, // Smaller for human evaluation
            elitism_count: 2,
            crossover_probability: 0.8,
            mutation_probability: 0.2,
            evaluation_mode: EvaluationMode::Rating,
            batch_size: 6,
            select_count: 2,
            min_coverage: 0.8, // 80% must be evaluated
            comparisons_per_candidate: 3,
            max_generations: 0, // Unlimited
            aggregation_model: AggregationModel::DirectRating {
                default_rating: 5.0,
            },
            selection_strategy: SelectionStrategy::Sequential,
        }
    }
}

/// Internal state machine for the algorithm
#[derive(Clone, Debug)]
enum AlgorithmState {
    /// Need to initialize population
    Initializing,
    /// Waiting for evaluation responses
    AwaitingEvaluation {
        /// Current request being processed
        pending_request_ids: Vec<CandidateId>,
    },
    /// Ready to perform selection and create next generation
    ReadyForEvolution,
    /// Evolution complete
    Terminated { reason: String },
}

/// Result of calling `step()` on the algorithm
#[derive(Clone, Debug)]
pub enum StepResult<G>
where
    G: EvolutionaryGenome,
{
    /// Algorithm needs user input
    NeedsEvaluation(EvaluationRequest<G>),

    /// Generation complete, ready to continue
    GenerationComplete {
        /// Generation number that completed
        generation: usize,
        /// Best fitness in the generation
        best_fitness: Option<f64>,
        /// Evaluation coverage achieved
        coverage: f64,
    },

    /// Evolution terminated
    Complete(Box<InteractiveResult<G>>),
}

/// Final result of interactive evolution
#[derive(Clone, Debug)]
pub struct InteractiveResult<G>
where
    G: EvolutionaryGenome,
{
    /// Best candidates found
    pub best_candidates: Vec<Candidate<G>>,
    /// Number of generations completed
    pub generations: usize,
    /// Total evaluation requests made
    pub total_evaluations: usize,
    /// Final session state
    pub session: InteractiveSession<G>,
    /// Termination reason
    pub termination_reason: String,
}

/// Step-based Interactive Genetic Algorithm
///
/// Unlike standard GA algorithms that run to completion, InteractiveGA yields
/// control between evaluations, allowing the caller to interact with users.
///
/// # Example
///
/// ```rust,ignore
/// use fugue_evo::interactive::prelude::*;
///
/// let mut iga = InteractiveGABuilder::<MyGenome>::new()
///     .population_size(12)
///     .evaluation_mode(EvaluationMode::BatchSelection)
///     .build()?;
///
/// let mut rng = rand::thread_rng();
///
/// loop {
///     match iga.step(&mut rng) {
///         StepResult::NeedsEvaluation(request) => {
///             let response = get_user_feedback(&request);
///             iga.provide_response(response);
///         }
///         StepResult::GenerationComplete { generation, .. } => {
///             println!("Generation {} complete", generation);
///         }
///         StepResult::Complete(result) => {
///             println!("Evolution complete: {}", result.termination_reason);
///             break;
///         }
///     }
/// }
/// ```
pub struct InteractiveGA<G, S, C, M>
where
    G: EvolutionaryGenome,
{
    config: InteractiveGAConfig,
    bounds: Option<MultiBounds>,
    selection: S,
    crossover: C,
    mutation: M,
    session: InteractiveSession<G>,
    state: AlgorithmState,
    /// Indices of candidates still needing evaluation this generation
    unevaluated_indices: Vec<usize>,
    /// Index for pairwise comparison scheduling
    comparison_index: usize,
}

impl<G, S, C, M> InteractiveGA<G, S, C, M>
where
    G: EvolutionaryGenome + Clone + Send + Sync,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
{
    /// Create a new InteractiveGA
    pub fn new(
        config: InteractiveGAConfig,
        bounds: Option<MultiBounds>,
        selection: S,
        crossover: C,
        mutation: M,
    ) -> Self {
        let aggregator = FitnessAggregator::new(config.aggregation_model.clone());
        Self {
            config,
            bounds,
            selection,
            crossover,
            mutation,
            session: InteractiveSession::new(aggregator),
            state: AlgorithmState::Initializing,
            unevaluated_indices: Vec::new(),
            comparison_index: 0,
        }
    }

    /// Resume from a saved session
    pub fn from_session(
        session: InteractiveSession<G>,
        config: InteractiveGAConfig,
        bounds: Option<MultiBounds>,
        selection: S,
        crossover: C,
        mutation: M,
    ) -> Self {
        let unevaluated: Vec<usize> = session
            .population
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_evaluated())
            .map(|(i, _)| i)
            .collect();

        let state = if session.population.is_empty() {
            AlgorithmState::Initializing
        } else if unevaluated.is_empty() {
            AlgorithmState::ReadyForEvolution
        } else {
            AlgorithmState::AwaitingEvaluation {
                pending_request_ids: Vec::new(),
            }
        };

        Self {
            config,
            bounds,
            selection,
            crossover,
            mutation,
            session,
            state,
            unevaluated_indices: unevaluated,
            comparison_index: 0,
        }
    }

    /// Get the current session
    pub fn session(&self) -> &InteractiveSession<G> {
        &self.session
    }

    /// Get mutable reference to session (for custom modifications)
    pub fn session_mut(&mut self) -> &mut InteractiveSession<G> {
        &mut self.session
    }

    /// Get the configuration
    pub fn config(&self) -> &InteractiveGAConfig {
        &self.config
    }

    /// Get coverage statistics
    pub fn coverage_stats(&self) -> CoverageStats {
        self.session.coverage_stats()
    }

    /// Check if algorithm should terminate
    fn should_terminate(&self) -> Option<String> {
        if self.config.max_generations > 0 && self.session.generation >= self.config.max_generations
        {
            return Some(format!(
                "Reached maximum generations ({})",
                self.config.max_generations
            ));
        }
        None
    }

    /// Initialize the population
    fn initialize_population<R: Rng>(&mut self, rng: &mut R) {
        // Need bounds for genome generation
        let bounds = self
            .bounds
            .clone()
            .unwrap_or_else(|| MultiBounds::symmetric(1.0, 1));

        for _ in 0..self.config.population_size {
            let genome = G::generate(rng, &bounds);
            self.session.add_candidate(genome);
        }

        self.unevaluated_indices = (0..self.config.population_size).collect();
        self.comparison_index = 0;
    }

    /// Create an evaluation request based on the current mode
    fn create_evaluation_request<R: Rng>(&mut self, rng: &mut R) -> Option<EvaluationRequest<G>> {
        match self.config.evaluation_mode {
            EvaluationMode::Rating => self.create_rating_request(rng),
            EvaluationMode::Pairwise => self.create_pairwise_request(rng),
            EvaluationMode::BatchSelection => self.create_batch_request(rng),
            EvaluationMode::Adaptive => self.create_adaptive_request(rng),
        }
    }

    fn create_rating_request<R: Rng>(&mut self, rng: &mut R) -> Option<EvaluationRequest<G>> {
        let batch_size = self.config.batch_size.min(self.session.population.len());
        if batch_size == 0 {
            return None;
        }

        // Use selection strategy to pick candidates
        let selected_indices = self.config.selection_strategy.select_batch(
            &self.session.population,
            &self.session.aggregator,
            batch_size,
            rng,
        );

        if selected_indices.is_empty() {
            return None;
        }

        let candidates: Vec<Candidate<G>> = selected_indices
            .iter()
            .filter_map(|&i| self.session.population.get(i).cloned())
            .collect();

        let ids: Vec<CandidateId> = candidates.iter().map(|c| c.id).collect();
        self.state = AlgorithmState::AwaitingEvaluation {
            pending_request_ids: ids,
        };

        Some(EvaluationRequest::rate(candidates))
    }

    fn create_pairwise_request<R: Rng>(&mut self, rng: &mut R) -> Option<EvaluationRequest<G>> {
        let pop_size = self.session.population.len();
        if pop_size < 2 {
            return None;
        }

        // Use selection strategy for intelligent pair selection
        let pair = self.config.selection_strategy.select_pair(
            &self.session.population,
            &self.session.aggregator,
            rng,
        );

        let (idx_a, idx_b) = match pair {
            Some(p) => p,
            None => {
                // Fallback to round-robin if strategy returns None
                let idx_a = self.comparison_index % pop_size;
                let idx_b = (self.comparison_index + 1) % pop_size;
                (idx_a, idx_b)
            }
        };

        self.comparison_index += 1;

        let candidate_a = self.session.population.get(idx_a)?.clone();
        let candidate_b = self.session.population.get(idx_b)?.clone();

        let ids = vec![candidate_a.id, candidate_b.id];
        self.state = AlgorithmState::AwaitingEvaluation {
            pending_request_ids: ids,
        };

        Some(EvaluationRequest::compare(candidate_a, candidate_b))
    }

    fn create_batch_request<R: Rng>(&mut self, rng: &mut R) -> Option<EvaluationRequest<G>> {
        let batch_size = self.config.batch_size.min(self.session.population.len());
        if batch_size < 2 {
            // Need at least 2 for selection
            return self.create_rating_request(rng); // Fall back
        }

        // Use selection strategy to pick candidates
        let selected_indices = self.config.selection_strategy.select_batch(
            &self.session.population,
            &self.session.aggregator,
            batch_size,
            rng,
        );

        if selected_indices.len() < 2 {
            return self.create_rating_request(rng);
        }

        let candidates: Vec<Candidate<G>> = selected_indices
            .iter()
            .filter_map(|&i| self.session.population.get(i).cloned())
            .collect();

        let ids: Vec<CandidateId> = candidates.iter().map(|c| c.id).collect();
        self.state = AlgorithmState::AwaitingEvaluation {
            pending_request_ids: ids,
        };

        let select_count = self.config.select_count.min(candidates.len() - 1);
        Some(EvaluationRequest::select_from_batch(
            candidates,
            select_count,
        ))
    }

    fn create_adaptive_request<R: Rng>(&mut self, rng: &mut R) -> Option<EvaluationRequest<G>> {
        // Simple adaptive strategy: use rating for initial coverage,
        // then switch to pairwise for refinement
        let coverage = self.session.coverage_stats().coverage;
        if coverage < 0.5 {
            self.create_rating_request(rng)
        } else {
            self.create_pairwise_request(rng)
        }
    }

    /// Provide user response to an evaluation request
    pub fn provide_response(&mut self, response: EvaluationResponse) {
        let was_skipped = response.is_skip();
        self.session.record_response(was_skipped);

        if was_skipped {
            // Put unevaluated candidates back if skipped
            if let AlgorithmState::AwaitingEvaluation {
                pending_request_ids,
            } = &self.state
            {
                for id in pending_request_ids {
                    if let Some(pos) = self.session.population.iter().position(|c| c.id == *id) {
                        if !self.unevaluated_indices.contains(&pos) {
                            self.unevaluated_indices.push(pos);
                        }
                    }
                }
            }
            self.state = AlgorithmState::AwaitingEvaluation {
                pending_request_ids: Vec::new(),
            };
            return;
        }

        // Process the response
        let updated = match &response {
            EvaluationResponse::Ratings(ratings) => {
                self.session.aggregator.process_response(&response);
                ratings.iter().map(|(id, _)| *id).collect::<Vec<_>>()
            }
            EvaluationResponse::PairwiseWinner(winner) => {
                if let AlgorithmState::AwaitingEvaluation {
                    pending_request_ids,
                } = &self.state
                {
                    if pending_request_ids.len() == 2 {
                        let id_a = pending_request_ids[0];
                        let id_b = pending_request_ids[1];
                        self.session
                            .aggregator
                            .process_pairwise(id_a, id_b, *winner);
                    }
                }
                winner.map(|w| vec![w]).unwrap_or_default()
            }
            EvaluationResponse::BatchSelected(selected) => {
                if let AlgorithmState::AwaitingEvaluation {
                    pending_request_ids,
                } = &self.state
                {
                    self.session
                        .aggregator
                        .process_batch_selection(pending_request_ids, selected);
                }
                selected.clone()
            }
            EvaluationResponse::Skip => Vec::new(),
        };

        // Update candidate fitness estimates with uncertainty
        for id in updated {
            if let Some(estimate) = self.session.aggregator.get_fitness_estimate(&id) {
                self.session.update_fitness_with_uncertainty(id, estimate);
            } else if let Some(fitness) = self.session.aggregator.get_fitness(&id) {
                // Fallback to point estimate only
                self.session.update_fitness(id, fitness);
            }
        }

        // Mark candidates as evaluated based on pending request
        if let AlgorithmState::AwaitingEvaluation {
            pending_request_ids,
        } = &self.state
        {
            for id in pending_request_ids {
                if let Some(candidate) = self.session.get_candidate_mut(*id) {
                    candidate.record_evaluation();
                }
            }
        }

        // Transition state
        self.state = AlgorithmState::AwaitingEvaluation {
            pending_request_ids: Vec::new(),
        };
    }

    /// Advance the algorithm one step
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> StepResult<G>
    where
        G: Serialize + for<'de> Deserialize<'de>,
    {
        loop {
            match &self.state {
                AlgorithmState::Initializing => {
                    self.initialize_population(rng);
                    self.state = AlgorithmState::AwaitingEvaluation {
                        pending_request_ids: Vec::new(),
                    };
                }

                AlgorithmState::AwaitingEvaluation {
                    pending_request_ids,
                } => {
                    // If we have a pending request, wait for response
                    if !pending_request_ids.is_empty() {
                        // This shouldn't happen in normal flow, but handle it
                        continue;
                    }

                    // Check if we have enough coverage
                    let coverage = self.session.coverage_stats();

                    // For pairwise mode, check comparison count instead
                    let enough_coverage = match self.config.evaluation_mode {
                        EvaluationMode::Pairwise => {
                            let target =
                                self.config.population_size * self.config.comparisons_per_candidate;
                            self.comparison_index >= target
                        }
                        _ => coverage.coverage >= self.config.min_coverage,
                    };

                    if enough_coverage {
                        self.state = AlgorithmState::ReadyForEvolution;
                        continue;
                    }

                    // Create next evaluation request
                    if let Some(request) = self.create_evaluation_request(rng) {
                        self.session.record_request(&request);
                        return StepResult::NeedsEvaluation(request);
                    } else {
                        // No more candidates to evaluate
                        self.state = AlgorithmState::ReadyForEvolution;
                    }
                }

                AlgorithmState::ReadyForEvolution => {
                    // Check termination
                    if let Some(reason) = self.should_terminate() {
                        self.state = AlgorithmState::Terminated {
                            reason: reason.clone(),
                        };
                        continue;
                    }

                    let generation = self.session.generation;
                    let best_fitness = self.session.best_candidate().and_then(|c| c.fitness());
                    let coverage = self.session.coverage_stats().coverage;

                    // Perform evolution
                    self.evolve_generation(rng);

                    return StepResult::GenerationComplete {
                        generation,
                        best_fitness,
                        coverage,
                    };
                }

                AlgorithmState::Terminated { reason } => {
                    let best_candidates = self
                        .session
                        .ranked_candidates()
                        .into_iter()
                        .take(self.config.elitism_count.max(3))
                        .cloned()
                        .collect();

                    return StepResult::Complete(Box::new(InteractiveResult {
                        best_candidates,
                        generations: self.session.generation,
                        total_evaluations: self.session.evaluations_requested,
                        session: self.session.clone(),
                        termination_reason: reason.clone(),
                    }));
                }
            }
        }
    }

    /// Perform selection and create next generation
    fn evolve_generation<R: Rng>(&mut self, rng: &mut R)
    where
        G: Serialize + for<'de> Deserialize<'de>,
    {
        let pop_size = self.config.population_size;

        // Get current population with fitness (genome, fitness) pairs
        let evaluated: Vec<(G, f64)> = self
            .session
            .population
            .iter()
            .filter_map(|c| c.fitness_estimate.map(|f| (c.genome.clone(), f)))
            .collect();

        if evaluated.is_empty() {
            // No evaluated individuals, can't evolve
            self.session.advance_generation();
            return;
        }

        // Preserve elites - collect first to avoid borrow issues
        let mut new_population: Vec<Candidate<G>> = Vec::with_capacity(pop_size);
        let elites: Vec<_> = self
            .session
            .ranked_candidates()
            .into_iter()
            .take(self.config.elitism_count)
            .map(|c| (c.genome.clone(), c.fitness_estimate))
            .collect();

        let next_gen = self.session.generation + 1;
        for (genome, fitness) in elites {
            let id = self.session.next_id();
            let mut candidate = Candidate::with_generation(id, genome, next_gen);
            // Preserve elite fitness
            candidate.fitness_estimate = fitness;
            new_population.push(candidate);
        }

        // Fill rest with offspring
        while new_population.len() < pop_size {
            // Selection - returns index into evaluated pool
            let parent1_idx = self.selection.select(&evaluated, rng);
            let parent2_idx = self.selection.select(&evaluated, rng);

            let parent1 = &evaluated[parent1_idx].0;
            let parent2 = &evaluated[parent2_idx].0;

            // Crossover
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_probability {
                match self.crossover.crossover(parent1, parent2, rng).genome() {
                    Some((c1, c2)) => (c1, c2),
                    None => (parent1.clone(), parent2.clone()),
                }
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Mutation (in-place)
            if rng.gen::<f64>() < self.config.mutation_probability {
                self.mutation.mutate(&mut child1, rng);
            }

            let id = self.session.next_id();
            new_population.push(Candidate::with_generation(
                id,
                child1,
                self.session.generation + 1,
            ));

            if new_population.len() < pop_size {
                if rng.gen::<f64>() < self.config.mutation_probability {
                    self.mutation.mutate(&mut child2, rng);
                }

                let id = self.session.next_id();
                new_population.push(Candidate::with_generation(
                    id,
                    child2,
                    self.session.generation + 1,
                ));
            }
        }

        // Update session
        self.session.replace_population(new_population);
        self.session.advance_generation();

        // Reset evaluation tracking for new generation
        self.unevaluated_indices =
            (self.config.elitism_count..self.config.population_size).collect();
        self.comparison_index = 0;
        self.state = AlgorithmState::AwaitingEvaluation {
            pending_request_ids: Vec::new(),
        };
    }

    /// Manually terminate the algorithm
    pub fn terminate(&mut self, reason: &str) {
        self.state = AlgorithmState::Terminated {
            reason: reason.to_string(),
        };
    }
}

/// Builder for InteractiveGA
pub struct InteractiveGABuilder<G, S, C, M>
where
    G: EvolutionaryGenome,
{
    config: InteractiveGAConfig,
    bounds: Option<MultiBounds>,
    selection: Option<S>,
    crossover: Option<C>,
    mutation: Option<M>,
    _phantom: PhantomData<G>,
}

impl<G> InteractiveGABuilder<G, (), (), ()>
where
    G: EvolutionaryGenome,
{
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: InteractiveGAConfig::default(),
            bounds: None,
            selection: None,
            crossover: None,
            mutation: None,
            _phantom: PhantomData,
        }
    }
}

impl<G> Default for InteractiveGABuilder<G, (), (), ()>
where
    G: EvolutionaryGenome,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<G, S, C, M> InteractiveGABuilder<G, S, C, M>
where
    G: EvolutionaryGenome,
{
    /// Set the population size
    pub fn population_size(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }

    /// Set the elitism count
    pub fn elitism_count(mut self, count: usize) -> Self {
        self.config.elitism_count = count;
        self
    }

    /// Set the crossover probability
    pub fn crossover_probability(mut self, prob: f64) -> Self {
        self.config.crossover_probability = prob;
        self
    }

    /// Set the mutation probability
    pub fn mutation_probability(mut self, prob: f64) -> Self {
        self.config.mutation_probability = prob;
        self
    }

    /// Set the evaluation mode
    pub fn evaluation_mode(mut self, mode: EvaluationMode) -> Self {
        self.config.evaluation_mode = mode;
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set the select count for batch selection mode
    pub fn select_count(mut self, count: usize) -> Self {
        self.config.select_count = count;
        self
    }

    /// Set the minimum coverage threshold
    pub fn min_coverage(mut self, coverage: f64) -> Self {
        self.config.min_coverage = coverage.clamp(0.0, 1.0);
        self
    }

    /// Set comparisons per candidate for pairwise mode
    pub fn comparisons_per_candidate(mut self, count: usize) -> Self {
        self.config.comparisons_per_candidate = count;
        self
    }

    /// Set maximum generations (0 = unlimited)
    pub fn max_generations(mut self, max: usize) -> Self {
        self.config.max_generations = max;
        self
    }

    /// Set the aggregation model
    pub fn aggregation_model(mut self, model: AggregationModel) -> Self {
        self.config.aggregation_model = model;
        self
    }

    /// Set the active learning selection strategy
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// .selection_strategy(SelectionStrategy::UncertaintySampling {
    ///     uncertainty_weight: 1.0,
    /// })
    /// ```
    pub fn selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.config.selection_strategy = strategy;
        self
    }

    /// Set the search space bounds
    pub fn bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Set the selection operator
    pub fn selection<NewS>(self, selection: NewS) -> InteractiveGABuilder<G, NewS, C, M>
    where
        NewS: SelectionOperator<G>,
    {
        InteractiveGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: Some(selection),
            crossover: self.crossover,
            mutation: self.mutation,
            _phantom: PhantomData,
        }
    }

    /// Set the crossover operator
    pub fn crossover<NewC>(self, crossover: NewC) -> InteractiveGABuilder<G, S, NewC, M>
    where
        NewC: CrossoverOperator<G>,
    {
        InteractiveGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: Some(crossover),
            mutation: self.mutation,
            _phantom: PhantomData,
        }
    }

    /// Set the mutation operator
    pub fn mutation<NewM>(self, mutation: NewM) -> InteractiveGABuilder<G, S, C, NewM>
    where
        NewM: MutationOperator<G>,
    {
        InteractiveGABuilder {
            config: self.config,
            bounds: self.bounds,
            selection: self.selection,
            crossover: self.crossover,
            mutation: Some(mutation),
            _phantom: PhantomData,
        }
    }
}

impl<G, S, C, M> InteractiveGABuilder<G, S, C, M>
where
    G: EvolutionaryGenome + Clone + Send + Sync,
    S: SelectionOperator<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
{
    /// Build the InteractiveGA
    pub fn build(self) -> Result<InteractiveGA<G, S, C, M>, EvolutionError> {
        let selection = self
            .selection
            .ok_or_else(|| EvolutionError::Configuration("Selection operator required".into()))?;
        let crossover = self
            .crossover
            .ok_or_else(|| EvolutionError::Configuration("Crossover operator required".into()))?;
        let mutation = self
            .mutation
            .ok_or_else(|| EvolutionError::Configuration("Mutation operator required".into()))?;

        Ok(InteractiveGA::new(
            self.config,
            self.bounds,
            selection,
            crossover,
            mutation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::operators::crossover::SbxCrossover;
    use crate::operators::mutation::PolynomialMutation;
    use crate::operators::selection::TournamentSelection;
    use rand::SeedableRng;

    #[test]
    fn test_interactive_ga_builder() {
        let result = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(10)
            .evaluation_mode(EvaluationMode::Rating)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build();

        assert!(result.is_ok());
        let iga = result.unwrap();
        assert_eq!(iga.config().population_size, 10);
    }

    #[test]
    fn test_interactive_ga_initialization() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(5)
            .evaluation_mode(EvaluationMode::Rating)
            .batch_size(2)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build()
            .unwrap();

        let result = iga.step(&mut rng);

        match result {
            StepResult::NeedsEvaluation(request) => {
                assert!(request.candidate_count() <= 2);
            }
            _ => panic!("Expected NeedsEvaluation"),
        }

        assert_eq!(iga.session().population.len(), 5);
    }

    #[test]
    fn test_provide_response() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(4)
            .evaluation_mode(EvaluationMode::Rating)
            .batch_size(4)
            .min_coverage(1.0)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build()
            .unwrap();

        // Get first request
        let result = iga.step(&mut rng);
        let request = match result {
            StepResult::NeedsEvaluation(r) => r,
            _ => panic!("Expected NeedsEvaluation"),
        };

        // Provide ratings
        let ids = request.candidate_ids();
        let ratings: Vec<_> = ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| (id, (i + 1) as f64 * 2.0))
            .collect();
        iga.provide_response(EvaluationResponse::ratings(ratings));

        // Should be ready for evolution
        let result = iga.step(&mut rng);
        match result {
            StepResult::GenerationComplete { generation, .. } => {
                assert_eq!(generation, 0);
            }
            _ => panic!("Expected GenerationComplete"),
        }
    }

    #[test]
    fn test_pairwise_mode() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(4)
            .evaluation_mode(EvaluationMode::Pairwise)
            .comparisons_per_candidate(2)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build()
            .unwrap();

        let result = iga.step(&mut rng);

        match result {
            StepResult::NeedsEvaluation(EvaluationRequest::PairwiseComparison { .. }) => {}
            _ => panic!("Expected PairwiseComparison request"),
        }
    }

    #[test]
    fn test_batch_selection_mode() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(6)
            .evaluation_mode(EvaluationMode::BatchSelection)
            .batch_size(4)
            .select_count(2)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build()
            .unwrap();

        let result = iga.step(&mut rng);

        match result {
            StepResult::NeedsEvaluation(EvaluationRequest::BatchSelection {
                candidates,
                select_count,
                ..
            }) => {
                assert_eq!(candidates.len(), 4);
                assert_eq!(select_count, 2);
            }
            _ => panic!("Expected BatchSelection request"),
        }
    }

    #[test]
    fn test_skip_response() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut iga = InteractiveGABuilder::<RealVector, (), (), ()>::new()
            .population_size(4)
            .evaluation_mode(EvaluationMode::Rating)
            .batch_size(2)
            .selection(TournamentSelection::new(2))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build()
            .unwrap();

        // Get request
        let _ = iga.step(&mut rng);

        // Skip it
        iga.provide_response(EvaluationResponse::skip());

        assert_eq!(iga.session().skipped, 1);
        assert_eq!(iga.session().responses_received, 0);
    }
}
