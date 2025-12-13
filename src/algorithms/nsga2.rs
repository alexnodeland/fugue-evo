//! NSGA-II (Non-dominated Sorting Genetic Algorithm II)
//!
//! Implements the NSGA-II algorithm for multi-objective optimization.
//!
//! Reference: Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
//! A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
//! IEEE Transactions on Evolutionary Computation, 6(2).

use std::marker::PhantomData;

use rand::Rng;

use crate::error::EvoResult;
use crate::fitness::traits::ParetoFitness;
use crate::genome::bounds::MultiBounds;
use crate::genome::traits::EvolutionaryGenome;
use crate::operators::traits::{
    BoundedCrossoverOperator, BoundedMutationOperator, CrossoverOperator, MutationOperator,
};
use crate::population::individual::Individual;

/// Multi-objective fitness function trait
#[cfg(feature = "parallel")]
pub trait MultiObjectiveFitness<G>: Send + Sync {
    /// Number of objectives
    fn num_objectives(&self) -> usize;

    /// Evaluate all objectives (all to be minimized by convention)
    fn evaluate(&self, genome: &G) -> Vec<f64>;
}

/// Multi-objective fitness function trait
#[cfg(not(feature = "parallel"))]
pub trait MultiObjectiveFitness<G> {
    /// Number of objectives
    fn num_objectives(&self) -> usize;

    /// Evaluate all objectives (all to be minimized by convention)
    fn evaluate(&self, genome: &G) -> Vec<f64>;
}

/// Implement MultiObjectiveFitness for closures
#[cfg(feature = "parallel")]
impl<G, F> MultiObjectiveFitness<G> for F
where
    F: Fn(&G) -> Vec<f64> + Send + Sync,
{
    fn num_objectives(&self) -> usize {
        // This is a limitation - closure doesn't know number of objectives
        // Users should use the explicit trait implementation for better type safety
        2 // Default assumption
    }

    fn evaluate(&self, genome: &G) -> Vec<f64> {
        self(genome)
    }
}

/// Implement MultiObjectiveFitness for closures
#[cfg(not(feature = "parallel"))]
impl<G, F> MultiObjectiveFitness<G> for F
where
    F: Fn(&G) -> Vec<f64>,
{
    fn num_objectives(&self) -> usize {
        2 // Default assumption
    }

    fn evaluate(&self, genome: &G) -> Vec<f64> {
        self(genome)
    }
}

/// NSGA-II individual with objectives and crowding info
#[derive(Clone, Debug)]
pub struct Nsga2Individual<G: EvolutionaryGenome> {
    /// The genome
    pub genome: G,
    /// Objective values
    pub objectives: Vec<f64>,
    /// Pareto rank (0 = first front)
    pub rank: usize,
    /// Crowding distance
    pub crowding_distance: f64,
}

impl<G: EvolutionaryGenome> Nsga2Individual<G> {
    /// Create a new individual with evaluated objectives
    pub fn new(genome: G, objectives: Vec<f64>) -> Self {
        Self {
            genome,
            objectives,
            rank: usize::MAX,
            crowding_distance: 0.0,
        }
    }

    /// Check if this individual dominates another
    /// (all objectives <= and at least one <, since we minimize)
    pub fn dominates(&self, other: &Self) -> bool {
        let at_least_as_good = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .all(|(a, b)| a <= b);
        let strictly_better = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .any(|(a, b)| a < b);
        at_least_as_good && strictly_better
    }

    /// Convert to ParetoFitness
    pub fn to_pareto_fitness(&self) -> ParetoFitness {
        let mut pf = ParetoFitness::new(self.objectives.clone());
        pf.rank = self.rank;
        pf.crowding_distance = self.crowding_distance;
        pf
    }

    /// Convert to Individual with ParetoFitness
    pub fn to_individual(self) -> Individual<G, ParetoFitness> {
        let fitness = self.to_pareto_fitness();
        Individual::with_fitness(self.genome, fitness)
    }
}

/// Fast non-dominated sort
///
/// Returns fronts where `front[0]` is the Pareto-optimal front
pub fn fast_non_dominated_sort<G: EvolutionaryGenome>(
    population: &mut [Nsga2Individual<G>],
) -> Vec<Vec<usize>> {
    let n = population.len();
    if n == 0 {
        return vec![];
    }

    // domination_count[i] = number of individuals that dominate i
    let mut domination_count = vec![0usize; n];
    // dominated_set[i] = set of individuals that i dominates
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n];

    // Calculate domination relationships
    for i in 0..n {
        for j in (i + 1)..n {
            if population[i].dominates(&population[j]) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if population[j].dominates(&population[i]) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // Build fronts
    let mut fronts: Vec<Vec<usize>> = vec![];
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    let mut rank = 0;
    while !current_front.is_empty() {
        // Assign rank to current front
        for &i in &current_front {
            population[i].rank = rank;
        }

        // Build next front
        let mut next_front = vec![];
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }

        fronts.push(current_front);
        current_front = next_front;
        rank += 1;
    }

    fronts
}

/// Calculate crowding distance for a front
pub fn calculate_crowding_distance<G: EvolutionaryGenome>(
    population: &mut [Nsga2Individual<G>],
    front: &[usize],
) {
    let n = front.len();
    if n <= 2 {
        for &i in front {
            population[i].crowding_distance = f64::INFINITY;
        }
        return;
    }

    // Reset distances
    for &i in front {
        population[i].crowding_distance = 0.0;
    }

    let num_objectives = population[front[0]].objectives.len();

    for obj in 0..num_objectives {
        // Sort front by this objective
        let mut sorted_indices: Vec<usize> = front.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            population[a].objectives[obj]
                .partial_cmp(&population[b].objectives[obj])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary individuals get infinite distance
        population[sorted_indices[0]].crowding_distance = f64::INFINITY;
        population[sorted_indices[n - 1]].crowding_distance = f64::INFINITY;

        // Calculate range
        let obj_min = population[sorted_indices[0]].objectives[obj];
        let obj_max = population[sorted_indices[n - 1]].objectives[obj];
        let obj_range = obj_max - obj_min;

        if obj_range > 0.0 {
            for i in 1..(n - 1) {
                let idx = sorted_indices[i];
                let prev_val = population[sorted_indices[i - 1]].objectives[obj];
                let next_val = population[sorted_indices[i + 1]].objectives[obj];
                population[idx].crowding_distance += (next_val - prev_val) / obj_range;
            }
        }
    }
}

/// Crowded comparison operator
///
/// Returns true if a is better than b (lower rank, or same rank with higher crowding distance)
pub fn crowded_comparison<G: EvolutionaryGenome>(
    a: &Nsga2Individual<G>,
    b: &Nsga2Individual<G>,
) -> bool {
    a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance)
}

/// NSGA-II algorithm
pub struct Nsga2<G, F, C, M> {
    /// Population size
    pub population_size: usize,
    /// Crossover probability
    pub crossover_probability: f64,
    /// Mutation probability
    pub mutation_probability: f64,
    /// Problem bounds
    pub bounds: Option<MultiBounds>,
    /// Marker for types
    _phantom: PhantomData<(G, F, C, M)>,
}

impl<G, F, C, M> Nsga2<G, F, C, M>
where
    G: EvolutionaryGenome,
    F: MultiObjectiveFitness<G>,
    C: CrossoverOperator<G>,
    M: MutationOperator<G>,
{
    /// Create a new NSGA-II algorithm
    pub fn new(population_size: usize) -> Self {
        Self {
            population_size,
            crossover_probability: 0.9,
            mutation_probability: 1.0,
            bounds: None,
            _phantom: PhantomData,
        }
    }

    /// Set crossover probability
    pub fn with_crossover_probability(mut self, prob: f64) -> Self {
        self.crossover_probability = prob;
        self
    }

    /// Set mutation probability
    pub fn with_mutation_probability(mut self, prob: f64) -> Self {
        self.mutation_probability = prob;
        self
    }

    /// Set bounds
    pub fn with_bounds(mut self, bounds: MultiBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Initialize random population
    pub fn initialize_population<R: Rng>(
        &self,
        fitness: &F,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> Vec<Nsga2Individual<G>> {
        (0..self.population_size)
            .map(|_| {
                let genome = G::generate(rng, bounds);
                let objectives = fitness.evaluate(&genome);
                Nsga2Individual::new(genome, objectives)
            })
            .collect()
    }

    /// Binary tournament selection with crowded comparison
    pub fn tournament_select<'a, R: Rng>(
        &self,
        population: &'a [Nsga2Individual<G>],
        rng: &mut R,
    ) -> &'a Nsga2Individual<G> {
        let i = rng.gen_range(0..population.len());
        let j = rng.gen_range(0..population.len());

        if crowded_comparison(&population[i], &population[j]) {
            &population[i]
        } else {
            &population[j]
        }
    }

    /// Create offspring population
    pub fn create_offspring<R: Rng>(
        &self,
        population: &[Nsga2Individual<G>],
        fitness: &F,
        crossover: &C,
        mutation: &M,
        rng: &mut R,
    ) -> Vec<Nsga2Individual<G>> {
        let mut offspring = Vec::with_capacity(self.population_size);

        while offspring.len() < self.population_size {
            // Select parents
            let parent1 = self.tournament_select(population, rng);
            let parent2 = self.tournament_select(population, rng);

            // Crossover
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_probability {
                match crossover.crossover(&parent1.genome, &parent2.genome, rng) {
                    crate::error::OperatorResult::Success((c1, c2)) => (c1, c2),
                    _ => (parent1.genome.clone(), parent2.genome.clone()),
                }
            } else {
                (parent1.genome.clone(), parent2.genome.clone())
            };

            // Mutation
            if rng.gen::<f64>() < self.mutation_probability {
                mutation.mutate(&mut child1, rng);
            }
            if rng.gen::<f64>() < self.mutation_probability {
                mutation.mutate(&mut child2, rng);
            }

            // Evaluate
            let obj1 = fitness.evaluate(&child1);
            let obj2 = fitness.evaluate(&child2);

            offspring.push(Nsga2Individual::new(child1, obj1));
            if offspring.len() < self.population_size {
                offspring.push(Nsga2Individual::new(child2, obj2));
            }
        }

        offspring
    }

    /// Run one generation of NSGA-II
    pub fn step<R: Rng>(
        &self,
        population: &mut Vec<Nsga2Individual<G>>,
        fitness: &F,
        crossover: &C,
        mutation: &M,
        rng: &mut R,
    ) {
        // Create offspring
        let offspring = self.create_offspring(population, fitness, crossover, mutation, rng);

        // Combine parent and offspring populations
        let mut combined: Vec<Nsga2Individual<G>> =
            population.drain(..).chain(offspring.into_iter()).collect();

        // Non-dominated sort
        let fronts = fast_non_dominated_sort(&mut combined);

        // Fill new population from fronts
        let mut new_pop = Vec::with_capacity(self.population_size);

        for front in fronts {
            if new_pop.len() + front.len() <= self.population_size {
                // Add entire front
                for &i in &front {
                    new_pop.push(combined[i].clone());
                }
            } else {
                // Partial front - sort by crowding distance
                calculate_crowding_distance(&mut combined, &front);

                let mut sorted_front: Vec<usize> = front.to_vec();
                sorted_front.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_distance
                        .partial_cmp(&combined[a].crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let remaining = self.population_size - new_pop.len();
                for &i in sorted_front.iter().take(remaining) {
                    new_pop.push(combined[i].clone());
                }
                break;
            }
        }

        // Update crowding distance for the entire new population
        let all_indices: Vec<usize> = (0..new_pop.len()).collect();
        calculate_crowding_distance(&mut new_pop, &all_indices);

        *population = new_pop;
    }

    /// Run NSGA-II for a fixed number of generations
    pub fn run<R: Rng>(
        &self,
        fitness: &F,
        crossover: &C,
        mutation: &M,
        bounds: &MultiBounds,
        max_generations: usize,
        rng: &mut R,
    ) -> EvoResult<Vec<Nsga2Individual<G>>> {
        let mut population = self.initialize_population(fitness, bounds, rng);

        // Initial non-dominated sort
        fast_non_dominated_sort(&mut population);
        let all_indices: Vec<usize> = (0..population.len()).collect();
        calculate_crowding_distance(&mut population, &all_indices);

        for _ in 0..max_generations {
            self.step(&mut population, fitness, crossover, mutation, rng);
        }

        Ok(population)
    }

    /// Get the Pareto front (rank 0 individuals)
    pub fn get_pareto_front(population: &[Nsga2Individual<G>]) -> Vec<&Nsga2Individual<G>> {
        population.iter().filter(|ind| ind.rank == 0).collect()
    }
}

/// Version with bounded operators
impl<G, F, C, M> Nsga2<G, F, C, M>
where
    G: EvolutionaryGenome,
    F: MultiObjectiveFitness<G>,
    C: BoundedCrossoverOperator<G>,
    M: BoundedMutationOperator<G>,
{
    /// Create offspring population with bounded operators
    pub fn create_offspring_bounded<R: Rng>(
        &self,
        population: &[Nsga2Individual<G>],
        fitness: &F,
        crossover: &C,
        mutation: &M,
        bounds: &MultiBounds,
        rng: &mut R,
    ) -> Vec<Nsga2Individual<G>> {
        let mut offspring = Vec::with_capacity(self.population_size);

        while offspring.len() < self.population_size {
            let parent1 = self.tournament_select(population, rng);
            let parent2 = self.tournament_select(population, rng);

            let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_probability {
                match crossover.crossover_bounded(&parent1.genome, &parent2.genome, bounds, rng) {
                    crate::error::OperatorResult::Success((c1, c2)) => (c1, c2),
                    _ => (parent1.genome.clone(), parent2.genome.clone()),
                }
            } else {
                (parent1.genome.clone(), parent2.genome.clone())
            };

            if rng.gen::<f64>() < self.mutation_probability {
                mutation.mutate_bounded(&mut child1, bounds, rng);
            }
            if rng.gen::<f64>() < self.mutation_probability {
                mutation.mutate_bounded(&mut child2, bounds, rng);
            }

            let obj1 = fitness.evaluate(&child1);
            let obj2 = fitness.evaluate(&child2);

            offspring.push(Nsga2Individual::new(child1, obj1));
            if offspring.len() < self.population_size {
                offspring.push(Nsga2Individual::new(child2, obj2));
            }
        }

        offspring
    }

    /// Run one generation with bounded operators
    pub fn step_bounded<R: Rng>(
        &self,
        population: &mut Vec<Nsga2Individual<G>>,
        fitness: &F,
        crossover: &C,
        mutation: &M,
        bounds: &MultiBounds,
        rng: &mut R,
    ) {
        let offspring =
            self.create_offspring_bounded(population, fitness, crossover, mutation, bounds, rng);

        let mut combined: Vec<Nsga2Individual<G>> =
            population.drain(..).chain(offspring.into_iter()).collect();

        let fronts = fast_non_dominated_sort(&mut combined);

        let mut new_pop = Vec::with_capacity(self.population_size);

        for front in fronts {
            if new_pop.len() + front.len() <= self.population_size {
                for &i in &front {
                    new_pop.push(combined[i].clone());
                }
            } else {
                calculate_crowding_distance(&mut combined, &front);

                let mut sorted_front: Vec<usize> = front.to_vec();
                sorted_front.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_distance
                        .partial_cmp(&combined[a].crowding_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let remaining = self.population_size - new_pop.len();
                for &i in sorted_front.iter().take(remaining) {
                    new_pop.push(combined[i].clone());
                }
                break;
            }
        }

        let all_indices: Vec<usize> = (0..new_pop.len()).collect();
        calculate_crowding_distance(&mut new_pop, &all_indices);

        *population = new_pop;
    }

    /// Run NSGA-II with bounded operators
    pub fn run_bounded<R: Rng>(
        &self,
        fitness: &F,
        crossover: &C,
        mutation: &M,
        bounds: &MultiBounds,
        max_generations: usize,
        rng: &mut R,
    ) -> EvoResult<Vec<Nsga2Individual<G>>> {
        let mut population = self.initialize_population(fitness, bounds, rng);

        fast_non_dominated_sort(&mut population);
        let all_indices: Vec<usize> = (0..population.len()).collect();
        calculate_crowding_distance(&mut population, &all_indices);

        for _ in 0..max_generations {
            self.step_bounded(&mut population, fitness, crossover, mutation, bounds, rng);
        }

        Ok(population)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::real_vector::RealVector;
    use crate::genome::traits::RealValuedGenome;
    use crate::operators::crossover::SbxCrossover;
    use crate::operators::mutation::PolynomialMutation;

    // ZDT1 test problem
    struct Zdt1;

    impl MultiObjectiveFitness<RealVector> for Zdt1 {
        fn num_objectives(&self) -> usize {
            2
        }

        fn evaluate(&self, genome: &RealVector) -> Vec<f64> {
            let x = genome.genes();
            let n = x.len() as f64;

            let f1 = x[0];

            let g: f64 = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
            let f2 = g * (1.0 - (f1 / g).sqrt());

            vec![f1, f2]
        }
    }

    #[test]
    fn test_domination() {
        let a = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![1.0, 2.0]);
        let b = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![2.0, 3.0]);
        let c = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![1.5, 1.5]);

        assert!(a.dominates(&b)); // a is better in both objectives
        assert!(!b.dominates(&a));
        assert!(!a.dominates(&c)); // c is better in second objective
        assert!(!c.dominates(&a)); // a is better in first objective
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let mut population = vec![
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![1.0, 4.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![2.0, 3.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![3.0, 2.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![4.0, 1.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![3.0, 3.0]),
        ];

        let fronts = fast_non_dominated_sort(&mut population);

        // First 4 should be in front 0 (none dominates each other)
        assert_eq!(fronts[0].len(), 4);
        // Last one should be in front 1 (dominated by [2,3] and [3,2])
        assert_eq!(fronts[1].len(), 1);

        for &i in &fronts[0] {
            assert_eq!(population[i].rank, 0);
        }
        for &i in &fronts[1] {
            assert_eq!(population[i].rank, 1);
        }
    }

    #[test]
    fn test_crowding_distance() {
        let mut population = vec![
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![0.0, 10.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![5.0, 5.0]),
            Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![10.0, 0.0]),
        ];

        let front: Vec<usize> = (0..population.len()).collect();
        calculate_crowding_distance(&mut population, &front);

        // Boundary points should have infinite distance
        assert!(population[0].crowding_distance.is_infinite());
        assert!(population[2].crowding_distance.is_infinite());
        // Middle point should have finite distance
        assert!(population[1].crowding_distance.is_finite());
        assert!(population[1].crowding_distance > 0.0);
    }

    #[test]
    fn test_crowded_comparison() {
        let mut a = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![1.0, 1.0]);
        a.rank = 0;
        a.crowding_distance = 2.0;

        let mut b = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![2.0, 2.0]);
        b.rank = 1;
        b.crowding_distance = 3.0;

        let mut c = Nsga2Individual::<RealVector>::new(RealVector::new(vec![0.0]), vec![1.5, 1.5]);
        c.rank = 0;
        c.crowding_distance = 1.0;

        assert!(crowded_comparison(&a, &b)); // a has lower rank
        assert!(crowded_comparison(&a, &c)); // same rank, a has higher crowding distance
        assert!(!crowded_comparison(&c, &a)); // c has lower crowding distance
    }

    #[test]
    fn test_nsga2_initialization() {
        use crate::genome::bounds::Bounds;
        let mut rng = rand::thread_rng();
        let fitness = Zdt1;
        let bounds = MultiBounds::new(vec![Bounds::new(0.0, 1.0); 10]);

        let nsga2: Nsga2<RealVector, Zdt1, SbxCrossover, PolynomialMutation> = Nsga2::new(20);
        let population = nsga2.initialize_population(&fitness, &bounds, &mut rng);

        assert_eq!(population.len(), 20);
        for ind in &population {
            assert_eq!(ind.objectives.len(), 2);
        }
    }

    #[test]
    fn test_nsga2_run() {
        use crate::genome::bounds::Bounds;
        let mut rng = rand::thread_rng();
        let fitness = Zdt1;
        let bounds = MultiBounds::new(vec![Bounds::new(0.0, 1.0); 10]);
        let crossover = SbxCrossover::new(15.0);
        let mutation = PolynomialMutation::new(20.0);

        let nsga2: Nsga2<RealVector, Zdt1, SbxCrossover, PolynomialMutation> = Nsga2::new(50);
        let population = nsga2
            .run_bounded(&fitness, &crossover, &mutation, &bounds, 10, &mut rng)
            .unwrap();

        assert_eq!(population.len(), 50);

        // Check that we have a Pareto front
        let pareto_front =
            Nsga2::<RealVector, Zdt1, SbxCrossover, PolynomialMutation>::get_pareto_front(
                &population,
            );
        assert!(!pareto_front.is_empty());
    }

    #[test]
    fn test_nsga2_pareto_front_quality() {
        use crate::genome::bounds::Bounds;
        let mut rng = rand::thread_rng();
        let fitness = Zdt1;
        let bounds = MultiBounds::new(vec![Bounds::new(0.0, 1.0); 10]);
        let crossover = SbxCrossover::new(15.0);
        let mutation = PolynomialMutation::new(20.0);

        let nsga2: Nsga2<RealVector, Zdt1, SbxCrossover, PolynomialMutation> = Nsga2::new(100);
        let population = nsga2
            .run_bounded(&fitness, &crossover, &mutation, &bounds, 50, &mut rng)
            .unwrap();

        let pareto_front =
            Nsga2::<RealVector, Zdt1, SbxCrossover, PolynomialMutation>::get_pareto_front(
                &population,
            );

        // Verify Pareto front properties
        for ind in &pareto_front {
            assert_eq!(ind.rank, 0);
            // ZDT1 Pareto front has f2 = 1 - sqrt(f1)
            // All objectives should be positive
            assert!(ind.objectives[0] >= 0.0);
            assert!(ind.objectives[1] >= 0.0);
        }
    }
}
