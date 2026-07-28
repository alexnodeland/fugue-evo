//! Multi-objective fitness abstraction
//!
//! The [`MultiObjectiveFitness`] trait is shared by the classic NSGA-II
//! algorithm (`algorithms::nsga2`, `classic` feature) and the Pareto-posterior
//! inference layer (`inference::pareto`, `ppl` feature), so it lives here in
//! the always-available core rather than in either layer.

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

/// Adapts a closure into a [`MultiObjectiveFitness`] with an explicit objective
/// count.
///
/// A bare `Fn(&G) -> Vec<f64>` cannot report how many objectives it produces,
/// so the previous blanket impl hardcoded `num_objectives() == 2`, silently
/// mis-reporting the count for any 3+ objective problem (EV-85). This wrapper
/// requires the caller to state the true objective count at construction.
///
/// ```
/// use fugue_evo::fitness::multi_objective::{ClosureMultiObjective, MultiObjectiveFitness};
/// use fugue_evo::genome::real_vector::RealVector;
/// use fugue_evo::genome::traits::RealValuedGenome;
///
/// let fitness = ClosureMultiObjective::new(3, |g: &RealVector| {
///     let x = g.genes()[0];
///     vec![x, x * x, x + 1.0]
/// });
/// assert_eq!(fitness.num_objectives(), 3);
/// ```
pub struct ClosureMultiObjective<G, F> {
    num_objectives: usize,
    f: F,
    _phantom: std::marker::PhantomData<fn() -> G>,
}

impl<G, F> ClosureMultiObjective<G, F>
where
    F: Fn(&G) -> Vec<f64>,
{
    /// Wrap `f`, declaring that it returns `num_objectives` objective values.
    pub fn new(num_objectives: usize, f: F) -> Self {
        Self {
            num_objectives,
            f,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "parallel")]
impl<G, F> MultiObjectiveFitness<G> for ClosureMultiObjective<G, F>
where
    F: Fn(&G) -> Vec<f64> + Send + Sync,
{
    fn num_objectives(&self) -> usize {
        self.num_objectives
    }

    fn evaluate(&self, genome: &G) -> Vec<f64> {
        (self.f)(genome)
    }
}

#[cfg(not(feature = "parallel"))]
impl<G, F> MultiObjectiveFitness<G> for ClosureMultiObjective<G, F>
where
    F: Fn(&G) -> Vec<f64>,
{
    fn num_objectives(&self) -> usize {
        self.num_objectives
    }

    fn evaluate(&self, genome: &G) -> Vec<f64> {
        (self.f)(genome)
    }
}
