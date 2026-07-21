//! Incremental, seeded engines for the explorable docs widgets and playground.
//!
//! Unlike the run-to-completion optimizers in `optimizers.rs`, every engine
//! here exposes a `step()` that advances the *real* fugue-evo algorithm by one
//! generation and streams the state a canvas needs back as JSON — population
//! positions, fitness, the live CMA-ES covariance, Pareto ranks, migration
//! events. WASM computes, JavaScript only draws.
//!
//! Conventions shared by every engine:
//! - fitness streamed to JS is the *raw* benchmark value (lower is better,
//!   optimum at `optimal_fitness()`), regardless of the sign convention the
//!   algorithm uses internally;
//! - every engine is explicitly seeded (`u64`, a `BigInt` at the JS boundary)
//!   and never touches OS entropy, so a seed is a replayable recording;
//! - genomes are 2-D (`RealVector`) so positions map straight onto a canvas,
//!   except NSGA-II which lives in objective space.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::json;
use wasm_bindgen::prelude::*;

use fugue_evo::algorithms::cmaes::{CmaEs, CmaEsFitness};
use fugue_evo::algorithms::eda::umda::{ContinuousUnivariateModel, UMDAConfig};
use fugue_evo::algorithms::island::{
    IslandModel, IslandModelBuilder, MigrationPolicy, MigrationTopology,
};
use fugue_evo::algorithms::nsga2::{Nsga2, Nsga2Individual};
use fugue_evo::fitness::benchmarks::{
    Ackley, BenchmarkFunction, Rastrigin, Rosenbrock, SchafferN1, Sphere, StyblinskiTang, Zdt1,
    Zdt2, Zdt3,
};
use fugue_evo::fitness::traits::Fitness;
use fugue_evo::genome::bounds::{Bounds, MultiBounds};
use fugue_evo::genome::real_vector::RealVector;
use fugue_evo::genome::traits::{EvolutionaryGenome, RealValuedGenome};
use fugue_evo::operators::crossover::SbxCrossover;
use fugue_evo::operators::mutation::PolynomialMutation;
use fugue_evo::operators::selection::TournamentSelection;
use fugue_evo::operators::traits::{
    BoundedCrossoverOperator, BoundedMutationOperator, SelectionOperator,
};

// ============================================================================
// Landscapes
// ============================================================================

fn make_landscape(name: &str) -> Result<Box<dyn BenchmarkFunction>, JsValue> {
    let f: Box<dyn BenchmarkFunction> = match name {
        "sphere" => Box::new(Sphere::new(2)),
        "rastrigin" => Box::new(Rastrigin::new(2)),
        "rosenbrock" => Box::new(Rosenbrock::new(2)),
        "ackley" => Box::new(Ackley::new(2)),
        "styblinski" => Box::new(StyblinskiTang::new(2)),
        _ => {
            return Err(JsValue::from_str(&format!(
            "unknown landscape '{name}' (expected sphere|rastrigin|rosenbrock|ackley|styblinski)"
        )))
        }
    };
    Ok(f)
}

fn landscape_multibounds(f: &dyn BenchmarkFunction) -> MultiBounds {
    let (lo, hi) = f.bounds();
    MultiBounds::uniform(Bounds::new(lo, hi), 2)
}

/// Raw landscape values on a grid, row-major (`out[j*nx + i]`, `j` indexing y).
/// Lower is better; JS normalizes for the heatmap.
#[wasm_bindgen]
pub fn explore_landscape_grid(
    name: &str,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
) -> Result<Vec<f64>, JsValue> {
    let f = make_landscape(name)?;
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        let y = ymin + (ymax - ymin) * (j as f64 + 0.5) / ny as f64;
        for i in 0..nx {
            let x = xmin + (xmax - xmin) * (i as f64 + 0.5) / nx as f64;
            out.push(f.evaluate_raw(&[x, y]));
        }
    }
    Ok(out)
}

/// Landscape metadata: `{name, lo, hi, optimum_fitness, optimum: [x,y]|null}`.
#[wasm_bindgen]
pub fn explore_landscape_info(name: &str) -> Result<String, JsValue> {
    let f = make_landscape(name)?;
    let (lo, hi) = f.bounds();
    Ok(json!({
        "name": f.name(),
        "lo": lo,
        "hi": hi,
        "optimum_fitness": f.optimal_fitness(),
        "optimum": f.optimal_solution(),
    })
    .to_string())
}

// ============================================================================
// Fitness adapters
// ============================================================================

/// Maximizing adapter over a raw (minimizing) benchmark, for GA-family
/// algorithms whose convention is higher-is-better.
struct MaximizeBench(Arc<Box<dyn BenchmarkFunction>>);

impl Fitness for MaximizeBench {
    type Genome = RealVector;
    type Value = f64;

    fn evaluate(&self, genome: &RealVector) -> f64 {
        -self.0.evaluate_raw(genome.genes())
    }
}

/// Minimizing adapter for CMA-ES (its native convention).
struct MinimizeBench(Arc<Box<dyn BenchmarkFunction>>);

impl CmaEsFitness for MinimizeBench {
    fn evaluate(&self, x: &RealVector) -> f64 {
        self.0.evaluate_raw(x.genes())
    }
}

fn xyf(genome: &RealVector, raw: f64) -> serde_json::Value {
    let g = genome.genes();
    json!([g[0], g[1], raw])
}

/// Mean pairwise Euclidean distance — the diversity readout the widgets plot.
fn mean_pairwise_distance(points: &[&RealVector]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = points[i].genes();
            let b = points[j].genes();
            let d2: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
            total += d2.sqrt();
            count += 1;
        }
    }
    total / count as f64
}

// ============================================================================
// GA anatomy engine — selection / crossover / mutation, phase by phase
// ============================================================================

/// One-generation-at-a-time GA over a 2-D benchmark landscape, driven directly
/// through fugue-evo's real operators (`TournamentSelection`, `SbxCrossover`,
/// `PolynomialMutation`) so each phase of a generation can be shown as it
/// happens: who got selected, what crossover produced, where mutation nudged.
#[wasm_bindgen]
pub struct ExploreGa {
    landscape: Arc<Box<dyn BenchmarkFunction>>,
    bounds: MultiBounds,
    selection: TournamentSelection,
    crossover: SbxCrossover,
    mutation: PolynomialMutation,
    crossover_prob: f64,
    mutation_prob: f64,
    pop: Vec<(RealVector, f64)>, // (genome, raw fitness — lower better)
    rng: StdRng,
    generation: usize,
    best: (Vec<f64>, f64),
}

#[wasm_bindgen]
impl ExploreGa {
    #[wasm_bindgen(constructor)]
    pub fn new(landscape: &str, pop_size: usize, seed: u64) -> Result<ExploreGa, JsValue> {
        let f = Arc::new(make_landscape(landscape)?);
        let bounds = landscape_multibounds(f.as_ref().as_ref());
        let mut rng = StdRng::seed_from_u64(seed);
        let pop: Vec<(RealVector, f64)> = (0..pop_size.clamp(4, 512))
            .map(|_| {
                let g = RealVector::generate(&mut rng, &bounds);
                let raw = f.evaluate_raw(g.genes());
                (g, raw)
            })
            .collect();
        let best = pop
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(g, raw)| (g.genes().to_vec(), *raw))
            .unwrap_or((vec![0.0, 0.0], f64::INFINITY));
        Ok(ExploreGa {
            landscape: f,
            bounds,
            selection: TournamentSelection::new(3),
            crossover: SbxCrossover::new(15.0),
            mutation: PolynomialMutation::new(20.0),
            crossover_prob: 0.9,
            mutation_prob: 0.6,
            pop,
            rng,
            generation: 0,
            best,
        })
    }

    #[wasm_bindgen(js_name = setTournamentSize)]
    pub fn set_tournament_size(&mut self, k: usize) {
        self.selection = TournamentSelection::new(k.clamp(1, 16));
    }

    #[wasm_bindgen(js_name = setMutationEta)]
    pub fn set_mutation_eta(&mut self, eta: f64) {
        self.mutation = PolynomialMutation::new(eta.clamp(1.0, 200.0));
    }

    #[wasm_bindgen(js_name = setRates)]
    pub fn set_rates(&mut self, crossover_prob: f64, mutation_prob: f64) {
        self.crossover_prob = crossover_prob.clamp(0.0, 1.0);
        self.mutation_prob = mutation_prob.clamp(0.0, 1.0);
    }

    /// Advance one generation. Returns the full anatomy of the step:
    /// `{gen, population, parents, pairs, offspring_pre, offspring,
    ///   elite, best, diversity, mean_fitness}` — positions as `[x, y, raw]`.
    pub fn step(&mut self) -> String {
        let n = self.pop.len();
        // Selection sees the GA convention (higher better).
        let maximized: Vec<(RealVector, f64)> =
            self.pop.iter().map(|(g, raw)| (g.clone(), -raw)).collect();

        // Elite carries over unchanged.
        let elite_idx = self
            .pop
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1 .1
                    .partial_cmp(&b.1 .1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut offspring_pre: Vec<RealVector> = Vec::new();
        let mut offspring: Vec<(RealVector, f64)> = Vec::new();
        offspring.push(self.pop[elite_idx].clone());

        while offspring.len() < n {
            let i = self.selection.select(&maximized, &mut self.rng);
            let j = self.selection.select(&maximized, &mut self.rng);
            pairs.push((i, j));

            use rand::Rng;
            let (mut c1, mut c2) = if self.rng.gen::<f64>() < self.crossover_prob {
                match self.crossover.crossover_bounded(
                    &self.pop[i].0,
                    &self.pop[j].0,
                    &self.bounds,
                    &mut self.rng,
                ) {
                    fugue_evo::error::OperatorResult::Success((a, b)) => (a, b),
                    _ => (self.pop[i].0.clone(), self.pop[j].0.clone()),
                }
            } else {
                (self.pop[i].0.clone(), self.pop[j].0.clone())
            };

            for c in [&mut c1, &mut c2] {
                offspring_pre.push(c.clone());
                if self.rng.gen::<f64>() < self.mutation_prob {
                    self.mutation.mutate_bounded(c, &self.bounds, &mut self.rng);
                }
                if offspring.len() < n {
                    let raw = self.landscape.evaluate_raw(c.genes());
                    offspring.push((c.clone(), raw));
                }
            }
        }

        self.generation += 1;

        let parents: Vec<usize> = pairs.iter().flat_map(|&(i, j)| [i, j]).collect();
        let population_json: Vec<_> = self.pop.iter().map(|(g, raw)| xyf(g, *raw)).collect();
        let pre_json: Vec<_> = offspring_pre
            .iter()
            .map(|g| json!([g.genes()[0], g.genes()[1]]))
            .collect();
        let off_json: Vec<_> = offspring.iter().map(|(g, raw)| xyf(g, *raw)).collect();

        self.pop = offspring;
        for (g, raw) in &self.pop {
            if *raw < self.best.1 {
                self.best = (g.genes().to_vec(), *raw);
            }
        }
        let genomes: Vec<&RealVector> = self.pop.iter().map(|(g, _)| g).collect();
        let diversity = mean_pairwise_distance(&genomes);
        let mean_raw: f64 =
            self.pop.iter().map(|(_, raw)| raw).sum::<f64>() / self.pop.len() as f64;

        json!({
            "gen": self.generation,
            "population": population_json,
            "parents": parents,
            "pairs": pairs.iter().map(|&(i, j)| json!([i, j])).collect::<Vec<_>>(),
            "offspring_pre": pre_json,
            "offspring": off_json,
            "elite": elite_idx,
            "best": [self.best.0[0], self.best.0[1], self.best.1],
            "diversity": diversity,
            "mean_fitness": mean_raw,
        })
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

// ============================================================================
// CMA-ES engine — the adapting covariance, live
// ============================================================================

/// Incremental CMA-ES over a 2-D benchmark, streaming the search
/// distribution's mean, step size, covariance, and eigenstructure each
/// generation — everything the covariance-ellipse widget draws.
#[wasm_bindgen]
pub struct ExploreCma {
    cma: CmaEs<MinimizeBench>,
    fitness: MinimizeBench,
    rng: StdRng,
}

#[wasm_bindgen]
impl ExploreCma {
    #[wasm_bindgen(constructor)]
    pub fn new(
        landscape: &str,
        x0: f64,
        y0: f64,
        sigma0: f64,
        lambda: usize,
        seed: u64,
    ) -> Result<ExploreCma, JsValue> {
        let f = Arc::new(make_landscape(landscape)?);
        let bounds = landscape_multibounds(f.as_ref().as_ref());
        let sigma = if sigma0 > 0.0 { sigma0 } else { 1.0 };
        let cma = if lambda >= 4 {
            CmaEs::with_lambda(vec![x0, y0], sigma, lambda.min(64))
        } else {
            CmaEs::new(vec![x0, y0], sigma)
        }
        .with_bounds(bounds);
        let fitness = MinimizeBench(f);
        Ok(ExploreCma {
            cma,
            fitness,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// One generation: `{gen, mean, sigma, cov: [c00,c01,c10,c11],
    /// eigenvalues, eigenvectors, population, best, evaluations, converged}`.
    pub fn step(&mut self) -> Result<String, JsValue> {
        let population = self
            .cma
            .step(&self.fitness, &mut self.rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let pop_json: Vec<_> = population
            .iter()
            .map(|ind| {
                let raw = ind.fitness.unwrap_or(f64::NAN);
                xyf(&ind.genome, raw)
            })
            .collect();
        let s = &self.cma.state;
        Ok(json!({
            "gen": s.generation,
            "mean": s.mean,
            "sigma": s.sigma,
            "cov": [s.covariance[0][0], s.covariance[0][1], s.covariance[1][0], s.covariance[1][1]],
            "eigenvalues": s.eigenvalues,
            "eigenvectors": s.eigenvectors,
            "population": pop_json,
            "best": [s.best_solution[0], s.best_solution[1], s.best_fitness],
            "evaluations": s.evaluations,
            "converged": s.has_converged(),
        })
        .to_string())
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.cma.state.generation
    }
}

// ============================================================================
// NSGA-II engine — a Pareto front forming live
// ============================================================================

enum MoProblem {
    Zdt1(Zdt1),
    Zdt2(Zdt2),
    Zdt3(Zdt3),
    Schaffer(SchafferN1),
}

impl MoProblem {
    fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        match self {
            MoProblem::Zdt1(p) => p.evaluate(x).to_vec(),
            MoProblem::Zdt2(p) => p.evaluate(x).to_vec(),
            MoProblem::Zdt3(p) => p.evaluate(x).to_vec(),
            MoProblem::Schaffer(p) => p.evaluate(x[0]).to_vec(),
        }
    }

    fn bounds(&self, dim: usize) -> MultiBounds {
        let (lo, hi) = match self {
            MoProblem::Schaffer(p) => p.bounds(),
            MoProblem::Zdt1(p) => p.bounds(),
            MoProblem::Zdt2(p) => p.bounds(),
            MoProblem::Zdt3(p) => p.bounds(),
        };
        MultiBounds::uniform(Bounds::new(lo, hi), dim)
    }
}

struct MoFitness {
    problem: MoProblem,
}

impl fugue_evo::algorithms::nsga2::MultiObjectiveFitness<RealVector> for MoFitness {
    fn num_objectives(&self) -> usize {
        2
    }

    fn evaluate(&self, genome: &RealVector) -> Vec<f64> {
        self.problem.evaluate(genome.genes())
    }
}

/// Incremental NSGA-II on a two-objective benchmark (`zdt1|zdt2|zdt3|schaffer`),
/// streaming objective-space points with their Pareto rank and crowding
/// distance each generation — the front forms on screen.
#[wasm_bindgen]
pub struct ExploreNsga2 {
    nsga: Nsga2<RealVector, MoFitness, SbxCrossover, PolynomialMutation>,
    fitness: MoFitness,
    crossover: SbxCrossover,
    mutation: PolynomialMutation,
    bounds: MultiBounds,
    pop: Vec<Nsga2Individual<RealVector>>,
    rng: StdRng,
    generation: usize,
}

#[wasm_bindgen]
impl ExploreNsga2 {
    #[wasm_bindgen(constructor)]
    pub fn new(problem: &str, pop_size: usize, seed: u64) -> Result<ExploreNsga2, JsValue> {
        let (mo, dim) = match problem {
            "zdt1" => (MoProblem::Zdt1(Zdt1::new(6)), 6),
            "zdt2" => (MoProblem::Zdt2(Zdt2::new(6)), 6),
            "zdt3" => (MoProblem::Zdt3(Zdt3::new(6)), 6),
            "schaffer" => (MoProblem::Schaffer(SchafferN1::new()), 1),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown problem '{problem}' (expected zdt1|zdt2|zdt3|schaffer)"
                )))
            }
        };
        let bounds = mo.bounds(dim);
        let fitness = MoFitness { problem: mo };
        let nsga = Nsga2::new(pop_size.clamp(8, 256));
        let mut rng = StdRng::seed_from_u64(seed);
        let pop = nsga.initialize_population(&fitness, &bounds, &mut rng);
        Ok(ExploreNsga2 {
            nsga,
            fitness,
            crossover: SbxCrossover::new(15.0),
            mutation: PolynomialMutation::new(20.0),
            bounds,
            pop,
            rng,
            generation: 0,
        })
    }

    /// One generation: `{gen, points: [[f1, f2, rank, crowding]...], fronts}`.
    pub fn step(&mut self) -> String {
        self.nsga.step_bounded(
            &mut self.pop,
            &self.fitness,
            &self.crossover,
            &self.mutation,
            &self.bounds,
            &mut self.rng,
        );
        self.generation += 1;
        self.snapshot()
    }

    /// Current population without stepping (for initial paint).
    pub fn snapshot(&self) -> String {
        let points: Vec<_> = self
            .pop
            .iter()
            .map(|ind| {
                let crowding = if ind.crowding_distance.is_finite() {
                    ind.crowding_distance
                } else {
                    -1.0 // JSON has no Infinity; -1 marks a boundary point
                };
                // Before the first step, individuals are unranked (usize::MAX).
                let rank: i64 = if ind.rank == usize::MAX {
                    -1
                } else {
                    ind.rank as i64
                };
                json!([ind.objectives[0], ind.objectives[1], rank, crowding])
            })
            .collect();
        let n_fronts = self
            .pop
            .iter()
            .map(|i| i.rank)
            .filter(|&r| r != usize::MAX)
            .max()
            .map_or(0, |r| r + 1);
        json!({
            "gen": self.generation,
            "points": points,
            "fronts": n_fronts,
        })
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

// ============================================================================
// Island model engine — divergence and migration
// ============================================================================

type IslandGa = IslandModel<
    RealVector,
    MaximizeBench,
    TournamentSelection,
    SbxCrossover,
    PolynomialMutation,
    f64,
>;

/// Incremental island-model GA on a 2-D benchmark: independent populations
/// diverging, then migration reseeding them on the configured interval.
#[wasm_bindgen]
pub struct ExploreIsland {
    landscape: Arc<Box<dyn BenchmarkFunction>>,
    model: IslandGa,
    migration_interval: usize,
    rng: StdRng,
}

#[wasm_bindgen]
impl ExploreIsland {
    #[wasm_bindgen(constructor)]
    pub fn new(
        landscape: &str,
        num_islands: usize,
        island_pop: usize,
        migration_interval: usize,
        topology: &str,
        seed: u64,
    ) -> Result<ExploreIsland, JsValue> {
        let f = Arc::new(make_landscape(landscape)?);
        let bounds = landscape_multibounds(f.as_ref().as_ref());
        let topo = match topology {
            "ring" => MigrationTopology::Ring,
            "full" => MigrationTopology::FullyConnected,
            "random" => MigrationTopology::Random,
            "star" => MigrationTopology::Star { hub_index: 0 },
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unknown topology '{topology}' (expected ring|full|random|star)"
                )))
            }
        };
        let interval = migration_interval.max(1);
        let mut rng = StdRng::seed_from_u64(seed);
        let model = IslandModelBuilder::new()
            .num_islands(num_islands.clamp(2, 12))
            .island_population_size(island_pop.clamp(4, 128))
            .migration_interval(interval)
            .topology(topo)
            .migration_policy(MigrationPolicy::Best(2))
            .bounds(bounds)
            .elitism(1)
            .fitness(MaximizeBench(Arc::clone(&f)))
            .selection(TournamentSelection::new(3))
            .crossover(SbxCrossover::new(15.0))
            .mutation(PolynomialMutation::new(20.0))
            .build(&mut rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(ExploreIsland {
            landscape: f,
            model,
            migration_interval: interval,
            rng,
        })
    }

    /// One generation across all islands:
    /// `{gen, migrated, islands: [{population, best, diversity}...], global_best}`.
    pub fn step(&mut self) -> Result<String, JsValue> {
        self.model
            .step(&mut self.rng)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let gen = self.model.generation;
        let migrated = gen % self.migration_interval == 0;

        let islands: Vec<_> = self
            .model
            .islands
            .iter()
            .map(|island| {
                let pop_json: Vec<_> = island
                    .population
                    .iter()
                    .map(|ind| {
                        let raw = self.landscape.evaluate_raw(ind.genome.genes());
                        xyf(&ind.genome, raw)
                    })
                    .collect();
                let genomes: Vec<&RealVector> =
                    island.population.iter().map(|ind| &ind.genome).collect();
                let best = island.best.as_ref().map(|b| {
                    let raw = self.landscape.evaluate_raw(b.genome.genes());
                    json!([b.genome.genes()[0], b.genome.genes()[1], raw])
                });
                json!({
                    "population": pop_json,
                    "best": best,
                    "diversity": mean_pairwise_distance(&genomes),
                })
            })
            .collect();

        let global_best = self
            .model
            .islands
            .iter()
            .filter_map(|i| i.best.as_ref())
            .map(|b| {
                (
                    b.genome.genes().to_vec(),
                    self.landscape.evaluate_raw(b.genome.genes()),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(g, raw)| json!([g[0], g[1], raw]));

        Ok(json!({
            "gen": gen,
            "migrated": migrated,
            "islands": islands,
            "global_best": global_best,
        })
        .to_string())
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.model.generation
    }
}

// ============================================================================
// UMDA engine — a distribution that learns
// ============================================================================

/// Incremental continuous UMDA (an EDA) on a 2-D benchmark, driven through the
/// crate's real `ContinuousUnivariateModel`: sample, select, refit — streaming
/// the model's mean and variance so the widget can draw the learned Gaussian
/// shrinking onto the optimum.
#[wasm_bindgen]
pub struct ExploreUmda {
    landscape: Arc<Box<dyn BenchmarkFunction>>,
    bounds: MultiBounds,
    model: ContinuousUnivariateModel,
    config: UMDAConfig,
    pop_size: usize,
    selection_ratio: f64,
    rng: StdRng,
    generation: usize,
    best: (Vec<f64>, f64),
}

#[wasm_bindgen]
impl ExploreUmda {
    #[wasm_bindgen(constructor)]
    pub fn new(
        landscape: &str,
        pop_size: usize,
        selection_ratio: f64,
        seed: u64,
    ) -> Result<ExploreUmda, JsValue> {
        let f = Arc::new(make_landscape(landscape)?);
        let bounds = landscape_multibounds(f.as_ref().as_ref());
        let model = ContinuousUnivariateModel::from_bounds(&bounds);
        Ok(ExploreUmda {
            landscape: f,
            bounds,
            model,
            config: UMDAConfig::default(),
            pop_size: pop_size.clamp(8, 512),
            selection_ratio: selection_ratio.clamp(0.05, 0.9),
            rng: StdRng::seed_from_u64(seed),
            generation: 0,
            best: (vec![0.0, 0.0], f64::INFINITY),
        })
    }

    /// One generation: `{gen, means, variances, population, selected, best}`.
    pub fn step(&mut self) -> String {
        let mut sampled: Vec<(RealVector, f64)> = (0..self.pop_size)
            .map(|_| {
                let g = self.model.sample(&self.bounds, &mut self.rng);
                let raw = self.landscape.evaluate_raw(g.genes());
                (g, raw)
            })
            .collect();
        sampled.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_select = ((self.pop_size as f64 * self.selection_ratio).round() as usize).max(2);
        let selected: Vec<&RealVector> = sampled.iter().take(n_select).map(|(g, _)| g).collect();
        self.model.update(&selected, &self.config);
        self.generation += 1;

        if let Some((g, raw)) = sampled.first() {
            if *raw < self.best.1 {
                self.best = (g.genes().to_vec(), *raw);
            }
        }

        json!({
            "gen": self.generation,
            "means": self.model.means,
            "variances": self.model.variances,
            "population": sampled.iter().map(|(g, raw)| xyf(g, *raw)).collect::<Vec<_>>(),
            "selected": n_select,
            "best": [self.best.0[0], self.best.0[1], self.best.1],
        })
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> usize {
        self.generation
    }
}

// ============================================================================
// Native tests (run with `cargo test -p fugue-evo-wasm`)
// ============================================================================

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn landscape_grid_shape_and_sphere_minimum() {
        let grid = explore_landscape_grid("sphere", -2.0, 2.0, -2.0, 2.0, 21, 21).unwrap();
        assert_eq!(grid.len(), 21 * 21);
        // Center cell (i=10, j=10) sits at the origin — the global minimum.
        let center = grid[10 * 21 + 10];
        assert!(grid.iter().all(|&v| v >= center));
    }

    #[test]
    fn ga_engine_improves_and_is_deterministic() {
        let run = |seed: u64| {
            let mut ga = ExploreGa::new("rastrigin", 40, seed).unwrap();
            let mut last = String::new();
            for _ in 0..30 {
                last = ga.step();
            }
            last
        };
        let a = run(7);
        let b = run(7);
        assert_eq!(a, b, "same seed must replay identically");

        let v: serde_json::Value = serde_json::from_str(&a).unwrap();
        let best = v["best"][2].as_f64().unwrap();
        assert!(
            best < 5.0,
            "GA should find a decent Rastrigin point, got {best}"
        );
    }

    #[test]
    fn cma_engine_shrinks_onto_sphere_optimum() {
        let mut cma = ExploreCma::new("sphere", 3.0, -3.0, 1.5, 0, 11).unwrap();
        let mut last = String::new();
        for _ in 0..40 {
            last = cma.step().unwrap();
        }
        let v: serde_json::Value = serde_json::from_str(&last).unwrap();
        let best = v["best"][2].as_f64().unwrap();
        assert!(best < 1e-3, "CMA-ES should nail the sphere, got {best}");
        assert!(
            v["sigma"].as_f64().unwrap() < 1.5,
            "step size should shrink"
        );
        assert_eq!(v["cov"].as_array().unwrap().len(), 4);
        assert_eq!(v["eigenvalues"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn nsga2_engine_front_advances() {
        let mut nsga = ExploreNsga2::new("zdt1", 60, 3).unwrap();
        let first: serde_json::Value = serde_json::from_str(&nsga.snapshot()).unwrap();
        assert_eq!(first["points"].as_array().unwrap().len(), 60);
        let mut last = String::new();
        for _ in 0..40 {
            last = nsga.step();
        }
        let v: serde_json::Value = serde_json::from_str(&last).unwrap();
        // After 40 generations most points should be rank 0 (on the front).
        let rank0 = v["points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p[2].as_u64() == Some(0))
            .count();
        assert!(
            rank0 > 30,
            "front should dominate the population, got {rank0}"
        );
    }

    #[test]
    fn island_engine_reports_migration_on_interval() {
        let mut island = ExploreIsland::new("ackley", 4, 20, 5, "ring", 5).unwrap();
        for gen in 1..=10 {
            let v: serde_json::Value = serde_json::from_str(&island.step().unwrap()).unwrap();
            assert_eq!(v["gen"].as_u64().unwrap(), gen);
            assert_eq!(v["migrated"].as_bool().unwrap(), gen % 5 == 0);
            assert_eq!(v["islands"].as_array().unwrap().len(), 4);
        }
    }

    #[test]
    fn umda_engine_variance_contracts() {
        let mut umda = ExploreUmda::new("sphere", 80, 0.3, 21).unwrap();
        let first: serde_json::Value = serde_json::from_str(&umda.step()).unwrap();
        let mut last = first.clone();
        for _ in 0..30 {
            last = serde_json::from_str(&umda.step()).unwrap();
        }
        let v0 = first["variances"][0].as_f64().unwrap();
        let v1 = last["variances"][0].as_f64().unwrap();
        assert!(v1 < v0, "model variance should contract ({v0} -> {v1})");
        assert!(last["best"][2].as_f64().unwrap() < 0.5);
    }
}
