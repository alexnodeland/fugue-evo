//! Parameter schedules for deterministic control
//!
//! Schedules provide predetermined parameter values based on generation number.

use std::f64::consts::PI;

/// Parameter schedule trait
///
/// Defines how a parameter changes over the course of evolution.
pub trait ParameterSchedule: Send + Sync {
    /// Get the parameter value at a given generation
    fn value_at(&self, generation: usize, max_generations: usize) -> f64;
}

/// Constant parameter (no change)
#[derive(Clone, Debug)]
pub struct ConstantSchedule {
    /// The constant value
    pub value: f64,
}

impl ConstantSchedule {
    /// Create a new constant schedule
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

impl ParameterSchedule for ConstantSchedule {
    fn value_at(&self, _generation: usize, _max_generations: usize) -> f64 {
        self.value
    }
}

/// Linear annealing: p(t) = p_start + (p_end - p_start) * t / T
#[derive(Clone, Debug)]
pub struct LinearAnnealing {
    /// Starting value
    pub start: f64,
    /// Ending value
    pub end: f64,
}

impl LinearAnnealing {
    /// Create a new linear annealing schedule
    pub fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    /// Create a schedule that decreases from start to end
    pub fn decreasing(start: f64, end: f64) -> Self {
        Self::new(start, end)
    }

    /// Create a schedule that increases from start to end
    pub fn increasing(start: f64, end: f64) -> Self {
        Self::new(start, end)
    }
}

impl ParameterSchedule for LinearAnnealing {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64 {
        if max_generations == 0 {
            return self.start;
        }
        let t = generation as f64 / max_generations as f64;
        self.start + (self.end - self.start) * t
    }
}

/// Exponential decay: p(t) = p₀ * e^(-λt)
#[derive(Clone, Debug)]
pub struct ExponentialDecay {
    /// Initial value
    pub initial: f64,
    /// Decay rate (λ)
    pub decay_rate: f64,
    /// Minimum value (floor)
    pub minimum: f64,
}

impl ExponentialDecay {
    /// Create a new exponential decay schedule
    pub fn new(initial: f64, decay_rate: f64) -> Self {
        Self {
            initial,
            decay_rate,
            minimum: 0.0,
        }
    }

    /// Set the minimum value
    pub fn with_minimum(mut self, minimum: f64) -> Self {
        self.minimum = minimum;
        self
    }
}

impl ParameterSchedule for ExponentialDecay {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        (self.initial * (-self.decay_rate * generation as f64).exp()).max(self.minimum)
    }
}

/// Cosine annealing with optional warm restarts
///
/// p(t) = p_min + 0.5 * (p_max - p_min) * (1 + cos(π * t / T))
#[derive(Clone, Debug)]
pub struct CosineAnnealing {
    /// Maximum value
    pub max_value: f64,
    /// Minimum value
    pub min_value: f64,
    /// Period for warm restarts (None = single annealing)
    pub period: Option<usize>,
}

impl CosineAnnealing {
    /// Create a new cosine annealing schedule
    pub fn new(max_value: f64, min_value: f64) -> Self {
        Self {
            max_value,
            min_value,
            period: None,
        }
    }

    /// Enable warm restarts with given period
    pub fn with_warm_restarts(mut self, period: usize) -> Self {
        self.period = Some(period);
        self
    }
}

impl ParameterSchedule for CosineAnnealing {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64 {
        let effective_gen = match self.period {
            Some(period) if period > 0 => generation % period,
            _ => generation,
        };
        let effective_max = match self.period {
            Some(period) if period > 0 => period,
            _ => max_generations,
        };

        if effective_max == 0 {
            return self.max_value;
        }

        let t = effective_gen as f64 / effective_max as f64;
        self.min_value + 0.5 * (self.max_value - self.min_value) * (1.0 + (PI * t).cos())
    }
}

/// Step schedule: changes at specific generations
#[derive(Clone, Debug)]
pub struct StepSchedule {
    /// List of (generation, value) pairs, sorted by generation
    pub steps: Vec<(usize, f64)>,
    /// Initial value (before first step)
    pub initial: f64,
}

impl StepSchedule {
    /// Create a new step schedule
    pub fn new(initial: f64, steps: Vec<(usize, f64)>) -> Self {
        let mut steps = steps;
        steps.sort_by_key(|(gen, _)| *gen);
        Self { steps, initial }
    }

    /// Create a schedule with a single step
    pub fn single_step(initial: f64, step_gen: usize, step_value: f64) -> Self {
        Self::new(initial, vec![(step_gen, step_value)])
    }
}

impl ParameterSchedule for StepSchedule {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        let mut value = self.initial;
        for &(step_gen, step_value) in &self.steps {
            if generation >= step_gen {
                value = step_value;
            } else {
                break;
            }
        }
        value
    }
}

/// Polynomial decay: p(t) = p₀ * (1 - t/T)^power + p_min
#[derive(Clone, Debug)]
pub struct PolynomialDecay {
    /// Initial value
    pub initial: f64,
    /// Power of the polynomial
    pub power: f64,
    /// Minimum value at the end
    pub minimum: f64,
}

impl PolynomialDecay {
    /// Create a new polynomial decay schedule
    pub fn new(initial: f64, power: f64) -> Self {
        Self {
            initial,
            power,
            minimum: 0.0,
        }
    }

    /// Set the minimum value
    pub fn with_minimum(mut self, minimum: f64) -> Self {
        self.minimum = minimum;
        self
    }
}

impl ParameterSchedule for PolynomialDecay {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64 {
        if max_generations == 0 {
            return self.initial;
        }
        let t = generation as f64 / max_generations as f64;
        let decay = (1.0 - t).max(0.0).powf(self.power);
        self.minimum + (self.initial - self.minimum) * decay
    }
}

/// Cyclical schedule with triangular waves
#[derive(Clone, Debug)]
pub struct CyclicalSchedule {
    /// Base (minimum) value
    pub base: f64,
    /// Maximum value
    pub max_value: f64,
    /// Step size (generations per half cycle)
    pub step_size: usize,
}

impl CyclicalSchedule {
    /// Create a new cyclical schedule
    pub fn new(base: f64, max_value: f64, step_size: usize) -> Self {
        Self {
            base,
            max_value,
            step_size,
        }
    }
}

impl ParameterSchedule for CyclicalSchedule {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        if self.step_size == 0 {
            return self.base;
        }

        let cycle = generation / (2 * self.step_size);
        let x = (generation as f64 / self.step_size as f64) - 2.0 * cycle as f64;
        let scale = (1.0 - (x - 1.0).abs()).max(0.0);
        self.base + (self.max_value - self.base) * scale
    }
}

/// Enum-based schedule for when you need to combine different schedule types
#[derive(Clone, Debug)]
pub enum DynamicSchedule {
    Constant(ConstantSchedule),
    Linear(LinearAnnealing),
    Exponential(ExponentialDecay),
    Cosine(CosineAnnealing),
    Step(StepSchedule),
    Polynomial(PolynomialDecay),
    Cyclical(CyclicalSchedule),
}

impl ParameterSchedule for DynamicSchedule {
    fn value_at(&self, generation: usize, max_generations: usize) -> f64 {
        match self {
            Self::Constant(s) => s.value_at(generation, max_generations),
            Self::Linear(s) => s.value_at(generation, max_generations),
            Self::Exponential(s) => s.value_at(generation, max_generations),
            Self::Cosine(s) => s.value_at(generation, max_generations),
            Self::Step(s) => s.value_at(generation, max_generations),
            Self::Polynomial(s) => s.value_at(generation, max_generations),
            Self::Cyclical(s) => s.value_at(generation, max_generations),
        }
    }
}

impl From<ConstantSchedule> for DynamicSchedule {
    fn from(s: ConstantSchedule) -> Self {
        Self::Constant(s)
    }
}

impl From<LinearAnnealing> for DynamicSchedule {
    fn from(s: LinearAnnealing) -> Self {
        Self::Linear(s)
    }
}

impl From<ExponentialDecay> for DynamicSchedule {
    fn from(s: ExponentialDecay) -> Self {
        Self::Exponential(s)
    }
}

impl From<CosineAnnealing> for DynamicSchedule {
    fn from(s: CosineAnnealing) -> Self {
        Self::Cosine(s)
    }
}

impl From<StepSchedule> for DynamicSchedule {
    fn from(s: StepSchedule) -> Self {
        Self::Step(s)
    }
}

impl From<PolynomialDecay> for DynamicSchedule {
    fn from(s: PolynomialDecay) -> Self {
        Self::Polynomial(s)
    }
}

impl From<CyclicalSchedule> for DynamicSchedule {
    fn from(s: CyclicalSchedule) -> Self {
        Self::Cyclical(s)
    }
}

/// Composite schedule using enum phases
#[derive(Clone, Debug)]
pub struct CompositeSchedule {
    /// List of (end_generation, schedule) pairs
    pub phases: Vec<(usize, DynamicSchedule)>,
}

impl CompositeSchedule {
    /// Create a new composite schedule
    pub fn new() -> Self {
        Self { phases: Vec::new() }
    }

    /// Add a phase
    pub fn add_phase<S: Into<DynamicSchedule>>(mut self, end_gen: usize, schedule: S) -> Self {
        self.phases.push((end_gen, schedule.into()));
        self.phases.sort_by_key(|(gen, _)| *gen);
        self
    }
}

impl Default for CompositeSchedule {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterSchedule for CompositeSchedule {
    fn value_at(&self, generation: usize, _max_generations: usize) -> f64 {
        let mut prev_end = 0;
        for (end_gen, schedule) in &self.phases {
            if generation < *end_gen {
                let phase_duration = end_gen - prev_end;
                let phase_gen = generation - prev_end;
                return schedule.value_at(phase_gen, phase_duration);
            }
            prev_end = *end_gen;
        }
        // If past all phases, use the last phase's final value
        if let Some((end_gen, schedule)) = self.phases.last() {
            let phase_duration = end_gen - self.phases.get(self.phases.len().saturating_sub(2))
                .map(|(e, _)| *e)
                .unwrap_or(0);
            schedule.value_at(phase_duration, phase_duration)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constant_schedule() {
        let schedule = ConstantSchedule::new(0.5);
        assert_relative_eq!(schedule.value_at(0, 100), 0.5);
        assert_relative_eq!(schedule.value_at(50, 100), 0.5);
        assert_relative_eq!(schedule.value_at(100, 100), 0.5);
    }

    #[test]
    fn test_linear_annealing() {
        let schedule = LinearAnnealing::new(1.0, 0.0);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert_relative_eq!(schedule.value_at(50, 100), 0.5);
        assert_relative_eq!(schedule.value_at(100, 100), 0.0);
    }

    #[test]
    fn test_linear_annealing_increasing() {
        let schedule = LinearAnnealing::increasing(0.1, 0.9);
        assert_relative_eq!(schedule.value_at(0, 100), 0.1);
        assert_relative_eq!(schedule.value_at(100, 100), 0.9);
    }

    #[test]
    fn test_exponential_decay() {
        let schedule = ExponentialDecay::new(1.0, 0.1);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert!(schedule.value_at(10, 100) < 1.0);
        assert!(schedule.value_at(50, 100) < schedule.value_at(10, 100));
    }

    #[test]
    fn test_exponential_decay_with_minimum() {
        let schedule = ExponentialDecay::new(1.0, 0.1).with_minimum(0.1);
        assert!(schedule.value_at(1000, 100) >= 0.1);
    }

    #[test]
    fn test_cosine_annealing() {
        let schedule = CosineAnnealing::new(1.0, 0.0);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert_relative_eq!(schedule.value_at(100, 100), 0.0, epsilon = 1e-10);
        // Mid-point should be halfway between max and min
        assert_relative_eq!(schedule.value_at(50, 100), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts() {
        let schedule = CosineAnnealing::new(1.0, 0.0).with_warm_restarts(50);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert_relative_eq!(schedule.value_at(50, 100), 1.0); // Restart
        assert_relative_eq!(schedule.value_at(25, 100), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_step_schedule() {
        let schedule = StepSchedule::new(1.0, vec![(25, 0.5), (75, 0.1)]);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert_relative_eq!(schedule.value_at(24, 100), 1.0);
        assert_relative_eq!(schedule.value_at(25, 100), 0.5);
        assert_relative_eq!(schedule.value_at(74, 100), 0.5);
        assert_relative_eq!(schedule.value_at(75, 100), 0.1);
    }

    #[test]
    fn test_polynomial_decay() {
        let schedule = PolynomialDecay::new(1.0, 2.0).with_minimum(0.0);
        assert_relative_eq!(schedule.value_at(0, 100), 1.0);
        assert_relative_eq!(schedule.value_at(100, 100), 0.0);
        // Quadratic decay: at t=0.5, value = (1-0.5)^2 = 0.25
        assert_relative_eq!(schedule.value_at(50, 100), 0.25);
    }

    #[test]
    fn test_cyclical_schedule() {
        let schedule = CyclicalSchedule::new(0.0, 1.0, 10);
        assert_relative_eq!(schedule.value_at(0, 100), 0.0);
        assert_relative_eq!(schedule.value_at(10, 100), 1.0);
        assert_relative_eq!(schedule.value_at(20, 100), 0.0);
        assert_relative_eq!(schedule.value_at(30, 100), 1.0);
    }
}
