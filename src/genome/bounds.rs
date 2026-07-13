//! Bounds for genome values
//!
//! This module provides bounds types for constraining genome values.

use serde::{Deserialize, Serialize};

use crate::error::GenomeError;

/// Bounds for a single dimension
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    /// Lower bound (inclusive)
    pub min: f64,
    /// Upper bound (inclusive)
    pub max: f64,
}

impl Bounds {
    /// Create new bounds.
    ///
    /// A degenerate (`min == max`) bound is allowed and represents a dimension
    /// pinned to a single constant value; see [`normalize`](Self::normalize) and
    /// [`denormalize`](Self::denormalize) for how such bounds behave.
    ///
    /// # Panics
    /// Panics if `min > max`. Use [`try_new`](Self::try_new) for a fallible
    /// constructor that returns a [`GenomeError`] instead of panicking.
    pub fn new(min: f64, max: f64) -> Self {
        Self::try_new(min, max).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallibly create new bounds, rejecting `min > max`.
    ///
    /// Returns `Err(GenomeError::InvalidStructure)` when `min > max`. A
    /// degenerate (`min == max`) bound is permitted.
    pub fn try_new(min: f64, max: f64) -> Result<Self, GenomeError> {
        // Reject `min > max` and any NaN operand. `partial_cmp` returns `None`
        // when either side is NaN, so NaN bounds are rejected exactly as the
        // prior `!(min <= max)` guard did (a NaN comparison is always false),
        // matching the invariant the original `assert!(min <= max)` held.
        if matches!(
            min.partial_cmp(&max),
            None | Some(std::cmp::Ordering::Greater)
        ) {
            return Err(GenomeError::InvalidStructure(format!(
                "Invalid bounds: min ({min}) must be <= max ({max})"
            )));
        }
        Ok(Self { min, max })
    }

    /// Create symmetric bounds centered at 0
    pub fn symmetric(half_width: f64) -> Self {
        Self::new(-half_width, half_width)
    }

    /// Create unit bounds [0, 1]
    pub fn unit() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Get the range (max - min)
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Get the center point
    pub fn center(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    /// Check if a value is within bounds
    pub fn contains(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    /// Clamp a value to be within bounds
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }

    /// Normalize a value from bounds to `[0, 1]`.
    ///
    /// For a degenerate (`min == max`) bound the range is zero, so there is no
    /// meaningful position within `[0, 1]`; this returns `0.5` (the midpoint)
    /// rather than dividing by zero and producing `NaN`/`±inf`.
    pub fn normalize(&self, value: f64) -> f64 {
        let range = self.range();
        if range <= 0.0 {
            return 0.5;
        }
        (value - self.min) / range
    }

    /// Denormalize a value from `[0, 1]` to bounds.
    ///
    /// For a degenerate (`min == max`) bound the range is zero, so every input
    /// maps to the single legal value `min`.
    pub fn denormalize(&self, value: f64) -> f64 {
        let range = self.range();
        if range <= 0.0 {
            return self.min;
        }
        self.min + value * range
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self::symmetric(5.12) // Common default for optimization benchmarks
    }
}

impl From<(f64, f64)> for Bounds {
    fn from((min, max): (f64, f64)) -> Self {
        Self::new(min, max)
    }
}

/// Multi-dimensional bounds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiBounds {
    /// Bounds for each dimension
    pub bounds: Vec<Bounds>,
}

impl MultiBounds {
    /// Create new multi-dimensional bounds
    pub fn new(bounds: Vec<Bounds>) -> Self {
        Self { bounds }
    }

    /// Create uniform bounds for all dimensions
    pub fn uniform(bound: Bounds, dimension: usize) -> Self {
        Self {
            bounds: vec![bound; dimension],
        }
    }

    /// Create symmetric bounds for all dimensions
    pub fn symmetric(half_width: f64, dimension: usize) -> Self {
        Self::uniform(Bounds::symmetric(half_width), dimension)
    }

    /// Get number of dimensions
    pub fn dimension(&self) -> usize {
        self.bounds.len()
    }

    /// Get bounds for a specific dimension
    pub fn get(&self, index: usize) -> Option<&Bounds> {
        self.bounds.get(index)
    }

    /// Clamp a vector to be within bounds
    pub fn clamp_vec(&self, values: &mut [f64]) {
        for (i, value) in values.iter_mut().enumerate() {
            if let Some(b) = self.bounds.get(i) {
                *value = b.clamp(*value);
            }
        }
    }

    /// Check if all values are within bounds
    pub fn contains_vec(&self, values: &[f64]) -> bool {
        values
            .iter()
            .enumerate()
            .all(|(i, &v)| self.bounds.get(i).is_some_and(|b| b.contains(v)))
    }
}

impl FromIterator<Bounds> for MultiBounds {
    fn from_iter<I: IntoIterator<Item = Bounds>>(iter: I) -> Self {
        Self {
            bounds: iter.into_iter().collect(),
        }
    }
}

impl FromIterator<(f64, f64)> for MultiBounds {
    fn from_iter<I: IntoIterator<Item = (f64, f64)>>(iter: I) -> Self {
        Self {
            bounds: iter.into_iter().map(Bounds::from).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_new() {
        let b = Bounds::new(-5.0, 5.0);
        assert_eq!(b.min, -5.0);
        assert_eq!(b.max, 5.0);
    }

    #[test]
    #[should_panic(expected = "Invalid bounds")]
    fn test_bounds_invalid() {
        Bounds::new(5.0, -5.0);
    }

    #[test]
    fn test_bounds_symmetric() {
        let b = Bounds::symmetric(3.0);
        assert_eq!(b.min, -3.0);
        assert_eq!(b.max, 3.0);
    }

    #[test]
    fn test_bounds_unit() {
        let b = Bounds::unit();
        assert_eq!(b.min, 0.0);
        assert_eq!(b.max, 1.0);
    }

    #[test]
    fn test_bounds_range() {
        let b = Bounds::new(-5.0, 5.0);
        assert_eq!(b.range(), 10.0);
    }

    #[test]
    fn test_bounds_center() {
        let b = Bounds::new(-2.0, 6.0);
        assert_eq!(b.center(), 2.0);
    }

    #[test]
    fn test_bounds_contains() {
        let b = Bounds::new(-5.0, 5.0);
        assert!(b.contains(0.0));
        assert!(b.contains(-5.0));
        assert!(b.contains(5.0));
        assert!(!b.contains(-5.1));
        assert!(!b.contains(5.1));
    }

    #[test]
    fn test_bounds_clamp() {
        let b = Bounds::new(-5.0, 5.0);
        assert_eq!(b.clamp(0.0), 0.0);
        assert_eq!(b.clamp(-10.0), -5.0);
        assert_eq!(b.clamp(10.0), 5.0);
    }

    #[test]
    fn test_bounds_normalize() {
        let b = Bounds::new(0.0, 10.0);
        assert_eq!(b.normalize(0.0), 0.0);
        assert_eq!(b.normalize(5.0), 0.5);
        assert_eq!(b.normalize(10.0), 1.0);
    }

    #[test]
    fn test_bounds_denormalize() {
        let b = Bounds::new(0.0, 10.0);
        assert_eq!(b.denormalize(0.0), 0.0);
        assert_eq!(b.denormalize(0.5), 5.0);
        assert_eq!(b.denormalize(1.0), 10.0);
    }

    #[test]
    fn test_bounds_try_new_rejects_min_gt_max() {
        // regression: EV-56 — the fallible constructor rejects min > max.
        let result = Bounds::try_new(5.0, -5.0);
        assert!(result.is_err());
        assert!(Bounds::try_new(-5.0, 5.0).is_ok());
        // Degenerate min == max is allowed.
        assert!(Bounds::try_new(3.0, 3.0).is_ok());
    }

    #[test]
    fn test_bounds_try_new_rejects_nan() {
        // regression: NaN bounds must be rejected. `partial_cmp` yields `None`
        // for a NaN operand, preserving the previous `!(min <= max)` behavior.
        assert!(Bounds::try_new(f64::NAN, 5.0).is_err());
        assert!(Bounds::try_new(-5.0, f64::NAN).is_err());
        assert!(Bounds::try_new(f64::NAN, f64::NAN).is_err());
    }

    #[test]
    fn test_bounds_degenerate_normalize_denormalize() {
        // regression: EV-56 — degenerate (min == max) bounds must not divide by
        // zero. normalize() previously returned NaN (0.0/0.0) for value == min.
        let b = Bounds::new(3.0, 3.0);
        assert_eq!(b.range(), 0.0);

        // normalize returns the midpoint 0.5 for any input (documented behavior),
        // and crucially is finite (never NaN / inf).
        assert_eq!(b.normalize(3.0), 0.5);
        assert!(b.normalize(3.0).is_finite());
        assert_eq!(b.normalize(100.0), 0.5);
        assert!(b.normalize(100.0).is_finite());

        // denormalize returns the single legal value min for any input.
        assert_eq!(b.denormalize(0.0), 3.0);
        assert_eq!(b.denormalize(0.5), 3.0);
        assert_eq!(b.denormalize(1.0), 3.0);
    }

    #[test]
    fn test_multi_bounds_uniform() {
        let mb = MultiBounds::symmetric(5.0, 3);
        assert_eq!(mb.dimension(), 3);
        assert_eq!(mb.get(0), Some(&Bounds::symmetric(5.0)));
        assert_eq!(mb.get(1), Some(&Bounds::symmetric(5.0)));
        assert_eq!(mb.get(2), Some(&Bounds::symmetric(5.0)));
        assert_eq!(mb.get(3), None);
    }

    #[test]
    fn test_multi_bounds_clamp_vec() {
        let mb = MultiBounds::symmetric(5.0, 3);
        let mut values = vec![-10.0, 0.0, 10.0];
        mb.clamp_vec(&mut values);
        assert_eq!(values, vec![-5.0, 0.0, 5.0]);
    }

    #[test]
    fn test_multi_bounds_contains_vec() {
        let mb = MultiBounds::symmetric(5.0, 3);
        assert!(mb.contains_vec(&[0.0, 0.0, 0.0]));
        assert!(mb.contains_vec(&[-5.0, 5.0, 0.0]));
        assert!(!mb.contains_vec(&[-6.0, 0.0, 0.0]));
    }

    #[test]
    fn test_multi_bounds_from_tuples() {
        let mb: MultiBounds = vec![(0.0, 1.0), (-10.0, 10.0)].into_iter().collect();
        assert_eq!(mb.dimension(), 2);
        assert_eq!(mb.get(0), Some(&Bounds::new(0.0, 1.0)));
        assert_eq!(mb.get(1), Some(&Bounds::new(-10.0, 10.0)));
    }
}
