//! Bounds for genome values
//!
//! This module provides bounds types for constraining genome values.

use serde::{Deserialize, Serialize};

/// Bounds for a single dimension
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    /// Lower bound (inclusive)
    pub min: f64,
    /// Upper bound (inclusive)
    pub max: f64,
}

impl Bounds {
    /// Create new bounds
    ///
    /// # Panics
    /// Panics if min > max
    pub fn new(min: f64, max: f64) -> Self {
        assert!(
            min <= max,
            "Invalid bounds: min ({}) must be <= max ({})",
            min,
            max
        );
        Self { min, max }
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

    /// Normalize a value from bounds to [0, 1]
    pub fn normalize(&self, value: f64) -> f64 {
        (value - self.min) / self.range()
    }

    /// Denormalize a value from [0, 1] to bounds
    pub fn denormalize(&self, value: f64) -> f64 {
        self.min + value * self.range()
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
            .all(|(i, &v)| self.bounds.get(i).map_or(false, |b| b.contains(v)))
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
