//! Property-based tests for fugue-evo
//!
//! Uses proptest to verify invariants and properties of the library.

use fugue_evo::prelude::*;
use proptest::prelude::*;

proptest! {
    // ==================== RealVector Properties ====================

    #[test]
    fn real_vector_dimension_preserved(dim in 1usize..20) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, dim);
        let genome = RealVector::generate(&mut rng, &bounds);
        prop_assert_eq!(genome.dimension(), dim);
    }

    #[test]
    fn real_vector_genes_within_bounds(
        dim in 1usize..20,
        half_width in 0.1f64..100.0
    ) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(half_width, dim);
        let genome = RealVector::generate(&mut rng, &bounds);

        for gene in genome.genes() {
            prop_assert!(*gene >= -half_width && *gene <= half_width);
        }
    }

    #[test]
    fn real_vector_trace_roundtrip(genes in prop::collection::vec(-10.0..10.0f64, 1..20)) {
        let original = RealVector::new(genes);
        let trace = original.to_trace();
        let recovered = RealVector::from_trace(&trace).unwrap();
        prop_assert_eq!(original.genes(), recovered.genes());
    }

    #[test]
    fn real_vector_distance_symmetric(
        genes1 in prop::collection::vec(-10.0..10.0f64, 5),
        genes2 in prop::collection::vec(-10.0..10.0f64, 5)
    ) {
        let g1 = RealVector::new(genes1);
        let g2 = RealVector::new(genes2);
        let d1 = g1.distance(&g2);
        let d2 = g2.distance(&g1);
        prop_assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn real_vector_distance_non_negative(
        genes1 in prop::collection::vec(-10.0..10.0f64, 5),
        genes2 in prop::collection::vec(-10.0..10.0f64, 5)
    ) {
        let g1 = RealVector::new(genes1);
        let g2 = RealVector::new(genes2);
        prop_assert!(g1.distance(&g2) >= 0.0);
    }

    #[test]
    fn real_vector_distance_identity(genes in prop::collection::vec(-10.0..10.0f64, 5)) {
        let g = RealVector::new(genes);
        prop_assert!((g.distance(&g) - 0.0).abs() < 1e-10);
    }

    // ==================== BitString Properties ====================

    #[test]
    fn bit_string_dimension_preserved(len in 1usize..100) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, len);
        let genome = BitString::generate(&mut rng, &bounds);
        prop_assert_eq!(genome.dimension(), len);
    }

    #[test]
    fn bit_string_count_consistency(bits in prop::collection::vec(any::<bool>(), 1..100)) {
        let genome = BitString::new(bits.clone());
        let ones = genome.count_ones();
        let zeros = genome.count_zeros();
        prop_assert_eq!(ones + zeros, bits.len());
    }

    #[test]
    fn bit_string_trace_roundtrip(bits in prop::collection::vec(any::<bool>(), 1..50)) {
        let original = BitString::new(bits);
        let trace = original.to_trace();
        let recovered = BitString::from_trace(&trace).unwrap();
        prop_assert_eq!(original.bits(), recovered.bits());
    }

    // ==================== Permutation Properties ====================

    #[test]
    fn permutation_is_valid(n in 2usize..20) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, n);
        let genome = Permutation::generate(&mut rng, &bounds);
        prop_assert!(genome.is_valid_permutation());
    }

    #[test]
    fn permutation_contains_all_elements(n in 2usize..20) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, n);
        let genome = Permutation::generate(&mut rng, &bounds);
        let perm = genome.permutation();

        // Should contain all elements 0..n
        let mut sorted = perm.to_vec();
        sorted.sort();
        prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn permutation_trace_roundtrip(n in 2usize..15) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(1.0, n);
        let original = Permutation::generate(&mut rng, &bounds);
        let trace = original.to_trace();
        let recovered = Permutation::from_trace(&trace).unwrap();
        prop_assert_eq!(original.permutation(), recovered.permutation());
    }

    // ==================== Bounds Properties ====================

    #[test]
    fn bounds_clamp_within_range(
        min in -100.0..0.0f64,
        max in 0.1..100.0f64,
        value in -200.0..200.0f64
    ) {
        let bounds = Bounds::new(min, max);
        let clamped = bounds.clamp(value);
        prop_assert!(clamped >= min && clamped <= max);
    }

    #[test]
    fn bounds_contains_clamped(
        min in -100.0..0.0f64,
        max in 0.1..100.0f64,
        value in -200.0..200.0f64
    ) {
        let bounds = Bounds::new(min, max);
        let clamped = bounds.clamp(value);
        prop_assert!(bounds.contains(clamped));
    }

    #[test]
    fn multi_bounds_dimension_correct(dim in 1usize..50) {
        let bounds = MultiBounds::symmetric(5.0, dim);
        prop_assert_eq!(bounds.dimension(), dim);
    }

    // ==================== Crossover Properties ====================

    #[test]
    fn sbx_crossover_produces_valid_offspring(
        eta in 1.0..30.0f64,
        dim in 2usize..10
    ) {
        let mut rng = rand::thread_rng();
        let crossover = SbxCrossover::new(eta);
        let bounds = MultiBounds::symmetric(5.0, dim);

        let parent1 = RealVector::generate(&mut rng, &bounds);
        let parent2 = RealVector::generate(&mut rng, &bounds);

        let result = crossover.crossover(&parent1, &parent2, &mut rng);
        if let Some((child1, child2)) = result.genome() {
            prop_assert_eq!(child1.dimension(), dim);
            prop_assert_eq!(child2.dimension(), dim);
        }
    }

    // ==================== Selection Properties ====================

    #[test]
    fn tournament_selection_returns_valid_index(
        size in 2usize..10,
        pop_size in 10usize..50
    ) {
        let mut rng = rand::thread_rng();
        let selection = TournamentSelection::new(size);

        // Create a population with fitness values
        let population: Vec<(RealVector, f64)> = (0..pop_size)
            .map(|i| (RealVector::new(vec![i as f64]), i as f64))
            .collect();

        let idx = selection.select(&population, &mut rng);
        prop_assert!(idx < pop_size);
    }

    // ==================== Fitness Function Properties ====================

    #[test]
    fn sphere_fitness_at_origin_is_optimal(dim in 1usize..20) {
        let fitness = Sphere::new(dim);
        let origin = RealVector::new(vec![0.0; dim]);
        let f = fitness.evaluate(&origin);
        // Sphere at origin should be 0 (or negated if maximizing)
        prop_assert!(f.abs() < 1e-10 || f == 0.0 || (-f).abs() < 1e-10);
    }

    #[test]
    fn rastrigin_has_global_minimum_at_origin(dim in 1usize..10) {
        let fitness = Rastrigin::new(dim);
        let origin = RealVector::new(vec![0.0; dim]);
        let at_origin = fitness.evaluate(&origin);

        // Test a nearby point
        let nearby = RealVector::new(vec![0.1; dim]);
        let at_nearby = fitness.evaluate(&nearby);

        // Origin should have better (higher) fitness since we negate for maximization
        prop_assert!(at_origin >= at_nearby);
    }

    // ==================== Population Properties ====================

    #[test]
    fn population_maintains_size(pop_size in 5usize..50, dim in 2usize..10) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, dim);
        let population: Population<RealVector, f64> = Population::random(pop_size, &bounds, &mut rng);
        prop_assert_eq!(population.len(), pop_size);
    }

    #[test]
    fn population_best_has_highest_fitness(pop_size in 5usize..30) {
        let mut rng = rand::thread_rng();
        let bounds = MultiBounds::symmetric(5.0, 5);
        let fitness = Sphere::new(5);

        let mut population: Population<RealVector, f64> = Population::random(pop_size, &bounds, &mut rng);
        population.evaluate(&fitness);

        if let Some(best) = population.best() {
            let best_fitness = best.fitness_value();
            for ind in population.iter() {
                if let Some(f) = ind.fitness.as_ref() {
                    prop_assert!(*f <= *best_fitness);
                }
            }
        }
    }
}
