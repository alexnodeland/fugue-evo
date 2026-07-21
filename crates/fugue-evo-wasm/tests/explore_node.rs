//! Node-target wasm tests for the incremental explore engines
//! (`wasm-pack test --node`): the same seeded-determinism and convergence
//! claims as the native tests, but executed as actual wasm32 — catching
//! anything that only breaks at the wasm boundary.

#![cfg(target_arch = "wasm32")]

use fugue_evo_wasm::{
    explore_landscape_grid, explore_landscape_info, ExploreCma, ExploreGa, ExploreIsland,
    ExploreNsga2, ExploreUmda,
};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn landscape_helpers_work_in_wasm() {
    let grid = explore_landscape_grid("sphere", -2.0, 2.0, -2.0, 2.0, 11, 11).unwrap();
    assert_eq!(grid.len(), 121);
    let center = grid[5 * 11 + 5];
    assert!(grid.iter().all(|&v| v >= center));

    let info = explore_landscape_info("rastrigin").unwrap();
    assert!(info.contains("Rastrigin"));
    assert!(explore_landscape_grid("nope", 0.0, 1.0, 0.0, 1.0, 2, 2).is_err());
}

#[wasm_bindgen_test]
fn ga_engine_is_deterministic_in_wasm() {
    let mut a = ExploreGa::new("rastrigin", 30, 7).unwrap();
    let mut b = ExploreGa::new("rastrigin", 30, 7).unwrap();
    let mut last_a = String::new();
    let mut last_b = String::new();
    for _ in 0..10 {
        last_a = a.step();
        last_b = b.step();
    }
    assert_eq!(last_a, last_b);
}

#[wasm_bindgen_test]
fn cma_engine_converges_in_wasm() {
    let mut cma = ExploreCma::new("sphere", 2.0, -2.0, 1.0, 0, 3).unwrap();
    let mut last = String::new();
    for _ in 0..35 {
        last = cma.step().unwrap();
    }
    let v: serde_json::Value = serde_json::from_str(&last).unwrap();
    assert!(v["best"][2].as_f64().unwrap() < 0.01);
}

#[wasm_bindgen_test]
fn nsga2_island_umda_step_in_wasm() {
    let mut nsga = ExploreNsga2::new("zdt1", 24, 5).unwrap();
    let v: serde_json::Value = serde_json::from_str(&nsga.step()).unwrap();
    assert_eq!(v["points"].as_array().unwrap().len(), 24);

    let mut island = ExploreIsland::new("ackley", 4, 12, 5, "ring", 5).unwrap();
    for gen in 1..=5 {
        let v: serde_json::Value = serde_json::from_str(&island.step().unwrap()).unwrap();
        assert_eq!(v["migrated"].as_bool().unwrap(), gen % 5 == 0);
    }

    let mut umda = ExploreUmda::new("sphere", 40, 0.3, 9).unwrap();
    let v: serde_json::Value = serde_json::from_str(&umda.step()).unwrap();
    assert_eq!(v["means"].as_array().unwrap().len(), 2);
}
