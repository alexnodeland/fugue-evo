//! Integration regression tests for the July 2026 audit findings in the
//! `interactive` module (EV-06, EV-26/EV-64, EV-63).
//!
//! These drive the public `InteractiveGA` / session API end-to-end so they pin
//! behavior that only manifests once the pieces are wired together.

use fugue_evo::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn build_ga(
    mode: EvaluationMode,
    pop: usize,
    elitism: usize,
) -> InteractiveGA<RealVector, TournamentSelection, SbxCrossover, PolynomialMutation> {
    InteractiveGABuilder::<RealVector, (), (), ()>::new()
        .population_size(pop)
        .elitism_count(elitism)
        .evaluation_mode(mode)
        .batch_size(pop)
        .min_coverage(1.0)
        .comparisons_per_candidate(1)
        .max_generations(3)
        .bounds(MultiBounds::symmetric(1.0, 2))
        .selection(TournamentSelection::new(2))
        .crossover(SbxCrossover::new(15.0))
        .mutation(PolynomialMutation::new(20.0))
        .build()
        .expect("builder should succeed")
}

fn first_request(
    iga: &mut InteractiveGA<RealVector, TournamentSelection, SbxCrossover, PolynomialMutation>,
    rng: &mut StdRng,
) -> EvaluationRequest<RealVector> {
    match iga.step(rng) {
        StepResult::NeedsEvaluation(r) => r,
        other => panic!("expected NeedsEvaluation, got {other:?}"),
    }
}

/// regression: EV-06 — Bradley-Terry MLE must be re-fit inside the live loop so
/// pairwise feedback actually orders candidates. A beats B beats C repeatedly
/// (via the live `process_pairwise` entry point) must make
/// `ranked_candidates()` order A > B > C with separated strengths. Pre-fix the
/// MLE ran only in tests, leaving every candidate at its initial strength.
#[test]
fn ev06_bradley_terry_drives_session_ranking() {
    let mut agg = FitnessAggregator::new(AggregationModel::BradleyTerry {
        initial_strength: 1.0,
        optimizer: BradleyTerryOptimizer::default(),
    });
    let a = CandidateId(0);
    let b = CandidateId(1);
    let c = CandidateId(2);

    // Record comparisons ONLY through the live loop entry point.
    for _ in 0..10 {
        agg.process_pairwise(a, b, Some(a));
        agg.process_pairwise(b, c, Some(b));
        agg.process_pairwise(a, c, Some(a));
    }

    let mut session: InteractiveSession<RealVector> = InteractiveSession::new(agg);
    session.replace_population(vec![
        Candidate::new(a, RealVector::new(vec![0.0])),
        Candidate::new(b, RealVector::new(vec![1.0])),
        Candidate::new(c, RealVector::new(vec![2.0])),
    ]);
    session.sync_fitness_estimates();

    let ranked = session.ranked_candidates();
    assert_eq!(ranked[0].id, a, "A should rank first");
    assert_eq!(ranked[1].id, b, "B should rank second");
    assert_eq!(ranked[2].id, c, "C should rank last");

    let fa = ranked[0].fitness_estimate.unwrap();
    let fb = ranked[1].fitness_estimate.unwrap();
    let fc = ranked[2].fitness_estimate.unwrap();
    assert!(
        fa > fb && fb > fc,
        "strengths must be ordered: {fa} {fb} {fc}"
    );
    assert!(
        (fa - fc).abs() > 1e-2,
        "strengths must be separated, not frozen"
    );
}

/// regression: EV-26 / EV-64 — a rated candidate's `evaluation_count` must
/// increase by exactly ONE per response (it was double-counted).
#[test]
fn ev26_rating_counts_exactly_once() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut iga = build_ga(EvaluationMode::Rating, 4, 1);

    let req = first_request(&mut iga, &mut rng);
    let ids = req.candidate_ids();
    assert_eq!(ids.len(), 4);
    let ratings: Vec<_> = ids.iter().map(|&id| (id, 5.0)).collect();
    iga.provide_response(EvaluationResponse::ratings(ratings));

    for id in &ids {
        let c = iga.session().get_candidate(*id).unwrap();
        assert_eq!(
            c.evaluation_count, 1,
            "rated candidate {id:?} counted {} times",
            c.evaluation_count
        );
    }
}

/// regression: EV-26 / EV-64 — in pairwise mode BOTH the winner and the loser
/// must increase by exactly one (pre-fix the winner got +2 and the loser +1,
/// an asymmetric bias).
#[test]
fn ev64_pairwise_counts_symmetric() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut iga = build_ga(EvaluationMode::Pairwise, 4, 1);

    let req = first_request(&mut iga, &mut rng);
    let ids = req.candidate_ids();
    assert_eq!(ids.len(), 2);
    // First id wins.
    iga.provide_response(EvaluationResponse::winner(ids[0]));

    let winner = iga.session().get_candidate(ids[0]).unwrap();
    let loser = iga.session().get_candidate(ids[1]).unwrap();
    assert_eq!(
        winner.evaluation_count, 1,
        "winner counted {} times",
        winner.evaluation_count
    );
    assert_eq!(
        loser.evaluation_count, 1,
        "loser counted {} times",
        loser.evaluation_count
    );
}

/// regression: EV-63 — elites carried into the next generation must keep their
/// `CandidateId` so the aggregator's accumulated feedback survives. Pre-fix a
/// fresh id was minted, orphaning the elite's stats.
#[test]
fn ev63_elite_keeps_candidate_id_and_history() {
    let mut rng = StdRng::seed_from_u64(123);
    let mut iga = build_ga(EvaluationMode::Rating, 4, 1);

    let req = first_request(&mut iga, &mut rng);
    let ids = req.candidate_ids();
    // Distinct ratings so there is an unambiguous best (elite).
    let ratings: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, (i + 1) as f64))
        .collect();
    iga.provide_response(EvaluationResponse::ratings(ratings));

    let elite_id = iga.session().ranked_candidates()[0].id;
    assert!(
        iga.session().aggregator.get_stats(&elite_id).is_some(),
        "elite should have aggregator history before evolution"
    );

    match iga.step(&mut rng) {
        StepResult::GenerationComplete { .. } => {}
        other => panic!("expected GenerationComplete, got {other:?}"),
    }

    assert!(
        iga.session().population.iter().any(|c| c.id == elite_id),
        "elite id {elite_id:?} was re-minted across generations"
    );
    assert!(
        iga.session().aggregator.get_stats(&elite_id).is_some(),
        "elite aggregator history must survive the generation boundary"
    );
}
