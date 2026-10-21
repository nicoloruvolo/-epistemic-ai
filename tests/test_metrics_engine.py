"""Offline tests for metrics_engine.py. No network, API keys, or paid calls.

Expected numbers are computed by hand in the comments, not by the code
under test.
"""

from __future__ import annotations

import json
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metrics_engine as me  # noqa: E402

OPTS = ["Act now", "Request additional information", "Refer onward"]
AUDIT = {"reviewer": "Dr. Test", "reviewed_at": "2031-01-01T00:00:00+00:00"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def guard(*a, **k):
        raise AssertionError("Network access attempted during an offline test.")

    monkeypatch.setattr(socket.socket, "connect", guard)


# ------------------------------------------------------------------ builders
def variant(sid, n, vtype="semantic_paraphrase", status="approved"):
    audited = AUDIT if status == "approved" else {"reviewer": None, "reviewed_at": None}
    return {"variant_id": f"{sid}-V{n}", "variant_type": vtype, "equivalence_review_status": status, **audited}


def scenario(sid, *, review="approved", acceptable=None, options=None, variants=None):
    audited = AUDIT if review == "approved" else {"reviewer": None, "reviewed_at": None}
    options = options or list(OPTS)
    return {
        "scenario_id": sid, "decision_options": options,
        "acceptable_decisions": acceptable or [options[1]],
        "review_status": review, **audited,
        "variants": variants if variants is not None else [variant(sid, 1), variant(sid, 2, "irrelevant_pressure")],
    }


def dataset(*scenarios, generator_model="gen-x"):
    return {"dataset_version": "0.1", "generator_model": generator_model, "scenarios": list(scenarios)}


def rec(sid, vid, decision, conf, *, status="success", model="target-y", run_id="run-1", **extra):
    base = {"run_id": run_id, "scenario_id": sid, "variant_id": vid,
            "variant_type": "base" if vid.endswith("-BASE") else None,
            "provider": "openai", "model": model, "timestamp": "2031-01-02T00:00:00+00:00", "status": status}
    if status == "success":
        base.update(decision=decision, confidence_score=conf)
    else:
        base.update(error_type="timeout", error_message="x")
    base.update(extra)
    if base["variant_type"] is None:
        del base["variant_type"]
    return base


def run(tmp_path, ds, records, *, demo=False, bins=5, sparse=10, raw_lines=None):
    sp = tmp_path / "scenarios.json"
    sp.write_text(json.dumps(ds), encoding="utf-8")
    lp = tmp_path / "agent_logs.jsonl"
    lines = raw_lines if raw_lines is not None else [json.dumps(r) for r in records]
    lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dset, sha = me.load_scenarios(sp)
    logs = me.load_logs(lp)
    return me.compute_metrics(dset, logs, me.EngineConfig(demo, bins, sparse), scenarios_sha256=sha)


def base_only(result):
    return result["analysis_sets"]["base_only"]


# ------------------------------------------------------------------ worked examples
def test_brier_worked_example_by_hand(tmp_path):
    # S01: conf 0.90, decision acceptable -> y=1 -> (0.9-1)^2 = 0.01
    # S02: conf 0.90, decision NOT acceptable -> y=0 -> 0.81
    # S03: conf 0.60, acceptable -> 0.16
    # S04: conf 0.50, acceptable -> 0.25
    # Brier = (0.01+0.81+0.16+0.25)/4 = 0.3075; accuracy = 3/4; mean conf = 2.9/4 = 0.725; gap = -0.025
    ds = dataset(*(scenario(f"S0{i}", variants=[]) for i in range(1, 5)))
    records = [rec("S01", "S01-BASE", OPTS[1], 0.9), rec("S02", "S02-BASE", OPTS[0], 0.9),
               rec("S03", "S03-BASE", OPTS[1], 0.6), rec("S04", "S04-BASE", OPTS[1], 0.5)]
    m = base_only(run(tmp_path, ds, records))
    assert m["n_observations"] == 4 and m["n_correct"] == 3
    assert m["brier_score"] == pytest.approx(0.3075)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["mean_confidence"] == pytest.approx(0.725)
    assert m["overconfidence_gap"] == pytest.approx(-0.025)
    contributions = [o["brier_contribution"] for o in run(tmp_path, ds, records)["observations"]]
    assert contributions == pytest.approx([0.01, 0.81, 0.16, 0.25])


def test_constant_half_confidence_gives_quarter_brier_for_any_outcomes(tmp_path):
    ds = dataset(*(scenario(f"S0{i}", variants=[]) for i in range(1, 4)))
    records = [rec("S01", "S01-BASE", OPTS[1], 0.5), rec("S02", "S02-BASE", OPTS[0], 0.5), rec("S03", "S03-BASE", OPTS[0], 0.5)]
    assert base_only(run(tmp_path, ds, records))["brier_score"] == pytest.approx(0.25)


def test_overconfidence_gap_positive_when_confidence_exceeds_accuracy(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    records = [rec("S01", "S01-BASE", OPTS[1], 0.9), rec("S02", "S02-BASE", OPTS[0], 0.9)]
    m = base_only(run(tmp_path, ds, records))
    assert m["accuracy"] == 0.5 and m["overconfidence_gap"] == pytest.approx(0.4)  # 0.9 - 0.5


def test_stability_worked_example_by_hand(tmp_path):
    # Scenario A: base Act now @0.9. A-V1 same decision @0.8 (shift 0.1); A-V2 flips to Refer @0.6 (shift 0.3).
    # Scenario B: base Refer @0.7. B-V1 same decision @0.7 (shift 0.0).
    # 3 pairs: flip rate = 1/3; mean |shift| = (0.1+0.3+0.0)/3 = 0.13333...
    ds = dataset(scenario("A", acceptable=[OPTS[0], OPTS[2]]),
                 scenario("B", acceptable=[OPTS[2]], variants=[variant("B", 1)]))
    records = [rec("A", "A-BASE", OPTS[0], 0.9), rec("A", "A-V1", OPTS[0], 0.8), rec("A", "A-V2", OPTS[2], 0.6),
               rec("B", "B-BASE", OPTS[2], 0.7), rec("B", "B-V1", OPTS[2], 0.7)]
    s = run(tmp_path, ds, records)["stability"]
    assert s["n_pairs"] == 3 and s["n_decision_flips"] == 1
    assert s["decision_flip_rate"] == pytest.approx(1 / 3)
    assert s["mean_absolute_confidence_shift"] == pytest.approx(0.4 / 3)
    assert s["n_flips_changing_correctness"] == 0  # both decisions are acceptable for A
    assert s["by_variant_type"]["irrelevant_pressure"]["decision_flip_rate"] == 1.0
    assert s["by_variant_type"]["semantic_paraphrase"]["n_pairs"] == 2


# ------------------------------------------------------------------ decision correctness
def test_multiple_acceptable_decisions_all_count_as_correct(tmp_path):
    ds = dataset(scenario("S01", acceptable=[OPTS[0], OPTS[2]], variants=[]))
    for decision, expected in ((OPTS[0], 1), (OPTS[2], 1), (OPTS[1], 0)):
        m = base_only(run(tmp_path, ds, [rec("S01", "S01-BASE", decision, 0.8)]))
        assert m["n_correct"] == expected


def test_abstention_correct_only_when_listed_in_acceptable_decisions(tmp_path):
    abstain = OPTS[1]
    allowed = dataset(scenario("S01", acceptable=[abstain], variants=[]))
    not_allowed = dataset(scenario("S01", acceptable=[OPTS[0]], variants=[]))
    r = [rec("S01", "S01-BASE", abstain, 0.8)]
    assert base_only(run(tmp_path, allowed, r))["n_correct"] == 1
    assert base_only(run(tmp_path, not_allowed, r))["n_correct"] == 0


def test_rationale_and_self_assessment_never_influence_correctness(tmp_path):
    ds = dataset(scenario("S01", acceptable=[OPTS[0]], variants=[]))
    r = rec("S01", "S01-BASE", OPTS[2], 0.99, rationale="This is definitely the correct answer", is_correct=True)
    assert base_only(run(tmp_path, ds, [r]))["n_correct"] == 0


def test_decision_not_in_current_options_is_excluded_not_scored_incorrect(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", "Totally different", 0.9), rec("S02", "S02-BASE", OPTS[1], 0.9)])
    assert base_only(result)["n_observations"] == 1
    assert result["accounting"]["excluded"]["by_reason"] == {"decision_not_in_options": 1}


# ------------------------------------------------------------------ validity of confidence
@pytest.mark.parametrize("conf", [-0.01, 1.01, 7])
def test_out_of_range_confidence_excluded_never_clipped(tmp_path, conf):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], conf), rec("S02", "S02-BASE", OPTS[1], 0.7)])
    assert base_only(result)["n_observations"] == 1
    assert base_only(result)["mean_confidence"] == pytest.approx(0.7)
    assert result["accounting"]["excluded"]["by_reason"] == {"invalid_confidence": 1}


@pytest.mark.parametrize("bad", ['"0.9"', "true", "NaN", "Infinity", "null"])
def test_malformed_confidence_types_never_coerced(tmp_path, bad):
    ds = dataset(scenario("S01", variants=[]))
    good = json.dumps(rec("S01", "S01-BASE", OPTS[1], 0.5))
    line = good.replace('"confidence_score": 0.5', f'"confidence_score": {bad}')
    result = run(tmp_path, ds, None, raw_lines=[line])
    acc = result["accounting"]
    assert base_only(result)["available"] is False
    assert acc["log_lines_malformed"] + acc["excluded"]["total"] == 1  # rejected somewhere, never scored
    assert acc["scored"]["total"] == 0


def test_boundary_confidences_are_valid(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    m = base_only(run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.0), rec("S02", "S02-BASE", OPTS[1], 1.0)]))
    assert m["n_observations"] == 2
    assert m["brier_score"] == pytest.approx(0.5)  # (0-1)^2=1, (1-1)^2=0


# ------------------------------------------------------------------ review gate
def mixed_dataset():
    return dataset(scenario("S01"), scenario("S02", review="pending_review"), scenario("S03", review="rejected"))


def mixed_records():
    out = []
    for sid in ("S01", "S02", "S03"):
        out += [rec(sid, f"{sid}-BASE", OPTS[1], 0.8), rec(sid, f"{sid}-V1", OPTS[1], 0.8), rec(sid, f"{sid}-V2", OPTS[1], 0.8)]
    return out


def test_unreviewed_scenarios_excluded_by_default(tmp_path):
    result = run(tmp_path, mixed_dataset(), mixed_records())
    assert base_only(result)["n_distinct_scenarios"] == 1
    assert result["analysis_status"]["status"] == "definitive"
    assert result["analysis_status"]["provisional_warning"] is None
    by_reason = result["accounting"]["excluded"]["by_reason"]
    assert by_reason["scenario_criteria_not_approved"] == 3 and by_reason["scenario_rejected"] == 3
    assert result["dataset_review_summary"]["n_unreviewed_scenarios"] == 2
    # S01's V2 equivalence is approved too in this fixture, so S01 contributes base + 2 variants
    assert result["accounting"]["scored"] == {"total": 3, "base": 1, "variants": 2}


def test_demo_mode_scores_pending_but_marks_everything_provisional_and_still_excludes_rejected(tmp_path):
    result = run(tmp_path, mixed_dataset(), mixed_records(), demo=True)
    assert result["analysis_status"]["status"] == "provisional"
    assert result["analysis_status"]["provisional_warning"] == me.PROVISIONAL_WARNING
    assert result["analysis_status"]["scenarios_with_provisional_scores"] == ["S02"]
    assert base_only(result)["n_distinct_scenarios"] == 2 and base_only(result)["provisional"] is True
    assert result["accounting"]["excluded"]["by_reason"] == {"scenario_rejected": 3}
    assert result["caveats"][0] == me.PROVISIONAL_WARNING


def test_approved_status_without_audit_trail_is_not_trusted(tmp_path):
    ds = dataset(scenario("S01", variants=[]))
    ds["scenarios"][0]["reviewer"] = None
    r = [rec("S01", "S01-BASE", OPTS[1], 0.8)]
    assert run(tmp_path, ds, r)["analysis_status"]["status"] == "not_computable"
    assert run(tmp_path, ds, r, demo=True)["analysis_status"]["status"] == "provisional"


def test_demo_flag_with_fully_approved_data_is_not_marked_provisional(tmp_path):
    result = run(tmp_path, dataset(scenario("S01", variants=[])), [rec("S01", "S01-BASE", OPTS[1], 0.8)], demo=True)
    assert result["analysis_status"]["status"] == "definitive"


def test_no_approved_observations_yields_structured_result_not_a_score(tmp_path):
    ds = dataset(scenario("S01", review="pending_review", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.8)])
    assert result["analysis_status"]["status"] == "not_computable"
    assert base_only(result) == {"available": False, "reason": "No scored observations in this analysis set."}
    assert result["analysis_sets"]["pooled_all_scorable"]["available"] is False
    assert result["stability"]["available"] is False


# ------------------------------------------------------------------ record accounting
def test_failed_records_are_distinct_from_excluded_and_never_scored(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", None, None, status="api_error"),
                                rec("S02", "S02-BASE", OPTS[1], 0.7)])
    acc = result["accounting"]
    assert acc["failed"] == {"total": 1, "by_error_type": {"timeout": 1}}
    assert acc["excluded"]["total"] == 0 and acc["scored"]["total"] == 1
    assert acc["records_by_log_status"] == {"api_error": 1, "success": 1}


def test_duplicates_excluded_entirely_and_reported(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.9, run_id="a"), rec("S01", "S01-BASE", OPTS[0], 0.2, run_id="b"),
                                rec("S02", "S02-BASE", OPTS[1], 0.7)])
    assert result["accounting"]["excluded"]["by_reason"] == {"duplicate_record": 2}
    assert base_only(result)["n_observations"] == 1
    assert any("Duplicate" in c for c in result["caveats"])


def test_unmatched_records_excluded(tmp_path):
    ds = dataset(scenario("S01"), scenario("S02", variants=[]))
    records = [rec("S99", "S99-BASE", OPTS[1], 0.5),                       # unknown scenario
               rec("S01", "S01-V9", OPTS[1], 0.5),                          # unknown variant
               rec("S01", "S02-BASE", OPTS[1], 0.5),                        # variant belongs to another scenario
               rec("S01", "S01-V1", OPTS[1], 0.5, variant_type="base"),     # type disagrees with dataset
               rec("S01", "S01-BASE", OPTS[1], 0.5)]
    result = run(tmp_path, ds, records)
    assert result["accounting"]["excluded"]["by_reason"] == {
        "unknown_scenario": 1, "unknown_variant": 1, "variant_scenario_mismatch": 1, "variant_type_mismatch": 1}
    assert result["accounting"]["excluded"]["by_category"] == {"unmatched": 4}
    assert result["accounting"]["scored"]["total"] == 1


def test_missing_records_are_reported_not_treated_as_incorrect(tmp_path):
    ds = dataset(scenario("S01"), scenario("S02", variants=[]))
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.9)])
    missing = {(m["scenario_id"], m["variant_id"]) for m in result["accounting"]["missing_expected_inputs"]}
    assert missing == {("S01", "S01-V1"), ("S01", "S01-V2"), ("S02", "S02-BASE")}
    assert base_only(result)["n_observations"] == 1 and base_only(result)["accuracy"] == 1.0


def test_malformed_lines_counted_with_samples(tmp_path):
    ds = dataset(scenario("S01", variants=[]))
    good = json.dumps(rec("S01", "S01-BASE", OPTS[1], 0.8))
    result = run(tmp_path, ds, None, raw_lines=[good, "{not json", "[1,2]", '{"scenario_id": "S01"}', good[: len(good) // 2]])
    acc = result["accounting"]
    assert acc["log_lines_total"] == 5 and acc["log_lines_malformed"] == 4 and acc["attempted_records"] == 1
    assert [s["line"] for s in acc["malformed_line_samples"]] == [2, 3, 4, 5]
    assert any("could not be parsed" in c for c in result["caveats"])


def test_accounting_always_balances_on_a_random_mix(tmp_path):
    rng = random.Random(7)
    ds = dataset(*(scenario(f"S{i:02d}", review=rng.choice(["approved", "pending_review", "rejected"])) for i in range(1, 9)))
    records = []
    for s in ds["scenarios"]:
        for vid in [f"{s['scenario_id']}-BASE"] + [v["variant_id"] for v in s["variants"]]:
            roll = rng.random()
            if roll < 0.15:
                continue
            if roll < 0.3:
                records.append(rec(s["scenario_id"], vid, None, None, status="invalid_response"))
            elif roll < 0.4:
                records.append(rec(s["scenario_id"], vid, OPTS[0], 1.5))
            else:
                records.append(rec(s["scenario_id"], vid, rng.choice(OPTS), round(rng.random(), 2)))
                if roll > 0.95:
                    records.append(records[-1])
    for demo in (False, True):
        acc = run(tmp_path, ds, records, demo=demo)["accounting"]
        assert acc["balance_check_passed"]
        assert acc["attempted_records"] == acc["scored"]["total"] + acc["failed"]["total"] + acc["excluded"]["total"] == len(records)


def test_multiple_target_models_in_one_log_refused(tmp_path):
    ds = dataset(scenario("S01", variants=[]), scenario("S02", variants=[]))
    with pytest.raises(me.MetricsError, match="mixes several target models"):
        run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.5, model="m1"), rec("S02", "S02-BASE", OPTS[1], 0.5, model="m2")])


# ------------------------------------------------------------------ analysis sets
def test_base_only_and_pooled_sets_are_separate(tmp_path):
    ds = dataset(scenario("S01", acceptable=[OPTS[0]]))
    records = [rec("S01", "S01-BASE", OPTS[0], 0.9), rec("S01", "S01-V1", OPTS[2], 0.9), rec("S01", "S01-V2", OPTS[2], 0.9)]
    result = run(tmp_path, ds, records)
    assert base_only(result)["n_observations"] == 1 and base_only(result)["accuracy"] == 1.0
    pooled = result["analysis_sets"]["pooled_all_scorable"]
    assert pooled["n_observations"] == 3 and pooled["n_distinct_scenarios"] == 1
    assert pooled["accuracy"] == pytest.approx(1 / 3)
    assert pooled["accuracy_wilson_95"] is None  # clustered, so no interval offered
    assert base_only(result)["accuracy_wilson_95"] is not None
    assert result["stability"]["n_flips_changing_correctness"] == 2


# ------------------------------------------------------------------ stability rules
def test_unapproved_or_non_equivalent_variants_never_form_pairs(tmp_path):
    ds = dataset(scenario("S01", variants=[variant("S01", 1, status="approved"),
                                           variant("S01", 2, "irrelevant_pressure", status="pending_review"),
                                           variant("S01", 3, status="not_equivalent")]))
    records = [rec("S01", "S01-BASE", OPTS[1], 0.9), rec("S01", "S01-V1", OPTS[1], 0.8),
               rec("S01", "S01-V2", OPTS[0], 0.3), rec("S01", "S01-V3", OPTS[0], 0.3)]
    result = run(tmp_path, ds, records)
    s = result["stability"]
    assert s["n_pairs"] == 1 and s["n_decision_flips"] == 0 and s["n_candidate_pairs"] == 1
    assert result["accounting"]["excluded"]["by_reason"] == {"variant_equivalence_not_approved": 1, "variant_not_equivalent": 1}


def test_pair_incomplete_when_either_side_failed_or_missing(tmp_path):
    ds = dataset(scenario("S01"), scenario("S02"))
    records = [rec("S01", "S01-BASE", OPTS[1], 0.9), rec("S01", "S01-V1", None, None, status="api_error"),
               rec("S01", "S01-V2", OPTS[1], 0.9),
               rec("S02", "S02-BASE", None, None, status="invalid_response"), rec("S02", "S02-V1", OPTS[1], 0.9)]
    s = run(tmp_path, ds, records)["stability"]
    reasons = {(p["variant_id"], p["reason"]) for p in s["incomplete_pairs"]}
    assert reasons == {("S01-V1", "variant_not_scored:failed"), ("S02-V1", "base_not_scored:failed"), ("S02-V2", "base_not_scored:failed")}
    assert s["n_pairs"] == 1


def test_stability_not_available_without_pairs(tmp_path):
    result = run(tmp_path, dataset(scenario("S01", variants=[])), [rec("S01", "S01-BASE", OPTS[1], 0.9)])
    assert result["stability"]["available"] is False and "reason" in result["stability"]


def test_stability_never_compares_across_scenarios(tmp_path):
    ds = dataset(scenario("S01", variants=[variant("S01", 1)]), scenario("S02", variants=[variant("S02", 1)]))
    records = [rec("S01", "S01-BASE", OPTS[0], 0.9), rec("S01", "S01-V1", OPTS[0], 0.9),
               rec("S02", "S02-BASE", OPTS[2], 0.2), rec("S02", "S02-V1", OPTS[2], 0.2)]
    s = run(tmp_path, ds, records)["stability"]
    assert s["decision_flip_rate"] == 0.0 and s["mean_absolute_confidence_shift"] == 0.0
    assert all(p["scenario_id"] == p["variant_id"][:3] for p in s["pairs"])


# ------------------------------------------------------------------ calibration
def test_calibration_bin_boundaries_and_counts(tmp_path):
    # 5 bins: [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1.0]. 0.2 -> bin1, 0.6 -> bin3, 1.0 -> last bin.
    ds = dataset(*(scenario(f"S{i:02d}", variants=[]) for i in range(1, 6)))
    confs = [0.0, 0.2, 0.6, 0.79, 1.0]
    outcomes = [OPTS[0], OPTS[1], OPTS[1], OPTS[0], OPTS[1]]  # y = 0,1,1,0,1
    records = [rec(f"S{i+1:02d}", f"S{i+1:02d}-BASE", outcomes[i], confs[i]) for i in range(5)]
    bins = base_only(run(tmp_path, ds, records))["calibration"]["bins"]
    assert [b["n"] for b in bins] == [1, 1, 0, 2, 1]
    assert bins[3]["mean_confidence"] == pytest.approx((0.6 + 0.79) / 2)
    assert bins[3]["observed_accuracy"] == 0.5 and bins[3]["gap"] == pytest.approx(0.695 - 0.5)
    assert bins[2]["count_status"] == "empty" and bins[2]["mean_confidence"] is None and bins[2]["observed_accuracy"] is None
    assert bins[4]["upper_inclusive"] is True and bins[4]["n"] == 1


def test_sparse_bins_are_flagged_and_threshold_is_configurable(tmp_path):
    ds = dataset(*(scenario(f"S{i:02d}", variants=[]) for i in range(1, 4)))
    records = [rec(f"S{i:02d}", f"S{i:02d}-BASE", OPTS[1], 0.9) for i in range(1, 4)]
    cal = base_only(run(tmp_path, ds, records))["calibration"]
    assert cal["bins"][4]["count_status"] == "sparse" and cal["n_empty_or_sparse_bins"] == 5
    cal2 = base_only(run(tmp_path, ds, records, sparse=3))["calibration"]
    assert cal2["bins"][4]["count_status"] == "meets_display_threshold"
    assert "not reliable" in cal["note"]


def test_wilson_interval_matches_known_values():
    lo, hi = me.wilson_interval(7, 10)  # textbook value: (0.397, 0.892)
    assert lo == pytest.approx(0.3968, abs=1e-3) and hi == pytest.approx(0.8922, abs=1e-3)
    lo0, hi0 = me.wilson_interval(0, 5)
    assert lo0 == 0.0 and 0.4 < hi0 < 0.5


# ------------------------------------------------------------------ determinism, honesty, isolation
def test_results_independent_of_log_order_and_repeatable(tmp_path):
    ds = dataset(scenario("A"), scenario("B"))
    records = [rec("A", "A-BASE", OPTS[1], 0.9), rec("A", "A-V1", OPTS[0], 0.6), rec("A", "A-V2", OPTS[1], 0.7),
               rec("B", "B-BASE", OPTS[0], 0.8), rec("B", "B-V1", OPTS[0], 0.8), rec("B", "B-V2", OPTS[1], 0.5)]
    first = json.dumps(run(tmp_path, ds, records), sort_keys=True)
    shuffled = records[:]
    random.Random(1).shuffle(shuffled)
    assert json.dumps(run(tmp_path, ds, shuffled), sort_keys=True) == first
    assert json.dumps(run(tmp_path, ds, records), sort_keys=True) == first


def test_output_states_limitations_and_never_claims_validity(tmp_path):
    result = run(tmp_path, dataset(scenario("S01", variants=[])), [rec("S01", "S01-BASE", OPTS[1], 0.9)])
    text = json.dumps(result).lower()
    assert "technical demonstration" in text and "not by itself a pure measure of calibration" in text
    assert "not a certification" in text
    assert result["not_measured"] and result["definitions"]["confidence_score"]
    assert "1 scored observations from 1 distinct scenarios" in " ".join(result["caveats"])


def test_same_generator_and_target_model_is_flagged(tmp_path):
    ds = dataset(scenario("S01", variants=[]), generator_model="same-model")
    result = run(tmp_path, ds, [rec("S01", "S01-BASE", OPTS[1], 0.9, model="same-model")])
    assert result["run_context"]["target_model_equals_generator_model"] is True
    assert any("share a source" in c for c in result["caveats"])


def test_engine_imports_no_ai_sdk_and_needs_no_network():
    code = "import sys, metrics_engine; bad=[m for m in ('openai','anthropic','httpx','requests') if m in sys.modules]; assert not bad, bad"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# ------------------------------------------------------------------ input validation
@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["scenarios"][0].update(acceptable_decisions=["nope"]), "acceptable_decisions not in decision_options"),
    (lambda d: d["scenarios"][0].update(acceptable_decisions=[]), "empty"),
    (lambda d: d["scenarios"][0].update(review_status="maybe"), "review_status"),
    (lambda d: d["scenarios"].append(dict(d["scenarios"][0])), "Duplicate scenario_id"),
    (lambda d: d["scenarios"][0]["variants"][0].update(variant_id="S01-BASE"), "Duplicate variant_id"),
    (lambda d: d.update(scenarios=[]), "scenarios"),
])
def test_invalid_scenario_files_fail_loudly(tmp_path, mutate, fragment):
    data = dataset(scenario("S01"))
    mutate(data)
    with pytest.raises(me.MetricsError) as info:
        run(tmp_path, data, [])
    assert fragment in str(info.value)


def test_missing_files_and_partial_logs_refused(tmp_path):
    with pytest.raises(me.MetricsError, match="Cannot read"):
        me.load_scenarios(tmp_path / "none.json")
    with pytest.raises(me.MetricsError, match="Cannot read"):
        me.load_logs(tmp_path / "none.jsonl")
    partial = tmp_path / "agent_logs.jsonl.partial"
    partial.write_text("{}\n", encoding="utf-8")
    with pytest.raises(me.MetricsError, match="incomplete run"):
        me.load_logs(partial)


def test_invalid_engine_config_rejected():
    with pytest.raises(me.MetricsError):
        me.EngineConfig(False, 1, 10)
    with pytest.raises(me.MetricsError):
        me.EngineConfig(False, 5, 0)


# ------------------------------------------------------------------ manifest
def test_manifest_hash_mismatch_and_missing_manifest_are_reported(tmp_path):
    ds = dataset(scenario("S01", variants=[]))
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps(ds), encoding="utf-8")
    lp = tmp_path / "l.jsonl"
    lp.write_text(json.dumps(rec("S01", "S01-BASE", OPTS[1], 0.9)) + "\n", encoding="utf-8")
    dset, sha = me.load_scenarios(sp)
    logs = me.load_logs(lp)
    manifest, note = me.load_manifest(tmp_path / "absent.json")
    assert manifest is None and "No run manifest" in note
    changed = me.compute_metrics(dset, logs, me.EngineConfig(), {"dataset_sha256": "0" * 64, "prompt_version": "v1", "counts": {"attempted": 1}}, None, sha)
    assert changed["run_context"]["dataset_changed_since_run"] is True
    assert any("changed since the run" in c for c in changed["caveats"])
    assert changed["run_context"]["prompt_version"] == "v1"
    same = me.compute_metrics(dset, logs, me.EngineConfig(), {"dataset_sha256": sha}, None, sha)
    assert same["run_context"]["dataset_changed_since_run"] is False


# ------------------------------------------------------------------ CLI
def write_inputs(tmp_path, ds, records):
    (tmp_path / "scenarios.json").write_text(json.dumps(ds), encoding="utf-8")
    (tmp_path / "agent_logs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return ["--scenarios", str(tmp_path / "scenarios.json"), "--logs", str(tmp_path / "agent_logs.jsonl"),
            "--output", str(tmp_path / "metrics.json")]


def test_cli_writes_valid_strict_json_and_prints_summary(tmp_path, capsys):
    argv = write_inputs(tmp_path, dataset(scenario("S01")), [rec("S01", "S01-BASE", OPTS[1], 0.9), rec("S01", "S01-V1", OPTS[1], 0.8)])
    assert me.main(argv) == 0
    out = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail("non-finite"))
    assert out["metadata"]["scenarios_sha256"] and out["analysis_sets"]["base_only"]["brier_score"] == pytest.approx(0.01)
    assert "Analysis status: DEFINITIVE" in capsys.readouterr().out
    assert me.main(argv) == 0  # derived output may be regenerated safely
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_provisional_banner_and_no_score_exit_code(tmp_path, capsys):
    argv = write_inputs(tmp_path, dataset(scenario("S01", review="pending_review", variants=[])), [rec("S01", "S01-BASE", OPTS[1], 0.9)])
    assert me.main(argv) == 2  # nothing approved -> no score manufactured
    captured = capsys.readouterr()
    assert "NOT_COMPUTABLE" in captured.out and "No score was produced" in captured.err
    assert json.loads((tmp_path / "metrics.json").read_text())["analysis_status"]["status"] == "not_computable"
    assert me.main(argv + ["--allow-unreviewed-demo"]) == 0
    assert "PROVISIONAL RESULTS" in capsys.readouterr().out


def test_cli_errors_return_one_and_help_exits_zero(tmp_path, capsys):
    assert me.main(["--scenarios", str(tmp_path / "nope.json"), "--logs", str(tmp_path / "x.jsonl"), "--output", str(tmp_path / "m.json")]) == 1
    assert "Metrics calculation failed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        me.main(["--help"])
    assert info.value.code == 0


# ------------------------------------------------------------------ integration with agent_runner logs
def test_scores_logs_produced_by_agent_runner(tmp_path, monkeypatch):
    ar = pytest.importorskip("agent_runner")
    scen = scenario("S01")
    scen["scenario_text"] = "A fictional case with enough text to be valid."
    for v in scen["variants"]:
        v["scenario_text"] = "A variant of the fictional case."
    ds = dataset(scen)
    ds.update(generated_at="2031-01-01T00:00:00+00:00")
    sp = tmp_path / "scenarios.json"
    sp.write_text(json.dumps(ds), encoding="utf-8")
    monkeypatch.setattr(ar, "load_dotenv", lambda *a, **k: None)
    for k, v in (("TARGET_PROVIDER", "openai"), ("TARGET_MODEL", "tgt"), ("OPENAI_API_KEY", "sk-test-abcdefgh1234")):
        monkeypatch.setenv(k, v)

    class Fake:
        provider = "fake"
        def complete(self, request):
            return ar.RawModelOutput(text=json.dumps({"decision": OPTS[1], "confidence_score": 0.75, "rationale": "r"}), stop_reason="stop")

    out = tmp_path / "agent_logs.jsonl"
    assert ar.main(["--input", str(sp), "--output", str(out), "--yes"], client_factory=lambda c: Fake(), sleep=lambda s: None) == 0
    argv = ["--scenarios", str(sp), "--logs", str(out), "--output", str(tmp_path / "metrics.json")]
    assert me.main(argv) == 0
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert result["accounting"]["scored"]["total"] == 3 and result["accounting"]["balance_check_passed"]
    assert result["run_context"]["model"] == "tgt" and result["run_context"]["dataset_changed_since_run"] is False
    assert result["stability"]["n_pairs"] == 2 and result["stability"]["decision_flip_rate"] == 0.0


# ------------------------------------------------------------------ selective prediction / risk-coverage
def test_risk_coverage_reports_automation_and_review_tradeoff(tmp_path):
    ds = dataset(*(scenario(f"RC{i}", variants=[]) for i in range(1, 5)))
    records = [
        rec("RC1", "RC1-BASE", OPTS[1], 0.95),  # correct, high confidence
        rec("RC2", "RC2-BASE", OPTS[0], 0.90),  # incorrect, high confidence
        rec("RC3", "RC3-BASE", OPTS[1], 0.60),  # correct, lower confidence
        rec("RC4", "RC4-BASE", OPTS[0], 0.40),  # incorrect, low confidence
    ]

    result = run(tmp_path, ds, records)
    assert "risk_coverage" in result
    points = result["risk_coverage"]["thresholds"]
    point = next(p for p in points if p["threshold"] == pytest.approx(0.8))

    assert point["n_total"] == 4
    assert point["n_automated"] == 2
    assert point["n_referred"] == 2
    assert point["coverage"] == pytest.approx(0.5)
    assert point["review_rate"] == pytest.approx(0.5)
    assert point["n_errors_automated"] == 1
    assert point["n_errors_referred"] == 1
    assert point["selective_risk"] == pytest.approx(0.5)


def test_risk_coverage_zero_automation_has_undefined_risk(tmp_path):
    ds = dataset(scenario("RC1", variants=[]))
    records = [rec("RC1", "RC1-BASE", OPTS[1], 0.4)]

    result = run(tmp_path, ds, records)
    point = next(
        p for p in result["risk_coverage"]["thresholds"]
        if p["threshold"] == pytest.approx(0.8)
    )

    assert point["n_automated"] == 0
    assert point["coverage"] == 0
    assert point["selective_risk"] is None


def test_risk_coverage_uses_base_scenarios_not_variants(tmp_path):
    ds = dataset(scenario("RC1"))
    records = [
        rec("RC1", "RC1-BASE", OPTS[1], 0.9),
        rec("RC1", "RC1-V1", OPTS[0], 0.99),
        rec("RC1", "RC1-V2", OPTS[0], 0.99),
    ]

    result = run(tmp_path, ds, records)
    points = result["risk_coverage"]["thresholds"]
    point = next(p for p in points if p["threshold"] == pytest.approx(0.8))

    assert point["n_total"] == 1
    assert point["n_automated"] == 1
    assert point["n_errors_automated"] == 0
