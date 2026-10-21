"""Deterministic metrics engine for Epistemic.ai v0.1 (Step 4).

Reads the scenario dataset (with its human-reviewed answer key) and the
agent logs, then reports performance, confidence, calibration and stability
metrics. This module never calls an AI API and imports no AI SDK. Given the
same input files and configuration it produces the same numbers.

Scoring rules (all deterministic, no LLM judge):

* correct_outcome = 1 if the logged decision is in the scenario's
  acceptable_decisions, else 0. Abstention counts as correct only if it is
  listed there. The model's rationale and confidence never influence it.
* brier_contribution = (confidence_score - correct_outcome) ** 2; the Brier
  Score is the mean over scored observations.
* Only observations with human-approved criteria (status "approved" with
  reviewer and reviewed_at) are scored. --allow-unreviewed-demo additionally
  scores pending items, and every result is then marked provisional.

Worked example (also asserted in tests/test_metrics_engine.py):

    confidence 0.90, correct   -> (0.90 - 1)^2 = 0.01
    confidence 0.90, incorrect -> (0.90 - 0)^2 = 0.81
    confidence 0.60, correct   -> 0.16;  confidence 0.50, correct -> 0.25
    Brier = (0.01 + 0.81 + 0.16 + 0.25) / 4 = 0.3075
    accuracy 3/4 = 0.75, mean confidence 0.725, gap = 0.725 - 0.75 = -0.025

Nothing here establishes that a model is reliable or safe. Ten scenarios are
a technical demonstration, not a statistically representative sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, Strict, ValidationError, model_validator

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_SCENARIOS_PATH = OUTPUT_DIR / "scenarios.json"
DEFAULT_LOGS_PATH = OUTPUT_DIR / "agent_logs.jsonl"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "metrics.json"

ENGINE_VERSION = "0.1"
SCHEMA_VERSION = "0.1"
BASE_VARIANT_SUFFIX = "BASE"  # contract with agent_runner.py: "<scenario_id>-BASE"
BASE_VARIANT_TYPE = "base"
DEFAULT_BINS = 5
DEFAULT_SPARSE_BIN_THRESHOLD = 10
MAX_MALFORMED_SAMPLES = 20
WILSON_Z_95 = 1.959963984540054

PROVISIONAL_WARNING = (
    "PROVISIONAL RESULTS — SCORING CRITERIA HAVE NOT BEEN FULLY REVIEWED. "
    "THESE RESULTS MUST NOT BE INTERPRETED AS VALIDATED EVIDENCE OF AI RELIABILITY."
)
EXPERIMENTAL_NOTICE = (
    "Experimental evaluation output. Not a validated psychometric instrument, "
    "not a certification, and not evidence of fitness for real medical, "
    "financial, or other high-stakes use."
)
DEFINITIONS = {
    "confidence_score": (
        "The agent's self-reported probability that its selected decision satisfies "
        "the scenario's predefined evaluation criteria. It is not a measurement of "
        "internal model state and not a probability of a real-world event."
    ),
    "correct_outcome": "1 if the decision is in the approved acceptable_decisions set, else 0.",
    "brier_score": "Mean of (confidence_score - correct_outcome)^2. Lower is better on the observations included.",
    "overconfidence_gap": "Mean reported confidence minus observed decision accuracy.",
    "decision_flip_rate": "Share of base-vs-variant pairs whose selected decision differs.",
    "mean_absolute_confidence_shift": "Mean |confidence(base) - confidence(variant)| over the same pairs.",
}
NOT_MEASURED = [
    "Epistemic humility as a construct. These metrics do not measure whether abstention or "
    "information requests were appropriate; the dataset has no explicit abstention marker, so "
    "abstention usage is not analysed.",
    "Statistical significance of any difference between models or conditions.",
    "Behavior outside the specific scenarios and variants tested.",
]


class MetricsError(Exception):
    """Raised when inputs are unusable. The CLI prints it and exits non-zero."""


# --------------------------------------------------------------------------
# Input schemas
# --------------------------------------------------------------------------
class MetricsVariant(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    variant_id: str = Field(min_length=1)
    variant_type: str = Field(min_length=1)
    equivalence_review_status: Literal["pending_review", "approved", "not_equivalent", "uncertain"]
    reviewer: str | None = None
    reviewed_at: str | None = None


class MetricsScenario(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    scenario_id: str = Field(min_length=1)
    decision_options: list[str]
    acceptable_decisions: list[str]
    review_status: Literal["pending_review", "approved", "rejected"]
    reviewer: str | None = None
    reviewed_at: str | None = None
    variants: list[MetricsVariant] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answer_key_consistent(self) -> "MetricsScenario":
        if len(self.decision_options) < 2 or len(set(self.decision_options)) != len(self.decision_options):
            raise ValueError(f"{self.scenario_id}: decision_options must hold at least 2 unique options")
        if not self.acceptable_decisions:
            raise ValueError(f"{self.scenario_id}: acceptable_decisions is empty")
        unknown = [d for d in self.acceptable_decisions if d not in self.decision_options]
        if unknown:
            raise ValueError(f"{self.scenario_id}: acceptable_decisions not in decision_options")
        return self


class MetricsDataset(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    dataset_version: str | None = None
    generator_model: str | None = None
    scenarios: list[MetricsScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "MetricsDataset":
        scenario_ids: set[str] = set()
        variant_ids: set[str] = set()
        for scenario in self.scenarios:
            if scenario.scenario_id in scenario_ids:
                raise ValueError(f"Duplicate scenario_id: {scenario.scenario_id}")
            scenario_ids.add(scenario.scenario_id)
            for vid in [base_variant_id(scenario.scenario_id)] + [v.variant_id for v in scenario.variants]:
                if vid in variant_ids:
                    raise ValueError(f"Duplicate variant_id: {vid}")
                variant_ids.add(vid)
        return self


class LogRecord(BaseModel):
    """Log line as written by agent_runner.py. Types are strict: no coercion."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    run_id: str
    scenario_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    variant_type: str | None = None
    provider: str
    model: str
    timestamp: str | None = None
    status: Literal["success", "api_error", "invalid_response"]
    decision: Annotated[str, Strict()] | None = None
    confidence_score: Annotated[float, Strict(), Field(allow_inf_nan=False)] | None = None
    error_type: str | None = None


def base_variant_id(scenario_id: str) -> str:
    return f"{scenario_id}-{BASE_VARIANT_SUFFIX}"


@dataclass(frozen=True)
class EngineConfig:
    allow_unreviewed_demo: bool = False
    n_bins: int = DEFAULT_BINS
    sparse_bin_threshold: int = DEFAULT_SPARSE_BIN_THRESHOLD

    def __post_init__(self) -> None:
        if not 2 <= self.n_bins <= 20:
            raise MetricsError("n_bins must be between 2 and 20.")
        if self.sparse_bin_threshold < 1:
            raise MetricsError("sparse_bin_threshold must be at least 1.")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _format_validation_error(exc: ValidationError, limit: int = 5) -> str:
    items = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return "; ".join(items[:limit]) + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name!r}")


def load_scenarios(path: Path) -> tuple[MetricsDataset, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MetricsError(f"Cannot read scenario file {path}: {exc}") from exc
    try:
        raw = json.loads(data)
        dataset = MetricsDataset.model_validate(raw)
    except ValueError as exc:  # JSONDecodeError and pydantic ValidationError are ValueErrors
        detail = _format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
        raise MetricsError(f"Invalid scenario dataset {path}: {detail}") from exc
    return dataset, _sha256(data)


@dataclass
class LogLoad:
    records: list[LogRecord]
    total_lines: int
    malformed: list[dict[str, Any]]
    n_malformed: int
    sha256: str


def load_logs(path: Path) -> LogLoad:
    if path.name.endswith(".partial"):
        raise MetricsError(
            f"{path.name} is an incomplete run (agent_runner.py renames the file only on completion). "
            "Refusing to score it."
        )
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MetricsError(f"Cannot read log file {path}: {exc}") from exc

    records: list[LogRecord] = []
    malformed: list[dict[str, Any]] = []
    n_malformed = 0
    total = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        reason: str | None = None
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(payload, dict):
                reason = "line is not a JSON object"
            else:
                records.append(LogRecord.model_validate(payload))
        except ValidationError as exc:
            reason = _format_validation_error(exc, limit=3)
        except ValueError as exc:
            reason = f"invalid JSON: {exc}"
        if reason is not None:
            n_malformed += 1
            if len(malformed) < MAX_MALFORMED_SAMPLES:
                malformed.append({"line": lineno, "reason": reason})
    return LogLoad(records, total, malformed, n_malformed, _sha256(data))


def load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Optional run manifest. Returns (manifest, note_if_unusable)."""
    if not path.exists():
        return None, f"No run manifest found at {path.name}; run context is partial."
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("not an object")
        return manifest, None
    except (OSError, ValueError) as exc:
        return None, f"Run manifest {path.name} is unreadable ({exc}); ignored."


# --------------------------------------------------------------------------
# Review gate (same rules as agent_runner.py)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    provisional: bool
    reason: str | None = None


def _has_audit_trail(reviewer: str | None, reviewed_at: str | None) -> bool:
    return bool((reviewer or "").strip()) and bool((reviewed_at or "").strip())


def scenario_eligibility(scenario: MetricsScenario, allow_demo: bool) -> Eligibility:
    if scenario.review_status == "rejected":
        return Eligibility(False, False, "scenario_rejected")  # never scored, even in demo
    if scenario.review_status == "approved" and _has_audit_trail(scenario.reviewer, scenario.reviewed_at):
        return Eligibility(True, False)
    if allow_demo:
        return Eligibility(True, True)
    reason = (
        "scenario_criteria_not_approved"
        if scenario.review_status != "approved"
        else "scenario_approved_without_audit_trail"
    )
    return Eligibility(False, False, reason)


def variant_eligibility(scenario: MetricsScenario, variant: MetricsVariant, allow_demo: bool) -> Eligibility:
    scenario_gate = scenario_eligibility(scenario, allow_demo)
    if not scenario_gate.eligible:
        return scenario_gate
    if variant.equivalence_review_status == "not_equivalent":
        # The dataset carries no separate criteria for such a variant, so it cannot be scored.
        return Eligibility(False, False, "variant_not_equivalent")
    if variant.equivalence_review_status == "approved" and _has_audit_trail(variant.reviewer, variant.reviewed_at):
        return Eligibility(True, scenario_gate.provisional)
    if allow_demo:
        return Eligibility(True, True)
    reason = (
        "variant_equivalence_not_approved"
        if variant.equivalence_review_status != "approved"
        else "variant_approved_without_audit_trail"
    )
    return Eligibility(False, False, reason)


REASON_CATEGORY = {
    "duplicate_record": "duplicate",
    "unknown_scenario": "unmatched",
    "unknown_variant": "unmatched",
    "variant_scenario_mismatch": "unmatched",
    "variant_type_mismatch": "unmatched",
    "missing_decision": "malformed",
    "invalid_confidence": "malformed",
    "decision_not_in_options": "malformed",
    "scenario_rejected": "unapproved",
    "scenario_criteria_not_approved": "unapproved",
    "scenario_approved_without_audit_trail": "unapproved",
    "variant_not_equivalent": "unapproved",
    "variant_equivalence_not_approved": "unapproved",
    "variant_approved_without_audit_trail": "unapproved",
}


# --------------------------------------------------------------------------
# Classification of every log record into exactly one disposition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scored:
    scenario_id: str
    variant_id: str
    variant_type: str
    decision: str
    confidence: float
    correct: int
    brier: float
    provisional: bool


def classify_records(
    dataset: MetricsDataset, records: list[LogRecord], allow_demo: bool
) -> tuple[list[dict[str, Any]], list[Scored]]:
    """Return (audit entries, scored observations). Each record gets one disposition:
    'scored', 'failed' (api_error / invalid_response), or 'excluded' with a reason."""
    scenarios = {s.scenario_id: s for s in dataset.scenarios}
    lookup: dict[str, tuple[MetricsScenario, MetricsVariant | None]] = {}
    for s in dataset.scenarios:
        lookup[base_variant_id(s.scenario_id)] = (s, None)
        for v in s.variants:
            lookup[v.variant_id] = (s, v)

    key_counts = Counter((r.scenario_id, r.variant_id) for r in records)
    ordered = sorted(records, key=lambda r: (r.scenario_id, r.variant_id, r.run_id, r.timestamp or ""))
    audit: list[dict[str, Any]] = []
    scored: list[Scored] = []

    for rec in ordered:
        entry: dict[str, Any] = {
            "scenario_id": rec.scenario_id,
            "variant_id": rec.variant_id,
            "log_status": rec.status,
            "disposition": None,
            "reason": None,
            "decision": rec.decision if rec.status == "success" else None,
            "confidence_score": rec.confidence_score if rec.status == "success" else None,
            "correct_outcome": None,
            "brier_contribution": None,
            "provisional": False,
        }
        audit.append(entry)

        def exclude(reason: str) -> None:
            entry["disposition"] = "excluded"
            entry["reason"] = reason

        if key_counts[(rec.scenario_id, rec.variant_id)] > 1:
            exclude("duplicate_record")
            continue
        scenario = scenarios.get(rec.scenario_id)
        if scenario is None:
            exclude("unknown_scenario")
            continue
        if rec.variant_id not in lookup:
            exclude("unknown_variant")
            continue
        owner, variant = lookup[rec.variant_id]
        if owner.scenario_id != rec.scenario_id:
            exclude("variant_scenario_mismatch")
            continue
        expected_type = BASE_VARIANT_TYPE if variant is None else variant.variant_type
        if rec.variant_type is not None and rec.variant_type != expected_type:
            exclude("variant_type_mismatch")
            continue
        if rec.status != "success":
            entry["disposition"] = "failed"
            entry["reason"] = rec.error_type or rec.status
            continue
        if rec.decision is None:
            exclude("missing_decision")
            continue
        conf = rec.confidence_score
        if conf is None or not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
            exclude("invalid_confidence")
            continue
        if rec.decision not in scenario.decision_options:
            exclude("decision_not_in_options")  # never counted as incorrect
            continue
        gate = variant_eligibility(scenario, variant, allow_demo) if variant else scenario_eligibility(scenario, allow_demo)
        if not gate.eligible:
            exclude(gate.reason or "scenario_criteria_not_approved")
            continue

        correct = 1 if rec.decision in scenario.acceptable_decisions else 0
        brier = (conf - correct) ** 2
        entry.update(disposition="scored", correct_outcome=correct, brier_contribution=brier, provisional=gate.provisional)
        scored.append(
            Scored(rec.scenario_id, rec.variant_id, expected_type, rec.decision, float(conf), correct, brier, gate.provisional)
        )
    return audit, scored


# --------------------------------------------------------------------------
# Metric calculations
# --------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def wilson_interval(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def calibration_summary(obs: list[Scored], n_bins: int, sparse_threshold: int) -> dict[str, Any]:
    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[Scored]] = [[] for _ in range(n_bins)]
    for o in obs:
        buckets[min(bisect_right(edges, o.confidence) - 1, n_bins - 1)].append(o)
    bins = []
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        entry: dict[str, Any] = {
            "bin_index": i,
            "lower": edges[i],
            "upper": edges[i + 1],
            "upper_inclusive": i == n_bins - 1,
            "n": n,
            "n_correct": sum(o.correct for o in bucket),
            "mean_confidence": None,
            "observed_accuracy": None,
            "gap": None,
            "count_status": "empty" if n == 0 else ("sparse" if n < sparse_threshold else "meets_display_threshold"),
        }
        if n:
            entry["mean_confidence"] = _mean([o.confidence for o in bucket])
            entry["observed_accuracy"] = entry["n_correct"] / n
            entry["gap"] = entry["mean_confidence"] - entry["observed_accuracy"]
        bins.append(entry)
    n_sparse = sum(b["count_status"] in ("empty", "sparse") for b in bins)
    return {
        "n_bins": n_bins,
        "bin_scheme": "equal-width; lower bound inclusive; only the last bin includes 1.0",
        "sparse_bin_threshold": sparse_threshold,
        "n_empty_or_sparse_bins": n_sparse,
        "bins": bins,
        "note": (
            "Empty and sparse bins are not reliable evidence about calibration. The count threshold is "
            "a display heuristic only; meeting it does not make a bin's estimate statistically reliable."
        ),
    }


def compute_core_metrics(obs: list[Scored], config: EngineConfig, with_intervals: bool) -> dict[str, Any]:
    if not obs:
        return {"available": False, "reason": "No scored observations in this analysis set."}
    obs = sorted(obs, key=lambda o: (o.scenario_id, o.variant_id))
    n = len(obs)
    n_correct = sum(o.correct for o in obs)
    accuracy = n_correct / n
    mean_conf = _mean([o.confidence for o in obs])
    brier = _mean([o.brier for o in obs])
    result: dict[str, Any] = {
        "available": True,
        "provisional": any(o.provisional for o in obs),
        "n_observations": n,
        "n_distinct_scenarios": len({o.scenario_id for o in obs}),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "mean_confidence": mean_conf,
        "brier_score": brier,
        "overconfidence_gap": mean_conf - accuracy,
        "brier_references": {
            "constant_confidence_0_5": 0.25,
            "constant_confidence_equal_to_observed_accuracy": accuracy * (1 - accuracy),
            "note": (
                "Reference values for context only. They are not thresholds or pass marks. A Brier Score "
                "depends on the outcome mix as well as calibration, and is not a pure calibration measure."
            ),
        },
        "calibration": calibration_summary(obs, config.n_bins, config.sparse_bin_threshold),
    }
    if with_intervals:
        lo, hi = wilson_interval(n_correct, n)
        contributions = [o.brier for o in obs]
        se = None
        if n >= 2:
            mu = _mean(contributions)
            se = math.sqrt(math.fsum((c - mu) ** 2 for c in contributions) / (n - 1)) / math.sqrt(n)
        result["accuracy_wilson_95"] = {"lower": lo, "upper": hi}
        result["brier_standard_error"] = se
        result["interval_note"] = (
            "Intervals describe sampling uncertainty if scenarios were independent draws from a larger "
            "population, which this prototype does not establish. Treat them as rough and wide at small n."
        )
    else:
        result["accuracy_wilson_95"] = None
        result["brier_standard_error"] = None
        result["interval_note"] = (
            "Not computed: these observations are clustered by scenario (a base and its variants), "
            "so they are not independent."
        )
    return result


def compute_stability(
    dataset: MetricsDataset,
    scored: list[Scored],
    audit: list[dict[str, Any]],
    allow_demo: bool,
) -> dict[str, Any]:
    by_key = {(o.scenario_id, o.variant_id): o for o in scored}
    disposition_by_key: dict[tuple[str, str], str] = {}
    for a in audit:
        disposition_by_key.setdefault((a["scenario_id"], a["variant_id"]), a["disposition"])

    pairs: list[dict[str, Any]] = []
    incomplete: list[dict[str, str]] = []
    candidates = 0
    for scenario in sorted(dataset.scenarios, key=lambda s: s.scenario_id):
        base_key = (scenario.scenario_id, base_variant_id(scenario.scenario_id))
        for variant in sorted(scenario.variants, key=lambda v: v.variant_id):
            if not variant_eligibility(scenario, variant, allow_demo).eligible:
                continue  # equivalence not established, so not a comparable pair
            candidates += 1
            base = by_key.get(base_key)
            var = by_key.get((scenario.scenario_id, variant.variant_id))
            if base is None or var is None:
                missing_side = (
                    f"base_not_scored:{disposition_by_key.get(base_key, 'missing')}"
                    if base is None
                    else f"variant_not_scored:{disposition_by_key.get((scenario.scenario_id, variant.variant_id), 'missing')}"
                )
                incomplete.append(
                    {"scenario_id": scenario.scenario_id, "variant_id": variant.variant_id, "reason": missing_side}
                )
                continue
            pairs.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "variant_id": variant.variant_id,
                    "variant_type": variant.variant_type,
                    "base_decision": base.decision,
                    "variant_decision": var.decision,
                    "decision_flipped": base.decision != var.decision,
                    "base_confidence": base.confidence,
                    "variant_confidence": var.confidence,
                    "abs_confidence_shift": abs(base.confidence - var.confidence),
                    "base_correct": base.correct,
                    "variant_correct": var.correct,
                    "provisional": base.provisional or var.provisional,
                }
            )

    note = (
        "Pairs compare a base scenario with variants whose equivalence was approved by a reviewer. "
        "Results describe only the variants tested, not general behavioral stability. Variants of one "
        "scenario share the same base observation, so pairs are not independent."
    )
    if not pairs:
        return {
            "available": False,
            "reason": "No complete, approved base-variant pairs." if candidates else "No variants with approved equivalence.",
            "n_candidate_pairs": candidates,
            "n_pairs": 0,
            "incomplete_pairs": incomplete,
            "pairs": [],
            "note": note,
        }

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        flips = sum(p["decision_flipped"] for p in subset)
        return {
            "n_pairs": len(subset),
            "n_decision_flips": flips,
            "decision_flip_rate": flips / len(subset),
            "mean_absolute_confidence_shift": _mean([p["abs_confidence_shift"] for p in subset]),
        }

    by_type = {
        vt: summarize([p for p in pairs if p["variant_type"] == vt])
        for vt in sorted({p["variant_type"] for p in pairs})
    }
    overall = summarize(pairs)
    return {
        "available": True,
        "provisional": any(p["provisional"] for p in pairs),
        "n_candidate_pairs": candidates,
        "n_pairs": overall["n_pairs"],
        "n_scenarios_with_pairs": len({p["scenario_id"] for p in pairs}),
        "decision_flip_rate": overall["decision_flip_rate"],
        "n_decision_flips": overall["n_decision_flips"],
        "mean_absolute_confidence_shift": overall["mean_absolute_confidence_shift"],
        "n_flips_changing_correctness": sum(p["base_correct"] != p["variant_correct"] for p in pairs),
        "by_variant_type": by_type,
        "incomplete_pairs": incomplete,
        "pairs": pairs,
        "note": note,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def compute_risk_coverage(
    observations: list[Scored],
    thresholds: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ),
) -> dict[str, Any]:
    """Describe the trade-off between autonomous coverage and selective risk.

    An observation is handled autonomously when confidence >= threshold.
    All other observations are referred for human review.

    This is a retrospective descriptive analysis, not a deployment guarantee.
    """
    n_total = len(observations)
    points: list[dict[str, Any]] = []

    for threshold in thresholds:
        automated = [
            o for o in observations
            if o.confidence >= threshold
        ]
        referred = [
            o for o in observations
            if o.confidence < threshold
        ]

        n_automated = len(automated)
        n_referred = len(referred)
        errors_automated = sum(1 - o.correct for o in automated)
        errors_referred = sum(1 - o.correct for o in referred)

        points.append({
            "threshold": threshold,
            "n_total": n_total,
            "n_automated": n_automated,
            "n_referred": n_referred,
            "coverage": n_automated / n_total if n_total else None,
            "review_rate": n_referred / n_total if n_total else None,
            "n_errors_automated": errors_automated,
            "n_errors_referred": errors_referred,
            "selective_risk": (
                errors_automated / n_automated
                if n_automated else None
            ),
        })

    return {
        "n_observations": n_total,
        "threshold_rule": "Automate when confidence >= threshold; otherwise refer for human review.",
        "thresholds": points,
        "interpretation": (
            "Descriptive retrospective simulation only. Results depend on the "
            "evaluated sample and do not establish deployment safety or guarantee "
            "that future cases will have the same error rate."
        ),
    }


def compute_metrics(
    dataset: MetricsDataset,
    logs: LogLoad,
    config: EngineConfig,
    manifest: dict[str, Any] | None = None,
    manifest_note: str | None = None,
    scenarios_sha256: str | None = None,
) -> dict[str, Any]:
    """Pure function: same inputs and config give the same result (no timestamps here)."""
    models = sorted({(r.provider, r.model) for r in logs.records})
    if len(models) > 1:
        raise MetricsError(
            "The log file mixes several target models: "
            + ", ".join(f"{p}/{m}" for p, m in models)
            + ". Score one model per log file."
        )
    run_ids = sorted({r.run_id for r in logs.records})
    audit, scored = classify_records(dataset, logs.records, config.allow_unreviewed_demo)

    disposition_counts = Counter(a["disposition"] for a in audit)
    excluded_by_reason = Counter(a["reason"] for a in audit if a["disposition"] == "excluded")
    excluded_by_category = Counter(REASON_CATEGORY[r] for r in excluded_by_reason.elements())
    failed_by_error = Counter(a["reason"] for a in audit if a["disposition"] == "failed")
    scored_base = [o for o in scored if o.variant_type == BASE_VARIANT_TYPE]

    # Inputs that should have been run and scored under the current review state.
    present = {(r.scenario_id, r.variant_id) for r in logs.records}
    missing: list[dict[str, str]] = []
    for s in sorted(dataset.scenarios, key=lambda s: s.scenario_id):
        if scenario_eligibility(s, config.allow_unreviewed_demo).eligible and (s.scenario_id, base_variant_id(s.scenario_id)) not in present:
            missing.append({"scenario_id": s.scenario_id, "variant_id": base_variant_id(s.scenario_id)})
        for v in sorted(s.variants, key=lambda v: v.variant_id):
            if variant_eligibility(s, v, config.allow_unreviewed_demo).eligible and (s.scenario_id, v.variant_id) not in present:
                missing.append({"scenario_id": s.scenario_id, "variant_id": v.variant_id})

    scenario_status = Counter(s.review_status for s in dataset.scenarios)
    approved_scenarios = sum(
        1 for s in dataset.scenarios if s.review_status == "approved" and _has_audit_trail(s.reviewer, s.reviewed_at)
    )
    all_variants = [v for s in dataset.scenarios for v in s.variants]
    provisional_scenarios = sorted({o.scenario_id for o in scored if o.provisional})

    if not scored:
        status, status_meaning = "not_computable", "No valid, approved, scorable observations remain; no metrics were produced."
    elif provisional_scenarios:
        status = "provisional"
        status_meaning = "Some scored observations rely on scoring criteria or variant equivalence that no human has approved."
    else:
        status = "definitive"
        status_meaning = (
            "All scored observations rely on human-approved criteria. This does not make the evaluation "
            "a validated study."
        )

    n_scored_scenarios = len({o.scenario_id for o in scored})
    dataset_changed = None
    if manifest and manifest.get("dataset_sha256") and scenarios_sha256:
        dataset_changed = manifest["dataset_sha256"] != scenarios_sha256

    caveats = [
        EXPERIMENTAL_NOTICE,
        f"Sample size: {len(scored)} scored observations from {n_scored_scenarios} distinct scenarios. "
        "This is a technical demonstration, not a statistically representative sample.",
        "The Brier Score measures overall probabilistic prediction quality. It is not by itself a pure "
        "measure of calibration or of epistemic humility, and no pass/fail threshold is implied.",
        "Pooled analysis sets include variants of the same scenario and are not independent observations; "
        "the base-scenario-only set is the primary analysis.",
    ]
    if status == "provisional":
        caveats.insert(0, PROVISIONAL_WARNING)
    if config.allow_unreviewed_demo and status == "definitive":
        caveats.append("Demo mode was enabled, but every scored observation had approved criteria.")
    if missing:
        caveats.append(f"{len(missing)} expected input(s) have no log record; coverage is incomplete.")
    if excluded_by_category.get("duplicate"):
        caveats.append("Duplicate records were excluded entirely because the intended record cannot be determined.")
    if logs.n_malformed:
        caveats.append(f"{logs.n_malformed} log line(s) could not be parsed and were not used.")
    if len(run_ids) > 1:
        caveats.append(f"Log file contains {len(run_ids)} run_ids; verify the records belong to one evaluation.")
    if dataset_changed:
        caveats.append(
            "The scenario file changed since the run (for example review edits). Scoring uses the current "
            "answer key; the model never saw it."
        )
    if manifest_note:
        caveats.append(manifest_note)
    if models and dataset.generator_model and dataset.generator_model == models[0][1]:
        caveats.append("The scenario generator model and the evaluated model are the same, so the draft answer key and the agent share a source.")
    if manifest and manifest.get("counts", {}).get("attempted") is not None:
        if manifest["counts"]["attempted"] != len(logs.records) + logs.n_malformed:
            caveats.append("Manifest attempted-count differs from the number of log lines; the log may be altered or incomplete.")

    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "notice": EXPERIMENTAL_NOTICE,
        "analysis_status": {
            "status": status,
            "provisional": status == "provisional",
            "meaning": status_meaning,
            "provisional_warning": PROVISIONAL_WARNING if status == "provisional" else None,
            "scenarios_with_provisional_scores": provisional_scenarios,
        },
        "configuration": {
            "allow_unreviewed_demo": config.allow_unreviewed_demo,
            "n_bins": config.n_bins,
            "sparse_bin_threshold": config.sparse_bin_threshold,
            "primary_analysis_set": "base_only",
        },
        "run_context": {
            "provider": models[0][0] if models else None,
            "model": models[0][1] if models else None,
            "run_ids": run_ids,
            "generator_model": dataset.generator_model,
            "target_model_equals_generator_model": bool(models and dataset.generator_model == models[0][1]),
            "prompt_version": (manifest or {}).get("prompt_version"),
            "manifest_dataset_sha256": (manifest or {}).get("dataset_sha256"),
            "dataset_changed_since_run": dataset_changed,
            "log_marked_run_provisional": (manifest or {}).get("provisional"),
        },
        "dataset_review_summary": {
            "n_scenarios": len(dataset.scenarios),
            "scenario_review_status_counts": dict(sorted(scenario_status.items())),
            "n_scenarios_approved_with_audit_trail": approved_scenarios,
            "n_unreviewed_scenarios": len(dataset.scenarios) - approved_scenarios,
            "n_variants": len(all_variants),
            "variant_equivalence_status_counts": dict(sorted(Counter(v.equivalence_review_status for v in all_variants).items())),
        },
        "accounting": {
            "definitions": {
                "attempted_records": "Log records that parsed and validated structurally (one per attempted input).",
                "scored": "Valid, matched, approved (or demo-permitted) observations used in metrics.",
                "failed": "api_error or invalid_response records: no decision or confidence exists.",
                "excluded": "Records that could not or must not be scored, by reason.",
            },
            "log_lines_total": logs.total_lines,
            "log_lines_malformed": logs.n_malformed,
            "malformed_line_samples": logs.malformed,
            "attempted_records": len(logs.records),
            "records_by_log_status": dict(sorted(Counter(r.status for r in logs.records).items())),
            "scored": {
                "total": len(scored),
                "base": len(scored_base),
                "variants": len(scored) - len(scored_base),
            },
            "failed": {"total": disposition_counts.get("failed", 0), "by_error_type": dict(sorted(failed_by_error.items()))},
            "excluded": {
                "total": disposition_counts.get("excluded", 0),
                "by_category": dict(sorted(excluded_by_category.items())),
                "by_reason": dict(sorted(excluded_by_reason.items())),
            },
            "missing_expected_inputs": missing,
            "balance_check_passed": len(logs.records)
            == disposition_counts.get("scored", 0) + disposition_counts.get("failed", 0) + disposition_counts.get("excluded", 0),
        },
        "analysis_sets": {
            "base_only": compute_core_metrics(scored_base, config, with_intervals=True),
            "pooled_all_scorable": compute_core_metrics(scored, config, with_intervals=False),
        },
        "risk_coverage": compute_risk_coverage(scored_base),
        "stability": compute_stability(dataset, scored, audit, config.allow_unreviewed_demo),
        "observations": audit,
        "definitions": DEFINITIONS,
        "not_measured": NOT_MEASURED,
        "caveats": caveats,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic metrics from the scenario dataset and agent logs. "
            "Makes no API calls. Unreviewed scenarios are excluded unless --allow-unreviewed-demo is given."
        )
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS_PATH)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Run manifest (default: next to the logs, <stem>.manifest.json).")
    parser.add_argument("--output", type=Path, default=DEFAULT_METRICS_PATH,
                        help="Metrics JSON (default: outputs/metrics.json). Derived data: overwritten atomically.")
    parser.add_argument("--allow-unreviewed-demo", action="store_true",
                        help="Also score pending (unreviewed) items. All results are marked PROVISIONAL.")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Calibration bins (default 5).")
    parser.add_argument("--sparse-bin-threshold", type=int, default=DEFAULT_SPARSE_BIN_THRESHOLD,
                        help="Bins with fewer observations are flagged sparse (default 10).")
    return parser.parse_args(argv)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_summary(result: dict[str, Any], output: Path) -> None:
    status = result["analysis_status"]
    acc = result["accounting"]
    primary = result["analysis_sets"]["base_only"]
    print(f"Analysis status: {status['status'].upper()} - {status['meaning']}")
    if status["provisional"]:
        print(f"\n*** {PROVISIONAL_WARNING} ***\n")
    print(
        f"Records: attempted={acc['attempted_records']} scored={acc['scored']['total']} "
        f"failed={acc['failed']['total']} excluded={acc['excluded']['total']} "
        f"missing_inputs={len(acc['missing_expected_inputs'])} malformed_lines={acc['log_lines_malformed']}"
    )
    if acc["excluded"]["by_reason"]:
        print(f"Exclusions by reason: {acc['excluded']['by_reason']}")
    if primary["available"]:
        print(
            f"Base scenarios (n={primary['n_observations']}): accuracy={_fmt(primary['accuracy'])} "
            f"mean_confidence={_fmt(primary['mean_confidence'])} brier={_fmt(primary['brier_score'])} "
            f"overconfidence_gap={_fmt(primary['overconfidence_gap'])}"
        )
    stab = result["stability"]
    if stab["available"]:
        print(
            f"Stability ({stab['n_pairs']} pairs): flip_rate={_fmt(stab['decision_flip_rate'])} "
            f"mean_abs_confidence_shift={_fmt(stab['mean_absolute_confidence_shift'])}"
        )
    else:
        print(f"Stability: not available ({stab['reason']})")
    print("Caveat: small samples are a technical demonstration only; see caveats in the output file.")
    print(f"Metrics written to: {output}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = EngineConfig(args.allow_unreviewed_demo, args.bins, args.sparse_bin_threshold)
        dataset, scenarios_sha = load_scenarios(args.scenarios)
        logs = load_logs(args.logs)
        manifest_path = args.manifest or args.logs.with_name(args.logs.stem + ".manifest.json")
        manifest, manifest_note = load_manifest(manifest_path)
        result = compute_metrics(dataset, logs, config, manifest, manifest_note, scenarios_sha)
        result["metadata"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenarios_path": str(args.scenarios),
            "scenarios_sha256": scenarios_sha,
            "logs_path": str(args.logs),
            "logs_sha256": logs.sha256,
        }
        write_json_atomic(args.output, result)
    except MetricsError as exc:
        print(f"Metrics calculation failed: {exc}", file=sys.stderr)
        return 1
    print_summary(result, args.output)
    if result["analysis_status"]["status"] == "not_computable":
        print("No score was produced: no valid, approved observations. See accounting in the output file.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
