import json
from pathlib import Path

out = Path("outputs/experiments")
out.mkdir(parents=True, exist_ok=True)

correctness = [True, False, True, True]

confidence_profiles = {
    "synthetic-agent-A": [0.90, 0.20, 0.80, 0.70],
    "synthetic-agent-B": [0.70, 0.70, 0.70, 0.70],
}

correct_decision = "Request additional information"
other_decision = "Act now"

scenarios = []
for i in range(1, 5):
    scenarios.append({
        "scenario_id": f"EXP1-{i:02d}",
        "decision_options": [correct_decision, other_decision],
        "acceptable_decisions": [correct_decision],
        "review_status": "pending_review",
        "reviewer": None,
        "reviewed_at": None,
        "variants": [],
    })

(out / "scenarios.json").write_text(
    json.dumps({
        "dataset_version": "0.1-experiment",
        "scenarios": scenarios,
    }, indent=2) + "\n"
)

for agent, profile in confidence_profiles.items():
    logs = []

    for i, is_correct in enumerate(correctness, start=1):
        scenario_id = f"EXP1-{i:02d}"

        logs.append({
            "run_id": f"offline-{agent}-001",
            "scenario_id": scenario_id,
            "variant_id": f"{scenario_id}-BASE",
            "variant_type": "base",
            "provider": "synthetic",
            "model": agent,
            "timestamp": "2031-01-02T00:00:00+00:00",
            "status": "success",
            "decision": correct_decision if is_correct else other_decision,
            "confidence_score": profile[i - 1],
        })

    filename = f"{agent}.jsonl"
    (out / filename).write_text(
        "".join(json.dumps(record) + "\n" for record in logs)
    )
    print(f"Created {filename}: {len(logs)} records")

print("Created shared scenarios and separate logs for both fictional agents.")
print("API calls: 0. All scenarios remain pending human review.")
