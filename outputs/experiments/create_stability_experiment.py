import json
from pathlib import Path

out = Path("outputs/experiments")
source = out / "scenarios.json"
destination = out / "stability_scenarios.json"

data = json.loads(source.read_text())

for scenario in data["scenarios"]:
    sid = scenario["scenario_id"]

    scenario["variants"] = [
        {
            "variant_id": f"{sid}-REFRAMED",
            "variant_type": "rephrased",
            "equivalence_review_status": "pending_review",
            "reviewer": None,
            "reviewed_at": None,
        }
    ]

data["dataset_version"] = "0.2-stability-demo"
destination.write_text(json.dumps(data, indent=2) + "\n")

print(f"Created {destination}")
print(f"Scenarios: {len(data['scenarios'])}")
print(f"Variants: {sum(len(s['variants']) for s in data['scenarios'])}")
print("Review status: pending; no variants have been approved.")
print("API calls: 0")
