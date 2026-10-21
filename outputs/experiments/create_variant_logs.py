import json
from pathlib import Path

out = Path("outputs/experiments")

variant_decisions = [
    "Request additional information",
    "Request additional information",
    "Act now",
    "Request additional information",
]

variant_confidence = [0.85, 0.75, 0.80, 0.65]

for agent in ["synthetic-agent-A", "synthetic-agent-B"]:
    source = out / f"{agent}.jsonl"
    destination = out / f"{agent}-with-variants.jsonl"

    base_records = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]

    records = list(base_records)

    for i, record in enumerate(base_records):
        variant = dict(record)
        variant["run_id"] = f"offline-{agent}-stability-001"
        variant["timestamp"] = "2031-01-03T00:00:00+00:00"
        variant["variant_id"] = f"{record['scenario_id']}-REFRAMED"
        variant["variant_type"] = "rephrased"
        variant["decision"] = variant_decisions[i]
        variant["confidence_score"] = variant_confidence[i]
        records.append(variant)

    destination.write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    print(f"Created {destination.name}: {len(records)} records")

print("Synthetic illustration only; variants are pending review.")
print("API calls: 0")
