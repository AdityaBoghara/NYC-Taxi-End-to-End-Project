"""Audit historical score provenance without modifying historical artifacts."""
import json
import pickle
import re

import numpy as np
import pandas as pd

from src.config import CFG, ROOT, TARGET, processed_path
from src.data_pipeline import sha256
from src.pretrip import metrics, split_by_time


def audit() -> dict:
    path = processed_path(2023, 1)
    model_path = ROOT / CFG["api"]["model_path"]
    metadata_path = ROOT / CFG["api"]["metadata_path"]
    meta = json.loads(metadata_path.read_text())
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    rows = pd.read_parquet(path)
    masks, boundaries = split_by_time(rows, CFG["training"]["val_days"])
    notebook = json.loads((ROOT / "notebooks" / "3. baseline_model_training.ipynb").read_text())
    output = "\n".join("".join(item.get("text", [])) for cell in notebook["cells"] for item in cell.get("outputs", []))
    match = re.search(r"Shape\s*:\s*\((\d+),", output)
    old_rows = int(match.group(1)) if match else None
    scores = {}
    for split in ["val", "test"]:
        subset = rows.loc[masks[split]]
        predicted = model.predict(subset[meta["features"]])
        if meta.get("log_transform"):
            predicted = np.expm1(predicted)
        scores[split] = metrics(subset[TARGET], predicted)
    return {
        "status": "historical_scores_not_reproducible_do_not_use_for_claims",
        "feature_order_matches_booster": model.get_booster().feature_names == meta["features"],
        "metadata_log_transform": meta.get("log_transform"),
        "model_class": type(model).__name__,
        "old_notebook_row_count": old_rows, "current_row_count": len(rows),
        "cohort_changed": old_rows != len(rows),
        "current_split": boundaries,
        "stored_cutoff_dates": {key: meta[key] for key in ["val_cutoff_date", "test_cutoff_date"]},
        "recorded": {split: meta[f"{split}_metrics"] for split in ["val", "test"]},
        "recomputed": scores,
        "artifact_sha256": sha256(model_path), "metadata_sha256": sha256(metadata_path),
        "current_processed_sha256": sha256(path),
        "conclusions": [
            "A changed row count proves the current cohort differs from the saved notebook run.",
            "Stored feature ordering matches the booster, and the recorded raw-target transform is respected.",
            "The original dataset hash and geographic encoding snapshot were not saved, so exact historical provenance cannot be reconstructed from these artifacts.",
            "The row-count change alone does not establish the entire cause of the MAE difference. Quarantine old scores rather than guessing a correction.",
        ],
    }


if __name__ == "__main__":
    report = audit()
    destination = ROOT / "reports" / "legacy_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
