"""Turn a probability file into the predictions file the evaluation expects.

Input : CSV with source_index + prob_<class> columns (e.g. Parisha's file).
Output: CSV with source_index, true_class, predicted_class, confidence + the prob_ columns.

Usage (from project root):
    python -m src.evaluation.probs_to_predictions <probs.csv> <validation|test> <out.csv>
"""
import sys
import pandas as pd

SPLIT_CSV = "data/metadata/dataset_split.csv"


def main(probs_path, split_name, out_path):
    probs = pd.read_csv(probs_path)
    split = pd.read_csv(SPLIT_CSV)

    prob_cols = [c for c in probs.columns if c.startswith("prob_")]
    classes = [c[len("prob_"):] for c in prob_cols]
    assert len(classes) == 9, f"Expected 9 prob_ columns, found {len(classes)}"

    # Safety: the file must contain exactly the documents of the chosen split.
    expected = split[split["split"] == split_name]
    if set(probs["source_index"]) != set(expected["source_index"]) or \
       len(probs) != len(expected):
        sys.exit(f"source_index mismatch: file has {len(probs)} rows, "
                 f"{split_name} split has {len(expected)}.")

    # Each row: the class with the highest probability is the prediction.
    best = probs[prob_cols].values.argmax(axis=1)
    out = probs.copy()
    out["predicted_class"] = [classes[i] for i in best]
    out["confidence"] = probs[prob_cols].values.max(axis=1)

    # Attach the true label from the frozen split file.
    labels = expected.set_index("source_index")["original_class"]
    out["true_class"] = out["source_index"].map(labels)

    cols = ["source_index", "true_class", "predicted_class", "confidence"] + prob_cols
    out[cols].to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] not in ("validation", "test"):
        sys.exit("Usage: python -m src.evaluation.probs_to_predictions "
                 "<probs.csv> <validation|test> <out.csv>")
    main(*sys.argv[1:])