
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import DOCUMENT_TO_ROUTING_OUTCOME, ROUTING_OUTCOMES  # noqa: E402

CACHE_DIR = ROOT / "outputs" / "ocr_cache"
OUTPUT_DIR = ROOT / "outputs" / "baseline"

CACHE_FILES = {
    "train": CACHE_DIR / "train_raw.jsonl",
    "validation": CACHE_DIR / "validation_raw.jsonl",
}


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path.name}:{line_number}"
                ) from exc

    return records


def analyse_ocr_quality():
    """Summarise extraction status and empty text by split and class."""
    rows = []

    for expected_split, path in CACHE_FILES.items():
        records = load_jsonl(path)

        for record in records:
            if record.get("split") != expected_split:
                raise ValueError(
                    f"Unexpected split in {path.name}: "
                    f"{record.get('split')}"
                )

            if record.get("split") == "test":
                raise ValueError("Test data must remain reserved.")

        frame = pd.DataFrame(records)

        required = {"split", "original_class", "status"}
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {sorted(missing)}"
            )

        frame["status"] = frame["status"].fillna("missing_status")
        frame["text"] = frame.get(
            "text", pd.Series("", index=frame.index)
        ).fillna("").astype(str)

        frame["is_success"] = frame["status"].eq("ok")
        frame["is_empty"] = frame["text"].str.strip().eq("")

        for (split, doc_class), group in frame.groupby(
            ["split", "original_class"], dropna=False
        ):
            total = len(group)
            successful = int(group["is_success"].sum())
            failed = total - successful
            empty = int(group["is_empty"].sum())
            successful_empty = int(
                (group["is_success"] & group["is_empty"]).sum()
            )

            rows.append({
                "split": split,
                "document_class": doc_class,
                "total_records": total,
                "successful_extractions": successful,
                "failed_extractions": failed,
                "empty_text_records": empty,
                "successful_empty_texts": successful_empty,
                "nonempty_successful_texts": int(
                    (group["is_success"] & ~group["is_empty"]).sum()
                ),
                "failure_rate": failed / total if total else 0.0,
                "successful_empty_text_rate": (
                    successful_empty / successful if successful else 0.0
                ),
            })

    quality = pd.DataFrame(rows).sort_values(
        ["split", "document_class"]
    )

    quality.to_csv(
        OUTPUT_DIR / "ocr_quality_by_split_class.csv",
        index=False,
    )

    return quality


def analyse_routing():
    """Evaluate routing outcomes using the existing validation predictions."""
    prediction_file = OUTPUT_DIR / "validation_predictions.csv"

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Validation predictions not found: {prediction_file}"
        )

    results = pd.read_csv(prediction_file)

    if not results["split"].eq("validation").all():
        raise ValueError("Predictions must contain validation records only.")

    if results["source_index"].duplicated().any():
        raise ValueError("Duplicate validation source indices detected.")

    required_classes = (
        set(results["true_class"].astype(str))
        | set(results["predicted_class"].astype(str))
    )
    unknown_classes = required_classes - set(
        DOCUMENT_TO_ROUTING_OUTCOME
    )

    if unknown_classes:
        raise ValueError(
            f"Classes missing from routing configuration: "
            f"{sorted(unknown_classes)}"
        )

    results["true_routing_outcome"] = (
        results["true_class"].map(DOCUMENT_TO_ROUTING_OUTCOME)
    )
    results["predicted_routing_outcome"] = (
        results["predicted_class"].map(DOCUMENT_TO_ROUTING_OUTCOME)
    )

    # Verify the saved ground-truth routing labels against project config.
    expected_routes = results["true_class"].map(
        DOCUMENT_TO_ROUTING_OUTCOME
    )
    saved_routes = results["routing_outcome"].astype(str)

    if not expected_routes.astype(str).equals(saved_routes):
        raise ValueError(
            "Saved routing outcomes disagree with src/config.py."
        )

    results["classification_correct"] = (
        results["true_class"] == results["predicted_class"]
    )
    results["routing_correct"] = (
        results["true_routing_outcome"]
        == results["predicted_routing_outcome"]
    )

    # Every document-class error is retained for traceable error analysis.
    errors = results.loc[
        ~results["classification_correct"]
    ].copy()

    errors.to_csv(
        OUTPUT_DIR / "validation_error_analysis.csv",
        index=False,
    )

    # Classification errors that nevertheless preserve the correct route.
    preserved = results.loc[
        ~results["classification_correct"] & results["routing_correct"]
    ].copy()

    preserved.to_csv(
        OUTPUT_DIR / "route_preserved_classification_errors.csv",
        index=False,
    )

    # Cases where the predicted destination is actually incorrect.
    routing_errors = results.loc[~results["routing_correct"]].copy()

    routing_errors.to_csv(
        OUTPUT_DIR / "validation_routing_errors.csv",
        index=False,
    )

    cm = confusion_matrix(
        results["true_routing_outcome"],
        results["predicted_routing_outcome"],
        labels=ROUTING_OUTCOMES,
    )

    pd.DataFrame(
        cm,
        index=ROUTING_OUTCOMES,
        columns=ROUTING_OUTCOMES,
    ).to_csv(
        OUTPUT_DIR / "routing_confusion_matrix.csv",
        index=True,
        index_label="true_routing_outcome",
    )

    report = classification_report(
        results["true_routing_outcome"],
        results["predicted_routing_outcome"],
        labels=ROUTING_OUTCOMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "routing_classification_report.csv",
        index=True,
        index_label="routing_outcome",
    )

    metrics = {
        "validation_records": int(len(results)),
        "document_class_accuracy": float(
            accuracy_score(
                results["true_class"], results["predicted_class"]
            )
        ),
        "document_class_macro_f1": float(
            f1_score(
                results["true_class"],
                results["predicted_class"],
                labels=sorted(DOCUMENT_TO_ROUTING_OUTCOME),
                average="macro",
                zero_division=0,
            )
        ),
        "routing_accuracy": float(
            accuracy_score(
                results["true_routing_outcome"],
                results["predicted_routing_outcome"],
            )
        ),
        "routing_macro_f1": float(
            f1_score(
                results["true_routing_outcome"],
                results["predicted_routing_outcome"],
                labels=ROUTING_OUTCOMES,
                average="macro",
                zero_division=0,
            )
        ),
        "routing_weighted_f1": float(
            f1_score(
                results["true_routing_outcome"],
                results["predicted_routing_outcome"],
                labels=ROUTING_OUTCOMES,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_errors": int(
            (~results["classification_correct"]).sum()
        ),
        "routing_errors": int((~results["routing_correct"]).sum()),
        "classification_errors_with_correct_route": int(len(preserved)),
        "test_split_used": False,
    }

    with (OUTPUT_DIR / "routing_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Analysing OCR quality in train and validation caches...")
    quality = analyse_ocr_quality()

    print("\nOCR quality summary by split:")
    print(
        quality.groupby("split")[
            [
                "total_records",
                "successful_extractions",
                "failed_extractions",
                "successful_empty_texts",
            ]
        ].sum().to_string()
    )

    print("\nAnalysing document classification and routing...")
    metrics = analyse_routing()

    print("\nValidation routing results:")
    print(f"Document-class accuracy: {metrics['document_class_accuracy']:.4f}")
    print(f"Routing accuracy:        {metrics['routing_accuracy']:.4f}")
    print(f"Routing macro F1:        {metrics['routing_macro_f1']:.4f}")
    print(f"Classification errors:   {metrics['classification_errors']}")
    print(f"Routing errors:           {metrics['routing_errors']}")
    print(
        "Class errors with correct route: "
        f"{metrics['classification_errors_with_correct_route']}"
    )
    print("\nTest split used: no")
    print(f"Analysis outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
