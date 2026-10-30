import json
import platform
import time
import uuid
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "outputs" / "ocr_cache"
OUTPUT_DIR = ROOT / "outputs" / "baseline"

TRAIN_FILE = CACHE_DIR / "train_raw.jsonl"
VALIDATION_FILE = CACHE_DIR / "validation_raw.jsonl"
SEED = 42


def load_cache(path):
    """Load successful OCR records while retaining empty OCR text."""
    if not path.exists():
        raise FileNotFoundError(f"OCR cache not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("status") != "ok":
                raise ValueError(
                    f"OCR record failed at {path.name}:{line_number}"
                )

            if record.get("split") not in {"train", "validation"}:
                raise ValueError(
                    f"Unexpected split in {path.name}:{line_number}"
                )

            if not isinstance(record.get("text"), str):
                raise ValueError(
                    f"Missing OCR text at {path.name}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return pd.DataFrame(records)


def main():
    started = time.time()
    run_id = uuid.uuid4().hex[:12]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_cache(TRAIN_FILE)
    validation = load_cache(VALIDATION_FILE)

    # Enforce the intended train/validation separation.
    if not (train["split"] == "train").all():
        raise ValueError("Training cache contains non-training records.")

    if not (validation["split"] == "validation").all():
        raise ValueError("Validation cache contains non-validation records.")

    # Prevent overlap between training and validation source records.
    train_indices = set(train["source_index"])
    validation_indices = set(validation["source_index"])

    overlap = train_indices.intersection(validation_indices)

    if overlap:
        raise ValueError(
            f"Train/validation source-index overlap detected: "
            f"{len(overlap)} records"
        )

    # Use OCR text as the only model input and original_class as the target.
    X_train = train["text"].fillna("").astype(str)
    y_train = train["original_class"].astype(str)

    X_val = validation["text"].fillna("").astype(str)
    y_val = validation["original_class"].astype(str)

    train_classes = sorted(y_train.unique())
    unknown_classes = sorted(set(y_val) - set(train_classes))

    if unknown_classes:
        raise ValueError(
            f"Validation contains unseen classes: {unknown_classes}"
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=SEED,
            ),
        ),
    ])

    print(f"Run ID: {run_id}")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")
    print(f"Classes: {train_classes}")
    print("Training TF-IDF + Logistic Regression...")

    # The vectorizer and classifier are fitted only on training data.
    model.fit(X_train, y_train)

    # Validation data is used only for evaluation.
    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)
    confidence = probabilities.max(axis=1)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_val,
        predictions,
        labels=train_classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv",
        index=True,
        index_label="class",
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=train_classes,
    )

    pd.DataFrame(
        cm,
        index=train_classes,
        columns=train_classes,
    ).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        index=True,
        index_label="true_class",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title("Validation Confusion Matrix - Raw OCR")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(train_classes)),
        yticks=range(len(train_classes)),
        xticklabels=train_classes,
        yticklabels=train_classes,
        xlabel="Predicted class",
        ylabel="True class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

    # Save validation predictions so individual errors can be traced.
    results = validation[
        [
            "source_index",
            "original_class",
            "routing_outcome",
            "image_path",
            "split",
        ]
    ].copy()

    results = results.rename(
        columns={"original_class": "true_class"}
    )

    results["predicted_class"] = predictions
    results["correct"] = (
        results["true_class"] == results["predicted_class"]
    )
    results["ocr_character_count"] = X_val.str.len().to_numpy()
    results["confidence"] = confidence

    results.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    # Save the fitted TF-IDF + Logistic Regression pipeline.
    joblib.dump(
        model,
        OUTPUT_DIR / "tfidf_logreg_pipeline.joblib",
    )

    empty_train = int((X_train.str.strip() == "").sum())
    empty_validation = int((X_val.str.strip() == "").sum())

    software_versions = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    metrics = {
        "run_id": run_id,
        "model": "TF-IDF + Logistic Regression",
        "ocr_variant": "raw",
        "random_seed": SEED,
        "training_records": int(len(train)),
        "validation_records": int(len(validation)),
        "classes": train_classes,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "empty_training_texts": empty_train,
        "empty_validation_texts": empty_validation,
        "test_split_used": False,
        "train_validation_source_overlap": len(overlap),
        "elapsed_seconds": round(time.time() - started, 2),
        "software_versions": software_versions,
    }

    with (OUTPUT_DIR / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    config = {
        "run_id": run_id,
        "text_source": str(TRAIN_FILE.relative_to(ROOT)),
        "validation_source": str(VALIDATION_FILE.relative_to(ROOT)),
        "target_column": "original_class",
        "ocr_variant": "raw",
        "random_seed": SEED,
        "tfidf": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "max_features": 100000,
            "sublinear_tf": True,
        },
        "logistic_regression": {
            "max_iter": 2000,
            "class_weight": None,
            "random_state": SEED,
        },
        "test_split_used": False,
        "software_versions": software_versions,
    }

    with (OUTPUT_DIR / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\nValidation results")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Empty training OCR texts: {empty_train}")
    print(f"Empty validation OCR texts: {empty_validation}")
    print(f"Train/validation overlap: {len(overlap)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("The test split was not used.")


if __name__ == "__main__":
    main()
