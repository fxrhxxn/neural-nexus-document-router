"""
Evaluation metrics (Uddish, Work Package C).

INPUT: a predictions CSV, one row per document. Required columns:
    file_id          - document id (Parisha's files call it source_index;
                       it is renamed when the file is loaded)
    actual_class     - true 9-class type, e.g. "invoice" (her files: true_class)
    predicted_class  - predicted 9-class type
    confidence       - highest class probability, between 0 and 1
Optional columns:
    prob_<class>     - one probability per class. If present, route_confidence
                       (summed probability of the predicted document's
                       department) is added when the file is loaded.
Department columns (actual_dept / predicted_dept) are derived from
config.DOCUMENT_TO_DEPARTMENT, so models do not need to supply them.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from config import DOCUMENT_TO_DEPARTMENT

REQUIRED_COLUMNS = ["file_id", "actual_class", "predicted_class", "confidence"]

def load_predictions(path):
    """Read a predictions CSV and check the required columns exist."""
    df = pd.read_csv(path)
    df = df.rename(columns={"source_index": "file_id", "true_class": "actual_class"})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Predictions file is missing columns: {missing}")
    
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        pred_dept = df["predicted_class"].map(DOCUMENT_TO_DEPARTMENT)
        route_conf = pd.Series(0.0, index=df.index)
        for dept in set(DOCUMENT_TO_DEPARTMENT.values()):
            cols = [f"prob_{c}" for c, d in DOCUMENT_TO_DEPARTMENT.items() if d == dept]
            mask = pred_dept == dept
            route_conf[mask] = df.loc[mask, cols].sum(axis=1)
        df["route_confidence"] = route_conf    
    
    return df


def overall_metrics(y_true, y_pred):
    """Return a dict with accuracy and macro-F1."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def per_class_report(y_true, y_pred):
    """Return a DataFrame of precision / recall / F1 / support per class."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def confusion(y_true, y_pred, labels):
    """Return a confusion matrix as a DataFrame (rows=actual, cols=predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def invoice_recall(df):
    """Of all REAL invoices, what fraction were predicted as invoice?"""
    invoices = df[df["actual_class"] == "invoice"]
    if len(invoices) == 0:
        return float("nan")
    found = (invoices["predicted_class"] == "invoice").sum()
    return found / len(invoices)


def plot_confusion_matrix(cm_df, out_path):
    """Save the confusion matrix as a PNG in outputs/figures/."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    top = cm_df.values.max()
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            value = cm_df.values[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value > top / 2 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
