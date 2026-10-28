"""
Confidence threshold + manual-review queue (Uddish).

Rule:  confidence >= tau -> AUTO_ROUTE
       otherwise         -> MANUAL_REVIEW with reason LOW_CONFIDENCE
       predicted handwritten -> MANUAL_REVIEW with reason HANDWRITTEN

tau is chosen on VALIDATION predictions only. Never on the test set.
"""
import pandas as pd
from config import DOCUMENT_TO_DEPARTMENT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]


def apply_threshold(df, tau, confidence_col="confidence"):
    """Add 'decision' and 'review_reason' columns to the predictions."""
    out = df.copy()
    is_handwritten = out["predicted_class"] == "handwritten"
    is_low = (out[confidence_col] < tau) & ~is_handwritten

    out["decision"] = "AUTO_ROUTE"
    out["review_reason"] = ""
    out.loc[is_handwritten, "decision"] = "MANUAL_REVIEW"
    out.loc[is_handwritten, "review_reason"] = "HANDWRITTEN"
    out.loc[is_low, "decision"] = "MANUAL_REVIEW"
    out.loc[is_low, "review_reason"] = "LOW_CONFIDENCE"
    return out


def coverage_accuracy_table(df, thresholds=DEFAULT_THRESHOLDS, confidence_col="confidence"):
    """One row per tau: auto-routed %, review % (by reason), auto-routed accuracy."""
    df = df.copy()
    df["actual_dept"] = df["actual_class"].map(DOCUMENT_TO_DEPARTMENT)
    df["predicted_dept"] = df["predicted_class"].map(DOCUMENT_TO_DEPARTMENT)

    rows = []
    for tau in thresholds:
        out = apply_threshold(df, tau, confidence_col)
        auto = out[out["decision"] == "AUTO_ROUTE"]
        total = len(out)
        if len(auto) > 0:
            accuracy = (auto["actual_dept"] == auto["predicted_dept"]).mean()
        else:
            accuracy = float("nan")
        rows.append({
            "threshold": tau,
            "auto_routed_pct": 100 * len(auto) / total,
            "manual_review_pct": 100 * (total - len(auto)) / total,
            "low_confidence_pct": 100 * (out["review_reason"] == "LOW_CONFIDENCE").sum() / total,
            "handwritten_pct": 100 * (out["review_reason"] == "HANDWRITTEN").sum() / total,
            "auto_routed_accuracy_pct": 100 * accuracy,
        })
    return pd.DataFrame(rows)


def choose_threshold(table, min_accuracy):
    """Pick the LOWEST tau whose auto-routed accuracy >= min_accuracy."""
    ok = table[table["auto_routed_accuracy_pct"] >= min_accuracy]
    if ok.empty:
        return None
    return ok["threshold"].min()


def plot_coverage_vs_accuracy(table, out_path):
    """Save the coverage-vs-accuracy plot in outputs/figures/."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["auto_routed_pct"], table["auto_routed_accuracy_pct"], marker="o")
    for _, row in table.iterrows():
        ax.annotate(f"tau={row['threshold']:.2f}",
                    (row["auto_routed_pct"], row["auto_routed_accuracy_pct"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Auto-routed documents (%)")
    ax.set_ylabel("Accuracy of auto-routed documents (%)")
    ax.set_title("Coverage vs accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
