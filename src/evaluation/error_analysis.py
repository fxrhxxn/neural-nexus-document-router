"""Error analysis (Uddish): find real failures to explain in the report."""
import pandas as pd

from src.config import DOCUMENT_TO_ROUTING_OUTCOME as DOCUMENT_TO_DEPARTMENT
from src.evaluation.thresholding import apply_threshold


def _add_departments(df):
    """Add actual_dept and predicted_dept columns using the shared mapping."""
    out = df.copy()
    out["actual_dept"] = out["actual_class"].map(DOCUMENT_TO_DEPARTMENT)
    out["predicted_dept"] = out["predicted_class"].map(DOCUMENT_TO_DEPARTMENT)
    return out


def get_misclassified(df, confidence_col="confidence"):
    """Routing errors (wrong department), most confident mistakes first."""
    out = _add_departments(df)
    errors = out[out["actual_dept"] != out["predicted_dept"]]
    return errors.sort_values(confidence_col, ascending=False)


def most_common_confusions(df, top_n=5):
    """Count (actual class, predicted class) pairs among the routing errors."""
    errors = get_misclassified(df)
    counts = errors.groupby(["actual_class", "predicted_class"]).size()
    counts = counts.reset_index(name="count")
    return counts.sort_values("count", ascending=False).head(top_n)


def unnecessary_reviews(df, tau, confidence_col="confidence"):
    """Of the documents sent to review for LOW_CONFIDENCE, the share whose
    department the model actually got right."""
    out = apply_threshold(_add_departments(df), tau, confidence_col)
    low = out[out["review_reason"] == "LOW_CONFIDENCE"]
    if len(low) == 0:
        return float("nan")
    return (low["actual_dept"] == low["predicted_dept"]).mean()

def silent_errors(df, tau, confidence_col="confidence"):
    """Auto-routed documents sent to the WRONG department: errors no human sees."""
    out = apply_threshold(_add_departments(df), tau, confidence_col)
    auto = out[out["decision"] == "AUTO_ROUTE"]
    wrong = auto[auto["actual_dept"] != auto["predicted_dept"]]
    return wrong.sort_values(confidence_col, ascending=False)