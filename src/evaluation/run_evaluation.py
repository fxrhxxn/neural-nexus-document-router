"""Run the full evaluation on a predictions CSV (validation or test)."""
import argparse
from pathlib import Path

from config import DOCUMENT_TO_DEPARTMENT
from src.evaluation.metrics import (
    load_predictions, overall_metrics, per_class_report, confusion,
    invoice_recall, plot_confusion_matrix,
)
from src.evaluation.thresholding import (
    DEFAULT_THRESHOLDS, coverage_accuracy_table, choose_threshold,
    plot_coverage_vs_accuracy,
)
from src.evaluation.error_analysis import (
    silent_errors, unnecessary_reviews, most_common_confusions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a predictions CSV.")
    parser.add_argument("predictions", help="path to the predictions CSV")
    parser.add_argument("--name", default="run", help="label used in output file names")
    parser.add_argument("--min-accuracy", type=float, default=95.0,
                        help="required accuracy (%%) of auto-routed documents")
    parser.add_argument("--confidence", default="confidence",
                        help="confidence column: 'confidence' or 'route_confidence'")
    parser.add_argument("--tau", type=float, default=None,
                        help="FROZEN threshold from validation. If given, tau is NOT "
                             "chosen from this file (use this for the test set)")
    args = parser.parse_args()
        
    if args.tau is None and "test" in args.name.lower():
        parser.error("Test-set runs must pass a frozen --tau chosen on validation.")

    df = load_predictions(args.predictions)
    df["actual_dept"] = df["actual_class"].map(DOCUMENT_TO_DEPARTMENT)
    df["predicted_dept"] = df["predicted_class"].map(DOCUMENT_TO_DEPARTMENT)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Primary task: 5 departments
    print("Department level:", overall_metrics(df["actual_dept"], df["predicted_dept"]))
    # Secondary: 9 document types
    print("Document type level:", overall_metrics(df["actual_class"], df["predicted_class"]))
    print("Invoice recall:", invoice_recall(df))

    per_class_report(df["actual_class"], df["predicted_class"]).to_csv(
        metrics_dir / f"{args.name}_per_class_report.csv")
    cm = confusion(df["actual_class"], df["predicted_class"],
                   labels=list(DOCUMENT_TO_DEPARTMENT))
    cm.to_csv(metrics_dir / f"{args.name}_confusion_matrix.csv")
    plot_confusion_matrix(cm, figures_dir / f"{args.name}_confusion_matrix.png")

    tau = None
    if args.tau is None:
        # Validation mode: try several thresholds and choose one
        table = coverage_accuracy_table(df, DEFAULT_THRESHOLDS,
                                        confidence_col=args.confidence)
        table.to_csv(metrics_dir / f"{args.name}_coverage_accuracy.csv", index=False)
        plot_coverage_vs_accuracy(
            table, figures_dir / f"{args.name}_coverage_vs_accuracy.png")
        print(table.round(2).to_string(index=False))
        tau = choose_threshold(table, args.min_accuracy)
        print(f"Required accuracy {args.min_accuracy}% -> chosen threshold tau = {tau}")
    else:
        # Frozen mode: use the given tau and do not choose anything
        table = coverage_accuracy_table(df, [args.tau],
                                        confidence_col=args.confidence)
        table.to_csv(metrics_dir / f"{args.name}_frozen_tau_result.csv", index=False)
        print(f"Using FROZEN tau = {args.tau} (not chosen from this file)")
        print(table.round(2).to_string(index=False))
        tau = args.tau

    if tau is not None:
        silent = silent_errors(df, tau, args.confidence)
        silent.to_csv(metrics_dir / f"{args.name}_silent_errors.csv", index=False)
        print(f"Silent errors (auto-routed to wrong department): {len(silent)}"
              f" (invoices: {(silent['actual_class'] == 'invoice').sum()})")
        print("Reviewed for low confidence but already correct:",
              round(unnecessary_reviews(df, tau, args.confidence), 3))
    most_common_confusions(df).to_csv(
        metrics_dir / f"{args.name}_top_confusions.csv", index=False)


if __name__ == "__main__":
    main()