"""
Summarise the completed manual OCR quality audit.

Reads outputs/ocr_audit/ocr_quality_audit.csv (read-only), validates it,
and reports:
  - average raw and Otsu manual scores
  - paired score changes (Otsu minus raw)
  - preferred-variant counts
  - the same measures by original document class

Saves:
  outputs/ocr_audit/ocr_audit_summary.csv   (per-class + overall rows)
  outputs/ocr_audit/ocr_audit_summary.txt   (readable text version)

Run from the repository root:
    python -m src.preprocessing.summarize_ocr_audit

This script never writes to the audit CSV itself.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "ocr_audit"
AUDIT_CSV = AUDIT_DIR / "ocr_quality_audit.csv"
SUMMARY_CSV = AUDIT_DIR / "ocr_audit_summary.csv"
SUMMARY_TXT = AUDIT_DIR / "ocr_audit_summary.txt"

MANUAL_COLUMNS = [
    "raw_manual_score",
    "otsu_manual_score",
    "preferred_variant",
    "audit_notes",
]
VALID_SCORES = {0, 1, 2}
VALID_PREFERENCES = {"raw", "otsu"}
EXPECTED_ROWS = 10


def load_and_validate():
    """Load the audit CSV and check it is complete and well-formed."""
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Audit file not found: {AUDIT_CSV}")

    df = pd.read_csv(AUDIT_CSV)

    missing_cols = [c for c in MANUAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    empty = df[MANUAL_COLUMNS].isna().sum()
    if empty.any():
        raise ValueError(f"Empty manual cells found:\n{empty[empty > 0]}")

    for col in ["raw_manual_score", "otsu_manual_score"]:
        bad = set(df[col].astype(int)) - VALID_SCORES
        if bad:
            raise ValueError(f"{col} has invalid values: {bad}")

    bad_pref = set(df["preferred_variant"]) - VALID_PREFERENCES
    if bad_pref:
        raise ValueError(f"preferred_variant has invalid values: {bad_pref}")

    df["raw_manual_score"] = df["raw_manual_score"].astype(int)
    df["otsu_manual_score"] = df["otsu_manual_score"].astype(int)
    df["score_change"] = df["otsu_manual_score"] - df["raw_manual_score"]
    return df


def summarise(group):
    """Summary statistics for one group of audit rows."""
    return {
        "documents": len(group),
        "mean_raw_score": round(group["raw_manual_score"].mean(), 2),
        "mean_otsu_score": round(group["otsu_manual_score"].mean(), 2),
        "mean_change_otsu_minus_raw": round(group["score_change"].mean(), 2),
        "otsu_score_higher": int((group["score_change"] > 0).sum()),
        "scores_equal": int((group["score_change"] == 0).sum()),
        "otsu_score_lower": int((group["score_change"] < 0).sum()),
        "prefer_raw": int((group["preferred_variant"] == "raw").sum()),
        "prefer_otsu": int((group["preferred_variant"] == "otsu").sum()),
    }


def main():
    df = load_and_validate()

    rows = []
    overall = {"original_class": "ALL"}
    overall.update(summarise(df))
    rows.append(overall)

    for cls, group in df.groupby("original_class", sort=True):
        row = {"original_class": cls}
        row.update(summarise(group))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    lines = []
    lines.append("OCR manual audit summary")
    lines.append(f"Source: {AUDIT_CSV.relative_to(PROJECT_ROOT)}")
    lines.append(f"Documents audited: {len(df)} (training split, seed 42)")
    lines.append("Scores: 0 = unusable, 1 = partially readable, 2 = mostly readable")
    lines.append("")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("Per-document scores:")
    lines.append(
        df[
            [
                "source_index",
                "original_class",
                "raw_manual_score",
                "otsu_manual_score",
                "score_change",
                "preferred_variant",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append(
        "Limitations: 10 documents from 5 of 9 classes (budget, email, memo and "
        "questionnaire not sampled); scores are manual judgements without "
        "reference transcriptions, so they are not OCR accuracy, CER or WER; "
        "Otsu here means grayscale + 1.5x upscale + Otsu, so any difference "
        "cannot be attributed to the threshold method alone."
    )

    text = "\n".join(lines)
    SUMMARY_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {SUMMARY_CSV}")
    print(f"Saved: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
