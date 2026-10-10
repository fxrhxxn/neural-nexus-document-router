"""
OCR quality audit.

Purpose:
1. Load the OCR comparison experiment.
2. Calculate simple automatic OCR-quality proxies.
3. Create fields for manual readability scoring.
4. Save a structured OCR audit CSV.

Important:
The automatic measurements used here are NOT true OCR accuracy
because the dataset does not provide ground-truth transcriptions.

Manual readability scoring:
    0 = Unusable
    1 = Partially readable
    2 = Mostly readable
"""

from pathlib import Path
import re

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_sample"
    / "ocr_preprocessing_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ocr_audit"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "ocr_quality_audit.csv"
)


# ============================================================
# OCR proxy functions
# ============================================================

def count_characters(text: str) -> int:
    """
    Count non-whitespace characters in OCR text.
    """

    if pd.isna(text):
        return 0

    return len(
        re.sub(
            r"\s+",
            "",
            str(text),
        )
    )


def count_words(text: str) -> int:
    """
    Count OCR tokens separated by whitespace.
    """

    if pd.isna(text):
        return 0

    words = str(text).split()

    return len(words)


def alphabetic_token_ratio(text: str) -> float:
    """
    Calculate the proportion of OCR tokens that contain
    mainly alphabetic characters.

    This is only a rough quality proxy.

    Example:

        ["Invoice", "Total", "$30,000", "@@@"]

    Alphabetic-looking tokens may suggest more readable OCR,
    but this is NOT equivalent to OCR accuracy.
    """

    if pd.isna(text):
        return 0.0

    tokens = str(text).split()

    if not tokens:
        return 0.0

    alphabetic_tokens = 0

    for token in tokens:

        # Remove common punctuation around words.
        cleaned_token = re.sub(
            r"[^A-Za-zÀ-ÿ]",
            "",
            token,
        )

        # Count token as alphabetic if meaningful letters remain.
        if (
            len(cleaned_token) >= 2
            and cleaned_token.isalpha()
        ):
            alphabetic_tokens += 1

    return round(
        alphabetic_tokens / len(tokens),
        4,
    )


# ============================================================
# Build audit table
# ============================================================

def build_audit_dataframe(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build OCR quality-audit measurements for every document.

    Manual score fields are deliberately left empty so they
    can be completed after visual inspection.

    Score:
        0 = Unusable
        1 = Partially readable
        2 = Mostly readable
    """

    audit_rows = []

    for _, row in comparison_df.iterrows():

        raw_text = row["raw_ocr_text"]
        otsu_text = row["otsu_ocr_text"]

        audit_rows.append(
            {
                "source_index": row["source_index"],
                "original_class": row["original_class"],

                # --------------------------------------------
                # Raw OCR automatic proxies
                # --------------------------------------------

                "raw_character_count": (
                    count_characters(raw_text)
                ),

                "raw_word_count": (
                    count_words(raw_text)
                ),

                "raw_alphabetic_token_ratio": (
                    alphabetic_token_ratio(raw_text)
                ),

                # --------------------------------------------
                # Otsu OCR automatic proxies
                # --------------------------------------------

                "otsu_character_count": (
                    count_characters(otsu_text)
                ),

                "otsu_word_count": (
                    count_words(otsu_text)
                ),

                "otsu_alphabetic_token_ratio": (
                    alphabetic_token_ratio(otsu_text)
                ),

                # --------------------------------------------
                # Manual audit fields
                # --------------------------------------------

                "raw_manual_score": "",
                "otsu_manual_score": "",
                "preferred_variant": "",
                "audit_notes": "",
            }
        )

    return pd.DataFrame(
        audit_rows
    )


# ============================================================
# Main program
# ============================================================

def main():

    print(
        "Loading OCR preprocessing comparison..."
    )

    comparison_df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"Documents loaded: "
        f"{len(comparison_df)}"
    )

    audit_df = build_audit_dataframe(
        comparison_df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print(
        "\nOCR quality audit created successfully."
    )

    print(
        f"\nSaved to:\n{OUTPUT_CSV}"
    )

    print(
        "\nManual scoring guide:"
    )

    print(
        "0 = Unusable"
    )

    print(
        "1 = Partially readable"
    )

    print(
        "2 = Mostly readable"
    )

    print(
        "\nImportant: automatic measurements are "
        "quality proxies, not true OCR accuracy."
    )


if __name__ == "__main__":
    main()