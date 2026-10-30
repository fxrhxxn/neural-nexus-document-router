"""
Review OCR preprocessing experiment results.

This script loads the saved OCR comparison CSV and prints
the raw OCR and Otsu OCR for each sampled document.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "preprocessing_sample"
    / "ocr_preprocessing_comparison.csv"
)


def main():
    dataframe = pd.read_csv(RESULTS_PATH)

    print(
        f"Loaded {len(dataframe)} OCR comparison records."
    )

    for _, row in dataframe.iterrows():

        print("\n" + "=" * 80)
        print(
            f"CLASS: {row['original_class']} | "
            f"SOURCE INDEX: {row['source_index']}"
        )
        print("=" * 80)

        print("\n--- RAW OCR ---\n")
        print(row["raw_ocr_text"])

        print("\n--- OTSU OCR ---\n")
        print(row["otsu_ocr_text"])

        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()