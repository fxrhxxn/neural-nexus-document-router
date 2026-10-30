"""
Validate the prepared Mauritian external test images.

This script checks that:

1. The prepared directory exists.
2. Every expected prepared PNG exists.
3. Each image can be opened successfully.
4. Each image is in RGB mode.
5. Each image has valid dimensions.
6. No image is suspiciously small.
7. Prepared filenames match the expected metadata naming.
8. The number of prepared images matches the metadata count.

Important:
This script does NOT run OCR.
This script does NOT run any ML model.
It only checks the prepared evaluation images.
"""

print("validate_prepared_mauritian_test.py has started")

from pathlib import Path

import pandas as pd
from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAURITIAN_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "mauritian_test"
)

PREPARED_DIR = (
    MAURITIAN_TEST_DIR
    / "prepared"
)

METADATA_CSV = (
    MAURITIAN_TEST_DIR
    / "mauritian_test_metadata.csv"
)


# ============================================================
# Validation configuration
# ============================================================

LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
}


# These are only sanity-check limits.
# A document page smaller than this may indicate
# a bad conversion or corrupted image.
MIN_WIDTH = 500
MIN_HEIGHT = 500


# ============================================================
# Helper functions
# ============================================================

def build_expected_filename(
    row: pd.Series,
) -> str:
    """
    Build the expected prepared PNG filename
    from the metadata.

    Example:
        MU001_questionnaire_en.png
        MU006_memo_fr.png
    """

    file_id = row["file_id"]
    document_type = row["document_type"]
    language = row["language"]

    language_code = LANGUAGE_CODES.get(
        language
    )

    if language_code is None:
        raise ValueError(
            f"{file_id}: unsupported language "
            f"'{language}'"
        )

    return (
        f"{file_id}_"
        f"{document_type}_"
        f"{language_code}.png"
    )


def validate_image(
    image_path: Path,
    file_id: str,
) -> list[str]:
    """
    Validate one prepared image.

    Checks:
    - image can be opened
    - image mode is RGB
    - dimensions are valid
    - image is not suspiciously small
    """

    errors = []

    try:

        with Image.open(image_path) as image:

            # Force Pillow to load the image data.
            # This helps detect corrupted files.
            image.load()

            width, height = image.size
            mode = image.mode

            # ------------------------------------------------
            # Check image mode
            # ------------------------------------------------

            if mode != "RGB":
                errors.append(
                    f"{file_id}: image mode is "
                    f"'{mode}', expected 'RGB'"
                )

            # ------------------------------------------------
            # Check dimensions
            # ------------------------------------------------

            if width <= 0 or height <= 0:
                errors.append(
                    f"{file_id}: invalid image dimensions "
                    f"{width}x{height}"
                )

            # ------------------------------------------------
            # Check that the image is not suspiciously small
            # ------------------------------------------------

            if width < MIN_WIDTH:
                errors.append(
                    f"{file_id}: image width is unusually "
                    f"small ({width}px)"
                )

            if height < MIN_HEIGHT:
                errors.append(
                    f"{file_id}: image height is unusually "
                    f"small ({height}px)"
                )

    except Exception as exc:

        errors.append(
            f"{file_id}: unable to open image -> {exc}"
        )

    return errors


# ============================================================
# Dataset summary
# ============================================================

def print_prepared_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print the dimensions and image mode
    of each prepared image.
    """

    print("\n" + "=" * 70)
    print("PREPARED MAURITIAN TEST IMAGE SUMMARY")
    print("=" * 70)

    rows = []

    for _, row in df.iterrows():

        expected_filename = build_expected_filename(
            row
        )

        image_path = (
            PREPARED_DIR
            / expected_filename
        )

        if not image_path.exists():

            rows.append(
                {
                    "file_id": row["file_id"],
                    "filename": expected_filename,
                    "width": "-",
                    "height": "-",
                    "mode": "MISSING",
                }
            )

            continue

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                rows.append(
                    {
                        "file_id": row["file_id"],
                        "filename": expected_filename,
                        "width": width,
                        "height": height,
                        "mode": image.mode,
                    }
                )

        except Exception:

            rows.append(
                {
                    "file_id": row["file_id"],
                    "filename": expected_filename,
                    "width": "-",
                    "height": "-",
                    "mode": "ERROR",
                }
            )

    summary_df = pd.DataFrame(
        rows
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


# ============================================================
# Main program
# ============================================================

def main():

    print(
        "Validating prepared Mauritian test images..."
    )

    # --------------------------------------------------------
    # Check required locations
    # --------------------------------------------------------

    if not METADATA_CSV.exists():

        raise FileNotFoundError(
            f"Metadata CSV not found:\n"
            f"{METADATA_CSV}"
        )

    if not PREPARED_DIR.exists():

        raise FileNotFoundError(
            f"Prepared directory not found:\n"
            f"{PREPARED_DIR}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_CSV
    )

    print(
        f"\nLoaded {len(df)} metadata records."
    )

    # --------------------------------------------------------
    # Collect validation issues
    # --------------------------------------------------------

    errors = []

    expected_filenames = []

    for _, row in df.iterrows():

        file_id = row["file_id"]

        try:

            expected_filename = build_expected_filename(
                row
            )

        except Exception as exc:

            errors.append(
                f"{file_id}: {exc}"
            )

            continue

        expected_filenames.append(
            expected_filename
        )

        image_path = (
            PREPARED_DIR
            / expected_filename
        )

        # ----------------------------------------------------
        # Check prepared image exists
        # ----------------------------------------------------

        if not image_path.exists():

            errors.append(
                f"{file_id}: prepared image "
                f"not found -> {expected_filename}"
            )

            continue

        # ----------------------------------------------------
        # Validate the actual image
        # ----------------------------------------------------

        errors.extend(
            validate_image(
                image_path=image_path,
                file_id=file_id,
            )
        )

    # --------------------------------------------------------
    # Check for unexpected PNG files
    # --------------------------------------------------------

    actual_png_files = {
        path.name
        for path in PREPARED_DIR.glob(
            "*.png"
        )
    }

    expected_png_files = set(
        expected_filenames
    )

    unexpected_files = (
        actual_png_files
        - expected_png_files
    )

    for filename in sorted(
        unexpected_files
    ):

        errors.append(
            f"Unexpected prepared image found: "
            f"{filename}"
        )

    # --------------------------------------------------------
    # Check image count
    # --------------------------------------------------------

    if len(actual_png_files) != len(df):

        errors.append(
            f"Prepared image count mismatch: "
            f"expected {len(df)}, "
            f"found {len(actual_png_files)}."
        )

    # --------------------------------------------------------
    # Print image summary
    # --------------------------------------------------------

    print_prepared_summary(
        df
    )

    # --------------------------------------------------------
    # Final validation result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if errors:

        print(
            "VALIDATION RESULT: ISSUES FOUND"
        )

        print("=" * 70)

        for error in errors:

            print(
                f"[ERROR] {error}"
            )

        print(
            f"\nTotal issues found: "
            f"{len(errors)}"
        )

    else:

        print(
            "VALIDATION RESULT: SUCCESS"
        )

        print("=" * 70)

        print(
            "\nAll prepared Mauritian test images "
            "passed validation successfully."
        )


if __name__ == "__main__":
    main()