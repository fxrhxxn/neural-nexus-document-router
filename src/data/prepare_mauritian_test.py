"""
Prepare the Mauritian external test documents for model evaluation.

The original documents are stored in:

    data/mauritian_test/documents/

This script:

1. Loads mauritian_test_metadata.csv.
2. Reads the selected_page value for each document.
3. Opens the source PDF.
4. Checks that the selected page actually exists.
5. Renders the selected page at 200 DPI.
6. Converts the page to RGB.
7. Saves it as a PNG inside:

    data/mauritian_test/prepared/

Important:
- Original source documents are NEVER modified.
- Page selection comes only from the metadata CSV.
- This script does NOT resize images to 224x224.
- This script does NOT run OCR.
- This script does NOT run any ML model.
"""

from pathlib import Path

import pymupdf
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

DOCUMENTS_DIR = (
    MAURITIAN_TEST_DIR
    / "documents"
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
# Preparation configuration
# ============================================================

# Render PDF pages at a reasonably high quality.
PDF_DPI = 200


LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
}


# ============================================================
# Helper functions
# ============================================================

def build_output_filename(row: pd.Series) -> str:
    """
    Create a standardized output filename.

    Example:
        MU001_questionnaire_en.png
        MU006_memo_fr.png

    The output filename is generated from metadata instead
    of simply copying the original source filename.
    """

    file_id = row["file_id"]
    document_type = row["document_type"]
    language = row["language"]

    language_code = LANGUAGE_CODES.get(language)

    if language_code is None:
        raise ValueError(
            f"{file_id}: unsupported language '{language}'"
        )

    return (
        f"{file_id}_"
        f"{document_type}_"
        f"{language_code}.png"
    )


def prepare_pdf(
    source_path: Path,
    output_path: Path,
    selected_page: int,
    file_id: str,
) -> None:
    """
    Extract one selected page from a PDF and save it
    as a high-quality RGB PNG.

    selected_page uses human-readable numbering:
        page 1 = first page

    PyMuPDF uses zero-based indexing internally:
        page 0 = first page
    """

    pdf = pymupdf.open(source_path)

    try:

        page_count = len(pdf)

        # ----------------------------------------------------
        # Make sure the requested page actually exists.
        # ----------------------------------------------------

        if selected_page < 1 or selected_page > page_count:
            raise ValueError(
                f"{file_id}: selected_page={selected_page}, "
                f"but PDF contains only {page_count} page(s)."
            )

        # Convert human page number to Python/PyMuPDF index.
        page_index = selected_page - 1

        page = pdf.load_page(page_index)

        # ----------------------------------------------------
        # Render page at 200 DPI.
        # alpha=False removes transparency and produces
        # a standard RGB-compatible page image.
        # ----------------------------------------------------

        pixmap = page.get_pixmap(
            dpi=PDF_DPI,
            alpha=False,
        )

        # Save the rendered page temporarily/directly as PNG.
        pixmap.save(output_path)

        # ----------------------------------------------------
        # Re-open using Pillow to guarantee RGB mode.
        # Some downstream image pipelines expect RGB.
        # ----------------------------------------------------

        with Image.open(output_path) as image:

            rgb_image = image.convert("RGB")

            rgb_image.save(
                output_path,
                format="PNG",
            )

    finally:

        pdf.close()


def prepare_image(
    source_path: Path,
    output_path: Path,
    selected_page: int,
    file_id: str,
) -> None:
    """
    Prepare an image source such as PNG, JPG or JPEG.

    Image files only contain one page, therefore
    selected_page must be 1.
    """

    if selected_page != 1:
        raise ValueError(
            f"{file_id}: image files can only use "
            f"selected_page=1."
        )

    with Image.open(source_path) as image:

        rgb_image = image.convert("RGB")

        rgb_image.save(
            output_path,
            format="PNG",
        )


# ============================================================
# Main preparation logic
# ============================================================

def prepare_document(
    row: pd.Series,
) -> Path:
    """
    Prepare one document based on a metadata row.
    """

    file_id = row["file_id"]
    filename = row["filename"]

    # Convert selected_page safely to integer.
    selected_page = int(row["selected_page"])

    source_path = (
        DOCUMENTS_DIR
        / filename
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"{file_id}: source file not found -> "
            f"{source_path}"
        )

    output_filename = build_output_filename(row)

    output_path = (
        PREPARED_DIR
        / output_filename
    )

    extension = source_path.suffix.lower()

    # --------------------------------------------------------
    # Handle PDF documents
    # --------------------------------------------------------

    if extension == ".pdf":

        prepare_pdf(
            source_path=source_path,
            output_path=output_path,
            selected_page=selected_page,
            file_id=file_id,
        )

    # --------------------------------------------------------
    # Handle image sources if they are added in the future
    # --------------------------------------------------------

    elif extension in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:

        prepare_image(
            source_path=source_path,
            output_path=output_path,
            selected_page=selected_page,
            file_id=file_id,
        )

    else:

        raise ValueError(
            f"{file_id}: unsupported source format "
            f"'{extension}'"
        )

    return output_path


# ============================================================
# Main program
# ============================================================

def main():

    print(
        "Preparing Mauritian external test documents..."
    )

    # --------------------------------------------------------
    # Check metadata
    # --------------------------------------------------------

    if not METADATA_CSV.exists():

        raise FileNotFoundError(
            f"Metadata CSV not found:\n"
            f"{METADATA_CSV}"
        )

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Documents directory not found:\n"
            f"{DOCUMENTS_DIR}"
        )

    # --------------------------------------------------------
    # Create prepared directory if it does not exist.
    # --------------------------------------------------------

    PREPARED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load metadata.
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA_CSV
    )

    print(
        f"\nLoaded {len(df)} document records."
    )

    print(
        f"Output directory:\n{PREPARED_DIR}"
    )

    print(
        f"\nPDF rendering quality: {PDF_DPI} DPI"
    )

    # --------------------------------------------------------
    # Prepare each document.
    # --------------------------------------------------------

    successful = 0
    errors = []

    print("\n" + "=" * 60)

    for _, row in df.iterrows():

        file_id = row["file_id"]
        filename = row["filename"]
        selected_page = row["selected_page"]

        print(
            f"\n[{file_id}]"
        )

        print(
            f"Source: {filename}"
        )

        print(
            f"Selected page: {selected_page}"
        )

        try:

            output_path = prepare_document(
                row
            )

            print(
                f"Prepared: {output_path.name}"
            )

            successful += 1

        except Exception as exc:

            error_message = (
                f"{file_id}: {exc}"
            )

            errors.append(
                error_message
            )

            print(
                f"[ERROR] {error_message}"
            )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)

    print(
        f"\nSuccessful: {successful}"
    )

    print(
        f"Failed: {len(errors)}"
    )

    if errors:

        print(
            "\nThe following documents "
            "could not be prepared:"
        )

        for error in errors:

            print(
                f"[ERROR] {error}"
            )

    else:

        print(
            "\nAll Mauritian external test "
            "documents were prepared successfully."
        )

        print(
            f"\nPrepared images are available in:\n"
            f"{PREPARED_DIR}"
        )


if __name__ == "__main__":
    main()