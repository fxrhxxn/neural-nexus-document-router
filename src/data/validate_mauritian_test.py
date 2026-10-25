"""
Validate the Mauritian external test dataset.

This script checks that:

1. The metadata CSV exists.
2. All required metadata columns are present.
3. File IDs are unique and follow the MU### format.
4. Filenames are unique.
5. Document classes belong to the 9 supported classes.
6. Routing outcomes match the expected class-to-route mapping.
7. Languages are valid.
8. Filenames follow the expected naming convention.
9. Every document listed in the metadata CSV actually exists.
10. The expected number of documents is present.
11. Every document has a valid selected_page value.

Important:
This script DOES NOT train or evaluate any model.
It only checks the integrity of the external Mauritian test set.
"""

print("validate_mauritian_test.py has started")

from pathlib import Path
import re

import pandas as pd


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

METADATA_CSV = (
    MAURITIAN_TEST_DIR
    / "mauritian_test_metadata.csv"
)


# ============================================================
# Expected dataset configuration
# ============================================================

EXPECTED_DOCUMENT_COUNT = 20


ALLOWED_CLASSES = {
    "budget",
    "email",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "questionnaire",
    "resume",
}


CLASS_TO_ROUTE = {
    "budget": "Finance",
    "invoice": "Finance",

    "email": "Correspondence",
    "letter": "Correspondence",
    "memo": "Correspondence",

    "form": "Administration",
    "questionnaire": "Administration",

    "resume": "HR",

    "handwritten": "Manual Review",
}


ALLOWED_LANGUAGES = {
    "English",
    "French",
}


REQUIRED_COLUMNS = {
    "file_id",
    "filename",
    "document_type",
    "routing_outcome",
    "language",
    "source_type",
    "source_description",
    "real_or_mock",
    "creator",
    "date_added",
    "notes",
    "selected_page",
}


# ============================================================
# Validation functions
# ============================================================

def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """
    Check that all required metadata columns exist.
    """

    errors = []

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return errors


def validate_unique_ids(df: pd.DataFrame) -> list[str]:
    """
    Check that every file_id is unique.
    """

    errors = []

    duplicates = df[
        df["file_id"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        duplicate_ids = sorted(
            duplicates["file_id"].unique()
        )

        errors.append(
            "Duplicate file_id values found: "
            + ", ".join(duplicate_ids)
        )

    return errors


def validate_unique_filenames(df: pd.DataFrame) -> list[str]:
    """
    Check that every filename is unique.
    """

    errors = []

    duplicates = df[
        df["filename"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        duplicate_names = sorted(
            duplicates["filename"].unique()
        )

        errors.append(
            "Duplicate filenames found: "
            + ", ".join(duplicate_names)
        )

    return errors


def validate_file_ids(df: pd.DataFrame) -> list[str]:
    """
    File IDs must follow the format:

        MU001
        MU002
        ...
        MU010
    """

    errors = []

    pattern = re.compile(r"^MU\d{3}$")

    for file_id in df["file_id"]:

        if not pattern.match(str(file_id)):
            errors.append(
                f"Invalid file_id format: {file_id}"
            )

    return errors


def validate_document_classes(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that every document belongs to one of the
    nine supported RVL-CDIP document classes.
    """

    errors = []

    for _, row in df.iterrows():

        document_type = row["document_type"]

        if document_type not in ALLOWED_CLASSES:
            errors.append(
                f"{row['file_id']}: "
                f"invalid document_type "
                f"'{document_type}'"
            )

    return errors


def validate_routing_mapping(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that every document class is mapped to
    the correct routing destination.
    """

    errors = []

    for _, row in df.iterrows():

        document_type = row["document_type"]
        actual_route = row["routing_outcome"]

        if document_type not in CLASS_TO_ROUTE:
            continue

        expected_route = CLASS_TO_ROUTE[
            document_type
        ]

        if actual_route != expected_route:
            errors.append(
                f"{row['file_id']}: "
                f"{document_type} should route to "
                f"'{expected_route}', "
                f"not '{actual_route}'"
            )

    return errors


def validate_languages(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that only English and French are used.
    """

    errors = []

    for _, row in df.iterrows():

        language = row["language"]

        if language not in ALLOWED_LANGUAGES:
            errors.append(
                f"{row['file_id']}: "
                f"invalid language '{language}'"
            )

    return errors


def validate_filename_format(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check filename naming convention.

    Expected format examples:

        MU001_questionnaire_en.pdf
        MU006_memo_fr.pdf
        MU007_handwritten_en.pdf
    """

    errors = []

    language_codes = {
        "English": "en",
        "French": "fr",
    }

    for _, row in df.iterrows():

        file_id = row["file_id"]
        document_type = row["document_type"]
        language = row["language"]
        filename = row["filename"]

        language_code = language_codes.get(
            language
        )

        if language_code is None:
            continue

        expected_prefix = (
            f"{file_id}_"
            f"{document_type}_"
            f"{language_code}"
        )

        if not str(filename).startswith(
            expected_prefix
        ):
            errors.append(
                f"{file_id}: filename "
                f"'{filename}' does not follow "
                f"expected naming convention "
                f"'{expected_prefix}.<extension>'"
            )

    return errors


def validate_files_exist(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that every document listed in the metadata
    actually exists inside data/mauritian_test/documents/.
    """

    errors = []

    for _, row in df.iterrows():

        filename = row["filename"]

        file_path = (
            DOCUMENTS_DIR
            / filename
        )

        if not file_path.exists():
            errors.append(
                f"{row['file_id']}: "
                f"file not found -> {filename}"
            )

    return errors


def validate_document_count(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that the expected 10 metadata records exist.
    """

    errors = []

    actual_count = len(df)

    if actual_count != EXPECTED_DOCUMENT_COUNT:
        errors.append(
            f"Expected "
            f"{EXPECTED_DOCUMENT_COUNT} "
            f"metadata records, "
            f"but found {actual_count}."
        )

    return errors


def validate_empty_fields(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check for empty values in important columns.
    """

    errors = []

    important_columns = [
        "file_id",
        "filename",
        "document_type",
        "routing_outcome",
        "language",
        "source_type",
        "source_description",
        "real_or_mock",
        "creator",
        "date_added",
        "selected_page",
    ]

    for column in important_columns:

        empty_rows = df[
            df[column].isna()
            | (
                df[column]
                .astype(str)
                .str.strip()
                == ""
            )
        ]

        for _, row in empty_rows.iterrows():

            errors.append(
                f"{row.get('file_id', 'UNKNOWN')}: "
                f"empty value in column "
                f"'{column}'"
            )

    return errors


def validate_selected_pages(
    df: pd.DataFrame,
) -> list[str]:
    """
    Check that selected_page contains a valid page number.

    Rules:
    - selected_page must not be empty
    - selected_page must be numeric
    - selected_page must be a whole number
    - selected_page must be 1 or greater

    Example valid values:
        1
        2
        3
    """

    errors = []

    for _, row in df.iterrows():

        file_id = row["file_id"]
        selected_page = row["selected_page"]

        # Check for missing values
        if pd.isna(selected_page):
            errors.append(
                f"{file_id}: selected_page is missing"
            )
            continue

        # Try converting the value to a number
        try:
            page_number = float(selected_page)

        except (ValueError, TypeError):
            errors.append(
                f"{file_id}: selected_page "
                f"'{selected_page}' is not numeric"
            )
            continue

        # The page number must be a whole number.
        # For example, 1 or 3 is valid.
        # 1.5 is not valid.
        if not page_number.is_integer():
            errors.append(
                f"{file_id}: selected_page "
                f"'{selected_page}' must be "
                f"a whole number"
            )
            continue

        page_number = int(page_number)

        # Page numbers start at 1 for human-readable PDFs.
        if page_number < 1:
            errors.append(
                f"{file_id}: selected_page "
                f"must be 1 or greater, "
                f"but found {page_number}"
            )

    return errors


# ============================================================
# Dataset summary
# ============================================================

def print_dataset_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a summary of the external dataset.
    """

    print("\n" + "=" * 60)
    print("MAURITIAN EXTERNAL TEST SET SUMMARY")
    print("=" * 60)

    print(
        f"\nMetadata records: {len(df)}"
    )

    print(
        "\nDocument classes:"
    )

    print(
        df["document_type"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nRouting outcomes:"
    )

    print(
        df["routing_outcome"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nLanguages:"
    )

    print(
        df["language"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSelected pages:"
    )

    print(
        df[
            [
                "file_id",
                "document_type",
                "selected_page",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Main program
# ============================================================

def main():

    print(
        "Validating Mauritian external test set..."
    )

    # --------------------------------------------------------
    # Check folders/files exist before loading metadata
    # --------------------------------------------------------

    if not METADATA_CSV.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_CSV}"
        )

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Documents folder not found:\n"
            f"{DOCUMENTS_DIR}"
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
    # Validate required columns first
    # --------------------------------------------------------

    errors = validate_required_columns(
        df
    )

    if errors:

        print("\nValidation failed.\n")

        for error in errors:
            print(
                f"[ERROR] {error}"
            )

        return

    # --------------------------------------------------------
    # Run remaining validation checks
    # --------------------------------------------------------

    validation_functions = [
        validate_unique_ids,
        validate_unique_filenames,
        validate_file_ids,
        validate_document_classes,
        validate_routing_mapping,
        validate_languages,
        validate_filename_format,
        validate_files_exist,
        validate_document_count,
        validate_empty_fields,
        validate_selected_pages,
    ]

    for validation_function in validation_functions:

        errors.extend(
            validation_function(df)
        )

    # --------------------------------------------------------
    # Show dataset summary
    # --------------------------------------------------------

    print_dataset_summary(
        df
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    if errors:

        print(
            "VALIDATION RESULT: ISSUES FOUND"
        )

        print("=" * 60)

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

        print("=" * 60)

        print(
            "\nAll Mauritian external test-set "
            "checks passed successfully."
        )


if __name__ == "__main__":
    main()