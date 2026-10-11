"""
Final validation for the prepared Neural Nexus dataset.

This script verifies that:

1. dataset_split.csv contains exactly 6,750 rows.
2. source_index values are unique.
3. all image files referenced in the CSV exist.
4. train/validation/test sizes are correct.
5. each original document class has the expected counts.
6. routing outcomes are distributed correctly.
7. there is no overlap between dataset splits.

This is the final quality-control step before the dataset
is handed over for preprocessing, OCR, modelling and evaluation.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_split.csv"
)


# ============================================================
# Expected values
# ============================================================

EXPECTED_TOTAL = 6750

EXPECTED_SPLIT_COUNTS = {
    "train": 4725,
    "validation": 1008,
    "test": 1017,
}

EXPECTED_CLASSES = [
    "budget",
    "email",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "questionnaire",
    "resume",
]

EXPECTED_CLASS_COUNTS = {
    "train": 525,
    "validation": 112,
    "test": 113,
}


def load_split_metadata():
    """
    Load the final dataset split metadata.
    """

    print("Loading final dataset metadata...")

    dataframe = pd.read_csv(
        DATASET_SPLIT_PATH
    )

    print(
        f"Rows loaded: {len(dataframe)}"
    )

    return dataframe


def validate_total_rows(dataframe):
    """
    Confirm that the dataset contains exactly 6,750 rows.
    """

    print("\nChecking total dataset size...")

    if len(dataframe) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"but found {len(dataframe)}."
        )

    print(
        f"✓ Total rows correct: {len(dataframe)}"
    )


def validate_unique_indices(dataframe):
    """
    Confirm that every source document appears only once.
    """

    print("\nChecking source_index uniqueness...")

    duplicate_count = (
        dataframe["source_index"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"source_index values."
        )

    print(
        "✓ All source_index values are unique."
    )


def validate_split_counts(dataframe):
    """
    Confirm exact train, validation and test sizes.
    """

    print("\nChecking split sizes...")

    actual_counts = (
        dataframe["split"]
        .value_counts()
        .to_dict()
    )

    for split_name, expected_count in (
        EXPECTED_SPLIT_COUNTS.items()
    ):

        actual_count = actual_counts.get(
            split_name,
            0,
        )

        print(
            f"{split_name:<12}: "
            f"{actual_count}"
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Incorrect {split_name} size. "
                f"Expected {expected_count}, "
                f"found {actual_count}."
            )

    print(
        "✓ Split sizes are correct."
    )


def validate_original_classes(dataframe):
    """
    Confirm that all expected document classes exist
    and have the correct counts in every split.
    """

    print(
        "\nChecking original class distribution..."
    )

    distribution = pd.crosstab(
        dataframe["original_class"],
        dataframe["split"],
    )

    print("\n")
    print(distribution)

    # Make sure no unexpected classes exist.
    actual_classes = set(
        dataframe["original_class"].unique()
    )

    expected_classes = set(
        EXPECTED_CLASSES
    )

    if actual_classes != expected_classes:
        raise ValueError(
            "Original document classes do not match "
            "the expected class list."
        )

    # Validate each class within every split.
    for class_name in EXPECTED_CLASSES:

        for split_name, expected_count in (
            EXPECTED_CLASS_COUNTS.items()
        ):

            actual_count = distribution.loc[
                class_name,
                split_name,
            ]

            if actual_count != expected_count:
                raise ValueError(
                    f"Incorrect count for "
                    f"{class_name} in {split_name}. "
                    f"Expected {expected_count}, "
                    f"found {actual_count}."
                )

    print(
        "\n✓ Original class distribution is correct."
    )


def validate_routing_outcomes(dataframe):
    """
    Display and validate routing-outcome distribution.
    """

    print(
        "\nChecking routing outcome distribution..."
    )

    routing_distribution = pd.crosstab(
        dataframe["routing_outcome"],
        dataframe["split"],
    )

    print("\n")
    print(routing_distribution)

    # Expected overall routing outcome totals.
    expected_totals = {
        "Administration": 1500,
        "Correspondence": 2250,
        "Finance": 1500,
        "HR": 750,
        "Manual Review": 750,
    }

    actual_totals = (
        dataframe["routing_outcome"]
        .value_counts()
        .to_dict()
    )

    for outcome, expected_count in (
        expected_totals.items()
    ):

        actual_count = actual_totals.get(
            outcome,
            0,
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Incorrect routing outcome total "
                f"for {outcome}. "
                f"Expected {expected_count}, "
                f"found {actual_count}."
            )

    print(
        "\n✓ Routing outcome totals are correct."
    )


def validate_image_files(dataframe):
    """
    Confirm that every referenced image file exists.
    """

    print(
        "\nChecking image files..."
    )

    missing_files = []

    for image_path in dataframe["image_path"]:

        absolute_path = (
            PROJECT_ROOT / image_path
        )

        if not absolute_path.exists():
            missing_files.append(
                image_path
            )

    if missing_files:
        print(
            f"Missing image files: "
            f"{len(missing_files)}"
        )

        # Show only the first few missing files
        # to keep terminal output readable.
        for path in missing_files[:10]:
            print(
                f"  Missing: {path}"
            )

        raise FileNotFoundError(
            "Some sampled images are missing."
        )

    print(
        f"✓ All {len(dataframe)} image files exist."
    )


def validate_split_overlap(dataframe):
    """
    Confirm that no source document appears in more
    than one dataset split.
    """

    print(
        "\nChecking split overlap..."
    )

    train_indices = set(
        dataframe[
            dataframe["split"] == "train"
        ]["source_index"]
    )

    validation_indices = set(
        dataframe[
            dataframe["split"]
            == "validation"
        ]["source_index"]
    )

    test_indices = set(
        dataframe[
            dataframe["split"] == "test"
        ]["source_index"]
    )

    if train_indices & validation_indices:
        raise ValueError(
            "Train/validation overlap detected."
        )

    if train_indices & test_indices:
        raise ValueError(
            "Train/test overlap detected."
        )

    if validation_indices & test_indices:
        raise ValueError(
            "Validation/test overlap detected."
        )

    print(
        "✓ No overlap detected between splits."
    )


def print_summary(dataframe):
    """
    Print a concise final dataset summary.
    """

    print("\n" + "=" * 60)

    print(
        "FINAL DATASET SUMMARY"
    )

    print("=" * 60)

    print(
        f"Total documents:      {len(dataframe)}"
    )

    print(
        f"Document classes:     "
        f"{dataframe['original_class'].nunique()}"
    )

    print(
        f"Routing outcomes:     "
        f"{dataframe['routing_outcome'].nunique()}"
    )

    print(
        f"Training documents:   "
        f"{(dataframe['split'] == 'train').sum()}"
    )

    print(
        f"Validation documents: "
        f"{(dataframe['split'] == 'validation').sum()}"
    )

    print(
        f"Test documents:       "
        f"{(dataframe['split'] == 'test').sum()}"
    )

    print(
        "Random seed:          42"
    )

    print("=" * 60)


def main():

    dataframe = load_split_metadata()

    validate_total_rows(
        dataframe
    )

    validate_unique_indices(
        dataframe
    )

    validate_split_counts(
        dataframe
    )

    validate_original_classes(
        dataframe
    )

    validate_routing_outcomes(
        dataframe
    )

    validate_image_files(
        dataframe
    )

    validate_split_overlap(
        dataframe
    )

    print_summary(
        dataframe
    )

    print(
        "\nFinal dataset validation successful."
    )


if __name__ == "__main__":
    main()