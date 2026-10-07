"""
Create the fixed train/validation/test split for the sampled RVL-CDIP dataset.

The dataset contains:
    6,750 documents
    9 document classes
    750 documents per class

Target split:
    Train      = 4,725 documents
    Validation = 1,008 documents
    Test       = 1,017 documents

The split is stratified by the original 9 document classes
and uses the shared fixed random seed for reproducibility.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SAMPLED_DATASET_PATH = (
    METADATA_DIR / "sampled_dataset.csv"
)

DATASET_SPLIT_PATH = (
    METADATA_DIR / "dataset_split.csv"
)


# ============================================================
# Expected dataset sizes
# ============================================================

EXPECTED_TOTAL = 6750

TRAIN_SIZE = 4725

VALIDATION_SIZE = 1008

TEST_SIZE = 1017


def load_dataset_metadata():
    """
    Load metadata describing the 6,750 sampled documents.
    """

    print("Loading sampled dataset metadata...")

    dataframe = pd.read_csv(
        SAMPLED_DATASET_PATH
    )

    print(
        f"Documents loaded: {len(dataframe)}"
    )

    return dataframe


def validate_source_dataset(dataframe):
    """
    Validate the source metadata before creating the split.
    """

    print("\nValidating source dataset...")

    if len(dataframe) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} documents, "
            f"but found {len(dataframe)}."
        )

    # Each source index should represent one unique
    # document from the original RVL-CDIP dataset.
    if dataframe["source_index"].duplicated().any():
        raise ValueError(
            "Duplicate source indices found."
        )

    print("Source dataset validation successful.")


def create_splits(dataframe):
    """
    Create reproducible stratified train, validation and test sets.

    Stratification uses the original 9 document classes.
    """

    print("\nCreating stratified dataset split...")

    # --------------------------------------------------------
    # First split:
    #
    # 6,750 total
    #       ↓
    # 4,725 train
    # 2,025 temporary
    # --------------------------------------------------------

    train_df, temporary_df = train_test_split(
        dataframe,
        train_size=TRAIN_SIZE,
        test_size=(
            VALIDATION_SIZE + TEST_SIZE
        ),
        stratify=dataframe["original_class"],
        random_state=RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Second split:
    #
    # 2,025 temporary
    #       ↓
    # 1,008 validation
    # 1,017 test
    # --------------------------------------------------------

    validation_df, test_df = train_test_split(
        temporary_df,
        train_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        stratify=temporary_df["original_class"],
        random_state=RANDOM_SEED,
    )

    return train_df, validation_df, test_df


def assign_split_labels(
    train_df,
    validation_df,
    test_df,
):
    """
    Add a split column to each dataframe and combine
    them into one master metadata file.
    """

    # Copy prevents accidental modification warnings.
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"

    validation_df["split"] = "validation"

    test_df["split"] = "test"

    combined_df = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    return combined_df


def validate_split(
    train_df,
    validation_df,
    test_df,
):
    """
    Validate split sizes and confirm that there is
    no overlap between train, validation and test sets.
    """

    print("\nValidating dataset split...")

    # --------------------------------------------------------
    # Validate exact sizes.
    # --------------------------------------------------------

    if len(train_df) != TRAIN_SIZE:
        raise ValueError(
            "Incorrect training set size."
        )

    if len(validation_df) != VALIDATION_SIZE:
        raise ValueError(
            "Incorrect validation set size."
        )

    if len(test_df) != TEST_SIZE:
        raise ValueError(
            "Incorrect test set size."
        )

    # --------------------------------------------------------
    # Check overlap using source_index.
    # --------------------------------------------------------

    train_indices = set(
        train_df["source_index"]
    )

    validation_indices = set(
        validation_df["source_index"]
    )

    test_indices = set(
        test_df["source_index"]
    )

    train_validation_overlap = (
        train_indices & validation_indices
    )

    train_test_overlap = (
        train_indices & test_indices
    )

    validation_test_overlap = (
        validation_indices & test_indices
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):
        raise ValueError(
            "Dataset split overlap detected."
        )

    print("No overlap detected.")

    print(
        f"Train:      {len(train_df)}"
    )

    print(
        f"Validation: {len(validation_df)}"
    )

    print(
        f"Test:       {len(test_df)}"
    )


def print_class_distribution(
    train_df,
    validation_df,
    test_df,
):
    """
    Print class counts for each dataset split.
    """

    print(
        "\nClass distribution by split:\n"
    )

    distribution = pd.DataFrame(
        {
            "train": (
                train_df[
                    "original_class"
                ]
                .value_counts()
                .sort_index()
            ),
            "validation": (
                validation_df[
                    "original_class"
                ]
                .value_counts()
                .sort_index()
            ),
            "test": (
                test_df[
                    "original_class"
                ]
                .value_counts()
                .sort_index()
            ),
        }
    )

    print(distribution)

    return distribution


def save_split(combined_df):
    """
    Save the complete split metadata.
    """

    combined_df.to_csv(
        DATASET_SPLIT_PATH,
        index=False,
    )

    print(
        f"\nDataset split saved to:\n"
        f"{DATASET_SPLIT_PATH}"
    )


def main():

    dataframe = load_dataset_metadata()

    validate_source_dataset(
        dataframe
    )

    train_df, validation_df, test_df = (
        create_splits(dataframe)
    )

    validate_split(
        train_df,
        validation_df,
        test_df,
    )

    print_class_distribution(
        train_df,
        validation_df,
        test_df,
    )

    combined_df = assign_split_labels(
        train_df,
        validation_df,
        test_df,
    )

    save_split(
        combined_df
    )

    print(
        "\nDataset splitting completed successfully."
    )


if __name__ == "__main__":
    main()