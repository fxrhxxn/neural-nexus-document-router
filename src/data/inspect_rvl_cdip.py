"""
Initial inspection script for the RVL-CDIP dataset.

Purpose:
1. Verify that the dataset can be streamed successfully.
2. Inspect the dataset features and label definitions.
3. Display a few records before building the sampling pipeline.
"""

from datasets import load_dataset


def main():
    # Stream the dataset instead of downloading the entire RVL-CDIP dataset.
    dataset = load_dataset(
    "Chan814/rvl-cdip",
    split="train",
    streaming=True
    )   

    print("Dataset loaded successfully.")
    print("\nDataset object:")
    print(dataset)

    print("\nDataset features:")
    print(dataset.features)

    print("\nFirst 3 samples:")

    for index, sample in enumerate(dataset.take(3)):
        label_id = sample["label"]
        label_name = dataset.features["label"].int2str(label_id)

        print(f"\n--- Sample {index + 1} ---")
        print(f"Label ID: {label_id}")
        print(f"Label Name: {label_name}")
        print(f"Image Mode: {sample['image'].mode}")
        print(f"Image Size: {sample['image'].size}")


if __name__ == "__main__":
    main()