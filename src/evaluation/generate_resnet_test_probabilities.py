from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.models.resnet_dataset import CLASS_NAMES, get_resnet_transform
from src.models.resnet_model import build_resnet18, get_device


CHECKPOINT_PATH = Path(
    "outputs/deep_learning/resnet/"
    "partial_finetune_layer4_fc_aug/"
    "best_resnet18.pt"
)

OUTPUT_PATH = Path(
    "outputs/deep_learning/resnet/"
    "partial_finetune_layer4_fc_aug/"
    "test_probabilities.csv"
)

METADATA_PATH = Path("data/metadata/dataset_split.csv")

BATCH_SIZE = 16


class FinalTestDataset(Dataset):
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]

        metadata = pd.read_csv(
            self.repo_root / METADATA_PATH
        )

        # Final held-out test split only
        self.data = (
            metadata[metadata["split"] == "test"]
            .reset_index(drop=True)
        )

        # Same preprocessing used for validation
        self.transform = get_resnet_transform(
            training=False,
            use_augmentation=False,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = (
            self.repo_root / row["image_path"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "source_index": int(row["source_index"]),
        }


def main():
    device = get_device()

    print(f"Using device: {device}")

    dataset = FinalTestDataset()

    print(f"Test documents: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            ).cpu()

            source_indices = (
                batch["source_index"]
                .cpu()
                .tolist()
            )

            for source_index, probability_vector in zip(
                source_indices,
                probabilities.tolist(),
            ):
                row = {
                    "source_index": source_index,
                }

                for class_name, probability in zip(
                    CLASS_NAMES,
                    probability_vector,
                ):
                    row[f"prob_{class_name}"] = probability

                rows.append(row)

    dataframe = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    probability_columns = [
        f"prob_{class_name}"
        for class_name in CLASS_NAMES
    ]

    probability_sums = dataframe[
        probability_columns
    ].sum(axis=1)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {list(dataframe.columns)}")

    print(
        "Probability sum range: "
        f"{probability_sums.min():.6f} "
        f"to {probability_sums.max():.6f}"
    )


if __name__ == "__main__":
    main()
