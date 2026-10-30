from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = [
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

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


def get_resnet_transform(
    training=False,
    use_augmentation=False,
):
    """Preprocessing used for ResNet-18 images."""

    steps = []

    # Small rotation can be enabled for training experiments
    if training and use_augmentation:
        steps.append(
            transforms.RandomRotation(
                degrees=2,
                fill=255,
            )
        )

    steps.extend(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(steps)


class NeuralNexusDataset(Dataset):
    """Loads the fixed train or validation split."""

    def __init__(
        self,
        split,
        csv_path="data/metadata/dataset_split.csv",
        use_augmentation=False,
    ):
        # Only train and validation are allowed here
        if split not in {"train", "validation"}:
            raise ValueError(
                "Only 'train' and 'validation' may be loaded."
            )

        self.repo_root = Path(__file__).resolve().parents[2]

        metadata = pd.read_csv(
            self.repo_root / csv_path
        )

        self.data = (
            metadata[metadata["split"] == split]
            .reset_index(drop=True)
        )

        self.split = split

        # Validation never uses augmentation
        self.transform = get_resnet_transform(
            training=(split == "train"),
            use_augmentation=use_augmentation,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = (
            self.repo_root / row["image_path"]
        )

        with Image.open(image_path) as image:
            # Pretrained ResNet expects three image channels
            image = image.convert("RGB")
            image = self.transform(image)

        class_name = row["original_class"]

        return {
            "image": image,
            "label": CLASS_TO_INDEX[class_name],
            "class_name": class_name,
            "routing_outcome": row["routing_outcome"],
            "source_index": int(row["source_index"]),
        }
