
"""Resumable OCR extraction for training and validation images only.

Outputs JSONL files under outputs/ocr_cache/.
Successful records are skipped on subsequent runs.
The test split is deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = ROOT / "data" / "metadata" / "dataset_split.csv"
OUTPUT_DIR = ROOT / "outputs" / "ocr_cache"
ALLOWED_SPLITS = {"train", "validation"}


def load_completed(output_file: Path) -> set[str]:
    """Return image paths with successful cached OCR results."""
    completed = set()

    if not output_file.exists():
        return completed

    with output_file.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Ignore a possibly incomplete final line after interruption.
                continue

            if record.get("status") == "ok":
                completed.add(str(record.get("image_path", "")))

    return completed


def prepare_image(image_path: Path, variant: str):
    """Load an image, optionally applying the audited Otsu candidate."""
    if variant == "raw":
        return Image.open(image_path).convert("RGB")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    # Match the candidate approach: grayscale, 1.5x upscale, Otsu threshold.
    image = cv2.resize(
        image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )
    _, image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["raw", "otsu"],
        default="raw",
        help="OCR raw images or the Otsu preprocessing candidate.",
    )
    parser.add_argument(
        "--lang",
        default="eng+fra",
        help="Tesseract language code, e.g. eng+fra or eng.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for a small test run; 0 means all records.",
    )
    args = parser.parse_args()

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Dataset split file not found: {SPLIT_FILE}")

    df = pd.read_csv(SPLIT_FILE)

    required = {
        "image_path",
        "split",
        "source_index",
        "class_id",
        "original_class",
        "routing_outcome",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    # Never extract OCR from the test set in this pipeline.
    df = df[df["split"].isin(ALLOWED_SPLITS)].copy()

    if not set(df["split"].unique()).issubset(ALLOWED_SPLITS):
        raise RuntimeError("Unexpected split detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "validation"):
        split_df = df[df["split"] == split_name].copy()
        output_file = OUTPUT_DIR / f"{split_name}_{args.variant}.jsonl"

        completed = load_completed(output_file)
        pending = split_df[
            ~split_df["image_path"].astype(str).isin(completed)
        ]

        if args.limit > 0:
            pending = pending.head(args.limit)

        print(
            f"\n{split_name}: {len(split_df)} total, "
            f"{len(completed)} cached successes, "
            f"{len(pending)} to process"
        )

        if pending.empty:
            print("Nothing to do for this split.")
            continue

        started = time.time()

        # Append after every record so progress survives interruptions.
        with output_file.open("a", encoding="utf-8") as output:
            for number, (_, row) in enumerate(pending.iterrows(), start=1):
                raw_path = str(row["image_path"]).replace("\\", "/")
                image_path = ROOT / Path(raw_path)

                record = {
                    "source_index": int(row["source_index"]),
                    "class_id": int(row["class_id"]),
                    "original_class": str(row["original_class"]),
                    "routing_outcome": str(row["routing_outcome"]),
                    "split": split_name,
                    "image_path": raw_path,
                    "variant": args.variant,
                    "language": args.lang,
                }

                try:
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Image not found: {image_path}")

                    image = prepare_image(image_path, args.variant)
                    text = pytesseract.image_to_string(
                        image,
                        lang=args.lang,
                    )

                    record.update({
                        "status": "ok",
                        "text": text,
                        "error": "",
                    })

                except Exception as exc:
                    record.update({
                        "status": "error",
                        "text": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    })

                output.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                output.flush()

                if number % 25 == 0 or number == len(pending):
                    print(
                        f"  {split_name}: {number}/{len(pending)} "
                        f"processed; latest status={record['status']}"
                    )

        elapsed = time.time() - started
        print(f"Saved checkpoint file: {output_file}")
        print(f"Elapsed: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
