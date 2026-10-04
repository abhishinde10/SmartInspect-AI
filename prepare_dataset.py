from pathlib import Path
import shutil
import random
import csv
from PIL import Image
import numpy as np

# ============================================================
# SMARTINSPECT AI - CLEAN DATASET PREPARATION
# ============================================================

SOURCE_DIR = Path("dataset")
OUTPUT_DIR = Path("SmartInspect_Final")

SEED = 42
random.seed(SEED)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
}

PRODUCTS = [
    "bottle",
    "BSD",
    "cable",
    "carpet",
    "grid",
    "leather",
    "metal_nut",
    "toothbrush",
    "wood",
    "zipper"
]


# ============================================================
# DELETE PREVIOUS FINAL DATASET
# ============================================================

if OUTPUT_DIR.exists():

    print("Removing previous SmartInspect_Final...")

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:

    for product in PRODUCTS:

        for status in ["good", "defective"]:

            folder = (
                OUTPUT_DIR
                / split
                / f"{product}_{status}"
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# STATISTICS
# ============================================================

metadata = []

total_original = 0
total_black_removed = 0
total_used = 0


# ============================================================
# PROCESS DATASET
# ============================================================

for product in PRODUCTS:

    print("\n" + "=" * 65)
    print(f"Processing: {product}")
    print("=" * 65)

    for original_status in ["good", "bad"]:

        source_folder = (
            SOURCE_DIR
            / product
            / original_status
        )

        if not source_folder.exists():

            print(
                f"WARNING: Missing folder: "
                f"{source_folder}"
            )

            continue

        status = (
            "good"
            if original_status == "good"
            else "defective"
        )

        valid_images = []

        black_removed = 0

        # ----------------------------------------------------
        # READ AND CHECK IMAGES
        # ----------------------------------------------------

        for image_path in source_folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_original += 1

            try:

                with Image.open(image_path) as img:

                    img = img.convert("RGB")

                    arr = np.array(img)

                    # Completely black image
                    if arr.max() == 0:

                        black_removed += 1
                        total_black_removed += 1

                        continue

                valid_images.append(image_path)

            except Exception as e:

                print(
                    f"WARNING: Could not read "
                    f"{image_path}: {e}"
                )

        # ----------------------------------------------------
        # SHUFFLE
        # ----------------------------------------------------

        random.shuffle(valid_images)

        total = len(valid_images)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train_images = valid_images[
            :train_count
        ]

        validation_images = valid_images[
            train_count:
            train_count + val_count
        ]

        test_images = valid_images[
            train_count + val_count:
        ]

        splits = {
            "train": train_images,
            "validation": validation_images,
            "test": test_images
        }

        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print(
            f"{status:10} | "
            f"Original: {total + black_removed:4} | "
            f"Black removed: {black_removed:4} | "
            f"Used: {total:4} | "
            f"Train: {len(train_images):4} | "
            f"Val: {len(validation_images):4} | "
            f"Test: {len(test_images):4}"
        )

        # ----------------------------------------------------
        # COPY IMAGES
        # ----------------------------------------------------

        for split_name, split_images in splits.items():

            destination_folder = (
                OUTPUT_DIR
                / split_name
                / f"{product}_{status}"
            )

            for index, image_path in enumerate(
                split_images
            ):

                new_filename = (
                    f"{product}_"
                    f"{status}_"
                    f"{index:04d}"
                    f"{image_path.suffix.lower()}"
                )

                destination = (
                    destination_folder
                    / new_filename
                )

                shutil.copy2(
                    image_path,
                    destination
                )

                metadata.append({

                    "filename":
                        new_filename,

                    "product":
                        product,

                    "status":
                        status,

                    "split":
                        split_name,

                    "original_path":
                        str(image_path)
                })

                total_used += 1


# ============================================================
# SAVE METADATA
# ============================================================

metadata_file = (
    OUTPUT_DIR / "metadata.csv"
)

with open(
    metadata_file,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "filename",
        "product",
        "status",
        "split",
        "original_path"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(metadata)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("SMARTINSPECT AI - CLEAN DATASET COMPLETE")
print("=" * 75)

print(
    f"Original images        : "
    f"{total_original}"
)

print(
    f"Completely black removed: "
    f"{total_black_removed}"
)

print(
    f"Images used             : "
    f"{total_used}"
)

print(
    f"Expected remaining      : "
    f"{total_original - total_black_removed}"
)

print(
    f"\nOutput directory:"
    f" {OUTPUT_DIR}"
)

print(
    f"Metadata:"
    f" {metadata_file}"
)

print("\nSplit:")
print("Train       : 70%")
print("Validation  : 15%")
print("Test        : 15%")

print("\nClasses:")
print("10 products × 2 statuses = 20 classes")

print("\nRandom seed:")
print(SEED)

print("=" * 75)