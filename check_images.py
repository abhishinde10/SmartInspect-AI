from pathlib import Path
from PIL import Image

DATASET_DIR = Path("dataset")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

total = 0
valid = 0
corrupted = 0
invalid = []

print("\nSMARTINSPECT AI - IMAGE VALIDATION")
print("=" * 60)

for product_dir in sorted(DATASET_DIR.iterdir()):

    if not product_dir.is_dir():
        continue

    for status_dir in ["good", "bad"]:

        folder = product_dir / status_dir

        if not folder.exists():
            continue

        for image_path in folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total += 1

            try:
                with Image.open(image_path) as img:
                    img.verify()

                valid += 1

            except Exception:
                corrupted += 1
                invalid.append(str(image_path))

print(f"Total images checked : {total}")
print(f"Valid images         : {valid}")
print(f"Corrupted images     : {corrupted}")

print("=" * 60)

if corrupted > 0:

    print("\nCORRUPTED FILES:")
    
    for file in invalid:
        print(file)

else:
    print("\nAll images are valid.")