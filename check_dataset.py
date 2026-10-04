from pathlib import Path

DATASET_DIR = Path("dataset")

valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

print("\nSMARTINSPECT AI - DATASET SUMMARY")
print("=" * 60)

total_good = 0
total_bad = 0

for product_dir in sorted(DATASET_DIR.iterdir()):

    if not product_dir.is_dir():
        continue

    good_dir = product_dir / "good"
    bad_dir = product_dir / "bad"

    good_count = 0
    bad_count = 0

    if good_dir.exists():
        good_count = sum(
            1 for f in good_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        )

    if bad_dir.exists():
        bad_count = sum(
            1 for f in bad_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        )

    total_good += good_count
    total_bad += bad_count

    print(f"{product_dir.name:15} Good: {good_count:4}   Bad: {bad_count:4}")

print("=" * 60)
print(f"TOTAL           Good: {total_good:4}   Bad: {total_bad:4}")
print(f"TOTAL IMAGES:   {total_good + total_bad}")