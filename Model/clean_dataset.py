"""
Scans the dataset folder for corrupt or unreadable images and deletes them.
Run this BEFORE training to fix crashes that happen during validation/evaluate.

Usage:
    python Model/clean_dataset.py
"""

import os
from PIL import Image

DATASET_PATH = "dataset/plantvillage dataset/color"

def scan_and_clean(dataset_path):
    total_checked = 0
    total_removed = 0
    removed_files = []

    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            total_checked += 1

            # Skip obviously non-image files
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"Removing non-image file: {fpath}")
                os.remove(fpath)
                total_removed += 1
                removed_files.append(fpath)
                continue

            try:
                with Image.open(fpath) as img:
                    img.verify()  # checks file integrity without fully decoding
                # re-open and actually load pixel data (verify() alone can miss some issues)
                with Image.open(fpath) as img:
                    img.convert('RGB').load()
            except Exception as e:
                print(f"Corrupt image found and removed: {fpath} ({e})")
                os.remove(fpath)
                total_removed += 1
                removed_files.append(fpath)

            if total_checked % 2000 == 0:
                print(f"...checked {total_checked} files so far")

    print(f"\nDone. Checked {total_checked} files, removed {total_removed} corrupt/invalid files.")
    if removed_files:
        print("\nRemoved files:")
        for f in removed_files:
            print(f"  - {f}")

if __name__ == '__main__':
    scan_and_clean(DATASET_PATH)