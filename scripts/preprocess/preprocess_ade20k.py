#!/usr/bin/env python3
"""
Set up ADE20K-150 dataset for UniMatch V2 (nclass=150).

The HuggingFace parquet on disk is ADE20K-2021 Full (3688 categories).
The split files and config expect the ADE20K-150 scene parsing benchmark from
MIT CSAIL (ADEChallengeData2016), whose annotation PNGs already contain
0-indexed class labels 0-149 (0=background, 150 classes total).

This script downloads ADEChallengeData2016.zip (~953 MB) from MIT CSAIL and
extracts it into the processed output directory.

Output: /petrobr/parceirosbr/spfm/datasets/image_seg/processed/ade20k/
          images/training/*.jpg           (20210 images)
          images/validation/*.jpg         (2000 images)
          annotations/training/*.png      (uint8, 0-indexed class labels)
          annotations/validation/*.png
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

DST_DIR = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/processed/ade20k")
ZIP_CACHE = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/ade20k")

DOWNLOAD_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ZIP_FILE = ZIP_CACHE / "ADEChallengeData2016.zip"
# Extracted subdirectory inside the zip
ZIP_INNER = "ADEChallengeData2016"


def download_zip():
    if ZIP_FILE.exists():
        print(f"  Found cached zip: {ZIP_FILE}")
        return
    ZIP_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ADE20K-150 (~953 MB) from MIT CSAIL...")
    print(f"  URL : {DOWNLOAD_URL}")
    print(f"  Dest: {ZIP_FILE}")
    print()

    def progress(count, block_size, total):
        if total > 0:
            pct = min(count * block_size / total * 100, 100)
            mb = count * block_size / 1e6
            print(f"\r  {mb:.0f} MB  ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_FILE, reporthook=progress)
    print(f"\n  Download complete.")


def extract_zip():
    sentinel = DST_DIR / "images" / "training"
    if sentinel.exists() and any(sentinel.glob("*.jpg")):
        print(f"  Already extracted: {DST_DIR}")
        return

    print(f"Extracting {ZIP_FILE.name} → {DST_DIR} ...")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_FILE) as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            if i % 1000 == 0:
                print(f"\r  {i}/{total} files", end="", flush=True)
            # Strip the leading ADEChallengeData2016/ prefix
            rel = member[len(ZIP_INNER) + 1:]
            if not rel:
                continue
            dst = DST_DIR / rel
            if member.endswith("/"):
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(zf.read(member))
    print(f"\n  Extraction complete.")


def verify():
    train_imgs = list((DST_DIR / "images" / "training").glob("*.jpg"))
    val_imgs = list((DST_DIR / "images" / "validation").glob("*.jpg"))
    train_anns = list((DST_DIR / "annotations" / "training").glob("*.png"))
    val_anns = list((DST_DIR / "annotations" / "validation").glob("*.png"))
    print(f"\nVerification:")
    print(f"  images/training:      {len(train_imgs):6d}  (expected 20210)")
    print(f"  images/validation:    {len(val_imgs):6d}  (expected  2000)")
    print(f"  annotations/training: {len(train_anns):6d}  (expected 20210)")
    print(f"  annotations/validation:{len(val_anns):5d}  (expected  2000)")


def main():
    print("=== ADE20K-150 setup ===")
    print(f"Output: {DST_DIR}")
    print()
    download_zip()
    extract_zip()
    verify()
    print(f"\nDone. Re-point the data/ symlink:")
    print(f"  ln -sfn {DST_DIR} data/ade20k")


if __name__ == "__main__":
    main()
