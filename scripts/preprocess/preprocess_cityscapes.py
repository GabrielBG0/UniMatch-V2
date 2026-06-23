#!/usr/bin/env python3
"""
Generate Cityscapes labelTrainIds PNG files from labelIds PNGs.

Input:  /petrobr/parceirosbr/spfm/datasets/image_seg/cityscapes/
          gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png
Output: /petrobr/parceirosbr/spfm/datasets/image_seg/processed/cityscapes/
          gtFine/{train,val,test}/<city>/*_gtFine_labelTrainIds.png
          leftImg8bit -> symlink to original

The split files expect:
  leftImg8bit/train/<city>/<name>_leftImg8bit.png
  gtFine/train/<city>/<name>_gtFine_labelTrainIds.png
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_DIR = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/cityscapes")
DST_DIR = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/processed/cityscapes")

NUM_WORKERS = 8

# Official Cityscapes label_id -> train_id mapping (255 = ignore/unlabeled)
LABEL_TO_TRAIN = np.full(256, 255, dtype=np.uint8)
_mapping = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}
for lid, tid in _mapping.items():
    LABEL_TO_TRAIN[lid] = tid


def convert_label(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    label_ids = np.array(Image.open(src_path))
    train_ids = LABEL_TO_TRAIN[label_ids]
    Image.fromarray(train_ids).save(dst_path)


def main():
    print("=== Cityscapes preprocessing ===")
    print(f"Source: {SRC_DIR}")
    print(f"Output: {DST_DIR}")

    # Symlink leftImg8bit so the processed dir is self-contained
    link_imgs = DST_DIR / "leftImg8bit"
    if not link_imgs.exists():
        DST_DIR.mkdir(parents=True, exist_ok=True)
        link_imgs.symlink_to(SRC_DIR / "leftImg8bit")
        print(f"  Created symlink: {link_imgs} -> {SRC_DIR / 'leftImg8bit'}")

    # Find all labelIds PNGs
    src_gtfine = SRC_DIR / "gtFine"
    label_files = sorted(src_gtfine.rglob("*_gtFine_labelIds.png"))

    if not label_files:
        sys.exit(f"No labelIds PNGs found under {src_gtfine}")

    print(f"\nFound {len(label_files)} labelIds files. Generating labelTrainIds...")

    tasks = []
    for src_path in label_files:
        rel = src_path.relative_to(SRC_DIR)
        dst_name = src_path.name.replace("_labelIds.png", "_labelTrainIds.png")
        dst_path = DST_DIR / rel.parent / dst_name
        if dst_path.exists():
            continue
        tasks.append((src_path, dst_path))

    if not tasks:
        print("All files already exist, nothing to do.")
    else:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futs = [pool.submit(convert_label, s, d) for s, d in tasks]
            for f in tqdm(as_completed(futs), total=len(futs)):
                f.result()

    print(f"\nDone. Update configs/cityscapes.yaml:")
    print(f"  data_root: {DST_DIR}")
    print(f"Or re-point the data/ symlink:")
    print(f"  ln -sfn {DST_DIR} data/cityscapes")


if __name__ == "__main__":
    main()
