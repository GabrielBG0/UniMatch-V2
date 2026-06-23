#!/usr/bin/env python3
"""
Generate COCO semantic segmentation masks from COCO instances annotations.

Downloads annotations_trainval2017.zip (~241 MB) if not present, then renders
per-image PNG masks for train2017 and val2017.

Output: /petrobr/parceirosbr/spfm/datasets/image_seg/processed/coco/
          masks/<image_id>.png   (uint8, 0=background, 1-80=COCO class, 255=crowd/ignore)
          train2017 -> symlink to original images
          val2017   -> symlink to original images

The split files reference:
  train2017/<image_id>.jpg
  masks/<image_id>.png

Config nclass=81 means 80 thing classes + background (index 0).
COCO category IDs (non-contiguous 1-90) are remapped to contiguous 1-80.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_DIR = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/coco")
DST_DIR = Path("/petrobr/parceirosbr/spfm/datasets/image_seg/processed/coco")

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP = SRC_DIR / "zips" / "annotations_trainval2017.zip"
ANNOTATIONS_DIR = SRC_DIR / "annotations"

NUM_WORKERS = 8

# Official COCO 80 thing category IDs (non-contiguous), in order → maps to train_id 1-80
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
CAT_TO_TRAIN = {cat_id: i + 1 for i, cat_id in enumerate(COCO_CAT_IDS)}


def download_annotations():
    if ANNOTATIONS_DIR.exists() and list(ANNOTATIONS_DIR.glob("instances_*.json")):
        return
    if not ANNOTATIONS_ZIP.exists():
        print(f"Downloading COCO annotations (~241 MB)...")
        print(f"  {ANNOTATIONS_URL}")
        print(f"  -> {ANNOTATIONS_ZIP}")
        SRC_DIR.joinpath("zips").mkdir(parents=True, exist_ok=True)

        def progress(count, block_size, total):
            mb_done = count * block_size / 1e6
            mb_total = total / 1e6
            print(f"\r  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(ANNOTATIONS_URL, ANNOTATIONS_ZIP, reporthook=progress)
        print()

    print(f"Extracting {ANNOTATIONS_ZIP.name}...")
    import zipfile
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as zf:
        zf.extractall(SRC_DIR)
    print(f"  Extracted to {SRC_DIR / 'annotations'}")


def build_masks_for_split(ann_json: Path, split: str, dst_masks: Path, img_dir: Path):
    print(f"\nLoading {ann_json.name}...")
    with open(ann_json) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Decode RLE or polygon segmentation to binary mask
    try:
        from pycocotools import mask as mask_util
        use_pycocotools = True
    except ImportError:
        use_pycocotools = False

    def rle_to_mask(rle, h, w):
        if use_pycocotools:
            return mask_util.decode(rle).astype(bool)
        # Manual RLE decode (COCO uncompressed RLE)
        counts = rle["counts"]
        if isinstance(counts, str):
            # compressed RLE — requires pycocotools
            raise RuntimeError(
                "pycocotools required for compressed RLE masks. "
                "Install with: pip install pycocotools"
            )
        flat = np.zeros(h * w, dtype=bool)
        pos, is_fg = 0, False
        for c in counts:
            if is_fg:
                flat[pos:pos + c] = True
            pos += c
            is_fg = not is_fg
        return flat.reshape(h, w, order="F")

    def poly_to_mask(poly, h, w):
        from PIL import ImageDraw
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        for p in poly:
            coords = list(zip(p[0::2], p[1::2]))
            if len(coords) >= 3:
                draw.polygon(coords, fill=1)
        return np.array(img, dtype=bool)

    def render_mask(img_id):
        info = img_info[img_id]
        h, w = info["height"], info["width"]
        canvas = np.zeros((h, w), dtype=np.uint8)  # 0 = background
        anns = sorted(anns_by_img.get(img_id, []), key=lambda a: a["area"], reverse=True)
        for ann in anns:
            train_id = CAT_TO_TRAIN.get(ann["category_id"])
            if train_id is None:
                continue
            seg = ann["segmentation"]
            try:
                if isinstance(seg, dict):  # RLE
                    bin_mask = rle_to_mask(seg, h, w)
                elif isinstance(seg, list) and seg:  # polygon
                    bin_mask = poly_to_mask(seg, h, w)
                else:
                    continue
            except Exception:
                continue
            fill = 255 if ann.get("iscrowd", 0) else train_id
            canvas[bin_mask] = fill
        stem = str(img_id).zfill(12)
        Image.fromarray(canvas).save(dst_masks / f"{stem}.png")

    image_ids = list(img_info.keys())
    print(f"  Rendering {len(image_ids)} masks for {split}...")

    pending = [img_id for img_id in image_ids
               if not (dst_masks / f"{str(img_id).zfill(12)}.png").exists()]
    if not pending:
        print("  All masks already exist.")
        return

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futs = [pool.submit(render_mask, img_id) for img_id in pending]
        for f in tqdm(as_completed(futs), total=len(futs), desc=split):
            f.result()


def main():
    print("=== COCO preprocessing ===")
    print(f"Source: {SRC_DIR}")
    print(f"Output: {DST_DIR}")

    download_annotations()

    dst_masks = DST_DIR / "masks"
    dst_masks.mkdir(parents=True, exist_ok=True)

    # Symlink images so the processed dir is self-contained
    for split in ("train2017", "val2017"):
        link = DST_DIR / split
        if not link.exists():
            link.symlink_to(SRC_DIR / split)
            print(f"  Created symlink: {link} -> {SRC_DIR / split}")

    ann_dir = ANNOTATIONS_DIR
    for ann_file, split in [
        (ann_dir / "instances_train2017.json", "train2017"),
        (ann_dir / "instances_val2017.json", "val2017"),
    ]:
        if not ann_file.exists():
            print(f"WARNING: {ann_file} not found, skipping {split}")
            continue
        build_masks_for_split(ann_file, split, dst_masks, SRC_DIR / split)

    print(f"\nDone. Update configs/coco.yaml:")
    print(f"  data_root: {DST_DIR}")
    print(f"Or re-point the data/ symlink:")
    print(f"  ln -sfn {DST_DIR} data/coco")


if __name__ == "__main__":
    main()
