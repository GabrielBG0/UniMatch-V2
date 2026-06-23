#!/bin/bash
# Run from the repo root: sh scripts/preprocess/run_all.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROCESSED="/petrobr/parceirosbr/spfm/datasets/image_seg/processed"
LOG_DIR="$PROCESSED"

cd "$REPO_ROOT"
mkdir -p "$PROCESSED"

echo "=== Cityscapes (already done, skipping if output exists) ==="
python3 scripts/preprocess/preprocess_cityscapes.py

echo ""
echo "=== ADE20K ==="
python3 scripts/preprocess/preprocess_ade20k.py 2>&1 | tee "$LOG_DIR/ade20k_preprocess.log"

echo ""
echo "=== COCO ==="
python3 scripts/preprocess/preprocess_coco.py 2>&1 | tee "$LOG_DIR/coco_preprocess.log"

echo ""
echo "=== Re-pointing data/ symlinks ==="
ln -sfn "$PROCESSED/cityscapes" data/cityscapes
ln -sfn "$PROCESSED/ade20k"     data/ade20k
ln -sfn "$PROCESSED/coco"       data/coco

echo "All done. Symlinks:"
ls -la data/
