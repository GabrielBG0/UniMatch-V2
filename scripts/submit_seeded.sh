#!/bin/bash
#
# Submit all seeded experiments that mirror the existing runs under exp/.
# Usage: bash scripts/submit_seeded.sh
#
# Results land in exp_seeded/<dataset>/<method>/<exp>/<split>/seed<seed>/

set -euo pipefail

SEED=0
METHOD=unimatch_v2
SCRIPT=scripts/slurm_train.sh

submit() {
    local dataset=$1 exp=$2 split=$3
    echo "Submitting: dataset=$dataset  exp=$exp  split=$split  seed=$SEED"
    sbatch --export=ALL,DATASET="$dataset",METHOD="$METHOD",EXP="$exp",SPLIT="$split",SEED="$SEED" \
           --job-name="${METHOD}_${dataset}_${exp}_${split}_s${SEED}" \
           "$SCRIPT"
}

# ── cityscapes ───────────────────────────────────────────────────────────────
submit cityscapes dinov2_small 1_2
submit cityscapes dinov2_small 1_4
submit cityscapes dinov2_small 1_8
submit cityscapes dinov2_small 1_16
submit cityscapes dinov2_small 1_30
submit cityscapes dinov2_base  1_8

# ── ade20k ───────────────────────────────────────────────────────────────────
submit ade20k dinov2_small 1_2
submit ade20k dinov2_small 1_4
submit ade20k dinov2_small 1_8
submit ade20k dinov2_small 1_16
submit ade20k dinov2_small 1_32
submit ade20k dinov2_small 1_64
submit ade20k dinov2_base  1_8

# ── coco ─────────────────────────────────────────────────────────────────────
submit coco dinov2_small 1_32
submit coco dinov2_small 1_64
submit coco dinov2_small 1_128
submit coco dinov2_small 1_256
submit coco dinov2_small 1_512
submit coco dinov2_base  1_128

echo ""
echo "All 19 jobs submitted. Check with: squeue -u \$USER"
