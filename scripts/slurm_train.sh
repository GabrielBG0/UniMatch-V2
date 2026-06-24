#!/bin/bash
#
# UniMatch V2 — semi-supervised semantic segmentation
# Single-node 4×H100 training via SLURM.
#
# Edit the four variables below, then submit:
#   sbatch scripts/slurm_train.sh
#
# dataset : pascal | cityscapes | ade20k | coco
# method  : unimatch_v2 | fixmatch | supervised
# exp     : arbitrary label used in the output path
# split   : see splits/<dataset>/ for available splits
#
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --job-name=unimatch_v2
#SBATCH --time=24:00:00
#SBATCH --exclude=sdumont2nd2025,sdumont2nd2046,sdumont2nd2020
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --chdir=/petrobr/parceirosbr/home/gabriel.gutierrez/github/UniMatch-V2
#SBATCH --output=logs/unimatch_v2/%x_%j.out
#SBATCH --error=logs/unimatch_v2/%x_%j.err

set -euo pipefail

# ── experiment config ────────────────────────────────────────────────────────
dataset='pascal'
method='unimatch_v2'
exp='dinov2_small'
split='366'
port=12345
# ────────────────────────────────────────────────────────────────────────────

config=configs/${dataset}.yaml
labeled_id_path=splits/${dataset}/${split}/labeled.txt
unlabeled_id_path=splits/${dataset}/${split}/unlabeled.txt
save_path=exp/${dataset}/${method}/${exp}/${split}

mkdir -p "$save_path" logs/unimatch_v2

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:        $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:       $SLURMD_NODENAME"
echo "Dataset:    $dataset  |  Method: $method  |  Split: $split"
echo "Save path:  $save_path"
echo "Started:    $(date)"
echo "========================================"

# Training auto-resumes from latest.pth if it exists (handled inside the script).
srun uv run python "$method.py" \
    --config="$config" \
    --labeled-id-path "$labeled_id_path" \
    --unlabeled-id-path "$unlabeled_id_path" \
    --save-path "$save_path" \
    --port "$port"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
