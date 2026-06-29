#!/bin/bash
#
# UniMatch V2 — DINOv2-Base validation sweep.
# Runs one representative split per dataset (cityscapes 1/8, ade20k 1/8, coco 1/128)
# to cross-check paper results against the small-backbone sweep.
#
# Submit:
#   sbatch scripts/slurm_sweep_base.sh
#
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --job-name=unimatch_v2_base
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-2
#SBATCH --chdir=/petrobr/parceirosbr/home/gabriel.gutierrez/github/UniMatch-V2
#SBATCH --output=logs/unimatch_v2_base/%x_%A_%a.out
#SBATCH --error=logs/unimatch_v2_base/%x_%A_%a.err

set -euo pipefail

method='unimatch_v2'
exp='dinov2_base'

# Task 0: cityscapes 1/8
# Task 1: ade20k    1/8
# Task 2: coco      1/128
TASKS=(
    "cityscapes 1_8"
    "ade20k     1_8"
    "coco       1_128"
)

read -r dataset split <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

config=configs/${dataset}_base.yaml
labeled_id_path=splits/${dataset}/${split}/labeled.txt
unlabeled_id_path=splits/${dataset}/${split}/unlabeled.txt
save_path=exp/${dataset}/${method}/${exp}/${split}
port=$((12345 + SLURM_ARRAY_TASK_ID))

mkdir -p "$save_path" logs/unimatch_v2_base

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:        $SLURM_JOB_NAME (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)"
echo "Node:       $SLURMD_NODENAME"
echo "Dataset:    $dataset  |  Split: $split  |  Backbone: dinov2_base"
echo "Save path:  $save_path"
echo "Started:    $(date)"
echo "========================================"

# Skip if already complete
if [ -f "${save_path}/best.pth" ] && [ ! -f "${save_path}/latest.pth" ]; then
    echo "Already complete, skipping."
    exit 0
fi

srun uv run python "${method}.py" \
    --config="$config" \
    --labeled-id-path "$labeled_id_path" \
    --unlabeled-id-path "$unlabeled_id_path" \
    --save-path "$save_path" \
    --port "$port"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
