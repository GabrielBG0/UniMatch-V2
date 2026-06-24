#!/bin/bash
#
# UniMatch V2 — job array sweep over all (dataset, split) combinations.
# Each array task is one independent 4×H100 job; all queue simultaneously.
#
# Submit:
#   sbatch scripts/slurm_sweep.sh
#
# To run a subset (e.g. only tasks 0-4):
#   sbatch --array=0-4 scripts/slurm_sweep.sh
#
#SBATCH --partition=ict-h100
#SBATCH --account=spfm
#SBATCH --job-name=unimatch_v2_sweep
#SBATCH --time=24:00:00
#SBATCH --exclude=sdumont2nd2025,sdumont2nd2046,sdumont2nd2020
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --array=0-15
#SBATCH --chdir=/petrobr/parceirosbr/home/gabriel.gutierrez/github/UniMatch-V2
#SBATCH --output=logs/unimatch_v2_sweep/%x_%A_%a.out
#SBATCH --error=logs/unimatch_v2_sweep/%x_%A_%a.err

set -euo pipefail

# ── experiment config ────────────────────────────────────────────────────────
method='unimatch_v2'
exp='dinov2_small'
# ────────────────────────────────────────────────────────────────────────────

# Build ordered task list: "dataset split"
# "all" splits are for the supervised baseline — excluded here.
# pascal is commented out until the dataset is downloaded.
TASKS=()

# cityscapes — 5 tasks (indices 0-4)
for split in 1_2 1_4 1_8 1_16 1_30; do
    TASKS+=("cityscapes $split")
done

# ade20k — 6 tasks (indices 5-10)
for split in 1_2 1_4 1_8 1_16 1_32 1_64; do
    TASKS+=("ade20k $split")
done

# coco — 5 tasks (indices 11-15)
for split in 1_32 1_64 1_128 1_256 1_512; do
    TASKS+=("coco $split")
done

# pascal — 5 tasks (indices 16-20); uncomment when dataset is available
# Update --array above to 0-20 when adding pascal.
# for split in 92 183 366 732 1464; do
#     TASKS+=("pascal $split")
# done

N_TASKS=${#TASKS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TASKS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} >= ${N_TASKS} — nothing to do."
    exit 0
fi

read -r dataset split <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

config=configs/${dataset}.yaml
labeled_id_path=splits/${dataset}/${split}/labeled.txt
unlabeled_id_path=splits/${dataset}/${split}/unlabeled.txt
save_path=exp/${dataset}/${method}/${exp}/${split}
# Offset port per task to avoid collisions when tasks share a node
port=$((12345 + SLURM_ARRAY_TASK_ID))

mkdir -p "$save_path" logs/unimatch_v2_sweep

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:        $SLURM_JOB_NAME (array $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID)"
echo "Node:       $SLURMD_NODENAME"
echo "Dataset:    $dataset  |  Split: $split  |  Method: $method"
echo "Save path:  $save_path"
echo "Started:    $(date)"
echo "========================================"

# Skip if already complete (safe on accidental resubmit)
if [ -f "${save_path}/best.pth" ] && [ ! -f "${save_path}/latest.pth" ]; then
    echo "best.pth exists and no latest.pth — run already complete, skipping."
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
