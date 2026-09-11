#!/bin/bash
#SBATCH -J earliest_lth_patch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Path patching into the EARLIEST identified LTH only, from every head above
# it, in the noising direction (--direction denoising to flip). Separate
# script/output tree from pp_pipeline.py -- see earliest_lth_patch.py.

set -euo pipefail
mkdir -p logs

REPO_ROOT="$SLURM_SUBMIT_DIR"

module load python/3.11.11-5e66
source "$REPO_ROOT/.venv/bin/activate"

export PYTHONPATH="$REPO_ROOT/src"
cd "$REPO_ROOT/src/pipeline/path_patching"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
# TASKS=("country-capital" "antonym" "country-currency")
TASKS=("country-capital")

K=20                  # matches identify_heads --k
THRESHOLD=0.2         # matches lth_pipeline --threshold (shared_heads_p{THRESHOLD}_k{K}.pkl)
PP_PROMPT_INDEX=10    # EP n_shot
DIRECTION=noising      # or: denoising

for TASK in "${TASKS[@]}"; do
    echo "=== $TASK: path patching into the earliest LTH (direction=$DIRECTION) ==="
    python earliest_lth_patch.py \
        --model_name "$MODEL" \
        --d_name "$TASK" \
        --project_root "$REPO_ROOT" \
        --save_root "$REPO_ROOT/output" \
        --dataset_folder "$REPO_ROOT/datasets/abstractive" \
        --k "$K" \
        --threshold "$THRESHOLD" \
        --pp_prompt_index "$PP_PROMPT_INDEX" \
        --receiver_input q k v \
        --direction "$DIRECTION" \
        --metric l2_norm_rel l2_norm_abs lprr \
        --exp_size 25
done
