#!/bin/bash
#SBATCH -J path_patching
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

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

K=20                 
THRESHOLD=0.2      
PP_PROMPT_INDEX=10

for TASK in "${TASKS[@]}"; do
    echo "=== $TASK: path patching (sender -> shared lexical-task heads) ==="
    python pp_pipeline.py \
        --model_name "$MODEL" \
        --d_name "$TASK" \
        --project_root "$REPO_ROOT" \
        --save_root "$REPO_ROOT/output" \
        --dataset_folder "$REPO_ROOT/datasets/abstractive" \
        --k "$K" \
        --threshold "$THRESHOLD" \
        --pp_prompt_index "$PP_PROMPT_INDEX" \
        --receiver_input q \
        --receiver_key ep_heads \
        --metric lprr \
        --exp_size 25 \
        --batch_size 16
done
