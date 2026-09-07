#!/bin/bash
#SBATCH -J identify_heads
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
cd "$REPO_ROOT/src/pipeline/identify_heads"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
# TASKS=("country-capital" "antonym" "country-currency")
TASKS=("country-capital")
SAVE_ROOT="$REPO_ROOT/output"

K=20                  # MAPS top-k; feeds shared_heads_p{P}_k{K}.pkl name
P=0.1                 # per-prompt-style threshold  <-- priority
EP_TEMPLATE_KEY=10    # EP n_shot for the receiver LTH (few-shot prompts)
N_MATCH=1
BATCH_SIZE=20

for TASK in "${TASKS[@]}"; do
    echo "=== $TASK: identify shared LTH (EP ${EP_TEMPLATE_KEY}-shot and best IP), p=${P} ==="
    python lth_pipeline.py \
        --model_name "$MODEL" \
        --d_name "$TASK" \
        --save_root "$SAVE_ROOT" \
        --project_root "$REPO_ROOT" \
        --dataset_folder "$REPO_ROOT/datasets/abstractive" \
        --ep_template_key "$EP_TEMPLATE_KEY" \
        --k "$K" \
        --threshold "$P" \
        --n_match "$N_MATCH" \
        --batch_size "$BATCH_SIZE"
done
