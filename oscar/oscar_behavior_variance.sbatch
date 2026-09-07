#!/bin/bash
#SBATCH -J behavior_variance
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

cd "$REPO_ROOT/src"


MODEL="meta-llama/Llama-3.2-1B-Instruct"
# TASKS=("country-capital" "antonym" "country-currency")
TASKS=("country-capital")

for TASK in "${TASKS[@]}"; do
    echo "=== $TASK: behavior/correctness (tlens) ==="
    export PYTHONPATH="$REPO_ROOT/src"
    cd "$REPO_ROOT/src/pipeline"
    python behavior_variance.py --model_name "$MODEL" --d_name "$TASK" --prompt_type IP \
      --project_root "$REPO_ROOT" --dataset_folder "$REPO_ROOT/datasets/abstractive"
done