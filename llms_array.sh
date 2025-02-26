#!/bin/bash -l
#SBATCH --job-name=track_ab_llm_array
#SBATCH --time=3-00:00:00
#SBATCH --gpus=8
#SBATCH -N 1
#SBATCH --array=0-4

# Optional: check GPU visibility
nvidia-smi

# -- 1) Define your model list, one per line --
MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "databricks/dolly-v2-12b"
)

# -- 2) SLURM array index picks which model to run --
IDX=$SLURM_ARRAY_TASK_ID
MODEL="${MODELS[$IDX]}"

echo "Starting array job for model = $MODEL"

# -- 3) Make an output directory if it doesn't exist --
OUTPATH="llm_track_ab_results"
mkdir -p $OUTPATH

# -- 4) Define the tasks and languages you want to evaluate --
TASKS=("binary")
# TASKS=("binary" "intensity")

# -- 5) For each combination of (task, language), run llms_single.py
for TASK in "${TASKS[@]}"; do
  SAFE_MODEL=${MODEL//\//_}

  echo "Running: Model=$MODEL | Task=$TASK"

  # Use either "srun" or just directly "python" (depending on environment)
  # Some HPC setups require "srun" or "mpirun"; adapt as needed:
  uv run python llms.py \
    --model_name "$MODEL" \
    --task "$TASK" \
    --tensor_parallel_size 8 \
    --skip_existing
done

echo "Done with all tasks & langs for model=$MODEL"
