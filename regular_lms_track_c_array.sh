#!/bin/bash -l
#SBATCH --job-name=track_c_lm_array
#SBATCH --time=2-08:00:00
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --array=0-5

# Display GPU info for debugging
nvidia-smi

# Define the array of model names to run
MODELS=(
    "google-bert/bert-base-multilingual-cased"
    "FacebookAI/xlm-roberta-large"
    "microsoft/mdeberta-v3-base"
    "sentence-transformers/LaBSE"
    "microsoft/infoxlm-large"
    "google/rembert"
)

# Select the current model based on the array index
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "========================================================="
echo "Running family leave-one-out with model: ${MODEL}"
echo "========================================================="

# If you use a virtual environment, activate it here:
# source /path/to/venv/bin/activate

# Run the training script
uv run python regular_lms_track_c.py \
    --model_name "${MODEL}" \
    --train_dir "./track_a/train" \
    --test_dir "./track_a/test" \
    --output_dir "./lm_track_c_results" \
    --num_epochs 2 \
    --learning_rate 1e-5
