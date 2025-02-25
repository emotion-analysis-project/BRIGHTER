#!/bin/bash -l
#SBATCH --job-name=track_ab_lm_array
#SBATCH --account=etechnik_gpu
#SBATCH --time=2-08:00:00
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH --array=0-5

# Display GPU info
nvidia-smi

# Define the array of model names.
MODELS=(
    "google-bert/bert-base-multilingual-cased"
    "FacebookAI/xlm-roberta-large"
    "microsoft/mdeberta-v3-base"
    "sentence-transformers/LaBSE"
    "microsoft/infoxlm-large"
    "google/rembert"
)

# Select the current model based on SLURM_ARRAY_TASK_ID.
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "Evaluating model: ${MODEL}"

# Define the tasks in an array (first binary, then intensity).
TASKS=("binary" "intensity")

# Loop over tasks
for TASK in "${TASKS[@]}"; do
  
  if [ "$TASK" = "binary" ]; then
    TRAIN_DIR="./track_a/train"
    TEST_DIR="./track_a/test"
    OUT_DIR_SUFFIX="_binary"
  else
    TRAIN_DIR="./track_b/train"
    TEST_DIR="./track_b/test"
    OUT_DIR_SUFFIX="_intensity"
  fi

  # Loop over all CSV files in the train directory
  for TRAIN_FILE in "${TRAIN_DIR}"/*.csv; do
    # Extract language code from filename, e.g. eng.csv -> eng
    LANG=$(basename "${TRAIN_FILE}" .csv)
    TEST_FILE="${TEST_DIR}/${LANG}.csv"

    # Construct an output directory for each language
    OUTPUT_DIR="./lm_track_ab_results/${MODEL//\//_}${OUT_DIR_SUFFIX}/${LANG}"

    echo "---------------------------------------"
    echo "Task: ${TASK}"
    echo "Language: ${LANG}"
    echo "Train file: ${TRAIN_FILE}"
    echo "Test file: ${TEST_FILE}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "---------------------------------------"

    # Create the directory if it doesn't exist
    mkdir -p "${OUTPUT_DIR}"

    # Run the evaluation
    uv run python regular_lms_track_ab.py \
        --model_name "${MODEL}" \
        --task "${TASK}" \
        --train_filepath "${TRAIN_FILE}" \
        --test_filepath "${TEST_FILE}" \
        --output_json_path "${OUTPUT_DIR}/${LANG}_results.json"

  done
done
