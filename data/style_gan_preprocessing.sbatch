#!/bin/bash
#SBATCH --job-name=stylegan-preprocess
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/style_gan_preprocessing-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/style_gan_preprocessing-%A.err
##SBATCH --partition=universe,asteroids 
##SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
IMAGE_DIR="${IMAGE_DIR:-${REPO_DIR}/data/dataset/final_patched_BTXRD}"
JSON_DIR="${JSON_DIR:-${REPO_DIR}/data/dataset/BTXRD/Annotations}"
PREPROCESS_OUTPUT_DIR="${PREPROCESS_OUTPUT_DIR:-${REPO_DIR}/data/dataset/BTXRD_anatomical_resized_sorted}"
SPLIT_PATH="${SPLIT_PATH:-${REPO_DIR}/data/dataset/dataset_split_final.json}"
INDEX_MAP_DATASET_DIR="${INDEX_MAP_DATASET_DIR:-${REPO_DIR}/data/dataset/final_patched_BTXRD}"
INDEX_MAP_OUTPUT_PATH="${INDEX_MAP_OUTPUT_PATH:-${REPO_DIR}/data/dataset/final_patched_index_map.json}"
CORRECT_SPLIT_DATASET_DIR="${CORRECT_SPLIT_DATASET_DIR:-${PREPROCESS_OUTPUT_DIR}}"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh

# Activate conda env
conda activate preproc

# Ensure Pillow is available (required for preprocessing)
python -m pip install --user pillow

python data/style_gan_preprocessing.py full-pipeline \
    --image-dir "${IMAGE_DIR}" \
    --json-dir "${JSON_DIR}" \
    --preprocess-output-dir "${PREPROCESS_OUTPUT_DIR}" \
    --target-size 256 \
    --use-anatomical-location \
    --split-path "${SPLIT_PATH}" \
    --index-map-dataset-dir "${INDEX_MAP_DATASET_DIR}" \
    --index-map-output-path "${INDEX_MAP_OUTPUT_PATH}" \
    --correct-split-dataset-dir "${CORRECT_SPLIT_DATASET_DIR}"
