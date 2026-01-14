#!/bin/bash
#SBATCH --job-name=correct-split
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/correct_split-%j.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/correct_split-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"
SPLIT_PATH="${SPLIT_PATH:-${REPO_DIR}/data/dataset/dataset_split_final.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data/dataset/BTXRD_anatomical_resized_sorted}"
INDEX_MAP="${INDEX_MAP:-${REPO_DIR}/data/dataset/final_patched_index_map.json}"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

python data/correct_split_new.py \
  --split-path "${SPLIT_PATH}" \
  --dataset-dir "${DATASET_DIR}" \
  --index-map "${INDEX_MAP}" \
  #$( [ "${DRY_RUN}" = "1" ] && printf "%s" "--dry-run" )
