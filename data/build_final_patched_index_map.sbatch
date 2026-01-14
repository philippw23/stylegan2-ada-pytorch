#!/bin/bash
#SBATCH --job-name=build-index-map
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/build_index_map-%j.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/build_index_map-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"
SPLIT_PATH="${SPLIT_PATH:-${REPO_DIR}/data/dataset/dataset_split.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data/dataset/final_patched_BTXRD}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_DIR}/data/dataset/final_patched_index_map.json}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

python data/build_final_patched_index_map.py \
  --split-path "${SPLIT_PATH}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-path "${OUTPUT_PATH}"
