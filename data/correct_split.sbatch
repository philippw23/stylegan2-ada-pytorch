#!/bin/bash
#SBATCH --job-name=correct-split
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/correct_split-%j.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/correct_split-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"
SPLIT_PATH="${SPLIT_PATH:-${REPO_DIR}/data/dataset/dataset_split.json}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data/dataset/BTXRD_resized_sorted}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

python data/correct_split.py \
  --split-path "${SPLIT_PATH}" \
  --dataset-dir "${DATASET_DIR}"
