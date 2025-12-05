#!/bin/bash
#SBATCH --job-name=push-corrected-ds
#SBATCH --chdir=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/move_corrected_dataset-%j.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/move_corrected_dataset-%j.err
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

SRC="${SRC:-/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/btxrd_corrected_dataset.zip}"
DEST_DIR="${DEST_DIR:-/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset}"
DEST="${DEST_DIR}/$(basename "${SRC}")"

if [[ ! -f "${SRC}" ]]; then
  echo "Source file not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

if [[ -e "${DEST}" ]]; then
  echo "Destination already exists: ${DEST}" >&2
  exit 1
fi

echo "Moving ${SRC} -> ${DEST_DIR}"
mv "${SRC}" "${DEST_DIR}"
echo "Done. Final location: ${DEST}"
