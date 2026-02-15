#!/bin/bash
#SBATCH --job-name=move-dataset
#SBATCH --output=logs/move-dataset-%j.out
#SBATCH --error=logs/move-dataset-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Default paths; override via --export if needed.
SRC="${SRC:-/vol/miltank/projects/practical_wise2526/bone-tumor-classification-gen-models/dataset}"
DEST_ROOT="${DEST_ROOT:-/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data}"

if [[ ! -d "${SRC}" ]]; then
  echo "Source directory not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "${DEST_ROOT}"

DEST_PATH="${DEST_ROOT}/$(basename "${SRC}")"

if [[ -e "${DEST_PATH}" ]]; then
  echo "Destination already exists: ${DEST_PATH}" >&2
  exit 1
fi

echo "Moving ${SRC} -> ${DEST_ROOT}"
mv "${SRC}" "${DEST_ROOT}"
echo "Done. Final location: ${DEST_PATH}"
