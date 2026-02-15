#!/bin/bash
#SBATCH --job-name=test-py
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-${REPO_DIR}/data/test.py}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
"${PYTHON_BIN}" "${SCRIPT_PATH}"
