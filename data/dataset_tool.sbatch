#!/bin/bash
#SBATCH --job-name=dataset-tool
#SBATCH --chdir=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/dataset_tool-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/dataset_tool-%A.err
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Ensure Python deps are available
python -m pip install --user pillow click tqdm numpy

REPO_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch"

# Override via env vars when submitting, e.g.:
# sbatch --export=ALL,SOURCE=/path/in,DEST=/path/out.zip data/dataset_tool.sbatch
SOURCE="${SOURCE:-${REPO_DIR}/data/dataset/BTXRD_resized_sorted}"
DEST="${DEST:-${REPO_DIR}/data/btxrd_corrected_dataset.zip}"
MAX_IMAGES="${MAX_IMAGES:-}"
WIDTH="${WIDTH:-256}"
HEIGHT="${HEIGHT:-256}"
TRANSFORM="${TRANSFORM:-}"
RESIZE_FILTER="${RESIZE_FILTER:-lanczos}"

python "${REPO_DIR}/data/dataset_tool.py" --source="${SOURCE}" --dest="${DEST}" \
    --width="${WIDTH}" --height="${HEIGHT}" --resize-filter=box
    
# cmd=(python data/dataset_tool.py --source "$SOURCE" --dest "$DEST" --resize-filter "$RESIZE_FILTER")
# [[ -n "$MAX_IMAGES" ]] && cmd+=(--max-images "$MAX_IMAGES")
# [[ -n "$WIDTH" ]] && cmd+=(--width "$WIDTH")
# [[ -n "$HEIGHT" ]] && cmd+=(--height "$HEIGHT")
# [[ -n "$TRANSFORM" ]] && cmd+=(--transform "$TRANSFORM")

# echo "Running: ${cmd[*]}"
# "${cmd[@]}"
