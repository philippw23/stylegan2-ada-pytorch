#!/bin/bash
#SBATCH --job-name=stylegan-preprocess
#SBATCH --chdir=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/style_gan_preprocessing-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/style_gan_preprocessing-%A.err
##SBATCH --partition=universe,asteroids 
##SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


LOG_DIR="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs"
mkdir -p "$LOG_DIR"

# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh

# Activate conda env
conda activate stylegan5

# Ensure Pillow is available (required for preprocessing)
python -m pip install --user pillow

python data/style_gan_preprocessing.py \
    --image-dir "/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/dataset/final_patched_BTXRD" \
    --json-dir "/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/dataset/BTXRD/Annotations" \
    --output-dir "/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/dataset/BTXRD_resized_sorted" \
    --target-size 256
