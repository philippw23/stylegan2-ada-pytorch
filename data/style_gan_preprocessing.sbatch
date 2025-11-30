#!/bin/bash
#SBATCH --job-name=test-env
#SBATCH --output=test-env-%A.out
#SBATCH --error=test-env-%A.err
##SBATCH --partition=universe,asteroids 
##SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh

# Activate conda env
conda activate /meta/users/cahu/my_env


python style_gan_preprocessing.py \
    --image-dir "/vol/miltank/users/cahu/stylegan2-ada-pytorch/data/dataset/final_patched_BTXRD" \
    --json-dir "/vol/miltank/users/cahu/stylegan2-ada-pytorch/data/dataset/BTXRD/Annotations" \
    --output-dir "/vol/miltank/users/cahu/stylegan2-ada-pytorch/data/dataset/BTXRD_resized_sorted" \
    --target-size 256