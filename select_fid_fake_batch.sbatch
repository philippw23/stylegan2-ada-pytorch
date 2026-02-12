#!/bin/bash
#SBATCH --job-name=select-fid-fake-batch
#SBATCH --output=select_fid_fake_batch-%j.out
#SBATCH --error=select_fid_fake_batch-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

cd /vol/miltank/users/wiep/Documents

ml purge
ml python/anaconda3

echo "Running on $(hostname)"
python stylegan2-ada-pytorch/select_fid_fake_batch.py \
  --src-base stylegan2-ada-pytorch \
  --src-glob "generated_images_020000" \
  --dst-base tmp \
  --dst-prefix "fid_fake_flat_" \
  --total 1000
