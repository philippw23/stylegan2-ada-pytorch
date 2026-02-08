#!/bin/bash
#SBATCH --job-name=fid50k
#SBATCH --output=fid50k-%j.out
#SBATCH --error=fid50k-%j.err
#SBATCH --partition=asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

DATA_ZIP=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/data/btxrd_train_anatomical_dataset.zip
NETWORK_PKL=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/checkpoints/stylegan2ada_cond_train_corrected/00000-btxrd_train_anatomical_dataset-cond-mirror-auto1-gamma6-batch16-ada/network-snapshot-015800.pkl
ENV_NAME=stylegan5

cd /vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch || exit 1

ml purge
ml python/anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python calc_metrics.py \
  --metrics=fid50k_full \
  --data="$DATA_ZIP" \
  --network="$NETWORK_PKL"
