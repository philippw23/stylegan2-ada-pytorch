#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=training-%A.out
#SBATCH --error=training-%A.err
##SBATCH --partition=universe,asteroids 
##SBATCH --qos=master-queuesave
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate conda env
conda activate /meta/users/cahu/my_env

# Install dependencies
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja
conda install -y gcc_linux-64=9 gxx_linux-64=9

# Set CUDA_HOME to the root of the conda environment
export CUDA_HOME=$CONDA_PREFIX

# Run the program
REPO_DIR="/vol/miltank/users/cahu/stylegan2-ada-pytorch"

# Override via env vars when submitting (e.g. `sbatch --export=ALL,BATCH=8 ...`).
DATASET_ZIP="${DATASET_ZIP:-${REPO_DIR}/data/btxrd_dataset.zip}"
OUTDIR="${OUTDIR:-${REPO_DIR}/checkpoints/stylegan2ada_cond}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-16}"
CFG="${CFG:-auto}"
SNAP="${SNAP:-10}"
MIRROR="${MIRROR:-1}"
AUG="${AUG:-ada}"
METRICS="${METRICS:-none}"
KIMG="${KIMG:-}"
RESUME="${RESUME:-}"
SEED="${SEED:-42}"

mkdir -p "${OUTDIR}" "${REPO_DIR}/logs"

# Run the training script
python train.py --outdir=./checkpoints/stylegan2ada_cond --data=./data/btxrd_dataset.zip --cond=1 --mirror=1 --batch=16 --gamma=0 --aug=ada