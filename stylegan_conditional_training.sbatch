#!/bin/bash
#SBATCH --job-name=stylegan_cond
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/stylegan_cond-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/stylegan_cond-%A.err
##SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Load python module
source ~/.bashrc
ml python/anaconda3

# Activate corresponding environment
# If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. The following guards against that.
#Not necessary if you always run this script from a clean terminal
#conda deactivate
# If the following does not work, try 'source activate <env-name>'
eval "$(conda shell.bash hook)"
conda activate stylegan


# Run the program
REPO_DIR="/vol/miltank/users/wiep/Documents/bone-tumour-classification"
# cd "${REPO_DIR}/stylegan2-ada-pytorch"

# Override via env vars when submitting (e.g. `sbatch --export=ALL,BATCH=8 ...`).
DATASET_ZIP="${DATASET_ZIP:-${REPO_DIR}/data/btxrd_dataset.zip}"
OUTDIR="${OUTDIR:-${REPO_DIR}/checkpoints/stylegan2ada_cond3}"
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

python train.py \
  --outdir="${OUTDIR}" \
  --data="${DATASET_ZIP}" \
  --gpus="${GPUS}" \
  --batch="${BATCH}" \
  --cfg="${CFG}" \
  --snap="${SNAP}" \
  --mirror="${MIRROR}" \
  --cond=1 \
  --seed="${SEED}" \
  --aug="${AUG}" \
  --metrics="${METRICS}" \
  ${KIMG:+--kimg="${KIMG}"} \
  ${RESUME:+--resume="${RESUME}"}