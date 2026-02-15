#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=./logs/style_gan_training-%A.out
#SBATCH --error=./logs/style_gan_training-%A.err
#SBATCH --partition=universe,asteroids 
#SBATCH --qos=master-queuesave
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


# Load Python module
source /meta/opt/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

# Activate conda env
conda activate stylegan5

# Detect GPU capability and export arch
export TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"

# Force rebuild for this arch
export TORCH_EXTENSIONS_DIR="/vol/miltank/users/wiep/.cache/torch_extensions"
rm -rf "${TORCH_EXTENSIONS_DIR}/bias_act_plugin" "${TORCH_EXTENSIONS_DIR}/upfirdn2d_plugin"

# Install dependencies
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja
conda install -y gcc_linux-64=9 gxx_linux-64=9
pip install mpmath

# Set CUDA_HOME to the root of the conda environment
export CUDA_HOME=$CONDA_PREFIX
# Force torch extensions to build in a workspace-writable cache (avoids ~/.cache symlink issues)
export TORCH_EXTENSIONS_DIR="/vol/miltank/users/wiep/.cache/torch_extensions"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
# Ensure PyTorch shared libs are discoverable for compiled extensions
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/u/home/wiep/.local/lib/python3.12/site-packages/torch/lib"
# Avoid arch autodetect failures if CUDA is not visible during build
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

# Run the program
REPO_DIR="."

# Override via env vars when submitting (e.g. `sbatch --export=ALL,BATCH=8 ...`).
DATASET_ZIP="${DATASET_ZIP:-${REPO_DIR}/data/btxrd_train_dataset.zip}"
OUTDIR="${OUTDIR:-${REPO_DIR}/checkpoints/stylegan2ada_cond_train_corrected}"
GPUS="${GPUS:-1}"
BATCH="${BATCH:-16}"
CFG="${CFG:-auto}"
SNAP="${SNAP:-50}"
MIRROR="${MIRROR:-1}"
AUG="${AUG:-ada}"
METRICS="${METRICS:-fid50k_full}"
KIMG="${KIMG:-}"
RESUME="${RESUME:-}"
SEED="${SEED:-42}"

mkdir -p "${OUTDIR}" "${REPO_DIR}/logs"

# Run the training script
# python train.py \
#     --outdir=./checkpoints/stylegan2ada_cond \
#     --data=./data/btxrd_dataset.zip \
#     --gpus="${GPUS}" \
#     --cond=1 \
#     --mirror=1 \
#     --batch=16 \
#     --gamma=0 \
#     --aug=ada

python train.py \
  --outdir="${OUTDIR}" \
  --data="${DATASET_ZIP}" \
  --gpus=1 \
  --batch="${BATCH}" \
  --gamma=6 \
  --cond=1 \
  --mirror="${MIRROR}" \
  --aug="${AUG}" \
  --cfg="${CFG}" \
  --snap="${SNAP}" \
  ${RESUME:+--resume="${RESUME}"} \
  ${KIMG:+--kimg="${KIMG}"} \
  --seed="${SEED}" \
  --metrics="${METRICS}"
