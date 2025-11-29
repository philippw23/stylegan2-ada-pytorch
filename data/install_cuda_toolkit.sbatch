#!/bin/bash
#SBATCH --job-name=install_cuda_toolkit
#SBATCH --output=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/install_cuda_toolkit-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/bone-tumour-classification/logs/install_cuda_toolkit-%A.err
#SBATCH --partition=universe
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

# Activate conda and the stylegan environment.
if [ -f "/meta/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/meta/opt/anaconda3/etc/profile.d/conda.sh"
fi
conda deactivate || true
conda activate /meta/users/wiep/envs/stylegan

# Install CUDA toolkit and nvcc from conda-forge (matching torch 1.7.1+cu110).
conda install -y -c conda-forge cudatoolkit-dev=11.0 nvcc_linux-64=11.0

echo "Done. Installed cudatoolkit-dev=11.0 and nvcc_linux-64=11.0 into ${CONDA_PREFIX}."
