#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
module purge 2>/dev/null || true  # falls kein modules, egal

nvidia-smi
which nvcc && nvcc --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.cuda.is_available()", torch.cuda.is_available())
PY
