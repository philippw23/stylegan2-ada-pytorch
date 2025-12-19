#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/vol/miltank/users/cahu/stylegan2-ada-pytorch/logs/stylegan_generate-%A.out
#SBATCH --error=/vol/miltank/users/cahu/stylegan2-ada-pytorch/logs/stylegan_generate-%A.err
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

# Set parameters for the output folder
truncation=1.0
snapshot=25000
gamma=8
class=6

# Class names
class_names=("giant_cell_tumor" "multiple_osteochondromas" "osteochondroma" "osteofibroma" "osteosarcoma" "simple_bone_cyst" "synovial_osteochondroma")

# Get the class name for the given class number
class_name=${class_names[$class]}

python generate.py \
  --outdir="/vol/miltank/users/cahu/stylegan2-ada-pytorch/generated_images/${class_name}gamma${gamma}snapshot${snapshot}trunc${truncation}" \
  --trunc=${truncation} \
  --seeds=0-99 \
  --network="/vol/miltank/users/cahu/stylegan2-ada-pytorch/checkpoints/stylegan2ada_cond/00041-btxrd_corrected_dataset-cond-mirror-auto1-gamma8-batch16-ada/network-snapshot-025000.pkl" \
  --class=${class} \