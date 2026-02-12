#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --output=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/stylegan_generate-%A.out
#SBATCH --error=/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/logs/stylegan_generate-%A.err
#SBATCH --partition=asteroids 
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
snapshot=020000
gamma=6
# Classes to generate (inclusive indices into class_names array)
start_class=0

# If you want to override the end index, set END_CLASS before submitting:
#   sbatch --export=END_CLASS=10 generate.sbatch
end_class=${END_CLASS:-0}

# Class names
#class_names=("giant_cell_tumor" "multiple_osteochondromas" "osteochondroma" "osteofibroma" "osteosarcoma" "simple_bone_cyst" "synovial_osteochondroma")

class_names=(
  "upper_limb_osteochondroma"
  "upper_limb_osteosarcoma"
  "upper_limb_multiple_osteochondromas"
  "upper_limb_simple_bone_cyst"
  "upper_limb_giant_cell_tumor"
  "upper_limb_synovial_osteochondroma"
  "upper_limb_osteofibroma"
  "lower_limb_osteochondroma"
  "lower_limb_osteosarcoma"
  "lower_limb_multiple_osteochondromas"
  "lower_limb_simple_bone_cyst"
  "lower_limb_giant_cell_tumor"
  "lower_limb_synovial_osteochondroma"
  "lower_limb_osteofibroma"
  "pelvis_osteochondroma"
  "pelvis_osteosarcoma"
  "pelvis_multiple_osteochondromas"
  "pelvis_simple_bone_cyst"
  "pelvis_giant_cell_tumor"
  "pelvis_synovial_osteochondroma"
)

if [ "$end_class" -le 0 ]; then
  end_class=$((${#class_names[@]} - 1))
fi

echo "Generating classes ${start_class}-${end_class}"

for class in $(seq "${start_class}" "${end_class}"); do
  class_name=${class_names[$class]}
  outdir="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/generated_images_${snapshot}/${class_name}_gamma${gamma}_snapshot${snapshot}_trunc${truncation}"
  echo "Class ${class}: ${class_name}"
  python generate.py \
    --outdir="${outdir}" \
    --trunc=${truncation} \
    --seeds=0-49 \
    --network="/vol/miltank/users/wiep/Documents/stylegan2-ada-pytorch/checkpoints/stylegan2ada_cond_train_corrected/00000-btxrd_train_anatomical_dataset-cond-mirror-auto1-gamma6-batch16-ada/network-snapshot-${snapshot}.pkl" \
    --class=${class}
done
