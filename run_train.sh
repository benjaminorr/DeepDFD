#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --job-name=train
#SBATCH --time 1-00:00:00
#SBATCH --account=nitin
#SBATCH --partition short
#SBATCH --mem=128GB
#SBATCH -o training_outputs/s_train%j.out
#SBATCH -e training_outputs/s_train%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

# `.venv` here is the code-review-graph MCP tooling env, not this project's -- use the conda env with the
# modernized deps (see requirements.txt / REPLICATION_UPGRADE_NOTES.md) instead.
#
# Calling the env's interpreter by absolute path rather than `conda activate` -- on this cluster, whatever
# shell init Slurm applies on the compute node doesn't reliably keep conda's PATH change in effect (a prior
# run hit `ModuleNotFoundError: No module named 'torch'` because `python` still resolved elsewhere despite
# `conda activate learned_defocus` reporting no error).
PYTHON=/home/bcorr/.conda/envs/learned_defocus/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "python: $PYTHON"
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# --no-mix_dualpixel_dataset: only SceneFlow is set up under data/training_data (see the FlyingThings3D_subset
# symlinks); the trainer defaults to mixing in DualPixel too, which isn't available here.
$PYTHON snapshotdepth_trainer.py \
  --experiment_name IMX585_f50_N6.3 \
  --batch_sz 2 \
  --max_epochs 100 \
  --optimize_optics \
  --psfjitter \
  --no-mix_dualpixel_dataset \
  --camera_pixel_pitch 2.9e-6 \
  --mask_sz 7940 \
  --full_size 2160 \
  --psf_size 128 \
  --crop_width 64 \
  --accelerator gpu \
  --devices 1 \
  --num_workers 10
