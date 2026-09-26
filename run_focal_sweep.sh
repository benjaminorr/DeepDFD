#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --job-name=focal_sweep
#SBATCH --time 1-00:00:00
#SBATCH --account=nitin
#SBATCH --partition short
#SBATCH --mem=128GB
#SBATCH -o training_outputs/s_focal_sweep_%A_%a.out
#SBATCH -e training_outputs/s_focal_sweep_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-2

# Focal-length sweep emulating the Sony IMX585 (2.9 um pixels), one array task per lens. Submit with:
#   sbatch run_focal_sweep.sh
# Re-run a single lens (e.g. after it hits the time limit) with:
#   sbatch --array=1 run_focal_sweep.sh
# Each task resumes from its own last.ckpt, since the experiment name is fixed per focal length.
#
# --signal=SIGUSR1@90 lets Lightning's SLURM auto-requeue save and requeue the task before the time limit;
# --open-mode=append keeps the requeued task's output in the same log file.

set -euo pipefail

PYTHON=/home/bcorr/.conda/envs/learned_defocus/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

F_NUMBER=6.3
FOCAL_DEPTH=0.5

# Indexed by SLURM_ARRAY_TASK_ID. mask_sz keeps the DOE sampling pitch at ~1 um (mask_sz ~= (f/N) / 1um) and
# must be divisible by 2 * mask_upsample_factor (20). Defocus blur at f/6.3 over 1-5 m is <= 28 px for all
# three lenses, so the default psf_size 64 / crop_width 32 window fits.
FOCAL_LENGTHS_MM=(16   25   35)
MASK_SZS=(2540 3960 5560)

i=${SLURM_ARRAY_TASK_ID:?must be submitted as an array job}
F_MM=${FOCAL_LENGTHS_MM[$i]}
MASK_SZ=${MASK_SZS[$i]}
EXPERIMENT_NAME="IMX585_f${F_MM}_N${F_NUMBER}"

echo "python: $PYTHON"
echo "task $i: focal_length=${F_MM}mm f_number=$F_NUMBER mask_sz=$MASK_SZ experiment=$EXPERIMENT_NAME"
$PYTHON -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

$PYTHON snapshotdepth_trainer.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --batch_sz 3 \
  --max_epochs 100 \
  --optimize_optics \
  --psfjitter \
  --no-mix_dualpixel_dataset \
  --camera_pixel_pitch 2.9e-6 \
  --focal_length "${F_MM}e-3" \
  --f_number "$F_NUMBER" \
  --focal_depth "$FOCAL_DEPTH" \
  --mask_sz "$MASK_SZ" \
  --psf_size 64 \
  --crop_width 32 \
  --accelerator gpu \
  --devices 1 \
  --num_workers 10
