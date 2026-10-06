## Project

This is the code repository
for [Depth from Defocus with Learned Optics for Imaging and Occlusion-aware Depth Estimation (ICCP 2021)](http://www.computationalimaging.org/publications/deepopticsdfd/)
.

## Environment

> This repo was originally written against Python 3.8 / PyTorch 1.7 / PyTorch Lightning 1.0.2. It has since been
> modernized to run on current libraries; see `REPLICATION_UPGRADE_NOTES.md` for what changed and why.

Create an environment (conda or venv both work) and install the pinned, verified-working dependencies:

```shell
conda create --name learned_defocus python=3.12 -y
conda activate learned_defocus
pip install -r requirements.txt
```

## Dataset for training

Download the datasets
from [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
and [DualPixel](https://github.com/google-research/google-research/blob/master/dual_pixels/README.md). To complete the
sparse depth map, the central view of DualPixel dataset is filled
with a python port of [NYU Depth V2's toolbox](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). After downloading the datasets,
place them under `data/training_data` directory. You can change the path to the dataset
in [dataset/dualpixel.py](dataset/dualpixel.py) and [dataset/sceneflow.py](dataset/sceneflow.py).

## How to run the training code

```shell
python snapshotdepth_trainer.py \
  --gpus 4 --batch_sz 3 --distributed_backend ddp  --max_epochs 100  --optimize_optics  --psfjitter  --replace_sampler_ddp False
```

New runs sample the PSF with `--psf_oversample 4` (4×4 samples per pixel on a finer radial grid; see
`precompute_H` in `optics/camera.py`). The original code's coarser sampling distorts PSFs that are only a few pixels
wide; it is `--psf_oversample 0`, which checkpoints trained before this option keep using, also when resumed.

## Checkpoint and captured data

An example of a trained checkpoint and a trained DOE is available from
GoogleDrive ([checkpoint](https://drive.google.com/file/d/1ZdenyEUhFq497QKQLI0VgIXYfLkMI778/view?usp=sharing)
and [image](https://drive.google.com/file/d/1rW6TspPHvpJSGFvxF4eP_gxroLy7Ico7/view?usp=sharing)).

## How to run the inference code on a real captured data

Download the captured image and the checkpoint, and place them in `data` directory.

```shell
python run_trained_snapshotdepth_on_captured_images.py \
  --ckpt_path data/checkpoint.ckpt \
  --captimg_path data/captured_data/outdoor1_predemosaic.tif 
```

This inference code runs on CPU. Note that it is memory-hungry: the Tikhonov solver builds a
depth×depth matrix per pixel (16 depths here) across the full image resolution and a 4-way
test-time-augmented batch, which can need tens of GB of RAM for a real ~1200×1920 capture — budget
accordingly (e.g. request more memory if running inside a resource-limited job/container).

Example input and output:

![Example input](result/indoor1_captimg.jpg)
![Example estimated image](result/indoor1_estimg.jpg)
![Example estimated depth](result/indoor1_estdepthmap.jpg)

## How to run a trained model on an ordinary RGB photo

`run_snapshotdepth_on_rgb_image.py` simulates a raw coded capture of an all-in-focus photo with a checkpoint's learned
DOE, then reconstructs it the same way as a real capture. The scene depth for the simulation comes from Depth Anything
V2 by default (see [third_party/README.md](third_party/README.md) for the weights); `--fit_depth_range` maps it onto the
model's trained depth range instead of clamping.

By default the capture is simulated independently of the model's own image formation, to avoid the "inverse crime"
(`optics/independent_sim.py`: PSFs from the 2D pupil by matrix Fourier transform, several wavelengths per color
channel, finer depth and spatial sampling, Poisson + read noise, 12-bit raw); `--simulator model` uses the model's own
image formation instead, as an optimistic reference. `test/test_independent_psf.py` checks that the two PSF
implementations agree. `--raw_path` reconstructs a real RGGB raw capture. See the script's docstring for the other
options.

```shell
python run_snapshotdepth_on_rgb_image.py --img_path data/sample_img/IMX585_test.jpg \
  --experiment IMX585_f25_1-3m_FD1.5 --fit_depth_range
```

## Raw data for the fabricated DOE

The design for the fabricated DOE is
available [here](https://drive.google.com/file/d/1kQtJn0rgH26193gOLoTiOfz8EKXZLIn4/view?usp=sharing). Unit is in meter,
and the pixel size is 1&mu;m.

## Citation

Hayato Ikoma, Cindy M. Nguyen, Christopher A. Metzler, Yifan Peng, Gordon Wetzstein, Depth from Defocus with Learned
Optics for Imaging and Occlusion-aware Depth Estimation, International Conference on Computational Photography 2021

```
@article{Ikoma:2021,
author = {Hayato Ikoma and Cindy M. Nguyen and Christopher A. Metzler and Yifan Peng and Gordon Wetzstein},
title = {Depth from Defocus with Learned Optics for Imaging and Occlusion-aware Depth Estimation},
journal = {IEEE International Conference on Computational Photography (ICCP)},
year={2021}
}
```

## Contact
Please direct questions to [hikoma@stanford.edu](hikoma@stanford.edu).

## Acknowledgement

We thank the open source software used in our project, which includes [pytorch](https://github.com/pytorch/pytorch),
[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning), [numpy](https://github.com/numpy/numpy),
[scipy](https://github.com/scipy/scipy), [kornia](https://github.com/kornia/kornia)
, [pytorch-debayer](https://github.com/cheind/pytorch-debayer), [matplotlib](https://github.com/matplotlib/matplotlib)
, [OpenCV](https://opencv.org/) and [Fiji](https://github.com/fiji/fiji).