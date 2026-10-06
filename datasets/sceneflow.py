from typing import Tuple
import glob
import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
from kornia.filters import gaussian_blur2d

# 'left' image has negative disparity.
DATA_ROOT = os.path.join('data', 'training_data', 'SceneFlow')


def _subset_dirs(root, dataset):
    """Image/disparity dirs for the FlyingThings3D subset layout:
    <root>/FlyingThings3D_subset{,_disparity}/{train,val}/..."""
    if dataset == 'example':
        return ([os.path.join(root, 'example', 'FlyingThings3D', 'RGB_cleanpass', 'left')],
                [os.path.join(root, 'example', 'FlyingThings3D', 'disparity')])
    image_dirs = [
        os.path.join(root, 'FlyingThings3D_subset', dataset, 'image_clean', s) for s in ['right']
    ]
    disparity_dirs = [
        os.path.join(root, 'FlyingThings3D_subset_disparity', dataset, 'disparity', s) for s in ['right']
    ]
    return image_dirs, disparity_dirs


def _full_dirs(root, dataset):
    """Image/disparity dirs for the complete FlyingThings3D layout:
    <root>/{frames_cleanpass,disparity}/{TRAIN,TEST}/{A,B,C}/<sequence>/right"""
    if dataset == 'example':
        raise ValueError(f'The complete FlyingThings3D layout ({root}) has no "example" split.')
    split = {'train': 'TRAIN', 'val': 'TEST'}[dataset]
    image_root = os.path.join(root, 'frames_cleanpass')
    # Sorted so that sample order (and hence the trainer's index-based train/val split) is deterministic.
    image_dirs = sorted(glob.glob(os.path.join(image_root, split, '*', '*', 'right')))
    disparity_dirs = [os.path.join(root, 'disparity', os.path.relpath(d, image_root)) for d in image_dirs]
    return image_dirs, disparity_dirs


def read_pfm(path):
    """Read a PFM file as an upright (top row first) float32 array.

    Not using imageio: depending on the installed plugins, imageio has either returned PFM rows bottom-up (the original
    code flipped them back) or decoded PFM via OpenCV as uint8 (destroying the disparity values).
    """
    with open(path, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            channels = 3
        elif header == b'Pf':
            channels = 1
        else:
            raise ValueError(f'Not a PFM file: {path}')
        width, height = map(int, f.readline().split())
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, dtype=endian + 'f4', count=width * height * channels)
    shape = (height, width, channels) if channels == 3 else (height, width)
    # PFM stores scanlines from bottom to top.
    return np.flipud(data.reshape(shape)).astype(np.float32)


class SceneFlow(Dataset):

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, singleplane: bool = False, n_depths: int = 16,
                 root: str = DATA_ROOT):
        """
        SceneFlow dataset is downloaded from
        https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
        Virtual image sensor size: 960 px x 540 px  or 32mm x 18mm
        Virtual focal length: 35mmx
        Baseline: 1 Blender unit

        root may be laid out either like the FlyingThings3D subset (the default) or like the complete
        FlyingThings3D (detected by a frames_cleanpass/ dir; 'val' then maps to its TEST split).
        """
        super().__init__()
        if dataset not in ('train', 'val', 'example'):
            raise ValueError(f'dataset ({dataset}) has to be "train," "val," or "example."')
        if os.path.isdir(os.path.join(root, 'frames_cleanpass')):
            image_dirs, disparity_dirs = _full_dirs(root, dataset)
        else:
            image_dirs, disparity_dirs = _subset_dirs(root, dataset)
        if not image_dirs:
            raise FileNotFoundError(f'No SceneFlow "{dataset}" images found under {root}')

        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)

        self.sample_ids = []
        for image_dir, disparity_dir in zip(image_dirs, disparity_dirs):
            for filename in sorted(os.listdir(image_dir)):
                if '.png' in filename:
                    id = os.path.splitext(filename)[0]
                    disparity_path = os.path.join(disparity_dir, f'{id}.pfm')
                    if os.path.exists(disparity_path):
                        sample_id = {
                            'image_dir': image_dir,
                            'disparity_dir': disparity_dir,
                            'id': id,
                        }
                        self.sample_ids.append(sample_id)
                    else:
                        print(f'Disparity image does not exist!: {disparity_path}')
        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths

    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_dir = sample_id['image_dir']
        disparity_dir = sample_id['disparity_dir']
        id = sample_id['id']

        disparity = read_pfm(os.path.join(disparity_dir, f'{id}.pfm'))
        img = imageio.v2.imread(os.path.join(image_dir, f'{id}.png')).astype(np.float32)
        img /= 255.  # Scale to [0, 1]

        img = np.pad(img,
                     ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        disparity = np.pad(disparity,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

        img = torch.from_numpy(img).permute(2, 0, 1)
        disparity = torch.from_numpy(disparity)[None, ...]

        # A far object is 0.
        depthmap = disparity
        depthmap -= depthmap.min()
        depthmap /= depthmap.max()

        # Flip the value. A near object is 0.
        depthmap = 1. - depthmap

        if self.is_training:
            img, depthmap = self.transform(img, depthmap)
        else:
            img = self.centercrop(img)
            depthmap = self.centercrop(depthmap)

        # SceneFlow's depthmap has some aliasing artifact.
        depthmap = gaussian_blur2d(depthmap, sigma=(0.8, 0.8), kernel_size=(5, 5))

        # Remove batch dim (Kornia adds batch dimension automatically.)
        img = img.squeeze(0)
        depthmap = depthmap.squeeze(0)

        if self.singleplane:
            if self.is_training:
                depthmap = torch.rand((1,), device=depthmap.device) * torch.ones_like(depthmap)
            else:
                depthmap = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(depthmap)



        sample = {'id': id, 'image': img, 'depthmap': depthmap, 'depth_conf': torch.ones_like(depthmap)}

        return sample
