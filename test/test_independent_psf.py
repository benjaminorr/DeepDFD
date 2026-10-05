"""Cross-check the independent simulator's PSF (2D pupil + matrix Fourier transform, optics/independent_sim.py) against
the model's PSF math (1D Hankel transform, optics/camera.py), evaluated exactly at the same sensor positions.

The comparison deliberately skips the model's last step, the cubic-spline resampling from its radial grid onto the
pixel grid: that is an approximation of the model, not of the physics.
"""
import glob
import math
import os
from argparse import Namespace

import pytest
import scipy.special
import torch

from optics.independent_sim import psf_stack
from snapshotdepth import SnapshotDepth

CKPT_GLOB = os.path.join('data', 'logs', 'IMX585_f25_1-3m_FD1.5', 'version_*', 'checkpoints', 'last.ckpt')


@pytest.fixture(scope='module')
def camera():
    ckpts = glob.glob(CKPT_GLOB)
    if not ckpts:
        pytest.skip(f'checkpoint not found: {CKPT_GLOB}')
    ckpt = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    model = SnapshotDepth(hparams=Namespace(**ckpt['hyper_parameters']))
    model.load_state_dict(ckpt['state_dict'])
    return model.camera.eval()


def hankel_psf_on_line(camera, modulate_phase, n=64):
    """Model PSF (camera.psf1d) evaluated exactly at sensor positions (1, k) px, k = 1..n. Returns 3 x D x n."""
    pts = torch.sqrt(1. + torch.arange(1, n + 1, dtype=torch.float64) ** 2) * camera.camera_pixel_pitch
    wl = camera.wavelengths.double().reshape(-1, 1, 1)
    rho = pts.reshape(1, 1, -1) / (wl * camera.sensor_distance())
    r = camera.mask_pitch * torch.linspace(1, camera.mask_size / 2, camera.mask_size // 2).double().reshape(1, -1, 1)
    # Same ring integrals as BaseRotationallySymmetricCamera.precompute_H
    J = r / (2 * math.pi * rho) * torch.from_numpy(scipy.special.jv(1, (2 * math.pi * rho * r).numpy()))
    H = torch.cat([J[:, :1], J[:, 1:] - J[:, :-1]], dim=1)
    with torch.no_grad():
        return camera.psf1d(H, camera.scene_distances, torch.tensor(modulate_phase)).float()


@pytest.mark.parametrize('modulate_phase', [False, True])
def test_mft_psf_matches_hankel_psf(camera, modulate_phase):
    n = 64
    expected = hankel_psf_on_line(camera, modulate_phase, n)
    # Odd grid with the axis at index n_out // 2, so row / column n_out // 2 + k sit at +k px.
    n_out = 2 * n + 1
    psf = psf_stack(camera, camera.scene_distances.tolist(), n_out, camera.camera_pixel_pitch,
                    wavelengths=[[w] for w in camera.wavelengths.tolist()],
                    diffraction_efficiency=1.0 if modulate_phase else 0.0,
                    device='cuda' if torch.cuda.is_available() else 'cpu').cpu()
    actual = psf[..., n + 1, n + 1:]

    expected = expected / expected.sum(dim=-1, keepdim=True)
    actual = actual / actual.sum(dim=-1, keepdim=True)
    rel_l1 = (actual - expected).abs().sum(dim=-1)
    assert rel_l1.max() < 0.01, rel_l1
