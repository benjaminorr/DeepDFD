"""Cross-check the independent simulator's PSF (2D pupil + matrix Fourier transform, optics/independent_sim.py) against
the model's PSF (1D Hankel transform, optics/camera.py).

test_mft_psf_matches_hankel_psf compares the physics exactly at the same sensor positions, skipping the model's
resampling from its radial grid onto pixels. test_oversampled_psf_matches_pixel_integrated_psf checks that resampling
for psf_oversample > 0 against the independent PSF integrated over each pixel.
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


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def ckpt():
    ckpts = glob.glob(CKPT_GLOB)
    if not ckpts:
        pytest.skip(f'checkpoint not found: {CKPT_GLOB}')
    return torch.load(ckpts[0], map_location='cpu', weights_only=False)


@pytest.fixture(scope='module')
def camera(ckpt):
    # A checkpoint from before psf_oversample existed: must load as before (original sampling).
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
                    diffraction_efficiency=1.0 if modulate_phase else 0.0, device=DEVICE).cpu()
    actual = psf[..., n + 1, n + 1:]

    expected = expected / expected.sum(dim=-1, keepdim=True)
    actual = actual / actual.sum(dim=-1, keepdim=True)
    rel_l1 = (actual - expected).abs().sum(dim=-1)
    assert rel_l1.max() < 0.01, rel_l1


@pytest.mark.parametrize('modulate_phase', [False, True])
def test_oversampled_psf_matches_pixel_integrated_psf(ckpt, modulate_phase):
    model = SnapshotDepth(hparams=Namespace(**{**ckpt['hyper_parameters'], 'psf_oversample': 4}))
    model.camera.heightmap1d_.data = ckpt['state_dict']['camera.heightmap1d_']
    camera = model.camera.to(DEVICE)
    with torch.no_grad():
        actual = camera._psf_at_camera_impl(camera.H, camera.rho_grid, camera.rho_sampling, camera.ind,
                                            camera.image_size, camera.scene_distances, torch.tensor(modulate_phase))

    # Independent PSF on an 8x finer grid, averaged over each pixel. The optical axis (fine index n * k // 2) is a
    # pixel corner, as in the model, so the k x k blocks are the pixels.
    n, k = camera.image_size[0], 8
    expected = psf_stack(camera, camera.scene_distances.tolist(), n * k, camera.camera_pixel_pitch / k,
                         wavelengths=[[w] for w in camera.wavelengths.tolist()],
                         diffraction_efficiency=1.0 if modulate_phase else 0.0, device=DEVICE)
    expected = torch.nn.functional.avg_pool2d(expected, k)

    actual = actual / actual.sum(dim=(-1, -2), keepdim=True)
    expected = expected / expected.sum(dim=(-1, -2), keepdim=True)
    rel_l1 = (actual - expected).abs().sum(dim=(-1, -2))
    # About 0.06-0.12 remains from the reference's samples sitting 1/16 px off the pixel-block centers; the original
    # sampling (psf_oversample=0) is at up to ~1.0.
    assert rel_l1.max() < 0.15, rel_l1
