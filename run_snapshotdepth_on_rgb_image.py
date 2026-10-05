"""
Run a trained SnapshotDepth model on an ordinary (all-in-focus, no coded aperture) RGB photo, or on a real raw capture.

Photo input: the photo is treated as the ground-truth scene radiance, a coded capture of it is simulated as a raw Bayer
mosaic, and the mosaic is reconstructed exactly as a real capture would be. Two simulators:

    independent  (default) an image formation independent of the model's, to avoid the "inverse crime"
                 (optics/independent_sim.py): PSFs from the 2D pupil by matrix Fourier transform, 3 wavelengths per
                 color channel, 64 depth layers, scene and PSFs on a 2x finer grid binned to the sensor pixels, Poisson
                 shot + read noise, 12-bit raw. The photo is the fine-grid scene, so the sensor image has half its
                 resolution.
    model        the model's own image formation (as in training): PSFs from optics/camera.py, 16 depth layers,
                 Gaussian noise. Inverse crime -- an optimistic reference.

Simulating needs a depth map, which the photo doesn't provide. Choose it with --depth_mode:

    depth_anything  (default) metric depth estimated from the photo with Depth Anything V2 (util/monodepth.py,
                    weights from --da_weights; see third_party/README.md)
    ramp            vertical depth gradient, far at the top and near at the bottom
    plane           one fronto-parallel plane at --plane_depth meters
    file            a per-pixel metric depth map (meters) from --depth_path (.npy, or a float/16-bit .tif/.png;
                    multiply by --depth_scale to get meters), e.g. from an RGB-D dataset or a depth sensor

The model only covers its trained depth range (min_depth-max_depth, e.g. 1-3 m). Depth outside it is clamped by
default, so a scene mostly beyond max_depth simulates as mostly one plane. --fit_depth_range instead maps the scene's
2nd-98th percentile inverse depth onto the trained range: relative depth ordering is kept, metric scale is not.

Raw input (--raw_path): an RGGB Bayer mosaic (R at row 0, col 0), e.g. a 16-bit TIFF/PNG; set --black_level,
--white_level and --wb_gains for the sensor. There is no ground truth, so no error is reported.

Reconstruction: demosaic (the model's bilinear Debayer3x3), normalize by the maximum, then the Tikhonov pre-inverse
and the U-Net decoder, using only the model's own PSFs. The image is processed in overlapping tiles (the Tikhonov
solver builds a depth x depth matrix per frequency); each tile carries the crop_width boundary used in training plus
--tile_overlap extra pixels per side.

Usage

python run_snapshotdepth_on_rgb_image.py --img_path data/sample_img/IMX585_test.jpg --fit_depth_range
python run_snapshotdepth_on_rgb_image.py --img_path data/sample_img/IMX585_test.jpg --fit_depth_range \
    --simulator model
python run_snapshotdepth_on_rgb_image.py --img_path data/sample_img/IMX585_test.jpg \
    --experiment IMX585_f25_1-3m_FD1.0 --depth_mode plane --plane_depth 2.0
python run_snapshotdepth_on_rgb_image.py --raw_path capture.tif --white_level 4095 --wb_gains 1.9 1.6

Fabrication / assembly sensitivity (independent simulator):

for s in 0.95 1.0 1.05; do
  python run_snapshotdepth_on_rgb_image.py --fit_depth_range --sim_height_scale $s \
      --out_dir data/result/sweep/height_$s
done
"""

import glob
import json
import os
import re
from argparse import ArgumentParser, Namespace

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import torch.nn.functional as F
import torchvision.utils

from optics import independent_sim
from snapshotdepth import SnapshotDepth
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, ips_to_metric, linear_to_srgb, metric_to_ips, srgb_to_linear, to_bayer
from util.monodepth import estimate_metric_depth

DEFAULT_EXPERIMENT = 'IMX585_f25_1-3m_FD1.5'


def find_checkpoint(log_root, experiment, which):
    """Pick a checkpoint of a training run: the lowest val_loss one ('best') or last.ckpt ('last')."""
    ckpts = glob.glob(os.path.join(log_root, experiment, 'version_*', 'checkpoints', '*.ckpt'))
    if which == 'last':
        ckpts = [c for c in ckpts if os.path.basename(c) == 'last.ckpt']
        if not ckpts:
            raise FileNotFoundError(f'No last.ckpt under {os.path.join(log_root, experiment)}')
        return max(ckpts, key=os.path.getmtime)

    scored = []
    for c in ckpts:
        m = re.search(r'val_loss=([0-9.]+?)\.ckpt$', os.path.basename(c))
        if m:
            scored.append((float(m.group(1)), c))
    if not scored:
        raise FileNotFoundError(f'No "val_loss=" checkpoints under {os.path.join(log_root, experiment)}')
    return min(scored)[1]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = Namespace(**ckpt['hyper_parameters'])
    model = SnapshotDepth(hparams=hparams)
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()


def load_image(path, resize, multiple):
    img = skimage.io.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    else:
        img = img.astype(np.float32)
    if resize != 1.0:
        img = skimage.transform.rescale(img, resize, channel_axis=-1, anti_aliasing=True).astype(np.float32)
    # Crop so the sensor image (after any binning) has an even size, which the camera model and Bayer pattern need.
    h, w = img.shape[:2]
    return img[:h - h % multiple, :w - w % multiple]


def load_raw(args):
    raw = skimage.io.imread(args.raw_path)
    if raw.ndim != 2:
        raise ValueError(f'Expected a single-channel Bayer mosaic, got shape {raw.shape}')
    white = args.white_level if args.white_level is not None else (
        np.iinfo(raw.dtype).max if np.issubdtype(raw.dtype, np.integer) else 1.0)
    raw = (raw.astype(np.float32) - args.black_level) / (white - args.black_level)
    h, w = raw.shape
    return torch.from_numpy(raw[:h - h % 2, :w - w % 2])


def make_depth_m(args, img, min_depth, max_depth, device):
    h, w = img.shape[:2]
    if args.depth_mode == 'depth_anything':
        return estimate_metric_depth(img, args.da_weights, input_size=args.da_input_size, device=device)
    if args.depth_mode == 'plane':
        return np.full((h, w), args.plane_depth, dtype=np.float32)
    if args.depth_mode == 'ramp':
        near = min_depth if args.ramp_near is None else args.ramp_near
        far = max_depth if args.ramp_far is None else args.ramp_far
        # Interpolate in inverse depth (like the model's depth sampling): top row = far, bottom row = near.
        t = np.linspace(0., 1., h, dtype=np.float32)[:, None]
        inv = (1 - t) / far + t / near
        return np.broadcast_to(1. / inv, (h, w)).astype(np.float32)
    if args.depth_mode == 'file':
        if args.depth_path is None:
            raise ValueError('--depth_mode file needs --depth_path')
        if args.depth_path.endswith('.npy'):
            depth = np.load(args.depth_path)
        else:
            depth = skimage.io.imread(args.depth_path)
        depth = np.squeeze(depth).astype(np.float32) * args.depth_scale
        if depth.ndim != 2:
            raise ValueError(f'Depth map must be single-channel, got shape {depth.shape}')
        if depth.shape != (h, w):
            depth = skimage.transform.resize(depth, (h, w), order=1, anti_aliasing=False).astype(np.float32)
        return depth
    raise ValueError(f'Unknown depth_mode: {args.depth_mode}')


def fit_depth_range(depth_m, min_depth, max_depth, percentile=2.):
    """Affinely map the scene's inverse depth (robust range) onto the model's [1/max_depth, 1/min_depth]."""
    inv = 1. / np.maximum(depth_m, 1e-3)
    lo, hi = np.percentile(inv, [percentile, 100 - percentile])
    t = np.clip((inv - lo) / max(hi - lo, 1e-12), 0., 1.)
    return (1. / ((1 - t) / max_depth + t / min_depth)).astype(np.float32)


def scene_depth_for_model(args, img, hp, device):
    """Depth map at the photo resolution [m], clamped or range-fitted to the model's depth range."""
    scene_depth_m = make_depth_m(args, img, hp.min_depth, hp.max_depth, device)
    print(f'scene depth ({args.depth_mode}): {scene_depth_m.min():.2f}-{scene_depth_m.max():.2f}m, '
          f'median {np.median(scene_depth_m):.2f}m')
    if args.fit_depth_range:
        print(f'mapped scene depth onto the trained range [{hp.min_depth}, {hp.max_depth}]m (relative depth only)')
        return scene_depth_m, fit_depth_range(scene_depth_m, hp.min_depth, hp.max_depth)
    below, above = np.mean(scene_depth_m < hp.min_depth), np.mean(scene_depth_m > hp.max_depth)
    if below + above > 0:
        print(f'note: clamping to the trained range [{hp.min_depth}, {hp.max_depth}]m: {100 * below:.1f}% of pixels '
              f'are nearer, {100 * above:.1f}% farther (see --fit_depth_range)')
    return scene_depth_m, np.clip(scene_depth_m, hp.min_depth, hp.max_depth)


def simulate_independent(model, img_linear, depth_ips, args, device):
    """Independent image formation (see optics/independent_sim.py). Inputs are on the fine (photo) grid.

    Returns the raw mosaic in [0, 1] and the ground truth binned to the sensor grid.
    """
    hp, cam, ss = model.hparams, model.camera, args.sim_supersample
    n_layers = args.sim_layers
    # Layer k covers inverse depth (k / L, (k + 1) / L]; its PSF is computed at the layer center.
    depths = ips_to_metric((torch.arange(n_layers) + 0.5) / n_layers, hp.min_depth, hp.max_depth).tolist()
    psf = independent_sim.psf_stack(cam, depths, out_size=args.sim_psf_size * ss,
                                    out_pitch=hp.camera_pixel_pitch / ss, height_scale=args.sim_height_scale,
                                    focus_offset=args.sim_focus_offset, device=device)
    irradiance = independent_sim.render_layered(cam, img_linear, depth_ips, psf, device=device)
    irradiance = F.avg_pool2d(irradiance, ss)
    raw = independent_sim.sensor_raw(irradiance, args.full_well, args.read_noise, args.bit_depth)
    raw = torch.from_numpy(raw.astype(np.float32) / (2 ** args.bit_depth - 1))
    return raw, F.avg_pool2d(img_linear, ss), F.avg_pool2d(depth_ips, ss)


def simulate_model(model, img_linear, depth_ips, args, device):
    """The model's own image formation, as in SnapshotDepth.forward (without PSF jitter)."""
    cam = model.camera
    psf = cam.normalize_psf(cam.psf_at_camera(size=cam.image_size))
    psf = fftshift(psf, dims=(-1, -2))  # center it, as render_layered expects
    irradiance = independent_sim.render_layered(cam, img_linear, depth_ips, psf, device=device)
    bayer = to_bayer(irradiance)[0, 0]
    return bayer + args.noise_sigma * torch.randn_like(bayer), img_linear, depth_ips


def flip_tta(x):
    return torch.cat([x, torch.flip(x, dims=(-1,)), torch.flip(x, dims=(-2,)), torch.flip(x, dims=(-2, -1))], dim=0)


def unflip_tta(x):
    return torch.stack([x[0], torch.flip(x[1], dims=(-1,)), torch.flip(x[2], dims=(-2,)),
                        torch.flip(x[3], dims=(-2, -1))], dim=0).mean(dim=0, keepdim=True)


@torch.no_grad()
def reconstruct(model, raw, args, device):
    """Raw RGGB mosaic (H x W, [0, 1]) -> captured image (linear), estimated image (sRGB), estimated depth ([0, 1]
    inverse-perspective), all 1 x C x H x W on the CPU. Uses only the model's own PSFs."""
    hp, cw = model.hparams, model.crop_width
    capt = model.debayer(raw[None, None].to(device)).cpu()
    capt[:, 0] *= args.wb_gains[0]
    capt[:, 2] *= args.wb_gains[1]
    capt = capt / capt.max()

    tile, m = args.tile_size, args.tile_overlap
    pad = cw + m
    tile_in = tile + 2 * pad
    psf_size = max(tile_in, max(model.camera.image_size))
    psf = model.camera.normalize_psf(model.camera.psf_at_camera(size=(psf_size, psf_size)).unsqueeze(0))
    psf_cropped = crop_psf(psf, tile_in)

    h, w = capt.shape[-2:]
    n_ty, n_tx = -(-h // tile), -(-w // tile)
    capt_p = F.pad(F.pad(capt, (pad,) * 4, mode='reflect'), (0, n_tx * tile - w, 0, n_ty * tile - h))
    est_img = torch.zeros(1, 3, n_ty * tile, n_tx * tile)
    est_depth = torch.zeros(1, 1, n_ty * tile, n_tx * tile)
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, x0 = ty * tile, tx * tile
            x = capt_p[..., y0:y0 + tile_in, x0:x0 + tile_in].to(device)
            x = flip_tta(x) if args.tta else x
            pinv_volumes = apply_tikhonov_inverse(x, psf_cropped, hp.reg_tikhonov, apply_edgetaper=True)
            out = model.decoder(captimgs=x, pinv_volumes=pinv_volumes)
            ei, ed = out.est_images, out.est_depthmaps
            if args.tta:
                ei, ed = unflip_tta(ei), unflip_tta(ed)
            est_img[..., y0:y0 + tile, x0:x0 + tile] = crop_boundary(ei, pad).cpu()
            est_depth[..., y0:y0 + tile, x0:x0 + tile] = crop_boundary(ed, pad).cpu()
            print(f'\rtile {ty * n_tx + tx + 1}/{n_ty * n_tx}', end='', flush=True)
    print()
    return capt, est_img[..., :h, :w], est_depth[..., :h, :w]


def to_hwc(x):
    return x.squeeze(0).permute(1, 2, 0).cpu().numpy()


def save_rgb(path, x):
    skimage.io.imsave(path, (255 * np.clip(x, 0, 1)).astype(np.uint8), check_contrast=False)


def save_psf_grid(path, model, size=96):
    psf = model.camera.normalize_psf(model.camera.psf_at_camera(size=(size * 4, size * 4)))
    psf = fftshift(crop_psf(psf, size), dims=(-1, -2))  # 3 x D x size x size, centered
    psf = psf / psf.max()
    grid = torchvision.utils.make_grid(psf.transpose(0, 1), nrow=psf.shape[1] // 2, pad_value=1, normalize=False)
    save_rgb(path, grid.permute(1, 2, 0).cpu().numpy() ** 0.5)  # sqrt to show the PSF tails


def save_summary(path, title, capt_srgb, est_srgb, est_depth_m, depth_kw, gt=None):
    """gt: None, or (gt_img_srgb, gt_depth_m, err_m, depth_title)."""
    if gt is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        panels = [(axes[0], capt_srgb, 'capture'), (axes[1], est_srgb, 'estimated image')]
        depth_ax = axes[2]
    else:
        gt_img, gt_depth_m, err_m, depth_title = gt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        panels = [(axes[0, 0], gt_img, 'input (ground truth)'), (axes[0, 1], capt_srgb, 'simulated coded capture'),
                  (axes[0, 2], est_srgb, 'estimated image')]
        axes[1, 0].imshow(gt_depth_m, **depth_kw)
        axes[1, 0].set_title(depth_title)
        eim = axes[1, 2].imshow(err_m, cmap='magma', vmin=0, vmax=depth_kw['vmax'] - depth_kw['vmin'])
        axes[1, 2].set_title(f'|error|, MAE {err_m.mean():.3f} m')
        fig.colorbar(eim, ax=axes[1, 2], label='[m]', shrink=0.8)
        depth_ax = axes[1, 1]
    for ax, x, t in panels:
        ax.imshow(np.clip(x, 0, 1))
        ax.set_title(t)
    im = depth_ax.imshow(est_depth_m, **depth_kw)
    depth_ax.set_title('estimated depth')
    fig.colorbar(im, ax=depth_ax, label='depth [m]', shrink=0.8)
    for ax in np.ravel(axes):
        ax.axis('off')
    fig.suptitle(title)
    fig.savefig(path, dpi=120)


def main(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)

    ckpt_path = args.ckpt_path or find_checkpoint(args.log_root, args.experiment, args.which_ckpt)
    print(f'checkpoint: {ckpt_path}')
    model = load_model(ckpt_path, device)
    hp = model.hparams
    min_depth, max_depth = hp.min_depth, hp.max_depth
    print(f'model: f={hp.focal_length * 1e3:g}mm N={hp.f_number} focal_depth={hp.focal_depth}m '
          f'depth range=[{min_depth}, {max_depth}]m pixel pitch={hp.camera_pixel_pitch * 1e6:g}um')

    exp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path)))) \
        if args.ckpt_path is None else os.path.splitext(os.path.basename(ckpt_path))[0]
    src = args.raw_path or args.img_path
    name = os.path.splitext(os.path.basename(src))[0]

    gt = None
    if args.raw_path:
        raw = load_raw(args)
        tag = 'raw'
    else:
        ss = args.sim_supersample if args.simulator == 'independent' else 1
        img = load_image(args.img_path, args.resize, multiple=2 * ss)
        scene_depth_m, depth_m = scene_depth_for_model(args, img, hp, device)
        img_linear = srgb_to_linear(torch.from_numpy(img).permute(2, 0, 1)[None])
        depth_ips = metric_to_ips(torch.from_numpy(depth_m), min_depth, max_depth)[None, None]
        simulate = simulate_independent if args.simulator == 'independent' else simulate_model
        raw, gt_img_linear, gt_depth_ips = simulate(model, img_linear, depth_ips, args, device)
        tag = f'{args.depth_mode}{"_fit" if args.fit_depth_range else ""}_{args.simulator}'
    print(f'sensor image: {raw.shape[1]}x{raw.shape[0]}')

    capt, est_img, est_depth = reconstruct(model, raw, args, device)
    est_depth_m = ips_to_metric(est_depth, min_depth, max_depth).squeeze().numpy()

    out_dir = args.out_dir or os.path.join('data', 'result', f'{name}_{exp}_{tag}')
    os.makedirs(out_dir, exist_ok=True)
    capt_srgb = to_hwc(linear_to_srgb(capt))
    est_srgb = to_hwc(est_img)  # the decoder is trained against sRGB targets, so its image output is already sRGB
    depth_kw = dict(cmap='inferno_r', vmin=min_depth, vmax=max_depth)
    skimage.io.imsave(os.path.join(out_dir, 'raw.png'), (65535 * raw.clamp(0, 1).numpy()).astype(np.uint16),
                      check_contrast=False)
    save_rgb(os.path.join(out_dir, 'captimg.png'), capt_srgb)
    save_rgb(os.path.join(out_dir, 'estimg.png'), est_srgb)
    plt.imsave(os.path.join(out_dir, 'est_depth.png'), est_depth_m, **depth_kw)
    np.save(os.path.join(out_dir, 'est_depth_m.npy'), est_depth_m)
    save_psf_grid(os.path.join(out_dir, 'psf_by_depth.png'), model)

    if not args.raw_path:
        gt_depth_m = ips_to_metric(gt_depth_ips, min_depth, max_depth).squeeze().numpy()
        err_ips = np.abs(est_depth.squeeze().numpy() - gt_depth_ips.squeeze().numpy())
        err_m = np.abs(est_depth_m - gt_depth_m)
        metrics = {
            'mae_ips': float(err_ips.mean()), 'mae_m': float(err_m.mean()), 'median_ae_m': float(np.median(err_m)),
            'checkpoint': ckpt_path, 'simulator': args.simulator, 'depth_mode': args.depth_mode,
            'fit_depth_range': args.fit_depth_range,
        }
        if args.simulator == 'independent':
            metrics.update({k: getattr(args, k) for k in ['sim_height_scale', 'sim_focus_offset', 'sim_supersample',
                                                          'sim_layers', 'full_well', 'read_noise', 'bit_depth']})
        else:
            metrics['noise_sigma'] = args.noise_sigma
        print(f'depth MAE: {metrics["mae_ips"]:.4f} (normalized inverse depth), {metrics["mae_m"]:.3f} m; '
              f'median abs error {metrics["median_ae_m"]:.3f} m')
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        gt_srgb = to_hwc(linear_to_srgb(gt_img_linear))
        save_rgb(os.path.join(out_dir, 'gt_img.png'), gt_srgb)
        plt.imsave(os.path.join(out_dir, 'gt_depth.png'), gt_depth_m, **depth_kw)
        plt.imsave(os.path.join(out_dir, 'scene_depth.png'), scene_depth_m, cmap='inferno_r',
                   vmin=np.percentile(scene_depth_m, 1), vmax=np.percentile(scene_depth_m, 99))
        np.save(os.path.join(out_dir, 'scene_depth_m.npy'), scene_depth_m)  # photo res, before clamping / fitting
        np.save(os.path.join(out_dir, 'gt_depth_m.npy'), gt_depth_m)
        depth_title = f'ground-truth depth ({args.depth_mode}, {"range-fit" if args.fit_depth_range else "clamped"})'
        gt = (gt_srgb, gt_depth_m, err_m, depth_title)

    save_summary(os.path.join(out_dir, 'summary.png'), f'{exp}  |  {os.path.basename(src)}  |  {tag}',
                 capt_srgb, est_srgb, est_depth_m, depth_kw, gt)
    print(f'saved results to {out_dir}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, default='data/sample_img/IMX585_test.jpg')
    parser.add_argument('--raw_path', type=str, default=None,
                        help='reconstruct a real RGGB raw capture instead of simulating one from --img_path')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='default: data/result/<image>_<experiment>_<depth_mode>_<simulator>')

    # model selection: --ckpt_path wins; otherwise the checkpoint is looked up from the experiment name
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--experiment', type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--log_root', type=str, default=os.path.join('data', 'logs'))
    parser.add_argument('--which_ckpt', choices=['best', 'last'], default='best',
                        help='best = lowest val_loss checkpoint of the experiment')

    # scene depth used to simulate the coded capture
    parser.add_argument('--depth_mode', choices=['depth_anything', 'ramp', 'plane', 'file'], default='depth_anything')
    parser.add_argument('--da_weights', type=str,
                        default=os.path.join('data', 'weights', 'depth_anything_v2_metric_vkitti_vitb.pth'),
                        help='Depth Anything V2 metric checkpoint (vkitti = outdoor, hypersim = indoor)')
    parser.add_argument('--da_input_size', type=int, default=518,
                        help='shorter side fed to Depth Anything; larger keeps finer detail, slower')
    parser.add_argument('--fit_depth_range', action='store_true',
                        help='map the scene inverse depth onto the trained range instead of clamping it')
    parser.add_argument('--plane_depth', type=float, default=2.0, help='[m], for --depth_mode plane')
    parser.add_argument('--ramp_near', type=float, default=None, help='[m] at the bottom row (default: min_depth)')
    parser.add_argument('--ramp_far', type=float, default=None, help='[m] at the top row (default: max_depth)')
    parser.add_argument('--depth_path', type=str, default=None, help='metric depth map for --depth_mode file')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='multiplies the depth file to get meters')

    # capture simulation
    parser.add_argument('--simulator', choices=['independent', 'model'], default='independent')
    parser.add_argument('--resize', type=float, default=1.0, help='rescale the photo before simulating')
    parser.add_argument('--sim_supersample', type=int, default=2, help='scene grid is this much finer than pixels')
    parser.add_argument('--sim_layers', type=int, default=64, help='depth layers of the independent simulator')
    parser.add_argument('--sim_psf_size', type=int, default=128, help='PSF window [sensor px]')
    parser.add_argument('--sim_height_scale', type=float, default=1.0, help='DOE height error (1 = nominal)')
    parser.add_argument('--sim_focus_offset', type=float, default=0.0, help='focus distance error [m]')
    parser.add_argument('--full_well', type=float, default=10000., help='[e-]; exposure puts highlights at 90%%')
    parser.add_argument('--read_noise', type=float, default=3., help='[e-]')
    parser.add_argument('--bit_depth', type=int, default=12)
    parser.add_argument('--noise_sigma', type=float, default=0.003,
                        help='Gaussian noise std for --simulator model (training sampled 0.001-0.005)')

    # real raw input
    parser.add_argument('--black_level', type=float, default=0.)
    parser.add_argument('--white_level', type=float, default=None, help='default: max of the file dtype')
    parser.add_argument('--wb_gains', type=float, nargs=2, default=[1., 1.], metavar=('R', 'B'),
                        help='white balance gains for red and blue (applied after demosaicking)')

    # reconstruction
    parser.add_argument('--tta', action='store_true', help='average over 4 flips (4x memory and time)')
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    main(parser.parse_args())
