"""Metric monocular depth estimation with Depth Anything V2 (vendored in third_party/depth_anything_v2)."""
import os

import numpy as np
import torch

from third_party.depth_anything_v2.dpt import DepthAnythingV2

# From the upstream metric_depth/README.md
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
# Output range each metric checkpoint was trained with: Hypersim (indoor) 20 m, Virtual KITTI (outdoor) 80 m.
DATASET_MAX_DEPTH = {'hypersim': 20., 'vkitti': 80.}


def _from_filename(weights_path, options, what):
    name = os.path.basename(weights_path)
    found = [k for k in options if k in name]
    if len(found) != 1:
        raise ValueError(f'Cannot tell the {what} from the weights filename "{name}"; pass it explicitly.')
    return found[0]


@torch.no_grad()
def estimate_metric_depth(img_rgb, weights_path, encoder=None, max_depth=None, input_size=518, device='cuda'):
    """
    Args:
        img_rgb: H x W x 3 float RGB image in [0, 1] (sRGB)
        weights_path: a Depth Anything V2 *metric* checkpoint (depth_anything_v2_metric_<dataset>_<encoder>.pth)
        encoder, max_depth: inferred from the checkpoint filename when None
        input_size: shorter side the image is resized to for the network (a multiple of 14 works best)

    Returns:
        H x W float32 depth map in meters
    """
    encoder = encoder or _from_filename(weights_path, MODEL_CONFIGS, 'encoder')
    if max_depth is None:
        max_depth = DATASET_MAX_DEPTH[_from_filename(weights_path, DATASET_MAX_DEPTH, 'training dataset')]

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder], max_depth=max_depth)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model = model.to(device).eval()

    # Depth Anything's preprocessing expects an 8-bit BGR image (as read by cv2.imread).
    img_bgr = (255 * np.clip(img_rgb, 0, 1)).round().astype(np.uint8)[..., ::-1]
    x, (h, w) = model.image2tensor(img_bgr, input_size)
    depth = model(x.to(device))  # image2tensor picks its own device, so move it explicitly
    depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode='bilinear', align_corners=True)[0, 0]
    depth = depth.cpu().numpy().astype(np.float32)

    del model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return depth
