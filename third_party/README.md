# Third-party code

## `depth_anything_v2/`

The metric-depth model code from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
(`metric_depth/depth_anything_v2/`), copied unmodified. Apache-2.0, see `depth_anything_v2/LICENSE`.
It is used by `util/monodepth.py` to estimate a metric depth map for an ordinary RGB photo, so that
`run_snapshotdepth_on_rgb_image.py --depth_mode depth_anything` can simulate a coded capture of it.

The weights are not part of this repo. Download a metric checkpoint (e.g.
`depth_anything_v2_metric_vkitti_vitb.pth` for outdoor scenes, `..._hypersim_...` for indoor) from the
upstream README and put it, or a symlink to it, under `data/weights/`. Note the ViT-B/L/G weights are
licensed CC-BY-NC-4.0 (ViT-S is Apache-2.0).
