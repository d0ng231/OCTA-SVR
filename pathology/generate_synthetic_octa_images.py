"""
Directly runnable script to generate synthetic OCTA images.

Pipeline:
  1) Generate vessel graphs (CSV) using the statistical simulator
  2) Optionally render raw MIP images (grayscale) from the graphs
  3) Optionally apply the pretrained GAN to create realistic images
  4) Optionally render label maps from the graphs

Examples:
  # Generate 50 graphs under ./generated/vessel_graphs and GAN images under ./generated/images
  python generate_synthetic_octa_images.py \
      --num_samples 50 \
      --use_gan \
      --gan_config /path/to/gan/config.yml \
      --gan_epoch 150

  # Generate only graphs and labels
  python generate_synthetic_octa_images.py \
      --num_samples 50 \
      --render_labels \
      --resolution 1216,1216,16

Notes:
  - The GAN step mirrors README Option B (Local python): it calls test.py with overrides
    so you don’t have to craft the command yourself.
  - For labels, we rasterize the graphs using the same utilities as visualize_vessel_graphs.py.


  Parallel graph generation
generate_vlm_dataset.py now spawns multiple processes to build graphs in parallel.
Flag: --workers (default -1 = all CPU cores but one).
Also disables stats to avoid slow Matplotlib I/O.
Parallel graph generation (single-runner)
generate_synthetic_octa_images.py now supports --workers to parallelize --num_samples.
Faster GAN inference on GPU
The dataset generator calls test.py once across all CSVs and sets --num_workers for dataloading.
Pass --use_gan --gan_device cuda to use your GPU.
Commands

VLM pairs (parallel CPU, GPU for GAN)
uv run python generate_vlm_dataset.py --num_samples 200 --out_dir ./vlm_dataset --workers -1 --use_gan --gan_config /path/to/gan/config.yml --gan_epoch 150 --gan_device cuda
Grayscale only (parallel CPU)
uv run python generate_vlm_dataset.py --num_samples 200 --out_dir ./vlm_dataset_gray --workers -1
Direct graph generation (parallel CPU)
uv run python generate_synthetic_octa_images.py --num_samples 50 --workers -1 --physical_dropout 0.6,0.45,0.12 --physical_neovasc 0.72,0.45,0.08


"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
import warnings
from glob import glob
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

# Make local imports robust regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Standardized default output roots (used for CLI + --out_dir aliasing)
DEFAULT_GRAPH_OUT_DIR = os.path.join(SCRIPT_DIR, "generated", "vessel_graphs")
DEFAULT_IMAGE_OUT_DIR = os.path.join(SCRIPT_DIR, "generated", "images")
DEFAULT_LABEL_OUT_DIR = os.path.join(SCRIPT_DIR, "generated", "labels")

from vessel_graph_generation.forest import Forest
from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.greenhouse_pathology import GreenhouseDropout
try:
    from vessel_graph_generation.nv_growth import grow_nv_branches
except Exception:
    grow_nv_branches = None  # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from vessel_graph_generation.utilities import prepare_output_dir, read_config
from vessel_graph_generation import tree2img


def _crop_black_borders_to_square(img):
    """
    Crop black borders from image and return a centered square crop.
    This handles FAZ shift padding artifacts by:
    1. Finding non-black content boundaries
    2. Using the smaller dimension as square side length
    3. Center-cropping to square within valid content area
    
    Args:
        img: PIL Image or numpy array
        
    Returns:
        PIL Image: Cropped square image with no black borders
    """
    try:
        from PIL import Image
        import numpy as np
        
        # Convert to PIL if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        arr = np.array(img)
        
        # Find content boundaries (non-black regions)
        if len(arr.shape) == 3:
            non_black = np.any(arr > 0, axis=-1)
        else:
            non_black = arr > 0
        
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)
        
        if not rows.any() or not cols.any():
            # No content found, return as-is
            return img
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Calculate valid content dimensions
        valid_height = y_max - y_min + 1
        valid_width = x_max - x_min + 1
        
        # Use smaller dimension to create square (avoid stretching)
        side = min(valid_height, valid_width)
        
        # Center the square crop within valid content area
        # Make sure crop stays within [y_min, y_max+1] and [x_min, x_max+1]
        center_y = y_min + valid_height // 2
        center_x = x_min + valid_width // 2
        
        crop_y_min = max(y_min, center_y - side // 2)
        crop_y_max = crop_y_min + side
        crop_x_min = max(x_min, center_x - side // 2)
        crop_x_max = crop_x_min + side
        
        # Ensure crop doesn't exceed valid bounds
        if crop_y_max > y_max + 1:
            crop_y_max = y_max + 1
            crop_y_min = crop_y_max - side
        if crop_x_max > x_max + 1:
            crop_x_max = x_max + 1
            crop_x_min = crop_x_max - side
        
        # Crop to square
        img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
        return img_cropped
    except Exception as e:
        print(f"[warn] Failed to crop black borders: {e}")
        return img


def _apply_non_gan_noise(img: np.ndarray, cfg: object, seed: int | None = None) -> np.ndarray:
    """Apply handcrafted OCTA noise model (ControlPoint beta noise) when GAN is disabled."""
    if cfg is None:
        return img
    if isinstance(cfg, dict):
        if not bool(cfg.get("enable", True)):
            return img
        noise_cfg: Dict[str, Any] = dict(cfg)
    elif isinstance(cfg, bool):
        if not cfg:
            return img
        noise_cfg = {}
    else:
        noise_cfg = {}

    try:
        import torch
        import torch.nn.functional as F
        from models.noise_model import NoiseModel
    except Exception:
        warnings.warn("NoiseModel dependencies missing; returning raw raster.", RuntimeWarning)
        return img

    device = torch.device(noise_cfg.get("device", "cpu"))
    grid_size = noise_cfg.get("grid_size", (9, 9))
    lambda_delta = float(noise_cfg.get("lambda_delta", 1.0))
    lambda_speckle = float(noise_cfg.get("lambda_speckle", 0.7))
    lambda_gamma = float(noise_cfg.get("lambda_gamma", 0.3))
    alpha = float(noise_cfg.get("alpha", 0.2))
    downsample_factor = int(max(1, int(noise_cfg.get("downsample_factor", 1))))
    background_kernel = int(max(1, int(noise_cfg.get("background_kernel", 41))))
    if background_kernel % 2 == 0:
        background_kernel += 1
    blend = float(noise_cfg.get("blend", 0.6))
    blend = max(0.0, min(1.0, blend))
    match_stats = bool(noise_cfg.get("match_mean_std", True))

    torch.manual_seed(seed if seed is not None else random.randint(0, 2**31 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed if seed is not None else random.randint(0, 2**31 - 1))

    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device=device, non_blocking=True)

    if background_kernel > 1:
        pad = background_kernel // 2
        background = F.avg_pool2d(
            F.pad(img_tensor, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=background_kernel,
            stride=1,
        )
    else:
        background = img_tensor.clone()

    noise_model = NoiseModel(
        grid_size=tuple(grid_size),
        lambda_delta=lambda_delta,
        lambda_speckle=lambda_speckle,
        lambda_gamma=lambda_gamma,
        alpha=alpha,
    )
    noise_model.requires_grad_(False)
    noise_model.to(device=device, non_blocking=True)

    try:
        with torch.no_grad():
            noisy = noise_model.forward(img_tensor, background, adversarial=False, downsample_factor=downsample_factor)
    except Exception as exc:  # Fallback to raw raster if the noise model cannot run (e.g., missing backend)
        warnings.warn(f"NoiseModel application failed ({exc}); returning raw raster.", RuntimeWarning)
        return img

    noisy = noisy.clamp(0.0, 1.0)
    if match_stats:
        raw_mean = float(img_tensor.mean())
        raw_std = float(img_tensor.std())
        noisy_mean = float(noisy.mean())
        noisy_std = float(noisy.std())
        if noisy_std > 1e-6:
            noisy = (noisy - noisy_mean) * (raw_std / noisy_std) + raw_mean
        else:
            noisy = noisy * 0.0 + raw_mean
        noisy = noisy.clamp(0.0, 1.0)

    if blend < 1.0:
        noisy = torch.lerp(img_tensor, noisy, blend)

    noisy = noisy.clamp(0.0, 1.0).squeeze().cpu().numpy()
    return (noisy * 255.0).astype(np.uint8)


def _generate_single_sample(
    config: dict,
    base_out_dir: str,
    physical_dropout: tuple[float, float, float] | list[tuple[float, float, float]] | None = None,
    physical_strength: float | None = None,
    physical_neovasc: tuple[float, float, float] | None = None,
    skip_venous: bool = False,
    write_config_per_sample: bool = True,
    save_pathology_masks: bool = False,
    mask_threshold: float = 0.5,
    mask_expand_px: int = 0,
    mask_vessel_sparing_px: int = 0,
    dropout_cfg_extra: dict | None = None,
    neovasc_cfg_extra: dict | None = None,
    faz_center_override: tuple[float, float] | None = None,
    view_shift_norm: tuple[float, float] | None = None,
) -> str:
    """Generate a single vessel graph sample and optional images.

    Returns:
        The concrete output directory used for this sample (timestamped subdir).
    """
    # Work on a local copy of the config so we can record actual sampled params
    cfg_local = copy.deepcopy(config)

    # If the caller provides a per-sample FAZ center, override and suppress jitter here
    if faz_center_override is not None:
        try:
            ghc = cfg_local.setdefault('Greenhouse', {})
            ghc['FAZ_center'] = [float(faz_center_override[0]), float(faz_center_override[1])]
            # Disable additional jitter to keep consistency with sampling logic upstream
            ghc['FAZ_center_jitter'] = 0.0
            ghc['FAZ_center_jitter_std'] = [0.0, 0.0]
        except Exception:
            pass

    # Initialize greenhouse and forests
    if physical_dropout is not None or physical_neovasc is not None:
        dropout_cfg = None
        nv_cfg = None
        if physical_dropout is not None:
            if isinstance(physical_dropout, (list, tuple)) and len(physical_dropout) > 0 and isinstance(physical_dropout[0], (list, tuple)):
                regs = []
                # Optional per-region strength sampling if a range is provided in extras
                sr_lo = sr_hi = None
                if dropout_cfg_extra and isinstance(dropout_cfg_extra.get("strength_range"), (list, tuple)) and len(dropout_cfg_extra.get("strength_range")) == 2:
                    try:
                        sr_lo = float(dropout_cfg_extra["strength_range"][0])
                        sr_hi = float(dropout_cfg_extra["strength_range"][1])
                    except Exception:
                        sr_lo = sr_hi = None
                for cx, cy, rr in physical_dropout:  # type: ignore
                    # Decide region strength: prefer per-region sampling from range, else fallback to physical_strength or extras' strength
                    if sr_lo is not None and sr_hi is not None and sr_hi >= sr_lo:
                        r_strength = float(np.random.uniform(sr_lo, sr_hi))
                    elif physical_strength is not None:
                        r_strength = float(physical_strength)
                    else:
                        # fallback: single provided strength in extras, else 1.0
                        r_strength = float(dropout_cfg_extra.get("strength", 1.0)) if dropout_cfg_extra else 1.0
                    cfg_i = {
                        "center": (float(cx), float(cy)),
                        "radius": float(rr),
                        "strength": float(max(0.0, min(1.0, r_strength))),
                    }
                    if dropout_cfg_extra:
                        # Do not override per-region strength; skip keys that would clobber it
                        cfg_i.update({
                            k: v for k, v in dropout_cfg_extra.items()
                            if v is not None and k not in ("strength_range", "strength")
                        })
                    regs.append(cfg_i)
                dropout_cfg = regs
            else:
                cx, cy, rr = physical_dropout  # type: ignore
                # Allow range-based sampling even for single region
                r_strength: float
                sr_lo = sr_hi = None
                if dropout_cfg_extra and isinstance(dropout_cfg_extra.get("strength_range"), (list, tuple)) and len(dropout_cfg_extra.get("strength_range")) == 2:
                    try:
                        sr_lo = float(dropout_cfg_extra["strength_range"][0])
                        sr_hi = float(dropout_cfg_extra["strength_range"][1])
                    except Exception:
                        sr_lo = sr_hi = None
                if sr_lo is not None and sr_hi is not None and sr_hi >= sr_lo:
                    r_strength = float(np.random.uniform(sr_lo, sr_hi))
                elif physical_strength is not None:
                    r_strength = float(physical_strength)
                else:
                    r_strength = float(dropout_cfg_extra.get("strength", 1.0)) if dropout_cfg_extra else 1.0
                dropout_cfg = {
                    "center": (float(cx), float(cy)),
                    "radius": float(rr),
                    "strength": float(max(0.0, min(1.0, r_strength))),
                }
                if dropout_cfg_extra:
                    # Merge shape/behavior parameters but keep per-region strength choice
                    dropout_cfg.update({k: v for k, v in dropout_cfg_extra.items() if v is not None and k not in ("strength_range", "strength")})
        if physical_neovasc is not None:
            nx, ny, nr = physical_neovasc
            nv_cfg = {
                "center": (float(nx), float(ny)),
                "radius": float(nr),
            }
            if neovasc_cfg_extra:
                nv_cfg.update({k: v for k, v in neovasc_cfg_extra.items() if v is not None})
        greenhouse = GreenhouseDropout(
            cfg_local['Greenhouse'],
            dropout_cfg=dropout_cfg or {"center": (0.5, 0.5), "radius": 0.0, "strength": 0.0},
            neovasc_cfg=nv_cfg,
        )
    else:
        greenhouse = Greenhouse(cfg_local['Greenhouse'])

    # Prepare output directory per sample (timestamp + uuid inside base)
    out_dir = prepare_output_dir({'directory': base_out_dir})
    os.makedirs(out_dir, exist_ok=True)
    if write_config_per_sample:
        # Record the actually used FAZ center/radius into the saved config for transparency
        try:
            ghc = cfg_local.setdefault('Greenhouse', {})
            ghc['FAZ_center'] = [float(greenhouse.FAZ_center[0]), float(greenhouse.FAZ_center[1])]
            # Save realized radius as a convenience
            ghc['FAZ_radius'] = float(greenhouse.FAZ_radius)
        except Exception:
            pass
        with open(os.path.join(out_dir, 'config.yml'), 'w') as f:
            yaml.dump(cfg_local, f)

    arterial_forest = Forest(
        cfg_local['Forest'], greenhouse.d, greenhouse.r, greenhouse.simspace,
        nerve_center=greenhouse.nerve_center, nerve_radius=greenhouse.nerve_radius
    )
    venous_forest = None if skip_venous else Forest(
        cfg_local['Forest'], greenhouse.d, greenhouse.r, greenhouse.simspace,
        arterial=False, nerve_center=greenhouse.nerve_center, nerve_radius=greenhouse.nerve_radius
    )

    greenhouse.set_forests(arterial_forest, venous_forest)
    greenhouse.develop_forest()

    if cfg_local["output"].get("save_stats"):
        greenhouse.save_stats(out_dir)

    # Optional NV-only radius boost for visibility in grayscale (does not affect growth)
    try:
        render_cfg0 = cfg_local.get("Render", {}) if isinstance(cfg_local, dict) else {}
        nv_radius_scale = float(render_cfg0.get("nv_radius_scale", 1.75))
    except Exception:
        nv_radius_scale = 1.75

    # Build edge lists; thicken growth‑time neovascular edges only (legacy flag)
    art_edges = []
    for tree in arterial_forest.get_trees():
        for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False):
            rad = float(getattr(current_node, "radius", 0.0))
            try:
                if bool(getattr(current_node, 'is_neovascular', False)):
                    rad *= nv_radius_scale
            except Exception:
                pass
            art_edges.append({
                "node1": current_node.position,
                "node2": current_node.get_proximal_node().position,
                "radius": rad,
            })

    ven_edges = [] if venous_forest is None else [{
        "node1": current_node.position,
        "node2": current_node.get_proximal_node().position,
        "radius": current_node.radius,
    } for tree in venous_forest.get_trees() for current_node in tree.get_tree_iterator(exclude_root=True, only_active=False)]

    # New NV generation via dedicated growth engine
    nv_polylines_norm: list[list[tuple[float, float]]] = []
    nv_host_regions: list[tuple[float, float, float]] = []
    try:
        _nv_cfg = (cfg_local.get('Pathology', {}) or {}).get('NV', {}) if isinstance(cfg_local, dict) else {}
        _sev = float(_nv_cfg.get('severity', 0.5) or 0.5)
        _p = float(_nv_cfg.get('probability', 0.6))
        _enabled = bool(_nv_cfg.get('augmentation', True))
    except Exception:
        _sev, _p, _enabled = (0.5, 0.6, True)

    try:
        if _enabled and (np.random.uniform(0.0, 1.0) < _p) and (grow_nv_branches is not None):
            new_edges, nv_polylines_norm, nv_host_regions = grow_nv_branches(art_edges, greenhouse, severity=_sev)
            if new_edges:
                art_edges.extend(new_edges)
    except Exception:
        nv_polylines_norm, nv_host_regions = ([], [])

    # Defer saving vessel graph CSV until after NV fallback drawing so those edges can be included

    # Optionally save 3D volumes
    save_3d = cfg_local["output"].get("save_3D_volumes")
    if save_3d:
        volume_dimension = [int(d) for d in greenhouse.simspace.shape * config['output']['image_scale_factor']]
        radius_list: List[float] = []
        art_mat, _ = tree2img.voxelize_forest(art_edges, volume_dimension, radius_list)
        ven_mat, _ = tree2img.voxelize_forest(ven_edges, volume_dimension, radius_list)
        art_ven_mat_gray = np.maximum(art_mat, ven_mat).astype(np.uint8)

        if save_3d == "npy":
            np.save(os.path.join(out_dir, 'art_ven_img_gray.npy'), art_ven_mat_gray)
        elif save_3d == "nifti":
            import nibabel as nib  # lazy import
            nifti = nib.Nifti1Image(art_ven_mat_gray, np.eye(4))
            nib.save(nifti, os.path.join(out_dir, "art_ven_img_gray.nii.gz"))
        else:
            raise ValueError("save_3D_volumes must be one of [None, 'npy', 'nifti']")

    # Determine 2D image resolution used for 2D outputs (image and masks)
    volume_dimension = [int(d) for d in greenhouse.simspace.shape * cfg_local['output']['image_scale_factor']]
    img_res = [*volume_dimension]
    proj_axis = cfg_local["output"].get("proj_axis", 2)
    del img_res[proj_axis]
    # Optional rendering clamp (does not affect growth): cap draw radius in mm
    render_cfg = cfg_local.get("Render", {}) if isinstance(cfg_local, dict) else {}
    try:
        render_max_r_mm = float(render_cfg.get("max_radius_mm")) if (render_cfg and render_cfg.get("max_radius_mm") is not None) else None
    except Exception:
        render_max_r_mm = None

    # Optionally save 2D MIP image
    if cfg_local["output"].get("save_2D_image"):
        radius_list: List[float] = []
        # Render with the precise 2D rasterizer so growth projects correctly.
        art_mat,_ = tree2img.rasterize_forest(art_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
        ven_mat,_ = tree2img.rasterize_forest(ven_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0)) if ven_edges else (np.zeros(img_res, dtype=np.uint8), {})
        art_ven_raw = np.maximum(art_mat, ven_mat).astype(np.uint8)
        tree2img.save_2d_img(art_ven_raw, out_dir, "art_ven_img_gray")

        noise_cfg = render_cfg.get("non_gan_noise") if isinstance(render_cfg, dict) else None
        noise_enabled = False
        if isinstance(noise_cfg, dict):
            noise_enabled = bool(noise_cfg.get("enable", True))
        elif isinstance(noise_cfg, bool):
            noise_enabled = noise_cfg
        elif noise_cfg is not None:
            noise_enabled = True

        if noise_enabled:
            noisy_img = _apply_non_gan_noise(art_ven_raw, noise_cfg, random.randint(0, 2**32 - 1))
            tree2img.save_2d_img(noisy_img, out_dir, "art_ven_img_gray_noisy")

        # Also produce a panned, square-cropped grayscale for GAN input consistency (even when no pathology overlays exist)
        try:
            if view_shift_norm is not None:
                from PIL import Image as _PILImage
                Hh, Ww = int(art_ven_raw.shape[0]), int(art_ven_raw.shape[1])
                sx, sy = float(view_shift_norm[0]), float(view_shift_norm[1])
                dx_px = int(round(sx * Ww))
                dy_px = int(round(sy * Hh))
                base_gray = _PILImage.fromarray(art_ven_raw).convert("L")
                canvas = _PILImage.new("L", (Ww, Hh), 0)
                # Paste with offset (dx along columns → +x; dy along rows → +y)
                canvas.paste(base_gray, (dx_px, dy_px))
                # Compute valid rectangle and center a square crop inside it
                valid_w = max(1, Ww - abs(dx_px))
                valid_h = max(1, Hh - abs(dy_px))
                side = int(max(1, min(valid_w, valid_h)))
                vx0 = max(0, dx_px)
                vy0 = max(0, dy_px)
                x0 = int(vx0 + max(0, (valid_w - side) // 2))
                y0 = int(vy0 + max(0, (valid_h - side) // 2))
                x1 = int(min(Ww, x0 + side))
                y1 = int(min(Hh, y0 + side))
                if (x1 - x0) > 0 and (y1 - y0) > 0:
                    canvas = canvas.crop((x0, y0, x1, y1))
                canvas.save(os.path.join(out_dir, "art_ven_img_gray_panned.png"))
        except Exception:
            pass

    # Always record per-sample pathology JSON; do not generate or save masks.
    if isinstance(greenhouse, GreenhouseDropout):
        try:
            from PIL import Image  # lazy import
            import numpy as _np

            def _dilate_bool_sq(mask_bool: _np.ndarray, radius: int) -> _np.ndarray:
                """Fast square-structuring dilation with 8-neighborhood using rolls.

                Uses Chebyshev distance (square) approximation; good enough for modest expansion.
                """
                if radius <= 0:
                    return mask_bool
                m = mask_bool.copy()
                for _ in range(int(radius)):
                    m = (
                        m
                        | _np.roll(mask_bool, 1, axis=0)
                        | _np.roll(mask_bool, -1, axis=0)
                        | _np.roll(mask_bool, 1, axis=1)
                        | _np.roll(mask_bool, -1, axis=1)
                        | _np.roll(mask_bool, (1, 1), axis=(0, 1))
                        | _np.roll(mask_bool, (1, -1), axis=(0, 1))
                        | _np.roll(mask_bool, (-1, 1), axis=(0, 1))
                        | _np.roll(mask_bool, (-1, -1), axis=(0, 1))
                    )
                    mask_bool = m
                return m

            def _mask_irregular(greenhouse_obj, Hh: int, Ww: int) -> _np.ndarray:
                """Render a natural-looking irregular dropout mask from greenhouse.dropouts.

                Uses a small set of angular harmonics and mild ellipticity; excludes FAZ region.
                Returns a uint8 array (0/255) of shape (Hh, Ww). Returns zeros if no dropouts.
                """
                try:
                    drop_regs = list(getattr(greenhouse_obj, 'dropouts', []) or [])
                except Exception:
                    drop_regs = []
                if not drop_regs:
                    return _np.zeros((Hh, Ww), dtype=_np.uint8)
                Xm, Ym = _np.meshgrid(_np.linspace(0.0, 1.0, Hh, dtype=_np.float32),
                                      _np.linspace(0.0, 1.0, Ww, dtype=_np.float32), indexing='ij')
                mask_ir = _np.zeros((Hh, Ww), dtype=_np.bool_)
                for idx, reg in enumerate(drop_regs):
                    r0 = float(getattr(reg, 'r0', 0.0))
                    if not (r0 > 0.0):
                        continue
                    cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5))
                    dx = Xm - cx; dy = Ym - cy
                    dist = _np.sqrt(dx * dx + dy * dy)
                    theta = _np.arctan2(dy, dx)
                    # Angular harmonics with small amplitudes
                    kset = _np.array([3, 5, 7], dtype=_np.float32)
                    phases = _np.mod(_np.array([1.1 + 0.7 * idx, 2.2 + 1.3 * idx, 3.3 + 0.9 * idx], dtype=_np.float32), 6.283185307179586)
                    amps = _np.array([0.06, 0.045, 0.03], dtype=_np.float32)
                    perturb = _np.zeros_like(theta, dtype=_np.float32)
                    for k, a, ph in zip(kset, amps, phases):
                        perturb += a * _np.sin(k * theta + ph)
                    ex = 1.0 + 0.10 * _np.sin(theta * 2.0 + 0.35 * idx)
                    r_b = r0 * _np.clip(1.0 + perturb, 0.65, 1.45) * ex
                    mask_ir |= (dist <= r_b)
                # Exclude FAZ with safety margin
                try:
                    fr = float(getattr(greenhouse_obj, 'FAZ_radius', 0.06))
                    fx, fy = float(getattr(greenhouse_obj, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(greenhouse_obj, 'FAZ_center', (0.5, 0.5))[1])
                    mask_ir &= (_np.sqrt((Xm - fx) ** 2 + (Ym - fy) ** 2) >= (1.2 * fr))
                except Exception:
                    pass
                mask_ir = _dilate_bool_sq(mask_ir, 1)
                return _np.asarray(mask_ir, dtype=_np.uint8) * 255

            # New behavior: abandon binary mask. Create a visualization overlay and save
            # dropout/MA parameters for this sample instead of a mask.
            # Prefer saved grayscale MIP to avoid an extra rasterize pass
            from PIL import ImageDraw, ImageFont, ImageFilter
            saved_mip = os.path.join(out_dir, "art_ven_img_gray.png")
            if os.path.isfile(saved_mip):
                base_img = Image.open(saved_mip).convert("RGB")
                H, W = base_img.size[1], base_img.size[0]
            else:
                radius_list_v: list[float] = []
                art_mat_v, _ = tree2img.rasterize_forest(art_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list_v, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                if ven_edges:
                    ven_mat_v, _ = tree2img.rasterize_forest(ven_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list_v, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                    img2d_g = _np.maximum(art_mat_v, ven_mat_v).astype(_np.uint8)
                else:
                    img2d_g = art_mat_v.astype(_np.uint8)
                H, W = int(img2d_g.shape[0]), int(img2d_g.shape[1])
                base_img = Image.fromarray(img2d_g).convert("RGB")
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            # NV-only layer for subtle feathering before compositing to the main overlay
            overlay_nv = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw_nv = ImageDraw.Draw(overlay_nv)
            # Secondary NV white layer (RGBA), later alpha-composited over base_img for dataset image
            nv_white_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw_white = ImageDraw.Draw(nv_white_layer)
            # Track whether any content (NV or MA) has been drawn into the white overlay
            ma_white_drawn = False
            # Placeholder for the post-composite white image
            base_img_white = base_img.copy()

            pathology = {"dropouts": [], "microaneurysms": []}
            # Draw dropouts (center + radius label only). Skip zero-radius placeholders.
            dropouts_all = list(getattr(greenhouse, 'dropouts', []) or [])
            dropouts_list = [reg for reg in dropouts_all if float(getattr(reg, 'r0', 0.0)) > 1e-6]
            # Identify largest dropout by radius (for emphasis)
            largest_idx = None
            if dropouts_list:
                try:
                    radii_tmp = [float(getattr(r, 'r0', 0.0)) for r in dropouts_list]
                    largest_idx = int(max(range(len(radii_tmp)), key=lambda i: radii_tmp[i]))
                except Exception:
                    largest_idx = None
            # Text font for annotations (fallback to default if no TTF available)
            try:
                _font = ImageFont.load_default()
            except Exception:
                _font = None

            # For readability: annotate radius text for all regions only if count is small
            _annotate_all = len(dropouts_list) <= 12

            # Helper: approximate irregular dropout boundary and draw translucent fill + outline
            def _draw_dropout_region_irregular(reg_obj) -> None:
                try:
                    cxm = float(getattr(reg_obj, 'cx', 0.5)); cym = float(getattr(reg_obj, 'cy', 0.5))
                    r0m = float(getattr(reg_obj, 'r0', 0.12))
                    N = 96
                    pts = []
                    for k in range(N):
                        th = (2.0 * _np.pi) * (k / N)
                        # coarse march outward then binary refine using inside_field
                        lo, hi = 0.0, r0m * 1.6
                        for _ in range(10):
                            mid = 0.5 * (lo + hi)
                            xm = cxm + mid * _np.cos(th)
                            ym = cym + mid * _np.sin(th)
                            try:
                                val = float(_np.asarray(reg_obj.inside_field(_np.array([xm]), _np.array([ym])))[0])
                            except Exception:
                                # radial fallback
                                val = 1.0 if mid <= r0m else 0.0
                            if val > 0.05:
                                lo = mid
                            else:
                                hi = mid
                        rr = max(0.0, lo)
                        ry = int(_np.clip(round((cxm + rr * _np.sin(th)) * H), 0, H - 1))
                        rx = int(_np.clip(round((cym + rr * _np.cos(th)) * W), 0, W - 1))
                        pts.append((rx, ry))
                    if len(pts) >= 3:
                        try:
                            draw.polygon(pts, fill=(255, 64, 64, 45), outline=(255, 80, 80, 220))
                        except Exception:
                            pass
                except Exception:
                    pass

            for i, reg in enumerate(dropouts_list):
                cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
                ds = float(getattr(reg, 'drop_strength', getattr(reg, 'strength', 1.0)))
                row = int(_np.clip(round(cx * H), 0, H - 1)); col = int(_np.clip(round(cy * W), 0, W - 1))
                r_px = int(max(1, round(r0 * min(H, W))))
                if r_px <= 1:
                    continue
                # Record only real dropouts
                pathology["dropouts"].append({"center": [cx, cy], "radius": r0, "strength": ds})
                # Irregular filled footprint + subtle outline
                _draw_dropout_region_irregular(reg)

                # Mark center with a small filled dot
                dot_px = max(2, int(max(2, r_px // 18)))
                try:
                    draw.ellipse([col - dot_px, row - dot_px, col + dot_px, row + dot_px], fill=(255, 64, 64, 255), outline=(255, 255, 255, 255))
                except Exception:
                    pass

                # Optionally annotate radius next to the center (short offset)
                show_text = _annotate_all or (largest_idx is not None and i == largest_idx)
                if show_text and _font is not None:
                    label = f"r={r0:.3f}"
                    # Place label right next to center
                    xt = int(_np.clip(col + (dot_px + 6), 2, W - 2))
                    yt = int(_np.clip(row - (dot_px + 6), 2, H - 12))
                    try:
                        draw.text((xt, yt), label, fill=(255, 200, 200, 255), font=_font, stroke_width=2, stroke_fill=(0, 0, 0, 200))
                    except TypeError:
                        # Older Pillow without stroke params
                        draw.text((xt, yt), label, fill=(255, 200, 200, 255), font=_font)

            # Extract and draw microaneurysms (yellow filled on vis; WHITE on M_ overlay; optional burn-in to gray)
            def _iter_ma_nodes(forest_obj):
                try:
                    for tree in forest_obj.get_trees():
                        for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                            if bool(getattr(node, 'is_microaneurysm', False)):
                                yield node
                except Exception:
                    return

            def _draw_irregular_blob(draw_obj, center_rc: tuple[int, int], base_r_px: int, fill_rgba, outline_rgba, seed: int | None = None, rnd_level: float = 0.0):
                """Draw a single random, connected blob within a small elliptical region.

                The shape is generated procedurally on a small patch as a connected boolean mask
                (union of random small disks inside an ellipse with optional carved holes), then
                converted to a boundary via polar sampling for a stable polygon. This produces
                varied concave/convex silhouettes beyond simple circles.
                """
                import random as _rand
                rng = _rand.Random(seed if seed is not None else (center_rc[0] * 73856093 ^ center_rc[1] * 19349663))
                try:
                    rl = max(0.0, min(1.0, float(rnd_level)))
                except Exception:
                    rl = 0.0

                # Patch size around center (square), keep modest for speed
                side = int(max(12, min(256, round(3.2 * float(max(1, base_r_px))))))
                if side % 2 == 0:
                    side += 1
                hh = ww = side
                cy = cx = side // 2
                Yg, Xg = _np.mgrid[0:hh, 0:ww]

                # Elliptical envelope parameters
                env_ax = float(base_r_px) * (0.9 + 0.35 * rng.random())
                env_ay = float(base_r_px) * (0.9 + 0.35 * rng.random())
                env_rot = rng.uniform(0.0, 2.0 * _np.pi)
                # Rotate grid
                X0 = Xg - cx
                Y0 = Yg - cy
                Xr = X0 * _np.cos(env_rot) + Y0 * _np.sin(env_rot)
                Yr = -X0 * _np.sin(env_rot) + Y0 * _np.cos(env_rot)
                ell = ((Xr / max(1e-3, env_ax)) ** 2 + (Yr / max(1e-3, env_ay)) ** 2) <= 1.0

                # Start with union of K random small disks inside the ellipse (ensure inside)
                K = int(rng.randint(2, 4) + round(4 * rl * rng.random()))
                mask = _np.zeros((hh, ww), dtype=_np.bool_)
                for _ in range(max(1, K)):
                    # sample center inside ellipse
                    for _try in range(20):
                        rx = rng.uniform(-0.8, 0.8) * env_ax
                        ry = rng.uniform(-0.8, 0.8) * env_ay
                        xx = int(round(cx + (rx * _np.cos(env_rot) - ry * _np.sin(env_rot))))
                        yy = int(round(cy + (rx * _np.sin(env_rot) + ry * _np.cos(env_rot))))
                        if 0 <= yy < hh and 0 <= xx < ww and ell[yy, xx]:
                            break
                    else:
                        xx, yy = cx, cy
                    rr = int(max(1, round(0.25 * base_r_px + rng.random() * 0.45 * base_r_px)))
                    yy0, xx0 = _np.ogrid[:hh, :ww]
                    disk = (yy0 - yy) ** 2 + (xx0 - xx) ** 2 <= rr * rr
                    mask |= (disk & ell)

                # Optionally carve a few random holes (concavities), but keep connectivity by keeping largest component
                Hn = int(max(0, rng.randint(0, 1) + round(3 * rl * rng.random())))
                for _ in range(Hn):
                    rx = rng.uniform(-0.6, 0.6) * env_ax
                    ry = rng.uniform(-0.6, 0.6) * env_ay
                    xx = int(round(cx + (rx * _np.cos(env_rot) - ry * _np.sin(env_rot))))
                    yy = int(round(cy + (rx * _np.sin(env_rot) + ry * _np.cos(env_rot))))
                    rr = int(max(1, round(0.18 * base_r_px + rng.random() * 0.35 * base_r_px)))
                    if 0 <= yy < hh and 0 <= xx < ww:
                        yy0, xx0 = _np.ogrid[:hh, :ww]
                        hole = (yy0 - yy) ** 2 + (xx0 - xx) ** 2 <= rr * rr
                        mask &= (~hole)

                # Mild smoothing/connection
                def _largest_component_bool(m: _np.ndarray) -> _np.ndarray:
                    # BFS to keep largest 8-connected component
                    H, W = m.shape
                    vis = _np.zeros_like(m, dtype=_np.uint8)
                    best_sz = 0
                    best = None
                    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                    for r in range(H):
                        row = m[r]
                        if not row.any():
                            continue
                        for c in _np.where(row & (vis[r] == 0))[0]:
                            if vis[r, c] or not m[r, c]:
                                continue
                            q = [(r, c)]
                            vis[r, c] = 1
                            qi = 0
                            sz = 0
                            while qi < len(q):
                                rr, cc = q[qi]; qi += 1; sz += 1
                                for dr, dc in neigh:
                                    r2 = rr + dr; c2 = cc + dc
                                    if 0 <= r2 < H and 0 <= c2 < W and not vis[r2, c2] and m[r2, c2]:
                                        vis[r2, c2] = 1
                                        q.append((r2, c2))
                            if sz > best_sz:
                                best_sz = sz
                                best = _np.zeros_like(m, dtype=_np.bool_)
                                for rr, cc in q:
                                    best[rr, cc] = True
                    return (best if best is not None else (m & m))

                mask = _dilate_bool_sq(mask, 1)
                mask = _largest_component_bool(mask)

                # Build boundary via polar sampling for a stable polygon
                N = int(48 + round(64 * rl))
                thetas = _np.linspace(0.0, 2.0 * _np.pi, N, endpoint=False)
                bpts = []
                for th in thetas:
                    step = 0.5
                    rmax = float(side)
                    # ray from center
                    rr = 0.0
                    last_in = (int(cy), int(cx))
                    while rr <= rmax:
                        x = cx + rr * _np.cos(th)
                        y = cy + rr * _np.sin(th)
                        iy = int(round(y)); ix = int(round(x))
                        if not (0 <= iy < hh and 0 <= ix < ww):
                            break
                        if not mask[iy, ix]:
                            break
                        last_in = (iy, ix)
                        rr += step
                    # Map to global coordinates
                    gy = int(center_rc[0] + (last_in[0] - cy))
                    gx = int(center_rc[1] + (last_in[1] - cx))
                    bpts.append((gx, gy))

                if len(bpts) >= 3:
                    draw_obj.polygon(bpts, fill=fill_rgba, outline=outline_rgba)

            # Resolve optional burn-in to grayscale for GAN/M_ (default True per request)
            try:
                _ma_burn_gray = bool(((config.get('Render', {}) or {}).get('ma_overlay_burn_gray', True)))
            except Exception:
                _ma_burn_gray = True

            # Ensure unique MA shapes per-instance by deriving a deterministic seed
            # from the sample-specific out_dir and each MA center. This avoids
            # accidental reuse across processes or samples.
            import os as _os, zlib as _zlib
            try:
                _sample_tag = _os.path.basename(out_dir.rstrip(_os.sep))
                _sample_salt = int(_zlib.adler32(_sample_tag.encode('utf-8')) & 0x7FFFFFFF)
            except Exception:
                _sample_salt = int(_np.random.randint(0, 2**31 - 1))

            def _seed_for_ma(_row: int, _col: int, _rpx: int, tweak: int = 0) -> int:
                return int(((_sample_salt ^ (_row * 73856093) ^ (_col * 19349663) ^ (_rpx * 83492791)) + tweak) & 0x7FFFFFFF)

            ma_seen_px: set[tuple[int, int]] = set()
            ma_records: list[tuple[float, float, float]] = []
            for node in _iter_ma_nodes(arterial_forest):
                try:
                    x = float(node.position[0]); y = float(node.position[1]); r_mm = float(getattr(node, 'radius', 0.012))
                except Exception:
                    continue
                row = int(_np.clip(round(x * H), 0, H - 1)); col = int(_np.clip(round(y * W), 0, W - 1))
                key_px = (row, col)
                if key_px in ma_seen_px:
                    continue
                ma_seen_px.add(key_px)
                pathology["microaneurysms"].append({"center": [x, y], "radius_mm": r_mm})
                ma_records.append((x, y, r_mm))
                r_px = int(max(1, round(r_mm * max(H, W))))
                # Unique, stable seed per MA instance to avoid any shape reuse
                _seed_inst = _seed_for_ma(row, col, r_px, tweak=0)
                _draw_irregular_blob(draw, (row, col), r_px,
                                     fill_rgba=(255, 220, 64, 180),
                                     outline_rgba=(255, 200, 32, 255),
                                     seed=_seed_inst,
                                     rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                # Also paint onto the white overlay (so M_ shows MA) and optionally burn into grayscale base
                try:
                    _draw_irregular_blob(draw_white, (row, col), r_px,
                                         fill_rgba=(255, 255, 255, 255),
                                         outline_rgba=(255, 255, 255, 255),
                                         seed=_seed_inst,
                                         rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                    ma_white_drawn = True
                except Exception:
                    pass
                if _ma_burn_gray:
                    try:
                        from PIL import ImageDraw as _ImageDraw
                        _draw_irregular_blob(_ImageDraw.Draw(base_img), (row, col), r_px,
                                             fill_rgba=(255, 255, 255), outline_rgba=(255, 255, 255),
                                             seed=_seed_inst,
                                             rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                    except Exception:
                        pass
            if venous_forest is not None:
                for node in _iter_ma_nodes(venous_forest):
                    try:
                        x = float(node.position[0]); y = float(node.position[1]); r_mm = float(getattr(node, 'radius', 0.012))
                    except Exception:
                        continue
                    row = int(_np.clip(round(x * H), 0, H - 1)); col = int(_np.clip(round(y * W), 0, W - 1))
                    key_px = (row, col)
                    if key_px in ma_seen_px:
                        continue
                    ma_seen_px.add(key_px)
                    pathology["microaneurysms"].append({"center": [x, y], "radius_mm": r_mm})
                    ma_records.append((x, y, r_mm))
                    r_px = int(max(1, round(r_mm * max(H, W))))
                    # Offset the seed slightly to avoid accidental clashes with arterial MAs
                    _seed_inst_v = _seed_for_ma(row, col, r_px, tweak=0x9E3779B)
                    _draw_irregular_blob(draw, (row, col), r_px,
                                         fill_rgba=(255, 220, 64, 180),
                                         outline_rgba=(255, 200, 32, 255),
                                         seed=_seed_inst_v,
                                         rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                    # White overlay and optional grayscale burn-in
                    try:
                        _draw_irregular_blob(draw_white, (row, col), r_px,
                                             fill_rgba=(255, 255, 255, 255),
                                             outline_rgba=(255, 255, 255, 255),
                                             seed=_seed_inst_v,
                                             rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                        ma_white_drawn = True
                    except Exception:
                        pass
                    if _ma_burn_gray:
                        try:
                            from PIL import ImageDraw as _ImageDraw
                            _draw_irregular_blob(_ImageDraw.Draw(base_img), (row, col), r_px,
                                                 fill_rgba=(255, 255, 255), outline_rgba=(255, 255, 255),
                                                 seed=_seed_inst_v,
                                                 rnd_level=float(getattr(greenhouse, 'ma_shape_randomness', 0.0)))
                        except Exception:
                            pass

            # Defer saving composed visualization until all overlays (incl. NV) are drawn below

            # NV visualization (overlay-only): we no longer render growth-time NV edges.
            # The tuft is drawn procedurally as a cyan overlay via draw_tuft_overlay below.
            # (No-op placeholder left intentionally to mark the change.)

            # Also write a concise YAML pathology spec for reporting
            def _quad_label(xm: float, ym: float) -> str:
                top = xm < 0.5
                left = ym < 0.5
                if top and left:
                    return "top-left"
                if top and not left:
                    return "top-right"
                if (not top) and (not left):
                    return "bottom-right"
                return "bottom-left"

            def _dedupe_nv_regions(regs: list[dict[str, object]]) -> list[dict[str, object]]:
                if not isinstance(regs, list) or not regs:
                    return []
                seen: set[tuple[float, float, float]] = set()
                deduped: list[dict[str, object]] = []
                for reg in regs:
                    if not isinstance(reg, dict):
                        continue
                    try:
                        center = reg.get('center_norm') or reg.get('center') or reg.get('center_xy') or [None, None]
                        if not isinstance(center, (list, tuple)) or len(center) != 2:
                            continue
                        cx = float(center[0])
                        cy = float(center[1])
                    except Exception:
                        continue
                    radius_val: float | None = None
                    radius_source = None
                    original_radius_norm: float | None = None
                    if reg.get('radius_norm') is not None:
                        try:
                            original_radius_norm = float(reg.get('radius_norm'))
                        except Exception:
                            original_radius_norm = None
                    for rk in ('radius_norm', 'radius'):
                        if reg.get(rk) is None:
                            continue
                        try:
                            radius_val = float(reg.get(rk))
                            radius_source = rk
                            break
                        except Exception:
                            continue
                    radius_from_mm = False
                    if radius_val is None and reg.get('radius_mm') is not None:
                        try:
                            radius_val = float(reg.get('radius_mm'))
                            radius_from_mm = True
                        except Exception:
                            radius_val = None
                    radius_key = round(radius_val, 4) if radius_val is not None else -1.0
                    key = (round(cx, 4), round(cy, 4), radius_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    reg_copy = dict(reg)
                    reg_copy['center_norm'] = [key[0], key[1]]
                    if radius_val is not None and not radius_from_mm:
                        if original_radius_norm is not None:
                            reg_copy['radius_norm'] = original_radius_norm
                        else:
                            reg_copy['radius_norm'] = round(radius_val, 4)
                    deduped.append(reg_copy)
                if not deduped:
                    return [reg for reg in regs if isinstance(reg, dict)]
                return deduped

            drop_regions_yaml = []
            largest_idx = None
            if hasattr(greenhouse, 'dropouts'):
                # Determine largest by radius
                radii = [float(getattr(reg, 'r0', 0.0)) for reg in getattr(greenhouse, 'dropouts', [])]
                if radii:
                    try:
                        largest_idx = int(max(range(len(radii)), key=lambda i: radii[i]))
                    except Exception:
                        largest_idx = None
                for i, reg in enumerate(getattr(greenhouse, 'dropouts', [])):
                    cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5))
                    r0 = float(getattr(reg, 'r0', 0.12))
                    ds = float(getattr(reg, 'drop_strength', getattr(reg, 'strength', 1.0)))
                    drop_regions_yaml.append({
                        'center_norm': [cx, cy],
                        'radius_norm': r0,
                        'strength': ds,
                        'quadrant': _quad_label(cx, cy),
                        'is_largest': bool(i == largest_idx),
                    })
            ma_records: list[tuple[float, float, float]] = []

            nv_regions_yaml = []
            nv_white_drawn = False
            nv_cyan_drawn = False
            nv_white_strength_cfg: float | None = None
            nv_white_fill_alpha_cfg: int | None = None
            try:
                nv_cfg = (config.get('Pathology', {}) or {}).get('NV', {}) if isinstance(config, dict) else {}
                ov_cfg = nv_cfg.get('overlay', {}) if isinstance(nv_cfg, dict) else {}
                if isinstance(ov_cfg, dict):
                    val = ov_cfg.get('white_strength', ov_cfg.get('white_intensity'))
                    if val is not None:
                        try:
                            nv_white_strength_cfg = float(val)
                        except Exception:
                            nv_white_strength_cfg = None
                    if ov_cfg.get('white_fill_alpha') is not None:
                        try:
                            nv_white_fill_alpha_cfg = int(ov_cfg.get('white_fill_alpha'))
                        except Exception:
                            nv_white_fill_alpha_cfg = None
                    elif ov_cfg.get('white_fill') is not None:
                        try:
                            nv_white_fill_alpha_cfg = int(float(ov_cfg.get('white_fill')) * 255.0)
                        except Exception:
                            nv_white_fill_alpha_cfg = None
            except Exception:
                nv_white_strength_cfg = None
            # Populate NV regions from host dropout regions used in augmentation
            try:
                if 'nv_host_regions' in locals() and nv_host_regions:
                    for (cxh, cyh, r0h) in nv_host_regions:
                        nv_regions_yaml.append({'center_norm': [float(cxh), float(cyh)], 'radius_norm': float(r0h), 'quadrant': _quad_label(float(cxh), float(cyh))})
            except Exception:
                pass

            # Also record physically simulated NV setup (if present)
            try:
                if physical_neovasc is not None and isinstance(physical_neovasc, (list, tuple)) and len(physical_neovasc) == 3:
                    _nx, _ny, _nr = float(physical_neovasc[0]), float(physical_neovasc[1]), float(physical_neovasc[2])
                    nv_regions_yaml.append({'center_norm': [_nx, _ny], 'radius_norm': _nr, 'quadrant': _quad_label(_nx, _ny)})
            except Exception:
                pass

            nv_regions_yaml = _dedupe_nv_regions(nv_regions_yaml)
            # Collect NV polylines across both growth engine and fallback for region summarization
            nv_fallback_polys_norm: list[list[tuple[float, float]]] = []

            # Draw NV overlays from the actual augmented vessel polylines (no fallback to synthetic overlay).
            try:
                if 'nv_polylines_norm' in locals() and nv_polylines_norm:
                    # Base thickness from image size; allow YAML to override with fixed or range
                    width_vis = int(max(4, round(0.006 * min(H, W))))
                    try:
                        _ov_cfg_line = ((config.get('Pathology', {}) or {}).get('NV', {}) or {}).get('overlay', {}) if isinstance(config, dict) else {}
                        if isinstance(_ov_cfg_line, dict):
                            if isinstance(_ov_cfg_line.get('thickness_px_range'), (list, tuple)) and len(_ov_cfg_line.get('thickness_px_range')) == 2:
                                _tlo = float(_ov_cfg_line['thickness_px_range'][0]); _thi = float(_ov_cfg_line['thickness_px_range'][1])
                                if _thi < _tlo:
                                    _tlo, _thi = _thi, _tlo
                                width_vis = int(max(1, round(_np.random.uniform(_tlo, _thi))))
                            elif _ov_cfg_line.get('thickness_px') is not None:
                                width_vis = int(max(1, round(float(_ov_cfg_line.get('thickness_px')))))
                    except Exception:
                        pass

                    def _safe_draw_line(dobj, pts, fill, width) -> None:
                        try:
                            dobj.line(pts, fill=fill, width=int(max(1, width)))
                        except Exception:
                            for i in range(1, len(pts)):
                                try:
                                    dobj.line([pts[i - 1], pts[i]], fill=fill, width=int(max(1, width)))
                                except Exception:
                                    pass

                    def _draw_polyline_feathered(dobj, pts, base_color, base_width):
                        # Multi-pass tapered stroke with increased smoothing: halo -> body -> bright core
                        base_thick = int(max(1, base_width))
                        w_outer = int(max(1, round(base_thick * 2.4)))  # Increased from 1.8
                        w_mid = int(max(1, round(base_thick * 1.3)))    # Increased from 1.0
                        w_core = int(max(1, round(max(1, base_thick * 0.6))))  # Adjusted
                        a = int(base_color[3]) if len(base_color) == 4 else 255
                        color_outer = (base_color[0], base_color[1], base_color[2], int(min(255, a * 0.25)))  # Reduced from 0.35
                        color_mid = (base_color[0], base_color[1], base_color[2], int(min(255, a * 0.75)))    # Reduced from 0.85
                        color_core = (base_color[0], base_color[1], base_color[2], int(min(255, a * 0.95)))   # Reduced from 1.10
                        _safe_draw_line(dobj, pts, fill=color_outer, width=w_outer)
                        _safe_draw_line(dobj, pts, fill=color_mid, width=w_mid)
                        _safe_draw_line(dobj, pts, fill=color_core, width=w_core)

                    for frag in nv_polylines_norm:
                        pts_px = [(int(_np.clip(round(y * W), 0, W - 1)), int(_np.clip(round(x * H), 0, H - 1))) for (x, y) in frag]
                        if len(pts_px) >= 2:
                            # Cyan, feathered
                            _draw_polyline_feathered(draw_nv, pts_px, (64, 220, 255, 220), width_vis)
                            # White, feathered (use configured base alpha if given)
                            a_white = int(nv_white_fill_alpha_cfg if nv_white_fill_alpha_cfg is not None else 255)
                            _draw_polyline_feathered(draw_white, pts_px, (255, 255, 255, a_white), max(2, width_vis))
                    nv_cyan_drawn = True
                    nv_white_drawn = True
                # Fallback: if no grown NV polylines but NV regions exist, procedurally grow curved sprouts
                elif isinstance(nv_regions_yaml, list) and len(nv_regions_yaml) > 0:
                    # Thickness selection from overlay config
                    width_vis = int(max(3, round(0.005 * min(H, W))))
                    try:
                        _ov_cfg_line = ((config.get('Pathology', {}) or {}).get('NV', {}) or {}).get('overlay', {}) if isinstance(config, dict) else {}
                        if isinstance(_ov_cfg_line, dict):
                            if isinstance(_ov_cfg_line.get('thickness_px_range'), (list, tuple)) and len(_ov_cfg_line.get('thickness_px_range')) == 2:
                                _tlo = float(_ov_cfg_line['thickness_px_range'][0]); _thi = float(_ov_cfg_line['thickness_px_range'][1])
                                if _thi < _tlo:
                                    _tlo, _thi = _thi, _tlo
                                width_vis = int(max(2, round((_tlo + _thi) * 0.5)))
                            elif _ov_cfg_line.get('thickness_px') is not None:
                                width_vis = int(max(2, round(float(_ov_cfg_line.get('thickness_px')))))
                    except Exception:
                        pass

                    # Base grayscale for steering into darker (avascular) areas
                    try:
                        base_gray_arr = _np.array(base_img.convert('L'))
                    except Exception:
                        base_gray_arr = None

                    def _grow_poly(host, cx_n: float, cy_n: float, r_n: float, n_rays: int = 3, steps: int = 60):
                        # host: greenhouse dropout object (or None). Ensure growth starts on dropout border and stays inside.
                        # Work in normalized model coords, then map to pixel when drawing.
                        rng = _np.random.default_rng()
                        # Host dropout params (fallback to NV region center/radius if host missing)
                        try:
                            hx = float(getattr(host, 'cx', cx_n)); hy = float(getattr(host, 'cy', cy_n)); hr = float(getattr(host, 'r0', r_n))
                        except Exception:
                            hx, hy, hr = float(cx_n), float(cy_n), float(r_n)

                        def _inside_dropout_norm(xn: float, yn: float) -> bool:
                            try:
                                val = float(_np.asarray(host.inside_field(_np.array([xn], dtype=_np.float32), _np.array([yn], dtype=_np.float32)))[0]) if host is not None else 1.0
                                return val > 0.0
                            except Exception:
                                dx, dy = (xn - hx), (yn - hy)
                                return (dx * dx + dy * dy) <= (hr * hr)

                        def _to_px(xn: float, yn: float) -> tuple[int, int]:
                            # map model (x=row, y=col) to pixels (col, row)
                            col = int(_np.clip(round(yn * W), 0, W - 1))
                            row = int(_np.clip(round(xn * H), 0, H - 1))
                            return (col, row)

                        for _k in range(int(max(1, n_rays))):
                            ang0 = rng.uniform(0.0, 2.0 * _np.pi)
                            sx = float(hx + hr * _np.cos(ang0))
                            sy = float(hy + hr * _np.sin(ang0))
                            # initial inward direction (toward host center)
                            dx = float(hx - sx); dy = float(hy - sy)
                            nrm = max(1e-6, _np.hypot(dx, dy)); dx /= nrm; dy /= nrm
                            pts_px = [_to_px(sx, sy)]
                            pts_norm: list[tuple[float, float]] = [(float(sx), float(sy))]
                            step_len = 0.0085  # normalized
                            curl = 0.55
                            jitter = 0.25
                            for _ in range(int(max(12, steps))):
                                # two curved options (left/right), choose darker and inside
                                tx, ty = -dy, dx
                                dx1 = (1.0 - curl) * dx + curl * tx
                                dy1 = (1.0 - curl) * dy + curl * ty
                                dx2 = (1.0 - curl) * dx - curl * tx
                                dy2 = (1.0 - curl) * dy - curl * ty
                                dx1 += jitter * rng.uniform(-0.3, 0.3); dy1 += jitter * rng.uniform(-0.3, 0.3)
                                dx2 += jitter * rng.uniform(-0.3, 0.3); dy2 += jitter * rng.uniform(-0.3, 0.3)
                                # normalize
                                n1 = max(1e-6, _np.hypot(dx1, dy1)); dx1 /= n1; dy1 /= n1
                                n2 = max(1e-6, _np.hypot(dx2, dy2)); dx2 /= n2; dy2 /= n2
                                candidates = []
                                for (ux, uy) in ((dx1, dy1), (dx2, dy2)):
                                    nx = float(sx + step_len * ux)
                                    ny = float(sy + step_len * uy)
                                    inside = _inside_dropout_norm(nx, ny)
                                    # evaluate grayscale (prefer darker)
                                    val = 255.0
                                    if base_gray_arr is not None:
                                        col, row = _to_px(nx, ny)
                                        val = float(base_gray_arr[row, col])
                                    # score: inside + darkness; penalize very bright
                                    score = (1.0 if inside else 0.0) + (1.0 - (val / 255.0)) - (0.35 if val > 220 else 0.0)
                                    candidates.append((score, nx, ny, ux, uy, inside, val))
                                candidates.sort(key=lambda t: (-t[0]))
                                _, nx, ny, dx, dy, inside, val = candidates[0]
                                if not inside:
                                    break
                                col, row = _to_px(nx, ny)
                                if len(pts_px) == 0 or (abs(col - pts_px[-1][0]) + abs(row - pts_px[-1][1])) > 1:
                                    pts_px.append((col, row))
                                    pts_norm.append((float(nx), float(ny)))
                                sx, sy = nx, ny
                                # stop if hitting bright vessel
                                if val > 220:
                                    break
                            if len(pts_px) >= 2:
                                # Tapered widths: thicker near border, thinner toward tip
                                w0 = int(max(2, round(width_vis * (1.10 + 0.25 * rng.random()))))
                                w1 = int(max(1, round(w0 * 0.55)))
                                for i in range(1, len(pts_px)):
                                    t = i / max(1, (len(pts_px) - 1))
                                    wi = int(max(1, round((1.0 - t) * w0 + t * w1)))
                                    # halo
                                    try:
                                        draw_nv.line([pts_px[i - 1], pts_px[i]], fill=(64, 220, 255, 160), width=int(max(1, round(wi * 1.6))))
                                    except Exception:
                                        pass
                                    # core
                                    try:
                                        draw_nv.line([pts_px[i - 1], pts_px[i]], fill=(64, 220, 255, 220), width=wi)
                                    except Exception:
                                        pass
                                    # white
                                    try:
                                        draw_white.line([pts_px[i - 1], pts_px[i]], fill=(255, 255, 255, 255), width=wi)
                                    except Exception:
                                        pass
                                # record normalized polyline for region summary
                                try:
                                    if len(pts_norm) >= 2:
                                        nv_fallback_polys_norm.append(pts_norm)
                                except Exception:
                                    pass

                    # Choose a host dropout for each NV region and grow 2-3 sprouts inside that dropout
                    # Host selection: nearest greenhouse dropout to the region center; fallback to region itself
                    # Build list once
                    _gh_dropouts = list(getattr(greenhouse, 'dropouts', []) or [])
                    for reg in nv_regions_yaml:
                        try:
                            c = reg.get('center_norm') or reg.get('center') or [None, None]
                            r = float(reg.get('radius_norm') or reg.get('radius') or 0.06)
                            if isinstance(c, (list, tuple)) and len(c) == 2:
                                cx_n, cy_n = float(c[0]), float(c[1])
                                host = None
                                if _gh_dropouts:
                                    try:
                                        # nearest by center distance
                                        _idx = int(min(range(len(_gh_dropouts)), key=lambda i: (float(getattr(_gh_dropouts[i], 'cx', 0.5)) - cx_n) ** 2 + (float(getattr(_gh_dropouts[i], 'cy', 0.5)) - cy_n) ** 2))
                                        host = _gh_dropouts[_idx]
                                    except Exception:
                                        host = None
                                _grow_poly(host, cx_n, cy_n, float(r), n_rays=int(2 + _np.random.randint(0, 2)), steps=int(50 + 90 * 0.6))
                        except Exception:
                            continue
                    nv_cyan_drawn = True
                    nv_white_drawn = True
            except Exception:
                pass

            if not (nv_white_drawn or ma_white_drawn):
                try:
                    base_img_white = base_img.copy()
                    nv_white_drawn = True
                except Exception:
                    nv_white_drawn = False

            # After drawing NV, enrich NV regions with per‑polyline summaries so YAML/JSON record all NV sites
            try:
                nv_poly_all = []
                try:
                    if 'nv_polylines_norm' in locals() and nv_polylines_norm:
                        nv_poly_all.extend([[(float(x), float(y)) for (x, y) in frag] for frag in nv_polylines_norm if isinstance(frag, (list, tuple)) and len(frag) >= 2])
                except Exception:
                    pass
                try:
                    if isinstance(nv_fallback_polys_norm, list) and nv_fallback_polys_norm:
                        nv_poly_all.extend(nv_fallback_polys_norm)
                except Exception:
                    pass
                def _regions_from_polys(polys: list[list[tuple[float, float]]]) -> list[dict]:
                    out: list[dict] = []
                    for pts in polys:
                        try:
                            xs = _np.array([p[0] for p in pts], dtype=_np.float32)
                            ys = _np.array([p[1] for p in pts], dtype=_np.float32)
                            cx = float(_np.mean(xs)); cy = float(_np.mean(ys))
                            rad = float(max(0.02, min(0.20, 1.35 * float(_np.max(_np.hypot(xs - cx, ys - cy))) )))
                            out.append({'center_norm': [cx, cy], 'radius_norm': rad, 'quadrant': _quad_label(cx, cy)})
                        except Exception:
                            continue
                    return out
                if nv_poly_all:
                    try:
                        # Prefer polyline-derived NV sites exclusively to avoid host/placeholder duplicates
                        nv_regions_from_polys = _regions_from_polys(nv_poly_all)
                        nv_regions_yaml = _dedupe_nv_regions(nv_regions_from_polys)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                ma_instances_yaml = [{
                    'center_norm': [float(cx), float(cy)],
                    'radius_mm': float(rmm),
                    'quadrant': _quad_label(float(cx), float(cy)),
                } for (cx, cy, rmm) in ma_records]
                faz_c = getattr(greenhouse, 'FAZ_center', (0.5, 0.5))
                faz_r = float(getattr(greenhouse, 'FAZ_radius', 0.06))
                fx, fy = float(faz_c[0]), float(faz_c[1])
            except Exception:
                fx, fy, faz_r = 0.5, 0.5, 0.06

            # Enrich JSON pathology with FAZ and comprehensive pathology summary
            try:
                pathology["FAZ"] = {"center_norm": [fx, fy], "radius_norm": faz_r}
                # Include neovascularization regions as a list (even if empty)
                nv_regions_json = _dedupe_nv_regions(nv_regions_yaml)
                pathology["neovascularization"] = [{
                    "center_norm": r.get('center_norm'),
                    "radius_norm": r.get('radius_norm'),
                    "quadrant": r.get('quadrant'),
                } for r in nv_regions_json]
                # Presence summary for 4 categories
                tort_gain = float(getattr(greenhouse, 'tortuosity_gain', 0.0)) if hasattr(greenhouse, 'tortuosity_gain') else 0.0
                pathology["summary"] = {
                    "dropout": {"present": bool(len(pathology.get("dropouts", [])) > 0), "count": int(len(pathology.get("dropouts", [])))},
                    "microaneurysm": {"present": bool(len(pathology.get("microaneurysms", [])) > 0), "count": int(len(pathology.get("microaneurysms", [])))},
                    "neovascularization": {"present": bool(len(nv_regions_yaml) > 0), "count": int(len(nv_regions_yaml))},
                    "tortuosity": {"present": bool(tort_gain > 0.0), "gain": float(tort_gain)},
                }
            except Exception:
                pass

            # Before view shift: finish NV composition
            try:
                # Feather the NV layer slightly to reduce harsh edges
                if nv_cyan_drawn:
                    try:
                        nv_ov_cfg = ((config.get('Pathology', {}) or {}).get('NV', {}) or {}).get('overlay', {}) if isinstance(config, dict) else {}
                        feather_px = 1.0
                        if isinstance(nv_ov_cfg, dict):
                            if isinstance(nv_ov_cfg.get('feather_px_range'), (list, tuple)) and len(nv_ov_cfg.get('feather_px_range')) == 2:
                                _flo = float(nv_ov_cfg['feather_px_range'][0]); _fhi = float(nv_ov_cfg['feather_px_range'][1])
                                if _fhi < _flo:
                                    _flo, _fhi = _fhi, _flo
                                feather_px = float(_np.random.uniform(_flo, _fhi))
                            elif nv_ov_cfg.get('feather_px') is not None:
                                feather_px = float(nv_ov_cfg.get('feather_px'))
                    except Exception:
                        feather_px = 1.0
                    feather_px = max(0.0, float(feather_px))
                    ov_nv = overlay_nv.filter(ImageFilter.GaussianBlur(radius=feather_px)) if feather_px > 0.0 else overlay_nv
                    # Compose NV onto main overlay
                    overlay = Image.alpha_composite(overlay, ov_nv)
                # Prepare white overlay compose for dataset image (pre-pan)
                if nv_white_drawn or ma_white_drawn:
                    # Instead of alpha compositing, which can create a gray background,
                    # create a mask from the white layer and use it to paste white vessels
                    # directly onto the base image. This preserves the black background.
                    mask = nv_white_layer.split()[3]  # Get alpha channel as mask
                    # Use unmodified base image (no dimming)
                    base_img_white = base_img.copy()
                    # Paste white (255) into the base image using the vessel mask
                    base_img_white.paste(255, mask=mask)
            except Exception:
                pass

            # Apply view shift (pan) if requested: shift both base_img and overlay together
            def _shift_img_rgba(img_rgba: Image.Image, dx_px: int, dy_px: int) -> Image.Image:
                if dx_px == 0 and dy_px == 0:
                    return img_rgba
                Wp, Hp = img_rgba.size
                out = Image.new("RGBA", (Wp, Hp), (0, 0, 0, 0))
                out.alpha_composite(img_rgba, (dx_px, dy_px))
                return out

            if nv_white_drawn:
                try:
                    strength = nv_white_strength_cfg
                    if strength is None:
                        strength = 0.0
                    strength = max(0.0, min(1.0, float(strength)))
                except Exception:
                    strength = 0.0
                if strength > 0.0:
                    base_img_white = Image.blend(base_img, base_img_white, strength)

            dx_px = dy_px = 0
            if view_shift_norm is not None:
                try:
                    sx, sy = float(view_shift_norm[0]), float(view_shift_norm[1])
                    dx_px = int(round(sx * W))
                    dy_px = int(round(sy * H))
                except Exception:
                    dx_px = dy_px = 0

            # Save composed visualization (now includes dropout, MA, NV) with optional panning
            try:
                base_rgba = base_img.convert("RGBA")
                if dx_px != 0 or dy_px != 0:
                    base_rgba = _shift_img_rgba(base_rgba, dx_px, dy_px)
                    overlay = _shift_img_rgba(overlay, dx_px, dy_px)
                    # Also pan the white overlay composition to keep M_ overlay consistent
                    try:
                        if nv_white_drawn or ma_white_drawn:
                            # Use SHIFTED base for correct FAZ alignment (no dimming)
                            nv_white_layer_shift = _shift_img_rgba(nv_white_layer, dx_px, dy_px)
                            base_img_white = Image.alpha_composite(base_rgba, nv_white_layer_shift).convert("RGB")
                    except Exception:
                        pass
                composed = Image.alpha_composite(base_rgba, overlay)

                # Center-crop to square while removing any padded edges introduced by panning
                crop_coords = None  # Store crop coordinates for later use
                try:
                    Wp, Hp = composed.size
                    valid_w = max(1, Wp - abs(int(dx_px)))
                    valid_h = max(1, Hp - abs(int(dy_px)))
                    side = int(max(1, min(valid_w, valid_h)))
                    # Valid content rectangle after shift
                    vx0 = max(0, int(dx_px))
                    vy0 = max(0, int(dy_px))
                    # Center a square inside the valid rectangle
                    x0 = int(vx0 + max(0, (valid_w - side) // 2))
                    y0 = int(vy0 + max(0, (valid_h - side) // 2))
                    x1 = int(min(Wp, x0 + side))
                    y1 = int(min(Hp, y0 + side))
                    if (x1 - x0) > 0 and (y1 - y0) > 0:
                        crop_coords = (x0, y0, x1, y1)
                        composed = composed.crop(crop_coords)
                        # Also crop the panned base image to produce a clean, view-shifted vessel map for GAN
                        try:
                            base_rgba_cropped = base_rgba.crop(crop_coords)
                            base_gray_cropped = base_rgba_cropped.convert("L")
                            base_gray_cropped.save(os.path.join(out_dir, "art_ven_img_gray_panned.png"))
                        except Exception:
                            pass
                        if nv_white_drawn and isinstance(base_img_white, Image.Image):
                            base_img_white = base_img_white.crop(crop_coords)
                except Exception:
                    pass
                # Cyan overlay visualization（包含 Dropout/MA/NV）
                composed.save(os.path.join(out_dir, "pathology_vis.png"))
                # White-burned overlay image for dataset usage (always save, even for healthy cases)
                # Auto-crop black borders to square for FAZ shift cases
                try:
                    if nv_white_drawn or ma_white_drawn:
                        img_to_save = _crop_black_borders_to_square(base_img_white)
                        img_to_save.save(os.path.join(out_dir, "pathology_overlay_white.png"))
                    else:
                        # For healthy cases (no NV/MA), save the cropped base vessel map so GAN has input
                        if isinstance(base_img_white, Image.Image):
                            img_to_save = _crop_black_borders_to_square(base_img_white)
                            img_to_save.save(os.path.join(out_dir, "pathology_overlay_white.png"))
                        elif crop_coords is not None:
                            # Crop the base image if view shift was applied
                            base_rgba_cropped = base_rgba.crop(crop_coords)
                            base_gray_cropped = base_rgba_cropped.convert("L")
                            img_to_save = _crop_black_borders_to_square(base_gray_cropped)
                            img_to_save.save(os.path.join(out_dir, "pathology_overlay_white.png"))
                        else:
                            img_to_save = _crop_black_borders_to_square(base_img)
                            img_to_save.save(os.path.join(out_dir, "pathology_overlay_white.png"))
                except Exception:
                    pass
                # 不再生成 dropout_mask_vis.png
            except Exception:
                pass

            # Save consolidated JSON
            import json as _json
            try:
                with open(os.path.join(out_dir, "dropout_ma.json"), 'w') as f:
                    _json.dump(pathology, f)
            except Exception as _e:
                print(f"[warn] Failed to save pathology JSON: {_e}")

            patho_yaml = {
                'FAZ': {
                    'center_norm': [fx, fy],
                    'radius_norm': faz_r,
                },
                'Dropout': {
                    'regions': drop_regions_yaml,
                },
                'MA': {
                    'density': float(getattr(greenhouse, 'ma_prob', 0.0)),
                    'near_dropout_only': bool(getattr(greenhouse, 'ma_only_near_dropout', True)),
                    'radius_mm_range': [float(getattr(greenhouse, 'ma_radius_mm_range', (0.010, 0.018))[0]), float(getattr(greenhouse, 'ma_radius_mm_range', (0.010, 0.018))[1])],
                    'prob_vs_dropout_strength_gain': float(getattr(greenhouse, 'ma_prob_strength_gain', 0.0)),
                    'count': int(len(ma_instances_yaml)),
                    'instances': ma_instances_yaml,
                },
                'NV': {
                    'regions': nv_regions_yaml,
                },
                'Tortuosity': {
                    'gain': float(getattr(greenhouse, 'tortuosity_gain', 0.0)),
                    'band': [float(getattr(greenhouse, 'tortuosity_band', (0.2, 0.8))[0]), float(getattr(greenhouse, 'tortuosity_band', (0.2, 0.8))[1])],
                },
            }
            try:
                with open(os.path.join(out_dir, 'pathology.yml'), 'w') as yf:
                    yaml.safe_dump(patho_yaml, yf, sort_keys=False)
            except Exception as _e:
                print(f"[warn] Failed to save pathology.yml: {_e}")
            # Finally, save vessel graph CSV including any fallback NV edges so GAN sees NV
            try:
                if config['output'].get('save_trees', True):
                    # Convert fallback NV polylines into graph edges with tapered radii (mm)
                    nv_edges_from_fallback: list[dict] = []
                    try:
                        if 'nv_fallback_polys_norm' in locals() and isinstance(nv_fallback_polys_norm, list):
                            for line in nv_fallback_polys_norm:
                                try:
                                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                                        continue
                                    L = len(line)
                                    r0_mm = 0.016
                                    r1_mm = 0.010
                                    for i in range(1, L):
                                        t = i / max(1, (L - 1))
                                        r_mm = (1.0 - t) * r0_mm + t * r1_mm
                                        nv_edges_from_fallback.append({
                                            "node1": (float(line[i - 1][0]), float(line[i - 1][1])),
                                            "node2": (float(line[i][0]), float(line[i][1])),
                                            "radius": float(r_mm),
                                        })
                                except Exception:
                                    continue
                    except Exception:
                        nv_edges_from_fallback = []
                    name = os.path.basename(out_dir.rstrip(os.sep))
                    filepath = os.path.join(out_dir, f"{name}.csv")
                    with open(filepath, 'w+', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["node1", "node2", "radius"])
                        for row in (art_edges + nv_edges_from_fallback + ven_edges):
                            writer.writerow([row["node1"], row["node2"], row["radius"]])
            except Exception as _e:
                print(f"[warn] Failed to save vessel CSV with NV fallback edges: {_e}")
            # Do not generate/save any dropout/neovasc masks; skip legacy mask flow entirely
            return out_dir
            def _fallback_dropout_mask_via_diffusion() -> _np.ndarray:
                # Re-render grayscale if not present in memory
                radius_list: list[float] = []
                art_mat, _ = tree2img.rasterize_forest(art_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                ven_mat, _ = tree2img.rasterize_forest(ven_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0)) if ven_edges else (_np.zeros(img_res, dtype=_np.uint8), {})
                img2d = _np.maximum(art_mat, ven_mat).astype(_np.uint8)
                H, W = int(img2d.shape[0]), int(img2d.shape[1])

                # Darkness map (higher = darker region)
                darkness = (255 - img2d).astype(_np.uint8)

                # Build normalized coordinate grid
                xs = (_np.arange(W) + 0.5) / max(W, 1)
                ys = (_np.arange(H) + 0.5) / max(H, 1)
                Xd, Yd = _np.meshgrid(xs, ys)  # shapes (H, W)

                def _otsu_threshold(vals: _np.ndarray) -> int:
                    # Compute Otsu threshold over uint8 array
                    if vals.size == 0:
                        return 0
                    hist = _np.bincount(vals.astype(_np.uint8).ravel(), minlength=256).astype(_np.float64)
                    total = vals.size
                    weight_b = 0.0
                    sum_total = _np.dot(_np.arange(256), hist)
                    sum_b = 0.0
                    var_max = -1.0
                    thresh = 0
                    for t in range(256):
                        weight_b += hist[t]
                        if weight_b == 0:
                            continue
                        weight_f = total - weight_b
                        if weight_f == 0:
                            break
                        sum_b += t * hist[t]
                        m_b = sum_b / weight_b
                        m_f = (sum_total - sum_b) / weight_f
                        var_between = weight_b * weight_f * (m_b - m_f) ** 2
                        if var_between > var_max:
                            var_max = var_between
                            thresh = t
                    return int(thresh)

                out = _np.zeros((H, W), dtype=_np.uint8)
                for reg in getattr(greenhouse, 'dropouts', []):
                    cx = float(getattr(reg, 'cx', 0.5))
                    cy = float(getattr(reg, 'cy', 0.5))
                    r0 = float(getattr(reg, 'r0', 0.12))
                    # Circular ROI in normalized coords with a generous margin factor
                    dist = _np.sqrt((Yd - cx) ** 2 + (Xd - cy) ** 2)
                    roi = dist <= (1.80 * r0)
                    if not _np.any(roi):
                        continue
                    vals = darkness[roi]
                    thr = _otsu_threshold(vals)
                    # Relaxed threshold to include more surrounding dark area
                    thr = max(0, min(255, int(0.85 * thr)))
                    mask_i = _np.zeros_like(out)
                    mask_i[roi] = (darkness[roi] >= thr).astype(_np.uint8)
                    out |= mask_i
                return (out * 255).astype(_np.uint8)
            def _fallback_dropout_mask_via_field() -> _np.ndarray:
                # Build coordinate grids in display space, then map to model (Xm,Ym)
                W = int(img_res[0])
                H = int(img_res[1])
                xs = (_np.arange(W) + 0.5) / max(W, 1)
                ys = (_np.arange(H) + 0.5) / max(H, 1)
                Xd, Yd = _np.meshgrid(xs, ys)  # shapes (H, W)
                Xm = Yd
                Ym = Xd
                # Accumulate inside score across all regions
                acc = _np.zeros((H, W), dtype=_np.float32)
                roi_union = _np.zeros((H, W), dtype=_np.bool_)
                for reg in getattr(greenhouse, 'dropouts', []):
                    acc = _np.maximum(acc, reg.inside_field(Xm, Ym).astype(_np.float32))
                    cx = float(getattr(reg, 'cx', 0.5))
                    cy = float(getattr(reg, 'cy', 0.5))
                    r0 = float(getattr(reg, 'r0', 0.12))
                    dist = _np.sqrt((Xm - cx) ** 2 + (Ym - cy) ** 2)
                    roi_union |= (dist <= (1.25 * r0))
                # Otsu on the score within ROI to separate inside vs outside
                vals_u8 = (acc[roi_union] * 255.0).astype(_np.uint8) if _np.any(roi_union) else (acc.ravel() * 255.0).astype(_np.uint8)
                def _otsu_threshold_u8(vals: _np.ndarray) -> int:
                    if vals.size == 0:
                        return 0
                    hist = _np.bincount(vals.astype(_np.uint8).ravel(), minlength=256).astype(_np.float64)
                    total = float(vals.size)
                    weight_b = 0.0
                    sum_total = float(_np.dot(_np.arange(256), hist))
                    sum_b = 0.0
                    var_max = -1.0
                    thresh = 0
                    for t in range(256):
                        weight_b += hist[t]
                        if weight_b == 0:
                            continue
                        weight_f = total - weight_b
                        if weight_f == 0:
                            break
                        sum_b += t * hist[t]
                        m_b = sum_b / weight_b
                        m_f = (sum_total - sum_b) / weight_f
                        var_between = weight_b * weight_f * (m_b - m_f) ** 2
                        if var_between > var_max:
                            var_max = var_between
                            thresh = t
                    return int(thresh)
                thr_u8 = _otsu_threshold_u8(vals_u8)
                thr_norm = max(0.18, min(0.50, float(thr_u8) / 255.0))
                mask_bool = (acc >= thr_norm)
                # Simple hole-filling/closing via a small dilation then erosion
                def _dilate_bool_sq_local(mask_bool_l: _np.ndarray, radius: int) -> _np.ndarray:
                    return _dilate_bool_sq(mask_bool_l, radius)
                def _erode_bool_sq(mask_bool_l: _np.ndarray, radius: int) -> _np.ndarray:
                    if radius <= 0:
                        return mask_bool_l
                    inv = ~mask_bool_l
                    inv_d = _dilate_bool_sq_local(inv, radius)
                    return ~inv_d
                mask_bool = _dilate_bool_sq_local(mask_bool, 1)
                mask_bool = _erode_bool_sq(mask_bool, 1)
                return (mask_bool.astype(_np.uint8) * 255)

            # Initialize dmask using irregular mask generation
            dmask = _mask_irregular(greenhouse, H, W)

            # Evaluate mask sufficiency
            dmask_arr = _np.asarray(dmask, dtype=_np.uint8)
            Hm, Wm = dmask_arr.shape[0], dmask_arr.shape[1]
            white_px = int((dmask_arr > 0).sum())
            # Expected area from configured dropout radii
            exp_px = 0.0
            for reg in getattr(greenhouse, 'dropouts', []):
                r0 = float(getattr(reg, 'r0', 0.12))
                exp_px += _np.pi * (r0 ** 2) * (Hm * Wm)
            too_small = (exp_px > 0.0) and (white_px < 0.15 * exp_px)

            if (white_px == 0) or too_small:
                # Try field-based fallback first; if it yields empty, use darkness diffusion
                dmask_fb = _fallback_dropout_mask_via_field()
                if not _np.any(dmask_fb):
                    dmask_fb = _fallback_dropout_mask_via_diffusion()
                dmask = dmask_fb
            # Expand into adjacent clearly dark regions via geodesic growth (within dropout ROIs)
            try:
                radius_list_g: list[float] = []
                art_mat_g, _ = tree2img.rasterize_forest(art_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list_g, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                if ven_edges:
                    ven_mat_g, _ = tree2img.rasterize_forest(ven_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list_g, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                else:
                    ven_mat_g = _np.zeros(img_res, dtype=_np.uint8)
                img2d_g = _np.maximum(art_mat_g, ven_mat_g).astype(_np.uint8)
                darkness = (255 - img2d_g).astype(_np.uint8)

                # ROI around each dropout region (slightly larger than simulated radius)
                W = int(img_res[0]); H = int(img_res[1])
                xs = (_np.arange(W) + 0.5) / max(W, 1)
                ys = (_np.arange(H) + 0.5) / max(H, 1)
                Xd, Yd = _np.meshgrid(xs, ys)
                Xm = Yd; Ym = Xd
                roi_union = _np.zeros((H, W), dtype=_np.bool_)
                for reg in getattr(greenhouse, 'dropouts', []):
                    cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
                    dist = _np.sqrt((Xm - cx) ** 2 + (Ym - cy) ** 2)
                    roi_union |= (dist <= (2.2 * r0))

                # Darkness threshold (Otsu) within ROI
                def _otsu_u8(vals: _np.ndarray) -> int:
                    if vals.size == 0:
                        return 0
                    hist = _np.bincount(vals.astype(_np.uint8).ravel(), minlength=256).astype(_np.float64)
                    total = float(vals.size)
                    weight_b = 0.0
                    sum_total = float(_np.dot(_np.arange(256), hist))
                    sum_b = 0.0
                    var_max = -1.0
                    thresh = 0
                    for t in range(256):
                        weight_b += hist[t]
                        if weight_b == 0:
                            continue
                        weight_f = total - weight_b
                        if weight_f == 0:
                            break
                        sum_b += t * hist[t]
                        m_b = sum_b / weight_b
                        m_f = (sum_total - sum_b) / weight_f
                        var_between = weight_b * weight_f * (m_b - m_f) ** 2
                        if var_between > var_max:
                            var_max = var_between
                            thresh = t
                    return int(thresh)

                roi_vals = darkness[roi_union]
                thr_dark = _otsu_u8(roi_vals) if roi_vals.size > 0 else _otsu_u8(darkness)
                thr_dark = int(max(0, min(255, 0.85 * thr_dark)))

                # Binary segmentation of vessel map and geodesic growth with relaxation if area too small
                def _grow_with_params(thr_extra: int, dilate_px: int, roi_scale: float) -> _np.ndarray:
                    roi_relax = _np.zeros_like(roi_union)
                    if roi_scale != 1.0:
                        # Relax ROI by recomputing with a larger scale
                        roi_relax = _np.zeros_like(roi_union)
                        for reg in getattr(greenhouse, 'dropouts', []):
                            cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
                            dist = _np.sqrt((Xm - cx) ** 2 + (Ym - cy) ** 2)
                            roi_relax |= (dist <= (roi_scale * r0))
                    roi_eff = roi_relax if roi_scale != 1.0 else roi_union
                    # Exclude FAZ from effective ROI
                    try:
                        fx, fy = float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[1])
                        fr = float(getattr(greenhouse, 'FAZ_radius', 0.06))
                    except Exception:
                        fx, fy, fr = 0.5, 0.5, 0.06
                    fr_excl = 1.2 * fr  # small margin beyond FAZ
                    faz_excl = (_np.sqrt((Xm - fx) ** 2 + (Ym - fy) ** 2) <= fr_excl)
                    roi_eff = roi_eff & (~faz_excl)

                    base_thr = _otsu_u8(img2d_g)
                    thr_vessel = int(max(0, min(255, base_thr + thr_extra)))
                    vessel_bin_rough = (img2d_g >= thr_vessel)
                    vessel_bin = _dilate_bool_sq(vessel_bin_rough, max(0, int(dilate_px))) if dilate_px > 0 else vessel_bin_rough

                    bg_cand = (~vessel_bin) & roi_eff
                    Hh, Ww = bg_cand.shape
                    mask_center = _np.zeros((Hh, Ww), dtype=_np.bool_)
                    for reg in getattr(greenhouse, 'dropouts', []):
                        cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
                        dist = _np.sqrt((Xm - cx) ** 2 + (Ym - cy) ** 2)
                        seed = _np.zeros_like(bg_cand)
                        found = False
                        for frac in [0.15, 0.22, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
                            disk = dist <= (frac * r0)
                            seed_try = disk & bg_cand
                            if seed_try.any():
                                seed = seed_try
                                found = True
                                break
                        if not found:
                            seed = (dist <= (0.85 * r0)) & bg_cand
                            if not seed.any():
                                continue
                        loops = 0
                        while loops < 512:
                            nb = _dilate_bool_sq(seed, 1)
                            recon = (nb & bg_cand) | seed
                            if _np.array_equal(recon, seed):
                                break
                            seed = recon
                            loops += 1
                        mask_center |= seed
                    mask_bool = mask_center
                    mask_bool = _dilate_bool_sq(mask_bool, 1)
                    mask_bool = (~_dilate_bool_sq(~mask_bool, 1))
                    return mask_bool.astype(_np.uint8) * 255

                # Try increasingly permissive settings if area remains too small vs expected
                exp_px = 0.0
                for reg in getattr(greenhouse, 'dropouts', []):
                    r0 = float(getattr(reg, 'r0', 0.12))
                    exp_px += _np.pi * (r0 ** 2) * (H * W)
                target_frac = 0.45
                tried = [
                    (8, 1, 2.2),   # base: slightly higher than Otsu, 1px thick boundary, ROI 2.2*r0
                    (16, 1, 2.4),  # raise threshold -> fewer vessel obstacles, slightly larger ROI
                    (24, 0, 2.6),  # thinner boundaries and larger ROI
                    (32, 0, 3.0),  # most permissive within safety bounds
                ]
                for thr_extra, dil_px, roi_scale in tried:
                    mask_try = _grow_with_params(thr_extra, dil_px, roi_scale)
                    area_try = int((mask_try > 0).sum())
                    if exp_px <= 0 or area_try >= target_frac * exp_px:
                        dmask = mask_try
                        break
                else:
                    dmask = mask_try  # fall back to last attempt
            except Exception:
                # If any error occurs during geodesic growth, keep previously computed dmask
                pass

            # dmask already computed (adaptive growth above)
            # Ray-based diffusion from centers to nearest vessel borders (no-try wrapper)
            def _ray_grow_to_vessels() -> _np.ndarray:
                base_thr = _otsu_u8(img2d_g)
                thr_vessel = int(max(8, min(255, base_thr)))
                vbin = _dilate_bool_sq((img2d_g >= thr_vessel), 2)
                # Exclude FAZ
                try:
                    fx, fy = float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[1])
                    fr = float(getattr(greenhouse, 'FAZ_radius', 0.06))
                except Exception:
                    fx, fy, fr = 0.5, 0.5, 0.06
                faz_mask = (_np.sqrt((Xm - fx) ** 2 + (Ym - fy) ** 2) <= (1.2 * fr))
                bg = (~vbin) & (~faz_mask)
                
                Hh, Ww = bg.shape
                out = _np.zeros((Hh, Ww), dtype=_np.bool_)
                step_norm = 1.0 / float(max(1, min(Hh, Ww)))
                thetas = _np.linspace(0.0, 2.0 * _np.pi, 256, endpoint=False)
                for reg in getattr(greenhouse, 'dropouts', []):
                    cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
                    r_center = int(_np.clip(cx * Hh, 0, Hh - 1))
                    c_center = int(_np.clip(cy * Ww, 0, Ww - 1))
                    if vbin[r_center, c_center]:
                        # nudge to nearby background
                        rr = int(max(1, round(0.05 * min(Hh, Ww))))
                        found_bg = False
                        for dy in range(-rr, rr + 1):
                            for dx in range(-rr, rr + 1):
                                r2, c2 = r_center + dy, c_center + dx
                                if 0 <= r2 < Hh and 0 <= c2 < Ww and bg[r2, c2]:
                                    r_center, c_center = r2, c2
                                    found_bg = True
                                    break
                            if found_bg:
                                break
                    max_r = 2.0 * r0
                    for th in thetas:
                        t = 0.0
                        last_r, last_c = r_center, c_center
                        for _ in range(int(max_r / step_norm)):
                            xm = cx + t * _np.cos(th)
                            ym = cy + t * _np.sin(th)
                            if xm < 0.0 or xm > 1.0 or ym < 0.0 or ym > 1.0:
                                break
                            rr_pix = int(_np.clip(xm * Hh, 0, Hh - 1))
                            cc_pix = int(_np.clip(ym * Ww, 0, Ww - 1))
                            if vbin[rr_pix, cc_pix] or not bg[rr_pix, cc_pix]:
                                break
                            out[rr_pix, cc_pix] = True
                            last_r, last_c = rr_pix, cc_pix
                            t += step_norm
                        # connect line back to center to close gaps
                        dr = last_r - r_center; dc = last_c - c_center
                        steps = int(max(abs(dr), abs(dc)))
                        if steps > 0:
                            for i in range(steps + 1):
                                rr_pix = r_center + int(round(dr * i / steps))
                                cc_pix = c_center + int(round(dc * i / steps))
                                if 0 <= rr_pix < Hh and 0 <= cc_pix < Ww and bg[rr_pix, cc_pix]:
                                    out[rr_pix, cc_pix] = True
                # close tiny gaps
                out = _dilate_bool_sq(out, 1)
                out = (~_dilate_bool_sq(~out, 1))
                return (out.astype(_np.uint8) * 255)

            dmask_rays = _ray_grow_to_vessels()
            # Prefer ray-grown mask; if it ends up empty, keep prior dmask
            if _np.any(dmask_rays):
                dmask = dmask_rays
            # Optional expansion
            if int(mask_expand_px) > 0:
                dmask_bool = _np.array(dmask, dtype=_np.uint8) > 0
                dmask_bool = _dilate_bool_sq(dmask_bool, int(mask_expand_px))
                dmask = (_np.asarray(dmask_bool, dtype=_np.uint8) * 255)
            # Remove vessel neighborhoods from final dropout mask if requested
            try:
                if int(mask_vessel_sparing_px) > 0:
                    # Build vessel binary at identical resolution using precise rasterizer
                    radius_list: list[float] = []
                    art_mat, _ = tree2img.rasterize_forest(art_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                    if ven_edges:
                        ven_mat, _ = tree2img.rasterize_forest(ven_edges, img_res, MIP_axis=proj_axis, radius_list=radius_list, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                        vimg = _np.maximum(art_mat, ven_mat)
                    else:
                        vimg = art_mat
                    vbin = (vimg.astype(_np.float32) > 0)
                    vbuf = _dilate_bool_sq(vbin, int(mask_vessel_sparing_px))
                    dmask_bool = (_np.array(dmask, dtype=_np.uint8) > 0)
                    dmask_bool = dmask_bool & (~vbuf)
                    dmask = (_np.asarray(dmask_bool, dtype=_np.uint8) * 255)
            except Exception:
                pass
            # Enforce FAZ exclusion on final mask before saving
            try:
                fx, fy = float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(greenhouse, 'FAZ_center', (0.5, 0.5))[1])
                fr = float(getattr(greenhouse, 'FAZ_radius', 0.06))
                fr_excl = 1.2 * fr
                faz_excl = (_np.sqrt((Xm - fx) ** 2 + (Ym - fy) ** 2) <= fr_excl)
                if isinstance(dmask, _np.ndarray) and dmask.ndim == 2:
                    dmask = dmask.copy()
                    dmask[faz_excl] = 0
            except Exception:
                pass
            Image.fromarray(dmask).save(os.path.join(out_dir, "dropout_mask.png"))
        except Exception as e:
            print(f"[warn] Failed to render/save pathology masks: {e}")
    elif False:
        # Masks disabled permanently; keep branch to avoid executing legacy path
        pass

    # If a view shift is requested, re-render a larger grayscale vessel map and crop (avoid black padding)
    try:
        if view_shift_norm is not None:
            sx, sy = float(view_shift_norm[0]), float(view_shift_norm[1])
            # Determine original target size from previous rendering plan
            img_res = [int(d) for d in greenhouse.simspace.shape * cfg_local['output']['image_scale_factor']]
            proj_axis = cfg_local["output"].get("proj_axis", 2)
            del img_res[proj_axis]
            # img_res is [W, H]; convert to H,W
            W0, H0 = int(img_res[0]), int(img_res[1])
            # Compute pixel shift for final crop window
            dx_px = int(round(sx * W0))
            dy_px = int(round(sy * H0))
            mx = abs(dx_px)
            my = abs(dy_px)
            Wb, Hb = W0 + 2 * mx, H0 + 2 * my
            # Render big grayscale via rasterization
            radius_list_v: list[float] = []
            art_mat_v, _ = tree2img.rasterize_forest(art_edges, [Wb, Hb], MIP_axis=proj_axis, radius_list=radius_list_v, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
            if ven_edges:
                ven_mat_v, _ = tree2img.rasterize_forest(ven_edges, [Wb, Hb], MIP_axis=proj_axis, radius_list=radius_list_v, max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0))
                img_big = np.maximum(art_mat_v, ven_mat_v).astype(np.uint8)
            else:
                img_big = art_mat_v.astype(np.uint8)
            # Crop window representing the shifted view
            x0 = mx + dx_px
            y0 = my + dy_px
            x1 = x0 + W0
            y1 = y0 + H0
            x0 = max(0, min(int(x0), max(0, Wb - W0)))
            y0 = max(0, min(int(y0), max(0, Hb - H0)))
            img_crop = img_big[y0:y1, x0:x1]
            # Overwrite saved grayscale
            Image.fromarray(img_crop).save(os.path.join(out_dir, "art_ven_img_gray.png"))
    except Exception:
        pass

    # For healthy samples (no pathology), ensure pathology_overlay_white.png exists so GAN can process them
    # Auto-crop black borders to square for FAZ shift cases
    try:
        overlay_path = os.path.join(out_dir, "pathology_overlay_white.png")
        if not os.path.exists(overlay_path):
            # Use panned version if available, otherwise regular gray
            panned_path = os.path.join(out_dir, "art_ven_img_gray_panned.png")
            gray_path = os.path.join(out_dir, "art_ven_img_gray.png")
            if os.path.exists(panned_path):
                from PIL import Image
                img = Image.open(panned_path)
                img_cropped = _crop_black_borders_to_square(img)
                img_cropped.save(overlay_path)
            elif os.path.exists(gray_path):
                from PIL import Image
                img = Image.open(gray_path)
                img_cropped = _crop_black_borders_to_square(img)
                img_cropped.save(overlay_path)
    except Exception as e:
        print(f"[warn] Failed to create pathology_overlay_white.png for healthy sample: {e}")

    # Save CSV for all samples (including healthy ones) so GAN can process them
    try:
        name = os.path.basename(out_dir.rstrip(os.sep))
        filepath = os.path.join(out_dir, f"{name}.csv")
        if not os.path.exists(filepath) and config.get('output', {}).get('save_trees', True):
            with open(filepath, 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["node1", "node2", "radius"])
                for row in (art_edges + ven_edges):
                    writer.writerow([row["node1"], row["node2"], row["radius"]])
    except Exception as e:
        print(f"[warn] Failed to save vessel CSV for healthy sample: {e}")

    return out_dir


def _call_gan_inference(
    gan_config: str,
    graph_root: str,
    image_out_dir: str,
    epoch: int | str,
    device: str | None,
    num_workers: int | None = None,
    batch_size: int | None = None,
    log_progress: bool | None = None,
):
    """Invoke test.py with overrides to transform inputs into images using the pretrained GAN.

    Preference order for inputs per-sample (so GAN can see NV):
      1) pathology_overlay_white.png (NV-inclusive overlay produced in graphs stage)
      2) art_ven_img_gray_panned.png (if available)
      3) CSV graph (fallback)
    """
    import subprocess, glob as _glob

    test_script = os.path.join(SCRIPT_DIR, "test.py")
    if not os.path.isfile(test_script):
        raise SystemExit(
            "GAN inference requires 'test.py' and the GAN models from the "
            "OCTA-autosegmentation repository. Either run this script inside "
            "a clone of https://github.com/aiforvision/OCTA-autosegmentation "
            "or copy it into that workspace."
        )

    root_abs = os.path.abspath(graph_root)
    patt_overlay = os.path.join(root_abs, "**", "pathology_overlay_white.png")
    patt_panned = os.path.join(root_abs, "**", "art_ven_img_gray_panned.png")
    patt_csv = os.path.join(root_abs, "**", "*.csv")
    # Prefer overlay inputs if present; else panned gray; else CSV
    try:
        has_overlay = bool(_glob.glob(patt_overlay, recursive=True))
    except Exception:
        has_overlay = False
    try:
        has_panned = bool(_glob.glob(patt_panned, recursive=True))
    except Exception:
        has_panned = False

    # Always enumerate by CSV files so every sample is included; the transform will
    # look for NV-inclusive overlays in the same folder and prefer them if present.
    chosen_pattern = patt_csv

    cmd = [
        sys.executable,
        test_script,
        "--config_file", gan_config,
        "--epoch", str(epoch),
        f"--Test.data.real_A.files={chosen_pattern}",
        f"--Test.save_dir={os.path.abspath(image_out_dir)}",
    ]
    if device:
        cmd.append(f"--General.device={device}")
    if num_workers is not None:
        cmd.append(f"--num_workers={int(num_workers)}")
    if batch_size is not None and int(batch_size) > 0:
        # Forward as a config override to the inner GAN test config
        cmd.append(f"--Test.batch_size={int(batch_size)}")
    if log_progress:
        cmd.append("--log_progress")

    subprocess.check_call(cmd)


def _render_labels_from_graphs(graph_root: str, label_out_dir: str, resolution: Sequence[int], mip_axis: int = 2, max_dropout_prob: float = 0.0, binarize: bool = True):
    os.makedirs(label_out_dir, exist_ok=True)
    csv_files = sorted(glob(os.path.join(graph_root, "**", "*.csv"), recursive=True))
    assert len(csv_files) > 0, f"No CSV graphs found under {graph_root}"

    if len(resolution) == 3:
        img_res = list(resolution)
        del img_res[mip_axis]
    else:
        img_res = list(resolution)

    import csv as _csv
    from PIL import Image
    # Local rendering clamp for labels (no external config available here)
    render_max_r_mm = None

    for file_path in csv_files:
        name = os.path.basename(file_path).removesuffix(".csv")
        f: list[dict] = []
        with open(file_path, newline='') as csvfile:
            reader = _csv.DictReader(csvfile)
            for row in reader:
                f.append(row)
        img, _ = tree2img.rasterize_forest(
            f,
            img_res,
            MIP_axis=mip_axis,
            max_dropout_prob=max_dropout_prob,
            max_radius=(render_max_r_mm if render_max_r_mm is not None else 1.0),
        )
        if binarize:
            img = (img >= 0.1).astype(np.uint8) * 255
            Image.fromarray(img).convert("1").save(os.path.join(label_out_dir, name + "_label.png"))
        else:
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(label_out_dir, name + ".png"))


def main():
    parser = argparse.ArgumentParser("Generate synthetic OCTA samples end-to-end")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--vessel_config", type=str, default=os.path.join(SCRIPT_DIR, "vessel_graph_generation", "configs", "dataset_18_June_2023.yml"))
    parser.add_argument("--graph_out_dir", type=str, default=DEFAULT_GRAPH_OUT_DIR)
    parser.add_argument("--save_raw_mip", action="store_true", help="Save raw grayscale MIP images from tree2img during graph generation", default=True)
    parser.add_argument("--save_3d", choices=["npy", "nifti"], default=None)
    parser.add_argument("--save_pathology_masks", action="store_true", help="Save binary masks for dropout and neovascularization regions", default=False)
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="Threshold on dropout inside_score to binarize mask [0-1]")

    # GAN step
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--gan_config", type=str, default=None)
    parser.add_argument("--gan_epoch", type=str, default="150")
    parser.add_argument("--gan_device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--image_out_dir", type=str, default=DEFAULT_IMAGE_OUT_DIR)

    # Labels
    parser.add_argument("--render_labels", action="store_true")
    parser.add_argument("--label_out_dir", type=str, default=DEFAULT_LABEL_OUT_DIR)
    parser.add_argument("--resolution", type=str, default="1216,1216,16", help="Px resolution for rendering labels (for 2D, pass H,W)")
    parser.add_argument("--mip_axis", type=int, default=2)

    # Unified base output directory (optional alias)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Base directory for all outputs. "
            "If set (and per-type dirs are not overridden), graphs, GAN images, and labels "
            "are written to '<out_dir>/vessel_graphs', '<out_dir>/images', and '<out_dir>/labels'."
        ),
    )

    # Pathology options (physical only)
    parser.add_argument("--physical_dropout", type=str, default=None, help="Physically simulate dropout during growth as 'cx,cy,r'")
    parser.add_argument("--physical_dropout_strength", type=float, default=1.0, help="Suppression strength inside dropout [0-1]")
    parser.add_argument("--physical_neovasc", type=str, default=None, help="Physically simulate neovascular tuft during growth as 'cx,cy,r'")

    parser.add_argument("--workers", type=int, default=-1, help="Parallel processes for graph gen (-1 => all but one)")
    parser.add_argument("--skip_venous", action="store_true", help="Skip venous tree generation for speed")
    parser.add_argument("--no_config_per_sample", action="store_true", help="Do not write per-sample config.yml to reduce I/O")
    parser.add_argument("--growth_progress", type=float, default=1.0, help="Fraction of total growth iterations across modes (0,1]. Early-stops growth to control depth.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # If a unified base --out_dir is provided, map it to per-type output roots
    if args.out_dir:
        base_root = os.path.abspath(args.out_dir)
        if args.graph_out_dir == DEFAULT_GRAPH_OUT_DIR:
            args.graph_out_dir = os.path.join(base_root, "vessel_graphs")
        if args.image_out_dir == DEFAULT_IMAGE_OUT_DIR:
            args.image_out_dir = os.path.join(base_root, "images")
        if args.label_out_dir == DEFAULT_LABEL_OUT_DIR:
            args.label_out_dir = os.path.join(base_root, "labels")

    if args.debug:
        warnings.filterwarnings('error')

    if args.use_gan:
        if not args.gan_config:
            raise SystemExit("Please provide --gan_config when --use_gan is enabled (GAN weights/config not bundled).")
        if not os.path.isfile(args.gan_config):
            raise SystemExit(f"GAN config not found: {args.gan_config}")

    assert os.path.isfile(args.vessel_config), f"Config not found: {args.vessel_config}"
    os.makedirs(args.graph_out_dir, exist_ok=True)

    config = read_config(args.vessel_config)
    # Override outputs according to CLI
    config.setdefault('output', {})
    config['output']['directory'] = os.path.abspath(args.graph_out_dir)
    config['output']['save_trees'] = True
    config['output']['save_2D_image'] = bool(args.save_raw_mip)
    config['output']['save_3D_volumes'] = args.save_3d

    # Parse pathology regions
    def _parse_triplet(s: str | None):
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        assert len(parts) == 3, "Expected triplet 'cx,cy,r'"
        return float(parts[0]), float(parts[1]), float(parts[2])

    physical_dropout = _parse_triplet(args.physical_dropout)
    physical_neovasc = _parse_triplet(args.physical_neovasc)

    # Optionally scale total growth iterations to control depth
    gp = float(args.growth_progress)
    if not (0 < gp <= 1.0):
        raise ValueError("--growth_progress must be in (0,1]")
    try:
        modes = config['Greenhouse']['modes']
        total_I = sum(int(m['I']) for m in modes)
        target = max(1, int(round(total_I * gp)))
        new_modes = []
        remaining = target
        for m in modes:
            I_orig = int(m['I'])
            I_new = min(I_orig, remaining)
            m_new = dict(m)
            m_new['I'] = I_new
            new_modes.append(m_new)
            remaining -= I_new
            if remaining <= 0:
                break
        # If target smaller than first mode, trim there; if larger, keep rest with I=0
        if remaining > 0:
            # keep structure but zero out remaining modes
            for m in modes[len(new_modes):]:
                m0 = dict(m); m0['I'] = 0; new_modes.append(m0)
        config['Greenhouse']['modes'] = new_modes
    except Exception as e:
        print(f"[warn] Failed to apply growth_progress: {e}")

    # Generate N samples (optionally parallel)
    out_dirs = []
    ns = max(1, args.num_samples)
    if ns > 1:
        if args.workers == -1:
            cpus = cpu_count()
            workers = max(1, cpus - 1)
        else:
            workers = max(1, int(args.workers))
    else:
        workers = 1

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _generate_single_sample,
                    config,
                    config['output']['directory'],
                    physical_dropout,
                    args.physical_dropout_strength,
                    physical_neovasc,
                    args.skip_venous,
                    not args.no_config_per_sample,
                    bool(args.save_pathology_masks),
                    float(args.mask_threshold),
                ) for _ in range(ns)
            ]
            for fut in as_completed(futs):
                out_dirs.append(fut.result())
    else:
        for _ in range(ns):
            out_dir = _generate_single_sample(
                config,
                base_out_dir=config['output']['directory'],
                physical_dropout=physical_dropout,
                physical_strength=args.physical_dropout_strength,
                physical_neovasc=physical_neovasc,
                skip_venous=args.skip_venous,
                write_config_per_sample=not args.no_config_per_sample,
                save_pathology_masks=bool(args.save_pathology_masks),
                mask_threshold=float(args.mask_threshold),
            )
            out_dirs.append(out_dir)

    # Optional GAN transform
    if args.use_gan:
        os.makedirs(args.image_out_dir, exist_ok=True)
        _call_gan_inference(
            gan_config=args.gan_config,
            graph_root=config['output']['directory'],
            image_out_dir=args.image_out_dir,
            epoch=args.gan_epoch,
            device=args.gan_device,
        )

    # Optional label rendering
    if args.render_labels:
        res = [int(x) for x in args.resolution.split(',') if len(x) > 0]
        _render_labels_from_graphs(
            graph_root=config['output']['directory'],
            label_out_dir=args.label_out_dir,
            resolution=res,
            mip_axis=args.mip_axis,
            binarize=True,
        )

    print("Generation complete.")
    print(f"Graphs: {config['output']['directory']}")
    if args.use_gan:
        print(f"GAN images: {os.path.abspath(args.image_out_dir)}")
    if args.render_labels:
        print(f"Labels: {os.path.abspath(args.label_out_dir)}")


if __name__ == "__main__":
    main()
