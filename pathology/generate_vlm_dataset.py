"""
Generate image–text pairs for VLM finetuning with a simplified, healthy‑first
pathology pipeline.

Pipeline (per sample):
  1) Grow healthy vasculature using the physical Greenhouse model.
  2) Introduce pathologies with minimal, interpretable controls:
     - Capillary dropout (biased toward a para-/perifoveal ring around the FAZ)
     - Optional microaneurysms (near dropout border when enabled)
     - Optional neovascularization (NV) as a compact tuft

Outputs under --out_dir:
  - images/: final images (GAN if --use_gan, else grayscale raster)
  - metadata.jsonl: machine‑readable per‑sample metadata
  - pairs.jsonl: VQA-style conversations per image
  - (no masks are generated)

Example:
  uv run python generate_vlm_dataset.py \
      --num_samples 100 \
      --out_dir ./vlm_dataset \
      --use_gan --gan_config /path/to/gan/config.yml --gan_epoch 150

Notes:
  - Dropout placement is sampled from a FAZ‑centric ring with a higher probability
    close to the FAZ, producing plausible parafoveal/perifoveal patterns.
  - Extras and shape noise have been trimmed to essentials.
  - If masks are saved, simple checks compare vessel density inside vs. around each mask.
    - VQA generation is simplified to a single mixed-style description per image.
  - `--stage pairs` regenerates pairs and metadata from an existing manifest/images.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import copy
import pickle
import yaml
from utils.config_overrides import apply_cli_overrides_from_unknown_args
from typing import Dict, List, Optional, Sequence, Tuple
from typing import Dict, List, Optional, Sequence, Tuple, Any

try:
    import numpy as np  # type: ignore
except Exception:  # Allow --stage pairs without numpy installed
    np = None  # type: ignore
try:
    from PIL import Image  # type: ignore
except Exception:  # Allow --stage pairs without Pillow
    Image = None  # type: ignore
import csv as _csv

# Ensure local imports work when run from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from vessel_graph_generation.utilities import read_config as _read_config  # type: ignore
except Exception:  # Avoid heavy deps for --stage pairs
    _read_config = None  # type: ignore
try:
    from generate_synthetic_octa_images import (  # type: ignore
        _generate_single_sample,
        _call_gan_inference,
    )
except Exception:  # Avoid heavy deps for --stage pairs
    _generate_single_sample = None  # type: ignore
    _call_gan_inference = None  # type: ignore
try:
    from vessel_graph_generation import tree2img  # type: ignore
except Exception:  # Avoid heavy deps for --stage pairs
    tree2img = None  # type: ignore
from vqa_mixed import build_mixed_paragraph

# -----------------------------
# Utilities for two-stage flow
# -----------------------------
def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "records_graphs.jsonl")


def _write_records_manifest(out_dir: str, records: List[Dict[str, object]]) -> str:
    path = _manifest_path(out_dir)
    with open(path, "w") as f:
        for rec in records:
            # Exclude large/transient keys if any; keep core fields
            kept = {
                "id": rec.get("id"),
                "graph_dir": rec.get("graph_dir"),
                "graph_csv": rec.get("graph_csv"),
                "dropout": rec.get("dropout"),
                "neovasc": rec.get("neovasc"),
                "nv_adjacent_to_dropout": rec.get("nv_adjacent_to_dropout"),
                # optional/maybe added later
                "gray_image": rec.get("gray_image"),
                # masks removed permanently
                # structured pathology if available (embedded for convenience)
                "dropouts": rec.get("dropouts"),
                "microaneurysms": rec.get("microaneurysms"),
                "FAZ": rec.get("FAZ"),
                # Prefer emitting a relative path to the per-sample JSON under graph_dir if present
                "dropout_ma_json": (rec.get("dropout_ma_json") or (
                    os.path.relpath(os.path.join(rec.get("graph_dir", ""), "dropout_ma.json"), out_dir)
                    if (rec.get("graph_dir") and os.path.isfile(os.path.join(rec.get("graph_dir", ""), "dropout_ma.json"))) else None
                )),
                "pathology_yml": (
                    os.path.relpath(os.path.join(rec.get("graph_dir", ""), "pathology.yml"), out_dir)
                    if (rec.get("graph_dir") and os.path.isfile(os.path.join(rec.get("graph_dir", ""), "pathology.yml"))) else None
                ),
                # carry per-sample view shift so later stages can pan final images
                "view_shift_norm": (
                    rec.get("view_shift_norm")
                    or (
                        rec.get("dropout_extras", {}).get("_view_shift_norm")
                        if isinstance(rec.get("dropout_extras"), dict)
                        else None
                    )
                ),
                # keep dropout_extras for completeness (may include strength_range etc.)
                "dropout_extras": rec.get("dropout_extras"),
            }
            f.write(json.dumps(kept) + "\n")
    return path


def _load_records_manifest(out_dir: str, manifest_path: Optional[str] = None) -> List[Dict[str, object]]:
    # Priority 1: explicit path
    if manifest_path:
        mpath = os.path.abspath(manifest_path)
        assert os.path.isfile(mpath), f"Manifest not found at --manifest: {mpath}"
    else:
        # Priority 2: expected location at dataset root
        mpath = _manifest_path(out_dir)
        if not os.path.isfile(mpath):
            # Priority 3: fallback — search recursively for any records_graphs.jsonl
            candidates: List[str] = []
            for root, _dirs, files in os.walk(out_dir):
                if "records_graphs.jsonl" in files:
                    candidates.append(os.path.join(root, "records_graphs.jsonl"))
            if len(candidates) == 1:
                mpath = candidates[0]
            elif len(candidates) > 1:
                # Prefer the one at dataset root; else choose the largest file
                roots = [p for p in candidates if os.path.dirname(os.path.abspath(p)) == os.path.abspath(out_dir)]
                if roots:
                    mpath = roots[0]
                else:
                    mpath = max(candidates, key=lambda p: os.path.getsize(p))
            else:
                raise AssertionError(
                    f"Manifest not found under {out_dir}. Run with --stage graphs first or pass --manifest FILE."
                )
    recs: List[Dict[str, object]] = []
    with open(mpath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _resolve_gan_image(images_dir: str, base_id: str) -> Optional[str]:
    """Return relative path to the GAN image matching this base id, regardless of prefix.

    Prefers 'generator_<id>.png' if present, then 'G_<id>.png', then 'pred_<id>.png'. If not found,
    scans the folder for any file ending with _<base_id>.png.
    """
    candidates = [
        os.path.join(images_dir, f"generator_{base_id}.png"),
        os.path.join(images_dir, f"G_{base_id}.png"),
        os.path.join(images_dir, f"pred_{base_id}.png"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.join("images", os.path.basename(c))
    # Fallback: scan directory
    try:
        for fn in os.listdir(images_dir):
            if fn.endswith(f"_{base_id}.png"):
                return os.path.join("images", fn)
    except Exception:
        pass
    return None

@dataclass
class PathologyParams:
    # Representative (largest) dropout region for metadata/QA
    dropout_center: Optional[Tuple[float, float]]
    dropout_radius: Optional[float]
    # Full list of dropout regions used for simulation
    dropout_regions: Optional[List[Tuple[float, float, float]]]
    nv_center: Optional[Tuple[float, float]]
    nv_radius: Optional[float]
    # Per-sample FAZ center actually used for sampling (model-normalized)
    faz_center: Optional[Tuple[float, float]] = None


def _clamp_01(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        return max(0.0, min(1.0, float(val)))
    except Exception:
        return None


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return float(a + (b - a) * t)


def _scale_range(lo_hi0: Sequence[float], lo_hi1: Sequence[float], t: float) -> List[float]:
    return [
        _lerp(float(lo_hi0[0]), float(lo_hi1[0]), t),
        _lerp(float(lo_hi0[1]), float(lo_hi1[1]), t),
    ]


PATHOLOGY_PRESETS: Dict[str, Dict[str, object]] = {
    "dropout_only": {
        "pathology": {
            "dropout_prob": 1.0,
            "nv_prob": 0.0,
            "dropout_ring_radius_range_norm": [0.18, 0.40],
            "dropout_radius_range_norm": [0.18, 0.36],
            "dropout_strength": 0.6,
            "regress_threshold": 0.5,
            "cap_radius_thresh_mm": 0.015,
            "dropout_gradient_alpha_range": [2.1, 2.7],
            "dropout_irregularity_amp_range": [0.08, 0.14],
            "dropout_noise_gain_range": [0.20, 0.32],
            "ring_tangential_bias": 0.5,
            "ring_band": [0.35, 0.7],
            "sparing_large_radius_mm": 0.02,
            "sparing_factor": 0.5,
            "degeneration_fraction_range": [0.22, 0.55],
            "degeneration_core_bias": 1.5,
            "degeneration_min_keep_frac": 0.16,
            "dropout_count_range": [1, 2],
            "small_dropout_radius_range_norm": [0.04, 0.08],
            "mask_vessel_sparing_px": 4,
            "ma_prob": 0.0,
            "ma_radius_mm_range": [0.010, 0.018],
            "ma_len_factor": 0.4,
            "ma_border_bias": 0.6,
            "ma_band": [0.25, 0.85],
            "dilation_scale_range": [1.0, 1.0],
            "dilation_band": [0.30, 0.75],
            "dilation_where": "none",
            "tortuosity_gain": 0.0,
            "tortuosity_band": [0.02, 0.06],
            "nv_radius_range_norm": [0.018, 0.050],
            "nv_anchor_to_dropout_prob": 0.75,
        },
        "general": {
            "skip_venous": False,
        },
        "defaults": {
            "severity": 0.55,
            "nv_severity": 0.0,
            "remodeling_severity": 0.0,
        },
    },
    "balanced_combo": {
        "pathology": {
            "dropout_prob": 0.9,
            "nv_prob": 0.55,
            "dropout_ring_radius_range_norm": [0.18, 0.40],
            "dropout_radius_range_norm": [0.12, 0.26],
            "nv_radius_range_norm": [0.021, 0.060],
            "dropout_strength": 0.65,
            "regress_threshold": 0.5,
            "cap_radius_thresh_mm": 0.015,
            "dropout_gradient_alpha_range": [2.0, 2.8],
            "dropout_irregularity_amp_range": [0.08, 0.16],
            "dropout_noise_gain_range": [0.20, 0.34],
            "ring_tangential_bias": 0.55,
            "ring_band": [0.35, 0.7],
            "sparing_large_radius_mm": 0.02,
            "sparing_factor": 0.5,
            "degeneration_fraction_range": [0.28, 0.65],
            "degeneration_core_bias": 1.55,
            "degeneration_min_keep_frac": 0.14,
            "dropout_count_range": [1, 2],
            "small_dropout_radius_range_norm": [0.04, 0.09],
            "mask_vessel_sparing_px": 4,
            "ma_prob": 0.06,
            "ma_radius_mm_range": [0.010, 0.019],
            "ma_len_factor": 0.42,
            "ma_border_bias": 0.6,
            "ma_band": [0.25, 0.85],
            "dilation_scale_range": [1.0, 1.12],
            "dilation_band": [0.30, 0.75],
            "dilation_where": "border",
            "tortuosity_gain": 0.015,
            "tortuosity_band": [0.02, 0.06],
            "nv_anchor_to_dropout_prob": 0.82,
        },
        "general": {
            "skip_venous": False,
        },
        "defaults": {
            "severity": 0.55,
            "nv_severity": 0.45,
            "remodeling_severity": 0.35,
        },
    },
    "pdr_combo": {
        "pathology": {
            "dropout_prob": 0.95,
            "nv_prob": 0.75,
            "dropout_ring_radius_range_norm": [0.20, 0.42],
            "dropout_radius_range_norm": [0.16, 0.30],
            "nv_radius_range_norm": [0.024, 0.070],
            "dropout_strength": 0.75,
            "regress_threshold": 0.5,
            "cap_radius_thresh_mm": 0.015,
            "dropout_gradient_alpha_range": [2.2, 2.9],
            "dropout_irregularity_amp_range": [0.10, 0.18],
            "dropout_noise_gain_range": [0.24, 0.38],
            "ring_tangential_bias": 0.6,
            "ring_band": [0.35, 0.7],
            "sparing_large_radius_mm": 0.02,
            "sparing_factor": 0.5,
            "degeneration_fraction_range": [0.32, 0.72],
            "degeneration_core_bias": 1.6,
            "degeneration_min_keep_frac": 0.12,
            "dropout_count_range": [1, 3],
            "small_dropout_radius_range_norm": [0.05, 0.10],
            "mask_vessel_sparing_px": 5,
            "ma_prob": 0.12,
            "ma_radius_mm_range": [0.011, 0.020],
            "ma_len_factor": 0.45,
            "ma_border_bias": 0.7,
            "ma_band": [0.22, 0.88],
            "dilation_scale_range": [1.0, 1.15],
            "dilation_band": [0.28, 0.78],
            "dilation_where": "border",
            "tortuosity_gain": 0.025,
            "tortuosity_band": [0.02, 0.07],
            "nv_anchor_to_dropout_prob": 0.85,
        },
        "general": {
            "skip_venous": False,
        },
        "defaults": {
            "severity": 0.75,
            "nv_severity": 0.7,
            "remodeling_severity": 0.55,
        },
    },
}


def _apply_severity_scaling(
    profile_name: str,
    base: Dict[str, object],
    severity: Optional[float],
    nv_severity: Optional[float],
    ma_severity: Optional[float],
    tortuosity_severity: Optional[float],
) -> None:
    defaults = PATHOLOGY_PRESETS.get(profile_name, {}).get("defaults", {})
    sev = _clamp_01(severity if severity is not None else defaults.get("severity"))
    nv_sev = _clamp_01(nv_severity if nv_severity is not None else defaults.get("nv_severity"))
    legacy_remodel = defaults.get("remodeling_severity")
    ma_sev = _clamp_01(ma_severity if ma_severity is not None else legacy_remodel)
    tort_sev = _clamp_01(tortuosity_severity if tortuosity_severity is not None else legacy_remodel)

    if profile_name == "dropout_only":
        nv_sev = 0.0
        ma_sev = 0.0
        tort_sev = 0.0

    if sev is not None:
        base["dropout_strength"] = _lerp(0.45, 0.9, sev)
        base["dropout_radius_range_norm"] = _scale_range([0.10, 0.20], [0.18, 0.32], sev)
        base["dropout_ring_radius_range_norm"] = _scale_range([0.16, 0.34], [0.22, 0.42], sev)
        base["dropout_irregularity_amp_range"] = _scale_range([0.06, 0.12], [0.12, 0.18], sev)
        base["dropout_noise_gain_range"] = _scale_range([0.18, 0.26], [0.26, 0.40], sev)
        base["dropout_gradient_alpha_range"] = _scale_range([1.9, 2.4], [2.5, 3.05], sev)
        base["small_dropout_radius_range_norm"] = _scale_range([0.03, 0.06], [0.06, 0.10], sev)
        count_max = 1
        if sev >= 0.75:
            count_max = 3
        elif sev >= 0.45:
            count_max = 2
        base["dropout_count_range"] = [1, count_max]
        base["mask_vessel_sparing_px"] = int(round(_lerp(3.0, 6.0, sev)))
        base["degeneration_fraction_range"] = _scale_range([0.20, 0.45], [0.45, 0.80], sev)
        base["degeneration_core_bias"] = _lerp(1.1, 1.9, sev)
        base["degeneration_min_keep_frac"] = _lerp(0.18, 0.08, sev)

    if nv_sev is not None:
        if nv_sev <= 0.0:
            base["nv_prob"] = 0.0
        else:
            # Make NV occurrence and size subtler across severities
            base["nv_prob"] = _lerp(0.10, 0.50, nv_sev)
        base["nv_radius_range_norm"] = _scale_range([0.015, 0.035], [0.022, 0.055], nv_sev or 0.0)
        base["nv_anchor_to_dropout_prob"] = _lerp(0.5, 0.78, nv_sev or 0.0)

    # Microaneurysm realism (detached)
    if ma_sev is not None:
        if ma_sev <= 0.0:
            base["ma_prob"] = 0.0
        else:
            base["ma_prob"] = _lerp(0.03, 0.12, ma_sev)
        base["ma_radius_mm_range"] = _scale_range([0.010, 0.016], [0.0115, 0.019], ma_sev or 0.0)
        base["ma_len_factor"] = _lerp(0.26, 0.38, ma_sev or 0.0)
        base["ma_border_bias"] = _lerp(0.55, 0.85, ma_sev or 0.0)
        base["ma_band"] = _scale_range([0.24, 0.82], [0.20, 0.88], ma_sev or 0.0)
    # Tortuosity (detached) — keep subtle; local dilation minimal (global sparsity handled elsewhere)
    if tort_sev is not None:
        base["tortuosity_gain"] = _lerp(0.0, 0.035, tort_sev)
        base.setdefault("tortuosity_band", [0.02, 0.07])
        base["dilation_scale_range"] = [1.0, 1.06] if (tort_sev or 0.0) > 0.0 else [1.0, 1.0]
        base.setdefault("dilation_band", [0.30, 0.70])
        base["dilation_where"] = "border"


def _resolve_pathology_profile(patho_cfg: Optional[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    if not patho_cfg:
        return {}, {}

    profile = patho_cfg.get("profile")
    general_defaults: Dict[str, object] = {}

    if profile:
        preset = PATHOLOGY_PRESETS.get(str(profile))
        if preset is None:
            raise ValueError(f"Unknown pathology profile '{profile}'. Known profiles: {sorted(PATHOLOGY_PRESETS.keys())}")
        base = copy.deepcopy(preset.get("pathology", {}))
        general_defaults = copy.deepcopy(preset.get("general", {}))
        _apply_severity_scaling(
            str(profile),
            base,
            patho_cfg.get("severity"),
            patho_cfg.get("nv_severity"),
            patho_cfg.get("ma_severity"),
            patho_cfg.get("tortuosity_severity") or patho_cfg.get("remodeling_severity") or patho_cfg.get("remodeling"),
        )
        special_keys = {"profile", "severity", "nv_severity", "remodeling_severity", "remodeling", "overrides", "general_overrides"}
        special_keys.update({"ma_severity", "tortuosity_severity"})
        direct = {k: v for k, v in patho_cfg.items() if k not in special_keys}
        base.update(direct)

        # Optional: support nested component blocks Pathology.Dropout/MA/NV/Tortuosity
        def _merge_components() -> None:
            comp = {
                "Dropout": patho_cfg.get("Dropout"),
                "MA": patho_cfg.get("MA"),
                "NV": patho_cfg.get("NV"),
                "Tortuosity": patho_cfg.get("Tortuosity"),
            }
            # Dropout mappings
            d = comp.get("Dropout")
            if isinstance(d, dict):
                if "probability" in d:
                    base["dropout_prob"] = float(d["probability"])  # 0..1
                if "count_range" in d and isinstance(d["count_range"], (list, tuple)) and len(d["count_range"]) == 2:
                    base["dropout_count_range"] = [int(d["count_range"][0]), int(d["count_range"][1])]
                if "radius_norm_range" in d and isinstance(d["radius_norm_range"], (list, tuple)) and len(d["radius_norm_range"]) == 2:
                    base["dropout_radius_range_norm"] = [float(d["radius_norm_range"][0]), float(d["radius_norm_range"][1])]
                if "ring_radius_norm_range" in d and isinstance(d["ring_radius_norm_range"], (list, tuple)) and len(d["ring_radius_norm_range"]) == 2:
                    base["dropout_ring_radius_range_norm"] = [float(d["ring_radius_norm_range"][0]), float(d["ring_radius_norm_range"][1])]
                # New: allow specifying strength or a range for strength
                if "dropout_strength" in d and d["dropout_strength"] is not None:
                    try:
                        base["dropout_strength"] = float(d["dropout_strength"])
                    except Exception:
                        pass
                if "strength" in d and d["strength"] is not None:
                    try:
                        base["dropout_strength"] = float(d["strength"])
                    except Exception:
                        pass
                if "dropout_strength_range" in d and isinstance(d["dropout_strength_range"], (list, tuple)) and len(d["dropout_strength_range"]) == 2:
                    try:
                        base["dropout_strength_range"] = [float(d["dropout_strength_range"][0]), float(d["dropout_strength_range"][1])]
                    except Exception:
                        pass
                if "strength_range" in d and isinstance(d["strength_range"], (list, tuple)) and len(d["strength_range"]) == 2:
                    try:
                        base["dropout_strength_range"] = [float(d["strength_range"][0]), float(d["strength_range"][1])]
                    except Exception:
                        pass
                if "radius_scale_vs_ring" in d and isinstance(d["radius_scale_vs_ring"], (list, tuple)) and len(d["radius_scale_vs_ring"]) == 2:
                    base["dropout_radius_scale_vs_ring"] = [float(d["radius_scale_vs_ring"][0]), float(d["radius_scale_vs_ring"][1])]
                if "shape_randomness" in d:
                    base["dropout_shape_randomness"] = float(d["shape_randomness"])  # 0..1
                # Whether to ensure one largest region among all dropouts
                if "ensure_one_large" in d:
                    base["dropout_ensure_one_large"] = bool(d["ensure_one_large"])  # default True
                if "ensure_one_large_region" in d:
                    base["dropout_ensure_one_large"] = bool(d["ensure_one_large_region"])  # alias
                # Optional vessel strength and border behavior, colocated under Dropout for convenience
                if "dilation_scale_range" in d and isinstance(d["dilation_scale_range"], (list, tuple)) and len(d["dilation_scale_range"]) == 2:
                    base["dilation_scale_range"] = [float(d["dilation_scale_range"][0]), float(d["dilation_scale_range"][1])]
                if "dilation_band" in d and isinstance(d["dilation_band"], (list, tuple)) and len(d["dilation_band"]) == 2:
                    base["dilation_band"] = [float(d["dilation_band"][0]), float(d["dilation_band"][1])]
                if "dilation_where" in d:
                    base["dilation_where"] = str(d["dilation_where"]).lower()
                if "regress_threshold" in d:
                    base["regress_threshold"] = float(d["regress_threshold"])  # 0..1 inside core
                if "degeneration_fraction_range" in d and isinstance(d["degeneration_fraction_range"], (list, tuple)) and len(d["degeneration_fraction_range"]) == 2:
                    base["degeneration_fraction_range"] = [float(d["degeneration_fraction_range"][0]), float(d["degeneration_fraction_range"][1])]
                if "degeneration_fraction" in d and d["degeneration_fraction"] is not None:
                    try:
                        base["degeneration_fraction"] = float(d["degeneration_fraction"])
                    except Exception:
                        pass
                if "cap_radius_thresh_mm" in d:
                    base["cap_radius_thresh_mm"] = float(d["cap_radius_thresh_mm"])  # fragile leaf cutoff
                if "sparing_large_radius_mm" in d:
                    base["sparing_large_radius_mm"] = float(d["sparing_large_radius_mm"])  # mm
                if "sparing_factor" in d:
                    base["sparing_factor"] = float(d["sparing_factor"])  # 0..1
            # Microaneurysms mappings
            ma = comp.get("MA")
            if isinstance(ma, dict):
                if "density" in ma:
                    base["ma_prob"] = float(ma["density"])  # per-new-segment probability in region
                if "size_mm_range" in ma and isinstance(ma["size_mm_range"], (list, tuple)) and len(ma["size_mm_range"]) == 2:
                    base["ma_radius_mm_range"] = [float(ma["size_mm_range"][0]), float(ma["size_mm_range"][1])]
                if "length_factor" in ma:
                    base["ma_len_factor"] = float(ma["length_factor"])  # optional
                if "near_dropout_only" in ma:
                    base["ma_only_near_dropout"] = bool(ma["near_dropout_only"])  # default True
                # Note: parent radius constraint removed; MA placement no longer filters by parent size
                # Optional MA-only radius clamps so balloons don't require global r_max
                if "r_min_mm" in ma and ma["r_min_mm"] is not None:
                    base["ma_r_min_mm"] = float(ma["r_min_mm"])  # applies only to MA nodes
                if "r_max_mm" in ma and ma["r_max_mm"] is not None:
                    base["ma_r_max_mm"] = float(ma["r_max_mm"])  # applies only to MA nodes
                # Shape randomness for MA overlay/visuals (0..1)
                if "shape_randomness" in ma:
                    try:
                        base["ma_shape_randomness"] = float(ma["shape_randomness"])
                    except Exception:
                        pass
                # Optional band region for MA placement relative to dropout core (0..1)
                if "band" in ma and isinstance(ma["band"], (list, tuple)) and len(ma["band"]) == 2:
                    try:
                        base["ma_band"] = [float(ma["band"][0]), float(ma["band"][1])]
                    except Exception:
                        pass
                # Optional border bias (amplifies MA probability inside band)
                if "border_bias" in ma and ma["border_bias"] is not None:
                    try:
                        base["ma_border_bias"] = float(ma["border_bias"])
                    except Exception:
                        pass
                # Optional: couple MA spawn probability to dropout strength
                # YAML key: MA.prob_vs_dropout_strength_gain (alias: prob_strength_gain)
                if "prob_vs_dropout_strength_gain" in ma and ma["prob_vs_dropout_strength_gain"] is not None:
                    try:
                        base["ma_prob_strength_gain"] = float(ma["prob_vs_dropout_strength_gain"])  # p *= 1 + gain * strength
                    except Exception:
                        pass
                elif "prob_strength_gain" in ma and ma["prob_strength_gain"] is not None:
                    try:
                        base["ma_prob_strength_gain"] = float(ma["prob_strength_gain"])  # alias
                    except Exception:
                        pass
            # Neovascularization mappings
            nv = comp.get("NV")
            if isinstance(nv, dict):
                if "enable" in nv and not bool(nv["enable"]):
                    base["nv_prob"] = 0.0
                if "probability" in nv:
                    base["nv_prob"] = float(nv["probability"])  # 0..1
                if "probability_range" in nv and isinstance(nv["probability_range"], (list, tuple)) and len(nv["probability_range"]) == 2:
                    base["nv_prob_range"] = [float(nv["probability_range"][0]), float(nv["probability_range"][1])]
                if "radius_norm_range" in nv and isinstance(nv["radius_norm_range"], (list, tuple)) and len(nv["radius_norm_range"]) == 2:
                    base["nv_radius_range_norm"] = [float(nv["radius_norm_range"][0]), float(nv["radius_norm_range"][1])]
                # Allow compact control: severity in [0,1] drives all growth knobs
                if nv.get("severity") is not None:
                    try:
                        s = max(0.0, min(1.0, float(nv["severity"])))
                        # Map severity to a consistent, visible tuft
                        base["nv_step_len_factor"] = float(_lerp(0.90, 1.30, s))
                        base["nv_edges_per_iter"] = int(round(_lerp(16.0, 48.0, s)))
                        base["nv_branch_prob"] = float(_lerp(0.20, 0.50, s))
                        base["nv_curl_factor"] = float(_lerp(0.50, 0.85, s))
                        base["nv_mean_radius_mm"] = float(_lerp(0.0014, 0.0025, s))
                        base["nv_std_radius_mm"] = float(_lerp(0.0003, 0.0006, s))
                        # Ensure sufficient NV growth iterations for visibility
                        base["nv_post_iters"] = int(round(_lerp(3.0, 6.0, s)))
                        base["nv_init_spokes"] = int(round(_lerp(3.0, 5.0, s)))
                    except Exception:
                        pass
                # Optional: explicitly control dropout adjacency probability with a friendly name
                if nv.get("adjacent_to_dropout_prob") is not None:
                    try:
                        base["nv_anchor_to_dropout_prob"] = float(nv["adjacent_to_dropout_prob"])  # alias
                    except Exception:
                        pass
                # Optional NV strength configuration
                if "strength" in nv and nv["strength"] is not None:
                    try:
                        base["nv_strength"] = float(nv["strength"])
                    except Exception:
                        pass
                if "strength_range" in nv and isinstance(nv["strength_range"], (list, tuple)) and len(nv["strength_range"]) == 2:
                    try:
                        base["nv_strength_range"] = [float(nv["strength_range"][0]), float(nv["strength_range"][1])]
                    except Exception:
                        pass
                # Growth/shape tuning (pass-through to GreenhouseDropout NVRegion via extras)
                if "step_len_factor" in nv:
                    base["nv_step_len_factor"] = float(nv["step_len_factor"])  # relative to d
                if "mean_radius_mm" in nv:
                    base["nv_mean_radius_mm"] = float(nv["mean_radius_mm"])  # mm
                if "std_radius_mm" in nv:
                    base["nv_std_radius_mm"] = float(nv["std_radius_mm"])  # mm
                if "branch_prob" in nv:
                    base["nv_branch_prob"] = float(nv["branch_prob"])  # 0..1
                if "curl_factor" in nv:
                    base["nv_curl_factor"] = float(nv["curl_factor"])  # 0..1 noise blend
                if "edges_per_iter" in nv:
                    base["nv_edges_per_iter"] = int(nv["edges_per_iter"])  # budget per iter
                if "irregularity_amp" in nv:
                    base["nv_irregularity_amp"] = float(nv["irregularity_amp"])  # 0..1
                if "harmonics" in nv and isinstance(nv["harmonics"], (list, tuple)):
                    try:
                        base["nv_harmonics"] = [int(h) for h in nv["harmonics"]]
                    except Exception:
                        pass
                if "ellipticity" in nv and isinstance(nv["ellipticity"], (list, tuple)) and len(nv["ellipticity"]) == 2:
                    try:
                        base["nv_ellipticity"] = [float(nv["ellipticity"][0]), float(nv["ellipticity"][1])]
                    except Exception:
                        pass
                if "gradient_alpha" in nv:
                    base["nv_gradient_alpha"] = float(nv["gradient_alpha"])
                if "noise_gain" in nv:
                    base["nv_noise_gain"] = float(nv["noise_gain"])
                if "connect_prob" in nv:
                    base["nv_connect_prob"] = float(nv["connect_prob"])  # loop/anastomosis chance
                if "connect_radius_norm" in nv:
                    base["nv_connect_radius_norm"] = float(nv["connect_radius_norm"])  # not used directly, passthrough
                if "border_band_norm" in nv:
                    base["nv_border_band_norm"] = float(nv["border_band_norm"])  # normalized
                if "border_bias" in nv:
                    base["nv_border_bias"] = float(nv["border_bias"])  # 0..1 tangential bias
                if "outward_bias" in nv:
                    base["nv_outward_bias"] = float(nv["outward_bias"])  # 0..1 radial preference
                if "init_spokes" in nv:
                    try:
                        base["nv_init_spokes"] = int(nv["init_spokes"])  # initial spoke count
                    except Exception:
                        pass
                if "spoke_jitter" in nv:
                    try:
                        base["nv_spoke_jitter"] = float(nv["spoke_jitter"])  # angle jitter factor
                    except Exception:
                        pass
                if "post_iters" in nv:
                    try:
                        base["nv_post_iters"] = int(nv["post_iters"])  # total NV growth passes
                    except Exception:
                        pass
                if "min_clearance_vessel_norm" in nv:
                    try:
                        base["nv_min_clearance_vessel_norm"] = float(nv["min_clearance_vessel_norm"])  # spacing vs existing vessels
                    except Exception:
                        pass
                if "min_clearance_nv_norm" in nv:
                    try:
                        base["nv_min_clearance_nv_norm"] = float(nv["min_clearance_nv_norm"])  # spacing vs own tuft
                    except Exception:
                        pass
                # Overlay sub-config: keep minimal knobs; pass through for later composition
                if isinstance(nv.get("overlay"), dict):
                    try:
                        ov = dict(nv.get("overlay") or {})
                        base["nv_overlay"] = {
                            k: ov[k] for k in ov.keys()
                            if k in ("enable", "spokes", "steps", "step_px", "curl", "jitter", "thickness_px", "fill_alpha")
                        }
                    except Exception:
                        pass
            # Tortuosity mappings
            tor = comp.get("Tortuosity")
            if isinstance(tor, dict):
                if "gain" in tor:
                    base["tortuosity_gain"] = float(tor["gain"])  # 0..1
                if "gain_range" in tor and isinstance(tor["gain_range"], (list, tuple)) and len(tor["gain_range"]) == 2:
                    base["tortuosity_gain_range"] = [float(tor["gain_range"][0]), float(tor["gain_range"][1])]
                if "probability" in tor:
                    base["tortuosity_prob"] = float(tor["probability"])  # 0..1
                if "probability_range" in tor and isinstance(tor["probability_range"], (list, tuple)) and len(tor["probability_range"]) == 2:
                    base["tortuosity_prob_range"] = [float(tor["probability_range"][0]), float(tor["probability_range"][1])]
                if "band" in tor and isinstance(tor["band"], (list, tuple)) and len(tor["band"]) == 2:
                    base["tortuosity_band"] = [float(tor["band"][0]), float(tor["band"][1])]
                # Dilation controls should be configured under Dropout; Tortuosity no longer carries them

        _merge_components()
        overrides = patho_cfg.get("overrides")
        if isinstance(overrides, dict):
            base.update(overrides)
        general_overrides = patho_cfg.get("general_overrides")
        if isinstance(general_overrides, dict):
            general_defaults.update(general_overrides)
        return base, general_defaults

    # Fallback to legacy behaviour: include optional nested components if present
    base = copy.deepcopy(patho_cfg)
    base.pop("general_overrides", None)
    base.pop("overrides", None)
    # Merge nested components into flat keys if provided
    def _merge_components_into(b: Dict[str, object], src: Dict[str, object]) -> None:
            d = src.get("Dropout") if isinstance(src.get("Dropout"), dict) else None
            if isinstance(d, dict):
                if "probability" in d:
                    b["dropout_prob"] = float(d["probability"])  # 0..1
                if "count_range" in d and isinstance(d["count_range"], (list, tuple)) and len(d["count_range"]) == 2:
                    b["dropout_count_range"] = [int(d["count_range"][0]), int(d["count_range"][1])]
                if "radius_norm_range" in d and isinstance(d["radius_norm_range"], (list, tuple)) and len(d["radius_norm_range"]) == 2:
                    b["dropout_radius_range_norm"] = [float(d["radius_norm_range"][0]), float(d["radius_norm_range"][1])]
                if "ring_radius_norm_range" in d and isinstance(d["ring_radius_norm_range"], (list, tuple)) and len(d["ring_radius_norm_range"]) == 2:
                    b["dropout_ring_radius_range_norm"] = [float(d["ring_radius_norm_range"][0]), float(d["ring_radius_norm_range"][1])]
                # Ensure legacy nested Dropout fields map to flat keys consistently
                if "small_dropout_radius_range_norm" in d and isinstance(d["small_dropout_radius_range_norm"], (list, tuple)) and len(d["small_dropout_radius_range_norm"]) == 2:
                    try:
                        b["small_dropout_radius_range_norm"] = [
                            float(d["small_dropout_radius_range_norm"][0]),
                            float(d["small_dropout_radius_range_norm"][1]),
                        ]
                    except Exception:
                        pass
                # New: allow specifying strength or range for strength from Dropout block
                if "dropout_strength" in d and d["dropout_strength"] is not None:
                    try:
                        b["dropout_strength"] = float(d["dropout_strength"])
                    except Exception:
                        pass
                if "strength" in d and d["strength"] is not None:
                    try:
                        b["dropout_strength"] = float(d["strength"])
                    except Exception:
                        pass
                if "dropout_strength_range" in d and isinstance(d["dropout_strength_range"], (list, tuple)) and len(d["dropout_strength_range"]) == 2:
                    try:
                        b["dropout_strength_range"] = [float(d["dropout_strength_range"][0]), float(d["dropout_strength_range"][1])]
                    except Exception:
                        pass
                if "strength_range" in d and isinstance(d["strength_range"], (list, tuple)) and len(d["strength_range"]) == 2:
                    try:
                        b["dropout_strength_range"] = [float(d["strength_range"][0]), float(d["strength_range"][1])]
                    except Exception:
                        pass
                # Whether to ensure one larger region among all dropouts
                if "ensure_one_large" in d:
                    try:
                        b["dropout_ensure_one_large"] = bool(d["ensure_one_large"])  # default True
                    except Exception:
                        pass
                if "ensure_one_large_region" in d:
                    try:
                        b["dropout_ensure_one_large"] = bool(d["ensure_one_large_region"])  # alias
                    except Exception:
                        pass
                # Optional radius-vs-center scaling: larger near FAZ, smaller towards periphery
                if "radius_scale_vs_ring" in d and isinstance(d["radius_scale_vs_ring"], (list, tuple)) and len(d["radius_scale_vs_ring"]) == 2:
                    try:
                        b["dropout_radius_scale_vs_ring"] = [float(d["radius_scale_vs_ring"][0]), float(d["radius_scale_vs_ring"][1])]
                    except Exception:
                        pass
                # Shape randomness for dropout overlay/visuals (0..1)
                if "shape_randomness" in d and d["shape_randomness"] is not None:
                    try:
                        b["dropout_shape_randomness"] = float(d["shape_randomness"])  # 0..1
                    except Exception:
                        pass
                if "regression_passes" in d:
                    try:
                        b["regression_passes"] = int(d["regression_passes"])
                    except Exception:
                        pass
                if "degeneration_fraction_range" in d and isinstance(d["degeneration_fraction_range"], (list, tuple)) and len(d["degeneration_fraction_range"]) == 2:
                    try:
                        b["degeneration_fraction_range"] = [float(d["degeneration_fraction_range"][0]), float(d["degeneration_fraction_range"][1])]
                    except Exception:
                        pass
                if "degeneration_fraction" in d and d["degeneration_fraction"] is not None:
                    try:
                        b["degeneration_fraction"] = float(d["degeneration_fraction"])
                    except Exception:
                        pass
                if "regress_threshold_range" in d and isinstance(d["regress_threshold_range"], (list, tuple)) and len(d["regress_threshold_range"]) == 2:
                    try:
                        b["regress_threshold_range"] = [float(d["regress_threshold_range"][0]), float(d["regress_threshold_range"][1])]
                    except Exception:
                        pass
                if "regression_passes" in d:
                    try:
                        b["regression_passes"] = int(d["regression_passes"])
                    except Exception:
                        pass
                if "degeneration_core_bias" in d:
                    try:
                        b["degeneration_core_bias"] = float(d["degeneration_core_bias"])
                    except Exception:
                        pass
                if "degeneration_min_keep_frac" in d:
                    try:
                        b["degeneration_min_keep_frac"] = float(d["degeneration_min_keep_frac"])
                    except Exception:
                        pass
                # Dropout-local dilation and ring shaping controls
                if "dilation_scale_range" in d and isinstance(d["dilation_scale_range"], (list, tuple)) and len(d["dilation_scale_range"]) == 2:
                    try:
                        b["dilation_scale_range"] = [float(d["dilation_scale_range"][0]), float(d["dilation_scale_range"][1])]
                    except Exception:
                        pass
                if "dilation_band" in d and isinstance(d["dilation_band"], (list, tuple)) and len(d["dilation_band"]) == 2:
                    try:
                        b["dilation_band"] = [float(d["dilation_band"][0]), float(d["dilation_band"][1])]
                    except Exception:
                        pass
                if "dilation_where" in d:
                    try:
                        b["dilation_where"] = str(d["dilation_where"]).lower()
                    except Exception:
                        pass
                if "ring_tangential_bias" in d and d["ring_tangential_bias"] is not None:
                    try:
                        b["ring_tangential_bias"] = float(d["ring_tangential_bias"])  # 0..1
                    except Exception:
                        pass
                if "ring_band" in d and isinstance(d["ring_band"], (list, tuple)) and len(d["ring_band"]) == 2:
                    try:
                        b["ring_band"] = [float(d["ring_band"][0]), float(d["ring_band"][1])]
                    except Exception:
                        pass
                if "cap_radius_thresh_mm" in d and d["cap_radius_thresh_mm"] is not None:
                    try:
                        b["cap_radius_thresh_mm"] = float(d["cap_radius_thresh_mm"])  # fragile leaf cutoff
                    except Exception:
                        pass
                if "sparing_large_radius_mm" in d and d["sparing_large_radius_mm"] is not None:
                    try:
                        b["sparing_large_radius_mm"] = float(d["sparing_large_radius_mm"])  # mm
                    except Exception:
                        pass
                if "sparing_factor" in d and d["sparing_factor"] is not None:
                    try:
                        b["sparing_factor"] = float(d["sparing_factor"])  # 0..1
                    except Exception:
                        pass
            ma = src.get("MA") if isinstance(src.get("MA"), dict) else None
            if isinstance(ma, dict):
                if "density" in ma:
                    b["ma_prob"] = float(ma["density"])  # 0..1
                if "size_mm_range" in ma and isinstance(ma["size_mm_range"], (list, tuple)) and len(ma["size_mm_range"]) == 2:
                    b["ma_radius_mm_range"] = [float(ma["size_mm_range"][0]), float(ma["size_mm_range"][1])]
                if "length_factor" in ma:
                    b["ma_len_factor"] = float(ma["length_factor"])  # optional
                if "near_dropout_only" in ma:
                    b["ma_only_near_dropout"] = bool(ma["near_dropout_only"])  # default True
                # Note: parent radius constraint removed; MA placement no longer filters by parent size
                # Optional MA-only radius clamps
                if "r_min_mm" in ma and ma["r_min_mm"] is not None:
                    b["ma_r_min_mm"] = float(ma["r_min_mm"])  # only affects MA nodes
                if "r_max_mm" in ma and ma["r_max_mm"] is not None:
                    b["ma_r_max_mm"] = float(ma["r_max_mm"])  # only affects MA nodes
                # Shape randomness for MA overlay/visuals (0..1)
                if "shape_randomness" in ma:
                    try:
                        b["ma_shape_randomness"] = float(ma["shape_randomness"])
                    except Exception:
                        pass
                # Band region for MA placement (relative to dropout core score 0..1)
                if "band" in ma and isinstance(ma["band"], (list, tuple)) and len(ma["band"]) == 2:
                    try:
                        b["ma_band"] = [float(ma["band"][0]), float(ma["band"][1])]
                    except Exception:
                        pass
                # Border bias for MA probability inside band
                if "border_bias" in ma and ma["border_bias"] is not None:
                    try:
                        b["ma_border_bias"] = float(ma["border_bias"])
                    except Exception:
                        pass
            nv = src.get("NV") if isinstance(src.get("NV"), dict) else None
            if isinstance(nv, dict):
                if "enable" in nv and not bool(nv["enable"]):
                    b["nv_prob"] = 0.0
                if "probability" in nv:
                    b["nv_prob"] = float(nv["probability"])  # 0..1
                if "probability_range" in nv and isinstance(nv["probability_range"], (list, tuple)) and len(nv["probability_range"]) == 2:
                    b["nv_prob_range"] = [float(nv["probability_range"][0]), float(nv["probability_range"][1])]
                if "radius_norm_range" in nv and isinstance(nv["radius_norm_range"], (list, tuple)) and len(nv["radius_norm_range"]) == 2:
                    b["nv_radius_range_norm"] = [float(nv["radius_norm_range"][0]), float(nv["radius_norm_range"][1])]
                if "strength" in nv and nv["strength"] is not None:
                    try:
                        b["nv_strength"] = float(nv["strength"])
                    except Exception:
                        pass
                if "strength_range" in nv and isinstance(nv["strength_range"], (list, tuple)) and len(nv["strength_range"]) == 2:
                    try:
                        b["nv_strength_range"] = [float(nv["strength_range"][0]), float(nv["strength_range"][1])]
                    except Exception:
                        pass
            tor = src.get("Tortuosity") if isinstance(src.get("Tortuosity"), dict) else None
            if isinstance(tor, dict):
                if "gain" in tor:
                    b["tortuosity_gain"] = float(tor["gain"])  # 0..1
                if "gain_range" in tor and isinstance(tor["gain_range"], (list, tuple)) and len(tor["gain_range"]) == 2:
                    b["tortuosity_gain_range"] = [float(tor["gain_range"][0]), float(tor["gain_range"][1])]
                if "probability" in tor:
                    b["tortuosity_prob"] = float(tor["probability"])  # 0..1
                if "probability_range" in tor and isinstance(tor["probability_range"], (list, tuple)) and len(tor["probability_range"]) == 2:
                    b["tortuosity_prob_range"] = [float(tor["probability_range"][0]), float(tor["probability_range"][1])]
                if "band" in tor and isinstance(tor["band"], (list, tuple)) and len(tor["band"]) == 2:
                    b["tortuosity_band"] = [float(tor["band"][0]), float(tor["band"][1])]
                # Dilation controls are configured under Dropout; ignore Tortuosity duplicates
    _merge_components_into(base, base)
    return base, general_defaults


_WORKER_CONFIG_BYTES: Optional[bytes] = None
_WORKER_GRAPH_ROOT: Optional[str] = None
_WORKER_SKIP_VENOUS: bool = False
_WORKER_WRITE_CONFIG: bool = True
_WORKER_SAVE_MASKS: bool = False
_WORKER_MASK_THRESHOLD: float = 0.5
_WORKER_MASK_EXPAND_PX: int = 0
_WORKER_MASK_VESSEL_SPARE_PX: int = 0


def _worker_init(
    config_bytes: bytes,
    graph_root: str,
    skip_venous: bool,
    write_config_per_sample: bool,
    save_masks: bool,
    mask_threshold: float,
    mask_expand_px: int,
    mask_vessel_spare_px: int,
) -> None:
    global _WORKER_CONFIG_BYTES, _WORKER_GRAPH_ROOT, _WORKER_SKIP_VENOUS, _WORKER_WRITE_CONFIG, _WORKER_SAVE_MASKS, _WORKER_MASK_THRESHOLD, _WORKER_MASK_EXPAND_PX, _WORKER_MASK_VESSEL_SPARE_PX
    _WORKER_CONFIG_BYTES = bytes(config_bytes)
    _WORKER_GRAPH_ROOT = graph_root
    _WORKER_SKIP_VENOUS = bool(skip_venous)
    _WORKER_WRITE_CONFIG = bool(write_config_per_sample)
    _WORKER_SAVE_MASKS = bool(save_masks)
    try:
        _WORKER_MASK_THRESHOLD = float(mask_threshold)
    except Exception:
        _WORKER_MASK_THRESHOLD = 0.5
    try:
        _WORKER_MASK_EXPAND_PX = int(mask_expand_px)
    except Exception:
        _WORKER_MASK_EXPAND_PX = 0
    try:
        _WORKER_MASK_VESSEL_SPARE_PX = int(mask_vessel_spare_px)
    except Exception:
        _WORKER_MASK_VESSEL_SPARE_PX = 0


def _sample_dropout_and_nv(param_scale_mm: float, rng: random.Random, patho_cfg: Dict | None = None, faz_center: Tuple[float, float] = (0.5, 0.5)) -> PathologyParams:
    """
    Sample clinically reasonable locations for 3x3 mm macula simulation.

    - Dropout: parafoveal/perifoveal ring, avoid FAZ center.
    - NV: if dropout exists, place near its border; else place in para/perifovea.
    """
    # Defaults (centered on a FAZ-centric ring; dropout/NV radii kept modest)
    p_dropout = 0.7
    p_nv = 0.5
    ring_lo, ring_hi = 0.18, 0.40
    drop_r_lo, drop_r_hi = 0.06, 0.14
    nv_r_lo, nv_r_hi = 0.04, 0.09
    nv_anchor_prob = 0.8

    if patho_cfg:
        p_dropout = float(patho_cfg.get("dropout_prob", p_dropout))
        # NV probability: support fixed or range
        if patho_cfg.get("nv_prob_range") is not None:
            try:
                lo, hi = patho_cfg.get("nv_prob_range")
                p_nv = float(random.uniform(float(lo), float(hi)))
            except Exception:
                p_nv = float(patho_cfg.get("nv_prob", p_nv))
        else:
            p_nv = float(patho_cfg.get("nv_prob", p_nv))
        rr = patho_cfg.get("dropout_ring_radius_range_norm") or [ring_lo, ring_hi]
        ring_lo, ring_hi = float(rr[0]), float(rr[1])
        dr = patho_cfg.get("dropout_radius_range_norm") or [drop_r_lo, drop_r_hi]
        drop_r_lo, drop_r_hi = float(dr[0]), float(dr[1])
        nr = patho_cfg.get("nv_radius_range_norm") or [nv_r_lo, nv_r_hi]
        nv_r_lo, nv_r_hi = float(nr[0]), float(nr[1])
        nv_anchor_prob = float(patho_cfg.get("nv_anchor_to_dropout_prob", nv_anchor_prob))

    have_dropout = rng.random() < p_dropout
    # Enforce NV only when dropout exists (anchor at dropout edge requirement)
    have_nv = have_dropout and (rng.random() < p_nv)

    fcx, fcy = float(faz_center[0]), float(faz_center[1])

    dc = dr = nc = nr = None
    all_dropouts: List[Tuple[float, float, float]] | None = None
    if have_dropout:
        # Determine number of regions (supports both small and large regions)
        count_min, count_max = 1, 1
        if patho_cfg and patho_cfg.get("dropout_count_range"):
            try:
                c = patho_cfg.get("dropout_count_range")
                count_min, count_max = int(c[0]), int(c[1])
            except Exception:
                count_min, count_max = 1, 1
        n_regions = rng.randint(count_min, max(count_min, count_max))
        # Small region radius range (optional)
        s_lo, s_hi = max(0.02, drop_r_lo * 0.5), max(0.03, drop_r_lo * 0.9)
        if patho_cfg and patho_cfg.get("small_dropout_radius_range_norm"):
            try:
                ss = patho_cfg.get("small_dropout_radius_range_norm")
                s_lo, s_hi = float(ss[0]), float(ss[1])
            except Exception:
                pass
        regs: List[Tuple[float, float, float]] = []
        # Control whether to ensure one larger region and smaller others
        large_first = True
        try:
            if patho_cfg is not None and patho_cfg.get("dropout_ensure_one_large") is not None:
                large_first = bool(patho_cfg.get("dropout_ensure_one_large"))
        except Exception:
            large_first = True
        # Optional radius-vs-center scaling: larger near FAZ, smaller towards periphery
        rad_vs_ring = patho_cfg.get("dropout_radius_scale_vs_ring") if patho_cfg else None
        if isinstance(rad_vs_ring, (list, tuple)) and len(rad_vs_ring) == 2:
            scale_center, scale_outer = float(rad_vs_ring[0]), float(rad_vs_ring[1])
        else:
            # Default modest bias: +20% near center, -5% at outer ring
            scale_center, scale_outer = 1.20, 0.95

        def _scale_for_ring(ring_val: float) -> float:
            t = (ring_val - ring_lo) / max(1e-6, (ring_hi - ring_lo))
            t = max(0.0, min(1.0, t))
            return _lerp(scale_center, scale_outer, t)

        for i in range(n_regions):
            if large_first:
                r_range = (drop_r_lo, drop_r_hi) if (i == 0) else (s_lo, s_hi)
            else:
                # All regions sampled in the same range
                r_range = (drop_r_lo, drop_r_hi)
            # Sample ring location uniformly within the configured band
            r_ring = rng.uniform(ring_lo, ring_hi)
            theta = rng.uniform(0, 2 * math.pi)
            cx = fcx + r_ring * math.cos(theta)
            cy = fcy + r_ring * math.sin(theta)
            cx = min(max(cx, 0.05), 0.95)
            cy = min(max(cy, 0.05), 0.95)
            rr = rng.uniform(r_range[0], r_range[1])
            rr *= _scale_for_ring(r_ring)
            regs.append((cx, cy, rr))
        all_dropouts = regs
        # Representative largest region for metadata
        if regs:
            cx, cy, rr = max(regs, key=lambda t: t[2])
            dc, dr = (cx, cy), rr

    if have_nv:
        if dc is not None and dr is not None:
            # Anchor NV along the inner side of the largest dropout border
            if rng.random() < max(0.6, nv_anchor_prob):
                theta = rng.uniform(0, 2 * math.pi)
                # Slightly inside the dropout border to align along the ring
                offset = dr * rng.uniform(0.70, 0.95)
                nc = (dc[0] + offset * math.cos(theta), dc[1] + offset * math.sin(theta))
            else:
                r_ring = rng.uniform(ring_lo, ring_hi)
                theta = rng.uniform(0, 2 * math.pi)
                nc = (fcx + r_ring * math.cos(theta), fcy + r_ring * math.sin(theta))
            nr = rng.uniform(nv_r_lo, nv_r_hi)
        else:
            # No dropout present => suppress NV entirely (strict adjacency requirement)
            nc = None
            nr = None
        nc = (min(max(nc[0], 0.05), 0.95), min(max(nc[1], 0.05), 0.95))

    return PathologyParams(dc, dr, all_dropouts, nc, nr, (fcx, fcy))


def _sample_faz_center(gh_cfg: Dict[str, object], rng: random.Random) -> Tuple[float, float]:
    """Sample a FAZ center consistent with Greenhouse jitter config.

    - Supports uniform disk jitter via 'FAZ_center_jitter'
    - Supports Gaussian jitter via 'FAZ_center_jitter_std' (float or [sx, sy])
    - Falls back to base 'FAZ_center' or (0.5, 0.5)
    """
    base = gh_cfg.get('FAZ_center', [0.5, 0.5])
    try:
        cx0, cy0 = float(base[0]), float(base[1])  # type: ignore[index]
    except Exception:
        cx0, cy0 = 0.5, 0.5

    # Prefer uniform disk jitter if provided
    try:
        jr = gh_cfg.get('FAZ_center_jitter')
        if jr is not None:
            rad = float(jr)
            if rad > 0:
                ang = rng.uniform(0.0, 2.0 * math.pi)
                r_u = (rng.random() ** 0.5) * rad
                cx0 += r_u * math.cos(ang)
                cy0 += r_u * math.sin(ang)
                return (max(0.05, min(0.95, cx0)), max(0.05, min(0.95, cy0)))
    except Exception:
        pass

    # Else Gaussian jitter
    try:
        js = gh_cfg.get('FAZ_center_jitter_std')
        if js is not None:
            if isinstance(js, (list, tuple)) and len(js) == 2:
                jx, jy = float(js[0]), float(js[1])
            else:
                jx = jy = float(js)  # type: ignore[arg-type]
            cx0 += rng.normalvariate(0.0, jx)
            cy0 += rng.normalvariate(0.0, jy)
    except Exception:
        pass
    return (max(0.05, min(0.95, cx0)), max(0.05, min(0.95, cy0)))


def _quadrant_label(x: float, y: float) -> str:
    # Superior = y < 0.5; Temporal = x > 0.5 (assume right-eye convention in absence of laterality)
    vert = "superior" if y < 0.5 else "inferior"
    horiz = "temporal" if x > 0.5 else "nasal"
    return f"{vert} {horiz}"


def _zone_label(dist_mm: float) -> str:
    if dist_mm < 0.5:
        return "foveal"
    if dist_mm < 1.5:
        return "parafoveal"
    return "perifoveal"


def _sector_index_and_label(x_model: float, y_model: float) -> tuple[int, str]:
    """Assign one of 4 sectors centered at FAZ (0.5, 0.5).

    The 4 sectors are separated by the horizontal and vertical axes through the
    foveal center and labeled with easy terms:
      0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left.

    Note: Display column (left→right) corresponds to y_model; display row (top→bottom)
    corresponds to x_model (matching tree2img rasterization conventions).
    """
    top = x_model < 0.5
    left = y_model < 0.5
    if top and left:
        return 0, "top-left"
    if top and not left:
        return 1, "top-right"
    if (not top) and (not left):
        return 2, "bottom-right"
    return 3, "bottom-left"


def _coarse_loc_label(x_model: float, y_model: float) -> str:
    """Return easy 4-sector label (top-left/top-right/bottom-right/bottom-left)."""
    return _sector_index_and_label(x_model, y_model)[1]


def _sector_index_and_label_relative(x_model: float, y_model: float, cx: float, cy: float) -> tuple[int, str]:
    """Return sector label relative to an arbitrary center (cx, cy)."""
    top = x_model < cx
    left = y_model < cy
    if top and left:
        return 0, "top-left"
    if top and not left:
        return 1, "top-right"
    if (not top) and (not left):
        return 2, "bottom-right"
    return 3, "bottom-left"


def _coarse_loc_label_relative(x_model: float, y_model: float, cx: float, cy: float) -> str:
    return _sector_index_and_label_relative(x_model, y_model, cx, cy)[1]


def _describe_region(center: Tuple[float, float], radius: float, param_scale_mm: float) -> Dict[str, object]:
    x, y = center
    d_norm = math.hypot(x - 0.5, y - 0.5)
    d_mm = d_norm * param_scale_mm
    r_mm = radius * param_scale_mm
    sector_idx, sector_lbl = _sector_index_and_label(x, y)
    return {
        "center_norm": [round(x, 4), round(y, 4)],
        "radius_norm": round(radius, 4),
        "center_mm": [round(x * param_scale_mm, 3), round(y * param_scale_mm, 3)],
        "radius_mm": round(r_mm, 3),
        "zone": _zone_label(d_mm),
        "simple_loc": _coarse_loc_label(x, y),
        "sector": sector_lbl,
        "sector_index": int(sector_idx),
        "dist_fovea_mm": round(d_mm, 3),
    }


def _rand_dropout_extras(rng: random.Random, patho_cfg: Dict | None, defaults: Dict[str, Tuple[float, float]] | None = None) -> Dict[str, object]:
    """Simplified dropout extras: strength and minimal MA/tortuosity controls.

    - strength or strength_range
    - ma_prob, ma_radius_mm_range, ma_only_near_dropout
    """
    extras: Dict[str, object] = {}
    if patho_cfg is not None:
        # Strength controls
        if patho_cfg.get("dropout_strength") is not None:
            try:
                extras["strength"] = float(patho_cfg["dropout_strength"])  # 0..1
            except Exception:
                pass
        if patho_cfg.get("dropout_strength_range") is not None and isinstance(patho_cfg.get("dropout_strength_range"), (list, tuple)):
            try:
                lo, hi = patho_cfg["dropout_strength_range"]
                extras["strength_range"] = [float(lo), float(hi)]
            except Exception:
                pass
        # Minimal MA controls
        if patho_cfg.get("ma_prob") is not None:
            extras["ma_prob"] = float(patho_cfg["ma_prob"])  # per-segment chance
        if patho_cfg.get("ma_radius_mm_range") is not None and isinstance(patho_cfg.get("ma_radius_mm_range"), (list, tuple)):
            try:
                lo, hi = patho_cfg["ma_radius_mm_range"]
                extras["ma_radius_mm_range"] = [float(lo), float(hi)]
            except Exception:
                pass
        if patho_cfg.get("ma_only_near_dropout") is not None:
            extras["ma_only_near_dropout"] = bool(patho_cfg["ma_only_near_dropout"])  # restrict to ring
        # Optional: expose coupling gain so downstream (per-sample records) keeps it
        if patho_cfg.get("ma_prob_strength_gain") is not None:
            try:
                extras["ma_prob_strength_gain"] = float(patho_cfg["ma_prob_strength_gain"])
            except Exception:
                pass
    # Optional tortuosity controls (re-enabled)
    # Probability of enabling tortuosity per sample
    t_prob = None
    try:
        if patho_cfg is not None:
            if patho_cfg.get("tortuosity_prob_range") is not None:
                lo, hi = patho_cfg.get("tortuosity_prob_range")
                t_prob = rng.uniform(float(lo), float(hi))
            elif patho_cfg.get("tortuosity_prob") is not None:
                t_prob = float(patho_cfg.get("tortuosity_prob"))
    except Exception:
        t_prob = None
    enable_tort = True
    if t_prob is not None:
        enable_tort = (rng.random() < max(0.0, min(1.0, t_prob)))
    # Gain: fixed or range
    t_gain = None
    try:
        if patho_cfg is not None:
            if patho_cfg.get("tortuosity_gain_range") is not None:
                lo, hi = patho_cfg.get("tortuosity_gain_range")
                t_gain = rng.uniform(float(lo), float(hi))
            elif patho_cfg.get("tortuosity_gain") is not None:
                t_gain = float(patho_cfg.get("tortuosity_gain"))
    except Exception:
        t_gain = None
    if t_gain is not None:
        extras["tortuosity_gain"] = float(t_gain if enable_tort else 0.0)
    if patho_cfg is not None and patho_cfg.get("tortuosity_band") is not None:
        tb = patho_cfg.get("tortuosity_band")
        if isinstance(tb, (list, tuple)) and len(tb) == 2:
            try:
                extras["tortuosity_band"] = [float(tb[0]), float(tb[1])]
            except Exception:
                pass

    if ("strength" not in extras) and ("strength_range" not in extras):
        extras["strength_range"] = [0.45, 0.85]
    return extras


def _rand_nv_extras(rng: random.Random, patho_cfg: Dict | None = None) -> Dict[str, object]:
    """Build NV extras from merged pathology config for GreenhouseDropout NVRegion."""
    extras: Dict[str, object] = {}
    if not isinstance(patho_cfg, dict):
        return extras
    def _copy_float(name_base: str, key_out: str | None = None):
        key = name_base
        if key in patho_cfg and patho_cfg[key] is not None:
            try:
                extras[key_out or key.replace('nv_', '')] = float(patho_cfg[key])
            except Exception:
                pass
    def _copy_int(name_base: str, key_out: str | None = None):
        key = name_base
        if key in patho_cfg and patho_cfg[key] is not None:
            try:
                extras[key_out or key.replace('nv_', '')] = int(patho_cfg[key])
            except Exception:
                pass
    def _copy_list(name_base: str, key_out: str | None = None, cast=float):
        key = name_base
        if key in patho_cfg and isinstance(patho_cfg[key], (list, tuple)):
            try:
                extras[key_out or key.replace('nv_', '')] = [cast(v) for v in patho_cfg[key]]
            except Exception:
                pass

    _copy_float('nv_step_len_factor', 'step_len_factor')
    _copy_float('nv_mean_radius_mm', 'mean_radius_mm')
    _copy_float('nv_std_radius_mm', 'std_radius_mm')
    _copy_float('nv_branch_prob', 'branch_prob')
    _copy_float('nv_curl_factor', 'curl_factor')
    _copy_int('nv_edges_per_iter', 'edges_per_iter')
    _copy_float('nv_irregularity_amp', 'irregularity_amp')
    _copy_list('nv_harmonics', 'harmonics', cast=int)
    _copy_list('nv_ellipticity', 'ellipticity', cast=float)
    _copy_float('nv_gradient_alpha', 'gradient_alpha')
    _copy_float('nv_noise_gain', 'noise_gain')
    _copy_float('nv_connect_prob', 'connect_prob')
    _copy_float('nv_connect_radius_norm', 'connect_radius_norm')
    _copy_float('nv_border_band_norm', 'border_band_norm')
    _copy_float('nv_border_bias', 'border_bias')
    _copy_float('nv_outward_bias', 'outward_bias')
    _copy_int('nv_init_spokes', 'init_spokes')
    _copy_float('nv_spoke_jitter', 'spoke_jitter')
    _copy_int('nv_post_iters', 'post_iters')
    _copy_float('nv_min_clearance_vessel_norm', 'min_clearance_vessel_norm')
    _copy_float('nv_min_clearance_nv_norm', 'min_clearance_nv_norm')

    # Pass-through overlay block if provided in compact config
    try:
        if isinstance(patho_cfg.get('nv_overlay'), dict):
            # Shallow copy; downstream will validate keys
            extras['overlay'] = dict(patho_cfg.get('nv_overlay'))
    except Exception:
        pass

    return extras


def _dilate_bool(mask, radius):
    """Binary dilation using a disk structuring element implemented with numpy shifts.

    This avoids external deps; suitable for small radii.
    """
    assert mask.dtype == np.bool_, "mask must be boolean"
    H, W = mask.shape
    pad = int(radius)
    padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
    yy, xx = np.ogrid[-pad : pad + 1, -pad : pad + 1]
    disk = (yy * yy + xx * xx) <= (pad * pad)
    out = np.zeros_like(padded)
    ys, xs = np.where(disk)
    ys = ys - pad
    xs = xs - pad
    for dy, dx in zip(ys, xs):
        out |= np.roll(np.roll(padded, int(dy), axis=0), int(dx), axis=1)
    return out[pad : pad + H, pad : pad + W]


def _render_binary_from_csv(csv_path: str, resolution_hw: Tuple[int, int], mip_axis: int = 2):
    """Rasterize a vessel binary from a graph CSV at given 2D resolution.

    resolution_hw: (H, W) for the output binary; internally converts to (W, H)
    for compatibility with tree2img.rasterize_forest.
    """
    f: list[dict] = []
    with open(csv_path, newline="") as csvfile:
        reader = _csv.DictReader(csvfile)
        for row in reader:
            f.append(row)
    H, W = int(resolution_hw[0]), int(resolution_hw[1])
    img, _ = tree2img.rasterize_forest(f, [W, H], MIP_axis=mip_axis)
    # Binarize: any signal considered vessel
    bin_img = (img.astype(np.float32) > 0).astype(np.uint8)
    return bin_img


def _validate_masks_for_record(rec: Dict, graph_csv: str, dropout_mask_path: Optional[str], neovasc_mask_path: Optional[str]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Compute simple correctness checks for dropout and NV masks.

    - Dropout should reduce vessel density inside vs immediate ring outside.
    - NV should increase vessel density inside vs ring outside.
    """
    d_check = None
    n_check = None
    try:
        # Collect resolution from existing mask (if any); else from rasterization default
        if dropout_mask_path and os.path.isfile(dropout_mask_path):
            dmask = np.array(Image.open(dropout_mask_path).convert("L")) > 127
            H, W = dmask.shape
        elif neovasc_mask_path and os.path.isfile(neovasc_mask_path):
            nmask = np.array(Image.open(neovasc_mask_path).convert("L")) > 127
            H, W = nmask.shape
        else:
            # Default to 1216x1216 if nothing present
            H, W = 1216, 1216

        vessel_bin = _render_binary_from_csv(graph_csv, (H, W), mip_axis=2)

        def _density(mask_bool) -> float:
            area = float(mask_bool.sum())
            if area <= 0:
                return 0.0
            return float((vessel_bin.astype(bool) & mask_bool).sum()) / area

        # Dropout check
        if dropout_mask_path and os.path.isfile(dropout_mask_path):
            dmask = np.array(Image.open(dropout_mask_path).convert("L")) > 127
            area_px = int(dmask.sum())
            ring = _dilate_bool(dmask, max(3, min(H, W) // 40)) & (~dmask)
            ring_px = int(ring.sum())
            dens_in = _density(dmask)
            dens_ring = _density(ring)
            # Valid if inside density sufficiently lower than ring
            valid = (area_px > 100) and (dens_ring > 0.001) and (dens_in <= 0.6 * dens_ring)
            d_check = {
                "valid": bool(valid),
                "area_px": int(area_px),
                "ring_px": int(ring_px),
                "inside_density": float(round(dens_in, 4)),
                "ring_density": float(round(dens_ring, 4)),
            }

        # NV check
        if neovasc_mask_path and os.path.isfile(neovasc_mask_path):
            nmask = np.array(Image.open(neovasc_mask_path).convert("L")) > 127
            area_px = int(nmask.sum())
            ring = _dilate_bool(nmask, max(3, min(H, W) // 40)) & (~nmask)
            ring_px = int(ring.sum())
            dens_in = _density(nmask)
            dens_ring = _density(ring)
            # Valid if inside density higher than ring
            valid = (area_px > 50) and (dens_in >= 1.2 * dens_ring)
            n_check = {
                "valid": bool(valid),
                "area_px": int(area_px),
                "ring_px": int(ring_px),
                "inside_density": float(round(dens_in, 4)),
                "ring_density": float(round(dens_ring, 4)),
            }
    except Exception as e:
        # Leave checks as None if validation fails
        print(f"[warn] Mask validation failed for {rec.get('id')}: {e}")
    return d_check, n_check


def _largest_component_center_from_mask(mask_path: str) -> Optional[Tuple[float, float]]:
    """Return (x_model_norm, y_model_norm) for the largest white component in a binary mask image."""
    try:
        if not (mask_path and os.path.isfile(mask_path)):
            return None
        m = np.array(Image.open(mask_path).convert("L")) >= 128
        H, W = int(m.shape[0]), int(m.shape[1])
        if not m.any():
            return None
        visited = np.zeros_like(m, dtype=np.uint8)
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        best_area = 0
        best_sum_r = 0.0
        best_sum_c = 0.0
        for r in range(H):
            row = m[r]
            if not row.any():
                continue
            for c in np.where(row & (visited[r] == 0))[0]:
                if visited[r, c] or not m[r, c]:
                    continue
                area = 0
                sum_r = 0.0
                sum_c = 0.0
                q = [(r, c)]
                visited[r, c] = 1
                qi = 0
                while qi < len(q):
                    rr, cc = q[qi]
                    qi += 1
                    area += 1
                    sum_r += rr
                    sum_c += cc
                    for dr, dc in neigh:
                        r2 = rr + dr
                        c2 = cc + dc
                        if 0 <= r2 < H and 0 <= c2 < W and not visited[r2, c2] and m[r2, c2]:
                            visited[r2, c2] = 1
                            q.append((r2, c2))
                if area > best_area:
                    best_area = area
                    best_sum_r = sum_r
                    best_sum_c = sum_c
        if best_area <= 0:
            return None
        cr = best_sum_r / float(best_area)
        cc = best_sum_c / float(best_area)
        x_model = (cr + 0.5) / float(H)
        y_model = (cc + 0.5) / float(W)
        x_model = max(0.0, min(1.0, float(x_model)))
        y_model = max(0.0, min(1.0, float(y_model)))
        return (x_model, y_model)
    except Exception:
        return None


def _worker_generate(
    physical_dropout: tuple[float, float, float] | None,
    physical_neovasc: tuple[float, float, float] | None,
    dropout_extras: Optional[Dict[str, object]] = None,
    nv_extras: Optional[Dict[str, object]] = None,
    faz_center_override: Optional[Tuple[float, float]] = None,
) -> Tuple[str, str]:
    assert _WORKER_CONFIG_BYTES is not None and _WORKER_GRAPH_ROOT is not None, "Worker initializer not called"
    # Ensure the generator function is available in the worker (import can be swallowed at module import time)
    global _generate_single_sample
    if _generate_single_sample is None:
        try:
            from generate_synthetic_octa_images import _generate_single_sample as _gss  # type: ignore
            _generate_single_sample = _gss  # type: ignore
        except Exception as e:
            import traceback as _tb
            raise RuntimeError(
                "Worker failed to import _generate_single_sample from generate_synthetic_octa_images."
                " This is required for --stage graphs."
                f"\nCause: {e}\nTraceback:\n{_tb.format_exc()}"
            )
    # Ensure per-sample stochasticity for numpy-based draws (e.g., FAZ jitter)
    try:
        import os as _os, time as _time
        if np is not None:
            seed_val = ((_os.getpid() & 0xFFFF) << 16) ^ (int(_time.time() * 1e6) & 0xFFFFFFFF) ^ random.randint(0, 2**31 - 1)
            np.random.seed(int(seed_val & 0xFFFFFFFF))
    except Exception:
        pass
    cfg = pickle.loads(_WORKER_CONFIG_BYTES)
    out_dir = _generate_single_sample(
        cfg,
        base_out_dir=_WORKER_GRAPH_ROOT,
        physical_dropout=physical_dropout,
        physical_strength=1.0,
        physical_neovasc=physical_neovasc,
        skip_venous=_WORKER_SKIP_VENOUS,
        write_config_per_sample=_WORKER_WRITE_CONFIG,
        save_pathology_masks=_WORKER_SAVE_MASKS,
        mask_threshold=float(_WORKER_MASK_THRESHOLD),
        mask_expand_px=int(_WORKER_MASK_EXPAND_PX),
        mask_vessel_sparing_px=int(_WORKER_MASK_VESSEL_SPARE_PX),
        dropout_cfg_extra=dropout_extras,
        neovasc_cfg_extra=nv_extras,
        faz_center_override=faz_center_override,
        view_shift_norm=(
            tuple(dropout_extras.get("_view_shift_norm"))  # type: ignore[arg-type]
            if isinstance(dropout_extras, dict) and isinstance(dropout_extras.get("_view_shift_norm"), (list, tuple))
            else None
        ),
    )
    base_name = os.path.basename(out_dir.rstrip(os.sep))
    return base_name, out_dir


def main():
    ap = argparse.ArgumentParser("Generate VLM image–text pairs with randomized dropout/NV")
    ap.add_argument("--config_file", type=str, default=os.path.join(SCRIPT_DIR, "configs", "generation_vlm.yml"))
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel processes for graph generation (-1 => all but one)")
    ap.add_argument("--skip_venous", action="store_true", help="Skip venous tree generation for speed")
    ap.add_argument("--no_config_per_sample", action="store_true", help="Do not write per-sample config.yml to reduce I/O")
    ap.add_argument("--save_masks", action="store_true", help="Save dropout/neovasc masks per sample and collect them in dataset folder")

    # Image options (CLI overrides)
    ap.add_argument("--use_gan", action="store_true")
    ap.add_argument("--gan_config", type=str, default=None)
    ap.add_argument("--gan_epoch", type=str, default=None)
    ap.add_argument("--gan_device", type=str, default=None)
    ap.add_argument("--stage", choices=["all", "graphs", "gan", "pairs"], default="all", help="Select pipeline stage: graphs (CPU), gan (GPU), pairs (only write pairs/metadata), or all (default)")
    ap.add_argument("--manifest", type=str, default=None, help="Optional explicit path to records_graphs.jsonl for --stage gan")
    # Mixed-only VQA; no subset selection needed

    args, unknown = ap.parse_known_args()
    
    cfg_path = os.path.abspath(args.config_file)
    if args.stage == "pairs":
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        apply_cli_overrides_from_unknown_args(cfg, unknown)
    else:
        assert os.path.isfile(cfg_path), f"Config file not found: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        apply_cli_overrides_from_unknown_args(cfg, unknown)

    gen = cfg.get("General", {}) or {}
    paths = cfg.get("Paths", {}) or {}
    gan_cfg = cfg.get("GAN", {}) or {}
    raw_patho_cfg = cfg.get("Pathology", {}) or {}
    patho_cfg, profile_general = _resolve_pathology_profile(raw_patho_cfg)
    vqa_cfg = cfg.get("VQA", {}) or {}

    # Robust parsing from config/CLI (tolerate nulls/strings)
    def _to_int(val, dflt):
        try:
            if val is None:
                return int(dflt)
            return int(val)
        except Exception:
            return int(dflt)

    num_samples = _to_int(gen.get("num_samples", 50), 50)
    seed = _to_int(args.seed if args.seed is not None else gen.get("seed", 1234), 1234)
    workers_cfg = gen.get("workers", -1)
    workers = _to_int(args.workers if args.workers is not None else workers_cfg, -1)
    use_gan = bool(args.use_gan or gen.get("use_gan", False))
    save_masks_cfg = gen.get("save_masks", profile_general.get("save_masks", False))
    save_masks = bool(args.save_masks or save_masks_cfg)
    skip_venous_cfg = bool(gen.get("skip_venous", profile_general.get("skip_venous", False)))
    skip_venous = bool(args.skip_venous or skip_venous_cfg)
    write_config_per_sample = not args.no_config_per_sample
    out_dir = os.path.abspath(args.out_dir or paths.get("out_dir") or os.path.join(SCRIPT_DIR, "vlm_dataset"))
    vessel_config = paths.get("vessel_config") or os.path.join(SCRIPT_DIR, "vessel_graph_generation", "configs", "dataset_18_June_2023.yml")
    gan_config = args.gan_config or paths.get("gan_config")
    gan_epoch = args.gan_epoch or str(gan_cfg.get("epoch", "150"))
    # Optional GAN inference batch size override from config
    try:
        gan_batch_size = int(gan_cfg.get("batch_size")) if gan_cfg.get("batch_size") is not None else None
    except Exception:
        gan_batch_size = None
    # Optional: cap GAN DataLoader workers (default to 2 to avoid overload on shared systems)
    try:
        gan_num_workers = int(gan_cfg.get("num_workers")) if gan_cfg.get("num_workers") is not None else 2
    except Exception:
        gan_num_workers = 2
    gan_device = args.gan_device or gen.get("gan_device")

    if (use_gan or args.stage == "gan"):
        if not gan_config:
            raise SystemExit("Please provide --gan_config or set Paths.gan_config when enabling GAN rendering (GAN weights/config are not bundled).")
        if not os.path.isfile(gan_config):
            raise SystemExit(f"GAN config not found: {gan_config}")

    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # If stage == gan, perform only GAN/image + metadata creation using existing graphs.
    if args.stage == "gan":
        # Load records from manifest created in graphs stage
        records = _load_records_manifest(out_dir, args.manifest)
        graph_root = os.path.join(out_dir, "vessel_graphs")
        # Run GAN inference regardless of use_gan flag in manifest time
        gan_workers = max(1, int(gan_num_workers))
        # Ensure log progress even if wrapper lacks the new argument (fallback via env var)
        try:
            os.environ.setdefault("LOG_PROGRESS", "1")
            _call_gan_inference(
                gan_config=gan_config,
                graph_root=graph_root,
                image_out_dir=images_dir,
                epoch=gan_epoch,
                device=gan_device,
                num_workers=gan_workers,
                batch_size=gan_batch_size,
                log_progress=True,
            )
        except TypeError:
            # Older _call_gan_inference without log_progress support
            os.environ.setdefault("LOG_PROGRESS", "1")
            _call_gan_inference(
                gan_config=gan_config,
                graph_root=graph_root,
                image_out_dir=images_dir,
                epoch=gan_epoch,
                device=gan_device,
                num_workers=gan_workers,
                batch_size=gan_batch_size,
            )
        for rec in records:
            base = rec["id"]
            resolved = _resolve_gan_image(images_dir, base)
            # Keep GAN path for reference; dataset will use vessel-map overlay
            rec["image_gan"] = resolved or os.path.join("images", f"generator_{base}.png")

        # Use NV-only overlay for dataset image when available (standardize to M_<id>.png)
        # ALWAYS refresh from graph_dir to ensure cropping is applied
        for rec in records:
            try:
                gdir = rec.get("graph_dir", "")
                # Prefer solid white NV-only overlay; DO NOT use pathology_vis (has annotations)
                white_src = os.path.join(gdir, "pathology_overlay_white.png")
                chosen = white_src if os.path.isfile(white_src) else None
                if not chosen:
                    # If overlay is not present, we will handle fallback below
                    # Clear any existing image ref to trigger fallback loop
                    rec.pop("image", None)
                    continue
                    
                dst_rel = os.path.join("images", f"M_{rec['id']}.png")
                dst_abs = os.path.join(out_dir, dst_rel)
                
                # Auto-crop black borders to square before saving (for FAZ shift cases)
                try:
                    from PIL import Image
                    import numpy as np
                    img = Image.open(chosen)
                    arr = np.array(img)
                    
                    # Find content boundaries (non-black regions)
                    if len(arr.shape) == 3:
                        # RGB/RGBA - check if all channels are 0
                        non_black = np.any(arr > 0, axis=-1)
                    else:
                        # Grayscale
                        non_black = arr > 0
                    
                    # Find bounding box of non-black content
                    rows = np.any(non_black, axis=1)
                    cols = np.any(non_black, axis=0)
                    
                    if rows.any() and cols.any():
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
                        img_cropped.save(dst_abs)
                    else:
                        # No content found, save as-is
                        shutil.copy2(chosen, dst_abs)
                except Exception as e:
                    # Fallback to direct copy if cropping fails
                    print(f"[warn] Failed to crop {chosen}: {e}")
                    shutil.copy2(chosen, dst_abs)
                    
                rec["image"] = dst_rel
            except Exception:
                continue

        # Ensure every record has an 'image' key; keep M_<id>.png strictly as vessel map
        for rec in records:
            if not rec.get("image"):
                base = rec.get("id")
                # Prefer vessel map from graph_dir (overlay white > overlay vis > noisy > raw)
                try:
                    gdir = rec.get("graph_dir", "")
                    cand = [
                        os.path.join(gdir, "pathology_overlay_white.png"),
                        os.path.join(gdir, "art_ven_img_gray_panned.png"),
                        os.path.join(gdir, "art_ven_img_gray_noisy.png"),
                        os.path.join(gdir, "art_ven_img_gray.png"),
                    ]
                    chosen = next((p for p in cand if os.path.isfile(p)), None)
                except Exception:
                    chosen = None
                if chosen and base:
                    try:
                        dst_rel = os.path.join("images", f"M_{base}.png")
                        dst_abs = os.path.join(out_dir, dst_rel)
                        
                        # Auto-crop black borders to square before saving
                        try:
                            from PIL import Image
                            import numpy as np
                            img = Image.open(chosen)
                            arr = np.array(img)
                            
                            # Find content boundaries (non-black regions)
                            if len(arr.shape) == 3:
                                non_black = np.any(arr > 0, axis=-1)
                            else:
                                non_black = arr > 0
                            
                            rows = np.any(non_black, axis=1)
                            cols = np.any(non_black, axis=0)
                            
                            if rows.any() and cols.any():
                                y_min, y_max = np.where(rows)[0][[0, -1]]
                                x_min, x_max = np.where(cols)[0][[0, -1]]
                                
                                # Calculate valid content dimensions
                                valid_height = y_max - y_min + 1
                                valid_width = x_max - x_min + 1
                                
                                # Use smaller dimension to create square
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
                                
                                img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                                img_cropped.save(dst_abs)
                            else:
                                shutil.copy2(chosen, dst_abs)
                        except Exception as e:
                            print(f"[warn] Failed to crop {chosen}: {e}")
                            shutil.copy2(chosen, dst_abs)
                            
                        rec["image"] = dst_rel
                        continue
                    except Exception:
                        pass
                # Ignore previously recorded dataset images; always refresh from graph_dir when possible
                # Last resort: point to GAN output, but do NOT copy into M_<id>.png
                img_rel = rec.get("image_gan")
                if not img_rel and base:
                    resolved = _resolve_gan_image(images_dir, str(base))
                    if resolved:
                        img_rel = resolved
                if not img_rel and base:
                    img_rel = os.path.join("images", f"generator_{base}.png")
                if img_rel:
                    rec["image"] = img_rel

        # No view shift panning applied to GAN; dataset image remains growth-centered overlay

        # Create VQA pairs and metadata based on loaded records
        def _normalize_text(s: str) -> str:
            try:
                return (
                    s.replace("\u2014", "-")  # em dash
                     .replace("\u2013", "-")  # en dash
                     .replace("\u2212", "-")  # minus sign
                     .replace("\u00A0", " ")  # nbsp
                     .replace("\u2019", "'")  # right single quote
                     .replace("\u2018", "'")  # left single quote
                     .replace("\u201C", '"')  # left double quote
                     .replace("\u201D", '"')  # right double quote
                )
            except Exception:
                return s

        def _build_conversation(question: str, answer: str) -> List[Dict[str, str]]:
            q = _normalize_text(question)
            a = _normalize_text(answer)
            return [
                {"from": "human", "value": "<image>\n" + q},
                {"from": "gpt", "value": a},
            ]

        def conv_variants(rec: Dict) -> List[List[Dict[str, str]]]:
            """Return a list of conversations. One per question style/variant.

            Ensures exactly one <image> token per conversation to satisfy loaders
            that validate image-token counts.
            """
            out: List[List[Dict[str, str]]] = []
            q = "What do you see in this OCTA image"
            a = build_mixed_paragraph(rec, out_dir)
            out.append(_build_conversation(q, a))
            return out

        meta_path = os.path.join(out_dir, "metadata.jsonl")
        pairs_path = os.path.join(out_dir, "pairs.jsonl")
        with open(meta_path, "w") as mf, open(pairs_path, "w") as pf:
            for rec in records:
                # Robustly resolve image path with fallbacks if overlay missing
                _base_id = rec["id"]
                _image_rel = (
                    rec.get("image")
                    or rec.get("image_gan")
                    or rec.get("gray_image")
                    or _resolve_gan_image(images_dir, _base_id)
                    or os.path.join("images", f"generator_{_base_id}.png")
                )
                # Determine relative path to per-sample pathology JSON/YAML for traceability
                djson_rel = rec.get("dropout_ma_json")
                if not djson_rel:
                    try:
                        gdir = rec.get("graph_dir", "")
                        cand = os.path.join(gdir, "dropout_ma.json")
                        if os.path.isfile(cand):
                            djson_rel = os.path.relpath(cand, out_dir)
                    except Exception:
                        djson_rel = None
                pyml_rel = None
                try:
                    gdir = rec.get("graph_dir", "")
                    candy = os.path.join(gdir, "pathology.yml")
                    if os.path.isfile(candy):
                        pyml_rel = os.path.relpath(candy, out_dir)
                except Exception:
                    pyml_rel = None
                mrec = {
                    "id": _base_id,
                    "image": _image_rel,
                    "graph_dir": rec.get("graph_dir"),
                    "dropout": rec.get("dropout"),
                    "neovasc": rec.get("neovasc"),
                    "nv_adjacent_to_dropout": rec.get("nv_adjacent_to_dropout"),
                    "dropout_ma_json": djson_rel,
                    "pathology_yml": pyml_rel,
                    "view_shift_norm": rec.get("view_shift_norm"),
                }
                # Embed structured dropout/FAZ/MA summary, preferring in-memory fields; fallback to per-sample JSON
                try:
                    if isinstance(rec.get("dropouts"), list):
                        mrec["dropouts"] = rec["dropouts"]
                    if isinstance(rec.get("microaneurysms"), list):
                        mrec["microaneurysms"] = rec["microaneurysms"]
                    if isinstance(rec.get("FAZ"), dict):
                        mrec["FAZ"] = rec["FAZ"]
                    if ("dropouts" not in mrec) or ("FAZ" not in mrec):
                        jsrc = os.path.join(rec.get("graph_dir", ""), "dropout_ma.json")
                        if os.path.isfile(jsrc):
                            with open(jsrc, "r") as jf:
                                ddata = json.load(jf)
                            if isinstance(ddata, dict):
                                if "dropouts" not in mrec and isinstance(ddata.get("dropouts"), list):
                                    mrec["dropouts"] = ddata["dropouts"]
                                if "microaneurysms" not in mrec and isinstance(ddata.get("microaneurysms"), list):
                                    mrec["microaneurysms"] = ddata["microaneurysms"]
                                if "FAZ" not in mrec and isinstance(ddata.get("FAZ"), dict):
                                    mrec["FAZ"] = ddata["FAZ"]
                except Exception:
                    pass
                mf.write(json.dumps(mrec) + "\n")

                # Build conversations using metadata-enriched record
                convs = conv_variants(mrec)
                # Write one entry per conversation variant; suffix id for uniqueness
                for i, conv in enumerate(convs):
                    pf.write(json.dumps({
                        "id": f"{rec['id']}_q{i+1}",
                        "image": rec["image"],
                        "conversations": conv,
                    }) + "\n")

        print(f"Stage 'gan' complete. Images: {images_dir}")
        print(f"Metadata: {meta_path}")
        print(f"VQA pairs: {pairs_path}")
        return

    # If stage == pairs, only (re)build metadata.jsonl and pairs.jsonl from existing records/images
    if args.stage == "pairs":
        records = _load_records_manifest(out_dir, args.manifest)
        images_dir = os.path.join(out_dir, "images")

        # FIRST: Refresh M_ images from graph_dir with auto-cropping for FAZ-shifted cases
        print(f"[pairs] Refreshing M_ images from graph_dir with auto-crop...")
        for rec in records:
            try:
                gdir = rec.get("graph_dir", "")
                if not gdir:
                    continue
                cid = rec.get("id")
                # Prefer pathology_overlay_white.png (has FAZ shifts applied)
                white_src = os.path.join(gdir, "pathology_overlay_white.png")
                if not os.path.isfile(white_src):
                    continue
                    
                dst_rel = os.path.join("images", f"M_{cid}.png")
                dst_abs = os.path.join(out_dir, dst_rel)
                
                # Auto-crop black borders to square before saving
                try:
                    from PIL import Image
                    import numpy as np
                    img = Image.open(white_src)
                    arr = np.array(img)
                    
                    # Find content boundaries (non-black regions)
                    if len(arr.shape) == 3:
                        non_black = np.any(arr > 0, axis=-1)
                    else:
                        non_black = arr > 0
                    
                    rows = np.any(non_black, axis=1)
                    cols = np.any(non_black, axis=0)
                    
                    if rows.any() and cols.any():
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        # Calculate valid content dimensions
                        valid_height = y_max - y_min + 1
                        valid_width = x_max - x_min + 1
                        
                        # Use smaller dimension to create square
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
                        
                        img_cropped = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
                        img_cropped.save(dst_abs)
                        print(f"  Cropped {cid}: {arr.shape} → {img_cropped.size}")
                    else:
                        shutil.copy2(white_src, dst_abs)
                except Exception as e:
                    print(f"[warn] Failed to crop {white_src}: {e}")
                    shutil.copy2(white_src, dst_abs)
            except Exception as e:
                print(f"[warn] Failed to process {rec.get('id')}: {e}")
                continue

        # THEN: Ensure each record has an image path; prefer conventional naming
        for rec in records:
            cid = rec.get("id")
            # Prefer M_<id>.png (graphs stage output, now cropped)
            m_path = os.path.join("images", f"M_{cid}.png")
            if os.path.isfile(os.path.join(out_dir, m_path)):
                rec["image"] = m_path
                continue
            # Fallback: existing image path in record (if valid)
            if rec.get("image") and os.path.isfile(os.path.join(out_dir, rec["image"])):
                continue
            # Next: grayscale vessel map
            if rec.get("gray_image") and os.path.isfile(os.path.join(out_dir, rec["gray_image"])):
                rec["image"] = rec["gray_image"]
                continue
            # Next: conventional names
            cand = [
                os.path.join("images", f"generator_{cid}.png"),
                os.path.join("images", f"M_{cid}.png"),
                os.path.join("images", f"{cid}.png"),
            ]
            for c in cand:
                if os.path.isfile(os.path.join(out_dir, c)):
                    rec["image"] = c
                    break
        # Backfill structured pathology directly from per-sample graph_dir JSON; do not create dataset-level masks
        for rec in records:
            # Structured pathology + compute representative dropout/NV and adjacency when missing
            try:
                gdir = rec.get("graph_dir", "")
                jsrc = os.path.join(gdir, "dropout_ma.json")
                if os.path.isfile(jsrc):
                    with open(jsrc, "r") as jf:
                        ddata = json.load(jf)
                    if isinstance(ddata, dict):
                        # Embed fine-grained lists used by CoT
                        d_list = ddata.get("dropouts") if isinstance(ddata.get("dropouts"), list) else None
                        ma_list = ddata.get("microaneurysms") if isinstance(ddata.get("microaneurysms"), list) else None
                        faz_dict = ddata.get("FAZ") if isinstance(ddata.get("FAZ"), dict) else None
                        nv_list = None
                        # Support both JSON keys for NV
                        if isinstance(ddata.get("neovascularization"), list):
                            nv_list = ddata.get("neovascularization")
                        elif isinstance(ddata.get("nv_regions"), list):
                            nv_list = ddata.get("nv_regions")

                        if d_list is not None:
                            rec["dropouts"] = d_list
                        if ma_list is not None:
                            rec["microaneurysms"] = ma_list
                        if faz_dict is not None:
                            rec["FAZ"] = faz_dict

                        # Make JSON path visible to downstream CoT builder
                        try:
                            rec["dropout_ma_json"] = os.path.relpath(jsrc, out_dir)
                        except Exception:
                            pass

                        # If record-level summaries are missing, derive them from per-site lists
                        # 1) Representative largest dropout (by radius)
                        if (rec.get("dropout") is None) and isinstance(d_list, list) and d_list:
                            try:
                                i0 = max(range(len(d_list)), key=lambda i: float(d_list[i].get("radius", 0.0)))
                                d0 = d_list[i0]
                                c0 = d0.get("center") or d0.get("center_norm") or [None, None]
                                r0 = d0.get("radius")
                                if c0 and r0 is not None and len(c0) == 2:
                                    cx, cy = float(c0[0]), float(c0[1])
                                    rr = float(r0)
                                    # param_scale_mm is unknown in pairs-only; keep normalized description
                                    rec["dropout"] = {
                                        "center_norm": [round(cx, 4), round(cy, 4)],
                                        "radius_norm": round(rr, 4),
                                    }
                                
                            except Exception:
                                pass
                        # 2) Representative NV site (first or largest if radius present)
                        if (rec.get("neovasc") is None) and isinstance(nv_list, list) and nv_list:
                            try:
                                def _nv_radius(n: dict) -> float:
                                    try:
                                        return float(n.get("radius", 0.0))
                                    except Exception:
                                        return 0.0
                                j = max(range(len(nv_list)), key=lambda i: _nv_radius(nv_list[i])) if nv_list else 0
                                nv0 = nv_list[j]
                                c1 = nv0.get("center") or nv0.get("center_norm") or [None, None]
                                r1 = nv0.get("radius")
                                if c1 and len(c1) == 2:
                                    nx, ny = float(c1[0]), float(c1[1])
                                    out = {"center_norm": [round(nx, 4), round(ny, 4)]}
                                    if r1 is not None:
                                        try:
                                            out["radius_norm"] = round(float(r1), 4)
                                        except Exception:
                                            pass
                                    rec["neovasc"] = out
                            except Exception:
                                pass
                        # 3) NV adjacency to dropout border (boolean) when both are present
                        if rec.get("nv_adjacent_to_dropout") is None:
                            try:
                                d0 = rec.get("dropout")
                                n0 = rec.get("neovasc")
                                if isinstance(d0, dict) and isinstance(n0, dict):
                                    dc = d0.get("center_norm")
                                    dr = d0.get("radius_norm")
                                    nc = n0.get("center_norm")
                                    if isinstance(dc, (list, tuple)) and isinstance(nc, (list, tuple)) and dr is not None:
                                        dx = float(nc[0]) - float(dc[0])
                                        dy = float(nc[1]) - float(dc[1])
                                        gap = math.hypot(dx, dy) - float(dr)
                                        rec["nv_adjacent_to_dropout"] = bool(abs(gap) <= 0.08)
                            except Exception:
                                pass
            except Exception:
                pass
            # Do not modify rec['image'] here; image resolution is handled above
            # via _resolve_gan_image or existing paths. Avoid overriding it based
            # on mask presence.

        # Before building conversations, prefer dropout/pathology info from existing metadata.jsonl
        meta_by_id: Dict[str, Dict[str, object]] = {}
        try:
            meta_path0 = os.path.join(out_dir, "metadata.jsonl")
            if os.path.isfile(meta_path0):
                with open(meta_path0, "r") as mf0:
                    for line in mf0:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict) and row.get("id"):
                                meta_by_id[str(row["id"])] = row
                        except Exception:
                            continue
            # Merge selected fields from metadata into records (source of truth)
            if meta_by_id:
                keys = ("dropouts", "microaneurysms", "FAZ", "dropout", "neovasc", "nv_adjacent_to_dropout")
                for rec in records:
                    m = meta_by_id.get(rec.get("id"))
                    if not isinstance(m, dict):
                        continue
                    for k in keys:
                        if k in m and m[k] is not None:
                            rec[k] = m[k]
        except Exception:
            pass

        # Conversations: mixed-style only

        def _faz_loc_phrase_from_xy(fx: float, fy: float) -> str:
            try:
                dx = float(fx) - 0.5
                dy = float(fy) - 0.5
                r = math.hypot(dx, dy)
                q = _coarse_loc_label(float(fx), float(fy))
                if r < 0.02:
                    return "centered"
                if r < 0.05:
                    return f"slightly shifted toward the {q} quadrant"
                if r < 0.10:
                    return f"displaced toward the {q} quadrant"
                return f"eccentrically positioned toward the {q} quadrant"
            except Exception:
                return "near the center"

        def _resolve_faz_center(rec: Dict) -> Optional[Tuple[float, float]]:
            # Prefer per-sample Greenhouse config.yml in graph_dir (source of truth)
            try:
                gdir = rec.get("graph_dir")
                if gdir:
                    cpath = os.path.join(gdir, "config.yml")
                    if os.path.isfile(cpath):
                        with open(cpath, "r") as yf:
                            cfg = yaml.safe_load(yf)
                        if isinstance(cfg, dict):
                            gh = cfg.get("Greenhouse") or {}
                            faz = gh.get("FAZ_center")
                            if isinstance(faz, (list, tuple)) and len(faz) == 2:
                                fx, fy = faz[0], faz[1]
                                return float(fx), float(fy)
            except Exception:
                pass
            try:
                faz = rec.get("FAZ")
                if isinstance(faz, dict) and isinstance(faz.get("center_norm"), (list, tuple)):
                    fx, fy = faz.get("center_norm")
                    return float(fx), float(fy)
            except Exception:
                pass
            try:
                djson_rel = rec.get("dropout_ma_json")
                if djson_rel:
                    path = os.path.join(out_dir, djson_rel)
                    if os.path.isfile(path):
                        with open(path, "r") as f:
                            data = json.load(f)
                        faz = data.get("FAZ") if isinstance(data, dict) else None
                        if isinstance(faz, dict) and isinstance(faz.get("center_norm"), (list, tuple)):
                            fx, fy = faz.get("center_norm")
                            return float(fx), float(fy)
            except Exception:
                pass
            try:
                gdir = rec.get("graph_dir")
                if gdir:
                    path = os.path.join(gdir, "dropout_ma.json")
                    if os.path.isfile(path):
                        with open(path, "r") as f:
                            data = json.load(f)
                        faz = data.get("FAZ") if isinstance(data, dict) else None
                        if isinstance(faz, dict) and isinstance(faz.get("center_norm"), (list, tuple)):
                            fx, fy = faz.get("center_norm")
                            return float(fx), float(fy)
            except Exception:
                pass
            return None

        def _normalize_text(s: str) -> str:
            try:
                return (
                    s.replace("\u2014", "-")
                     .replace("\u2013", "-")
                     .replace("\u2212", "-")
                     .replace("\u00A0", " ")
                     .replace("\u2019", "'")
                     .replace("\u2018", "'")
                     .replace("\u201C", '"')
                     .replace("\u201D", '"')
                )
            except Exception:
                return s

        def _build_conversation(question: str, answer: str) -> List[Dict[str, str]]:
            q = _normalize_text(question)
            a = _normalize_text(answer)
            return [
                {"from": "human", "value": "<image>\n" + q},
                {"from": "gpt", "value": a},
            ]

        def conv_variants(rec: Dict) -> List[List[Dict[str, str]]]:
            """Return only the mixed-style VQA conversation."""
            q = "What do you see in this OCTA image"
            a = build_mixed_paragraph(rec, out_dir)
            return [_build_conversation(q, a)]

        meta_path = os.path.join(out_dir, "metadata.jsonl")
        pairs_path = os.path.join(out_dir, "pairs.jsonl")
        with open(meta_path, "w") as mf, open(pairs_path, "w") as pf:
            for rec in records:
                # Include dataset-local paths to per-sample pathology artifacts for downstream CoT (YAML/JSON)
                try:
                    gdir = rec.get("graph_dir") or ""
                    djson_rel = None
                    pyml_rel = None
                    cand_json = os.path.join(gdir, "dropout_ma.json")
                    cand_yml = os.path.join(gdir, "pathology.yml")
                    if os.path.isfile(cand_json):
                        djson_rel = os.path.relpath(cand_json, out_dir)
                    if os.path.isfile(cand_yml):
                        pyml_rel = os.path.relpath(cand_yml, out_dir)
                except Exception:
                    djson_rel = None
                    pyml_rel = None

                mrec = {
                    "id": rec["id"],
                    "image": rec["image"],
                    "graph_dir": rec.get("graph_dir"),
                    "dropout": rec.get("dropout"),
                    "neovasc": rec.get("neovasc"),
                    "nv_adjacent_to_dropout": rec.get("nv_adjacent_to_dropout"),
                    # Provide pointers so vqa_mixed can load Tortuosity/summary
                    "dropout_ma_json": djson_rel,
                    "pathology_yml": pyml_rel,
                    # Preserve view shift if present (used in CoT FAZ phrasing)
                    "view_shift_norm": rec.get("view_shift_norm"),
                }
                # Embed compact dropout/MA/FAZ summary for easier consumption
                try:
                    if isinstance(rec.get("dropouts"), list):
                        mrec["dropouts"] = rec["dropouts"]
                    if isinstance(rec.get("microaneurysms"), list):
                        mrec["microaneurysms"] = rec["microaneurysms"]
                    if isinstance(rec.get("FAZ"), dict):
                        mrec["FAZ"] = rec["FAZ"]
                    if ("dropouts" not in mrec) or ("FAZ" not in mrec):
                        jsrc = os.path.join(rec.get("graph_dir", ""), "dropout_ma.json")
                        if os.path.isfile(jsrc):
                            with open(jsrc, "r") as jf:
                                ddata = json.load(jf)
                            if isinstance(ddata, dict):
                                if "dropouts" not in mrec and isinstance(ddata.get("dropouts"), list):
                                    mrec["dropouts"] = ddata["dropouts"]
                                if "microaneurysms" not in mrec and isinstance(ddata.get("microaneurysms"), list):
                                    mrec["microaneurysms"] = ddata["microaneurysms"]
                                if "FAZ" not in mrec and isinstance(ddata.get("FAZ"), dict):
                                    mrec["FAZ"] = ddata["FAZ"]
                except Exception:
                    pass
                mf.write(json.dumps(mrec) + "\n")

                # Build conversations using metadata-enriched record
                # Use the richer record (includes extras/paths) so CoT can use YAML/JSON fallbacks
                convs = conv_variants(rec | mrec if hasattr(dict, "__or__") else {**rec, **mrec})
                for i, conv in enumerate(convs):
                    pf.write(json.dumps({
                        "id": f"{rec['id']}_q{i+1}",
                        "image": rec["image"],
                        "conversations": conv,
                    }) + "\n")

        print(f"Stage 'pairs' complete. Images: {images_dir}")
        print(f"Metadata: {meta_path}")
        print(f"VQA pairs: {pairs_path}")
        return

    # Load config to obtain mm scaling
    if _read_config is None:
        raise RuntimeError("read_config unavailable (missing deps). Run non-'pairs' stages in a full environment.")
    base_cfg = _read_config(vessel_config)
    # Optional FAZ controls from config file (normalize into Greenhouse config)
    try:
        faz_cfg = cfg.get("FAZ", {}) or {}
        if faz_cfg:
            gh = base_cfg.setdefault("Greenhouse", {})
            # Always keep growth FAZ centered (disable jitter); still allow radius bound
            gh["FAZ_center_jitter"] = 0.0
            gh["FAZ_center_jitter_std"] = [0.0, 0.0]
            if "radius_bound" in faz_cfg and isinstance(faz_cfg["radius_bound"], (list, tuple)) and len(faz_cfg["radius_bound"]) == 2:
                gh["FAZ_radius_bound"] = [float(faz_cfg["radius_bound"][0]), float(faz_cfg["radius_bound"][1])]
    except Exception as _e:
        print(f"[warn] Failed to apply FAZ overrides: {_e}")
    # Optional Render controls (do not affect growth; only rasterization)
    try:
        render_cfg = cfg.get("Render", {}) or {}
        if render_cfg:
            base_cfg["Render"] = dict(render_cfg)
    except Exception as _e:
        print(f"[warn] Failed to apply Render overrides: {_e}")
    param_scale_mm = float(base_cfg["Greenhouse"].get("param_scale", 3.0))

    # Note: removed growth scaling shortcuts; generation uses full configured iterations.

    # Optional global sparsity knob from tortuosity severity (detached)
    def _clamp01(x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0
    tsev_raw = raw_patho_cfg.get("tortuosity_severity") if isinstance(raw_patho_cfg, dict) else None
    if tsev_raw is None:
        tsev_raw = raw_patho_cfg.get("remodeling_severity") if isinstance(raw_patho_cfg, dict) else None
    tsev = _clamp01(tsev_raw) if tsev_raw is not None else None
    if tsev is not None and tsev > 0.0:
        try:
            gh = base_cfg.get("Greenhouse", {})
            # Scale base radius down as tortuosity increases
            r0 = float(gh.get("r", 0.0025))
            r_scale = (1.0 - 0.30 * tsev)  # 1.0 -> 0.70
            gh["r"] = float(max(1e-4, r0 * r_scale))
            # Increase spacing, reduce sinks and iterations to produce a sparser map
            for mode in gh.get("modes", []) or []:
                try:
                    mode["N"] = int(max(100, round(float(mode.get("N", 1000)) * (1.0 - 0.45 * tsev))))
                    mode["I"] = int(max(10, round(float(mode.get("I", 100)) * (1.0 - 0.15 * tsev))))
                    mode["eps_s"] = float(mode.get("eps_s", 0.1)) * (1.0 + 0.25 * tsev)
                    mode["eps_n"] = float(mode.get("eps_n", 0.1)) * (1.0 + 0.20 * tsev)
                    mode["eps_k"] = float(mode.get("eps_k", 0.1)) * (1.0 + 0.10 * tsev)
                except Exception:
                    continue
            base_cfg["Greenhouse"] = gh
        except Exception:
            pass

    # Generate graphs with randomized pathology (stage: graphs or all)
    base_cfg.setdefault('output', {})
    graph_root = os.path.join(out_dir, "vessel_graphs")
    base_cfg['output']['directory'] = os.path.abspath(graph_root)
    base_cfg['output']['save_trees'] = True
    # Always save grayscale MIP in graphs stage so vessel maps are available on disk
    if args.stage == "graphs":
        base_cfg['output']['save_2D_image'] = True
    else:
        base_cfg['output']['save_2D_image'] = (not use_gan)  # save grayscale when not using GAN
    base_cfg['output']['save_3D_volumes'] = None
    base_cfg['output']['save_stats'] = False  # speed up

    config_bytes = pickle.dumps(base_cfg, protocol=pickle.HIGHEST_PROTOCOL)

    records: List[Dict] = []
    # Mask rendering controls (optional) from Pathology
    try:
        mask_threshold = float(patho_cfg.get("mask_threshold", 0.5)) if patho_cfg else 0.5
    except Exception:
        mask_threshold = 0.5
    try:
        mask_expand_px = int(patho_cfg.get("mask_expand_px", 0)) if patho_cfg else 0
    except Exception:
        mask_expand_px = 0
    try:
        mask_vessel_spare_px = int(patho_cfg.get("mask_vessel_sparing_px", 0)) if patho_cfg else 0
    except Exception:
        mask_vessel_spare_px = 0
    job_params: List[Tuple[PathologyParams, Dict[str, object], Dict[str, object]]] = []
    gh_cfg = base_cfg.get("Greenhouse", {}) or {}
    # Use FAZ center directly from config to avoid influencing healthy growth
    try:
        faz_cfg_center = gh_cfg.get('FAZ_center') if isinstance(gh_cfg, dict) else None
        if isinstance(faz_cfg_center, (list, tuple)) and len(faz_cfg_center) == 2:
            faz_center_for_sampling = (float(faz_cfg_center[0]), float(faz_cfg_center[1]))
        else:
            faz_center_for_sampling = (0.5, 0.5)
    except Exception:
        faz_center_for_sampling = (0.5, 0.5)
    # View shift (crop) derived from FAZ jitter settings — simulate motion without changing growth
    faz_view_jitter_rad = None
    faz_view_jitter_std = None
    faz_view_jitter_limit = None
    try:
        fz = cfg.get("FAZ", {}) or {}
        if fz.get("center_jitter") is not None:
            faz_view_jitter_rad = float(fz.get("center_jitter"))
        if fz.get("center_jitter_std") is not None:
            faz_view_jitter_std = fz.get("center_jitter_std")
        if fz.get("max_center_shift") is not None:
            try:
                faz_view_jitter_limit = float(fz.get("max_center_shift"))
            except Exception:
                faz_view_jitter_limit = None
    except Exception:
        pass

    def _sample_view_shift(rng_obj: random.Random) -> tuple[float, float] | None:
        try:
            sx = sy = None
            if faz_view_jitter_rad is not None and faz_view_jitter_rad > 0.0:
                ang = rng_obj.uniform(0.0, 2.0 * math.pi)
                r_u = math.sqrt(rng_obj.random()) * float(faz_view_jitter_rad)
                sx, sy = (r_u * math.cos(ang), r_u * math.sin(ang))
            elif faz_view_jitter_std is not None:
                if isinstance(faz_view_jitter_std, (list, tuple)) and len(faz_view_jitter_std) == 2:
                    jx, jy = float(faz_view_jitter_std[0]), float(faz_view_jitter_std[1])
                else:
                    jx = jy = float(faz_view_jitter_std)
                sx, sy = (rng_obj.gauss(0.0, jx), rng_obj.gauss(0.0, jy))
            else:
                return None
            # Clamp to max_center_shift if provided
            if faz_view_jitter_limit is not None and faz_view_jitter_limit > 0.0:
                lim = float(faz_view_jitter_limit)
                sx = max(-lim, min(lim, float(sx)))
                sy = max(-lim, min(lim, float(sy)))
            return (float(sx), float(sy))
        except Exception:
            return None

    for _ in range(num_samples):
        # Keep growth FAZ centered; apply view shift as crop later
        params = _sample_dropout_and_nv(param_scale_mm, rng, patho_cfg, faz_center=faz_center_for_sampling)
        d_extras = _rand_dropout_extras(rng, patho_cfg) if params.dropout_center is not None else {}
        nv_extras = _rand_nv_extras(rng, patho_cfg) if params.nv_center is not None else {}
        vs = _sample_view_shift(rng)
        if not isinstance(d_extras, dict):
            d_extras = {}
        if vs is not None:
            d_extras["_view_shift_norm"] = [float(vs[0]), float(vs[1])]
        job_params.append((params, d_extras, nv_extras))

    # Parallel graph generation
    if workers == -1:
        cpus = cpu_count()
        workers = max(1, cpus - 1)
    else:
        workers = max(1, int(workers))

    def _emit_record(p: PathologyParams, d_extras: Dict[str, object], nv_extras: Dict[str, object], out_dir_concrete: str):
        base_name = os.path.basename(out_dir_concrete.rstrip(os.sep))
        rec: Dict[str, object] = {
            "id": base_name,
            "graph_dir": out_dir_concrete,
            "graph_csv": os.path.join(out_dir_concrete, f"{base_name}.csv"),
            "dropout": None,
            "neovasc": None,
            "dropout_extras": d_extras or None,
            "neovasc_extras": nv_extras or None,
        }
        # Carry per-sample view shift for later image panning
        try:
            if isinstance(d_extras, dict) and isinstance(d_extras.get("_view_shift_norm"), (list, tuple)):
                rec["view_shift_norm"] = [float(d_extras["_view_shift_norm"][0]), float(d_extras["_view_shift_norm"][1])]
        except Exception:
            pass
        # Default: assume not adjacent unless determined later
        rec["nv_adjacent_to_dropout"] = None
        if p.dropout_center is not None:
            rec["dropout"] = _describe_region(p.dropout_center, p.dropout_radius, param_scale_mm)
        if p.nv_center is not None:
            rec["neovasc"] = _describe_region(p.nv_center, p.nv_radius, param_scale_mm)
            # Determine adjacency to dropout border when both exist
            adj = None
            if p.dropout_regions:
                gaps = []
                for (cx, cy, rr) in p.dropout_regions:
                    d = math.hypot(p.nv_center[0] - cx, p.nv_center[1] - cy)
                    gaps.append(d - rr)
                if gaps:
                    gmin = min(gaps)
                    adj = bool(abs(gmin) <= 0.08)
            elif p.dropout_center is not None and p.dropout_radius is not None:
                d = math.hypot(p.nv_center[0] - p.dropout_center[0], p.nv_center[1] - p.dropout_center[1])
                gap = d - p.dropout_radius
                adj = bool(abs(gap) <= 0.08)
            rec["nv_adjacent_to_dropout"] = adj
        records.append(rec)

    if workers > 1:
        futures = []
        fut2info: Dict[object, Tuple[PathologyParams, Dict[str, object], Dict[str, object]]] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                config_bytes,
                base_cfg['output']['directory'],
                skip_venous,
                write_config_per_sample,
                save_masks,
                float(mask_threshold),
                int(mask_expand_px),
                int(mask_vessel_spare_px),
            ),
        ) as ex:
            for params, d_extras, nv_extras in job_params:
                if params.dropout_regions:
                    physical_dropout = [(cx, cy, rr) for (cx, cy, rr) in params.dropout_regions]
                else:
                    physical_dropout = None if params.dropout_center is None else (
                        params.dropout_center[0], params.dropout_center[1], params.dropout_radius
                    )
                physical_neovasc = None if params.nv_center is None else (
                    params.nv_center[0], params.nv_center[1], params.nv_radius
                )
                fut = ex.submit(
                    _worker_generate,
                    physical_dropout,
                    physical_neovasc,
                    d_extras,
                    nv_extras,
                    None,
                )
                futures.append(fut)
                fut2info[fut] = (params, d_extras, nv_extras)

            try:
                for fut in as_completed(futures):
                    params, d_extras, nv_extras = fut2info[fut]
                    base_name, out_dir_concrete = fut.result()
                    _emit_record(params, d_extras, nv_extras, out_dir_concrete)
            except KeyboardInterrupt:
                # Cancel remaining futures and attempt a fast shutdown
                for f in futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                raise
    else:
        # Inline generation to allow live preview
        for params, d_extras, nv_extras in job_params:
            if params.dropout_regions:
                physical_dropout = [(cx, cy, rr) for (cx, cy, rr) in params.dropout_regions]
            else:
                physical_dropout = None if params.dropout_center is None else (
                    params.dropout_center[0], params.dropout_center[1], params.dropout_radius
                )
            physical_neovasc = None if params.nv_center is None else (
                params.nv_center[0], params.nv_center[1], params.nv_radius
            )
            cfg_inline = pickle.loads(config_bytes)
            # Extract per-sample view shift (virtual camera pan)
            vs = None
            try:
                if isinstance(d_extras, dict) and isinstance(d_extras.get("_view_shift_norm"), (list, tuple)):
                    vs = (float(d_extras["_view_shift_norm"][0]), float(d_extras["_view_shift_norm"][1]))
            except Exception:
                vs = None

            out_dir_concrete = _generate_single_sample(
                cfg_inline,
                base_out_dir=base_cfg['output']['directory'],
                physical_dropout=physical_dropout,
                physical_strength=1.0,
                physical_neovasc=physical_neovasc,
                skip_venous=skip_venous,
                write_config_per_sample=write_config_per_sample,
                save_pathology_masks=save_masks,
                mask_threshold=float(mask_threshold),
                mask_expand_px=int(mask_expand_px),
                dropout_cfg_extra=d_extras,
                neovasc_cfg_extra=nv_extras,
                faz_center_override=None,
                view_shift_norm=vs,
            )
            _emit_record(params, d_extras, nv_extras, out_dir_concrete)

    # If only generating graphs (stage == graphs), skip image rendering and write manifest then exit.
    if args.stage == "graphs":
        # Embed structured pathology directly from per-sample JSON; do not create dataset-level masks
        for rec in records:
            jsrc = os.path.join(rec["graph_dir"], "dropout_ma.json")
            if os.path.isfile(jsrc):
                try:
                    with open(jsrc, "r") as jf:
                        pdata = json.load(jf)
                    if isinstance(pdata, dict):
                        if isinstance(pdata.get("dropouts"), list):
                            rec["dropouts"] = pdata["dropouts"]
                        if isinstance(pdata.get("microaneurysms"), list):
                            rec["microaneurysms"] = pdata["microaneurysms"]
                        if isinstance(pdata.get("FAZ"), dict):
                            rec["FAZ"] = pdata["FAZ"]
                except Exception as e:
                    print(f"[warn] Failed to read pathology JSON for {rec['id']}: {e}")

        # Collect per-sample images into dataset images folder; prefer NV-only overlay if present (never use vis)
        os.makedirs(images_dir, exist_ok=True)
        copied = 0
        for rec in records:
            raw_src = os.path.join(rec["graph_dir"], "art_ven_img_gray.png")
            noisy_src = os.path.join(rec["graph_dir"], "art_ven_img_gray_noisy.png")
            overlay_white_src = os.path.join(rec["graph_dir"], "pathology_overlay_white.png")
            overlay_vis_src = os.path.join(rec["graph_dir"], "pathology_vis.png")

            dst_raw = os.path.join(images_dir, f"M_{rec['id']}.png")
            chosen_src: Optional[str] = None
            # Always prefer NV-only overlay when available; else prefer noisy, then raw
            if os.path.isfile(overlay_white_src):
                chosen_src = overlay_white_src
            elif os.path.isfile(noisy_src):
                chosen_src = noisy_src
            elif os.path.isfile(raw_src):
                chosen_src = raw_src

            if chosen_src is not None:
                if chosen_src == raw_src:
                    # If a view shift is requested, re-rasterize larger and crop to avoid padding
                    vs = None
                    try:
                        if isinstance(rec.get("view_shift_norm"), (list, tuple)):
                            vs = (float(rec["view_shift_norm"][0]), float(rec["view_shift_norm"][1]))
                        elif isinstance(rec.get("dropout_extras"), dict) and isinstance(rec.get("dropout_extras").get("_view_shift_norm"), (list, tuple)):
                            de = rec.get("dropout_extras")
                            vs = (float(de["_view_shift_norm"][0]), float(de["_view_shift_norm"][1]))
                    except Exception:
                        vs = None
                    if vs is not None:
                        # Only attempt CSV-based re-render when the CSV exists; else silently copy raw
                        if isinstance(rec.get("graph_csv"), str) and os.path.isfile(rec["graph_csv"]):
                            try:
                                # Determine output size from existing raw_src
                                from PIL import Image as _PILImage
                                import csv as _csv
                                import numpy as _np
                                im0 = _PILImage.open(raw_src)
                                W0, H0 = im0.size
                                dx = int(round(vs[0] * W0))
                                dy = int(round(vs[1] * H0))
                                mx, my = abs(dx), abs(dy)
                                Wb, Hb = W0 + 2 * mx, H0 + 2 * my
                                # Load graph CSV and rasterize larger, then crop to square valid region
                                forest_rows: list[dict] = []
                                with open(rec["graph_csv"], newline="") as csvfile:
                                    reader = _csv.DictReader(csvfile)
                                    for row in reader:
                                        forest_rows.append(row)
                                art_mat, _ = tree2img.rasterize_forest(forest_rows, [Wb, Hb], MIP_axis=2)
                                img_big = art_mat.astype(_np.uint8)
                                # Valid rectangle after shift
                                vx0 = mx + max(0, dx)
                                vy0 = my + max(0, dy)
                                valid_w = W0 - abs(dx)
                                valid_h = H0 - abs(dy)
                                side = max(1, min(valid_w, valid_h))
                                # Center square within the valid rectangle
                                x0 = int(vx0 + max(0, (valid_w - side) // 2))
                                y0 = int(vy0 + max(0, (valid_h - side) // 2))
                                x1 = int(x0 + side)
                                y1 = int(y0 + side)
                                x0 = max(0, min(x0, max(0, Wb - side)))
                                y0 = max(0, min(y0, max(0, Hb - side)))
                                x1 = min(Wb, max(x1, side))
                                y1 = min(Hb, max(y1, side))
                                img_crop = img_big[y0:y1, x0:x1]
                                _PILImage.fromarray(img_crop).save(dst_raw)
                                rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                                copied += 1
                            except Exception as e:
                                print(f"[warn] Failed to re-render/crop grayscale for {rec['id']}: {e}")
                                try:
                                    shutil.copy2(chosen_src, dst_raw)
                                    rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                                    copied += 1
                                except Exception as e2:
                                    print(f"[warn] Failed to copy grayscale for {rec['id']}: {e2}")
                        else:
                            # CSV not present — skip re-render and copy the raw vessel map without warning
                            try:
                                shutil.copy2(chosen_src, dst_raw)
                                rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                                copied += 1
                            except Exception as e:
                                print(f"[warn] Failed to copy grayscale for {rec['id']}: {e}")
                    else:
                        try:
                            shutil.copy2(chosen_src, dst_raw)
                            rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                            copied += 1
                        except Exception as e:
                            print(f"[warn] Failed to copy grayscale for {rec['id']}: {e}")
                else:
                    try:
                        shutil.copy2(chosen_src, dst_raw)
                        rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                        rec["gray_image_overlay"] = True
                        copied += 1
                    except Exception as e:
                        print(f"[warn] Failed to copy NV overlay for {rec['id']}: {e}")
                        # Fallback to raw if available
                        if os.path.isfile(raw_src):
                            try:
                                shutil.copy2(raw_src, dst_raw)
                                rec["gray_image"] = os.path.join("images", f"M_{rec['id']}.png")
                                copied += 1
                            except Exception as e2:
                                print(f"[warn] Failed to copy fallback grayscale for {rec['id']}: {e2}")

            if os.path.isfile(noisy_src):
                dst_noisy = os.path.join(images_dir, f"M_{rec['id']}_noisy.png")
                try:
                    shutil.copy2(noisy_src, dst_noisy)
                    rec["gray_image_noisy"] = os.path.join("images", f"M_{rec['id']}_noisy.png")
                except Exception as e:
                    print(f"[warn] Failed to copy noisy grayscale for {rec['id']}: {e}")
        print(f"Copied {copied} vessel maps to {images_dir} (NV overlay applied when available)")

        man_path = _write_records_manifest(out_dir, records)
        print(f"Stage 'graphs' complete. Graphs: {graph_root}")
        print(f"Manifest: {man_path}")
        return

    # Image rendering (stage == all)
    if use_gan:
        # Use multiple workers for dataloading if provided
        gan_workers = max(1, int(gan_num_workers))
        # Ensure log progress even if wrapper lacks the new argument (fallback via env var)
        try:
            os.environ.setdefault("LOG_PROGRESS", "1")
            _call_gan_inference(
                gan_config=gan_config,
                graph_root=graph_root,
                image_out_dir=images_dir,
                epoch=gan_epoch,
                device=gan_device,
                num_workers=gan_workers,
                batch_size=gan_batch_size,
                log_progress=True,
            )
        except TypeError:
            os.environ.setdefault("LOG_PROGRESS", "1")
            _call_gan_inference(
                gan_config=gan_config,
                graph_root=graph_root,
                image_out_dir=images_dir,
                epoch=gan_epoch,
                device=gan_device,
                num_workers=gan_workers,
                batch_size=gan_batch_size,
            )
        # Record GAN image path for reference only
        for rec in records:
            base = rec['id']
            resolved = _resolve_gan_image(images_dir, base)
            rec['image'] = resolved or os.path.join("images", f"generator_{base}.png")
        # For GAN runs we do not overlay; naming stays as generator_<id>.png
    else:
        # Copy per-sample grayscale MIP into images dir using GAN-like naming for consistency
        for rec in records:
            raw_src = os.path.join(rec["graph_dir"], "art_ven_img_gray.png")
            noisy_src = os.path.join(rec["graph_dir"], "art_ven_img_gray_noisy.png")
            chosen_src = noisy_src if os.path.isfile(noisy_src) else raw_src
            dst = os.path.join(images_dir, f"generator_{rec['id']}.png")
            if chosen_src and os.path.isfile(chosen_src):
                shutil.copy2(chosen_src, dst)
            elif os.path.isfile(raw_src):
                shutil.copy2(raw_src, dst)
            rec['image'] = os.path.join("images", f"generator_{rec['id']}.png")

        # Prefer NV-only overlay image for dataset 'image' when available; standardize to M_<id>.png
        for rec in records:
            try:
                gdir = rec.get("graph_dir", "")
                white_src = os.path.join(gdir, "pathology_overlay_white.png")
                chosen = white_src if os.path.isfile(white_src) else None
                if not chosen:
                    continue
                dst_rel = os.path.join("images", f"M_{rec['id']}.png")
                dst_abs = os.path.join(out_dir, dst_rel)
                shutil.copy2(chosen, dst_abs)
                rec["image"] = dst_rel
            except Exception:
                continue

        # Keep non-vis images raw (no colored overlay) only if overlay missing; otherwise M_<id>.png set above.

    # Do not create dataset-level masks; embed pathology info directly when writing metadata

    # Create VQA pairs (mixed style only)

    def _normalize_text(s: str) -> str:
        try:
            return (
                s.replace("\u2014", "-")
                 .replace("\u2013", "-")
                 .replace("\u2212", "-")
                 .replace("\u00A0", " ")
                 .replace("\u2019", "'")
                 .replace("\u2018", "'")
                 .replace("\u201C", '"')
                 .replace("\u201D", '"')
            )
        except Exception:
            return s

    def _build_conversation(question: str, answer: str) -> List[Dict[str, str]]:
        q = _normalize_text(question)
        a = _normalize_text(answer)
        return [
            {"from": "human", "value": "<image>\n" + q},
            {"from": "gpt", "value": a},
        ]

    def conv_variants(rec: Dict) -> List[List[Dict[str, str]]]:
        """Return only the mixed-style VQA conversation."""
        q = "What do you see in this OCTA image"
        a = build_mixed_paragraph(rec, out_dir)
        return [_build_conversation(q, a)]

    # Save metadata and pairs
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    pairs_path = os.path.join(out_dir, "pairs.jsonl")
    with open(meta_path, "w") as mf, open(pairs_path, "w") as pf:
        for rec in records:
            # metadata
            mrec = {
                "id": rec["id"],
                "image": rec["image"],
                "dropout": rec["dropout"],
                "neovasc": rec["neovasc"],
                "nv_adjacent_to_dropout": rec.get("nv_adjacent_to_dropout"),
                "dropout_ma_json": (rec.get("dropout_ma_json") or (
                    os.path.relpath(os.path.join(rec.get("graph_dir", ""), "dropout_ma.json"), out_dir)
                    if (rec.get("graph_dir") and os.path.isfile(os.path.join(rec.get("graph_dir", ""), "dropout_ma.json"))) else None
                )),
                "pathology_yml": (
                    os.path.relpath(os.path.join(rec.get("graph_dir", ""), "pathology.yml"), out_dir)
                    if (rec.get("graph_dir") and os.path.isfile(os.path.join(rec.get("graph_dir", ""), "pathology.yml"))) else None
                ),
                "view_shift_norm": rec.get("view_shift_norm"),
                # masks removed; no paths emitted
            }
            # Embed optional structured dropout/FAZ/MA summary for convenience
            try:
                # Prefer structured info already embedded on the record; else read from per-sample JSON
                if isinstance(rec.get("dropouts"), list):
                    mrec["dropouts"] = rec["dropouts"]
                if isinstance(rec.get("microaneurysms"), list):
                    mrec["microaneurysms"] = rec["microaneurysms"]
                if isinstance(rec.get("FAZ"), dict):
                    mrec["FAZ"] = rec["FAZ"]
                if ("dropouts" not in mrec) or ("FAZ" not in mrec):
                    jsrc = os.path.join(rec.get("graph_dir", ""), "dropout_ma.json")
                    if os.path.isfile(jsrc):
                        with open(jsrc, "r") as jf:
                            ddata = json.load(jf)
                        if isinstance(ddata, dict):
                            if "dropouts" not in mrec and isinstance(ddata.get("dropouts"), list):
                                mrec["dropouts"] = ddata["dropouts"]
                            if "microaneurysms" not in mrec and isinstance(ddata.get("microaneurysms"), list):
                                mrec["microaneurysms"] = ddata["microaneurysms"]
                            if "FAZ" not in mrec and isinstance(ddata.get("FAZ"), dict):
                                mrec["FAZ"] = ddata["FAZ"]
            except Exception:
                pass
            mf.write(json.dumps(mrec) + "\n")

            # pairs: one entry per conversation variant
            convs = conv_variants(rec)
            for i, conv in enumerate(convs):
                pf.write(json.dumps({
                    "id": f"{rec['id']}_q{i+1}",
                    "image": rec["image"],
                    "conversations": conv,
                }) + "\n")

    print(f"Done. Images: {images_dir}")
    print(f"Metadata: {meta_path}")
    print(f"VQA pairs: {pairs_path}")


if __name__ == "__main__":
    main()
