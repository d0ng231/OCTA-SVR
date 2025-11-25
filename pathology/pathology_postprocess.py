#!/usr/bin/env python3
"""Post-process OCTA synthetic dataset outputs to ensure:
- Every sample has a pathology overlay (falls back to art_ven_img_gray if none generated)
- Microaneurysm entries are deduplicated between arterial/venous passes
- Generator images exist for all samples (copy overlay or raw vessel map)

Usage:
  uv run python pathology_postprocess.py --dataset vlm_dataset_OCT_22
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def _dedupe_mas(items: Iterable[Dict]) -> List[Dict]:
    seen: set[Tuple[int, int]] = set()
    out: List[Dict] = []
    for item in items:
        center = item.get("center") or item.get("center_norm")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        try:
            row = int(round(float(center[0]) * 10000))
            col = int(round(float(center[1]) * 10000))
        except Exception:
            continue
        key = (row, col)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _ensure_overlay(sample_dir: Path) -> Path:
    overlay = sample_dir / "pathology_overlay_white.png"
    if overlay.exists():
        return overlay
    fallback = sample_dir / "art_ven_img_gray_panned.png"
    if not fallback.exists():
        fallback = sample_dir / "art_ven_img_gray.png"
    if fallback.exists():
        shutil.copy2(fallback, overlay)
    return overlay


def _update_json(sample_dir: Path) -> None:
    json_path = sample_dir / "dropout_ma.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return
    mas = data.get("microaneurysms")
    if isinstance(mas, list):
        deduped = _dedupe_mas(mas)
        if len(deduped) != len(mas):
            data["microaneurysms"] = deduped
            json_path.write_text(json.dumps(data))


def _update_yaml(sample_dir: Path) -> None:
    yaml_path = sample_dir / "pathology.yml"
    if not yaml_path.exists():
        return
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return
    ma_block = data.get("MA")
    if isinstance(ma_block, dict):
        mas = ma_block.get("instances")
        if isinstance(mas, list):
            deduped = _dedupe_mas(mas)
            if len(deduped) != len(mas):
                ma_block["instances"] = deduped
                ma_block["count"] = len(deduped)
                yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _ensure_generator(sample_id: str, sample_dir: Path, images_dir: Path) -> None:
    overlay = _ensure_overlay(sample_dir)
    target = images_dir / f"generator_{sample_id}.png"
    if target.exists():
        return
    if overlay.exists():
        shutil.copy2(overlay, target)
    else:
        fallback = sample_dir / "art_ven_img_gray.png"
        if fallback.exists():
            shutil.copy2(fallback, target)


def postprocess(dataset_root: Path) -> None:
    graph_root = dataset_root / "vessel_graphs"
    images_dir = dataset_root / "images"
    if not graph_root.is_dir():
        raise SystemExit(f"vessel_graphs folder not found under {dataset_root}")
    images_dir.mkdir(exist_ok=True)
    for sample_dir in sorted(graph_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        _ensure_overlay(sample_dir)
        _update_json(sample_dir)
        _update_yaml(sample_dir)
        _ensure_generator(sample_id, sample_dir, images_dir)


def main() -> None:
    parser = argparse.ArgumentParser("Post-process OCTA synthetic dataset outputs")
    parser.add_argument("--dataset", required=True, help="Path to dataset root (e.g., ./vlm_dataset_OCT_22)")
    args = parser.parse_args()
    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    postprocess(dataset_root)
    print(f"Post-processed dataset at {dataset_root}")


if __name__ == "__main__":
    main()
