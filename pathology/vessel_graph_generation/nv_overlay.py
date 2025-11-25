from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _noise2(x: float, y: float, kx: float, ky: float, phx: float, phy: float) -> float:
    return 0.5 * (math.sin(kx * x + phx) + math.sin(ky * y + phy))


def generate_tuft_polylines(
    center_rc: Tuple[int, int],
    radius_px: int,
    canvas_hw: Tuple[int, int],
    n_spokes: int = 32,
    steps: int = 64,
    step_px: float = 3.0,
    curl: float = 0.65,
    jitter: float = 0.25,
) -> list[list[Tuple[int, int]]]:
    """Generate radial polylines representing an NV tuft in pixel space.

    - center_rc: (row, col) center in pixels.
    - radius_px: footprint radius in pixels.
    - canvas_hw: (H, W) size of the target image.
    - n_spokes: number of seed directions.
    - steps: segments per spoke.
    - step_px: step length in pixels.
    - curl: 0..1 amount of tangential turning.
    - jitter: 0..1 random perturbation.
    """
    H, W = int(canvas_hw[0]), int(canvas_hw[1])
    r0, c0 = int(center_rc[0]), int(center_rc[1])
    polylines: list[list[Tuple[int, int]]] = []

    # Coherent phases for noise-driven swirl
    kx = 2.0 * math.pi / max(8.0, float(radius_px))
    ky = 2.0 * math.pi / max(10.0, float(radius_px))
    phx = np.random.uniform(0.0, 2.0 * math.pi)
    phy = np.random.uniform(0.0, 2.0 * math.pi)

    for k in range(max(1, int(n_spokes))):
        th0 = 2.0 * math.pi * (k / max(1, int(n_spokes))) + np.random.uniform(-0.15, 0.15)
        # Initialize at exact center with a small nudge
        x = float(c0)
        y = float(r0)
        dx = math.cos(th0)
        dy = math.sin(th0)

        line: list[Tuple[int, int]] = [(int(round(y)), int(round(x)))]
        for s in range(max(1, int(steps))):
            # Local curl using perpendicular blend + smooth noise field
            tx, ty = -dy, dx
            n = _noise2(x, y, kx, ky, phx, phy)
            kcurl = _clamp(curl, 0.0, 1.0)
            kjit = _clamp(jitter, 0.0, 1.0)
            dx = (1.0 - kcurl) * dx + kcurl * (tx * n)
            dy = (1.0 - kcurl) * dy + kcurl * (ty * n)
            # Add mild jitter
            dx += kjit * np.random.uniform(-0.6, 0.6)
            dy += kjit * np.random.uniform(-0.6, 0.6)
            # Normalize
            nrm = math.hypot(dx, dy)
            if nrm < 1e-6:
                # re-seed direction
                ang = np.random.uniform(0.0, 2.0 * math.pi)
                dx, dy = math.cos(ang), math.sin(ang)
                nrm = 1.0
            dx /= nrm
            dy /= nrm
            # Step
            x += float(step_px) * dx
            y += float(step_px) * dy
            # Clamp to footprint disk
            vx, vy = x - float(c0), y - float(r0)
            r = math.hypot(vx, vy)
            if r > float(radius_px):
                if r <= 1e-6:
                    break
                scale = float(radius_px) / r
                x = float(c0) + vx * scale
                y = float(r0) + vy * scale
                # End spoke when reaching boundary
                line.append((int(round(y)), int(round(x))))
                break
            # Clip to canvas
            yy = int(_clamp(round(y), 0, H - 1))
            xx = int(_clamp(round(x), 0, W - 1))
            if (yy, xx) != line[-1]:
                line.append((yy, xx))
        if len(line) >= 2:
            polylines.append(line)
    return polylines


def draw_tuft_overlay(
    draw_obj,  # PIL.ImageDraw.Draw
    center_rc: Tuple[int, int],
    radius_px: int,
    canvas_hw: Tuple[int, int],
    color_rgba: Tuple[int, int, int, int] = (64, 220, 255, 220),
    thickness_px: int = 3,
    n_spokes: int = 36,
    steps: int = 72,
    step_px: float = 3.0,
    curl: float = 0.7,
    jitter: float = 0.25,
    fill_alpha: int = 48,
) -> np.ndarray:
    """Draw a synthetic NV tuft (procedurally simulated) directly onto a PIL overlay.

    Returns a boolean mask (H,W) where NV was drawn, suitable for saving as NV mask.
    """
    H, W = int(canvas_hw[0]), int(canvas_hw[1])

    # Optional light fill of the footprint and a mask that matches stroke thickness
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        Image = None  # type: ignore
        ImageDraw = None  # type: ignore

    # Build mask we will return (thick version via PIL, not just centerline)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask_img = None
    mask_draw = None
    if Image is not None and ImageDraw is not None:
        mask_img = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask_img)

    # Light fill circle (very subtle) to reduce harsh boundary when composited
    try:
        if fill_alpha > 0:
            cy, cx = int(center_rc[0]), int(center_rc[1])
            r = int(max(1, radius_px))
            draw_obj.ellipse([cx - r, cy - r, cx + r, cy + r],
                             fill=(color_rgba[0], color_rgba[1], color_rgba[2], int(fill_alpha)))
            if mask_draw is not None:
                mask_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    except Exception:
        pass

    # Generate line geometry
    polylines = generate_tuft_polylines(
        center_rc, radius_px, (H, W),
        n_spokes=n_spokes, steps=steps, step_px=step_px, curl=curl, jitter=jitter,
    )

    # Taper profile: outer soft halo, mid body, bright core
    base_thick = max(1, int(thickness_px))
    w_outer = max(1, int(round(base_thick * 1.8)))
    w_mid = max(1, int(round(base_thick * 1.0)))
    w_core = max(1, int(round(max(1, base_thick // 2))))
    a_outer = int(round(_clamp(color_rgba[3] * 0.35, 0, 255)))
    a_mid = int(round(_clamp(color_rgba[3] * 0.85, 0, 255)))
    a_core = int(round(_clamp(color_rgba[3] * 1.10, 0, 255)))
    color_outer = (color_rgba[0], color_rgba[1], color_rgba[2], a_outer)
    color_mid = (color_rgba[0], color_rgba[1], color_rgba[2], a_mid)
    color_core = (color_rgba[0], color_rgba[1], color_rgba[2], a_core)

    def _safe_draw_line(dobj, pts, fill, width) -> None:
        try:
            dobj.line([(x[1], x[0]) for x in pts], fill=fill, width=int(max(1, width)))
        except Exception:
            # Break the line into small segments if PIL fails on long polylines
            for i in range(1, len(pts)):
                y0, x0 = pts[i - 1]
                y1, x1 = pts[i]
                try:
                    dobj.line([(x0, y0), (x1, y1)], fill=fill, width=int(max(1, width)))
                except Exception:
                    pass

    # Draw lines and set mask
    for line in polylines:
        # Update mask as a thick line approximating the visible stroke
        if mask_draw is not None:
            _safe_draw_line(mask_draw, line, fill=255, width=w_mid)
        else:
            for (ry, rx) in line:
                if 0 <= ry < H and 0 <= rx < W:
                    mask[ry, rx] = 255

        # Draw multi-pass tapered stroke: halo -> body -> bright core
        try:
            _safe_draw_line(draw_obj, line, fill=color_outer, width=w_outer)
            _safe_draw_line(draw_obj, line, fill=color_mid, width=w_mid)
            _safe_draw_line(draw_obj, line, fill=color_core, width=w_core)
        except Exception:
            # If anything fails, fall back to single pass
            _safe_draw_line(draw_obj, line, fill=color_rgba, width=base_thick)

    if mask_img is not None:
        try:
            mask = np.array(mask_img, dtype=np.uint8)
        except Exception:
            pass

    return mask
