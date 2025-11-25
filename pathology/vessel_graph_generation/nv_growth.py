from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def _unit(vx: float, vy: float) -> tuple[float, float]:
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return 0.0, 0.0
    return vx / n, vy / n


def _segments_from_edges(edges: Iterable[Dict[str, Any]]) -> list[tuple[float, float, float, float, float]]:
    """Pack edges to a simple segment list: (x1,y1,x2,y2,r)."""
    segs = []
    for e in edges:
        (x1, y1) = e["node1"]; (x2, y2) = e["node2"]
        rr = float(e.get("radius", 0.012))
        segs.append((float(x1), float(y1), float(x2), float(y2), rr))
    return segs


def _quantize_xy(x: float, y: float, q: float = 1e-4) -> tuple[float, float]:
    return (round(float(x) / q) * q, round(float(y) / q) * q)


def _build_tip_candidates(edges: Iterable[Dict[str, Any]]) -> list[tuple[float, float, float, float, float]]:
    """Find leaf tips and their outward heading.

    Returns a list of (tx, ty, dirx, diry, parent_radius), where (tx,ty) is the tip position
    and (dirx,diry) is the outward direction (from neighbor -> tip).
    """
    deg: Dict[tuple[float, float], int] = {}
    last_edge_for: Dict[tuple[float, float], Dict[str, Any]] = {}

    for e in edges:
        x1, y1 = float(e["node1"][0]), float(e["node1"][1])
        x2, y2 = float(e["node2"][0]), float(e["node2"][1])
        k1 = _quantize_xy(x1, y1)
        k2 = _quantize_xy(x2, y2)
        deg[k1] = deg.get(k1, 0) + 1
        deg[k2] = deg.get(k2, 0) + 1
        # Remember an incident edge to recover tangent and radius
        last_edge_for[k1] = e
        last_edge_for[k2] = e

    tips: list[tuple[float, float, float, float, float]] = []
    for k, d in deg.items():
        if d != 1:
            continue
        e = last_edge_for.get(k)
        if e is None:
            continue
        (x1, y1) = (float(e["node1"][0]), float(e["node1"][1]))
        (x2, y2) = (float(e["node2"][0]), float(e["node2"][1]))
        # Determine which endpoint is the tip
        k1 = _quantize_xy(x1, y1)
        k2 = _quantize_xy(x2, y2)
        if k == k1:
            tx, ty = x1, y1
            nx, ny = x2, y2
        else:
            tx, ty = x2, y2
            nx, ny = x1, y1
        dirx, diry = _unit(tx - nx, ty - ny)  # outward
        parent_r = float(e.get("radius", 0.010))
        tips.append((tx, ty, dirx, diry, parent_r))
    return tips


def grow_nv_branches(
    art_edges: List[Dict[str, Any]],
    greenhouse_obj,
    severity: float = 0.5,
    rng_seed: int | None = None,
    n_regions_hint: int | None = None,
) -> tuple[List[Dict[str, Any]], List[List[Tuple[float, float]]], List[Tuple[float, float, float]]]:
    """Generate neovascular sprouts by branching from existing vessel tips.

    Default behavior prefers tips near dropout borders and grows inward so NV occurs
    within dropout regions. If no dropout exists, falls back to free sprouting from
    central tips (still avoiding FAZ and image borders).

    Inputs
    - art_edges: existing arterial edges list with keys node1, node2 (norm coords), radius (mm)
    - greenhouse_obj: Greenhouse/GreenhouseDropout (used for FAZ avoidance and dropout borders)
    - severity: 0..1 scales number/length/thickness/branching

    Returns
    - new_edges: list of edges to append (same format)
    - polylines_norm: list of grown centerlines in normalized coords
    - host_regions: list of (cx,cy,r) host regions used (dropout(s) or local footprint)
    """
    rng = np.random.default_rng(rng_seed if rng_seed is not None else np.random.SeedSequence().entropy)

    # Extract FAZ to avoid placing NV clearly inside it
    try:
        fx, fy = float(getattr(greenhouse_obj, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(greenhouse_obj, 'FAZ_center', (0.5, 0.5))[1])
        fr = float(getattr(greenhouse_obj, 'FAZ_radius', 0.06))
    except Exception:
        fx, fy, fr = 0.5, 0.5, 0.06

    tips = _build_tip_candidates(art_edges)
    if not tips:
        return [], [], []

    # Try to anchor to dropout borders if available
    try:
        dropouts = list(getattr(greenhouse_obj, 'dropouts', []) or [])
    except Exception:
        dropouts = []

    # Severity → parameters
    sev = float(max(0.0, min(1.0, severity)))
    n_groups = int(round(2 + sev * 10)) if n_regions_hint is None else int(max(1, n_regions_hint))
    n_groups = max(1, min(n_groups, len(tips)))
    steps_main = int(round(40 + 120 * sev))
    steps_side = int(round(14 + 60 * sev))
    step_len = 0.004 + 0.004 * sev         # normalized units (~0.4%–0.8% of field)
    curl = 0.45 + 0.35 * sev               # turning amount
    jitter = 0.20 + 0.25 * sev             # heading noise
    side_prob = 0.25 + 0.45 * sev          # probability to create a side branch
    th_scale = 1.8 + 0.9 * sev             # thickness relative to parent
    r_min_mm = 0.010 + 0.018 * sev         # minimal mm radius for visibility

    # Prefer tips away from FAZ and away from image borders
    def _tip_score(tx: float, ty: float) -> float:
        dfaz = math.hypot(tx - fx, ty - fy)
        border = min(tx, ty, 1.0 - tx, 1.0 - ty)
        return 1.2 * dfaz + 0.8 * border + rng.uniform(0.0, 0.2)

    seeds: list[tuple[float, float, float, float, float, object | None]] = []  # (sx,sy,dx,dy,pr, host)
    host_regions_meta: list[tuple[float, float, float]] = []

    if dropouts:
        # Prefer the largest few dropouts; anchor tips in a thin ring near each border
        # Weight by area (~r0^2) and dropout strength
        def _drop_strength(d) -> float:
            try:
                return float(getattr(d, 'drop_strength', getattr(d, 'strength', 1.0)))
            except Exception:
                return 1.0
        weights = []
        for i in range(len(dropouts)):
            r0i = float(getattr(dropouts[i], 'r0', 0.0))
            si = _drop_strength(dropouts[i])
            weights.append(max(0.0, (r0i * r0i) * (0.5 + 0.5 * si)))
        order = sorted(range(len(dropouts)), key=lambda i: weights[i], reverse=True)
        n_hosts = min(len(order), max(1, int(round(1 + 3 * sev))))
        ring_frac = 0.20  # ring thickness as fraction of r0
        inward_blend = 0.65  # blend between inward vector and tip tangent
        # Normalize weights among chosen hosts
        chosen = order[:n_hosts]
        w_chosen = [weights[i] for i in chosen]
        w_max = max(1e-6, max(w_chosen))
        for i in chosen:
            reg = dropouts[i]
            cx = float(getattr(reg, 'cx', 0.5)); cy = float(getattr(reg, 'cy', 0.5)); r0 = float(getattr(reg, 'r0', 0.12))
            # Collect tips whose distance to center is close to r0 (border ring)
            ring_tips: list[tuple[float, float, float, float, float]] = []
            for (tx, ty, dx, dy, pr) in tips:
                d = math.hypot(tx - cx, ty - cy)
                if r0 <= 1e-6:
                    continue
                if abs(d - r0) <= (ring_frac * r0):
                    # Blend heading toward interior to ensure growth into dropout
                    icx, icy = _unit(cx - tx, cy - ty)
                    gdx = (1.0 - inward_blend) * dx + inward_blend * icx
                    gdy = (1.0 - inward_blend) * dy + inward_blend * icy
                    gdx, gdy = _unit(gdx, gdy)
                    ring_tips.append((tx, ty, gdx, gdy, pr))
            if not ring_tips:
                continue
            # Rank ring tips by how strongly they point inward and are away from borders and FAZ
            def _border_inward_score(t: tuple[float, float, float, float, float]) -> float:
                tx, ty, gdx, gdy, _ = t
                dfaz = math.hypot(tx - fx, ty - fy)
                border = min(tx, ty, 1.0 - tx, 1.0 - ty)
                inward = max(-1.0, min(1.0, gdx * (cx - tx) + gdy * (cy - ty)))
                return 1.0 * dfaz + 0.8 * border + 0.5 * inward
            ring_tips.sort(key=_border_inward_score, reverse=True)

            # Number of sprouts per dropout scales with severity and host weight
            w_norm = weights[i] / w_max
            base = 1.0 + 3.0 * sev
            scale = 0.60 + 0.80 * max(0.0, min(1.0, w_norm))
            n_here = max(1, int(round(base * scale)))
            for k in range(min(n_here, len(ring_tips))):
                sx, sy, sdx, sdy, pr = ring_tips[k]
                seeds.append((sx, sy, sdx, sdy, pr, reg))
            host_regions_meta.append((cx, cy, r0))

        # If not enough seeds collected (e.g., no tips on borders), fall back to best central tips
        if len(seeds) < n_groups:
            tips_sorted = sorted(tips, key=lambda t: _tip_score(t[0], t[1]), reverse=True)
            for (sx, sy, sdx, sdy, pr) in tips_sorted:
                if len(seeds) >= n_groups:
                    break
                seeds.append((sx, sy, sdx, sdy, pr, None))
    else:
        # No dropout: choose central tips
        tips_sorted = sorted(tips, key=lambda t: _tip_score(t[0], t[1]), reverse=True)
        for (sx, sy, sdx, sdy, pr) in tips_sorted[:n_groups]:
            seeds.append((sx, sy, sdx, sdy, pr, None))

    new_edges: list[dict] = []
    polylines: list[list[tuple[float, float]]] = []
    host_regions: list[tuple[float, float, float]] = []
    host_region_ids: set[int] = set()

    # Global smooth swirl field for coherence across sprouts
    phx = rng.uniform(0.0, 2.0 * math.pi)
    phy = rng.uniform(0.0, 2.0 * math.pi)
    kx = 2.0 * math.pi / 0.18
    ky = 2.0 * math.pi / 0.24

    def _curl_dir(xm: float, ym: float, dirx: float, diry: float) -> tuple[float, float]:
        n = 0.5 * (math.sin(kx * xm + phx) + math.sin(ky * ym + phy))
        tx, ty = -diry, dirx
        dx = (1.0 - curl) * dirx + curl * (tx * n)
        dy = (1.0 - curl) * diry + curl * (ty * n)
        # jitter
        dx += jitter * rng.uniform(-0.7, 0.7)
        dy += jitter * rng.uniform(-0.7, 0.7)
        return _unit(dx, dy)

    def _inside_dropout(reg, x: float, y: float, cx: float, cy: float, r0: float) -> bool:
        try:
            val = float(np.asarray(reg.inside_field(np.array([x], dtype=np.float32), np.array([y], dtype=np.float32)))[0])
            return val > 0.0
        except Exception:
            dx, dy = (x - cx), (y - cy)
            return (dx * dx + dy * dy) <= (r0 * r0)

    for (sx, sy, sdx, sdy, parent_r, reg_host) in seeds:
        # Skip if clearly inside FAZ
        if math.hypot(sx - fx, sy - fy) <= (1.05 * fr):
            continue

        # main sprout
        px = float(sx + 0.6 * step_len * sdx)
        py = float(sy + 0.6 * step_len * sdy)
        dirx, diry = float(sdx), float(sdy)
        pts = [(px, py)]
        for _ in range(steps_main):
            # Respect bounds and FAZ avoidance
            if not (0.0 < px < 1.0 and 0.0 < py < 1.0):
                break
            if math.hypot(px - fx, py - fy) <= (1.02 * fr):
                break
            # If anchored to a dropout, constrain growth to remain inside the dropout
            if reg_host is not None:
                cx = float(getattr(reg_host, 'cx', 0.5)); cy = float(getattr(reg_host, 'cy', 0.5)); r0 = float(getattr(reg_host, 'r0', 0.12))
                # Add a slight inward pull to keep threads in the dropout
                icx, icy = _unit(cx - px, cy - py)
                dirx = 0.80 * dirx + 0.20 * icx
                diry = 0.80 * diry + 0.20 * icy
            dirx, diry = _curl_dir(px, py, dirx, diry)
            nx = px + step_len * dirx
            ny = py + step_len * diry
            if reg_host is not None:
                cx = float(getattr(reg_host, 'cx', 0.5)); cy = float(getattr(reg_host, 'cy', 0.5)); r0 = float(getattr(reg_host, 'r0', 0.12))
                if not _inside_dropout(reg_host, nx, ny, cx, cy, r0):
                    break
            if (abs(nx - pts[-1][0]) + abs(ny - pts[-1][1])) > (step_len * 0.4):
                pts.append((nx, ny))
            px, py = nx, ny

        # Optional side branch along the main path
        side_lines: list[list[tuple[float, float]]] = []
        if len(pts) >= 6 and rng.random() < side_prob:
            j = int(rng.integers(2, max(3, len(pts) - 2)))
            bx, by = pts[j]
            # derive local tangent at j
            vx = pts[min(len(pts) - 1, j + 1)][0] - pts[max(0, j - 1)][0]
            vy = pts[min(len(pts) - 1, j + 1)][1] - pts[max(0, j - 1)][1]
            vx, vy = _unit(vx, vy)
            ang = (math.pi / 3.0) * (1.0 if rng.random() < 0.5 else -1.0)
            ca, sa = math.cos(ang), math.sin(ang)
            bdx = ca * vx - sa * vy
            bdy = sa * vx + ca * vy
            qx, qy = bx, by
            b_pts = [(qx, qy)]
            for _ in range(steps_side):
                if not (0.0 < qx < 1.0 and 0.0 < qy < 1.0):
                    break
                if math.hypot(qx - fx, qy - fy) <= (1.02 * fr):
                    break
                bdx, bdy = _curl_dir(qx, qy, bdx, bdy)
                qx2, qy2 = qx + 0.85 * step_len * bdx, qy + 0.85 * step_len * bdy
                if reg_host is not None:
                    cx = float(getattr(reg_host, 'cx', 0.5)); cy = float(getattr(reg_host, 'cy', 0.5)); r0 = float(getattr(reg_host, 'r0', 0.12))
                    if not _inside_dropout(reg_host, qx2, qy2, cx, cy, r0):
                        break
                b_pts.append((qx2, qy2))
                qx, qy = qx2, qy2
            if len(b_pts) >= 2:
                side_lines.append(b_pts)

        # Emit edges with tapering radius
    def _emit_polyline(line_pts: list[tuple[float, float]]) -> bool:
        if len(line_pts) < 2:
            return False
        polylines.append(line_pts)
        r_parent = float(parent_r)
        r_start = max(r_parent * th_scale, r_min_mm)
        r_end = max(0.6 * r_start, 0.7 * r_min_mm)
        for i in range(1, len(line_pts)):
            t = i / max(1, (len(line_pts) - 1))
            r_seg = (1.0 - t) * r_start + t * r_end
            new_edges.append({
                "node1": (line_pts[i - 1][0], line_pts[i - 1][1]),
                "node2": (line_pts[i][0], line_pts[i][1]),
                "radius": float(r_seg),
            })
        return True

        any_emitted = _emit_polyline(pts)
        for b in side_lines:
            any_emitted = _emit_polyline(b) or any_emitted

        # Host region metadata: record only when any NV polyline was emitted
        if any_emitted:
            if reg_host is not None:
                cx = float(getattr(reg_host, 'cx', 0.5)); cy = float(getattr(reg_host, 'cy', 0.5)); r0 = float(getattr(reg_host, 'r0', 0.12))
                key = id(reg_host)
                if key not in host_region_ids:
                    host_regions.append((cx, cy, r0))
                    host_region_ids.add(key)
            else:
                all_pts = pts[:]
                for b in side_lines:
                    all_pts.extend(b)
                if all_pts:
                    cx = float(np.mean([p[0] for p in all_pts]))
                    cy = float(np.mean([p[1] for p in all_pts]))
                    rad = float(max(0.02, 1.25 * max(math.hypot(p[0] - cx, p[1] - cy) for p in all_pts)))
                    host_regions.append((cx, cy, min(rad, 0.18)))

    return new_edges, polylines, host_regions
