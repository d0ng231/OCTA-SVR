import os
import json
import math
from typing import Dict, List, Optional, Tuple
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------
# Direction helpers (8-sector, FAZ-relative)
# -----------------------------
def _dir8_label_relative(x: float, y: float, cx: float, cy: float) -> str:
    """Map a point to one of 8 directions relative to (cx, cy).

    Labels: right, top-right, top, top-left, left, bottom-left, bottom, bottom-right.
    Coordinate convention: image origin at top-left; x=row (down +), y=col (right +).
    """
    dx = float(x) - float(cx)  # down is +
    dy = float(y) - float(cy)  # right is +
    # 0 deg = right, 90 = up (top), 180 = left, 270 = down (bottom)
    ang = math.degrees(math.atan2(-dx, dy))
    if ang < 0:
        ang += 360.0
    dirs = [
        (0.0, "right"),
        (45.0, "top-right"),
        (90.0, "top"),
        (135.0, "top-left"),
        (180.0, "left"),
        (225.0, "bottom-left"),
        (270.0, "bottom"),
        (315.0, "bottom-right"),
    ]
    def circ_dist(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)
    return min(dirs, key=lambda t: circ_dist(ang, t[0]))[1]


def _dir8_is_diagonal(label: str) -> bool:
    """Return True if an 8-way direction label is a diagonal (e.g., top-right)."""
    return label in {"top-right", "top-left", "bottom-left", "bottom-right"}


def _coarse_loc_label(x: float, y: float) -> str:
    return _dir8_label_relative(x, y, 0.5, 0.5)


def _coarse_loc_label_relative(x: float, y: float, cx: float, cy: float) -> str:
    return _dir8_label_relative(x, y, cx, cy)


def _qual_strength(alpha: Optional[float]) -> Optional[str]:
    if alpha is None:
        return None
    try:
        a = max(0.0, min(1.0, float(alpha)))
    except Exception:
        return None
    if a >= 0.8:
        return "marked"
    if a >= 0.6:
        return "moderate"
    if a >= 0.4:
        return "mild"
    return "subtle"


def _safe_load_pathology_json(dataset_root: str, path_rel: Optional[str]) -> Dict[str, List[Dict]]:
    out = {"dropouts": [], "microaneurysms": [], "neovascularization": [], "FAZ": None, "summary": None}
    if not path_rel:
        return out
    try:
        path = os.path.join(dataset_root, path_rel)
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                d = data.get("dropouts") or []
                m = data.get("microaneurysms") or []
                nv = data.get("neovascularization") or data.get("nv_regions") or []
                faz = data.get("FAZ")
                summ = data.get("summary")
                if isinstance(d, list):
                    out["dropouts"] = d
                if isinstance(m, list):
                    out["microaneurysms"] = m
                if isinstance(nv, list):
                    out["neovascularization"] = nv
                if isinstance(faz, dict):
                    out["FAZ"] = faz
                if isinstance(summ, dict):
                    out["summary"] = summ
    except Exception:
        pass
    return out


def _safe_load_pathology_yml(dataset_root: str, path_rel: Optional[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not (yaml and path_rel):
        return out
    try:
        path = os.path.join(dataset_root, path_rel)
        if os.path.isfile(path):
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                out = data
    except Exception:
        return {}
    return out


def build_mixed_paragraph(rec: Dict, dataset_root: str) -> str:
    """Compose a single descriptive, CoT‑style paragraph summarizing findings.

    Includes FAZ position/size (qualitative), dropout presence/strength and
    locations relative to the FAZ (robust to any pan/crop), microaneurysm
    density with emphasis on the largest few and where others lie, NV presence
    and adjacency, and tortuosity. Does not report count ranges.
    """
    # Gather structured info
    drop_meta = rec.get("dropout") or None
    nv_meta = rec.get("neovasc") or None
    nv_adj = rec.get("nv_adjacent_to_dropout")

    pathology = _safe_load_pathology_json(dataset_root, rec.get("dropout_ma_json"))
    patho_yml = _safe_load_pathology_yml(dataset_root, rec.get("pathology_yml"))
    # Prefer embedded fields in metadata when available; else fall back to JSON file
    dropouts = rec.get("dropouts") or pathology.get("dropouts") or []
    mas = rec.get("microaneurysms") or pathology.get("microaneurysms") or []
    # Merge NV from JSON and, if needed, from YAML (NV.regions)
    nv_list_json = pathology.get("neovascularization") or []
    nv_list_yaml = []
    try:
        nv_yaml_block = patho_yml.get("NV") if isinstance(patho_yml, dict) else None
        if isinstance(nv_yaml_block, dict) and isinstance(nv_yaml_block.get("regions"), list):
            nv_list_yaml = nv_yaml_block.get("regions")
    except Exception:
        nv_list_yaml = []
    nv_list_raw = (nv_list_json if isinstance(nv_list_json, list) else []) + (nv_list_yaml if isinstance(nv_list_yaml, list) else [])
    nv_list: List[Dict] = []
    if isinstance(nv_list_raw, list):
        seen_nv: set[tuple[float, float]] = set()
        for item in nv_list_raw:
            if not isinstance(item, dict):
                continue
            center = item.get("center_norm") or item.get("center")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                continue
            try:
                cx = round(float(center[0]), 4)
                cy = round(float(center[1]), 4)
            except Exception:
                continue
            key = (cx, cy)
            if key in seen_nv:
                continue
            seen_nv.add(key)
            cleaned = dict(item)
            cleaned["center_norm"] = [cx, cy]
            nv_list.append(cleaned)

    # Helpers for distributions (8-direction, FAZ-relative)
    def _dir_hist_from_points(pts: List[Tuple[float, float]], cx: float, cy: float) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for (x, y) in pts:
            lab = _coarse_loc_label_relative(float(x), float(y), cx, cy)
            hist[lab] = hist.get(lab, 0) + 1
        return hist

    def _approx_count_range(n: int) -> str:
        """Return coarse count buckets for CoT: 1-10, 10-20, >20."""
        try:
            n = int(max(0, n))
        except Exception:
            return ""
        if n <= 10:
            return "1-10"
        if n <= 20:
            return "10-20"
        return ">20"

    def _dir_hist_to_text(hist: Dict[str, int], top_k: int = 3) -> str:
        if not hist:
            return ""
        items = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, top_k)]
        pieces = [f"{k} ({_approx_count_range(v)})" for k, v in items]
        if len(pieces) == 1:
            return pieces[0]
        if len(pieces) == 2:
            return f"{pieces[0]} and {pieces[1]}"
        return ", ".join(pieces[:-1]) + f", and {pieces[-1]}"

    # Determine FAZ center for relative location references
    # Resolve FAZ center: prefer per-sample Greenhouse config.yml under graph_dir
    def _faz_from_config() -> Optional[Tuple[float, float]]:
        try:
            gdir = rec.get("graph_dir")
            if gdir:
                cpath = os.path.join(gdir, "config.yml")
                if os.path.isfile(cpath) and yaml is not None:
                    with open(cpath, "r") as yf:
                        cfg = yaml.safe_load(yf)
                    if isinstance(cfg, dict):
                        gh = cfg.get("Greenhouse") or {}
                        fc = gh.get("FAZ_center")
                        if isinstance(fc, (list, tuple)) and len(fc) == 2:
                            return float(fc[0]), float(fc[1])
        except Exception:
            return None
        return None

    faz_info = rec.get("FAZ") or pathology.get("FAZ") or {}
    try:
        cfg_xy = _faz_from_config()
        if cfg_xy is not None:
            fx, fy = cfg_xy
        else:
            fx, fy = faz_info.get("center_norm", [0.5, 0.5])
            fx = float(fx); fy = float(fy)
    except Exception:
        fx, fy = 0.5, 0.5
    # FAZ size (qualitative)
    faz_radius = None
    try:
        rnorm = faz_info.get("radius_norm")
        if rnorm is not None:
            faz_radius = float(rnorm)
    except Exception:
        faz_radius = None

    # Determine representative (largest) dropout and additional regions
    largest_loc = None  # str
    largest_strength = None  # float
    extra_regions: List[Tuple[str, Optional[float], Optional[float]]] = []  # (loc, strength, radius)
    quad_counts: Dict[str, int] = {}
    all_radii: List[float] = []
    if dropouts:
        try:
            # JSON uses keys: center [x,y] and radius (normalized)
            order = sorted(range(len(dropouts)), key=lambda i: float(dropouts[i].get("radius", 0.0)), reverse=True)
            if order:
                i0 = order[0]
                d0 = dropouts[i0]
                c0 = d0.get("center") or d0.get("center_norm") or [0.5, 0.5]
                largest_loc = _coarse_loc_label_relative(float(c0[0]), float(c0[1]), fx, fy)
                largest_strength = float(dropouts[i0].get("strength", 0.0))
                # Capture up to 3 additional regions by size
                for j in order[1:4]:
                    dj = dropouts[j]
                    cj = dj.get("center") or dj.get("center_norm") or [0.5, 0.5]
                    lj = _coarse_loc_label_relative(float(cj[0]), float(cj[1]), fx, fy)
                    sj = dj.get("strength")
                    rj = dj.get("radius")
                    extra_regions.append((lj, float(sj) if sj is not None else None, float(rj) if rj is not None else None))
                # Collect quadrant counts and radii for general description
                for d in dropouts:
                    try:
                        cc = d.get("center") or d.get("center_norm") or [0.5, 0.5]
                        cx, cy = cc
                        q = _coarse_loc_label_relative(float(cx), float(cy), fx, fy)
                        quad_counts[q] = quad_counts.get(q, 0) + 1
                        rr = d.get("radius")
                        if rr is not None:
                            all_radii.append(float(rr))
                    except Exception:
                        continue
        except Exception:
            largest_loc = None
            largest_strength = None
    # Fallback to metadata summary
    if largest_loc is None and isinstance(drop_meta, dict) and drop_meta.get("center_norm"):
        try:
            cx, cy = drop_meta["center_norm"]
            largest_loc = _coarse_loc_label_relative(float(cx), float(cy), fx, fy)
        except Exception:
            largest_loc = None

    # Additional dropout quadrants (exclude largest quadrant to avoid redundancy)
    extra_quads: List[str] = []
    if dropouts:
        for i, d in enumerate(dropouts):
            try:
                cx, cy = d.get("center", [0.5, 0.5])
                q = _coarse_loc_label_relative(float(cx), float(cy), fx, fy)
                if q and q != largest_loc and q not in extra_quads:
                    extra_quads.append(q)
            except Exception:
                continue

    # Microaneurysm qualitative density
    ma_density = None
    n_ma = len(mas) if isinstance(mas, list) else 0
    # Track largest MA for a single-location callout relative to FAZ
    largest_ma: Optional[Tuple[float, float, float]] = None  # (x, y, radius_mm)
    if n_ma > 0:
        for m in mas:
            try:
                rmm = float(m.get("radius_mm", 0.0))
                cx, cy = m.get("center", [None, None])
                if cx is None or cy is None:
                    continue
                if (largest_ma is None) or (rmm > largest_ma[2]):
                    largest_ma = (float(cx), float(cy), rmm)
            except Exception:
                continue
    if n_ma > 0:
        if n_ma <= 3:
            ma_density = "a few"
        elif n_ma <= 10:
            ma_density = "several"
        else:
            ma_density = "numerous"

    # NV location (relative to FAZ)
    nv_loc = None
    if nv_list:
        try:
            nx, ny = nv_list[0].get("center_norm", [None, None])
            if nx is not None and ny is not None:
                nv_loc = _coarse_loc_label_relative(float(nx), float(ny), fx, fy)
        except Exception:
            nv_loc = None
    if nv_loc is None and isinstance(nv_meta, dict) and nv_meta.get("center_norm"):
        try:
            nx, ny = nv_meta["center_norm"]
            nv_loc = _coarse_loc_label_relative(float(nx), float(ny), fx, fy)
        except Exception:
            nv_loc = None

    # FAZ apparent position in the final image
    # If a view shift was applied, describe FAZ displacement using that shift;
    # otherwise, fall back to Greenhouse FAZ center vs image center.
    faz_loc_phrase = None
    try:
        vs = rec.get("view_shift_norm")
        if isinstance(vs, (list, tuple)) and len(vs) == 2:
            # Map normalized view shift (sx along width/columns, sy along height/rows)
            sx = float(vs[0])  # columns → y
            sy = float(vs[1])  # rows → x
            # Build a virtual point offset from image center
            vx = 0.5 + sy  # row coordinate
            vy = 0.5 + sx  # col coordinate
            q = _coarse_loc_label(float(vx), float(vy))
            r = math.hypot(sx, sy)
        else:
            # Fall back to model FAZ center relative to image center
            q = _coarse_loc_label_relative(float(fx), float(fy), 0.5, 0.5)
            dx = float(fx) - 0.5
            dy = float(fy) - 0.5
            r = math.hypot(dx, dy)
        # Thresholds in normalized units; tuned for typical jitter scales; keep 'centered' for small offsets
        if r < 0.04:
            faz_loc_phrase = "roughly centered"
        elif r < 0.08:
            faz_loc_phrase = f"slightly off center toward the {q}"
        elif r < 0.16:
            faz_loc_phrase = f"off center toward the {q}"
        else:
            faz_loc_phrase = f"clearly off center toward the {q}"
    except Exception:
        faz_loc_phrase = None

    # Compose paragraph
    parts: List[str] = []

    # Step 0: FAZ (state displacement only; do not mention view panning)
    if faz_loc_phrase:
        parts.append(f"I first locate the foveal avascular zone: it is {faz_loc_phrase}.")
    else:
        parts.append("I first locate the foveal avascular zone: it lies near the center.")
    # Optional FAZ size qualifier
    if faz_radius is not None:
        try:
            if faz_radius < 0.045:
                parts.append("The FAZ appears small for this field of view.")
            elif faz_radius > 0.085:
                parts.append("The FAZ appears relatively large for this field of view.")
        except Exception:
            pass

    # Step 1: dropout
    if drop_meta or dropouts:
        strength_word = _qual_strength(largest_strength)
        n_do = len(dropouts) if isinstance(dropouts, list) else (1 if drop_meta else 0)
        if strength_word:
            if n_do >= 2:
                lead = f"There are multiple areas of {strength_word} capillary flow loss"
            else:
                lead = f"There is {strength_word} capillary dropout"
        else:
            lead = "There is capillary dropout" if n_do < 2 else "There are multiple areas of capillary flow loss"
        if largest_loc:
            lead += f"; the largest sits in the {largest_loc} area relative to the FAZ"
        lead += ", with uneven edges."

        # Emphasize the next two largest regions (by radius), then mention where others lie
        try:
            # Determine top-3 by radius
            order = sorted(range(len(dropouts)), key=lambda i: float(dropouts[i].get("radius", 0.0)), reverse=True) if dropouts else []
            top_extra_locs: List[str] = []
            for j in order[1:3]:
                dj = dropouts[j]
                cj = dj.get("center") or dj.get("center_norm") or [0.5, 0.5]
                lj = _coarse_loc_label_relative(float(cj[0]), float(cj[1]), fx, fy)
                if lj not in top_extra_locs and lj != largest_loc:
                    top_extra_locs.append(lj)
            if top_extra_locs:
                if len(top_extra_locs) == 1:
                    lead += f" Another sizeable region lies in the {top_extra_locs[0]} area."
                else:
                    lead += f" Additional sizeable regions lie in the {top_extra_locs[0]} and {top_extra_locs[1]} areas."
            # Remaining smaller foci — list distinct areas without counts
            other_dirs: List[str] = []
            for d in (dropouts or []):
                cc = d.get("center") or d.get("center_norm") or [0.5, 0.5]
                q = _coarse_loc_label_relative(float(cc[0]), float(cc[1]), fx, fy)
                if q and q != largest_loc and q not in top_extra_locs and q not in other_dirs:
                    other_dirs.append(q)
            if other_dirs:
                if len(other_dirs) == 1:
                    lead += f" Smaller foci are present in the {other_dirs[0]} area."
                elif len(other_dirs) == 2:
                    lead += f" Smaller foci are present in the {other_dirs[0]} and {other_dirs[1]} areas."
                else:
                    lead += f" Smaller foci are present in the {', '.join(other_dirs[:-1])}, and {other_dirs[-1]} areas."
        except Exception:
            pass
        # Removed radial-band summary sentence to keep CoT concise and avoid over-specifying distribution.
        parts.append(f"First, I check for areas of flow loss: {lead}")
    else:
        parts.append("First, I check for areas of flow loss: no clear capillary dropout is seen; blood flow looks continuous.")

    # Step 2: microaneurysms
    # Calibrate qualitative density by generator parameters when available (pathology_yml.MA.density)
    ma_density_cfg = None
    try:
        if isinstance(patho_yml.get("MA"), dict) and patho_yml["MA"].get("density") is not None:
            ma_density_cfg = float(patho_yml["MA"]["density"])  # per-segment probability
    except Exception:
        ma_density_cfg = None
    if ma_density or (ma_density_cfg is not None and ma_density_cfg > 0.0):
        # Prefer config-driven qualitative mapping when available
        if ma_density_cfg is not None and ma_density_cfg > 0.0:
            if ma_density_cfg < 0.004:
                ma_density_q = "a few"
            elif ma_density_cfg < 0.010:
                ma_density_q = "several"
            else:
                ma_density_q = "numerous"
        else:
            ma_density_q = ma_density if ma_density else "a few"
        if drop_meta or dropouts:
            sent = f"Next, I look along the edges of flow loss: {ma_density_q} small bulges (microaneurysms) are seen."
        else:
            sent = f"Next, I look at the capillaries: {ma_density_q} small bulges (microaneurysms) are present."
        # Emphasize up to three largest MAs (by radius), then mention where others lie
        try:
            ma_sorted = []
            for m in (mas or []):
                try:
                    rmm = float(m.get("radius_mm", 0.0))
                    cx, cy = m.get("center", [None, None])
                    if cx is None or cy is None:
                        continue
                    ma_sorted.append((rmm, float(cx), float(cy)))
                except Exception:
                    continue
            ma_sorted.sort(key=lambda t: t[0], reverse=True)
            top_ma = ma_sorted[:3]
            top_locs_ma: List[str] = []
            for _, x0, y0 in top_ma:
                loc = _coarse_loc_label_relative(float(x0), float(y0), fx, fy)
                if loc not in top_locs_ma:
                    top_locs_ma.append(loc)
            if top_locs_ma:
                if len(top_locs_ma) == 1:
                    sent += f" The largest lies {top_locs_ma[0]} of the FAZ."
                elif len(top_locs_ma) == 2:
                    sent += f" The largest lie {top_locs_ma[0]} and {top_locs_ma[1]} of the FAZ."
                else:
                    sent += f" The largest lie {top_locs_ma[0]}, {top_locs_ma[1]}, and {top_locs_ma[2]} of the FAZ."
            # Others (without counts): list distinct areas
            other_dirs_ma: List[str] = []
            for m in (mas or []):
                try:
                    cx, cy = m.get("center", [None, None])
                    if cx is None or cy is None:
                        continue
                    loc = _coarse_loc_label_relative(float(cx), float(cy), fx, fy)
                    if loc and loc not in top_locs_ma and loc not in other_dirs_ma:
                        other_dirs_ma.append(loc)
                except Exception:
                    continue
            if other_dirs_ma:
                if len(other_dirs_ma) == 1:
                    sent += f" Additional tiny dots appear in the {other_dirs_ma[0]} area."
                elif len(other_dirs_ma) == 2:
                    sent += f" Additional tiny dots appear in the {other_dirs_ma[0]} and {other_dirs_ma[1]} areas."
                else:
                    sent += f" Additional tiny dots appear in the {', '.join(other_dirs_ma[:-1])}, and {other_dirs_ma[-1]} areas."
        except Exception:
            pass
        parts.append(sent)
    else:
        parts.append("Next, I look at the capillaries: no clear microaneurysms are seen.")

    # Step 3: neovascularization — prefer per-site NV list (JSON/YAML) for direction; fallback to meta
    if nv_meta or nv_list:
        # Cluster NV sites by proximity to avoid double-counting and stabilize directions
        pts: List[Tuple[float, float]] = []
        for n in (nv_list or []):
            try:
                nx, ny = n.get("center_norm", [None, None])
                if nx is None or ny is None:
                    continue
                pts.append((float(nx), float(ny)))
            except Exception:
                continue

        # Simple greedy clustering with tolerance (normalized units)
        def _cluster_points(points: List[Tuple[float, float]], tol: float = 0.035) -> List[Tuple[float, float]]:
            centers: List[Tuple[float, float]] = []
            for (x, y) in points:
                assigned = False
                for i, (cx0, cy0) in enumerate(centers):
                    if (abs(x - cx0) + abs(y - cy0)) <= (2.0 * tol) or ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 <= tol:
                        # merge by averaging
                        centers[i] = ((cx0 + x) * 0.5, (cy0 + y) * 0.5)
                        assigned = True
                        break
                if not assigned:
                    centers.append((x, y))
            return centers

        centers = _cluster_points(pts) if pts else []
        nv_locs: List[str] = []
        if centers:
            try:
                labs: Dict[str, int] = {}
                for (nx, ny) in centers:
                    lab = _coarse_loc_label_relative(float(nx), float(ny), fx, fy)
                    labs[lab] = labs.get(lab, 0) + 1
                nv_locs = [
                    k
                    for k, _ in sorted(
                        labs.items(),
                        key=lambda kv: (-kv[1], 0 if _dir8_is_diagonal(kv[0]) else 1, kv[0]),
                    )[:2]
                ]
            except Exception:
                nv_locs = []
        # Fallback to record-level meta only when no site list
        if not nv_locs and isinstance(nv_meta, dict) and nv_meta.get("center_norm") is not None:
            try:
                nx, ny = nv_meta["center_norm"]
                nv_locs = [_coarse_loc_label_relative(float(nx), float(ny), fx, fy)]
            except Exception:
                pass
        # Final fallback
        if not nv_locs and nv_loc:
            nv_locs = [nv_loc]

        count_nv = max(1 if nv_locs else 0, len(centers))

        if count_nv > 1:
            stem = "Finally, I look for new vessels: in multiple areas, thicker, curved branches sprout from dropout borders and extend inward"
        else:
            stem = "Finally, I look for new vessels: thicker, curved branches sprout from the border of dropout and extend inward"

        if nv_locs:
            parts.append(
                f"{stem}, located in the {nv_locs[0]} area relative to the FAZ."
                if len(nv_locs) == 1
                else f"{stem}, located in the {nv_locs[0]} and {nv_locs[1]} areas relative to the FAZ."
            )
        else:
            parts.append(f"{stem}.")
    else:
        parts.append("Finally, I look for new vessels: none are seen.")

    # Step 4: tortuosity/course — prefer pathology_yml.Tortuosity.gain; fallback to JSON summary
    tor = pathology.get("summary", {}).get("tortuosity") if isinstance(pathology.get("summary"), dict) else None
    tor_present = bool(tor.get("present", False)) if isinstance(tor, dict) else False
    gain_cfg = None
    try:
        if isinstance(patho_yml.get("Tortuosity"), dict) and patho_yml["Tortuosity"].get("gain") is not None:
            gain_cfg = float(patho_yml["Tortuosity"]["gain"])
    except Exception:
        gain_cfg = None
    # Fallback: use JSON summary gain if YAML missing
    if (gain_cfg is None or gain_cfg <= 0.0) and isinstance(tor, dict) and tor.get("gain") is not None:
        try:
            gain_cfg = float(tor.get("gain"))
        except Exception:
            pass
    # Fallback 2: use per-record extras if available (records_graphs.jsonl.dropout_extras.tortuosity_gain)
    if (gain_cfg is None or gain_cfg <= 0.0):
        try:
            de = rec.get("dropout_extras") or {}
            if isinstance(de, dict) and de.get("tortuosity_gain") is not None:
                gain_cfg = float(de.get("tortuosity_gain"))
        except Exception:
            pass

    if gain_cfg is not None and gain_cfg > 0.0:
        if gain_cfg < 0.012:
            parts.append("Lastly, I look at vessel shape: capillaries show subtle extra twisting.")
        elif gain_cfg < 0.022:
            parts.append("Lastly, I look at vessel shape: capillaries show mild extra twisting.")
        else:
            parts.append("Lastly, I look at vessel shape: capillaries show moderate extra twisting.")
    elif tor_present:
        # Present but no gain value — state subtle twisting conservatively
        parts.append("Lastly, I look at vessel shape: capillaries show subtle extra twisting.")
    else:
        parts.append("Lastly, I look at vessel shape: vessels look smooth without extra twisting.")

    return " " + " ".join(parts).strip()
