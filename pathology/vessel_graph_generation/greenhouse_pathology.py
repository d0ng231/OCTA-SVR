import math
import numpy as np
import random
from math import sqrt
from typing import Tuple, Sequence, Optional

from vessel_graph_generation.greenhouse import Greenhouse
from vessel_graph_generation.forest import Forest
from vessel_graph_generation.element_mesh import CoordKdTree, NodeKdTree, SpacePartitioner
from vessel_graph_generation.utilities import norm_vector, get_angle_between_vectors, get_angle_between_two_vectors
from vessel_graph_generation.arterial_tree import Node


class _DropoutRegion:
    def __init__(
        self,
        center_xy: Tuple[float, float],
        radius_xy: float,
        drop_strength: float = 1.0,
        irregularity_amp: float = 0.2,
        harmonics: tuple[int, ...] = (2, 3, 5),
        ellipticity: tuple[float, float] = (1.0, 1.2),
        gradient_alpha: float = 2.0,
        noise_gain: float = 0.35,
    ) -> None:
        self.cx = float(center_xy[0])
        self.cy = float(center_xy[1])
        self.r0 = float(radius_xy)
        self.drop_strength = float(drop_strength)
        self.irregularity_amp = float(irregularity_amp)
        self.harmonics = tuple(harmonics)
        self.a = max(1e-3, float(ellipticity[0]))
        self.b = max(1e-3, float(ellipticity[1]))
        self.gradient_alpha = float(gradient_alpha)
        self.noise_gain = float(noise_gain)

        # Coherent randomization per region
        self._phases = np.random.uniform(0, 2 * np.pi, size=len(self.harmonics))
        # Increase edge roughness when more harmonics are present, without exploding amplitude
        denom = max(np.sqrt(max(len(self.harmonics), 1)), 1.0)
        self._amps = (self.irregularity_amp * np.random.uniform(0.4, 1.0, size=len(self.harmonics))) / denom
        # Slightly broaden spatial noise frequencies to induce less-round boundaries
        self._fx = np.random.uniform(2.0, 9.0)
        self._fy = np.random.uniform(2.0, 9.0)
        self._fxy = np.random.uniform(2.0, 9.0)
        self._phx = np.random.uniform(0, 2 * np.pi)
        self._phy = np.random.uniform(0, 2 * np.pi)
        self._phxy = np.random.uniform(0, 2 * np.pi)

    def _boundary_radius(self, theta: float) -> float:
        ct = np.cos(theta)
        st = np.sin(theta)
        r_ell = self.r0 / np.sqrt((ct / self.a) ** 2 + (st / self.b) ** 2)
        if len(self.harmonics) == 0:
            return float(r_ell)
        perturb = 1.0 + np.sum(self._amps * np.cos(np.array(self.harmonics) * theta + self._phases))
        return float(r_ell * max(0.6, perturb))

    def _smooth_value_noise(self, x: float, y: float) -> float:
        v = 0.5 * (np.sin(2 * np.pi * self._fx * x + self._phx) + np.sin(2 * np.pi * self._fy * y + self._phy))
        v += 0.35 * np.sin(2 * np.pi * self._fxy * (0.7 * x + 0.3 * y) + self._phxy)
        return float(np.clip(0.5 * (v + 1.0), 0.0, 1.0))

    def inside_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized inside_score evaluated over arrays of coordinates."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        dx = x_arr - self.cx
        dy = y_arr - self.cy
        theta = np.arctan2(dy, dx)

        ct = np.cos(theta)
        st = np.sin(theta)
        denom = (ct / self.a) ** 2 + (st / self.b) ** 2
        r_ell = self.r0 / np.sqrt(np.maximum(denom, 1e-9))
        if len(self.harmonics) > 0:
            perturb = np.ones_like(r_ell, dtype=float)
            for amp, phase, harm in zip(self._amps, self._phases, self.harmonics):
                perturb += amp * np.cos(harm * theta + phase)
            r_ell = r_ell * np.maximum(0.6, perturb)

        rp = np.hypot(dx, dy)
        core = np.clip(1.0 - rp / (r_ell + 1e-8), 0.0, 1.0)
        core = core ** self.gradient_alpha

        noise = 0.5 * (
            np.sin(2 * np.pi * self._fx * x_arr + self._phx)
            + np.sin(2 * np.pi * self._fy * y_arr + self._phy)
        )
        noise += 0.35 * np.sin(2 * np.pi * self._fxy * (0.7 * x_arr + 0.3 * y_arr) + self._phxy)
        noise = np.clip(0.5 * (noise + 1.0), 0.0, 1.0)
        core *= np.clip(0.75 + self.noise_gain * (noise - 0.5), 0.0, 1.2)
        return np.clip(core, 0.0, 1.0)

    def inside_score(self, x: float, y: float) -> float:
        return float(np.asarray(self.inside_field(x, y)))

    def suppress_prob(self, x: float, y: float) -> float:
        return float(np.clip(self.drop_strength * self.inside_score(x, y), 0.0, 1.0))


class NVRegion:
    def __init__(
        self,
        center_xy: Tuple[float, float],
        radius_xy: float,
        step_len_factor: float = 0.45,
        mean_radius_mm: float = 0.007,
        std_radius_mm: float = 0.002,
        branch_prob: float = 0.35,
        curl_factor: float = 0.5,
        edges_per_iter: int = 20,
        # shape realism
        irregularity_amp: float = 0.12,
        harmonics: tuple[int, ...] = (3, 5, 7),
        ellipticity: tuple[float, float] = (1.0, 1.35),
        gradient_alpha: float = 1.6,
        noise_gain: float = 0.25,
        connect_prob: float = 0.12,
        connect_radius_norm: float = 0.025,
        # Optional legacy params kept for compatibility; core logic no longer depends on them
        border_center: Tuple[float, float] | None = None,
        border_radius: float | None = None,
        border_band: float = 0.06,
        border_bias: float = 0.5,
        outward_bias: float = 0.85,
        init_spokes: int = 6,
        spoke_jitter: float = 0.25,
        min_clearance_vessel_norm: float = 0.012,
        min_clearance_nv_norm: float = 0.010,
    ) -> None:
        self.center = (float(center_xy[0]), float(center_xy[1]))
        self.radius = float(radius_xy)
        self.step_len_factor = float(step_len_factor)
        self.mean_radius_mm = float(mean_radius_mm)
        self.std_radius_mm = float(std_radius_mm)
        self.branch_prob = float(branch_prob)
        self.curl_factor = float(curl_factor)
        self.edges_per_iter = int(edges_per_iter)
        self.anchor: Node | None = None
        self.tips: list[Node] = []
        # Boundary shape (irregular disk similar to ischemic patch but smaller amplitude)
        self.irregularity_amp = float(irregularity_amp)
        self.harmonics = tuple(harmonics)
        self.a = max(1e-3, float(ellipticity[0]))
        self.b = max(1e-3, float(ellipticity[1]))
        self.gradient_alpha = float(gradient_alpha)
        self.noise_gain = float(noise_gain)
        # Phases for coherent irregular boundary
        self._phases = np.random.uniform(0, 2 * np.pi, size=len(self.harmonics))
        self._amps = self.irregularity_amp * np.random.uniform(0.4, 1.0, size=len(self.harmonics)) / max(len(self.harmonics), 1)
        self._fx = np.random.uniform(2.0, 6.0)
        self._fy = np.random.uniform(2.0, 6.0)
        self._fxy = np.random.uniform(2.0, 6.0)
        self._phx = np.random.uniform(0, 2 * np.pi)
        self._phy = np.random.uniform(0, 2 * np.pi)
        self._phxy = np.random.uniform(0, 2 * np.pi)
        # Loop/anastomosis heuristics (not used in minimal tuft core loop, retained for compatibility)
        self.connect_prob = float(connect_prob)
        self.connect_radius_norm = float(connect_radius_norm)
        # Border-related params kept for compatibility (ignored in core growth)
        self.border_center = border_center
        self.border_radius = border_radius
        self.border_band = float(border_band)
        self.border_bias = float(border_bias)
        try:
            self.outward_bias = float(outward_bias)
        except Exception:
            self.outward_bias = 0.85
        try:
            self.init_spokes = max(1, int(init_spokes))
        except Exception:
            self.init_spokes = 6
        try:
            self.spoke_jitter = float(spoke_jitter)
        except Exception:
            self.spoke_jitter = 0.25
        try:
            self.min_clearance_v = float(min_clearance_vessel_norm)
        except Exception:
            self.min_clearance_v = 0.012
        try:
            self.min_clearance_nv = float(min_clearance_nv_norm)
        except Exception:
            self.min_clearance_nv = 0.010

        # Hard guarantees for visible growth
        self._min_edges_guarantee = max(4, int(self.edges_per_iter // 2))
        self._force_inside_threshold = 1e-4  # allow clamping to inside with almost-zero core

    def _boundary_radius(self, theta: float) -> float:
        ct = np.cos(theta)
        st = np.sin(theta)
        r_ell = self.radius / np.sqrt((ct / self.a) ** 2 + (st / self.b) ** 2)
        if len(self.harmonics) == 0:
            return float(r_ell)
        perturb = 1.0 + np.sum(self._amps * np.cos(np.array(self.harmonics) * theta + self._phases))
        return float(r_ell * max(0.6, perturb))

    def _smooth_value_noise(self, x: float, y: float) -> float:
        v = 0.5 * (np.sin(2 * np.pi * self._fx * x + self._phx) + np.sin(2 * np.pi * self._fy * y + self._phy))
        v += 0.35 * np.sin(2 * np.pi * self._fxy * (0.7 * x + 0.3 * y) + self._phxy)
        return float(np.clip(0.5 * (v + 1.0), 0.0, 1.0))

    def inside_field(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        dx = x_arr - self.center[0]
        dy = y_arr - self.center[1]
        theta = np.arctan2(dy, dx)

        ct = np.cos(theta)
        st = np.sin(theta)
        denom = (ct / self.a) ** 2 + (st / self.b) ** 2
        r_ell = self.radius / np.sqrt(np.maximum(denom, 1e-9))
        if len(self.harmonics) > 0:
            perturb = np.ones_like(r_ell, dtype=float)
            for amp, phase, harm in zip(self._amps, self._phases, self.harmonics):
                perturb += amp * np.cos(harm * theta + phase)
            r_ell = r_ell * np.maximum(0.6, perturb)

        rp = np.hypot(dx, dy)
        core = np.clip(1.0 - rp / (r_ell + 1e-8), 0.0, 1.0)
        core = core ** self.gradient_alpha

        noise = 0.5 * (
            np.sin(2 * np.pi * self._fx * x_arr + self._phx)
            + np.sin(2 * np.pi * self._fy * y_arr + self._phy)
        )
        noise += 0.35 * np.sin(2 * np.pi * self._fxy * (0.7 * x_arr + 0.3 * y_arr) + self._phxy)
        noise = np.clip(0.5 * (noise + 1.0), 0.0, 1.0)
        core *= np.clip(0.75 + self.noise_gain * (noise - 0.5), 0.0, 1.2)
        return np.clip(core, 0.0, 1.0)

    def inside_score(self, x: float, y: float) -> float:
        return float(np.asarray(self.inside_field(x, y)))

    def _within(self, pos: np.ndarray) -> bool:
        return self.inside_score(float(pos[0]), float(pos[1])) > 0.0

    def _ensure_anchor(self, greenhouse: Greenhouse):
        # 如果已经有anchor和tips，直接返回（避免重复初始化）
        if self.anchor is not None and self.tips:
            return
        
        # 找到最近的动脉节点作为anchor（用于连接到正常血管网络）
        center3 = np.array([self.center[0], self.center[1], greenhouse.simspace.shape[2] * 0.5], dtype=float)
        cand = greenhouse.art_node_mesh.find_nearest_element(center3, max_dist=np.inf)
        if cand is None:
            return
        
        self.anchor = cand
        # Robust central seeding at region center; never fall back to anchor-only
        z_level = float(self.anchor.position[2])
        seed_pos = np.array([self.center[0], self.center[1], z_level], dtype=float)
        r_mm = float(np.clip(np.random.normal(self.mean_radius_mm, self.std_radius_mm), 0.0030, 0.012))
        # Create central seed as child of the nearest existing vessel so it renders/exports consistently
        central_seed = cand.tree.add_node(seed_pos, r_mm, cand, cand.kappa)
        try:
            setattr(central_seed, 'is_neovascular', True)
        except Exception:
            pass

        # Radial spokes from the exact NV center
        step_len = max(0.18 * greenhouse.d, self.step_len_factor * greenhouse.d)
        spoke_count = max(1, self.init_spokes)
        base_ang = float(np.random.uniform(0.0, 2.0 * np.pi))
        seed_tips: list[Node] = []
        for k in range(spoke_count):
            ang = base_ang + 2.0 * np.pi * (k / spoke_count)
            ang += float(np.random.uniform(-self.spoke_jitter, self.spoke_jitter)) * (2.0 * np.pi / max(spoke_count, 1))
            dir2 = np.array([np.cos(ang), np.sin(ang)], dtype=float)
            dir2 /= (np.linalg.norm(dir2) + 1e-8)
            swirl = np.array([-dir2[1], dir2[0]], dtype=float)
            swirl /= (np.linalg.norm(swirl) + 1e-8)
            dir2 = dir2 * 0.65 + swirl * np.random.uniform(-0.5, 0.5)
            dir2 /= (np.linalg.norm(dir2) + 1e-8)
            # attempt with several scales; clamp inside
            placed = False
            for scale in (1.0, 0.8, 0.6):
                cand_pos = central_seed.position + np.array([dir2[0], dir2[1], 0.0], dtype=float) * (step_len * 0.75 * scale)
                cand_pos[2] = z_level + np.random.uniform(-0.002, 0.002)
                cand_pos = self._clamp_inside(cand_pos)
                rk = float(np.clip(np.random.normal(self.mean_radius_mm, self.std_radius_mm), 0.0030, 0.013))
                sp = central_seed.tree.add_node(cand_pos, rk, central_seed, central_seed.kappa)
                try:
                    setattr(sp, 'is_neovascular', True)
                except Exception:
                    pass
                dir_vec = np.array([dir2[0], dir2[1], 0.0], dtype=float)
                setattr(sp, '_nv_last_dir', dir_vec)
                seed_tips.append(sp)
                placed = True
                break
            if not placed:
                # even if none place (unlikely), keep at least a tiny spoke
                cand_pos = central_seed.position + np.array([dir2[0], dir2[1], 0.0], dtype=float) * (0.25 * step_len)
                cand_pos[2] = z_level
                cand_pos = self._clamp_inside(cand_pos)
                rk = float(max(0.0030, self.mean_radius_mm * 0.75))
                sp = central_seed.tree.add_node(cand_pos, rk, central_seed, central_seed.kappa)
                try:
                    setattr(sp, 'is_neovascular', True)
                except Exception:
                    pass
                setattr(sp, '_nv_last_dir', np.array([dir2[0], dir2[1], 0.0], dtype=float))
                seed_tips.append(sp)

        self.tips = seed_tips if seed_tips else [central_seed]

    def _flow_direction(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute a smoothed outward-flow direction with gentle curl."""
        x = float(pos[0]); y = float(pos[1])
        radial = np.array([x - self.center[0], y - self.center[1]], dtype=float)
        n_rad = np.linalg.norm(radial)
        if n_rad < 1e-6:
            radial = np.random.randn(2)
            n_rad = np.linalg.norm(radial)
        radial /= (n_rad + 1e-8)

        core = float(self.inside_score(x, y))

        # Blend in the gradient of the dropout field so the flow follows the basin.
        eps = max(1e-3, 0.006 * self.radius)
        try:
            gx = self.inside_score(x + eps, y) - self.inside_score(x - eps, y)
            gy = self.inside_score(x, y + eps) - self.inside_score(x, y - eps)
            grad = np.array([gx, gy], dtype=float)
            n_grad = np.linalg.norm(grad)
            if n_grad > 1e-6:
                grad = -grad / (n_grad + 1e-8)  # negative gradient points outward
                radial = 0.7 * radial + 0.3 * grad
                radial /= (np.linalg.norm(radial) + 1e-8)
        except Exception:
            pass

        # Rotate the radial vector by a curl angle derived from coherent noise.
        noise_phase = float(self._smooth_value_noise(x, y) - 0.5)
        swirl_strength = np.clip((0.35 + 0.45 * (1.0 - core)) * self.curl_factor, 0.0, 0.95)
        angle_offset = float(np.clip(noise_phase * swirl_strength, -0.9, 0.9) * np.pi)
        cos_a = math.cos(angle_offset)
        sin_a = math.sin(angle_offset)
        flow = np.array([radial[0] * cos_a - radial[1] * sin_a, radial[0] * sin_a + radial[1] * cos_a], dtype=float)
        flow /= (np.linalg.norm(flow) + 1e-8)
        return flow, core

    def _rand_dir(self, prev_pos: np.ndarray, prev_dir: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample the next growth direction following the smooth flow field."""
        base2d, core = self._flow_direction(prev_pos)

        if prev_dir is not None:
            try:
                prev2 = np.array([float(prev_dir[0]), float(prev_dir[1])], dtype=float)
            except Exception:
                prev2 = None
            else:
                n_prev = np.linalg.norm(prev2)
                if n_prev > 1e-8:
                    prev2 /= n_prev
                    align = float(max(-1.0, min(1.0, np.dot(prev2, base2d))))
                    history_w = 0.35 + 0.25 * max(0.0, align)
                    base2d = (1.0 - history_w) * base2d + history_w * prev2
                    base2d /= (np.linalg.norm(base2d) + 1e-8)

        # Gentle meander noise to avoid perfectly smooth arcs.
        noise = np.random.randn(2)
        noise /= (np.linalg.norm(noise) + 1e-8)
        noise_strength = np.clip(0.08 + 0.35 * (1.0 - core) * self.curl_factor, 0.0, 0.45)
        d2 = base2d * (1.0 - noise_strength) + noise * noise_strength
        d2 /= (np.linalg.norm(d2) + 1e-8)

        return np.array([d2[0], d2[1], 0.0])

    # ---- Minimal helpers for guaranteed growth ----
    def _clamp_inside(self, p: np.ndarray) -> np.ndarray:
        """Project a 3D point to lie inside the NV footprint (irregular ellipse)."""
        x, y = float(p[0]), float(p[1])
        vx, vy = x - float(self.center[0]), y - float(self.center[1])
        r = math.hypot(vx, vy)
        if r < 1e-12:
            # near center – nudge radially
            ang = float(np.random.uniform(0.0, 2.0 * np.pi))
            r_ok = 0.25 * float(self.radius)
            return np.array([self.center[0] + r_ok * math.cos(ang), self.center[1] + r_ok * math.sin(ang), p[2]], dtype=float)
        theta = math.atan2(vy, vx)
        r_max = float(self._boundary_radius(theta))
        if r <= max(self._force_inside_threshold, 0.999 * r_max):
            return p
        scale = 0.98 * r_max / max(r, 1e-8)
        return np.array([self.center[0] + vx * scale, self.center[1] + vy * scale, p[2]], dtype=float)

    def grow(self, greenhouse: Greenhouse) -> list[Node]:
        self._ensure_anchor(greenhouse)
        if self.anchor is None:
            return []
        new_nodes: list[Node] = []
        if not self.tips:
            self.tips = [self.anchor]
        z_level = float(self.anchor.position[2])
        step_len = max(0.12 * greenhouse.d, self.step_len_factor * greenhouse.d)
        budget = int(max(1, self.edges_per_iter))

        # Track NV points for spacing checks
        nv_pts: list[np.ndarray] = [np.array(t.position, dtype=float) for t in self.tips]

        def _has_clearance(p3: np.ndarray) -> bool:
            # Only keep a light self-spacing to avoid exact overlaps; never block by legacy vasculature.
            clearance = float(max(1e-4, 0.5 * self.min_clearance_nv))
            m2 = clearance * clearance
            for q in nv_pts:
                dx = float(p3[0] - q[0]); dy = float(p3[1] - q[1])
                if (dx*dx + dy*dy) <= m2:
                    return False
            return True

        def _in_dropout_core(p3: np.ndarray, thr: float = 0.35) -> bool:
            try:
                x, y = float(p3[0]), float(p3[1])
                if not greenhouse.dropouts:
                    return True
                score = max(d.inside_score(x, y) for d in greenhouse.dropouts)
                return bool(score >= float(thr))
            except Exception:
                return True

        next_tips: list[Node] = []
        grown_edges = 0
        for tip in list(self.tips):
            if budget <= 0:
                break
            # main step with retries
            prev_dir = getattr(tip, '_nv_last_dir', None)
            core_tip = float(self.inside_score(float(tip.position[0]), float(tip.position[1])))
            local_step = step_len * (0.85 + 0.35 * (1.0 - core_tip))
            local_step *= float(np.random.uniform(0.85, 1.15))
            last_child: Optional[Node] = None
            # Try progressively more permissive placements; finally force-add inside region
            for attempt in range(8):
                direction = self._rand_dir(tip.position, prev_dir)
                candidate = np.array(tip.position, dtype=float) + direction * local_step
                candidate[2] = z_level + np.random.uniform(-0.003, 0.003)
                # Keep candidate inside footprint
                candidate = self._clamp_inside(candidate)
                ok = _has_clearance(candidate) and _in_dropout_core(candidate, thr=0.35)
                if not ok and attempt >= 3:
                    # Relax spacing after a few failed tries
                    ok = _in_dropout_core(candidate, thr=0.20)
                if ok:
                    r_mm = float(np.clip(np.random.normal(self.mean_radius_mm, self.std_radius_mm), 0.0030, 0.012))
                    child = tip.tree.add_node(candidate, r_mm, tip, tip.kappa)
                    try:
                        setattr(child, 'is_neovascular', True)
                    except Exception:
                        pass
                    new_nodes.append(child)
                    next_tips.append(child)
                    nv_pts.append(np.array(child.position, dtype=float))
                    setattr(child, '_nv_last_dir', direction)
                    last_child = child
                    budget -= 1
                    grown_edges += 1
                    break
            if budget <= 0:
                break
            # optional branch
            # Strengthen branching if we struggled to grow
            dyn_branch_prob = float(self.branch_prob)
            if grown_edges < self._min_edges_guarantee:
                dyn_branch_prob = min(1.0, self.branch_prob * 1.8)
            if np.random.rand() < dyn_branch_prob:
                branch_from = last_child if last_child is not None else tip
                prev_branch_dir = getattr(branch_from, '_nv_last_dir', None)
                branch_step = local_step * float(np.random.uniform(0.75, 1.2))
                for attempt in range(6):
                    direction_b = self._rand_dir(branch_from.position, prev_branch_dir)
                    cand_b = np.array(branch_from.position, dtype=float) + direction_b * branch_step
                    cand_b[2] = z_level + np.random.uniform(-0.003, 0.003)
                    cand_b = self._clamp_inside(cand_b)
                    ok_b = _has_clearance(cand_b) and _in_dropout_core(cand_b, thr=0.35)
                    if not ok_b and attempt >= 2:
                        ok_b = _in_dropout_core(cand_b, thr=0.20)
                    if ok_b:
                        r_b = float(np.clip(np.random.normal(self.mean_radius_mm, self.std_radius_mm), 0.0030, 0.012))
                        child_b = branch_from.tree.add_node(cand_b, r_b, branch_from, branch_from.kappa)
                        try:
                            setattr(child_b, 'is_neovascular', True)
                        except Exception:
                            pass
                        new_nodes.append(child_b)
                        next_tips.append(child_b)
                        nv_pts.append(np.array(child_b.position, dtype=float))
                        setattr(child_b, '_nv_last_dir', direction_b)
                        budget -= 1
                        grown_edges += 1
                        break
            # drop reconnect/loop step from core minimal tuft (kept simple and directional)

        self.tips = next_tips if next_tips else self.tips
        return new_nodes


class GreenhouseDropout(Greenhouse):
    """
    Growth model that first generates a healthy vessel forest with the base simulator,
    then applies post-growth dropout degeneration and secondary lesions (MA, tortuosity, NV).
    """

    def __init__(self, config: dict, dropout_cfg: dict | list[dict], neovasc_cfg=None) -> None:
        super().__init__(config)
        # Build one or more dropout regions
        self.dropouts: list[_DropoutRegion] = []
        if isinstance(dropout_cfg, (list, tuple)):
            for cfg in dropout_cfg:
                if not isinstance(cfg, dict):
                    continue
                self.dropouts.append(_DropoutRegion(
                    center_xy=tuple(cfg.get("center", (0.5, 0.5))),
                    radius_xy=float(cfg.get("radius", 0.12)),
                    drop_strength=float(cfg.get("strength", 1.0)),
                    irregularity_amp=float(cfg.get("irregularity_amp", 0.22)),
                    harmonics=tuple(cfg.get("harmonics", (2, 3, 5))),
                    ellipticity=tuple(cfg.get("ellipticity", (1.0, 1.25))),
                    gradient_alpha=float(cfg.get("gradient_alpha", 2.0)),
                    noise_gain=float(cfg.get("noise_gain", 0.35)),
                ))
            cfg0 = dropout_cfg[0] if (len(dropout_cfg) > 0 and isinstance(dropout_cfg[0], dict)) else {}
        else:
            cfg = dropout_cfg if isinstance(dropout_cfg, dict) else {}
            self.dropouts.append(_DropoutRegion(
                center_xy=tuple(cfg.get("center", (0.5, 0.5))),
                radius_xy=float(cfg.get("radius", 0.12)),
                drop_strength=float(cfg.get("strength", 1.0)),
                irregularity_amp=float(cfg.get("irregularity_amp", 0.22)),
                harmonics=tuple(cfg.get("harmonics", (2, 3, 5))),
                ellipticity=tuple(cfg.get("ellipticity", (1.0, 1.25))),
                gradient_alpha=float(cfg.get("gradient_alpha", 2.0)),
                noise_gain=float(cfg.get("noise_gain", 0.35)),
            ))
            cfg0 = cfg
        # Primary region for legacy access
        self.dropout = max(self.dropouts, key=lambda d: d.r0)
        self.cap_radius_thresh_mm = float(cfg0.get("cap_radius_thresh_mm", 0.010))
        self.regress_threshold = float(cfg0.get("regress_threshold", 0.35))
        # Border behavior and perivascular sparing
        self.ring_tangential_bias = float(cfg0.get("ring_tangential_bias", 0.45))
        band = cfg0.get("ring_band") or (0.35, 0.7)
        self.ring_band = (float(band[0]), float(band[1]))
        self.sparing_large_radius_mm = float(cfg0.get("sparing_large_radius_mm", 0.02))
        self.sparing_factor = float(cfg0.get("sparing_factor", 0.5))  # reduces suppression/regression near trunks
        # Radius clamps (optional). Default: no lower clamp; cap at ~2.5x base radius.
        rmn = cfg0.get("r_min_mm", None)
        rmx = cfg0.get("r_max_mm", None)
        self.r_min_mm = None if (rmn is None or str(rmn).lower() == "none") else float(rmn)
        # Cap new segment radius to avoid unrealistic over-thickening; relative to base self.r if not provided
        self.r_max_mm = float(rmx) if (rmx is not None and str(rmx).lower() != "none") else float(self.r * 2.5)
        # Post-growth degeneration controls
        try:
            self.dropout_regression_passes: int = max(1, int(cfg0.get("regression_passes", 1)))
        except Exception:
            self.dropout_regression_passes = 1

        # Fraction of leaf vessels to prune inside dropout (sampled per pass, smallest radii first)
        # Map global dropout strength -> removal fraction so that:
        #   strength=0.0 => 0% removed; strength=1.0 => ~100% removed in effected area.
        # Clamp to avoid removing protected core minimum entirely.
        self.degeneration_fraction_range: tuple[float, float] = (0.25, 0.60)
        try:
            if cfg0.get("degeneration_fraction") is not None:
                val = float(cfg0.get("degeneration_fraction"))
                val = max(0.0, min(1.0, val))
                self.degeneration_fraction_range = (val, val)
            elif cfg0.get("degeneration_fraction_range") is not None:
                dfr = cfg0.get("degeneration_fraction_range")
                if isinstance(dfr, (list, tuple)) and len(dfr) == 2:
                    lo = float(dfr[0]); hi = float(dfr[1])
                    if hi < lo:
                        lo, hi = hi, lo
                    lo = max(0.0, min(1.0, lo))
                    hi = max(0.0, min(1.0, hi))
                    self.degeneration_fraction_range = (lo, hi)
        except Exception:
            pass

        # If no explicit degeneration fraction provided, derive it from configured dropout strengths
        try:
            if (cfg0.get("degeneration_fraction") is None) and (cfg0.get("degeneration_fraction_range") is None):
                s_vals = [max(0.0, min(1.0, float(r.drop_strength))) for r in self.dropouts] or [0.0]
                s_global = float(max(s_vals))
                # Use full strength value - no artificial caps
                # When strength = 1.0, all vessels should be removed
                s_clip = float(max(0.0, min(1.0, s_global)))
                self.degeneration_fraction_range = (s_clip, s_clip)
                # For very high strengths, lower the regression threshold so the whole region participates
                if s_global >= 0.9:
                    # Allow near-core vessels to participate but keep a small cutoff so non-dropout trunks survive.
                    base_thresh = float(cfg0.get("regress_threshold", 0.35))
                    self.regress_threshold = min(base_thresh, 0.08)
        except Exception:
            pass

        try:
            self.degeneration_core_bias: float = float(cfg0.get("degeneration_core_bias", 1.5))
        except Exception:
            self.degeneration_core_bias = 1.5
        try:
            min_keep = float(cfg0.get("degeneration_min_keep_frac", 0.1))
            self.degeneration_min_keep_frac: float = max(0.0, min(0.9, float(min_keep)))
        except Exception:
            self.degeneration_min_keep_frac = 0.1
        try:
            self.degeneration_core_keep_threshold: float = float(cfg0.get("degeneration_core_keep_threshold", 0.82))
        except Exception:
            self.degeneration_core_keep_threshold = 0.82
        try:
            core_keep_frac = float(cfg0.get("degeneration_core_keep_frac", 0.10))
            self.degeneration_core_keep_frac: float = max(0.0, min(0.9, core_keep_frac))
        except Exception:
            self.degeneration_core_keep_frac = 0.10

        # Vessel elongation and dilation in dropout areas
        try:
            elongation_range = cfg0.get("vessel_elongation_range", [1.0, 1.5])
            self.vessel_elongation_range: tuple[float, float] = (float(elongation_range[0]), float(elongation_range[1]))
        except Exception:
            self.vessel_elongation_range = (1.0, 1.5)
        try:
            dilation_range = cfg0.get("vessel_dilation_range", [1.0, 1.2])
            self.vessel_dilation_range: tuple[float, float] = (float(dilation_range[0]), float(dilation_range[1]))
        except Exception:
            self.vessel_dilation_range = (1.0, 1.2)

        # NV growth controls
        # We default to overlay-only NV (no growth-time modification of the graph).
        # If a caller explicitly enables growth via cfg0['nv_growth_enable'] = True,
        # the older sprout simulation can be triggered, but this is off by default.
        try:
            self.nv_growth_enable: bool = bool(cfg0.get("nv_growth_enable", False))
        except Exception:
            self.nv_growth_enable = False
        try:
            self.nv_post_iters: int = int(cfg0.get("nv_post_iters", 5))
        except Exception:
            self.nv_post_iters = 5
        # NV regions (optional)
        self.nv_regions: list[NVRegion] = []
        if neovasc_cfg is not None:
            if isinstance(neovasc_cfg, dict):
                self.nv_regions = [NVRegion(
                    center_xy=tuple(neovasc_cfg.get("center", (0.6, 0.5))),
                    radius_xy=float(neovasc_cfg.get("radius", 0.08)),
                    step_len_factor=float(neovasc_cfg.get("step_len_factor", 0.45)),
                    mean_radius_mm=float(neovasc_cfg.get("mean_radius_mm", 0.007)),
                    std_radius_mm=float(neovasc_cfg.get("std_radius_mm", 0.002)),
                    branch_prob=float(neovasc_cfg.get("branch_prob", 0.35)),
                    curl_factor=float(neovasc_cfg.get("curl_factor", 0.5)),
                    edges_per_iter=int(neovasc_cfg.get("edges_per_iter", 20)),
                    irregularity_amp=float(neovasc_cfg.get("irregularity_amp", 0.12)),
                    harmonics=tuple(neovasc_cfg.get("harmonics", (3, 5, 7))),
                    ellipticity=tuple(neovasc_cfg.get("ellipticity", (1.0, 1.35))),
                    gradient_alpha=float(neovasc_cfg.get("gradient_alpha", 1.6)),
                    noise_gain=float(neovasc_cfg.get("noise_gain", 0.25)),
                    connect_prob=float(neovasc_cfg.get("connect_prob", 0.12)),
                    connect_radius_norm=float(neovasc_cfg.get("connect_radius_norm", 0.025)),
                    border_center=(self.dropout.cx, self.dropout.cy),
                    border_radius=self.dropout.r0,
                    border_band=float(neovasc_cfg.get("border_band_norm", 0.06)),
                    border_bias=float(neovasc_cfg.get("border_bias", 0.5)),
                    outward_bias=float(neovasc_cfg.get("outward_bias", 0.85)),
                    init_spokes=int(neovasc_cfg.get("init_spokes", 6)),
                    spoke_jitter=float(neovasc_cfg.get("spoke_jitter", 0.25)),
                    min_clearance_vessel_norm=float(neovasc_cfg.get("min_clearance_vessel_norm", 0.012)),
                    min_clearance_nv_norm=float(neovasc_cfg.get("min_clearance_nv_norm", 0.010)),
                )]
                # Allow NV post iterations to be overridden by neovasc config
                try:
                    if neovasc_cfg.get("post_iters") is not None:
                        self.nv_post_iters = int(neovasc_cfg.get("post_iters"))
                    elif neovasc_cfg.get("nv_post_iters") is not None:
                        self.nv_post_iters = int(neovasc_cfg.get("nv_post_iters"))
                except Exception:
                    pass
            elif isinstance(neovasc_cfg, (list, tuple)):
                self.nv_regions = [NVRegion(
                    center_xy=tuple(cfg.get("center", (0.6, 0.5))),
                    radius_xy=float(cfg.get("radius", 0.08)),
                    step_len_factor=float(cfg.get("step_len_factor", 0.45)),
                    mean_radius_mm=float(cfg.get("mean_radius_mm", 0.007)),
                    std_radius_mm=float(cfg.get("std_radius_mm", 0.002)),
                    branch_prob=float(cfg.get("branch_prob", 0.35)),
                    curl_factor=float(cfg.get("curl_factor", 0.5)),
                    edges_per_iter=int(cfg.get("edges_per_iter", 20)),
                    irregularity_amp=float(cfg.get("irregularity_amp", 0.12)),
                    harmonics=tuple(cfg.get("harmonics", (3, 5, 7))),
                    ellipticity=tuple(cfg.get("ellipticity", (1.0, 1.35))),
                    gradient_alpha=float(cfg.get("gradient_alpha", 1.6)),
                    noise_gain=float(cfg.get("noise_gain", 0.25)),
                    connect_prob=float(cfg.get("connect_prob", 0.12)),
                    connect_radius_norm=float(cfg.get("connect_radius_norm", 0.025)),
                    border_center=(self.dropout.cx, self.dropout.cy),
                    border_radius=self.dropout.r0,
                    border_band=float(cfg.get("border_band_norm", 0.06)),
                    border_bias=float(cfg.get("border_bias", 0.5)),
                    outward_bias=float(cfg.get("outward_bias", 0.85)),
                    init_spokes=int(cfg.get("init_spokes", 6)),
                    spoke_jitter=float(cfg.get("spoke_jitter", 0.25)),
                    min_clearance_vessel_norm=float(cfg.get("min_clearance_vessel_norm", 0.012)),
                    min_clearance_nv_norm=float(cfg.get("min_clearance_nv_norm", 0.010)),
                ) for cfg in neovasc_cfg]
        else:
            self.nv_regions = []

    # NV placement now follows sampling strategy (dropout border anchoring) without extra relocation.

        # Additional pathology knobs (microaneurysms, dilation, tortuosity)
        self.ma_prob = float(cfg0.get("ma_prob", 0.0))
        mr = cfg0.get("ma_radius_mm_range", [0.010, 0.018])
        if isinstance(mr, (list, tuple)) and len(mr) == 2:
            self.ma_radius_mm_range = (float(mr[0]), float(mr[1]))
        else:
            self.ma_radius_mm_range = (0.010, 0.018)
        self.ma_len_factor = float(cfg0.get("ma_len_factor", 0.4))
        self.ma_border_bias = float(cfg0.get("ma_border_bias", 0.5))
        mb = cfg0.get("ma_band", [0.25, 0.85])
        self.ma_band = (float(mb[0]), float(mb[1]))
        # Optional coupling: scale MA probability with dropout strength (0..1)
        # p_ma *= (1 + gain * strength_local)
        try:
            self.ma_prob_strength_gain = float(cfg0.get("ma_prob_strength_gain", 0.0))
        except Exception:
            self.ma_prob_strength_gain = 0.0
        # Visualization-only shape randomness for MA overlays (0..1)
        try:
            self.ma_shape_randomness = float(cfg0.get("ma_shape_randomness", 0.0))
        except Exception:
            self.ma_shape_randomness = 0.0
        # Restrict MA placement to border of dropout and to capillary-scale parents
        self.ma_only_near_dropout = bool(cfg0.get("ma_only_near_dropout", True))
        # Optional: MA-specific radius clamps (do not affect regular segments)
        _ma_rmn = cfg0.get("ma_r_min_mm", None)
        _ma_rmx = cfg0.get("ma_r_max_mm", None)
        try:
            self.ma_r_min_mm = None if (_ma_rmn is None or str(_ma_rmn).lower() == "none") else float(_ma_rmn)
        except Exception:
            self.ma_r_min_mm = None
        try:
            self.ma_r_max_mm = None if (_ma_rmx is None or str(_ma_rmx).lower() == "none") else float(_ma_rmx)
        except Exception:
            self.ma_r_max_mm = None
        # Note: parent radius limit for MA placement removed; allow placement regardless of parent size

        ds = cfg0.get("dilation_scale_range", [1.0, 1.0])
        if isinstance(ds, (list, tuple)) and len(ds) == 2:
            self.dilation_scale_range = (float(ds[0]), float(ds[1]))
        else:
            self.dilation_scale_range = (1.0, 1.0)
        db = cfg0.get("dilation_band", [0.30, 0.75])
        self.dilation_band = (float(db[0]), float(db[1]))
        self.dilation_where = str(cfg0.get("dilation_where", "border")).lower()  # border|inside|all|none

        self.tortuosity_gain = float(cfg0.get("tortuosity_gain", 0.0))  # 0..1
        tb = cfg0.get("tortuosity_band", [0.20, 0.80])
        self.tortuosity_band = (float(tb[0]), float(tb[1]))

    # Utilities for additional pathology
    @staticmethod
    def _smoothstep(z: float) -> float:
        z = np.clip(z, 0.0, 1.0)
        return float(z * z * (3.0 - 2.0 * z))

    @staticmethod
    def _in_band(core: float, band: tuple[float, float]) -> bool:
        lo, hi = float(band[0]), float(band[1])
        return (core > lo) and (core < hi)

    def _radius_scale_at(self, core: float) -> float:
        lo, hi = self.dilation_scale_range
        if lo == 1.0 and hi == 1.0:
            return 1.0
        if self.dilation_where not in ("border", "inside", "all"):
            return 1.0
        w = 0.0
        if self.dilation_where == "border" and self._in_band(core, self.dilation_band):
            z = (core - self.dilation_band[0]) / max(1e-6, (self.dilation_band[1] - self.dilation_band[0]))
            w = self._smoothstep(z)
        elif self.dilation_where == "inside" and core >= self.dilation_band[1]:
            w = 1.0
        elif self.dilation_where == "all":
            w = 1.0
        return float(lo + (hi - lo) * np.clip(w, 0.0, 1.0))

    def _apply_radius_dilation_posthoc(self) -> None:
        """Scale vessel radii near dropout borders after healthy growth."""
        if not self.dropouts:
            return
        lo, hi = self.dilation_scale_range
        if abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9:
            return
        forests = [self.arterial_forest]
        if self.venous_forest is not None:
            forests.append(self.venous_forest)
        for forest in forests:
            for tree in forest.get_trees():
                for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    try:
                        core = max(d.inside_score(float(node.position[0]), float(node.position[1])) for d in self.dropouts)
                    except Exception:
                        continue
                    scale = self._radius_scale_at(core)
                    if abs(scale - 1.0) < 1e-6:
                        continue
                    radius_new = float(max(0.0, node.radius * scale))
                    if self.r_max_mm is not None:
                        radius_new = float(min(radius_new, self.r_max_mm))
                    if self.r_min_mm is not None:
                        radius_new = float(max(radius_new, self.r_min_mm))
                    if abs(radius_new - node.radius) < 1e-6:
                        continue
                    node.radius = radius_new
                    try:
                        node.optimize_edge_radius_to_root()
                    except Exception:
                        pass

    def _apply_vessel_elongation_in_dropout(self) -> None:
        """Apply random elongation to vessels in dropout areas."""
        if not self.dropouts:
            return
        lo, hi = self.vessel_elongation_range
        if abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9:
            return

        forests = [self.arterial_forest]
        if self.venous_forest is not None:
            forests.append(self.venous_forest)

        for forest in forests:
            for tree in forest.get_trees():
                for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    try:
                        x = float(node.position[0])
                        y = float(node.position[1])
                        core = max(d.inside_score(x, y) for d in self.dropouts)
                    except Exception:
                        continue

                    # Only apply elongation if inside dropout region
                    if core < 0.3:  # Threshold for being inside dropout
                        continue

                    # Sample elongation factor based on position in dropout
                    elongation_factor = random.uniform(lo, hi)

                    # Apply elongation by moving node position along the vessel direction
                    if hasattr(node, 'parent') and node.parent is not None:
                        try:
                            parent_pos = node.parent.position
                            # Calculate direction vector from parent to node
                            dx = x - float(parent_pos[0])
                            dy = y - float(parent_pos[1])
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 1e-6:
                                # Elongate by moving node further along the direction
                                # Scale by core to make elongation stronger at dropout center
                                effective_elongation = 1.0 + (elongation_factor - 1.0) * core
                                new_x = float(parent_pos[0]) + dx * effective_elongation
                                new_y = float(parent_pos[1]) + dy * effective_elongation
                                node.position = np.array([new_x, new_y], dtype=float)
                        except Exception:
                            pass

    def _apply_vessel_dilation_in_dropout(self) -> None:
        """Apply random dilation to vessel radii in dropout areas."""
        if not self.dropouts:
            return
        lo, hi = self.vessel_dilation_range
        if abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9:
            return

        forests = [self.arterial_forest]
        if self.venous_forest is not None:
            forests.append(self.venous_forest)

        for forest in forests:
            for tree in forest.get_trees():
                for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    try:
                        x = float(node.position[0])
                        y = float(node.position[1])
                        core = max(d.inside_score(x, y) for d in self.dropouts)
                    except Exception:
                        continue

                    # Only apply dilation if inside dropout region
                    if core < 0.3:  # Threshold for being inside dropout
                        continue

                    # Sample dilation factor based on position in dropout
                    # Stronger dilation closer to dropout center
                    dilation_factor = lo + (hi - lo) * core

                    try:
                        radius_new = float(node.radius * dilation_factor)
                        if self.r_max_mm is not None:
                            radius_new = float(min(radius_new, self.r_max_mm))
                        if self.r_min_mm is not None:
                            radius_new = float(max(radius_new, self.r_min_mm))
                        node.radius = radius_new

                        # Optimize edge radii
                        if hasattr(node, 'optimize_edge_radius_to_root'):
                            node.optimize_edge_radius_to_root()
                    except Exception:
                        pass

    def _grow_neovascularization_posthoc(self) -> None:
        """Legacy growth-time NV sprouting (now disabled by default).

        Overlay-only NV rendering is the default. Set nv_growth_enable=True in
        config to opt back into graph modification if needed.
        """
        if not self.nv_growth_enable:
            return
        if not self.nv_regions:
            return
        try:
            for _ in range(int(max(1, self.nv_post_iters))):
                nv_new: list[Node] = []
                for region in self.nv_regions:
                    nv_new.extend(region.grow(self))
                if not nv_new:
                    continue
                self.art_node_mesh.extend(nv_new)
                self.active_art_node_mesh.extend(nv_new)
                to_remove: set[tuple[float, ...]] = set()
                to_add: set[tuple[float, ...]] = set()
                for node in nv_new:
                    for oxy in self.oxy_mesh.find_elements_in_distance(node.position, self.eps_k):
                        to_remove.add(oxy)
                        if self.venous_forest is not None:
                            to_add.add(tuple(oxy))
                if to_remove:
                    self.oxy_mesh.delete_all(list(to_remove))
                if self.venous_forest is not None and to_add:
                    self.co2_mesh.extend(list(to_add))
        except Exception:
            pass

    def _apply_post_growth_pathology(self) -> None:
        """Execute dropout degeneration and secondary pathologies after healthy growth."""
        if self.dropouts:
            passes = max(1, int(getattr(self, "dropout_regression_passes", 1)))
            for _ in range(passes):
                self._degenerate_fraction_in_region(self.arterial_forest, self.art_node_mesh, self.active_art_node_mesh)
                if self.venous_forest is not None:
                    self._degenerate_fraction_in_region(self.venous_forest, self.ven_node_mesh, self.active_ven_node_mesh)
            try:
                self._apply_vessel_elongation_in_dropout()
            except Exception:
                pass
            try:
                self._apply_vessel_dilation_in_dropout()
            except Exception:
                pass
            try:
                self._apply_radius_dilation_posthoc()
            except Exception:
                pass
            try:
                self._apply_tortuosity_posthoc()
            except Exception:
                pass
            try:
                self._add_microaneurysms_posthoc()
            except Exception:
                pass
        # Place NV seeds along the border of the largest non-FAZ dropout in locally dark areas
        # When overlay-only NV is used (default), we still position regions for visualization
        # but we do not grow/attach new edges to the graph unless nv_growth_enable=True.
        try:
            self._place_nv_regions_on_dropout_border_dark()
        except Exception:
            pass
        self._grow_neovascularization_posthoc()

    # -----------------------------
    # NV placement helpers
    # -----------------------------
    def _count_nodes_within(self, x: float, y: float, r_norm: float) -> int:
        try:
            center3 = np.array([x, y, self.simspace.shape[2] * 0.5], dtype=float)
            # Search in arterial and venous meshes if present; approximate by arterial first
            cnt = 0
            for mesh in [self.art_node_mesh, self.ven_node_mesh if self.venous_forest is not None else None]:
                if mesh is None:
                    continue
                elems = list(mesh.find_elements_in_distance(center3, r_norm))
                cnt += len(elems)
                if cnt > 0:
                    return cnt
            return cnt
        except Exception:
            return 0

    def _place_nv_regions_on_dropout_border_dark(self) -> None:
        """
        Place each NV region at the geometric center of the largest non‑FAZ dropout.
        This guarantees sprouts originate from the middle of the ischemic core.
        """
        if not self.nv_regions:
            return
        if not self.dropouts:
            return
        
        # 获取FAZ（中心凹无血管区）的信息
        try:
            fx, fy = float(getattr(self, 'FAZ_center', (0.5, 0.5))[0]), float(getattr(self, 'FAZ_center', (0.5, 0.5))[1])
            fr = float(getattr(self, 'FAZ_radius', 0.06))
        except Exception:
            fx, fy, fr = 0.5, 0.5, 0.06
        
        # 选择不与FAZ核心重叠的dropout区域
        valid = []
        for d in self.dropouts:
            try:
                cx, cy, r0 = float(d.cx), float(d.cy), float(d.r0)
                # 保持与FAZ核心足够距离
                dist_f = math.hypot(cx - fx, cy - fy)
                if dist_f <= (1.2 * fr):
                    continue
                valid.append(d)
            except Exception:
                continue
        
        if not valid:
            valid = list(self.dropouts)
        
        # Choose the largest dropout region as the NV target
        target = max(valid, key=lambda t: float(getattr(t, 'r0', 0.0)))
        cx0, cy0, r0 = float(target.cx), float(target.cy), float(target.r0)

        # Place every NV center exactly at dropout center and adjust radius modestly
        for region in self.nv_regions:
            region.center = (
                float(np.clip(cx0, 0.05, 0.95)),
                float(np.clip(cy0, 0.05, 0.95)),
            )
            # Cover most of dropout core but keep a natural transition near border
            region.radius = float(max(region.radius, 0.6 * r0))
            region.radius = float(min(region.radius, 0.85 * r0))
            # Reset anchor to start growth from the new center
            region.anchor = None

    # (NV dark-area relocation helpers removed per edge-anchoring requirement)

    def _degenerate_fraction_in_region(self, forest: Forest, node_mesh: NodeKdTree, active_node_mesh: NodeKdTree) -> None:
        lo, hi = self.degeneration_fraction_range
        if hi <= 0.0:
            return
        # Global dropout strength sampled from config controls overall removal percentage.
        # Treat strength as direct removal fraction so artists can reason about percentage.
        avg_strength = float(np.mean([d.drop_strength for d in self.dropouts]))
        high_strength_threshold = 0.85  # When strength > 0.85, start removing large vessels
        complete_removal_threshold = 0.95
        force_full_purge = avg_strength >= complete_removal_threshold
        target_frac = float(np.clip(avg_strength, lo, hi))
        if force_full_purge:
            target_frac = 1.0
        if target_frac <= 0.0:
            return

        # Scale keep fractions down as strength rises so full-strength purges are allowed.
        base_min_keep = float(self.degeneration_min_keep_frac)
        min_keep_frac_effective = base_min_keep * max(0.0, 1.0 - target_frac)
        base_core_keep = float(self.degeneration_core_keep_frac)
        core_keep_frac_effective = base_core_keep * max(0.0, 1.0 - target_frac)

        # Lower core keep threshold slightly when strength is high so deep-core vessels can be culled.
        core_keep_threshold = float(self.degeneration_core_keep_threshold)
        if target_frac > 0.75:
            core_keep_threshold *= max(0.35, 1.0 - 0.5 * (target_frac - 0.75) / 0.25)

        while True:
            candidates: list[tuple[float, float, Node]] = []
            for tree in forest.get_trees():
                for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                    if not node.is_leaf:
                        continue
                    try:
                        x = float(node.position[0])
                        y = float(node.position[1])
                        core = max(d.inside_score(x, y) for d in self.dropouts)
                    except Exception:
                        continue
                    if core < self.regress_threshold:
                        continue
                    try:
                        radius = float(getattr(node, "radius", 0.0))
                    except Exception:
                        radius = 0.0
                    candidates.append((radius, core, node))

            if not candidates:
                return

            radii = np.array([max(1e-6, float(r)) for r, _, _ in candidates], dtype=float)
            cores = np.array([min(1.0, max(0.0, float(c))) for _, c, _ in candidates], dtype=float)
            try:
                edge = np.power(np.maximum(1e-3, 1.0 - cores), float(self.degeneration_core_bias))

                # When strength is high, modify weighting to include large vessels
                # But still prioritize small vessels (优先小血管)
                if avg_strength > high_strength_threshold:
                    # Blend between pure small-vessel preference and more uniform removal
                    # strength_factor: 0.85->0.0, 1.0->1.0
                    strength_factor = min(1.0, (avg_strength - high_strength_threshold) / (1.0 - high_strength_threshold))
                    # At low strength_factor, weights = edge/radii (prefer small vessels)
                    # At high strength_factor, weights = edge/radii^0.3 (less preference for small)
                    radius_exponent = 1.0 - 0.7 * strength_factor  # 1.0 -> 0.3
                    weights = edge / np.power(radii, radius_exponent)
                else:
                    # Normal behavior: strongly prefer small vessels
                    weights = edge / radii

                total_w = float(np.sum(weights))
                if total_w <= 0.0 or not np.isfinite(total_w):
                    weights = None
                else:
                    weights = (weights / total_w).astype(float)
            except Exception:
                weights = None

            # When strength is very high (>= 0.95), remove ALL vessels - no caps or limits
            if force_full_purge:
                sel_idx = np.arange(len(candidates), dtype=int)
            else:
                remove_count = int(round(len(candidates) * target_frac))
                remove_count = min(len(candidates), max(1, remove_count))
                min_keep = int(math.ceil(len(candidates) * min_keep_frac_effective))
                if min_keep >= len(candidates):
                    return
                remove_count = min(remove_count, len(candidates) - min_keep)
                if remove_count <= 0:
                    return

                # Split into core-protected and others, enforce minimum keep in core
                protect_mask = cores >= core_keep_threshold
                idx_protect = np.nonzero(protect_mask)[0]
                idx_other = np.nonzero(~protect_mask)[0]
                keep_core_target = int(math.ceil(len(idx_protect) * core_keep_frac_effective)) if len(idx_protect) > 0 else 0
                max_rm_core = max(0, len(idx_protect) - keep_core_target)
                # Decide removal counts per group
                rm_other = min(remove_count, len(idx_other))
                rm_core = min(remove_count - rm_other, max_rm_core)
                if rm_core < 0:
                    rm_core = 0
                # Sample from each group with weights
                sel_core = np.array([], dtype=int)
                sel_other = np.array([], dtype=int)
                if rm_core > 0 and len(idx_protect) > 0:
                    pw = weights[idx_protect] if weights is not None else None
                    if pw is not None:
                        pw = (pw / float(np.sum(pw))).astype(float)
                    sel_core = np.random.default_rng().choice(idx_protect, size=rm_core, replace=False, p=pw)
                if rm_other > 0 and len(idx_other) > 0:
                    ow = weights[idx_other] if weights is not None else None
                    if ow is not None:
                        ow = (ow / float(np.sum(ow))).astype(float)
                    sel_other = np.random.default_rng().choice(idx_other, size=rm_other, replace=False, p=ow)
                sel_idx = np.concatenate([sel_core, sel_other]).astype(int)

            to_remove_nodes = [candidates[int(i)][2] for i in np.atleast_1d(sel_idx)]
            if len(to_remove_nodes) == 0:
                return

            removed_any = False
            for n in to_remove_nodes:
                try:
                    node_mesh.delete(n)
                    removed_any = True
                except Exception:
                    pass
                try:
                    active_node_mesh.delete(n)
                except Exception:
                    pass
                try:
                    parent = n.parent if hasattr(n, "parent") else None
                    n.parent = None
                    if parent is not None and hasattr(parent, "_update_node_status"):
                        parent._update_node_status()
                except Exception:
                    pass

            # When forcing full purge, iterate until no nodes remain in the dropout core
            if not force_full_purge or not removed_any:
                return


    def develop_forest(self):
        """Run healthy growth with the base simulator, then apply dropout pathology."""
        super().develop_forest()
        try:
            self._healthy_metrics = {
                "arterial_nodes": len(self.art_node_mesh.get_all_elements()),
                "venous_nodes": len(self.ven_node_mesh.get_all_elements()) if hasattr(self, "ven_node_mesh") else None,
            }
        except Exception:
            self._healthy_metrics = {}
        self._apply_post_growth_pathology()

    def _apply_tortuosity_posthoc(self) -> None:
        """Apply a small position jitter perpendicular to local direction for nodes in tortuosity band."""
        import numpy as _np
        if self.tortuosity_gain <= 0.0:
            return
        amp = float(self.tortuosity_gain) * 0.35 * float(self.d)
        lo, hi = self.tortuosity_band
        for tree in self.arterial_forest.get_trees():
            for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                p = node.position.copy()
                x, y = float(p[0]), float(p[1])
                core = max([d.inside_score(x, y) for d in self.dropouts])
                if not self._in_band(core, (lo, hi)):
                    continue
                # Approximate local tangent by proximal->node vector
                try:
                    parent = node.get_proximal_node()
                    if parent is None:
                        continue
                    v = p - parent.position
                    v[:2] /= (float(_np.linalg.norm(v[:2])) + 1e-8)
                    # Perpendicular
                    perp = _np.array([-v[1], v[0], 0.0], dtype=float)
                    jitter = (2.0 * _np.random.rand() - 1.0) * amp
                    p2 = p + perp * jitter
                    # Keep within [0,1] bounds and same z
                    p2[0] = float(_np.clip(p2[0], 0.0, 1.0))
                    p2[1] = float(_np.clip(p2[1], 0.0, 1.0))
                    node.position = p2
                except Exception:
                    continue

    def _add_microaneurysms_posthoc(self) -> None:
        """Spawn microaneurysms after regression near the dropout border on small parents."""
        import numpy as _np
        if self.ma_prob <= 0.0:
            return
        for tree in self.arterial_forest.get_trees():
            for node in tree.get_tree_iterator(exclude_root=True, only_active=False):
                try:
                    parent = node.get_proximal_node()
                    if parent is None:
                        continue
                    # Parent radius constraint removed; consider all parents
                    x, y = float(node.position[0]), float(node.position[1])
                    core = max([d.inside_score(x, y) for d in self.dropouts])
                    if self.ma_only_near_dropout and not self._in_band(core, self.ma_band):
                        continue
                    p = self.ma_prob
                    # Prefer larger dropout regions: scale probability by region radius relative to max
                    try:
                        if self.dropouts:
                            # choose the dropout contributing most to core at this point
                            cores = [d.inside_score(x, y) for d in self.dropouts]
                            if cores:
                                k = int(_np.argmax(_np.asarray(cores)))
                                r0 = float(self.dropouts[k].r0)
                                rmax = float(max(d.r0 for d in self.dropouts))
                                if rmax > 0:
                                    scale = 0.5 + 0.5 * (r0 / rmax)
                                    p *= float(scale)
                                # Couple MA spawn probability to local dropout strength if configured
                                try:
                                    gain = float(getattr(self, 'ma_prob_strength_gain', 0.0))
                                except Exception:
                                    gain = 0.0
                                if gain != 0.0:
                                    try:
                                        s_local = float(getattr(self.dropouts[k], 'drop_strength', 0.0))
                                        s_local = max(0.0, min(1.0, s_local))
                                        p *= (1.0 + gain * s_local)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    if self._in_band(core, self.ma_band):
                        p = p * (1.0 + self.ma_border_bias)
                    if _np.random.rand() >= p:
                        continue
                    # Perpendicular short protrusion with larger radius
                    # Tangent from parent->node; perpendicular for MA
                    v = node.position - parent.position
                    v[:2] /= (float(_np.linalg.norm(v[:2])) + 1e-8)
                    d2 = _np.array([-v[1], v[0], 0.0])
                    d2 /= (float(_np.linalg.norm(d2)) + 1e-8)
                    length = max(0.1 * self.d, float(self.ma_len_factor) * self.d)
                    p_ma = _np.real(node.position + d2 * length)
                    r_ma = float(_np.random.uniform(self.ma_radius_mm_range[0], self.ma_radius_mm_range[1]))
                    if self.ma_r_max_mm is not None:
                        r_ma = float(min(r_ma, self.ma_r_max_mm))
                    if self.ma_r_min_mm is not None:
                        r_ma = float(max(r_ma, self.ma_r_min_mm))
                    elif self.r_min_mm is not None:
                        r_ma = float(max(r_ma, self.r_min_mm))
                    ma_node = node.tree.add_node(p_ma, r_ma, node, self.kappa)
                    setattr(ma_node, 'is_microaneurysm', True)
                    # Optionally add a small irregular cluster around the MA to create varied shapes
                    try:
                        n_extra = int(_np.random.randint(0, 3))  # 0..2 extra nodes
                        for _ in range(n_extra):
                            th = _np.random.uniform(0.0, 2.0 * _np.pi)
                            rad = _np.random.uniform(0.2, 0.5) * length
                            jitter = _np.array([rad * _np.cos(th), rad * _np.sin(th), 0.0])
                            p_ex = _np.real(p_ma + jitter)
                            r_ex = float(_np.random.uniform(self.ma_radius_mm_range[0], self.ma_radius_mm_range[1])) * _np.random.uniform(0.65, 0.95)
                            if self.ma_r_max_mm is not None:
                                r_ex = float(min(r_ex, self.ma_r_max_mm))
                            if self.ma_r_min_mm is not None:
                                r_ex = float(max(r_ex, self.ma_r_min_mm))
                            elif self.r_min_mm is not None:
                                r_ex = float(max(r_ex, self.r_min_mm))
                            child = node.tree.add_node(p_ex, r_ex, ma_node, self.kappa)
                            setattr(child, 'is_microaneurysm', True)
                    except Exception:
                        pass
                except Exception:
                    continue

    # -----------------------------
    # Rendering helpers for pathology masks (2D)
    # -----------------------------
    def render_pathology_masks(self, image_resolution_xy: Sequence[int], mip_axis: int = 2, dropout_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Render binary masks (uint8 0/255) for dropout and neovascular regions aligned with rasterized images.

        Coordinate conventions are matched to tree2img.rasterize_forest:
          - `image_resolution_xy` is (no_pixels_x, no_pixels_y). This is the same tuple you pass as
            `image_resolution` to rasterize_forest.
          - For the typical case mip_axis=2 (z-projection), rasterize_forest plots edges using
            (x_display, y_display) = (node[1], node[0]) and inverts the y-axis.
            To align, we evaluate pathology fields in model space at:
               x_model = y_display = (row + 0.5) / H
               y_model = x_display = (col + 0.5) / W
          - Other MIP axes are not supported here.
        """
        if len(image_resolution_xy) != 2:
            raise ValueError("image_resolution_xy must be (no_pixels_x, no_pixels_y)")
        if mip_axis != 2:
            raise NotImplementedError("render_pathology_masks currently supports mip_axis=2 only")

        W = int(image_resolution_xy[0])  # no_pixels_x
        H = int(image_resolution_xy[1])  # no_pixels_y
        xs = (np.arange(W) + 0.5) / max(W, 1)  # x_display in [0,1]
        ys = (np.arange(H) + 0.5) / max(H, 1)  # y_display in [0,1]
        Xd, Yd = np.meshgrid(xs, ys)  # shapes (H, W)

        # Map display coords to model (x,y) as used during growth
        Xm = Yd  # x_model corresponds to y_display (row direction)
        Ym = Xd  # y_model corresponds to x_display (col direction)

        # Dropout mask (thresholded inside score)
        # Union across all dropout regions
        acc = np.zeros((H, W), dtype=np.float32)
        for reg in self.dropouts:
            acc = np.maximum(acc, reg.inside_field(Xm, Ym).astype(np.float32))
        dscore = acc
        dmask = (dscore >= float(dropout_threshold)).astype(np.uint8) * 255

        # NV mask: union across all NV regions
        if not self.nv_regions:
            nmask = np.zeros((H, W), dtype=np.uint8)
        else:
            acc = np.zeros((H, W), dtype=np.float32)
            for reg in self.nv_regions:
                acc = np.maximum(acc, reg.inside_field(Xm, Ym).astype(np.float32))
            nmask = (acc > 0.0).astype(np.uint8) * 255

        return dmask, nmask
