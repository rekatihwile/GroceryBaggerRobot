from __future__ import annotations

"""Autonomous candidate selection for pick/place scripts.

The selector is intentionally boring and easy to tune:
filter impossible or unwanted survey candidates first, then choose the remaining
object with the largest estimated 3D volume.
"""

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from planning.aabb_utils import AxisAlignedBox3D, aabb_from_object_candidate
from vision.pick_candidate_builder import CandidateDebug, SurveyState


@dataclass(frozen=True)
class BestCandidateConfig:
    require_positive_platform_xy: bool = True
    platform_min_x_mm: float = 0.0
    platform_min_y_mm: float = 0.0

    workspace_x_min_mm: float | None = 0.0
    workspace_x_max_mm: float | None = 900.0
    workspace_y_min_mm: float | None = 0.0
    workspace_y_max_mm: float | None = 900.0

    # Optional hard gates on robot-frame centroid derived from calibrated 3D estimate.
    require_robotframe_centroid_xy_in_platform_bounds: bool = False
    robotframe_centroid_x_min_mm: float | None = None
    robotframe_centroid_x_max_mm: float | None = None
    robotframe_centroid_y_min_mm: float | None = None
    robotframe_centroid_y_max_mm: float | None = None
    require_min_robotframe_centroid_z: bool = False
    min_robotframe_centroid_z_mm: float | None = None

    use_robot_reach_check: bool = True
    robot_reach_margin_mm: float = 2.0
    require_soft_pose_safe: bool = True
    soft_pose_check_z_mm: float = 250.0

    center_gate_enabled: bool = True
    max_image_center_norm_radius: float = 0.85

    cluster_gate_enabled: bool = True
    max_cluster_distance_mm: float = 260.0

    reject_placed_overlap: bool = True
    placed_overlap_margin_mm: float = 25.0

    min_volume_mm3: float = 1.0
    max_volume_mm3: float | None = 3_000_000.0


@dataclass
class CandidateDecision:
    dbg: CandidateDebug
    box: AxisAlignedBox3D | None
    xy_mm: np.ndarray
    size_xyz_mm: np.ndarray | None
    volume_mm3: float
    image_center_norm_radius: float | None
    cluster_distance_mm: float | None
    passed: bool
    reject_reasons: list[str]
    notes: list[str]

    @property
    def candidate_index(self) -> int:
        return int(getattr(self.dbg.candidate, "index", -1))

    @property
    def class_name(self) -> str:
        return str(getattr(self.dbg.candidate.yolo, "class_name", "unknown"))

    @property
    def volume_cm3(self) -> float:
        return float(self.volume_mm3) / 1000.0


@dataclass
class BestCandidateResult:
    selected: CandidateDebug | None
    selected_decision: CandidateDecision | None
    decisions: list[CandidateDecision]
    cluster_center_xy_mm: np.ndarray | None

    @property
    def valid_count(self) -> int:
        return sum(1 for d in self.decisions if d.passed)

    def display_lines(self, *, max_lines: int = 5) -> list[str]:
        total = len(self.decisions)
        valid = self.valid_count
        lines: list[str] = []
        if self.selected_decision is None:
            lines.append(f"AUTO PICK: no valid candidate ({valid}/{total} passed filters)")
        else:
            d = self.selected_decision
            lines.append(
                f"AUTO PICK: #{d.candidate_index} {d.class_name} | largest volume "
                f"{d.volume_cm3:.0f} cm3 among {valid}/{total} valid"
            )

        if self.cluster_center_xy_mm is not None:
            cc = self.cluster_center_xy_mm
            lines.append(f"cluster_xy=({cc[0]:.1f},{cc[1]:.1f}) | filters: workspace + center + not-in-box")
        else:
            lines.append("cluster_xy unavailable | filters: workspace + center + not-in-box")

        ranked = sorted(
            self.decisions,
            key=lambda d: (0 if d.passed else 1, -d.volume_mm3, d.candidate_index),
        )
        for d in ranked[: max(0, max_lines - len(lines))]:
            status = "PASS" if d.passed else "REJECT"
            reason = "ok" if d.passed else ";".join(d.reject_reasons[:2])
            lines.append(
                f"#{d.candidate_index} {status} {d.class_name} "
                f"vol={d.volume_cm3:.0f}cm3 xy=({d.xy_mm[0]:.0f},{d.xy_mm[1]:.0f}) {reason}"
            )
        return lines[:max_lines]

    def print_debug(self, prefix: str = "[BEST]") -> None:
        total = len(self.decisions)
        valid = self.valid_count
        print(f"{prefix} survey candidates={total}; passed_filters={valid}")
        if self.cluster_center_xy_mm is not None:
            cc = self.cluster_center_xy_mm
            print(f"{prefix} cluster_center_xy_mm=({cc[0]:.1f},{cc[1]:.1f})")
        else:
            print(f"{prefix} cluster_center_xy_mm=<unavailable>")

        for d in sorted(self.decisions, key=lambda item: item.candidate_index):
            status = "PASS" if d.passed else "REJECT"
            size_txt = "--"
            if d.size_xyz_mm is not None:
                s = d.size_xyz_mm
                size_txt = f"{s[0]:.1f}x{s[1]:.1f}x{s[2]:.1f}mm"
            center_txt = "--" if d.image_center_norm_radius is None else f"{d.image_center_norm_radius:.3f}"
            cluster_txt = "--" if d.cluster_distance_mm is None else f"{d.cluster_distance_mm:.1f}mm"
            reason_txt = "ok" if d.passed else "; ".join(d.reject_reasons)
            note_txt = "" if not d.notes else " | " + "; ".join(d.notes)
            print(
                f"{prefix} [{d.candidate_index}] {status:6s} {d.class_name:16s} "
                f"xy=({d.xy_mm[0]:7.1f},{d.xy_mm[1]:7.1f}) "
                f"vol={d.volume_cm3:8.1f}cm3 size={size_txt:>20s} "
                f"center_r={center_txt:>6s} cluster={cluster_txt:>8s} "
                f"reason={reason_txt}{note_txt}"
            )

        if self.selected_decision is None:
            print(f"{prefix} SELECTED: none. Every candidate was rejected by the active filters.")
            return

        d = self.selected_decision
        print(
            f"{prefix} SELECTED [{d.candidate_index}] {d.class_name}: "
            f"largest estimated volume among valid candidates "
            f"({d.volume_cm3:.1f} cm3, xyz={_fmt_size(d.size_xyz_mm)})."
        )


def choose_best_candidate(
    survey: SurveyState | list[CandidateDebug] | tuple[CandidateDebug, ...] | None,
    *,
    config: BestCandidateConfig,
    robot: Any = None,
    placed_boxes: list[Any] | tuple[Any, ...] | None = None,
) -> BestCandidateResult:
    if survey is None:
        candidates: list[CandidateDebug] = []
    elif isinstance(survey, SurveyState):
        candidates = list(survey.candidates)
    else:
        candidates = list(survey)

    placed_boxes = list(placed_boxes or [])
    cluster_center = _compute_cluster_center(candidates, config)

    decisions = [
        _evaluate_candidate(
            dbg,
            config=config,
            robot=robot,
            placed_boxes=placed_boxes,
            cluster_center_xy_mm=cluster_center,
        )
        for dbg in candidates
    ]

    passed = [d for d in decisions if d.passed]
    if not passed:
        return BestCandidateResult(None, None, decisions, cluster_center)

    selected_decision = max(
        passed,
        key=lambda d: (
            float(d.volume_mm3),
            float(getattr(d.dbg.candidate.yolo, "confidence", 0.0)),
            int(getattr(d.dbg.candidate, "valid_point_count", 0)),
        ),
    )
    return BestCandidateResult(selected_decision.dbg, selected_decision, decisions, cluster_center)


def _evaluate_candidate(
    dbg: CandidateDebug,
    *,
    config: BestCandidateConfig,
    robot: Any,
    placed_boxes: list[Any],
    cluster_center_xy_mm: np.ndarray | None,
) -> CandidateDecision:
    c = dbg.candidate
    reject: list[str] = []
    notes: list[str] = []

    xy = _candidate_xy(c)
    if xy is None:
        xy = np.array([float("nan"), float("nan")], dtype=np.float64)
        reject.append("missing_xy")

    box = None
    size = None
    volume = 0.0
    try:
        box = aabb_from_object_candidate(c, default_label=f"candidate{getattr(c, 'index', '?')}")
        size = np.asarray(box.size_xyz_mm, dtype=np.float64).reshape(3)
        volume = float(np.prod(size))
    except Exception as exc:
        reject.append(f"aabb_failed:{exc}")

    if xy is not None and np.all(np.isfinite(xy)):
        x = float(xy[0])
        y = float(xy[1])
        if config.require_positive_platform_xy:
            if x < float(config.platform_min_x_mm):
                reject.append(f"box_or_wrong_quadrant_x<{config.platform_min_x_mm:.1f}")
            if y < float(config.platform_min_y_mm):
                reject.append(f"box_or_wrong_quadrant_y<{config.platform_min_y_mm:.1f}")

        _check_bounds("x", x, config.workspace_x_min_mm, config.workspace_x_max_mm, reject)
        _check_bounds("y", y, config.workspace_y_min_mm, config.workspace_y_max_mm, reject)

        if config.use_robot_reach_check:
            ok, reason = _robot_reach_ok(robot, x, y, config.robot_reach_margin_mm)
            if not ok:
                reject.append(reason)

        if config.require_soft_pose_safe and robot is not None and hasattr(robot, "check_cartesian_pose_safe"):
            try:
                ok, reason = robot.check_cartesian_pose_safe(x, y, float(config.soft_pose_check_z_mm))
                if not ok:
                    reject.append(f"soft_pose_unsafe:{reason}")
            except Exception as exc:
                notes.append(f"soft_pose_check_failed:{exc}")

    robot_xyz = _candidate_robot_xyz(c)
    if config.require_robotframe_centroid_xy_in_platform_bounds:
        if robot_xyz is None:
            reject.append("missing_robotframe_centroid_xy")
        else:
            _check_bounds("robotframe_centroid_x", float(robot_xyz[0]), config.robotframe_centroid_x_min_mm, config.robotframe_centroid_x_max_mm, reject)
            _check_bounds("robotframe_centroid_y", float(robot_xyz[1]), config.robotframe_centroid_y_min_mm, config.robotframe_centroid_y_max_mm, reject)

    if config.require_min_robotframe_centroid_z:
        z_floor = config.min_robotframe_centroid_z_mm
        if robot_xyz is None:
            reject.append("missing_robotframe_centroid_z")
        elif z_floor is not None and float(robot_xyz[2]) < float(z_floor):
            reject.append(f"robotframe_centroid_z<{float(z_floor):.1f}")

    center_r = _image_center_norm_radius(dbg)
    if config.center_gate_enabled:
        if center_r is None:
            reject.append("image_center_unavailable")
        elif center_r > float(config.max_image_center_norm_radius):
            reject.append(f"far_from_frame_center:{center_r:.3f}>{config.max_image_center_norm_radius:.3f}")

    cluster_dist = None
    if cluster_center_xy_mm is not None and xy is not None and np.all(np.isfinite(xy)):
        cluster_dist = float(np.linalg.norm(xy - cluster_center_xy_mm))
        if config.cluster_gate_enabled and cluster_dist > float(config.max_cluster_distance_mm):
            reject.append(f"far_from_cluster:{cluster_dist:.1f}>{config.max_cluster_distance_mm:.1f}mm")
    elif config.cluster_gate_enabled:
        reject.append("cluster_distance_unavailable")

    if config.reject_placed_overlap and box is not None:
        overlap_label = _overlaps_placed(box, placed_boxes, float(config.placed_overlap_margin_mm))
        if overlap_label is not None:
            reject.append(f"already_in_placed_box:{overlap_label}")

    if not np.isfinite(volume) or volume < float(config.min_volume_mm3):
        reject.append(f"volume_too_small:{volume:.1f}<{config.min_volume_mm3:.1f}mm3")
    if config.max_volume_mm3 is not None and np.isfinite(volume) and volume > float(config.max_volume_mm3):
        reject.append(f"volume_absurdly_large:{volume / 1000.0:.1f}>{float(config.max_volume_mm3) / 1000.0:.1f}cm3")

    return CandidateDecision(
        dbg=dbg,
        box=box,
        xy_mm=xy,
        size_xyz_mm=size,
        volume_mm3=volume,
        image_center_norm_radius=center_r,
        cluster_distance_mm=cluster_dist,
        passed=not reject,
        reject_reasons=reject,
        notes=notes,
    )


def _candidate_xy(candidate: Any) -> np.ndarray | None:
    xy = getattr(candidate, "target_xy", None)
    if xy is None:
        return None
    arr = np.asarray(xy, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    out = arr[:2].astype(np.float64)
    if not np.all(np.isfinite(out)):
        return None
    return out


def _candidate_robot_xyz(candidate: Any) -> np.ndarray | None:
    for attr in ("object_robot_xyz_corrected", "object_robot_xyz_raw", "target_cam_xyz"):
        xyz = getattr(candidate, attr, None)
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            continue
        out = arr[:3].astype(np.float64)
        if np.all(np.isfinite(out)):
            return out
    return None


def _compute_cluster_center(
    candidates: list[CandidateDebug],
    config: BestCandidateConfig,
) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for dbg in candidates:
        xy = _candidate_xy(dbg.candidate)
        if xy is None:
            continue
        x = float(xy[0])
        y = float(xy[1])
        if config.require_positive_platform_xy:
            if x < float(config.platform_min_x_mm) or y < float(config.platform_min_y_mm):
                continue
        points.append(xy)

    if not points:
        for dbg in candidates:
            xy = _candidate_xy(dbg.candidate)
            if xy is not None:
                points.append(xy)

    if not points:
        return None
    return np.median(np.vstack(points), axis=0).astype(np.float64)


def _check_bounds(label: str, value: float, lo: float | None, hi: float | None, reject: list[str]) -> None:
    if lo is not None and value < float(lo):
        reject.append(f"workspace_{label}<{float(lo):.1f}")
    if hi is not None and value > float(hi):
        reject.append(f"workspace_{label}>{float(hi):.1f}")


def _robot_reach_ok(robot: Any, x_mm: float, y_mm: float, margin_mm: float) -> tuple[bool, str]:
    cfg = getattr(robot, "cfg", None)
    if cfg is None:
        return True, "robot_reach_check_skipped:no_cfg"
    try:
        l1 = float(cfg.L1_mm)
        l2 = float(cfg.L2_mm)
    except Exception:
        return True, "robot_reach_check_skipped:no_link_lengths"

    r = math.hypot(float(x_mm), float(y_mm))
    r_min = max(0.0, abs(l1 - l2) - float(margin_mm))
    r_max = l1 + l2 + float(margin_mm)
    if r < r_min:
        return False, f"outside_reach:r<{r_min:.1f}mm"
    if r > r_max:
        return False, f"outside_reach:r>{r_max:.1f}mm"
    return True, "reachable"


def _image_center_norm_radius(dbg: CandidateDebug) -> float | None:
    det = getattr(dbg, "best_detection", None)
    if det is None:
        det = getattr(dbg.candidate, "yolo", None)
    centroid = getattr(det, "centroid_px", None)
    if centroid is None:
        return None

    shape = None
    mask = getattr(det, "mask", None)
    if mask is not None:
        shape = np.asarray(mask).shape[:2]
    if (not shape) and getattr(dbg, "disparity", None) is not None:
        shape = np.asarray(dbg.disparity).shape[:2]
    if not shape or len(shape) < 2:
        return None

    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return None

    c = np.asarray(centroid, dtype=np.float64).reshape(-1)
    if c.size < 2 or not np.all(np.isfinite(c[:2])):
        return None

    dx = (float(c[0]) - 0.5 * w) / max(1.0, 0.5 * w)
    dy = (float(c[1]) - 0.5 * h) / max(1.0, 0.5 * h)
    return float(math.hypot(dx, dy))


def _overlaps_placed(candidate_box: AxisAlignedBox3D, placed_boxes: list[Any], margin_mm: float) -> str | None:
    cmin = np.asarray(candidate_box.min_xyz_mm[:2], dtype=np.float64) - margin_mm
    cmax = np.asarray(candidate_box.max_xyz_mm[:2], dtype=np.float64) + margin_mm
    center = np.asarray(candidate_box.center_xyz_mm[:2], dtype=np.float64)

    for idx, placed in enumerate(placed_boxes, start=1):
        placed_box = getattr(placed, "padded_box", placed)
        pmin_src = getattr(placed_box, "min_xyz_mm", None)
        pmax_src = getattr(placed_box, "max_xyz_mm", None)
        if pmin_src is None or pmax_src is None:
            continue
        pmin = np.asarray(pmin_src[:2], dtype=np.float64) - margin_mm
        pmax = np.asarray(pmax_src[:2], dtype=np.float64) + margin_mm
        if np.all(center >= pmin) and np.all(center <= pmax):
            return str(getattr(placed_box, "label", f"placed{idx}") or f"placed{idx}")
        separated = cmax[0] < pmin[0] or pmax[0] < cmin[0] or cmax[1] < pmin[1] or pmax[1] < cmin[1]
        if not separated:
            return str(getattr(placed_box, "label", f"placed{idx}") or f"placed{idx}")
    return None


def _fmt_size(size_xyz_mm: np.ndarray | None) -> str:
    if size_xyz_mm is None:
        return "--"
    s = np.asarray(size_xyz_mm, dtype=np.float64).reshape(-1)
    if s.size < 3:
        return "--"
    return f"{s[0]:.1f}x{s[1]:.1f}x{s[2]:.1f}mm"
