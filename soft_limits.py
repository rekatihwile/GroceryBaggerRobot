from __future__ import annotations

"""
soft_limits.py

Small soft-limit / obstacle-avoidance helper for the grocery bagger.

Model:
    A forbidden overhead frame region is represented as an XY rectangle/polygon
    that only becomes forbidden when z_mm >= z_min_mm.

    Example: if the gripper is retracted high, it may hit the overhead frame
    while passing through a certain XY range. At lower Z, that same XY may be safe.

Main functions:
    is_pose_safe(x, y, z)
    is_segment_safe(p0, p1)
    plan_safe_path(p0, p1)

This is intentionally conservative and easy to debug.
"""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Iterable

Point3 = tuple[float, float, float]
Point2 = tuple[float, float]


@dataclass
class ForbiddenBox:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    z_min: float
    margin_xy: float = 20.0
    margin_z: float = 0.0

    def expanded(self) -> "ForbiddenBox":
        return ForbiddenBox(
            name=self.name,
            xmin=min(self.xmin, self.xmax) - self.margin_xy,
            xmax=max(self.xmin, self.xmax) + self.margin_xy,
            ymin=min(self.ymin, self.ymax) - self.margin_xy,
            ymax=max(self.ymin, self.ymax) + self.margin_xy,
            z_min=self.z_min - self.margin_z,
            margin_xy=0.0,
            margin_z=0.0,
        )

    def contains_xy(self, x: float, y: float, expanded: bool = True) -> bool:
        b = self.expanded() if expanded else self
        return b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax

    def contains_pose(self, x: float, y: float, z: float, expanded: bool = True) -> bool:
        b = self.expanded() if expanded else self
        return z >= b.z_min and b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax

    def corners_expanded(self) -> list[Point2]:
        b = self.expanded()
        return [(b.xmin, b.ymin), (b.xmax, b.ymin), (b.xmax, b.ymax), (b.xmin, b.ymax)]


@dataclass
class SoftLimitConfig:
    boxes: list[ForbiddenBox]
    sample_step_mm: float = 5.0
    default_detour_margin_mm: float = 35.0
    max_waypoints: int = 8

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        payload = {
            "sample_step_mm": self.sample_step_mm,
            "default_detour_margin_mm": self.default_detour_margin_mm,
            "max_waypoints": self.max_waypoints,
            "boxes": [asdict(b) for b in self.boxes],
        }
        p.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "SoftLimitConfig":
        p = Path(path)
        payload = json.loads(p.read_text())
        return cls(
            boxes=[ForbiddenBox(**b) for b in payload.get("boxes", [])],
            sample_step_mm=float(payload.get("sample_step_mm", 5.0)),
            default_detour_margin_mm=float(payload.get("default_detour_margin_mm", 35.0)),
            max_waypoints=int(payload.get("max_waypoints", 8)),
        )


def default_config() -> SoftLimitConfig:
    """EDIT THESE NUMBERS FIRST.

    Coordinates are robot-frame mm. This example box is fake; replace with the
    rough XY rectangle where your overhead frame would collide with a raised claw.
    """
    return SoftLimitConfig(
        boxes=[
            ForbiddenBox(
                name="overhead_frame_bar",
                xmin=-800.0,
                xmax=210.0,
                ymin=270.0,
                ymax=730.0,
                z_min=65.0,
                margin_xy=5.0,
                margin_z=5.0,
            ),
        ],
        sample_step_mm=5.0,
        default_detour_margin_mm=35.0,
        max_waypoints=8,
    )


def is_pose_safe(p: Point3, cfg: SoftLimitConfig) -> tuple[bool, str]:
    x, y, z = p
    for box in cfg.boxes:
        if box.contains_pose(x, y, z, expanded=True):
            return False, f"pose inside forbidden region '{box.name}' at z={z:.1f}"
    return True, "safe"


def lerp3(a: Point3, b: Point3, t: float) -> Point3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def dist3(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def is_segment_safe(a: Point3, b: Point3, cfg: SoftLimitConfig) -> tuple[bool, str]:
    length = dist3(a, b)
    n = max(2, int(math.ceil(length / max(cfg.sample_step_mm, 1e-6))) + 1)
    for i in range(n):
        p = lerp3(a, b, i / (n - 1))
        ok, reason = is_pose_safe(p, cfg)
        if not ok:
            return False, f"segment collision near ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}): {reason}"
    return True, "safe"


def path_is_safe(path: list[Point3], cfg: SoftLimitConfig) -> tuple[bool, str]:
    if len(path) < 2:
        return True, "safe"
    for p in path:
        ok, reason = is_pose_safe(p, cfg)
        if not ok:
            return False, reason
    for a, b in zip(path[:-1], path[1:]):
        ok, reason = is_segment_safe(a, b, cfg)
        if not ok:
            return False, reason
    return True, "safe"


def _segment_bbox_maybe_intersects_xy(a: Point3, b: Point3, box: ForbiddenBox) -> bool:
    """Fast conservative XY overlap test."""
    be = box.expanded()
    xmin = min(a[0], b[0])
    xmax = max(a[0], b[0])
    ymin = min(a[1], b[1])
    ymax = max(a[1], b[1])
    return not (xmax < be.xmin or xmin > be.xmax or ymax < be.ymin or ymin > be.ymax)


def _active_boxes_for_segment(a: Point3, b: Point3, cfg: SoftLimitConfig) -> list[ForbiddenBox]:
    active = []
    zmax = max(a[2], b[2])
    for box in cfg.boxes:
        if zmax >= box.expanded().z_min and _segment_bbox_maybe_intersects_xy(a, b, box):
            active.append(box)
    return active


def _candidate_detour_paths(a: Point3, b: Point3, box: ForbiddenBox, cfg: SoftLimitConfig) -> list[list[Point3]]:
    """Generate simple one/two-waypoint paths around an expanded rectangle."""
    be = box.expanded()
    z = max(a[2], b[2])
    m = cfg.default_detour_margin_mm

    left_x = be.xmin - m
    right_x = be.xmax + m
    bottom_y = be.ymin - m
    top_y = be.ymax + m

    candidates: list[list[Point3]] = []

    # Single-side detours.
    candidates.append([a, (left_x, a[1], z), (left_x, b[1], z), b])
    candidates.append([a, (right_x, a[1], z), (right_x, b[1], z), b])
    candidates.append([a, (a[0], bottom_y, z), (b[0], bottom_y, z), b])
    candidates.append([a, (a[0], top_y, z), (b[0], top_y, z), b])

    # Corner detours. Useful if start/end are diagonal around the box.
    corners = [
        (left_x, bottom_y, z),
        (right_x, bottom_y, z),
        (right_x, top_y, z),
        (left_x, top_y, z),
    ]
    for c in corners:
        candidates.append([a, c, b])

    # Keep target z at the end; intermediate Z is intentionally conservative.
    fixed = []
    for path in candidates:
        path2 = [path[0]]
        for p in path[1:-1]:
            path2.append(p)
        path2.append(b)
        fixed.append(path2)
    return fixed


def _path_length(path: list[Point3]) -> float:
    return sum(dist3(a, b) for a, b in zip(path[:-1], path[1:]))


def plan_safe_path(a: Point3, b: Point3, cfg: SoftLimitConfig) -> tuple[bool, list[Point3], str]:
    """Return a safe waypoint path from a to b.

    Conservative behavior:
      - If the final target is inside a forbidden raised region, refuse.
      - If the direct segment is safe, use it.
      - If the direct segment collides with one box, try simple detours around it.
      - If multiple boxes are involved, this still often works, but it is not a full
        motion planner. For capstone use, keep the forbidden regions simple.
    """
    ok, reason = is_pose_safe(a, cfg)
    if not ok:
        return False, [a], f"start unsafe: {reason}"
    ok, reason = is_pose_safe(b, cfg)
    if not ok:
        return False, [a], f"target unsafe: {reason}"

    ok, reason = is_segment_safe(a, b, cfg)
    if ok:
        return True, [a, b], "direct path safe"

    active = _active_boxes_for_segment(a, b, cfg)
    if not active:
        return False, [a, b], reason

    candidates: list[list[Point3]] = []
    for box in active:
        candidates.extend(_candidate_detour_paths(a, b, box, cfg))

    safe_candidates = []
    for path in candidates:
        ok, why = path_is_safe(path, cfg)
        if ok and len(path) <= cfg.max_waypoints:
            safe_candidates.append(path)

    if not safe_candidates:
        return False, [a, b], "direct path unsafe and no simple detour found; add more margin or split the move"

    best = min(safe_candidates, key=_path_length)
    return True, best, f"detour path with {len(best) - 2} waypoint(s)"


def sample_path(path: list[Point3], step_mm: float = 10.0) -> list[Point3]:
    pts: list[Point3] = []
    for a, b in zip(path[:-1], path[1:]):
        length = dist3(a, b)
        n = max(2, int(math.ceil(length / max(step_mm, 1e-6))) + 1)
        for i in range(n):
            if pts and i == 0:
                continue
            pts.append(lerp3(a, b, i / (n - 1)))
    return pts
