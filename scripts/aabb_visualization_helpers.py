from __future__ import annotations

"""Shared 3D plotting helpers for AABB placement validation."""

import numpy as np


def corners_from_box(box) -> np.ndarray:
    mins = box.min_xyz_mm
    maxs = box.max_xyz_mm
    return np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ], dtype=np.float64)


def draw_box(ax, box, color: str, linestyle: str = "-", alpha: float = 0.85) -> None:
    c = corners_from_box(box)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [c[i, 0], c[j, 0]],
            [c[i, 1], c[j, 1]],
            [c[i, 2], c[j, 2]],
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )


def set_axes_equal(ax, points_xyz: np.ndarray) -> None:
    mins = np.min(points_xyz, axis=0)
    maxs = np.max(points_xyz, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span <= 0.0:
        span = 1.0
    half = 0.6 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def draw_robot_axes(ax, origin_xyz_mm: np.ndarray | None = None, axis_len_mm: float = 80.0) -> None:
    origin = np.asarray(origin_xyz_mm if origin_xyz_mm is not None else [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
    L = float(max(1.0, axis_len_mm))
    ax.plot([origin[0], origin[0] + L], [origin[1], origin[1]], [origin[2], origin[2]], color="red", linewidth=2.0)
    ax.plot([origin[0], origin[0]], [origin[1], origin[1] + L], [origin[2], origin[2]], color="green", linewidth=2.0)
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + L], color="blue", linewidth=2.0)
    ax.text(origin[0] + L, origin[1], origin[2], "X")
    ax.text(origin[0], origin[1] + L, origin[2], "Y")
    ax.text(origin[0], origin[1], origin[2] + L, "Z")
