from __future__ import annotations

"""Offline 3D visualization for raw/padded AABBs and adjacent placement."""

# ============================================================
# USER SETTINGS
# ============================================================

USE_SYNTHETIC_BOXES = True
BOX1_SIZE_MM = [120, 80, 100]
BOX2_SIZE_MM = [90, 70, 60]
BOX1_CENTER_MM = [0, 0, 50]
BOX2_CENTER_MM = [0, 0, 30]
PAD_X_MM = 10
PAD_Y_MM = 10
PAD_Z_MM = 0
ADJACENT_DIRECTION = "right"
SURFACE_Z_MM = 0.0

# ============================================================

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.aabb_utils import make_aabb_from_center_size, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement
from scripts.aabb_visualization_helpers import corners_from_box, draw_box, draw_robot_axes, set_axes_equal


def main() -> int:
    if not USE_SYNTHETIC_BOXES:
        print("[INFO] USE_SYNTHETIC_BOXES=False currently not wired to live candidate loading; using synthetic.")

    box1 = make_aabb_from_center_size(BOX1_CENTER_MM, BOX1_SIZE_MM, label="box1_raw")
    box2 = make_aabb_from_center_size(BOX2_CENTER_MM, BOX2_SIZE_MM, label="box2_raw")
    box1_padded = pad_aabb(box1, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
    box2_padded = pad_aabb(box2, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)

    plan = compute_adjacent_placement(
        reference_padded_box=box1_padded,
        moving_padded_box=box2_padded,
        direction=ADJACENT_DIRECTION,
        surface_z_mm=SURFACE_Z_MM,
        place_phi_deg=0.0,
    )

    print("[BOX SUMMARY]")
    print(f"box1 raw size mm = {box1.size_xyz_mm.tolist()}")
    print(f"box1 padded size mm = {box1_padded.padded_box.size_xyz_mm.tolist()}")
    print(f"box2 raw size mm = {box2.size_xyz_mm.tolist()}")
    print(f"box2 padded size mm = {box2_padded.padded_box.size_xyz_mm.tolist()}")
    print(f"proposed box2 center mm = {plan.target_center_xyz_mm.tolist()}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR] matplotlib is required for visualization: {exc}")
        return 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_box(ax, box1, color="tab:blue", linestyle="-")
    draw_box(ax, box1_padded.padded_box, color="tab:blue", linestyle="--")
    draw_box(ax, box2, color="tab:orange", linestyle="-")
    draw_box(ax, box2_padded.padded_box, color="tab:orange", linestyle="--")
    draw_box(ax, plan.placed_box, color="tab:green", linestyle="-")

    centers = np.vstack([
        box1.center_xyz_mm,
        box2.center_xyz_mm,
        plan.target_center_xyz_mm,
    ])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=["blue", "orange", "green"], s=50)

    label_points = [
        (box1.center_xyz_mm, "box1_raw"),
        (box2.center_xyz_mm, "box2_raw"),
        (plan.target_center_xyz_mm, "box2_adjacent_target"),
    ]
    for pt, text in label_points:
        ax.text(float(pt[0]), float(pt[1]), float(pt[2]), text)

    all_points = np.vstack([
        corners_from_box(box1_padded.padded_box),
        corners_from_box(box2_padded.padded_box),
        corners_from_box(plan.placed_box),
        centers,
    ])
    set_axes_equal(ax, all_points)
    draw_robot_axes(ax)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Raw/Padded AABBs and Adjacent Placement")
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
