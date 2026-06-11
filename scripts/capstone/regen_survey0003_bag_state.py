"""Regenerate survey_0003 bag state with correct side-by-side Bandaid+ChexMix placements.

Root cause: original viz used first-cycle rows 1+2 (both at place_xy=[161,742.9] — stacked).
Ground truth from placed_boxes: Bandaid=[160,744.7], ChexMix=[270.6,747.4] — side-by-side.

Does NOT require YOLO/torch — uses only manifest data and the existing overhead image.

Outputs (in presentation_outputs/.../survey_0003_corrected/):
  overhead_with_seg.png   — overhead + YOLO detections (cropped from existing combined PNG)
  bag_state_targets.pdf   — vectorized bag state (editable text in Illustrator)
  bag_state_targets.png   — raster preview at 300 DPI
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType fonts → fully editable in Illustrator
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

REPO_ROOT = _REPO_ROOT
RUN_DIR = REPO_ROOT / "data" / "run_snapshots" / "run_20260531_185839"
EXISTING_COMBINED = (
    REPO_ROOT
    / "presentation_outputs"
    / "run_20260531_185839_pickplace_planner"
    / "survey_0003_row_03_Pringles"
    / "03_overhead_segmentation_and_valid_targets.png"
)
OUT_DIR = (
    REPO_ROOT
    / "presentation_outputs"
    / "run_20260531_185839_pickplace_planner"
    / "survey_0003_corrected"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ── Color palette (mirrors visualize_pickplace_planner.py) ────────────────────
CLASS_COLORS = {
    "Bandaid":       "#3b82f6",
    "Chex Mix":      "#f59e0b",
    "Pringles":      "#fb923c",
    "Carmex Lip Balm": "#a78bfa",
    "Burts Bees":    "#facc15",
    "Dove Deodorant":"#f87171",
    "Lays Chips":    "#38bdf8",
}
WINNER_COLOR = "#22c55e"

# ── Bag geometry (from _load_bag() / surface_zones.json "New Bag Test") ───────
# Matches BagVolume in the planner script
BAG = dict(x_min=80.0, x_max=340.0, y_min=680.0, y_max=830.0)

# ── Correct placed items from manifest placed_boxes (second-cycle ground truth) ─
# placed_boxes[0] = Bandaid  at [160, 744.7]   ← side-by-side
# placed_boxes[1] = Chex Mix at [270.6, 747.4] ← side-by-side
PLACED = [
    dict(name="Bandaid",  cx=159.96, cy=744.69, sx=119.93, sy=89.38),
    # Chex Mix: sx/sy swapped so long axis (134.88mm) runs in Y (vertical orientation).
    # pick_phi≈7.5° + place_phi=180° rotates the item ~172.5°, flipping its XY footprint.
    dict(name="Chex Mix", cx=270.56, cy=747.44, sx=98.89,  sy=134.88),
]

# ── Platform candidates at survey_0003 time ───────────────────────────────────
# Positions derived from second-cycle placed_boxes (what the robot actually did).
# All stacked items share the same XY as Bandaid [160, 744.7]; items not stacking
# there go to the upper bag region [145.6, 792.6].
# We spread labels so none overlap at identical XY in this 2D view.
CANDIDATES = [
    # Pringles (winner) — stacks on Bandaid column; winner shown with green border
    dict(label="1", name="Pringles",        cx=159.96, cy=744.69, sx=59.64,  sy=51.65, winner=True),
    # Dove Deodorant — goes to upper region of bag (elongated, placed horizontally)
    dict(label="2", name="Dove Deodorant",  cx=155.0,  cy=804.0,  sx=111.13, sy=34.70, winner=False),
    # Burts Bees — stacks on top of Chex Mix footprint (right column)
    dict(label="3", name="Burts Bees",      cx=270.56, cy=747.44, sx=76.27,  sy=54.13, winner=False),
    # Carmex Lip Balm — upper right region
    dict(label="4", name="Carmex Lip Balm", cx=282.0,  cy=804.0,  sx=100.22, sy=59.78, winner=False),
]


# ── Step 1: crop overhead from existing combined image ────────────────────────

def export_overhead_png() -> None:
    img = np.array(Image.open(str(EXISTING_COMBINED)))
    h, w = img.shape[:2]
    left_half = img[:, : w // 2, :]
    out = OUT_DIR / "overhead_with_seg.png"
    Image.fromarray(left_half).save(str(out), dpi=(DPI, DPI))
    print(f"  Overhead PNG:  {out}")


# ── Step 2: re-render bag state ───────────────────────────────────────────────

def render_bag_state(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_xlabel("Robot X (mm)", fontsize=11)
    ax.set_ylabel("Robot Y (mm)", fontsize=11)
    ax.set_title("Valid placement target for each current platform item", fontsize=11)

    # Bag outline
    bw = BAG["x_max"] - BAG["x_min"]
    bd = BAG["y_max"] - BAG["y_min"]
    ax.add_patch(Rectangle(
        (BAG["x_min"], BAG["y_min"]), bw, bd,
        facecolor="#f8fafc", edgecolor="#111827", linewidth=2.0, zorder=1,
    ))

    # Already-placed items (correct second-cycle positions)
    for item in PLACED:
        color = CLASS_COLORS.get(item["name"], "#94a3b8")
        ax.add_patch(Rectangle(
            (item["cx"] - 0.5 * item["sx"], item["cy"] - 0.5 * item["sy"]),
            item["sx"], item["sy"],
            facecolor=color, edgecolor=color, alpha=0.28, linewidth=1.4, zorder=2,
        ))
        ax.text(
            item["cx"], item["cy"], item["name"],
            ha="center", va="center", fontsize=7, color="#111827",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.70),
            zorder=3,
        )

    # Placement targets for each detected platform candidate
    for cand in CANDIDATES:
        color = CLASS_COLORS.get(cand["name"], "#94a3b8")
        edge  = WINNER_COLOR if cand["winner"] else color
        lw    = 2.0          if cand["winner"] else 1.2
        ax.add_patch(Rectangle(
            (cand["cx"] - 0.5 * cand["sx"], cand["cy"] - 0.5 * cand["sy"]),
            cand["sx"], cand["sy"],
            facecolor=color, edgecolor=edge, alpha=0.28, linewidth=lw, zorder=4,
        ))
        ax.text(
            cand["cx"], cand["cy"], cand["label"],
            ha="center", va="center",
            color="#111827", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=5,
        )

    ax.set_xlim(BAG["x_min"] - 20.0, BAG["x_max"] + 20.0)
    ax.set_ylim(BAG["y_min"] - 20.0, BAG["y_max"] + 20.0)


def export_bag_state() -> None:
    for ext, kwargs in [(".pdf", {}), (".png", {"dpi": DPI})]:
        fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor="white")
        render_bag_state(ax)
        fig.tight_layout()
        out = OUT_DIR / f"bag_state_targets{ext}"
        fig.savefig(str(out), **kwargs, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Bag state {ext.upper()[1:]}:  {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Exporting overhead PNG (cropped from existing combined image)...")
    export_overhead_png()

    print("Rendering corrected bag state...")
    for item in PLACED:
        print(f"  placed: {item['name']} XY=[{item['cx']:.1f}, {item['cy']:.1f}] "
              f"size=[{item['sx']:.1f} x {item['sy']:.1f}] mm")
    export_bag_state()

    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
