"""run_snapshot_view_config.py — interactive 3D view configurator for run_snapshot_placement_gif.py.

Launches a GUI with sliders and inputs that let you adjust every visual constant
used by the GIF script, previewing the result on a real scene from the run snapshot.
Press "Save" to write view_config.json next to this script — the GIF script reads
it automatically on startup.

Usage:
    python scripts/capstone/run_snapshot_view_config.py [--run-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")  # interactive backend — falls back gracefully on headless systems
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from scripts.capstone.pointcloud_color import (
    load_stereo_calibration,
    photo_colors_for_points,
    project_bundle_camera_points_to_uv,
)

# ─── IDE defaults ─────────────────────────────────────────────────────────────
IDE_DEFAULT_RUN_DIR_STR: str | None = (
    r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
)

# ─── Output path (same folder as this script) ────────────────────────────────
_OUT_CONFIG = Path(__file__).resolve().parent / "view_config.json"

# ─── Default values (mirror GIF script constants) ────────────────────────────
DEFAULTS = {
    "elev_deg":     24.0,
    "azim_deg":    -58.0,
    "point_size":    1.0,
    "point_alpha":   0.55,
    "gif_dpi":      90.0,
    "gif_frame_ms": 70.0,
    "n_anim_steps": 16.0,
    "fig_w_in":     10.0,
    "fig_h_in":      8.0,
    "x_min":      None,
    "x_max":      None,
    "y_min":      None,
    "y_max":      None,
    "z_min":      None,
    "z_max":      None,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_run_dir() -> Path | None:
    if IDE_DEFAULT_RUN_DIR_STR:
        p = Path(IDE_DEFAULT_RUN_DIR_STR)
        if p.exists():
            return p
    candidates = sorted((_REPO_ROOT / "data" / "run_snapshots").glob("run_*"))
    return candidates[-1] if candidates else None


def _load_bundle() -> dict:
    bundle_path = _REPO_ROOT / "robot_calibration_bundle.npz"
    if not bundle_path.exists():
        return {}
    raw = np.load(bundle_path, allow_pickle=True)
    return {k: raw[k].item() if raw[k].ndim == 0 else raw[k] for k in raw.files}


def _cam_to_robot(pts_cam: np.ndarray, bundle: dict) -> np.ndarray:
    if len(pts_cam) == 0 or "A_robot_from_cam_xyz_3x4" not in bundle:
        return pts_cam
    try:
        A = np.asarray(bundle["A_robot_from_cam_xyz_3x4"], dtype=np.float64)
        h = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)])
        pts_r = (A @ h.T).T
        return pts_r[np.all(np.isfinite(pts_r), axis=1)]
    except Exception:
        return np.empty((0, 3), dtype=np.float64)


def _load_scene(run_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud from the first usable manifest row. Returns (pts_robot, colors)."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return np.empty((0, 3)), None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("objects", [])
    except Exception:
        return np.empty((0, 3)), None

    bundle = _load_bundle()
    stereo_calib = load_stereo_calibration(_REPO_ROOT / "stereo_calibration.npz")

    for row in rows:
        pc_file = row.get("points_cam")
        if not pc_file:
            continue
        pc_path = run_dir / pc_file
        if not pc_path.exists():
            continue
        try:
            d = np.load(pc_path)
            pts_cam = np.asarray(d["points_cam"], dtype=np.float64).reshape(-1, 3)
        except Exception:
            continue
        stereo_left = row.get("stereo_left")
        image_bgr = cv2.imread(str(run_dir / stereo_left), cv2.IMREAD_COLOR) if stereo_left else None
        image_rgb = (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if image_bgr is not None
            else None
        )
        uv_px = project_bundle_camera_points_to_uv(pts_cam, stereo_calib)
        colors, color_valid = photo_colors_for_points(image_rgb, uv_px, len(pts_cam))
        if colors is None or color_valid is None:
            continue
        pts_cam = pts_cam[color_valid]
        pts_robot = _cam_to_robot(pts_cam, bundle)
        if len(pts_robot) == 0:
            continue
        # Subsample for speed
        if len(pts_robot) > 4000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts_robot), 4000, replace=False)
            pts_robot = pts_robot[idx]
            colors = colors[idx]
        return pts_robot, colors
    return np.empty((0, 3)), None


def _load_zone_bounds() -> tuple[float, float, float, float] | None:
    try:
        ws_cfg = get_workspace_filter_config("default")
        x_min, x_max, y_min, y_max = workspace_bounds_mm(ws_cfg)
        return x_min, x_max, y_min, y_max
    except Exception:
        return None


def _draw_workspace_box(ax, bounds: tuple[float, float, float, float], z: float = 0.0) -> None:
    x0, x1, y0, y1 = bounds
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    zs = [z, z, z, z, z]
    ax.plot(xs, ys, zs, color="#f2c66d", linewidth=1.0, alpha=0.6)


def _set_limits(ax, cfg: dict, pts: np.ndarray) -> None:
    def _axis_lim(key_min, key_max, data_col):
        lo = cfg[key_min]
        hi = cfg[key_max]
        if lo is not None and hi is not None:
            return float(lo), float(hi)
        if len(pts) > 0:
            mn = float(pts[:, data_col].min())
            mx = float(pts[:, data_col].max())
            pad = max((mx - mn) * 0.1, 10.0)
            return mn - pad, mx + pad
        return -300.0, 300.0

    ax.set_xlim(*_axis_lim("x_min", "x_max", 0))
    ax.set_ylim(*_axis_lim("y_min", "y_max", 1))
    ax.set_zlim(*_axis_lim("z_min", "z_max", 2))


def _parse_float_or_none(s: str) -> float | None:
    s = s.strip()
    if s in ("", "auto", "none", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_display(val: float | None) -> str:
    return "" if val is None else f"{val:.1f}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Interactive view configurator for GIF script")
    p.add_argument("--run-dir", type=Path, default=None)
    args = p.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else _resolve_run_dir()

    print(f"[INFO] run_dir = {run_dir}")
    pts, _ = _load_scene(run_dir) if run_dir else (np.empty((0, 3)), None)
    bounds = _load_zone_bounds()

    cfg = dict(DEFAULTS)

    # ── Load existing view_config.json if present ─────────────────────────────
    if _OUT_CONFIG.exists():
        try:
            saved = json.loads(_OUT_CONFIG.read_text(encoding="utf-8"))
            v = saved.get("view", {})
            s = saved.get("style", {})
            lims = saved.get("limits", {})
            cfg["elev_deg"]     = float(v.get("elev_deg",     cfg["elev_deg"]))
            cfg["azim_deg"]     = float(v.get("azim_deg",     cfg["azim_deg"]))
            cfg["point_size"]   = float(s.get("point_size",   cfg["point_size"]))
            cfg["point_alpha"]  = float(s.get("point_alpha",  cfg["point_alpha"]))
            cfg["gif_dpi"]      = float(s.get("gif_dpi",      cfg["gif_dpi"]))
            cfg["gif_frame_ms"] = float(s.get("gif_frame_ms", cfg["gif_frame_ms"]))
            cfg["n_anim_steps"] = float(s.get("n_anim_steps", cfg["n_anim_steps"]))
            cfg["fig_w_in"]     = float(s.get("fig_w_in",     cfg["fig_w_in"]))
            cfg["fig_h_in"]     = float(s.get("fig_h_in",     cfg["fig_h_in"]))
            for ax_key in ("x", "y", "z"):
                lv = lims.get(ax_key)
                if isinstance(lv, list) and len(lv) == 2:
                    cfg[f"{ax_key}_min"] = lv[0]
                    cfg[f"{ax_key}_max"] = lv[1]
            print(f"[INFO] Loaded existing {_OUT_CONFIG}")
        except Exception as exc:
            print(f"[WARN] Could not load {_OUT_CONFIG}: {exc}")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    fig.canvas.manager.set_window_title("GIF View Configurator — run_snapshot_placement_gif")

    # 3D preview panel
    ax3d = fig.add_axes([0.0, 0.0, 0.58, 1.0], projection="3d")
    ax3d.set_facecolor("#1a1a2e")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.set_xlabel("X (mm)", color="#aaaacc", fontsize=7)
    ax3d.set_ylabel("Y (mm)", color="#aaaacc", fontsize=7)
    ax3d.set_zlabel("Z (mm)", color="#aaaacc", fontsize=7)
    ax3d.tick_params(colors="#aaaacc", labelsize=6)

    # Control panel area
    ctrl_left = 0.60
    ctrl_w    = 0.38

    sliders: dict[str, Slider] = {}
    textboxes: dict[str, TextBox] = {}

    # Slider definitions: (key, label, valmin, valmax, step)
    slider_defs = [
        ("elev_deg",     "Elev (deg)",        0,    90,   1),
        ("azim_deg",     "Azim (deg)",      -180,  180,   1),
        ("point_size",   "Point size",       0.3,   6.0,  0.1),
        ("point_alpha",  "Point alpha",      0.05,  1.0,  0.05),
        ("gif_dpi",      "GIF DPI",          40,   200,   5),
        ("gif_frame_ms", "Frame ms",         20,   500,  10),
        ("n_anim_steps", "Anim steps",        4,    48,   1),
        ("fig_w_in",     "Fig width (in)",    4,    24,   0.5),
        ("fig_h_in",     "Fig height (in)",   3,    18,   0.5),
    ]

    n_sliders = len(slider_defs)
    slider_h  = 0.03
    slider_gap = 0.015
    block_h   = n_sliders * (slider_h + slider_gap)
    top_y     = 0.95
    slider_x  = ctrl_left + 0.08
    slider_w  = ctrl_w - 0.12

    for i, (key, label, vmin, vmax, _vstep) in enumerate(slider_defs):
        y = top_y - i * (slider_h + slider_gap) - slider_h
        ax_s = fig.add_axes([slider_x, y, slider_w, slider_h], facecolor="#2a2a4e")
        ax_s.tick_params(labelcolor="#aaaacc")
        sl = Slider(ax_s, label, vmin, vmax, valinit=cfg[key], color="#4a90d9")
        sl.label.set_color("#ccccee")
        sl.label.set_fontsize(8)
        sl.valtext.set_color("#ccccee")
        sl.valtext.set_fontsize(7)
        sliders[key] = sl

    # Axis limit text boxes
    limit_y = top_y - block_h - 0.07
    lim_keys = [
        ("x_min", "X min"), ("x_max", "X max"),
        ("y_min", "Y min"), ("y_max", "Y max"),
        ("z_min", "Z min"), ("z_max", "Z max"),
    ]
    tb_w = 0.065
    tb_h = 0.030
    tb_gap = 0.010
    pair_w = tb_w * 2 + tb_gap + 0.02  # label + two boxes
    cols = 3
    for j, (key, lbl) in enumerate(lim_keys):
        col = j % cols
        row = j // cols
        x_off = ctrl_left + col * (pair_w + 0.01)
        y_off = limit_y - row * (tb_h + 0.02)
        # label
        fig.text(x_off, y_off + 0.01, lbl + ":", color="#aaaacc", fontsize=7, va="bottom",
                 transform=fig.transFigure)
        ax_tb = fig.add_axes([x_off + 0.04, y_off, tb_w, tb_h], facecolor="#2a2a4e")
        tb = TextBox(ax_tb, "", initial=_to_display(cfg[key]), color="#2a2a4e", hovercolor="#3a3a6e")
        tb.text_disp.set_color("#ccccee")
        tb.text_disp.set_fontsize(7)
        textboxes[key] = tb

    # Buttons
    btn_y = limit_y - 0.12
    ax_save  = fig.add_axes([ctrl_left + 0.04, btn_y, 0.12, 0.05], facecolor="#2a4a2a")
    ax_reset = fig.add_axes([ctrl_left + 0.20, btn_y, 0.12, 0.05], facecolor="#4a2a2a")
    btn_save  = Button(ax_save,  "Save config", color="#2a4a2a", hovercolor="#3a6a3a")
    btn_reset = Button(ax_reset, "Reset defaults", color="#4a2a2a", hovercolor="#6a3a3a")
    btn_save.label.set_color("#aaffaa")
    btn_save.label.set_fontsize(9)
    btn_reset.label.set_color("#ffaaaa")
    btn_reset.label.set_fontsize(9)

    # Status text
    status_text = fig.text(
        ctrl_left + 0.02, btn_y - 0.06, "",
        color="#aaaacc", fontsize=7, va="top", transform=fig.transFigure,
    )

    # ── Preview render ────────────────────────────────────────────────────────
    def _render_preview() -> None:
        ax3d.cla()
        ax3d.set_facecolor("#1a1a2e")
        ax3d.set_xlabel("X (mm)", color="#aaaacc", fontsize=7)
        ax3d.set_ylabel("Y (mm)", color="#aaaacc", fontsize=7)
        ax3d.set_zlabel("Z (mm)", color="#aaaacc", fontsize=7)
        ax3d.tick_params(colors="#aaaacc", labelsize=6)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

        # Draw workspace boundary
        if bounds is not None:
            _draw_workspace_box(ax3d, bounds, z=0.0)

        # Draw point cloud
        if len(pts) > 0 and colors is not None:
            ax3d.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=cfg["point_size"], alpha=cfg["point_alpha"],
                c=colors, depthshade=False,
            )

        ax3d.view_init(elev=cfg["elev_deg"], azim=cfg["azim_deg"])
        _set_limits(ax3d, cfg, pts)
        fig.canvas.draw_idle()

    # ── Sync cfg from widgets ─────────────────────────────────────────────────
    def _sync_from_widgets() -> None:
        for key, sl in sliders.items():
            cfg[key] = sl.val
        for key, tb in textboxes.items():
            cfg[key] = _parse_float_or_none(tb.text)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_slider_changed(_val) -> None:
        _sync_from_widgets()
        _render_preview()

    def _on_text_submit(_text) -> None:
        _sync_from_widgets()
        _render_preview()

    def _on_save(_event) -> None:
        _sync_from_widgets()
        def _lim(kmin, kmax):
            lo, hi = cfg[kmin], cfg[kmax]
            if lo is not None and hi is not None:
                return [lo, hi]
            return None
        out = {
            "view": {
                "elev_deg": cfg["elev_deg"],
                "azim_deg": cfg["azim_deg"],
            },
            "limits": {
                "x": _lim("x_min", "x_max"),
                "y": _lim("y_min", "y_max"),
                "z": _lim("z_min", "z_max"),
            },
            "style": {
                "point_size":   cfg["point_size"],
                "point_alpha":  cfg["point_alpha"],
                "gif_dpi":      int(round(cfg["gif_dpi"])),
                "gif_frame_ms": int(round(cfg["gif_frame_ms"])),
                "n_anim_steps": int(round(cfg["n_anim_steps"])),
                "fig_w_in":     cfg["fig_w_in"],
                "fig_h_in":     cfg["fig_h_in"],
            },
        }
        _OUT_CONFIG.write_text(json.dumps(out, indent=2), encoding="utf-8")
        msg = f"Saved to {_OUT_CONFIG}"
        print(f"[SAVE] {msg}")
        status_text.set_text(msg)
        status_text.set_color("#aaffaa")
        fig.canvas.draw_idle()

    def _on_reset(_event) -> None:
        for key, sl in sliders.items():
            sl.set_val(DEFAULTS[key])
        for key, tb in textboxes.items():
            tb.set_val(_to_display(DEFAULTS[key]))
        _sync_from_widgets()
        status_text.set_text("Reset to defaults")
        status_text.set_color("#ffaaaa")
        _render_preview()

    for sl in sliders.values():
        sl.on_changed(_on_slider_changed)
    for tb in textboxes.values():
        tb.on_submit(_on_text_submit)
    btn_save.on_clicked(_on_save)
    btn_reset.on_clicked(_on_reset)

    # ── Initial render ────────────────────────────────────────────────────────
    _render_preview()
    n_pts = len(pts)
    status_text.set_text(
        f"Preview: {n_pts} pts from {run_dir.name if run_dir else '(no run)'}  |  "
        f"Config: {_OUT_CONFIG.name}"
    )
    status_text.set_color("#aaaacc")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
