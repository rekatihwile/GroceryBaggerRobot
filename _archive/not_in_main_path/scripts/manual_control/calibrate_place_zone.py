from __future__ import annotations

"""
Manual placement-scene calibration GUI.

This extends scripts/manual_control/workspace_click_jog.py with a small zone
capture/edit/save panel. It does not close the claw and does not run any
autonomous pick logic.

Controls added here:
  n       set/edit zone name
  c       capture current FK as zone center/floor_z/phi
  z       edit width/depth/floor_z numerically
  b       toggle drawing current zone bounds
  Ctrl+S  save scene to config/surface_zones.json
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

PLACE_SCENE_CONFIG_PATH = Path("config/surface_zones.json")
DEFAULT_SCENE_NAME = "default_test_zone"
DEFAULT_ZONE_WIDTH_MM = 120.0
DEFAULT_ZONE_DEPTH_MM = 120.0

# ============================================================

import json
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "run_pickplace_fast.py").exists()),
    Path(__file__).resolve().parents[2],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.surface_zone_io import get_surface_zone, upsert_surface_zone
from config.robot_config import print_startup_config
from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from scripts.manual_control.workspace_click_jog import WorkspaceGUI


class PlaceSceneCalibrationGUI(WorkspaceGUI):
    def __init__(self, root: tk.Tk):
        self.zone_name = DEFAULT_SCENE_NAME
        self.zone_center_xy_mm = [450.0, 250.0]
        self.zone_floor_z_mm = 85.0
        self.zone_phi_deg = 0.0
        self.zone_width_mm = DEFAULT_ZONE_WIDTH_MM
        self.zone_depth_mm = DEFAULT_ZONE_DEPTH_MM
        self.zone_notes = "Saved from calibrate_place_zone.py."
        self.show_zone_bounds = True
        self._zone_labels: dict[str, tk.Label] = {}
        self._load_zone(DEFAULT_SCENE_NAME, quiet=True)
        super().__init__(root)
        self._update_zone_labels()

    def _build_ui(self):
        super()._build_ui()

        panel = tk.Frame(self.root, bg="#111827", width=260)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        def label(text, bold=False):
            font = ("Consolas", 10, "bold") if bold else ("Consolas", 10)
            tk.Label(panel, text=text, bg="#111827", fg="#e5e7eb", font=font, anchor="w").pack(
                fill=tk.X, padx=8, pady=1
            )

        label("PLACE ZONE", bold=True)
        for key, text in [
            ("name", "name:"),
            ("xy", "xy:"),
            ("floor_z", "floor z:"),
            ("phi", "phi:"),
            ("size", "size:"),
        ]:
            row = tk.Frame(panel, bg="#111827")
            row.pack(fill=tk.X, padx=8)
            tk.Label(row, text=text, bg="#111827", fg="#93c5fd", font=("Consolas", 10), width=9, anchor="w").pack(
                side=tk.LEFT
            )
            lbl = tk.Label(row, text="-", bg="#111827", fg="#ffffff", font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._zone_labels[key] = lbl

        tk.Label(
            panel,
            text="n=name  c=capture FK  z=edit dims/Z\nb=toggle bounds  Ctrl+S=save",
            bg="#111827",
            fg="#9ca3af",
            font=("Consolas", 8),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 4))

        for txt, cmd in [
            ("Set Name (n)", self._cmd_zone_name),
            ("Capture FK (c)", self._cmd_capture_zone),
            ("Edit Size/Z (z)", self._cmd_edit_zone_dims),
            ("Toggle Bounds (b)", self._cmd_toggle_bounds),
            ("Save Zone (Ctrl+S)", self._save_zone),
        ]:
            tk.Button(
                panel,
                text=txt,
                command=cmd,
                bg="#1f4d7a",
                fg="white",
                font=("Consolas", 10),
                relief=tk.FLAT,
                activebackground="#2563eb",
                activeforeground="white",
            ).pack(fill=tk.X, padx=8, pady=2)

        self.root.bind("<Control-s>", lambda _e: self._save_zone())

    def _load_zone(self, name: str, *, quiet: bool = False) -> None:
        try:
            zone = get_surface_zone(name, PLACE_SCENE_CONFIG_PATH)
        except Exception as exc:
            if not quiet:
                messagebox.showwarning("Place Zone", f"Could not load zone {name!r}:\n{exc}")
            return
        self.zone_name = str(zone["name"])
        self.zone_center_xy_mm = [float(zone["center_xy_mm"][0]), float(zone["center_xy_mm"][1])]
        self.zone_floor_z_mm = float(zone["surface_z_mm"])
        self.zone_phi_deg = float(zone["default_phi_deg"])
        self.zone_width_mm = float(zone["width_mm"])
        self.zone_depth_mm = float(zone["depth_mm"])
        self.zone_notes = str(zone.get("notes", ""))

    def _update_zone_labels(self) -> None:
        if not self._zone_labels:
            return
        self._zone_labels["name"].config(text=self.zone_name)
        self._zone_labels["xy"].config(text=f"{self.zone_center_xy_mm[0]:.1f}, {self.zone_center_xy_mm[1]:.1f} mm")
        self._zone_labels["floor_z"].config(text=f"{self.zone_floor_z_mm:.1f} mm")
        self._zone_labels["phi"].config(text=f"{self.zone_phi_deg:.1f} deg")
        self._zone_labels["size"].config(text=f"{self.zone_width_mm:.1f} x {self.zone_depth_mm:.1f} mm")

    def _draw_workspace(self):
        super()._draw_workspace()
        if self.show_zone_bounds:
            self._draw_zone_bounds()

    def _draw_zone_bounds(self) -> None:
        cx, cy = self.zone_center_xy_mm
        hw = 0.5 * self.zone_width_mm
        hd = 0.5 * self.zone_depth_mm
        x1, y1 = self._world_to_canvas(cx - hw, cy - hd)
        x2, y2 = self._world_to_canvas(cx + hw, cy + hd)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#facc15", width=3, dash=(6, 3))
        ccx, ccy = self._world_to_canvas(cx, cy)
        self.canvas.create_oval(ccx - 5, ccy - 5, ccx + 5, ccy + 5, fill="#facc15", outline="#111827")
        self.canvas.create_text(
            ccx + 8,
            ccy - 8,
            anchor="sw",
            fill="#facc15",
            font=("Consolas", 9, "bold"),
            text=self.zone_name,
        )

    def _cmd_zone_name(self):
        name = simpledialog.askstring("Place Zone Name", "Zone name:", initialvalue=self.zone_name, parent=self.root)
        if not name:
            return
        self.zone_name = name.strip()
        self._load_zone(self.zone_name, quiet=True)
        self._update_zone_labels()
        self._refresh()

    def _cmd_capture_zone(self):
        if not self.robot:
            messagebox.showwarning("Place Zone", "Robot is not connected yet.")
            return
        x, y, z, phi = self.robot.fk()
        self.zone_center_xy_mm = [float(x), float(y)]
        self.zone_floor_z_mm = float(z)
        self.zone_phi_deg = float(phi)
        print(
            f"[PLACE ZONE] captured FK: x={x:.1f} y={y:.1f} "
            f"floor_z={z:.1f} phi={phi:.1f}"
        )
        self._update_zone_labels()
        self._refresh()

    def _cmd_edit_zone_dims(self):
        width = simpledialog.askfloat(
            "Zone Width", "Width in robot X direction (mm):", initialvalue=self.zone_width_mm, parent=self.root
        )
        if width is None:
            return
        depth = simpledialog.askfloat(
            "Zone Depth", "Depth in robot Y direction (mm):", initialvalue=self.zone_depth_mm, parent=self.root
        )
        if depth is None:
            return
        place_z = simpledialog.askfloat(
            "Floor Z", "Floor/table Z (mm):", initialvalue=self.zone_floor_z_mm, parent=self.root
        )
        if place_z is None:
            return
        phi = simpledialog.askfloat(
            "Place Phi", "Place phi (deg):", initialvalue=self.zone_phi_deg, parent=self.root
        )
        if phi is None:
            return
        self.zone_width_mm = float(width)
        self.zone_depth_mm = float(depth)
        self.zone_floor_z_mm = float(place_z)
        self.zone_phi_deg = float(phi)
        self._update_zone_labels()
        self._refresh()

    def _cmd_toggle_bounds(self):
        self.show_zone_bounds = not self.show_zone_bounds
        self._refresh()

    def _save_zone(self):
        try:
            zone = upsert_surface_zone(
                name=self.zone_name,
                center_xy_mm=self.zone_center_xy_mm,
                surface_z_mm=self.zone_floor_z_mm,
                default_phi_deg=self.zone_phi_deg,
                width_mm=self.zone_width_mm,
                depth_mm=self.zone_depth_mm,
                notes=self.zone_notes,
                path=PLACE_SCENE_CONFIG_PATH,
            )
        except Exception as exc:
            messagebox.showerror("Save Place Zone", str(exc))
            return
        print("[PLACE ZONE] saved:")
        print(json.dumps(zone, indent=2))
        self._set_status(f"Saved scene {zone['name']}", "#4ae04a")
        self._update_zone_labels()
        self._refresh()

    def _on_key(self, event):
        k = event.keysym.lower()
        if k == "s" and (int(getattr(event, "state", 0)) & 0x4):
            self._save_zone()
        elif k == "n":
            self._cmd_zone_name()
        elif k == "c":
            self._cmd_capture_zone()
        elif k == "z":
            self._cmd_edit_zone_dims()
        elif k == "b":
            self._cmd_toggle_bounds()
        else:
            super()._on_key(event)


def main():
    print_startup_config("calibrate_place_zone.py", OVERHEAD_INDEX, STEREO_INDEX)
    root = tk.Tk()
    root.resizable(True, True)
    app = PlaceSceneCalibrationGUI(root)

    def on_close():
        if app.robot:
            app.robot.close()
        if app._overhead_cap is not None:
            app._overhead_cap.release()
        if app._stereo_cam is not None:
            app._stereo_cam.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
