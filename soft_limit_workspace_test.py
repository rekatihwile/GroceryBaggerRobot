from __future__ import annotations

"""
soft_limit_workspace_test.py

Click-to-move soft-limit tester for the grocery bagger.

What it does:
  - Opens a Tkinter XY workspace like workspace_click_jog.py.
  - Draws forbidden overhead-frame regions.
  - Left-click plans a path from current robot pose to clicked XY at current Z.
  - If current Z is above a forbidden region's z_min, the planner avoids that XY box.
  - If no safe path exists, the move is refused.
  - Press b, then click two corners to create a new forbidden box.
  - Press w to save boxes to soft_limits_config.json.

Required in same folder:
  robot.py
  soft_limits.py

Controls:
  Left-click  : plan + execute safe XY move
  b           : define new forbidden box with two clicks
  w           : save boxes to JSON
  l           : load boxes from JSON
  [ / ]       : Z down/up
  h           : HOME
  c           : sync from Teensy POS
  e/d         : enable/disable
  p           : print pose
  q/Esc       : quit

IMPORTANT:
  This is not a certified collision avoidance system. It is a practical soft-limit
  guard that checks the commanded Cartesian path before sending moves.
"""

import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

from robot import Robot
from robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    SOFT_LIMITS_CONFIG_PATH,
    print_startup_config,
    require_robot_soft_limits_loaded,
    require_soft_limits_configured,
)
from camera_config import OVERHEAD_INDEX, STEREO_INDEX
from soft_limits import (
    SoftLimitConfig,
    ForbiddenBox,
    default_config,
    plan_safe_path,
    path_is_safe,
)

# ============================================================
# CONFIGURATION — EDIT THIS TO MATCH YOUR ROBOT
# ============================================================

CONFIG_JSON = SOFT_LIMITS_CONFIG_PATH

CANVAS_W = 800
CANVAS_H = 800
WORLD_HALF = 850.0
Z_STEP_MM = 5.0
PHI_STEP_DEG = 5.0
MOVE_TIME_PER_SEGMENT_S = 0.75

CLAW_OPEN_DEG = 70
CLAW_CLOSED_DEG = 20

# ============================================================
# HELPERS
# ============================================================

def load_soft_config() -> SoftLimitConfig:
    if CONFIG_JSON.exists():
        print(f"[SOFT] Loading {CONFIG_JSON.resolve()}")
        return SoftLimitConfig.load_json(CONFIG_JSON)
    print("[SOFT] Using default soft-limit config. Edit/save as needed.")
    return default_config()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Grocery Bagger Soft-Limit Workspace Test")

        self.world_half = WORLD_HALF
        self.robot: Robot | None = None
        self.busy = False
        self.soft_cfg = load_soft_config()
        self.pending_box_first_corner = None
        self.last_planned_path = []

        self._build_ui()
        self._draw()
        threading.Thread(target=self._connect, daemon=True).start()

    # -------------------------------------------------- UI

    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg="#171725", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(self.root, bg="#101827", width=270)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        def label(text, bold=False, color="#e5e7eb"):
            font = ("Consolas", 10, "bold") if bold else ("Consolas", 10)
            tk.Label(panel, text=text, bg="#101827", fg=color, font=font, anchor="w", wraplength=250, justify=tk.LEFT).pack(fill=tk.X, padx=8, pady=1)

        label("SOFT LIMIT TEST", True)
        self.lbl_status = tk.Label(panel, text="Connecting...", bg="#101827", fg="#fbbf24", font=("Consolas", 10), anchor="w", wraplength=250, justify=tk.LEFT)
        self.lbl_status.pack(fill=tk.X, padx=8, pady=4)

        tk.Frame(panel, bg="#374151", height=1).pack(fill=tk.X, pady=5)
        label("POSE", True)
        self.lbl_pose = tk.Label(panel, text="—", bg="#101827", fg="#ffffff", font=("Consolas", 10), anchor="w", justify=tk.LEFT)
        self.lbl_pose.pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(panel, bg="#374151", height=1).pack(fill=tk.X, pady=5)
        label("KEYS", True)
        for line in [
            "Left-click: safe move XY",
            "b: define forbidden box",
            "w/l: save/load boxes",
            "[ / ]: Z down/up",
            ", / .: phi -/+",
            "o/c: claw open/close",
            "h: home",
            "s: sync POS",
            "e/d: enable/disable",
            "p: print pose",
            "q/Esc: quit",
        ]:
            label(line)

        tk.Frame(panel, bg="#374151", height=1).pack(fill=tk.X, pady=5)
        label("FORBIDDEN BOXES", True)
        self.lbl_boxes = tk.Label(panel, text="", bg="#101827", fg="#c7d2fe", font=("Consolas", 9), anchor="w", wraplength=250, justify=tk.LEFT)
        self.lbl_boxes.pack(fill=tk.X, padx=8, pady=2)

        for txt, cmd in [
            ("HOME", self._cmd_home),
            ("Sync POS", self._cmd_sync),
            ("Enable", self._cmd_enable),
            ("Disable", self._cmd_disable),
            ("Save Boxes", self._save_boxes),
        ]:
            tk.Button(panel, text=txt, command=cmd, bg="#1f4f75", fg="white", relief=tk.FLAT).pack(fill=tk.X, padx=8, pady=2)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.root.bind("<KeyPress>", self._on_key)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    # -------------------------------------------------- coords/drawing

    def _world_to_canvas(self, wx, wy):
        cx = CANVAS_W / 2 + wx * (CANVAS_W / 2) / self.world_half
        cy = CANVAS_H / 2 - wy * (CANVAS_H / 2) / self.world_half
        return cx, cy

    def _canvas_to_world(self, cx, cy):
        wx = (cx - CANVAS_W / 2) * self.world_half / (CANVAS_W / 2)
        wy = -(cy - CANVAS_H / 2) * self.world_half / (CANVAS_H / 2)
        return wx, wy

    def _nice_grid_step(self):
        approx = self.world_half / 5
        for s in [25, 50, 100, 150, 200, 250, 300, 400, 500]:
            if approx <= s:
                return s
        return 500

    def _draw(self):
        self.canvas.delete("all")
        step = self._nice_grid_step()
        x = int(-self.world_half // step) * step
        while x <= self.world_half:
            cx, _ = self._world_to_canvas(x, 0)
            color = "#303044" if abs(x) > 1e-6 else "#4b5563"
            self.canvas.create_line(cx, 0, cx, CANVAS_H, fill=color)
            if abs(x) > 1e-6:
                self.canvas.create_text(cx + 3, CANVAS_H - 18, anchor="sw", fill="#6b7280", font=("Consolas", 8), text=f"{x:.0f}")
            x += step

        y = int(-self.world_half // step) * step
        while y <= self.world_half:
            _, cy = self._world_to_canvas(0, y)
            color = "#303044" if abs(y) > 1e-6 else "#4b5563"
            self.canvas.create_line(0, cy, CANVAS_W, cy, fill=color)
            if abs(y) > 1e-6:
                self.canvas.create_text(4, cy - 3, anchor="sw", fill="#6b7280", font=("Consolas", 8), text=f"{y:.0f}")
            y += step

        # reachability circles
        L1 = ROBOT_CONFIG.L1_mm
        L2 = ROBOT_CONFIG.L2_mm
        for r, color in [(abs(L1 - L2), "#7f1d1d"), (L1 + L2, "#14532d")]:
            cx, cy = self._world_to_canvas(0, 0)
            rp = r * (CANVAS_W / 2) / self.world_half
            self.canvas.create_oval(cx-rp, cy-rp, cx+rp, cy+rp, outline=color, width=2)

        # forbidden boxes
        for box in self.soft_cfg.boxes:
            be = box.expanded()
            x1, y1 = self._world_to_canvas(be.xmin, be.ymin)
            x2, y2 = self._world_to_canvas(be.xmax, be.ymax)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ef4444", width=3, dash=(6, 4))
            self.canvas.create_text(x1 + 5, y2 + 16, anchor="nw", fill="#fca5a5", font=("Consolas", 9), text=f"{box.name}\nz >= {box.z_min:.0f} mm")

        # planned path
        if self.last_planned_path:
            pts = [self._world_to_canvas(p[0], p[1]) for p in self.last_planned_path]
            for a, b in zip(pts[:-1], pts[1:]):
                self.canvas.create_line(*a, *b, fill="#fbbf24", width=3, arrow=tk.LAST)
            for i, p in enumerate(pts):
                self.canvas.create_oval(p[0]-4, p[1]-4, p[0]+4, p[1]+4, fill="#fbbf24", outline="")
                self.canvas.create_text(p[0]+6, p[1]-6, fill="#fde68a", anchor="sw", font=("Consolas", 8), text=str(i))

        if self.robot:
            self._draw_arm()
        self._update_labels()

    def _draw_arm(self):
        import math
        q = self.robot.q_est
        L1 = ROBOT_CONFIG.L1_mm
        L2 = ROBOT_CONFIG.L2_mm
        q1 = math.radians(q.q1_deg)
        q2 = math.radians(q.q2_deg)
        shoulder = (0.0, 0.0)
        elbow = (L1 * math.cos(q1), L1 * math.sin(q1))
        ee = (elbow[0] + L2 * math.cos(q1 + q2), elbow[1] + L2 * math.sin(q1 + q2))

        sp = self._world_to_canvas(*shoulder)
        ep = self._world_to_canvas(*elbow)
        tp = self._world_to_canvas(*ee)
        self.canvas.create_line(*sp, *ep, fill="#60a5fa", width=5, capstyle=tk.ROUND)
        self.canvas.create_line(*ep, *tp, fill="#2dd4bf", width=4, capstyle=tk.ROUND)
        for p, r, color in [(sp, 7, "#ffffff"), (ep, 6, "#60a5fa"), (tp, 7, "#f87171")]:
            self.canvas.create_oval(p[0]-r, p[1]-r, p[0]+r, p[1]+r, fill=color, outline="#111827")

    def _update_labels(self):
        if self.robot:
            q = self.robot.q_est
            x, y, z, phi = self.robot.fk()
            self.lbl_pose.config(text=f"x={x:+.1f} y={y:+.1f}\nz={z:+.1f} phi={phi:+.1f}\nq1={q.q1_deg:+.2f} q2={q.q2_deg:+.2f}")
        box_lines = []
        for b in self.soft_cfg.boxes:
            box_lines.append(f"{b.name}:\n  x {b.xmin:.0f}..{b.xmax:.0f}\n  y {b.ymin:.0f}..{b.ymax:.0f}\n  z>={b.z_min:.0f}, margin={b.margin_xy:.0f}")
        self.lbl_boxes.config(text="\n".join(box_lines) if box_lines else "none")

    # -------------------------------------------------- robot

    def _connect(self):
        self._set_status("Connecting...", "#fbbf24")
        try:
            self.robot = Robot(ROBOT_CONFIG, connect=True)
            require_robot_soft_limits_loaded(self.robot, "soft_limit_workspace_test.py")
            self._set_status("Connected. Enable/sync/home before moving.", "#22c55e")
            self.root.after(0, self._draw)
        except Exception as exc:
            self._set_status(f"Connect failed: {exc}", "#ef4444")

    def _set_status(self, text, color="#e5e7eb"):
        self.root.after(0, lambda: self.lbl_status.config(text=text, fg=color))

    def _execute_path(self, path):
        if self.busy or not self.robot:
            return
        def run():
            self.busy = True
            try:
                self._set_status(f"Executing {len(path)-1} segment path...", "#fbbf24")
                for i, p in enumerate(path[1:], start=1):
                    x, y, z = p
                    print(f"[MOVE] waypoint {i}/{len(path)-1}: x={x:.1f}, y={y:.1f}, z={z:.1f}")
                    ok = self.robot.move_cartesian(x_mm=x, y_mm=y, z_mm=z, move_time_s=MOVE_TIME_PER_SEGMENT_S)
                    if not ok:
                        self._set_status(f"Move failed at waypoint {i}", "#ef4444")
                        return
                    self.robot.sync_estimate_from_teensy_steps()
                self._set_status("Move complete", "#22c55e")
            finally:
                self.busy = False
                self.root.after(0, self._draw)
        threading.Thread(target=run, daemon=True).start()

    # -------------------------------------------------- events

    def _on_left_click(self, event):
        wx, wy = self._canvas_to_world(event.x, event.y)

        if self.pending_box_first_corner is not None:
            x0, y0 = self.pending_box_first_corner
            z_min = simpledialog.askfloat("Forbidden box", "Forbidden when z >= ? [mm]", initialvalue=80.0)
            if z_min is None:
                self.pending_box_first_corner = None
                self._set_status("Box creation cancelled", "#fbbf24")
                return
            name = f"box_{len(self.soft_cfg.boxes)+1}"
            self.soft_cfg.boxes.append(ForbiddenBox(name, min(x0, wx), max(x0, wx), min(y0, wy), max(y0, wy), z_min, margin_xy=25.0, margin_z=5.0))
            self.pending_box_first_corner = None
            self._set_status(f"Added {name}. Press w to save.", "#22c55e")
            self._draw()
            return

        if not self.robot or self.busy:
            return

        x0, y0, z0, _ = self.robot.fk()
        start = (float(x0), float(y0), float(z0))
        target = (float(wx), float(wy), float(z0))
        ok, path, reason = plan_safe_path(start, target, self.soft_cfg)
        self.last_planned_path = path
        self._draw()
        print(f"[PLAN] {reason}")
        for i, p in enumerate(path):
            print(f"  {i}: ({p[0]:+.1f}, {p[1]:+.1f}, {p[2]:+.1f})")
        if not ok:
            self._set_status(f"Refused: {reason}", "#ef4444")
            return
        self._set_status(reason, "#22c55e")
        self._execute_path(path)

    def _on_scroll(self, event):
        if event.delta > 0:
            self.world_half *= 0.88
        else:
            self.world_half *= 1.14
        self.world_half = max(150.0, min(2000.0, self.world_half))
        self._draw()

    def _on_key(self, event):
        k = event.char.lower() if event.char else event.keysym.lower()
        if k in ("q", "escape"):
            self.root.destroy()
        elif k == "b":
            self.pending_box_first_corner = None
            self._set_status("Box mode: click first corner, then second corner", "#fbbf24")
            self.pending_box_first_corner = "WAITING"
            # The next click stores actual coordinates.
            self.canvas.bind("<Button-1>", self._on_box_first_click)
        elif k == "w":
            self._save_boxes()
        elif k == "l":
            self._load_boxes()
        elif k == "h":
            self._cmd_home()
        elif k == "s":
            self._cmd_sync()
        elif k == "e":
            self._cmd_enable()
        elif k == "d":
            self._cmd_disable()
        elif k == "p" and self.robot:
            self.robot.print_estimate()
        elif k == "[":
            self._jog_z(-Z_STEP_MM)
        elif k == "]":
            self._jog_z(+Z_STEP_MM)
        elif k == ",":
            self._jog_phi(-PHI_STEP_DEG)
        elif k == ".":
            self._jog_phi(+PHI_STEP_DEG)
        elif k == "o" and self.robot:
            self.robot.servo(CLAW_OPEN_DEG)
        elif k == "c" and self.robot:
            # Note: c conflicts with close claw in some older scripts. Here c closes claw only if shifted? keep simple:
            self.robot.servo(CLAW_CLOSED_DEG)

    def _on_box_first_click(self, event):
        wx, wy = self._canvas_to_world(event.x, event.y)
        self.pending_box_first_corner = (wx, wy)
        self._set_status(f"First corner ({wx:.1f},{wy:.1f}). Click second corner.", "#fbbf24")
        self.canvas.bind("<Button-1>", self._on_left_click)

    # -------------------------------------------------- commands

    def _cmd_enable(self):
        if self.robot:
            threading.Thread(target=lambda: (self.robot.enable(True), self.robot.init_drivers()), daemon=True).start()

    def _cmd_disable(self):
        if self.robot:
            threading.Thread(target=lambda: self.robot.enable(False), daemon=True).start()

    def _cmd_home(self):
        if not self.robot or self.busy:
            return
        def run():
            self.busy = True
            self._set_status("Homing...", "#fbbf24")
            ok = self.robot.home()
            self._set_status("Homed" if ok else "Home failed", "#22c55e" if ok else "#ef4444")
            self.busy = False
            self.root.after(0, self._draw)
        threading.Thread(target=run, daemon=True).start()

    def _cmd_sync(self):
        if self.robot:
            self.robot.sync_estimate_from_teensy_steps()
            self._draw()

    def _jog_z(self, dz):
        if not self.robot or self.busy:
            return
        x, y, z, phi = self.robot.fk()
        target = (x, y, z + dz)
        ok, path, reason = plan_safe_path((x, y, z), target, self.soft_cfg)
        self.last_planned_path = path
        self._draw()
        if not ok:
            self._set_status(f"Z refused: {reason}", "#ef4444")
            return
        self._execute_path(path)

    def _jog_phi(self, dphi):
        if not self.robot or self.busy:
            return
        def run():
            self.busy = True
            q = self.robot.q_est
            ok = self.robot.move_cartesian(phi_deg=q.phi_deg + dphi)
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
            self.busy = False
            self.root.after(0, self._draw)
        threading.Thread(target=run, daemon=True).start()

    def _save_boxes(self):
        self.soft_cfg.save_json(CONFIG_JSON)
        self._set_status(f"Saved {CONFIG_JSON}", "#22c55e")

    def _load_boxes(self):
        self.soft_cfg = load_soft_config()
        self._set_status(f"Loaded {CONFIG_JSON}", "#22c55e")
        self._draw()


def main():
    print_startup_config("soft_limit_workspace_test.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("soft_limit_workspace_test.py")
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    if app.robot:
        app.robot.close()


if __name__ == "__main__":
    main()
