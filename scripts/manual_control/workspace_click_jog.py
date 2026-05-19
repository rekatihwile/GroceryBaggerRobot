# workspace_click_jog.py
#
# Click-to-move GUI for the grocery bagger robot.
# Shows the XY workspace as a canvas. Click to send the robot to that position.
# The drawn arm updates to reflect the estimated pose after each move.
#
# Requires only the Python standard library (tkinter) + robot.py.
#
# Controls:
#   Left-click        : move robot to clicked XY position
#   Right-click       : set Z offset (popup entry)
#   [ / ]             : decrease / increase Z by Z_STEP_MM
#   Mouse wheel       : zoom in/out
#   h                 : run HOME
#   e                 : enable motors
#   d                 : disable motors
#   p                 : print current estimate to console
#   s                 : sync estimate from Teensy step counters (no move)
#   q / Escape        : quit

import threading
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "run_pickplace_fast.py").exists()),
    Path(__file__).resolve().parents[1],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from hardware.robot import Robot
from config.robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
)
from config.camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    STEREO_FOURCC,
    STEREO_FPS,
    STEREO_HEIGHT,
    STEREO_INDEX,
    STEREO_WIDTH,
)
from hardware.cameras.overhead_camera import SimpleOverheadCamera

# Optional camera deps — the GUI works without them.
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    from hardware.cameras.stereo_apriltag_viewer import (
        SimpleStereoCamera,
        build_detector,
        detect_tags,
        draw_detection,
    )
    _CAMERAS_AVAILABLE = True
except ImportError as _cam_import_err:
    _CAMERAS_AVAILABLE = False
    print(f"[cameras] Optional deps not found ({_cam_import_err}). Camera views disabled.")

# ============================================================
# CONFIGURATION  —  Edit to match your setup
# ============================================================

# Step for Z keyboard jog (mm).
Z_STEP_MM = 10.0
# Step for phi keyboard jog (deg).
PHI_STEP_DEG = 5.0

# Canvas size in pixels.
CANVAS_W = 700
CANVAS_H = 700

# Initial view: world mm range shown on each axis (half-width).
WORLD_HALF = 950.0  # shows -950..+950 mm initially

# ============================================================
# CAMERA VIEW CONFIGURATION  —  Set CAMERA_ENABLE=False to skip
# ============================================================

CAMERA_ENABLE = True       # set False to run without cameras
EE_TAG_IDS    = [EE_TAG_ID]  # AprilTag IDs to detect and highlight

# Pixel widths for the camera column inside the main window.
# Both use the same width so they stack cleanly.
OVERHEAD_DISPLAY_W = 640
STEREO_DISPLAY_W   = 640

# How often the camera window refreshes (ms).
CAMERA_UPDATE_MS = 40   # ≈25 fps


# ============================================================
# HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# APPLICATION
# ============================================================

class WorkspaceGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Robot Workspace — Click to Move")

        # ---- view state ----
        self.world_half = WORLD_HALF   # mm half-width of the current view

        # ---- robot ----
        self.robot = None
        self.busy = False  # True while a move command is in flight

        # ---- cameras ----
        self._overhead_cap   = None
        self._stereo_cam     = None
        self._detector       = None
        self._overhead_lbl   = None   # created in _build_ui
        self._stereo_lbl     = None
        self._overhead_photo = None   # keep refs to prevent GC
        self._stereo_photo   = None
        self._cam_update_id  = None
        self._serial_entry   = None
        self._serial_resp    = None

        self._build_ui()
        self._draw_workspace()

        # Connect in a background thread so the window opens immediately.
        threading.Thread(target=self._connect, daemon=True).start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # Canvas
        self.canvas = tk.Canvas(
            self.root, width=CANVAS_W, height=CANVAS_H,
            bg="#1a1a2e", cursor="crosshair"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Camera column (middle) — labels filled by the update loop once cameras open
        _cbg = "#0d0d1a"
        cam_col = tk.Frame(self.root, bg=_cbg)
        cam_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 0))

        tk.Label(cam_col, text="Overhead Camera", bg=_cbg, fg="#9ec8ff",
                 font=("Consolas", 10, "bold")).pack(pady=(6, 0))
        self._overhead_lbl = tk.Label(cam_col, bg="#111111",
                                      text="opening camera…", fg="#444444",
                                      font=("Consolas", 9),
                                      width=OVERHEAD_DISPLAY_W // 8,
                                      height=12)
        self._overhead_lbl.pack(padx=4, pady=2)

        tk.Frame(cam_col, bg="#333", height=1).pack(fill=tk.X, pady=2)

        tk.Label(cam_col, text="Stereo Camera  (Left | Right)", bg=_cbg, fg="#9ec8ff",
                 font=("Consolas", 10, "bold")).pack(pady=(4, 0))
        self._stereo_lbl = tk.Label(cam_col, bg="#111111",
                                    text="opening camera…", fg="#444444",
                                    font=("Consolas", 9),
                                    width=STEREO_DISPLAY_W // 8,
                                    height=8)
        self._stereo_lbl.pack(padx=4, pady=(2, 6))

        # Side panel
        panel = tk.Frame(self.root, bg="#16213e", width=220)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        def label(text, bold=False):
            font = ("Consolas", 10, "bold") if bold else ("Consolas", 10)
            tk.Label(panel, text=text, bg="#16213e", fg="#e0e0e0", font=font,
                     anchor="w").pack(fill=tk.X, padx=8, pady=1)

        label("STATUS", bold=True)
        self.lbl_status = tk.Label(panel, text="Connecting…", bg="#16213e",
                                   fg="#f5a623", font=("Consolas", 10), anchor="w",
                                   wraplength=200, justify=tk.LEFT)
        self.lbl_status.pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("ESTIMATED POSE", bold=True)

        for attr, text in [("lbl_q1", "q1:"), ("lbl_q2", "q2:"),
                            ("lbl_z",  "z: "), ("lbl_phi","phi:")]:
            row = tk.Frame(panel, bg="#16213e")
            row.pack(fill=tk.X, padx=8)
            tk.Label(row, text=text, bg="#16213e", fg="#9ec8ff",
                     font=("Consolas", 10), width=5, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="—", bg="#16213e", fg="#ffffff",
                           font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            setattr(self, attr, lbl)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("CARTESIAN", bold=True)
        for attr, text in [("lbl_x","x:"), ("lbl_y","y:"),
                            ("lbl_zc","z:"), ("lbl_phic","phi:")]:
            row = tk.Frame(panel, bg="#16213e")
            row.pack(fill=tk.X, padx=8)
            tk.Label(row, text=text, bg="#16213e", fg="#9ec8ff",
                     font=("Consolas", 10), width=5, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="—", bg="#16213e", fg="#ffffff",
                           font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            setattr(self, attr, lbl)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("TARGET", bold=True)
        for attr, text in [("lbl_tx","x:"), ("lbl_ty","y:")]:
            row = tk.Frame(panel, bg="#16213e")
            row.pack(fill=tk.X, padx=8)
            tk.Label(row, text=text, bg="#16213e", fg="#9ec8ff",
                     font=("Consolas", 10), width=5, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="—", bg="#16213e", fg="#ffffff",
                           font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            setattr(self, attr, lbl)

        # Buttons
        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        for txt, cmd in [("HOME (h)",    self._cmd_home),
                         ("Enable (e)",  self._cmd_enable),
                         ("Disable (d)", self._cmd_disable),
                         ("Sync (s)",    self._cmd_sync),
                         ("Print (p)",   self._cmd_print)]:
            tk.Button(panel, text=txt, command=cmd,
                      bg="#0f3460", fg="white", font=("Consolas", 10),
                      relief=tk.FLAT, activebackground="#1a5276",
                      activeforeground="white").pack(fill=tk.X, padx=8, pady=2)

        # ---- Serial command interface ----
        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("SERIAL CMD", bold=True)

        cmd_row = tk.Frame(panel, bg="#16213e")
        cmd_row.pack(fill=tk.X, padx=8, pady=2)
        self._serial_entry = tk.Entry(
            cmd_row, bg="#0a1628", fg="#e0e0e0",
            insertbackground="#e0e0e0", font=("Consolas", 9), relief=tk.FLAT)
        self._serial_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._serial_entry.bind("<Return>", lambda e: self._cmd_send_raw_serial())
        tk.Button(cmd_row, text="Send", command=self._cmd_send_raw_serial,
                  bg="#0f3460", fg="white", font=("Consolas", 9),
                  relief=tk.FLAT, activebackground="#1a5276",
                  activeforeground="white").pack(side=tk.LEFT, padx=(3, 0))

        for _quick in [
            [("HOME",     "HOME"),   ("HOMEJ1",  "HOMEJ1"),
             ("HOMEJ2",   "HOMEJ2"), ("HOMEJ3",  "HOMEJ3")],
            [("POS",      "POS"),    ("LIMITS",  "LIMITS"),
             ("ZERO",     "ZERO"),   ("INITDRVS","INITDRIVERS")],
        ]:
            qrow = tk.Frame(panel, bg="#16213e")
            qrow.pack(fill=tk.X, padx=8, pady=1)
            for btn_label, serial_cmd in _quick:
                tk.Button(
                    qrow, text=btn_label,
                    command=lambda c=serial_cmd: self._cmd_send_raw_serial(c),
                    bg="#1a2a4e", fg="#9ec8ff", font=("Consolas", 7),
                    relief=tk.FLAT, padx=2, pady=1,
                    activebackground="#1a5276", activeforeground="white",
                ).pack(side=tk.LEFT, padx=1)

        self._serial_resp = tk.Text(
            panel, height=3, bg="#050c14", fg="#4aff8f",
            font=("Consolas", 8), relief=tk.FLAT,
            state=tk.DISABLED, wrap=tk.WORD)
        self._serial_resp.pack(fill=tk.X, padx=8, pady=(2, 0))

        # Command reference
        tk.Frame(panel, bg="#333", height=1).pack(fill=tk.X, pady=3)
        label("TEENSY CMDS", bold=True)
        _ref = tk.Text(
            panel, height=8, bg="#0a0f1a", fg="#666688",
            font=("Consolas", 7), relief=tk.FLAT,
            state=tk.NORMAL, wrap=tk.NONE)
        _ref.insert(tk.END,
            "HOME\n"
            "HOMEJ1 / HOMEJ2 / HOMEJ3\n"
            "ZERO  POS  LIMITS\n"
            "INITDRIVERS\n"
            "EN 1|0   SERVO <angle>\n"
            "MOVE j1 j2 j3 j4\n"
            "MOVESYNC j1 j2 j3 j4\n"
            "MOVESYNC_T j1..j4 ms")
        _ref.config(state=tk.DISABLED)
        _ref.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ---- event bindings ----
        self.canvas.bind("<Button-1>",      self._on_click)
        self.canvas.bind("<Button-3>",      self._on_right_click)
        self.canvas.bind("<MouseWheel>",    self._on_scroll)
        self.canvas.bind("<Motion>",        self._on_motion)
        self.root.bind("<KeyPress>",        self._on_key)
        self.root.bind("<Escape>",          lambda e: self.root.destroy())

        # Hover label on canvas
        self.hover_text = self.canvas.create_text(
            6, CANVAS_H - 6, anchor="sw", fill="#888888",
            font=("Consolas", 9), text="")

    # ------------------------------------------------------------------ coordinate helpers

    def _world_to_canvas(self, wx, wy):
        """World mm → canvas pixel. World y+ is up; canvas y+ is down."""
        cx = CANVAS_W / 2 + wx * (CANVAS_W / 2) / self.world_half
        cy = CANVAS_H / 2 - wy * (CANVAS_H / 2) / self.world_half
        return cx, cy

    def _canvas_to_world(self, cx, cy):
        wx = (cx - CANVAS_W / 2) * self.world_half / (CANVAS_W / 2)
        wy = -(cy - CANVAS_H / 2) * self.world_half / (CANVAS_H / 2)
        return wx, wy

    # ------------------------------------------------------------------ drawing

    def _draw_workspace(self):
        self.canvas.delete("all")

        # Re-create hover text on top after clearing.
        self.hover_text = self.canvas.create_text(
            6, CANVAS_H - 6, anchor="sw", fill="#888888",
            font=("Consolas", 9), text="")

        L1 = ROBOT_CONFIG.L1_mm
        L2 = ROBOT_CONFIG.L2_mm
        max_r = L1 + L2
        min_r = abs(L1 - L2)

        # Grid lines
        step_mm = self._nice_grid_step()
        x_start = int(-self.world_half // step_mm) * step_mm
        y_start = int(-self.world_half // step_mm) * step_mm
        x = x_start
        while x <= self.world_half:
            cx, _ = self._world_to_canvas(x, 0)
            self.canvas.create_line(cx, 0, cx, CANVAS_H, fill="#2a2a3e", width=1)
            if abs(x) < step_mm * 0.1:
                self.canvas.create_line(cx, 0, cx, CANVAS_H, fill="#3a3a5e", width=1)
            if abs(x) > 0.1:
                self.canvas.create_text(cx + 2, CANVAS_H - 12, anchor="sw",
                                        fill="#444466", font=("Consolas", 7),
                                        text=f"{x:.0f}")
            x += step_mm
        y = y_start
        while y <= self.world_half:
            _, cy = self._world_to_canvas(0, y)
            self.canvas.create_line(0, cy, CANVAS_W, cy, fill="#2a2a3e", width=1)
            if abs(y) < step_mm * 0.1:
                self.canvas.create_line(0, cy, CANVAS_W, cy, fill="#3a3a5e", width=1)
            if abs(y) > 0.1:
                self.canvas.create_text(4, cy - 2, anchor="sw",
                                        fill="#444466", font=("Consolas", 7),
                                        text=f"{y:.0f}")
            y += step_mm

        # Reachability annulus
        def draw_circle(r_mm, color, dash=None, width=1):
            cx, cy = self._world_to_canvas(0, 0)
            r_px = r_mm * (CANVAS_W / 2) / self.world_half
            kw = dict(outline=color, width=width)
            if dash:
                kw["dash"] = dash
            self.canvas.create_oval(cx - r_px, cy - r_px,
                                    cx + r_px, cy + r_px, **kw)

        draw_circle(max_r, "#1e5c1e", width=2)
        draw_circle(min_r, "#5c1e1e", dash=(4, 4), width=1)

        # Origin crosshair
        ox, oy = self._world_to_canvas(0, 0)
        self.canvas.create_line(ox - 8, oy, ox + 8, oy, fill="#666688", width=1)
        self.canvas.create_line(ox, oy - 8, ox, oy + 8, fill="#666688", width=1)

        # Robot arm
        if self.robot:
            self._draw_arm()

    def _draw_arm(self):
        q = self.robot.q_est
        L1 = ROBOT_CONFIG.L1_mm
        L2 = ROBOT_CONFIG.L2_mm

        import math
        q1r = math.radians(q.q1_deg)
        q2r = math.radians(q.q2_deg)

        shoulder = (0.0, 0.0)
        elbow = (L1 * math.cos(q1r),
                 L1 * math.sin(q1r))
        ee = (elbow[0] + L2 * math.cos(q1r + q2r),
              elbow[1] + L2 * math.sin(q1r + q2r))

        def px(pt):
            return self._world_to_canvas(*pt)

        sp = px(shoulder)
        ep = px(elbow)
        tp = px(ee)

        # Link 1
        self.canvas.create_line(*sp, *ep, fill="#4a9eff", width=4, capstyle=tk.ROUND)
        # Link 2
        self.canvas.create_line(*ep, *tp, fill="#4aefcf", width=3, capstyle=tk.ROUND)

        r_sh = 7
        r_el = 5
        r_ee = 6

        # Shoulder joint
        self.canvas.create_oval(sp[0]-r_sh, sp[1]-r_sh, sp[0]+r_sh, sp[1]+r_sh,
                                fill="#ffffff", outline="#4a9eff", width=2)
        # Elbow joint
        self.canvas.create_oval(ep[0]-r_el, ep[1]-r_el, ep[0]+r_el, ep[1]+r_el,
                                fill="#4a9eff", outline="#ffffff", width=1)
        # End-effector
        self.canvas.create_oval(tp[0]-r_ee, tp[1]-r_ee, tp[0]+r_ee, tp[1]+r_ee,
                                fill="#ff4a4a", outline="#ffffff", width=2)

    def _update_labels(self):
        if not self.robot:
            return
        q = self.robot.q_est
        x, y, z, phi = self.robot.fk()
        self.lbl_q1.config(text=f"{q.q1_deg:.2f}°")
        self.lbl_q2.config(text=f"{q.q2_deg:.2f}°")
        self.lbl_z.config(text=f"{q.z_mm:.2f} mm")
        self.lbl_phi.config(text=f"{q.phi_deg:.2f}°")
        self.lbl_x.config(text=f"{x:.2f} mm")
        self.lbl_y.config(text=f"{y:.2f} mm")
        self.lbl_zc.config(text=f"{z:.2f} mm")
        self.lbl_phic.config(text=f"{phi:.2f}°")

    def _nice_grid_step(self):
        approx = self.world_half / 4
        for s in [25, 50, 100, 150, 200, 250, 300, 400, 500]:
            if approx <= s:
                return s
        return 500

    # ------------------------------------------------------------------ robot connection

    def _connect(self):
        self._set_status("Connecting…", "#f5a623")
        try:
            self.robot = Robot(ROBOT_CONFIG, connect=True)
            self._set_status("Connected — enable motors to move", "#4ae04a")
            self.root.after(0, self._refresh)
        except Exception as exc:
            self._set_status(f"Connect failed:\n{exc}", "#ff4a4a")
        if _CAMERAS_AVAILABLE and CAMERA_ENABLE:
            self._open_cameras()

    def _set_status(self, msg, color="#e0e0e0"):
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color))

    def _refresh(self):
        self._draw_workspace()
        self._update_labels()

    # ------------------------------------------------------------------ cameras

    def _open_cameras(self):
        """Open cameras in the background thread, then schedule the GUI window."""
        try:
            self._detector = build_detector()
        except Exception as exc:
            print(f"[cameras] build_detector failed: {exc}")
            return

        try:
            cap = SimpleOverheadCamera().cap
            if cap.isOpened():
                self._overhead_cap = cap
                print(f"[cameras] Overhead opened (index {OVERHEAD_INDEX})")
            else:
                cap.release()
                print(f"[cameras] Could not open overhead index {OVERHEAD_INDEX}")
        except Exception as exc:
            print(f"[cameras] Overhead error: {exc}")

        try:
            self._stereo_cam = SimpleStereoCamera(
                index=STEREO_INDEX,
                width=STEREO_WIDTH,
                height=STEREO_HEIGHT,
                fps=STEREO_FPS,
                fourcc=STEREO_FOURCC,
            )
            print(f"[cameras] Stereo opened (index {STEREO_INDEX})")
        except Exception as exc:
            print(f"[cameras] Stereo error: {exc}")

        if self._overhead_cap is not None or self._stereo_cam is not None:
            self.root.after(0, self._schedule_cam_update)

    def _schedule_cam_update(self):
        try:
            self._update_camera_views()
        except Exception as exc:
            print(f"[cameras] update error: {exc}")
        self._cam_update_id = self.root.after(CAMERA_UPDATE_MS, self._schedule_cam_update)

    def _update_camera_views(self):
        # --- Overhead ---
        if self._overhead_cap is not None and self._overhead_cap.isOpened() \
                and self._overhead_lbl is not None:
            ok, frame = self._overhead_cap.read()
            if ok and frame is not None:
                dets = detect_tags(self._detector, frame)
                drawn = frame
                for det in dets.values():
                    drawn = draw_detection(drawn, det, f"ID {det.marker_id}")
                if not dets:
                    drawn = draw_detection(drawn, None, "No tags detected")
                dw = OVERHEAD_DISPLAY_W
                dh = int(frame.shape[0] * dw / frame.shape[1])
                drawn = cv2.resize(drawn, (dw, dh))
                rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
                self._overhead_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._overhead_lbl.config(image=self._overhead_photo, width=dw, height=dh)

        # --- Stereo ---
        if self._stereo_cam is not None and self._stereo_lbl is not None:
            ok, _, left, right = self._stereo_cam.read_pair()
            if ok and left is not None and right is not None:
                dets_l = detect_tags(self._detector, left)
                dets_r = detect_tags(self._detector, right)

                drawn_l = left
                for det in dets_l.values():
                    drawn_l = draw_detection(drawn_l, det, f"LEFT  ID {det.marker_id}")
                if not dets_l:
                    drawn_l = draw_detection(drawn_l, None, "LEFT — no tags")

                drawn_r = right
                for det in dets_r.values():
                    drawn_r = draw_detection(drawn_r, det, f"RIGHT ID {det.marker_id}")
                if not dets_r:
                    drawn_r = draw_detection(drawn_r, None, "RIGHT — no tags")

                stereo_view = np.hstack([drawn_l, drawn_r])
                dw = STEREO_DISPLAY_W
                dh = int(stereo_view.shape[0] * dw / stereo_view.shape[1])
                stereo_view = cv2.resize(stereo_view, (dw, dh))
                rgb = cv2.cvtColor(stereo_view, cv2.COLOR_BGR2RGB)
                self._stereo_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self._stereo_lbl.config(image=self._stereo_photo, width=dw, height=dh)

    def _move_to(self, x_mm, y_mm):
        if self.busy or not self.robot:
            return
        self.lbl_tx.config(text=f"{x_mm:.1f} mm")
        self.lbl_ty.config(text=f"{y_mm:.1f} mm")

        def run():
            self.busy = True
            self._set_status(f"Moving to ({x_mm:.1f}, {y_mm:.1f})…", "#f5a623")
            ok = self.robot.move_cartesian(x_mm=x_mm, y_mm=y_mm)
            if ok:
                self._set_status("Done", "#4ae04a")
                self.robot.sync_estimate_from_teensy_steps()
            else:
                self._set_status("Move failed or out of reach", "#ff4a4a")
            self.root.after(0, self._refresh)
            self.busy = False

        threading.Thread(target=run, daemon=True).start()

    def _jog_z(self, dz):
        if self.busy or not self.robot:
            return

        def run():
            self.busy = True
            new_z = self.robot.q_est.z_mm + dz
            self._set_status(f"Jogging Z to {new_z:.1f} mm…", "#f5a623")
            ok = self.robot.move_cartesian(z_mm=new_z)
            if ok:
                self._set_status("Done", "#4ae04a")
                self.robot.sync_estimate_from_teensy_steps()
            else:
                self._set_status("Z jog failed", "#ff4a4a")
            self.root.after(0, self._refresh)
            self.busy = False

        threading.Thread(target=run, daemon=True).start()

    def _jog_phi(self, dphi):
        if self.busy or not self.robot:
            return

        def run():
            self.busy = True
            new_phi = self.robot.q_est.phi_deg + dphi
            self._set_status(f"Rotating phi to {new_phi:.1f}°…", "#f5a623")
            ok = self.robot.move_cartesian(phi_deg=new_phi)
            if ok:
                self._set_status("Done", "#4ae04a")
                self.robot.sync_estimate_from_teensy_steps()
            else:
                self._set_status("Phi jog failed", "#ff4a4a")
            self.root.after(0, self._refresh)
            self.busy = False

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------ commands

    def _cmd_home(self):
        if self.busy or not self.robot:
            return
        def run():
            self.busy = True
            self._set_status("Homing…", "#f5a623")
            ok = self.robot.home()
            self._set_status("Homed" if ok else "HOME failed", "#4ae04a" if ok else "#ff4a4a")
            self.root.after(0, self._refresh)
            self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def _cmd_enable(self):
        if not self.robot:
            return
        def run():
            self.robot.enable(True)
            self._set_status("Motors enabled", "#4ae04a")
        threading.Thread(target=run, daemon=True).start()

    def _cmd_disable(self):
        if not self.robot:
            return
        def run():
            self.robot.enable(False)
            self._set_status("Motors disabled", "#aaaaaa")
        threading.Thread(target=run, daemon=True).start()

    def _cmd_sync(self):
        if self.busy or not self.robot:
            return
        def run():
            self.busy = True
            self._set_status("Syncing from Teensy…", "#f5a623")
            ok = self.robot.sync_estimate_from_teensy_steps()
            self._set_status("Synced" if ok else "Sync failed", "#4ae04a" if ok else "#ff4a4a")
            self.root.after(0, self._refresh)
            self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def _cmd_print(self):
        if self.robot:
            self.robot.print_estimate()

    def _cmd_send_raw_serial(self, cmd: str | None = None):
        if not self.robot or not self.robot.ser:
            self._serial_log("[no serial connection]")
            return
        if cmd is None:
            cmd = self._serial_entry.get().strip()
            self._serial_entry.delete(0, tk.END)
        if not cmd:
            return
        if self.busy:
            self._serial_log("[warn] robot busy — response may be incomplete")
        self._serial_log(f"> {cmd}")
        # HOME commands can take up to 30 s; everything else time out in 5 s.
        timeout = 30.0 if cmd.upper().startswith("HOME") else 5.0
        terminal = {"DONE", "HOMED", "ZEROED", "ENABLED", "DISABLED", "READY"}

        def run():
            try:
                self.robot.ser.write((cmd + "\n").encode())
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if not self.robot or not self.robot.ser:
                        break
                    line = self.robot.ser.readline().decode(errors="replace").strip()
                    if line:
                        self.root.after(0, lambda l=line: self._serial_log(l))
                        if line in terminal:
                            break
            except Exception as exc:
                self.root.after(0, lambda: self._serial_log(f"[err] {exc}"))

        threading.Thread(target=run, daemon=True).start()

    def _serial_log(self, text: str):
        if self._serial_resp is None:
            return
        self._serial_resp.config(state=tk.NORMAL)
        self._serial_resp.insert(tk.END, text + "\n")
        self._serial_resp.see(tk.END)
        # Cap at 50 lines to prevent unbounded growth.
        total = int(self._serial_resp.index("end-1c").split(".")[0])
        if total > 50:
            self._serial_resp.delete("1.0", f"{total - 50}.0")
        self._serial_resp.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ event handlers

    def _on_click(self, event):
        wx, wy = self._canvas_to_world(event.x, event.y)
        # Check reachability before sending.
        import math
        r = math.hypot(wx, wy)
        max_r = ROBOT_CONFIG.L1_mm + ROBOT_CONFIG.L2_mm
        min_r = abs(ROBOT_CONFIG.L1_mm - ROBOT_CONFIG.L2_mm)
        if r > max_r:
            self._set_status(f"({wx:.0f}, {wy:.0f}) is outside max reach ({max_r:.0f} mm)", "#ff4a4a")
            return
        if r < min_r:
            self._set_status(f"({wx:.0f}, {wy:.0f}) is inside min reach ({min_r:.0f} mm)", "#ff4a4a")
            return
        self._move_to(wx, wy)

    def _on_right_click(self, event):
        if not self.robot:
            return
        val = simpledialog.askfloat(
            "Set Z", f"Enter target Z (mm).\nCurrent: {self.robot.q_est.z_mm:.2f} mm",
            initialvalue=self.robot.q_est.z_mm,
            parent=self.root,
        )
        if val is not None:
            self._jog_z(val - self.robot.q_est.z_mm)

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.0 / 0.85
        self.world_half = clamp(self.world_half * factor, 100.0, 2000.0)
        self._draw_workspace()

    def _on_motion(self, event):
        wx, wy = self._canvas_to_world(event.x, event.y)
        import math
        r = math.hypot(wx, wy)
        self.canvas.itemconfig(self.hover_text,
                               text=f"({wx:.0f}, {wy:.0f}) mm   r={r:.0f} mm")

    def _on_key(self, event):
        k = event.keysym.lower()
        if k in ("q", "escape"):
            self.root.destroy()
        elif k == "bracketleft":
            self._jog_z(-Z_STEP_MM)
        elif k == "bracketright":
            self._jog_z(+Z_STEP_MM)
        elif k == "comma":
            self._jog_phi(-PHI_STEP_DEG)
        elif k == "period":
            self._jog_phi(+PHI_STEP_DEG)
        elif k == "h":
            self._cmd_home()
        elif k == "e":
            self._cmd_enable()
        elif k == "d":
            self._cmd_disable()
        elif k == "s":
            self._cmd_sync()
        elif k == "p":
            self._cmd_print()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    print_startup_config("workspace_click_jog.py", OVERHEAD_INDEX, STEREO_INDEX)
    root = tk.Tk()
    root.resizable(True, True)
    app = WorkspaceGUI(root)

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
