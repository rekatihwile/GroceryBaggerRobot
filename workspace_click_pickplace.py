from __future__ import annotations

"""
workspace_click_pickplace.py

Combined manual jog GUI + camera-assisted pick/place helper.

What this adds over the separate scripts:
  - click-to-move / jog / serial controls from workspace_click_jog.py
  - live overhead + stereo camera panes
  - latch target geometry once, then keep using it after moving the robot away
  - set drop/place zone from the robot's current pose, no tag visibility required

Typical flow:
    1. Put target ID2 in view and press k, or press Capture Target if you want to save it first.
  2. Manually move the robot out of the way if needed.
  3. Jog the robot to a place location and press Set Drop From Current.
  4. Press Pick Latched, then Place Held.

Controls:
  Left-click        move robot to clicked XY
  Right-click       set absolute Z by popup
  [ / ]             decrease / increase Z
  , / .             decrease / increase phi
    g                 move to survey pose from robot config
  h/e/d/s/p         home / enable / disable / sync / print
    t                 capture current live target
    m                 move to current target or latched hover pose
    k                 pick current target, or fall back to latched target
  n                 set drop zone from current pose
  f                 place at stored drop zone
  o/l               open / close claw
  q / Escape        quit
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog

from robot import Robot
from robot_config import (
    ROBOT_CONFIG,
    print_startup_config,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    OVERHEAD_INDEX,
    STEREO_INDEX,
    STEREO_FOURCC,
    STEREO_FPS,
    STEREO_HEIGHT,
    STEREO_WIDTH,
    TARGET_TAG_ID,
)
from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
)
from test_calibration_bundle_live_stereo_z_pickplace import (
    BUNDLE_PATH,
    STEREO_CALIBRATION_PATH,
    CLAW_CLOSED_DEG,
    CLAW_OPEN_DEG,
    MAX_EE_ERROR_MM,
    WARN_EE_ERROR_MM,
    choose_safe_travel_z,
    compute_stereo_target_geometry,
    execute_pick,
    execute_place,
    load_bundle,
    load_stereo_calibration,
    map_uv_z_to_robot_xy,
    nearest_support_distance,
    print_geometry_sanity,
    read_stereo_tags_once,
    require_soft_limits,
    tag_phi_from_det,
)

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    _CAMERAS_AVAILABLE = True
except ImportError as exc:
    _CAMERAS_AVAILABLE = False
    print(f"[cameras] Optional deps not found ({exc}). Camera views disabled.")


Z_STEP_MM = 10.0
PHI_STEP_DEG = 5.0

CANVAS_W = 700
CANVAS_H = 700
WORLD_HALF = 950.0

CAMERA_ENABLE = True
CAMERA_UPDATE_MS = 60
OVERHEAD_DISPLAY_W = 640
STEREO_DISPLAY_W = 640


def clamp(value, low, high):
    return max(low, min(high, value))


class WorkspacePickPlaceGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robot Workspace - Click Move + PickPlace")

        self.world_half = WORLD_HALF
        self.robot: Robot | None = None
        self.busy = False

        self.bundle = None
        self.stereo_calib = None

        self._overhead_cap = None
        self._stereo_cam = None
        self._detector = None
        self._cam_update_id = None
        self._overhead_photo = None
        self._stereo_photo = None

        self.live_target = None
        self.latched_target = None
        self.last_live_geom = None
        self.has_item = False
        self.drop_zone_xy = None
        self.drop_zone_phi = None
        self.last_grasp_robot_z = None

        self._serial_entry = None
        self._serial_resp = None

        self._build_ui()
        self._draw_workspace()
        threading.Thread(target=self._connect, daemon=True).start()

    def _build_ui(self):
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#1a1a2e",
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        cam_bg = "#0d0d1a"
        cam_col = tk.Frame(self.root, bg=cam_bg)
        cam_col.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            cam_col,
            text="Overhead Camera",
            bg=cam_bg,
            fg="#9ec8ff",
            font=("Consolas", 10, "bold"),
        ).pack(pady=(6, 0))
        self._overhead_lbl = tk.Label(
            cam_col,
            bg="#111111",
            text="opening camera...",
            fg="#444444",
            font=("Consolas", 9),
            width=OVERHEAD_DISPLAY_W // 8,
            height=12,
        )
        self._overhead_lbl.pack(padx=4, pady=2)

        tk.Frame(cam_col, bg="#333", height=1).pack(fill=tk.X, pady=2)

        tk.Label(
            cam_col,
            text="Stereo Camera (Left | Right)",
            bg=cam_bg,
            fg="#9ec8ff",
            font=("Consolas", 10, "bold"),
        ).pack(pady=(4, 0))
        self._stereo_lbl = tk.Label(
            cam_col,
            bg="#111111",
            text="opening camera...",
            fg="#444444",
            font=("Consolas", 9),
            width=STEREO_DISPLAY_W // 8,
            height=8,
        )
        self._stereo_lbl.pack(padx=4, pady=(2, 6))

        panel = tk.Frame(self.root, bg="#16213e", width=320)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        def label(text, bold=False):
            font = ("Consolas", 10, "bold") if bold else ("Consolas", 10)
            tk.Label(panel, text=text, bg="#16213e", fg="#e0e0e0", font=font, anchor="w").pack(fill=tk.X, padx=8, pady=1)

        def value_row(attr, text):
            row = tk.Frame(panel, bg="#16213e")
            row.pack(fill=tk.X, padx=8)
            tk.Label(row, text=text, bg="#16213e", fg="#9ec8ff", font=("Consolas", 10), width=10, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="-", bg="#16213e", fg="#ffffff", font=("Consolas", 10), anchor="w")
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            setattr(self, attr, lbl)

        label("STATUS", bold=True)
        self.lbl_status = tk.Label(panel, text="Connecting...", bg="#16213e", fg="#f5a623", font=("Consolas", 10), anchor="w", wraplength=300, justify=tk.LEFT)
        self.lbl_status.pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("ESTIMATED POSE", bold=True)
        for attr, text in (("lbl_q1", "q1:"), ("lbl_q2", "q2:"), ("lbl_z", "z:"), ("lbl_phi", "phi:"), ("lbl_x", "x:"), ("lbl_y", "y:")):
            value_row(attr, text)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("LIVE TARGET", bold=True)
        for attr, text in (("lbl_live_xy", "xy:"), ("lbl_live_z", "z:"), ("lbl_live_phi", "phi:"), ("lbl_live_support", "support:")):
            value_row(attr, text)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("LATCHED TARGET", bold=True)
        for attr, text in (("lbl_latched_xy", "xy:"), ("lbl_latched_z", "z:"), ("lbl_latched_phi", "phi:"), ("lbl_pick_state", "state:")):
            value_row(attr, text)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("DROP ZONE", bold=True)
        for attr, text in (("lbl_drop_xy", "xy:"), ("lbl_drop_phi", "phi:"), ("lbl_drop_z", "place z:")):
            value_row(attr, text)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        button_specs = [
            ("HOME (h)", self._cmd_home),
            ("Assume Home (a)", self._cmd_assume_home),
            ("Survey Pose (g)", self._cmd_survey_pose),
            ("Enable (e)", self._cmd_enable),
            ("Disable (d)", self._cmd_disable),
            ("Sync (s)", self._cmd_sync),
            ("Print (p)", self._cmd_print),
            ("Capture Target (t)", self._cmd_capture_target),
            ("Move Hover (m)", self._cmd_move_hover),
            ("Pick Target (k)", self._cmd_pick_latched),
            ("Set Drop From Current (n)", self._cmd_set_drop_from_current),
            ("Place Held (f)", self._cmd_place_held),
            ("Open Claw (o)", self._cmd_open_claw),
            ("Close Claw (l)", self._cmd_close_claw),
            ("Geom Print (b)", self._cmd_print_geometry),
        ]
        for text, cmd in button_specs:
            tk.Button(panel, text=text, command=cmd, bg="#0f3460", fg="white", font=("Consolas", 10), relief=tk.FLAT, activebackground="#1a5276", activeforeground="white").pack(fill=tk.X, padx=8, pady=2)

        tk.Frame(panel, bg="#444", height=1).pack(fill=tk.X, pady=4)
        label("SERIAL CMD", bold=True)
        cmd_row = tk.Frame(panel, bg="#16213e")
        cmd_row.pack(fill=tk.X, padx=8, pady=2)
        self._serial_entry = tk.Entry(cmd_row, bg="#0a1628", fg="#e0e0e0", insertbackground="#e0e0e0", font=("Consolas", 9), relief=tk.FLAT)
        self._serial_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
        self._serial_entry.bind("<Return>", lambda _event: self._cmd_send_raw_serial())
        tk.Button(cmd_row, text="Send", command=self._cmd_send_raw_serial, bg="#0f3460", fg="white", font=("Consolas", 9), relief=tk.FLAT, activebackground="#1a5276", activeforeground="white").pack(side=tk.LEFT, padx=(3, 0))

        self._serial_resp = tk.Text(panel, height=4, bg="#050c14", fg="#4aff8f", font=("Consolas", 8), relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD)
        self._serial_resp.pack(fill=tk.X, padx=8, pady=(2, 0))

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Motion>", self._on_motion)
        self.root.bind("<KeyPress>", self._on_key)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self.hover_text = self.canvas.create_text(6, CANVAS_H - 6, anchor="sw", fill="#888888", font=("Consolas", 9), text="")

    def _connect(self):
        self._set_status("Loading calibration and connecting...", "#f5a623")
        try:
            require_soft_limits_configured("workspace_click_pickplace.py")
            self.bundle = load_bundle(BUNDLE_PATH)
            self.stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
            self.robot = Robot(ROBOT_CONFIG, connect=True)
            require_soft_limits(self.robot)
            self._set_status("Connected - use HOME, Sync, or Assume Home", "#4ae04a")
            self.root.after(0, self._refresh)
        except Exception as exc:
            self._set_status(f"Connect failed:\n{exc}", "#ff4a4a")
            return

        if _CAMERAS_AVAILABLE and CAMERA_ENABLE:
            self._open_cameras()

    def _open_cameras(self):
        try:
            self._detector = build_detector()
        except Exception as exc:
            print(f"[cameras] build_detector failed: {exc}")
            return

        try:
            self._overhead_cap = SimpleOverheadCamera()
            print(f"[cameras] Overhead opened (index {OVERHEAD_INDEX})")
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

        self.root.after(0, self._schedule_cam_update)

    def _schedule_cam_update(self):
        try:
            self._update_camera_views()
        except Exception as exc:
            print(f"[cameras] update error: {exc}")
        self._cam_update_id = self.root.after(CAMERA_UPDATE_MS, self._schedule_cam_update)

    def _update_camera_views(self):
        if not _CAMERAS_AVAILABLE:
            return

        live_target = None
        live_geom = None

        overhead_frame = None
        overhead_dets = {}
        if self._overhead_cap is not None:
            ok, frame = self._overhead_cap.read()
            if ok:
                overhead_frame = frame
                overhead_dets = detect_tags(self._detector, frame)

        stereo_left = None
        stereo_right = None
        det_l_all = {}
        det_r_all = {}
        stereo_tags = {}
        if self._stereo_cam is not None and self.stereo_calib is not None:
            stereo_tags, stereo_left, stereo_right, det_l_all, det_r_all = read_stereo_tags_once(
                self._stereo_cam,
                self._detector,
                self.stereo_calib,
            )

        if self.robot is not None:
            x_fk, y_fk, z_fk, _phi_fk = self.robot.fk()
        else:
            x_fk = y_fk = z_fk = 0.0

        det_ee_overhead = overhead_dets.get(EE_TAG_ID)
        det_tg_overhead = overhead_dets.get(TARGET_TAG_ID)

        if self.robot is not None and self.bundle is not None:
            live_geom = compute_stereo_target_geometry(self.robot, self.bundle, stereo_tags)
            self.last_live_geom = live_geom if live_geom is not None else self.last_live_geom

            if live_geom is not None and det_tg_overhead is not None:
                tg_xy, _tg_uv, _lo, _hi, _alpha = map_uv_z_to_robot_xy(
                    det_tg_overhead.center,
                    live_geom["lookup_z_used"],
                    self.bundle,
                )
                support_dist, _support_idx = nearest_support_distance(
                    tg_xy,
                    live_geom["lookup_z_used"],
                    self.bundle,
                )
                live_target = {
                    "xy": np.asarray(tg_xy, dtype=np.float64),
                    "phi": tag_phi_from_det(det_tg_overhead, live_geom["lookup_z_used"], self.bundle),
                    "support_dist": float(support_dist),
                    "geom": live_geom,
                    "captured_at": time.time(),
                }

            if det_ee_overhead is not None:
                ee_xy, _uv, _lo, _hi, _alpha = map_uv_z_to_robot_xy(det_ee_overhead.center, z_fk, self.bundle)
                ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                ee_err_norm = float(np.linalg.norm(ee_err))
                if ee_err_norm > MAX_EE_ERROR_MM:
                    status = "BAD"
                elif ee_err_norm > WARN_EE_ERROR_MM:
                    status = "WARN"
                else:
                    status = "OK"
                self._set_status(f"Connected | Overhead EE {status} |e|={ee_err_norm:.1f} mm", "#4ae04a")

        self.live_target = live_target
        self._update_target_labels()
        self._render_overhead_frame(overhead_frame, overhead_dets)
        self._render_stereo_frame(stereo_left, stereo_right, det_l_all, det_r_all)

    def _render_overhead_frame(self, frame, detections):
        if self._overhead_lbl is None:
            return
        if frame is None:
            self._overhead_lbl.config(text="overhead camera unavailable", image="")
            return

        drawn = frame.copy()
        for tag_id, det in detections.items():
            label = f"EE {tag_id}" if tag_id == EE_TAG_ID else (f"TGT {tag_id}" if tag_id == TARGET_TAG_ID else f"ID {tag_id}")
            drawn = draw_detection(drawn, det, label)
        drawn = cv2.resize(drawn, (OVERHEAD_DISPLAY_W, int(frame.shape[0] * OVERHEAD_DISPLAY_W / frame.shape[1])))
        rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        self._overhead_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._overhead_lbl.config(image=self._overhead_photo, width=drawn.shape[1], height=drawn.shape[0], text="")

    def _render_stereo_frame(self, left, right, dets_l, dets_r):
        if self._stereo_lbl is None:
            return
        if left is None or right is None:
            self._stereo_lbl.config(text="stereo camera unavailable", image="")
            return

        drawn_l = left.copy()
        for tag_id, det in dets_l.items():
            drawn_l = draw_detection(drawn_l, det, f"L {tag_id}")

        drawn_r = right.copy()
        for tag_id, det in dets_r.items():
            drawn_r = draw_detection(drawn_r, det, f"R {tag_id}")

        stereo_view = np.hstack([drawn_l, drawn_r])
        stereo_view = cv2.resize(stereo_view, (STEREO_DISPLAY_W, int(stereo_view.shape[0] * STEREO_DISPLAY_W / stereo_view.shape[1])))
        rgb = cv2.cvtColor(stereo_view, cv2.COLOR_BGR2RGB)
        self._stereo_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._stereo_lbl.config(image=self._stereo_photo, width=stereo_view.shape[1], height=stereo_view.shape[0], text="")

    def _draw_workspace(self):
        self.canvas.delete("all")
        self.hover_text = self.canvas.create_text(6, CANVAS_H - 6, anchor="sw", fill="#888888", font=("Consolas", 9), text="")

        l1 = ROBOT_CONFIG.L1_mm
        l2 = ROBOT_CONFIG.L2_mm
        max_r = l1 + l2
        min_r = abs(l1 - l2)

        step_mm = self._nice_grid_step()
        x = int(-self.world_half // step_mm) * step_mm
        while x <= self.world_half:
            cx, _ = self._world_to_canvas(x, 0)
            self.canvas.create_line(cx, 0, cx, CANVAS_H, fill="#2a2a3e", width=1)
            if abs(x) < step_mm * 0.1:
                self.canvas.create_line(cx, 0, cx, CANVAS_H, fill="#3a3a5e", width=1)
            if abs(x) > 0.1:
                self.canvas.create_text(cx + 2, CANVAS_H - 12, anchor="sw", fill="#444466", font=("Consolas", 7), text=f"{x:.0f}")
            x += step_mm

        y = int(-self.world_half // step_mm) * step_mm
        while y <= self.world_half:
            _, cy = self._world_to_canvas(0, y)
            self.canvas.create_line(0, cy, CANVAS_W, cy, fill="#2a2a3e", width=1)
            if abs(y) < step_mm * 0.1:
                self.canvas.create_line(0, cy, CANVAS_W, cy, fill="#3a3a5e", width=1)
            if abs(y) > 0.1:
                self.canvas.create_text(4, cy - 2, anchor="sw", fill="#444466", font=("Consolas", 7), text=f"{y:.0f}")
            y += step_mm

        ox, oy = self._world_to_canvas(0, 0)
        self.canvas.create_line(ox - 8, oy, ox + 8, oy, fill="#666688", width=1)
        self.canvas.create_line(ox, oy - 8, ox, oy + 8, fill="#666688", width=1)

        self._draw_circle(max_r, "#1e5c1e", width=2)
        self._draw_circle(min_r, "#5c1e1e", dash=(4, 4), width=1)

        if self.robot is not None:
            self._draw_arm()
        self._draw_saved_points()

    def _draw_circle(self, radius_mm, color, dash=None, width=1):
        cx, cy = self._world_to_canvas(0, 0)
        r_px = radius_mm * (CANVAS_W / 2) / self.world_half
        kwargs = {"outline": color, "width": width}
        if dash:
            kwargs["dash"] = dash
        self.canvas.create_oval(cx - r_px, cy - r_px, cx + r_px, cy + r_px, **kwargs)

    def _draw_arm(self):
        import math

        q = self.robot.q_est
        q1r = math.radians(q.q1_deg)
        q2r = math.radians(q.q2_deg)

        shoulder = (0.0, 0.0)
        elbow = (ROBOT_CONFIG.L1_mm * math.cos(q1r), ROBOT_CONFIG.L1_mm * math.sin(q1r))
        ee = (
            elbow[0] + ROBOT_CONFIG.L2_mm * math.cos(q1r + q2r),
            elbow[1] + ROBOT_CONFIG.L2_mm * math.sin(q1r + q2r),
        )

        def px(point):
            return self._world_to_canvas(*point)

        sp = px(shoulder)
        ep = px(elbow)
        tp = px(ee)
        self.canvas.create_line(*sp, *ep, fill="#4a9eff", width=4, capstyle=tk.ROUND)
        self.canvas.create_line(*ep, *tp, fill="#4aefcf", width=3, capstyle=tk.ROUND)
        self.canvas.create_oval(sp[0] - 7, sp[1] - 7, sp[0] + 7, sp[1] + 7, fill="#ffffff", outline="#4a9eff", width=2)
        self.canvas.create_oval(ep[0] - 5, ep[1] - 5, ep[0] + 5, ep[1] + 5, fill="#4a9eff", outline="#ffffff", width=1)
        self.canvas.create_oval(tp[0] - 6, tp[1] - 6, tp[0] + 6, tp[1] + 6, fill="#ff4a4a", outline="#ffffff", width=2)

    def _draw_saved_points(self):
        points = []
        if self.latched_target is not None:
            points.append((self.latched_target["xy"], "#ffd166", "TARGET"))
        if self.drop_zone_xy is not None:
            points.append((self.drop_zone_xy, "#7cf29a", "DROP"))

        for xy, color, label in points:
            cx, cy = self._world_to_canvas(float(xy[0]), float(xy[1]))
            self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, outline=color, width=2)
            self.canvas.create_text(cx + 12, cy - 12, text=label, anchor="sw", fill=color, font=("Consolas", 9, "bold"))

    def _update_labels(self):
        if self.robot is None:
            return
        q = self.robot.q_est
        x, y, z, phi = self.robot.fk()
        self.lbl_q1.config(text=f"{q.q1_deg:.2f} deg")
        self.lbl_q2.config(text=f"{q.q2_deg:.2f} deg")
        self.lbl_z.config(text=f"{q.z_mm:.2f} mm")
        self.lbl_phi.config(text=f"{q.phi_deg:.2f} deg")
        self.lbl_x.config(text=f"{x:.2f} mm")
        self.lbl_y.config(text=f"{y:.2f} mm")
        self._update_target_labels()

    def _update_target_labels(self):
        live = self.live_target
        if live is None:
            self.lbl_live_xy.config(text="not visible")
            self.lbl_live_z.config(text="-")
            self.lbl_live_phi.config(text="-")
            self.lbl_live_support.config(text="-")
        else:
            geom = live["geom"]
            xy = live["xy"]
            self.lbl_live_xy.config(text=f"({xy[0]:.1f}, {xy[1]:.1f})")
            self.lbl_live_z.config(text=f"hover {geom['hover_robot_z']:.1f} | grasp {geom['grasp_robot_z']:.1f}")
            self.lbl_live_phi.config(text="current" if live["phi"] is None else f"{float(live['phi']):.1f} deg")
            self.lbl_live_support.config(text=f"{live['support_dist']:.1f} mm")

        latched = self.latched_target
        if latched is None:
            self.lbl_latched_xy.config(text="none")
            self.lbl_latched_z.config(text="-")
            self.lbl_latched_phi.config(text="-")
            self.lbl_pick_state.config(text="empty" if not self.has_item else "holding")
        else:
            geom = latched["geom"]
            xy = latched["xy"]
            self.lbl_latched_xy.config(text=f"({xy[0]:.1f}, {xy[1]:.1f})")
            self.lbl_latched_z.config(text=f"hover {geom['hover_robot_z']:.1f} | grasp {geom['grasp_robot_z']:.1f}")
            self.lbl_latched_phi.config(text="current" if latched["phi"] is None else f"{float(latched['phi']):.1f} deg")
            state = "holding" if self.has_item else "ready"
            self.lbl_pick_state.config(text=f"{state} | support {latched['support_dist']:.1f}")

        if self.drop_zone_xy is None:
            self.lbl_drop_xy.config(text="none")
            self.lbl_drop_phi.config(text="-")
        else:
            self.lbl_drop_xy.config(text=f"({self.drop_zone_xy[0]:.1f}, {self.drop_zone_xy[1]:.1f})")
            self.lbl_drop_phi.config(text=f"{self.drop_zone_phi:.1f} deg")
        self.lbl_drop_z.config(text="-" if self.last_grasp_robot_z is None else f"{self.last_grasp_robot_z:.1f} mm")
        self._draw_workspace()

    def _set_status(self, msg, color="#e0e0e0"):
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color))

    def _refresh(self):
        self._draw_workspace()
        self._update_labels()

    def _nice_grid_step(self):
        approx = self.world_half / 4
        for step in (25, 50, 100, 150, 200, 250, 300, 400, 500):
            if approx <= step:
                return step
        return 500

    def _world_to_canvas(self, wx, wy):
        cx = CANVAS_W / 2 + wx * (CANVAS_W / 2) / self.world_half
        cy = CANVAS_H / 2 - wy * (CANVAS_H / 2) / self.world_half
        return cx, cy

    def _canvas_to_world(self, cx, cy):
        wx = (cx - CANVAS_W / 2) * self.world_half / (CANVAS_W / 2)
        wy = -(cy - CANVAS_H / 2) * self.world_half / (CANVAS_H / 2)
        return wx, wy

    def _run_robot_action(self, status_text, fn):
        if self.busy or self.robot is None:
            return

        def run():
            self.busy = True
            self._set_status(status_text, "#f5a623")
            try:
                fn()
            except Exception as exc:
                self._set_status(f"Action failed: {exc}", "#ff4a4a")
            finally:
                if self.robot is not None:
                    self.root.after(0, self._refresh)
                self.busy = False

        threading.Thread(target=run, daemon=True).start()

    def _move_to(self, x_mm, y_mm):
        def action():
            ok = self.robot.move_cartesian(x_mm=x_mm, y_mm=y_mm)
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                self._set_status(f"Moved to ({x_mm:.1f}, {y_mm:.1f})", "#4ae04a")
            else:
                self._set_status("Move failed or out of reach", "#ff4a4a")

        self._run_robot_action(f"Moving to ({x_mm:.1f}, {y_mm:.1f})...", action)

    def _jog_z(self, dz):
        def action():
            new_z = self.robot.q_est.z_mm + dz
            ok = self.robot.move_cartesian(z_mm=new_z)
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                self._set_status(f"Z -> {new_z:.1f} mm", "#4ae04a")
            else:
                self._set_status("Z jog failed", "#ff4a4a")

        self._run_robot_action(f"Jogging Z by {dz:+.1f} mm...", action)

    def _jog_phi(self, dphi):
        def action():
            new_phi = self.robot.q_est.phi_deg + dphi
            ok = self.robot.move_cartesian(phi_deg=new_phi)
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                self._set_status(f"phi -> {new_phi:.1f} deg", "#4ae04a")
            else:
                self._set_status("Phi jog failed", "#ff4a4a")

        self._run_robot_action(f"Jogging phi by {dphi:+.1f} deg...", action)

    def _cmd_home(self):
        def action():
            ok = self.robot.home()
            self._set_status("Homed" if ok else "HOME failed", "#4ae04a" if ok else "#ff4a4a")

        self._run_robot_action("Homing...", action)

    def _cmd_assume_home(self):
        if self.robot is None:
            return
        self.robot.assume_homed()
        self._set_status("Assumed configured home pose", "#4ae04a")
        self._refresh()

    def _cmd_survey_pose(self):
        def action():
            ok = self.robot.move_cartesian(
                x_mm=ROBOT_CONFIG.x_survey_mm,
                y_mm=ROBOT_CONFIG.y_survey_mm,
                z_mm=ROBOT_CONFIG.z_survey_mm,
                phi_deg=ROBOT_CONFIG.phi_survey_deg,
                move_time_s=1.0,
            )
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                self._set_status(
                    (
                        f"Moved to survey pose "
                        f"({ROBOT_CONFIG.x_survey_mm:.1f}, {ROBOT_CONFIG.y_survey_mm:.1f}, "
                        f"{ROBOT_CONFIG.z_survey_mm:.1f}, {ROBOT_CONFIG.phi_survey_deg:.1f})"
                    ),
                    "#4ae04a",
                )
            else:
                self._set_status("Survey move failed", "#ff4a4a")

        self._run_robot_action("Moving to survey pose...", action)

    def _cmd_enable(self):
        if self.robot is None:
            return
        self.robot.enable(True)
        self.robot.init_drivers()
        self._set_status("Motors enabled", "#4ae04a")

    def _cmd_disable(self):
        if self.robot is None:
            return
        self.robot.enable(False)
        self._set_status("Motors disabled", "#aaaaaa")

    def _cmd_sync(self):
        def action():
            ok = self.robot.sync_estimate_from_teensy_steps()
            self._set_status("Synced" if ok else "Sync failed", "#4ae04a" if ok else "#ff4a4a")

        self._run_robot_action("Syncing from Teensy...", action)

    def _cmd_print(self):
        if self.robot is not None:
            self.robot.print_estimate()

    def _cmd_capture_target(self):
        if self.live_target is None:
            messagebox.showwarning("Capture Target", "No live target available. Need overhead ID2 and stereo ID2 visible once.")
            return
        self.latched_target = self._clone_target(self.live_target)
        self._set_status("Latched current target", "#4ae04a")
        self._update_target_labels()

    def _clone_target(self, target: dict):
        return {
            "xy": target["xy"].copy(),
            "phi": target["phi"],
            "support_dist": target["support_dist"],
            "geom": {
                key: (value.copy() if hasattr(value, "copy") else value)
                for key, value in target["geom"].items()
            },
            "captured_at": time.time(),
        }

    def _get_pick_target(self, *, auto_latch_live: bool):
        if self.live_target is not None:
            target = self._clone_target(self.live_target)
            if auto_latch_live:
                self.latched_target = self._clone_target(self.live_target)
                self.root.after(0, self._update_target_labels)
            return target, "live"
        if self.latched_target is not None:
            return self.latched_target, "latched"
        return None, None

    def _cmd_move_hover(self):
        target, source = self._get_pick_target(auto_latch_live=True)
        if target is None:
            messagebox.showwarning("Move Hover", "Need a current live target or a previously captured target.")
            return

        def action():
            xy = target["xy"]
            geom = target["geom"]
            travel_z = choose_safe_travel_z(
                self.robot,
                float(xy[0]),
                float(xy[1]),
                geom["hover_robot_z"],
                geom["target_tag_robot_xyz_corrected"][2],
            )
            if travel_z is None:
                self._set_status("No soft-limit-safe hover travel Z", "#ff4a4a")
                return
            ok = self.robot.move_cartesian(z_mm=travel_z, move_time_s=0.75)
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                ok = self.robot.move_cartesian(
                    x_mm=float(xy[0]),
                    y_mm=float(xy[1]),
                    z_mm=travel_z,
                    phi_deg=self.robot.q_est.phi_deg,
                    move_time_s=1.10,
                )
            if ok:
                self.robot.sync_estimate_from_teensy_steps()
                self._set_status(f"Moved to {source} target hover", "#4ae04a")
            else:
                self._set_status("Hover move failed", "#ff4a4a")

        self._run_robot_action("Moving to latched hover...", action)

    def _cmd_pick_latched(self):
        target, source = self._get_pick_target(auto_latch_live=True)
        if target is None:
            messagebox.showwarning("Pick Target", "Need a current live target or a previously captured target.")
            return

        def action():
            ok, grasp_z = execute_pick(
                self.robot,
                target["xy"],
                target["phi"],
                target["geom"]["target_tag_robot_xyz_corrected"][2],
                target["geom"]["hover_robot_z"],
                target["geom"]["grasp_robot_z"],
            )
            self.has_item = bool(ok)
            if ok:
                self.last_grasp_robot_z = grasp_z
                self._set_status(f"Pick complete using {source} target", "#4ae04a")
            else:
                self._set_status("Pick failed", "#ff4a4a")

        self._run_robot_action("Running pick sequence...", action)

    def _cmd_set_drop_from_current(self):
        if self.robot is None:
            return
        x_cur, y_cur, _z_cur, phi_cur = self.robot.fk()
        self.drop_zone_xy = np.array([x_cur, y_cur], dtype=np.float64)
        self.drop_zone_phi = float(phi_cur)
        self._set_status(f"Drop zone set to ({x_cur:.1f}, {y_cur:.1f})", "#4ae04a")
        self._update_target_labels()

    def _cmd_place_held(self):
        if not self.has_item:
            messagebox.showwarning("Place Held", "No item held. Pick first.")
            return
        if self.drop_zone_xy is None or self.drop_zone_phi is None:
            messagebox.showwarning("Place Held", "Set drop zone from current pose first.")
            return
        if self.last_grasp_robot_z is None:
            messagebox.showwarning("Place Held", "No grasp height recorded from pick.")
            return

        def action():
            ok = execute_place(self.robot, self.drop_zone_xy, self.drop_zone_phi, self.last_grasp_robot_z)
            if ok:
                self.has_item = False
                self._set_status("Place complete", "#4ae04a")
            else:
                self._set_status("Place failed", "#ff4a4a")

        self._run_robot_action("Running place sequence...", action)

    def _cmd_open_claw(self):
        if self.robot is None:
            return
        self.robot.servo(CLAW_OPEN_DEG)
        self._set_status("Claw opened", "#4ae04a")

    def _cmd_close_claw(self):
        if self.robot is None:
            return
        self.robot.servo(CLAW_CLOSED_DEG)
        self._set_status("Claw closed", "#4ae04a")

    def _cmd_print_geometry(self):
        print_geometry_sanity(self.last_live_geom)

    def _cmd_send_raw_serial(self, cmd: str | None = None):
        if self.robot is None or not self.robot.ser:
            self._serial_log("[no serial connection]")
            return
        if cmd is None:
            cmd = self._serial_entry.get().strip()
            self._serial_entry.delete(0, tk.END)
        if not cmd:
            return
        self._serial_log(f"> {cmd}")
        timeout = 30.0 if cmd.upper().startswith("HOME") else 5.0
        terminal = {"DONE", "HOMED", "ZEROED", "ENABLED", "DISABLED", "READY"}

        def run():
            try:
                self.robot.ser.write((cmd + "\n").encode())
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if self.robot is None or not self.robot.ser:
                        break
                    line = self.robot.ser.readline().decode(errors="replace").strip()
                    if line:
                        self.root.after(0, lambda text=line: self._serial_log(text))
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
        total = int(self._serial_resp.index("end-1c").split(".")[0])
        if total > 50:
            self._serial_resp.delete("1.0", f"{total - 50}.0")
        self._serial_resp.config(state=tk.DISABLED)

    def _on_click(self, event):
        if self.robot is None:
            return
        import math

        wx, wy = self._canvas_to_world(event.x, event.y)
        radius = math.hypot(wx, wy)
        max_r = ROBOT_CONFIG.L1_mm + ROBOT_CONFIG.L2_mm
        min_r = abs(ROBOT_CONFIG.L1_mm - ROBOT_CONFIG.L2_mm)
        if radius > max_r:
            self._set_status(f"({wx:.0f}, {wy:.0f}) is outside max reach ({max_r:.0f} mm)", "#ff4a4a")
            return
        if radius < min_r:
            self._set_status(f"({wx:.0f}, {wy:.0f}) is inside min reach ({min_r:.0f} mm)", "#ff4a4a")
            return
        self._move_to(wx, wy)

    def _on_right_click(self, _event):
        if self.robot is None:
            return
        value = simpledialog.askfloat(
            "Set Z",
            f"Enter target Z (mm).\nCurrent: {self.robot.q_est.z_mm:.2f} mm",
            initialvalue=self.robot.q_est.z_mm,
            parent=self.root,
        )
        if value is not None:
            self._jog_z(value - self.robot.q_est.z_mm)

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.0 / 0.85
        self.world_half = clamp(self.world_half * factor, 100.0, 2000.0)
        self._draw_workspace()

    def _on_motion(self, event):
        import math

        wx, wy = self._canvas_to_world(event.x, event.y)
        radius = math.hypot(wx, wy)
        self.canvas.itemconfig(self.hover_text, text=f"({wx:.0f}, {wy:.0f}) mm   r={radius:.0f} mm")

    def _on_key(self, event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.root.destroy()
        elif key == "bracketleft":
            self._jog_z(-Z_STEP_MM)
        elif key == "bracketright":
            self._jog_z(+Z_STEP_MM)
        elif key == "comma":
            self._jog_phi(-PHI_STEP_DEG)
        elif key == "period":
            self._jog_phi(+PHI_STEP_DEG)
        elif key == "h":
            self._cmd_home()
        elif key == "a":
            self._cmd_assume_home()
        elif key == "g":
            self._cmd_survey_pose()
        elif key == "e":
            self._cmd_enable()
        elif key == "d":
            self._cmd_disable()
        elif key == "s":
            self._cmd_sync()
        elif key == "p":
            self._cmd_print()
        elif key == "t":
            self._cmd_capture_target()
        elif key == "m":
            self._cmd_move_hover()
        elif key == "k":
            self._cmd_pick_latched()
        elif key == "n":
            self._cmd_set_drop_from_current()
        elif key == "f":
            self._cmd_place_held()
        elif key == "o":
            self._cmd_open_claw()
        elif key == "l":
            self._cmd_close_claw()
        elif key == "b":
            self._cmd_print_geometry()


def main():
    print_startup_config("workspace_click_pickplace.py", OVERHEAD_INDEX, STEREO_INDEX)
    root = tk.Tk()
    root.resizable(True, True)
    app = WorkspacePickPlaceGUI(root)

    def on_close():
        if app._cam_update_id is not None:
            root.after_cancel(app._cam_update_id)
        if app.robot is not None:
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