from __future__ import annotations

"""Display and console printing helpers for real-object pick validation."""

from typing import Any

import cv2
import numpy as np

from vision.pick_candidate_builder import CandidateDebug, SurveyState


BURST_COUNT: int = 10
COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 560
STEREO_DRAW_H_PX: int = 390
STATUS_H_PX: int = 140
Z_MAX_MM: float = 275.0
GRIPPER_OFFSET_MM: float = 125.0
PLACE_RELEASE_GAP_MM: float = 8.0
PLACE_ZONE_FLOOR_Z_MM: float | None = None


def hr(title: str = "", char: str = "=", width: int = 96) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"{char * 3} {title} {char * pad}"[:width])
    else:
        print(char * width)


def fmt_xy(xy: Any) -> str:
    if xy is None:
        return "(--, --)"
    a = np.asarray(xy, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return "(--, --)"
    return f"({a[0]:7.1f}, {a[1]:7.1f})"


def fmt_xyz(xyz: Any) -> str:
    if xyz is None:
        return "(--, --, --)"
    a = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if a.size < 3:
        return "(--, --, --)"
    return f"({a[0]:7.1f}, {a[1]:7.1f}, {a[2]:7.1f})"


def fmt_candidate_phi(candidate: Any) -> str:
    if candidate.pick_phi_deg is None:
        return f"current ({candidate.pick_phi_source})"
    return f"{float(candidate.pick_phi_deg):+6.1f} ({candidate.pick_phi_source})"


def print_survey_candidate_summary(candidates: list[CandidateDebug]) -> None:
    hr("CANDIDATES", "-")
    if not candidates:
        print("[CANDIDATES] none")
        return

    for dbg in candidates:
        c = dbg.candidate
        dxy = "--" if dbg.xy_disagreement_mm is None else f"{dbg.xy_disagreement_mm:.1f}mm"
        z_debug = getattr(c, "z_debug", None)
        z_tag = ""
        if z_debug is not None and z_debug.top_spread_mm > 0.0:
            z_tag = f" zspread={z_debug.top_spread_mm:.1f} clear={z_debug.uncertainty_clearance_mm:.1f}"
        print(
            f"  [{c.index}] {c.yolo.class_name:14s} "
            f"hits={dbg.track.hit_count:2d}/{BURST_COUNT} "
            f"best_frame={dbg.best_frame_i + 1:02d} "
            f"pts={dbg.point_count:5d} "
            f"stereo_xy={fmt_xy(dbg.stereo_xy_mm)} "
            f"overhead_xy={fmt_xy(dbg.overhead_xy_mm)} "
            f"cmd_xy={fmt_xy(c.target_xy)} "
            f"dXY={dxy:>8s} "
            f"Z={c.object_robot_xyz_raw[2]:7.1f} "
            f"grasp={c.grasp_robot_z:7.1f} "
            f"phi={fmt_candidate_phi(c)}"
            f"{z_tag}"
        )


def print_validation(state: SurveyState | None, selected_index: int) -> None:
    if state is None or not state.candidates:
        print("[VALIDATE] no survey candidates. Press s first.")
        return

    hr("VALIDATION - ALL FRAME INFORMATION", "=")
    print(f"[BURST] frames captured={len(state.burst_frames)} requested={BURST_COUNT}")
    for frame in state.burst_frames:
        print(
            f"  frame {frame.frame_i + 1:02d}: "
            f"detections={len(frame.detections):2d} "
            + ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in frame.detections)
        )

    hr("TRACKS", "-")
    for track in state.tracks:
        keep = "KEEP" if track in state.kept_tracks else "drop"
        c = track.mean_centroid_px
        print(
            f"  track[{track.track_id:02d}] {keep:4s} {track.class_name:14s} "
            f"hits={track.hit_count:2d}/{BURST_COUNT} mean_px=({c[0]:.1f},{c[1]:.1f})"
        )

    hr("CANDIDATE GEOMETRY", "-")
    print_survey_candidate_summary(state.candidates)

    dbg = state.candidates[selected_index % len(state.candidates)]
    c = dbg.candidate

    hr(f"SELECTED [{c.index}] {c.yolo.class_name}", "-")
    print(f"best_frame_i           = {dbg.best_frame_i + 1}/{BURST_COUNT}")
    print(f"burst_hits             = {dbg.track.hit_count}/{BURST_COUNT}")
    print(f"valid_point_count      = {c.valid_point_count}")
    print(f"mask centroid px       = {np.round(c.yolo.centroid_px, 2)}")
    print(f"mask area px           = {c.yolo.mask_area}")
    print(f"bbox px                = {tuple(round(float(v), 1) for v in c.yolo.bbox)}")
    print(f"major axis px/deg      = {c.yolo.major_axis_length_px:.1f} @ {c.yolo.major_axis_angle_deg:.1f} deg")
    print(f"minor axis px/deg      = {c.yolo.minor_axis_length_px:.1f} @ {c.yolo.minor_axis_angle_deg:.1f} deg")
    print(f"pick phi               = {fmt_candidate_phi(c)}")
    print(f"phi_stereo_deg         = {dbg.phi_stereo_deg}")
    print(f"phi_overhead_deg       = {dbg.phi_overhead_deg}")
    print(f"phi_stereo_confidence  = {dbg.phi_stereo_confidence}")
    print(f"phi_overhead_confidence= {dbg.phi_overhead_confidence}")
    print(f"phi_disagreement_deg   = {dbg.phi_disagreement_deg}")
    print(f"phi_blend_source       = {dbg.phi_blend_source}")
    print()
    print(f"centroid_cam_xyz       = {fmt_xyz(c.centroid_cam_xyz)}")
    print(f"top_cam_xyz            = {fmt_xyz(c.top_cam_xyz)}")
    print(f"target_cam_xyz         = {fmt_xyz(c.target_cam_xyz)}")
    print()
    print(f"object_robot_xyz_raw   = {fmt_xyz(c.object_robot_xyz_raw)}")
    print(f"object_robot_xyz_corr  = {fmt_xyz(c.object_robot_xyz_corrected)}")
    print(f"stereo_xy              = {fmt_xy(dbg.stereo_xy_mm)}")
    print(f"overhead_centroid_px   = {None if c.overhead_centroid_px is None else np.round(c.overhead_centroid_px, 2)}")
    print(f"overhead_xy            = {fmt_xy(dbg.overhead_xy_mm)}")
    print(f"target_xy              = {fmt_xy(c.target_xy)}")
    print(f"xy_source              = {c.target_xy_source_effective}")
    print(f"blend_weight_overhead  = {dbg.blend_weight_overhead:.2f}")
    print(f"xy_disagreement_mm     = {dbg.xy_disagreement_mm}")
    print()
    print(f"lookup_z_used          = {c.lookup_z_used:.1f}  clamped={c.lookup_z_clamped}")
    print(f"support_distance_mm    = {c.support_distance_mm:.1f} nearest_idx={c.support_index}")
    print(f"pointcloud_height_cm   = {getattr(c, 'pointcloud_height_cm', None)}")
    print(f"pointcloud_z_range_mm  = {getattr(c, 'pointcloud_height_robot_z_range_mm', None)}")
    z_debug = getattr(c, "z_debug", None)
    if z_debug is not None:
        print()
        print("ROBUST Z DIAGNOSTICS")
        print(f"z_source               = {z_debug.source}")
        print(f"raw_min_z_mm           = {z_debug.raw_min_z_mm:.1f}")
        print(f"raw_max_z_mm           = {z_debug.raw_max_z_mm:.1f}")
        print(f"z_p50_mm               = {z_debug.z_p50_mm:.1f}")
        print(f"z_p90_mm               = {z_debug.z_p90_mm:.1f}")
        print(f"z_p95_mm               = {z_debug.z_p95_mm:.1f}")
        print(f"z_p99_mm               = {z_debug.z_p99_mm:.1f}")
        print(f"robust_top_z_mm        = {z_debug.robust_top_z_mm:.1f}")
        print(f"robust_bottom_z_mm     = {z_debug.robust_bottom_z_mm:.1f}")
        print(f"object_height_mm       = {z_debug.object_height_mm:.1f}")
        print(f"top_spread_mm          = {z_debug.top_spread_mm:.1f}")
        print(f"uncertainty_clearance  = {z_debug.uncertainty_clearance_mm:.1f}")
        print(f"final_grasp_z_mm       = {z_debug.grasp_z_mm:.1f}")
        if PLACE_ZONE_FLOOR_Z_MM is not None:
            dynamic_place_z = (
                float(PLACE_ZONE_FLOOR_Z_MM)
                + z_debug.object_height_mm
                + float(PLACE_RELEASE_GAP_MM)
                + z_debug.uncertainty_clearance_mm
            )
            dynamic_place_z = max(0.0, min(float(Z_MAX_MM), dynamic_place_z))
            print(f"dynamic_place_z_mm     = {dynamic_place_z:.1f}")
        elif z_debug.place_release_z_mm is not None:
            print(f"dynamic_place_z_mm     = {z_debug.place_release_z_mm:.1f}")
        print(f"z_warnings             = {z_debug.warnings}")
    print()
    print(f"Z_MAX_MM               = {Z_MAX_MM:.1f}")
    print(f"GRIPPER_OFFSET_MM      = {GRIPPER_OFFSET_MM:.1f}")
    print(f"hover_robot_z          = {c.hover_robot_z:.1f}")
    print(f"grasp_robot_z          = {c.grasp_robot_z:.1f}")
    print("=" * 96)


def put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def resize_to_panel(img: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
    if img is None:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        put_text_outline(out, label + " unavailable", (20, height // 2), scale=0.7, color=(80, 80, 255), thickness=2)
        return out
    out = cv2.resize(img, (width, height))
    put_text_outline(out, label, (10, 26), scale=0.65, color=(255, 255, 255), thickness=2)
    return out


def draw_overhead_panel(state: SurveyState | None, live_frame: np.ndarray | None) -> np.ndarray:
    frame = None
    if state is not None and state.overhead_frame is not None:
        frame = state.overhead_frame.copy()
    elif live_frame is not None:
        frame = live_frame.copy()

    if frame is None:
        return resize_to_panel(None, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX, "OVERHEAD")

    if state is not None:
        for i, det in enumerate(state.overhead_detections):
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cx, cy = np.round(det.centroid_px).astype(int)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
            put_text_outline(frame, f"OH {i} {det.class_name}", (x1, max(20, y1 - 6)), scale=0.5, color=(255, 160, 0), thickness=1)

        for idx, dbg in enumerate(state.candidates):
            c = dbg.candidate
            if c.overhead_centroid_px is None:
                continue
            px, py = np.round(c.overhead_centroid_px).astype(int)
            sel = idx == state.selected_index
            color = (0, 255, 0) if sel else (0, 220, 255)
            cv2.circle(frame, (int(px), int(py)), 14 if sel else 9, color, 2)
            put_text_outline(frame, str(idx + 1), (int(px) + 14, int(py) - 8), scale=0.65, color=color, thickness=2)

    return resize_to_panel(frame, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX, "OVERHEAD")


def make_display(
    state: SurveyState | None,
    live_overhead: np.ndarray | None,
    live_left: np.ndarray | None,
    live_right: np.ndarray | None,
) -> np.ndarray:
    W = COMBINED_WIDTH_PX
    top = draw_overhead_panel(state, live_overhead)

    half = W // 2
    if state is not None and state.candidates:
        dbg = state.candidates[state.selected_index % len(state.candidates)]
        left_panel = resize_to_panel(dbg.left_overlay, half, STEREO_DRAW_H_PX, f"BEST BURST LEFT #{state.selected_index + 1}")
        right_panel = resize_to_panel(dbg.disparity_overlay, half, STEREO_DRAW_H_PX, "RAFT DISPARITY + MASK")
    else:
        left_panel = resize_to_panel(live_left, half, STEREO_DRAW_H_PX, "LIVE LEFT")
        right_panel = resize_to_panel(live_right, half, STEREO_DRAW_H_PX, "LIVE RIGHT")

    bottom = np.hstack([left_panel, right_panel])

    status = np.zeros((STATUS_H_PX, W, 3), dtype=np.uint8)
    if state is None or not state.candidates:
        lines = [
            "No survey yet. Press s to run 10-frame burst survey.",
            "Keys: s=survey | r=rotate kept candidates | v=validate | t=test slow descent/no claw close | q=quit",
        ]
    else:
        dbg = state.candidates[state.selected_index % len(state.candidates)]
        c = dbg.candidate
        c_oh = dbg.phi_overhead_confidence
        c_st = dbg.phi_stereo_confidence
        phi_conf_tag = ""
        if c_oh is not None or c_st is not None:
            phi_conf_tag = f" [cOH={(c_oh or 0.0):.2f} cST={(c_st or 0.0):.2f}]"
        lines = [
            f"Selected [{state.selected_index + 1}/{len(state.candidates)}] {c.yolo.class_name} | hits={dbg.track.hit_count}/{BURST_COUNT} | best_frame={dbg.best_frame_i + 1}",
            f"cmd_xy={fmt_xy(c.target_xy)}  stereo={fmt_xy(dbg.stereo_xy_mm)}  overhead={fmt_xy(dbg.overhead_xy_mm)}  wOH={dbg.blend_weight_overhead:.2f}",
            f"Z={c.object_robot_xyz_raw[2]:.1f}  grasp={c.grasp_robot_z:.1f}  phi={fmt_candidate_phi(c)}{phi_conf_tag}  pts={c.valid_point_count}",
            "Keys: s=survey | r=rotate candidates | v=validate print | t=test descent/no close | q=quit",
        ]
        z_debug = getattr(c, "z_debug", None)
        if z_debug is not None:
            lines[2] += f"  zspread={z_debug.top_spread_mm:.1f} clear={z_debug.uncertainty_clearance_mm:.1f}"
    for i, line in enumerate(lines):
        put_text_outline(status, line, (8, 24 + i * 25), scale=0.58, color=(230, 230, 230), thickness=1)

    return np.vstack([top, bottom, status])


_hr = hr
_fmt_xy = fmt_xy
_fmt_xyz = fmt_xyz
_fmt_candidate_phi = fmt_candidate_phi
