from __future__ import annotations

"""Gripper phi resolution for validation picks."""

import numpy as np

from vision.object_geometry import ObjectCandidate
from vision.pick_candidate_builder import CandidateDebug
from vision.pick_xy_resolver import project_overhead_centroid_to_robot_xy
from vision.pointcloud import estimate_mask_centroid_ray_angle_deg
from vision.yolo_segmenter import YOLODetection


PICK_PHI_MODE: str = "centroid_shortest_ray_parallel"
OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI: bool = True
USE_CONFIDENCE_PHI_BLEND: bool = False
PHI_DISAGREEMENT_WARN_DEG: float = 25.0
PHI_MIN_CONFIDENCE: float = 0.20
PHI_ASPECT_DECAY: float = 0.8
PHI_STEREO_HEIGHT_DECAY_CM: float = 8.0
PHI_FALLBACK_TO_CURRENT_EE_PHI: bool = False


def wrapped_phi_blend_deg(
    phi_a_deg: float,
    w_a: float,
    phi_b_deg: float,
    w_b: float,
) -> float:
    """Weighted blend of two angles that are equivalent mod 180 deg."""
    ang_a = np.deg2rad(2.0 * float(phi_a_deg))
    ang_b = np.deg2rad(2.0 * float(phi_b_deg))
    vx = float(w_a) * np.cos(ang_a) + float(w_b) * np.cos(ang_b)
    vy = float(w_a) * np.sin(ang_a) + float(w_b) * np.sin(ang_b)
    if vx == 0.0 and vy == 0.0:
        return float(0.5 * (float(phi_a_deg) + float(phi_b_deg)))
    return float(np.rad2deg(np.arctan2(vy, vx)) / 2.0)


def wrapped_phi_diff_deg(phi_a_deg: float, phi_b_deg: float) -> float:
    """Signed difference (phi_a - phi_b) wrapped to [-90, 90)."""
    return float(((float(phi_a_deg) - float(phi_b_deg) + 90.0) % 180.0) - 90.0)


def aspect_phi_confidence(det: YOLODetection) -> float:
    """Confidence that the YOLO mask's minor axis is well defined."""
    minor = max(float(det.minor_axis_length_px), 1.0)
    major = max(float(det.major_axis_length_px), minor)
    ratio = major / minor
    decay = max(1e-6, float(PHI_ASPECT_DECAY))
    return float(np.clip(1.0 - np.exp(-(ratio - 1.0) / decay), 0.0, 1.0))


def overhead_obliquity_confidence(det: YOLODetection, bundle: dict) -> float:
    """Penalize overhead phi for detections far from the overhead principal point."""
    K = bundle.get("overhead_camera_matrix")
    if K is None:
        return 1.0
    K = np.asarray(K, dtype=np.float64)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if cx <= 0.0 or cy <= 0.0:
        return 1.0
    dx = float(det.centroid_px[0]) - cx
    dy = float(det.centroid_px[1]) - cy
    radial = float(np.hypot(dx, dy)) / float(np.hypot(cx, cy))
    return float(np.clip(1.0 - radial, 0.0, 1.0))


def stereo_phi_confidence(cand: ObjectCandidate) -> float:
    """Penalize stereo phi when the object is tall."""
    h_cm = getattr(cand, "pointcloud_height_cm", None)
    if h_cm is None or not np.isfinite(float(h_cm)):
        return 1.0
    decay = max(1e-6, float(PHI_STEREO_HEIGHT_DECAY_CM))
    return float(np.clip(np.exp(-max(0.0, float(h_cm)) / decay), 0.0, 1.0))


def _overhead_centroid_ray_phi(
    det: YOLODetection,
    z_mm: float,
    bundle: dict,
    *,
    select: str,
    perpendicular: bool,
) -> tuple[float | None, str]:
    image_angle, source = estimate_mask_centroid_ray_angle_deg(
        det,
        select=select,
        perpendicular=perpendicular,
    )
    if image_angle is None:
        return None, source

    centroid = np.asarray(det.centroid_px, dtype=np.float64).reshape(2)
    theta = np.deg2rad(float(image_angle))
    axis_px = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    half_len_px = max(
        12.0,
        0.5 * float(det.minor_axis_length_px if select == "shortest" else det.major_axis_length_px),
    )
    p0 = centroid - axis_px * half_len_px
    p1 = centroid + axis_px * half_len_px

    try:
        xy0, *_ = project_overhead_centroid_to_robot_xy(p0, z_mm, bundle)
        xy1, *_ = project_overhead_centroid_to_robot_xy(p1, z_mm, bundle)
    except Exception as exc:
        print(f"[PHI] overhead centroid ray projection failed: {exc}")
        return None, f"{source}_overhead_projection_failed"

    delta = np.asarray(xy1, dtype=np.float64).reshape(2) - np.asarray(xy0, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-6:
        return None, f"{source}_overhead_degenerate_robot_delta"

    phi = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0)
    return phi, f"{source}_overhead"


def _overhead_semi_minor_phi(
    det: YOLODetection,
    z_mm: float,
    bundle: dict,
) -> tuple[float | None, str]:
    """Project the YOLO semi-minor axis direction through the overhead homography to robot XY.

    Unlike _overhead_centroid_ray_phi (which ray-casts to find the short/long
    chord), this uses the YOLO minor_axis_angle_deg directly — same projection
    math, different input angle.
    """
    image_angle = float(det.minor_axis_angle_deg)
    if not np.isfinite(image_angle):
        return None, "overhead_semi_minor_nan"

    centroid = np.asarray(det.centroid_px, dtype=np.float64).reshape(2)
    theta = np.deg2rad(image_angle)
    axis_px = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    half_len_px = max(12.0, 0.5 * float(det.minor_axis_length_px))
    p0 = centroid - axis_px * half_len_px
    p1 = centroid + axis_px * half_len_px

    try:
        xy0, *_ = project_overhead_centroid_to_robot_xy(p0, z_mm, bundle)
        xy1, *_ = project_overhead_centroid_to_robot_xy(p1, z_mm, bundle)
    except Exception as exc:
        print(f"[PHI] overhead semi-minor projection failed: {exc}")
        return None, "overhead_semi_minor_projection_failed"

    delta = np.asarray(xy1, dtype=np.float64).reshape(2) - np.asarray(xy0, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-6:
        return None, "overhead_semi_minor_degenerate_delta"

    phi = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0)
    return phi, "overhead_semi_minor_projected"


def resolve_pick_phi(
    debug: CandidateDebug,
    overhead_det: YOLODetection | None,
    bundle: dict,
) -> None:
    """Resolve cand.pick_phi_deg and fill debug phi metadata."""
    cand = debug.candidate
    stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))

    if PICK_PHI_MODE == "overhead_semi_minor_projected":
        phi: float | None = None
        source: str = "overhead_semi_minor_unavailable"
        if overhead_det is not None:
            phi, source = _overhead_semi_minor_phi(overhead_det, stereo_z, bundle)
        if phi is None:
            phi = cand.pick_phi_deg
            source = cand.pick_phi_source
        cand.pick_phi_deg = None if phi is None else float(phi)
        cand.pick_phi_source = source
        debug.phi_stereo_deg = cand.pick_phi_deg
        debug.phi_overhead_deg = cand.pick_phi_deg if overhead_det is not None else None
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = None
        debug.phi_blend_source = source
        return

    if PICK_PHI_MODE in {"centroid_longest_ray_perp", "centroid_shortest_ray_parallel"}:
        select = "longest" if PICK_PHI_MODE == "centroid_longest_ray_perp" else "shortest"
        perpendicular = PICK_PHI_MODE == "centroid_longest_ray_perp"
        phi: float | None = None
        source: str = "unknown"
        if overhead_det is not None:
            phi, source = _overhead_centroid_ray_phi(
                overhead_det,
                stereo_z,
                bundle,
                select=select,
                perpendicular=perpendicular,
            )
        if phi is None:
            phi = cand.pick_phi_deg
            source = cand.pick_phi_source
        cand.pick_phi_deg = None if phi is None else float(phi)
        cand.pick_phi_source = source
        debug.phi_stereo_deg = cand.pick_phi_deg
        debug.phi_overhead_deg = cand.pick_phi_deg if overhead_det is not None else None
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = None
        debug.phi_blend_source = source
        return

    if PICK_PHI_MODE == "pointcloud_shortest_path":
        cand.pick_phi_source = cand.pick_phi_source or "pointcloud_shortest_path"
        debug.phi_stereo_deg = cand.pick_phi_deg
        debug.phi_overhead_deg = None
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = None
        debug.phi_blend_source = cand.pick_phi_source
        return

    phi_stereo = float(cand.yolo.minor_axis_angle_deg)
    debug.phi_stereo_deg = phi_stereo if np.isfinite(phi_stereo) else None

    phi_overhead: float | None = None
    if overhead_det is not None and OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI:
        p = float(overhead_det.minor_axis_angle_deg)
        if np.isfinite(p):
            phi_overhead = p
    debug.phi_overhead_deg = phi_overhead

    if not USE_CONFIDENCE_PHI_BLEND:
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = (
            wrapped_phi_diff_deg(phi_overhead, phi_stereo)
            if (phi_overhead is not None and debug.phi_stereo_deg is not None) else None
        )
        if phi_overhead is not None:
            cand.pick_phi_deg = phi_overhead
            cand.pick_phi_source = "overhead_minor_axis"
            debug.phi_blend_source = cand.pick_phi_source
        return

    c_stereo = aspect_phi_confidence(cand.yolo) * stereo_phi_confidence(cand)
    debug.phi_stereo_confidence = float(c_stereo)

    c_overhead = 0.0
    if phi_overhead is not None:
        c_overhead = aspect_phi_confidence(overhead_det) * overhead_obliquity_confidence(overhead_det, bundle)
    debug.phi_overhead_confidence = float(c_overhead)

    if phi_overhead is not None and debug.phi_stereo_deg is not None:
        diff = wrapped_phi_diff_deg(phi_overhead, phi_stereo)
        debug.phi_disagreement_deg = diff
        if abs(diff) > float(PHI_DISAGREEMENT_WARN_DEG):
            print(
                f"[PHI WARN] cand[{cand.index}] overhead={phi_overhead:+.1f} "
                f"stereo={phi_stereo:+.1f} diff={diff:+.1f} deg  "
                f"c_oh={c_overhead:.2f} c_st={c_stereo:.2f}"
            )
    else:
        debug.phi_disagreement_deg = None

    usable: list[tuple[str, float, float]] = []
    if phi_overhead is not None and c_overhead >= float(PHI_MIN_CONFIDENCE):
        usable.append(("overhead", phi_overhead, c_overhead))
    if debug.phi_stereo_deg is not None and c_stereo >= float(PHI_MIN_CONFIDENCE):
        usable.append(("stereo", debug.phi_stereo_deg, c_stereo))

    if not usable:
        if PHI_FALLBACK_TO_CURRENT_EE_PHI:
            cand.pick_phi_deg = None
            cand.pick_phi_source = "fallback_current_ee_phi"
        else:
            if debug.phi_stereo_deg is not None:
                cand.pick_phi_deg = debug.phi_stereo_deg
                cand.pick_phi_source = "fallback_stereo_low_conf"
            else:
                cand.pick_phi_source = "fallback_no_phi_available"
        debug.phi_blend_source = cand.pick_phi_source
        return

    if len(usable) == 1:
        label, phi, conf = usable[0]
        cand.pick_phi_deg = float(phi)
        cand.pick_phi_source = f"{label}_minor_axis_c{conf:.2f}"
        debug.phi_blend_source = cand.pick_phi_source
        return

    (la, pa, ca), (lb, pb, cb) = usable
    total = ca + cb
    blended = wrapped_phi_blend_deg(pa, ca / total, pb, cb / total)
    cand.pick_phi_deg = float(blended)
    cand.pick_phi_source = f"blend_{la[:2]}{ca:.2f}_{lb[:2]}{cb:.2f}"
    debug.phi_blend_source = cand.pick_phi_source


_resolve_pick_phi = resolve_pick_phi
