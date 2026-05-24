from __future__ import annotations

"""Overhead/stereo XY matching and blend resolution for validation picks."""

import numpy as np

from test_calibration_bundle_live_stereo_z_pickplace import (
    clamp_lookup_z_to_bundle,
    map_uv_z_to_robot_xy,
    nearest_support_distance,
)
from vision.pick_candidate_builder import CandidateDebug
from vision.pick_z_resolver import resolve_pick_z_from_stereo
from vision.yolo_segmenter import YOLODetection


OVERHEAD_MATCH_MAX_DIST_MM: float = 140.0
OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True


def project_overhead_centroid_to_robot_xy(
    centroid_px: np.ndarray,
    z_mm: float,
    bundle: dict,
) -> tuple[np.ndarray, float, bool, float, int]:
    lookup_z, clamped = clamp_lookup_z_to_bundle(max(0.0, float(z_mm)), bundle)
    xy_raw, _uv_undist, _lo, _hi, _alpha = map_uv_z_to_robot_xy(
        np.asarray(centroid_px, dtype=np.float64).reshape(2),
        lookup_z,
        bundle,
    )
    xy = np.asarray(xy_raw, dtype=np.float64).reshape(2)
    support_dist, support_idx = nearest_support_distance(xy, lookup_z, bundle)
    return xy, float(lookup_z), bool(clamped), float(support_dist), int(support_idx)


def weighted_xy(
    stereo_xy: np.ndarray,
    overhead_xy: np.ndarray | None,
    w_overhead: float,
) -> tuple[np.ndarray, str]:
    st = np.asarray(stereo_xy, dtype=np.float64).reshape(2)
    if overhead_xy is None:
        return st.copy(), "stereo_only_no_overhead"

    w = float(np.clip(w_overhead, 0.0, 1.0))
    oh = np.asarray(overhead_xy, dtype=np.float64).reshape(2)
    xy = w * oh + (1.0 - w) * st
    return xy.astype(np.float64), f"weighted_overhead_{w:.2f}_stereo_{1.0 - w:.2f}"


def apply_xy_blend(
    debug: CandidateDebug,
    bundle: dict,
    w_overhead: float | None = None,
) -> None:
    cand = debug.candidate
    if w_overhead is None:
        w_overhead = debug.blend_weight_overhead

    z_debug = getattr(cand, "z_debug", None)
    if z_debug is not None:
        stereo_z = float(z_debug.robust_top_z_mm)
        hover_z = float(z_debug.hover_z_mm)
        grasp_z = float(z_debug.grasp_z_mm)
    else:
        z_plan = resolve_pick_z_from_stereo(float(cand.object_robot_xyz_raw[2]))
        stereo_z = z_plan.object_z
        hover_z = z_plan.hover_z
        grasp_z = z_plan.grasp_z
    debug.stereo_xy_mm = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2)

    overhead_xy = None
    if cand.overhead_centroid_px is not None:
        try:
            overhead_xy, lookup_z, lookup_clamped, support_dist, support_idx = project_overhead_centroid_to_robot_xy(
                cand.overhead_centroid_px,
                stereo_z,
                bundle,
            )
            cand.lookup_z_used = float(lookup_z)
            cand.lookup_z_clamped = bool(lookup_clamped)
            cand.support_distance_mm = float(support_dist)
            cand.support_index = int(support_idx)
        except Exception as exc:
            print(f"[XY BLEND] overhead mapping failed for cand[{cand.index}]: {exc}")
            cand.overhead_centroid_px = None
            overhead_xy = None

    debug.overhead_xy_mm = None if overhead_xy is None else overhead_xy.copy()

    xy, source = weighted_xy(debug.stereo_xy_mm, overhead_xy, float(w_overhead))
    debug.blend_weight_overhead = float(np.clip(w_overhead, 0.0, 1.0))

    if overhead_xy is not None:
        debug.xy_disagreement_mm = float(np.linalg.norm(overhead_xy - debug.stereo_xy_mm))
    else:
        debug.xy_disagreement_mm = None

    cand.target_xy = xy.copy()
    cand.target_xy_source_effective = source
    cand.object_robot_xyz_corrected = np.array([xy[0], xy[1], stereo_z], dtype=np.float64)
    cand.hover_robot_z = hover_z
    cand.grasp_robot_z = grasp_z


def match_overhead_xy_to_candidate(
    overhead_dets: list[YOLODetection],
    debug: CandidateDebug,
    bundle: dict,
) -> YOLODetection | None:
    cand = debug.candidate
    stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))
    stereo_xy = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2)

    pool = overhead_dets
    if OVERHEAD_MATCH_PREFER_SAME_CLASS:
        same = [d for d in overhead_dets if str(d.class_name) == str(cand.yolo.class_name)]
        if same:
            pool = same

    best_det = None
    best_dist = float("inf")
    best_xy = None

    for det in pool:
        try:
            xy, *_ = project_overhead_centroid_to_robot_xy(det.centroid_px, stereo_z, bundle)
            dist = float(np.linalg.norm(xy - stereo_xy))
        except Exception:
            continue
        if dist < best_dist:
            best_dist = dist
            best_det = det
            best_xy = xy

    if best_det is not None and best_dist <= OVERHEAD_MATCH_MAX_DIST_MM:
        cand.overhead_centroid_px = np.asarray(best_det.centroid_px, dtype=np.float64).reshape(2)
        cand.overhead_bbox_px = tuple(float(v) for v in best_det.bbox)
        debug.overhead_xy_mm = np.asarray(best_xy, dtype=np.float64).reshape(2) if best_xy is not None else None

        print(
            f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} "
            f"-> overhead {best_det.class_name:14s} dist={best_dist:6.1f} mm"
        )
        return best_det

    reason = "no projected overhead candidate" if best_det is None else f"best_dist={best_dist:.1f} > {OVERHEAD_MATCH_MAX_DIST_MM:.1f}"
    print(f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} -> NO MATCH ({reason})")
    return None


def match_overhead_xy_to_candidates(
    overhead_dets: list[YOLODetection],
    candidate_debugs: list[CandidateDebug],
    bundle: dict,
) -> list[YOLODetection | None]:
    return [match_overhead_xy_to_candidate(overhead_dets, dbg, bundle) for dbg in candidate_debugs]


_project_overhead_centroid_to_robot_xy = project_overhead_centroid_to_robot_xy
_weighted_xy = weighted_xy
_apply_xy_blend = apply_xy_blend
