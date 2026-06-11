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

# "legacy"     — original per-candidate nearest match (one overhead det can match many stereo)
# "one_to_one" — new bijective matching: each OH det and each stereo cand matched at most once
OVERHEAD_MATCH_MODE: str = "legacy"
OVERHEAD_MATCH_SCORE_MODE: str = "bounded_distance"  # "bounded_distance" or "inverse_distance"


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


def _oh_match_score(dist_mm: float) -> float:
    if OVERHEAD_MATCH_SCORE_MODE == "inverse_distance":
        return 1.0 / max(float(dist_mm), 1e-6)
    return max(0.0, 1.0 - float(dist_mm) / max(1e-6, float(OVERHEAD_MATCH_MAX_DIST_MM)))


def match_overhead_xy_to_candidates_one_to_one(
    overhead_dets: list[YOLODetection],
    candidate_debugs: list[CandidateDebug],
    bundle: dict,
) -> list[YOLODetection | None]:
    """Bijective overhead→stereo matching: each detection and each candidate matched at most once.

    Sets candidate.overhead_centroid_px / overhead_bbox_px and debug.overhead_xy_mm on
    matched candidates exactly like match_overhead_xy_to_candidate() does for legacy mode.
    The caller is responsible for calling resolve_pick_phi() and apply_xy_blend() afterward.
    Prints an audit table.
    """
    if not overhead_dets or not candidate_debugs:
        print("[OVERHEAD MATCH ONE-TO-ONE] no overhead detections or no candidates")
        return [None] * len(candidate_debugs)

    n_oh = len(overhead_dets)
    n_st = len(candidate_debugs)

    # Build all pairwise distances and scores
    dist_matrix: list[list[float]] = [[float("inf")] * n_st for _ in range(n_oh)]
    xy_matrix: list[list[np.ndarray | None]] = [[None] * n_st for _ in range(n_oh)]
    same_class_matrix: list[list[bool]] = [[False] * n_st for _ in range(n_oh)]

    for oh_i, oh_det in enumerate(overhead_dets):
        for st_j, dbg in enumerate(candidate_debugs):
            cand = dbg.candidate
            stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))
            try:
                xy, *_ = project_overhead_centroid_to_robot_xy(oh_det.centroid_px, stereo_z, bundle)
                stereo_xy = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2)
                dist = float(np.linalg.norm(xy - stereo_xy))
                dist_matrix[oh_i][st_j] = dist
                xy_matrix[oh_i][st_j] = xy.copy()
                same_class_matrix[oh_i][st_j] = (
                    str(oh_det.class_name) == str(cand.yolo.class_name)
                )
            except Exception:
                pass

    # Hungarian-style greedy: prefer same-class pairs first, then by best score
    # Build candidate pairs sorted by score (best first)
    pairs: list[tuple[float, bool, int, int]] = []
    for oh_i in range(n_oh):
        for st_j in range(n_st):
            d = dist_matrix[oh_i][st_j]
            if d > float(OVERHEAD_MATCH_MAX_DIST_MM):
                continue
            score = _oh_match_score(d)
            same = same_class_matrix[oh_i][st_j]
            # same-class pairs sort first within the same score range
            pairs.append((-score, not same, oh_i, st_j))
    pairs.sort()  # ascending: best (lowest -score) first; same-class (False) before diff-class

    matched_oh: set[int] = set()
    matched_st: set[int] = set()
    assignments: dict[int, tuple[int, float, bool, np.ndarray]] = {}  # st_j → (oh_i, dist, same, xy)

    for _neg_score, _not_same, oh_i, st_j in pairs:
        if oh_i in matched_oh or st_j in matched_st:
            continue
        d = dist_matrix[oh_i][st_j]
        assignments[st_j] = (oh_i, d, same_class_matrix[oh_i][st_j], xy_matrix[oh_i][st_j])
        matched_oh.add(oh_i)
        matched_st.add(st_j)

    # Apply assignments and print audit table
    print("[OVERHEAD MATCH ONE-TO-ONE]")
    results: list[YOLODetection | None] = [None] * n_st
    for st_j, dbg in enumerate(candidate_debugs):
        cand = dbg.candidate
        if st_j in assignments:
            oh_i, dist, same, xy = assignments[st_j]
            oh_det = overhead_dets[oh_i]
            score = _oh_match_score(dist)
            class_tag = "same" if same else "diff"
            print(
                f"  OH[{oh_i}] {oh_det.class_name:14s} -> "
                f"ST[{getattr(cand, 'index', st_j)}] {cand.yolo.class_name:14s} "
                f"dist={dist:.1f}mm score={score:.3f} class={class_tag}"
            )
            cand.overhead_centroid_px = np.asarray(oh_det.centroid_px, dtype=np.float64).reshape(2)
            cand.overhead_bbox_px = tuple(float(v) for v in oh_det.bbox)
            dbg.overhead_xy_mm = xy.copy() if xy is not None else None
            if not same:
                print(f"    NOTE: class mismatch ({oh_det.class_name} vs {cand.yolo.class_name})")
            results[st_j] = oh_det
        else:
            # Find why: was best distance too far or already taken?
            best_d = min((dist_matrix[oh_i][st_j] for oh_i in range(n_oh)), default=float("inf"))
            if best_d > float(OVERHEAD_MATCH_MAX_DIST_MM):
                note = f"best_dist={best_d:.1f}mm > {OVERHEAD_MATCH_MAX_DIST_MM:.1f}"
            else:
                note = "best match already assigned to another candidate"
            print(
                f"  ST[{getattr(cand, 'index', st_j)}] {cand.yolo.class_name:14s} "
                f"-> stereo_only_no_overhead ({note})"
            )

    for oh_i, oh_det in enumerate(overhead_dets):
        if oh_i not in matched_oh:
            best_d = min((dist_matrix[oh_i][st_j] for st_j in range(n_st)), default=float("inf"))
            print(
                f"  OH[{oh_i}] {oh_det.class_name:14s} -> no match: "
                f"best_dist={best_d:.1f}mm > {OVERHEAD_MATCH_MAX_DIST_MM:.1f}"
                if best_d > float(OVERHEAD_MATCH_MAX_DIST_MM)
                else f"  OH[{oh_i}] {oh_det.class_name:14s} -> no match: all stereo candidates already matched"
            )

    return results


def match_overhead_xy_to_candidates(
    overhead_dets: list[YOLODetection],
    candidate_debugs: list[CandidateDebug],
    bundle: dict,
) -> list[YOLODetection | None]:
    return [match_overhead_xy_to_candidate(overhead_dets, dbg, bundle) for dbg in candidate_debugs]


_project_overhead_centroid_to_robot_xy = project_overhead_centroid_to_robot_xy
_weighted_xy = weighted_xy
_apply_xy_blend = apply_xy_blend
