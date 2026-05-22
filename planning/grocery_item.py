from __future__ import annotations

"""planning/grocery_item.py

GroceryItem data model for the 2D BLB bag planner.

Conversion from ObjectCandidate (run_pickplace_fast.py) is handled via
GroceryItem.from_object_candidate().  The classmethod uses safe placeholder
parsing so it can be called even if ObjectCandidate geometry is incomplete.
"""

from dataclasses import dataclass, field
import math
from typing import Any


PACKING_CLEARANCE_CM: float = 1.0
HEIGHT_PADDING_CM: float = 1.0
MIN_FOOTPRINT_CM: float = 2.0
MAX_REASONABLE_FOOTPRINT_CM: float = 40.0

_DEFAULT_HEIGHT_CM: float = 10.0
_DEFAULT_PADDED_RECT_XY_CM: tuple[float, float] = (12.0, 12.0)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_footprint_cm(width_cm: Any, depth_cm: Any) -> tuple[float, float] | None:
    width = _finite_float(width_cm)
    depth = _finite_float(depth_cm)
    if width is None or depth is None:
        return None
    if width <= 0.0 or depth <= 0.0:
        return None
    if width > MAX_REASONABLE_FOOTPRINT_CM or depth > MAX_REASONABLE_FOOTPRINT_CM:
        return None
    return (max(width, MIN_FOOTPRINT_CM), max(depth, MIN_FOOTPRINT_CM))


def _pair_from_value(value: Any) -> tuple[float, float] | None:
    try:
        values = tuple(value)
    except TypeError:
        return None
    if len(values) < 2:
        return None
    return _valid_footprint_cm(values[0], values[1])


def _candidate_topdown_footprint_cm(
    candidate: Any,
) -> tuple[tuple[float, float], str] | None:
    """Return measured overhead top-down footprint in cm when the pipeline has it.

    The main buildup runner projects matched overhead YOLO mask/bbox corners
    through map_uv_z_to_robot_xy() at the candidate's stereo height and stores
    the resulting robot-frame AABB on ObjectCandidate. This planning module
    intentionally reads only those already-projected dimensions so it does not
    need calibration bundle or homography imports.
    """
    width = getattr(candidate, "topdown_width_cm", None)
    depth = getattr(candidate, "topdown_depth_cm", None)
    footprint = _valid_footprint_cm(width, depth)
    if footprint is not None:
        return footprint, str(getattr(candidate, "topdown_footprint_source", None) or "overhead_projected_bbox")

    for attr_name in ("topdown_aabb_cm", "topdown_rect_xy_cm", "measured_rect_xy_cm"):
        footprint = _pair_from_value(getattr(candidate, attr_name, None))
        if footprint is not None:
            return footprint, str(getattr(candidate, "topdown_footprint_source", None) or "overhead_projected_bbox")

    return None


def _candidate_height_cm(candidate: Any) -> tuple[float | None, str | None]:
    """Return object height in cm from pointcloud/stereo candidate geometry.

    Explicit height fields win if a future pointcloud estimator provides them.
    Current ObjectCandidate instances expose the stereo/pointcloud object Z in
    object_robot_xyz_corrected/raw[2] in millimetres, so that becomes the live
    Z dimension when no explicit height field is present.
    """
    for attr_name, scale in (
        ("pointcloud_height_cm", 1.0),
        ("height_cm_from_pointcloud", 1.0),
        ("pointcloud_height_mm", 0.1),
        ("height_mm_from_pointcloud", 0.1),
    ):
        value = _finite_float(getattr(candidate, attr_name, None))
        if value is not None and value > 0.0:
            return value * scale, "pointcloud"

    for attr_name in ("object_robot_xyz_corrected", "object_robot_xyz_raw"):
        try:
            z_mm = _finite_float(getattr(candidate, attr_name)[2])
        except (AttributeError, IndexError, TypeError):
            continue
        if z_mm is not None and z_mm > 0.0:
            return z_mm * 0.1, "stereo_z"

    return None, None


def _fallback_padded_rect_xy_cm(db_entry: dict[str, Any]) -> tuple[tuple[float, float], str]:
    if "padded_rect_xy_cm" in db_entry:
        rect = _pair_from_value(db_entry.get("padded_rect_xy_cm"))
        if rect is not None:
            return rect, "attrs_db"

    if "padded_box_xyz_cm" in db_entry:
        rect = _pair_from_value(db_entry.get("padded_box_xyz_cm"))
        if rect is not None:
            return rect, "attrs_db"

    return _DEFAULT_PADDED_RECT_XY_CM, "default"


def _corners_as_tuples(value: Any) -> tuple[tuple[float, float], ...] | None:
    try:
        corners = [tuple(corner) for corner in value]
    except TypeError:
        return None
    result: list[tuple[float, float]] = []
    for corner in corners:
        if len(corner) < 2:
            return None
        x = _finite_float(corner[0])
        y = _finite_float(corner[1])
        if x is None or y is None:
            return None
        result.append((x, y))
    return tuple(result) if result else None


@dataclass
class GroceryItem:
    # Identity
    id: str
    class_name: str
    confidence: float | None

    # Coordinate estimates (mm)
    grasp_xyz_robot_mm: tuple[float, float, float] | None
    grasp_xyz_overhead: tuple[float, float, float] | None
    grasp_xyz_stereo_cam_mm: tuple[float, float, float] | None

    # Physical estimates
    cross_sectional_area_cm2: float      # footprint area
    height_cm: float                     # estimated object height
    weight_score: float                  # 0-1, higher = heavier (place first)
    fragility_score: float               # 0-1, higher = more fragile (place last / top)

    # 2D packing footprint in bag-plane cm
    padded_rect_xy_cm: tuple[float, float]   # (width_cm, depth_cm) with padding
    padded_box_xyz_cm: tuple[float, float, float]  # (w, d, h) with padding

    # Optional raw source / debug data
    source_candidate: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_object_candidate(
        cls,
        candidate: Any,
        attrs_db: dict[str, Any] | None = None,
        *,
        item_id: str | None = None,
    ) -> "GroceryItem":
        """Convert an ObjectCandidate from run_pickplace_fast.py into a GroceryItem.

        Uses pointcloud-derived height when available and conservative defaults
        where exact footprint geometry is not available.
        """
        # --- Basic identity ---
        class_name = ""
        confidence: float | None = None

        try:
            yolo = candidate.yolo
            class_name = str(yolo.class_name)
            confidence = float(yolo.confidence)
        except AttributeError:
            pass  # TODO: handle non-YOLO candidates if needed

        uid = item_id or f"{class_name}_{id(candidate)}"

        # --- Robot XYZ ---
        grasp_xyz_robot_mm: tuple[float, float, float] | None = None
        grasp_xyz_stereo_cam_mm: tuple[float, float, float] | None = None
        try:
            xyz = candidate.object_robot_xyz_corrected
            grasp_xyz_robot_mm = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except (AttributeError, IndexError, TypeError):
            pass

        try:
            cam_xyz = candidate.target_cam_xyz
            grasp_xyz_stereo_cam_mm = (float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2]))
        except (AttributeError, IndexError, TypeError):
            pass

        # --- Physical estimates from live geometry, attrs_db, or safe defaults ---
        db_entry: dict[str, Any] = {}
        if attrs_db and class_name in attrs_db and isinstance(attrs_db[class_name], dict):
            db_entry = attrs_db[class_name]

        weight_score = _finite_float(db_entry.get("weight_score"))
        fragility_score = _finite_float(db_entry.get("fragility_score"))
        weight_score = 0.5 if weight_score is None else weight_score
        fragility_score = 0.5 if fragility_score is None else fragility_score

        # Z dimension: use the stereo/pointcloud object height when present.
        # The buildup pipeline stores this as robot-frame object Z in mm;
        # attrs_db/default height is only a fallback for non-live or incomplete
        # ObjectCandidate data.
        db_height = _finite_float(db_entry.get("height_cm"))
        db_height_cm = db_height if db_height is not None and db_height > 0.0 else _DEFAULT_HEIGHT_CM
        live_height_cm, size_source_z = _candidate_height_cm(candidate)
        if live_height_cm is None:
            height_cm = db_height_cm
            size_source_z = "attrs_db/default"
        else:
            height_cm = live_height_cm

        # XY footprint: prefer the overhead-camera top-down robot-frame AABB.
        # That AABB is measured before packing clearance. If overhead geometry
        # is unavailable or bogus, padded_rect_xy_cm falls back to attrs_db (or
        # the conservative default), preserving offline/fake candidate behavior.
        measured_rect_xy_cm: tuple[float, float] | None = None
        size_source_xy_detail: str | None = None
        topdown = _candidate_topdown_footprint_cm(candidate)
        if topdown is not None:
            measured_rect_xy_cm, size_source_xy_detail = topdown
            size_source_xy = "overhead_projected_bbox"
            cross_sectional_area_cm2 = measured_rect_xy_cm[0] * measured_rect_xy_cm[1]
            padded_rect_xy_cm = (
                measured_rect_xy_cm[0] + 2.0 * PACKING_CLEARANCE_CM,
                measured_rect_xy_cm[1] + 2.0 * PACKING_CLEARANCE_CM,
            )
        else:
            padded_rect_xy_cm, size_source_xy = _fallback_padded_rect_xy_cm(db_entry)
            size_source_xy_detail = size_source_xy
            db_area = _finite_float(db_entry.get("cross_sectional_area_cm2"))
            cross_sectional_area_cm2 = db_area if db_area is not None and db_area > 0.0 else 100.0

        # Rectangular-prism AABB consumed by BLB: X/Y are the chosen padded
        # top-down footprint; Z is the live stereo/pointcloud height plus a
        # small release/packing margin.
        padded_box_xyz_cm = (
            padded_rect_xy_cm[0],
            padded_rect_xy_cm[1],
            height_cm + HEIGHT_PADDING_CM,
        )

        topdown_aabb_cm = _pair_from_value(getattr(candidate, "topdown_aabb_cm", None))
        if topdown_aabb_cm is None and measured_rect_xy_cm is not None:
            topdown_aabb_cm = measured_rect_xy_cm

        metadata = {
            "size_source_xy": size_source_xy,
            "size_source_xy_detail": size_source_xy_detail,
            "size_source_z": size_source_z,
            "measured_rect_xy_cm": measured_rect_xy_cm,
            "packing_clearance_cm": PACKING_CLEARANCE_CM,
            "height_padding_cm": HEIGHT_PADDING_CM,
            "topdown_aabb_cm": topdown_aabb_cm,
            "topdown_oriented_rect_cm": _pair_from_value(
                getattr(candidate, "topdown_oriented_rect_cm", None)
            ),
            "topdown_bbox_robot_xy_mm": _corners_as_tuples(
                getattr(candidate, "topdown_bbox_robot_xy_mm", None)
            ),
        }

        return cls(
            id=uid,
            class_name=class_name,
            confidence=confidence,
            grasp_xyz_robot_mm=grasp_xyz_robot_mm,
            grasp_xyz_overhead=None,  # TODO: wire in overhead homography XY
            grasp_xyz_stereo_cam_mm=grasp_xyz_stereo_cam_mm,
            cross_sectional_area_cm2=cross_sectional_area_cm2,
            height_cm=height_cm,
            weight_score=weight_score,
            fragility_score=fragility_score,
            padded_rect_xy_cm=padded_rect_xy_cm,
            padded_box_xyz_cm=padded_box_xyz_cm,
            source_candidate=candidate,
            metadata=metadata,
        )
