from __future__ import annotations

"""
build_calibration_from_csv.py

Rebuild robot_calibration_bundle.npz + robot_calibration_report.txt from an
existing (possibly partial) robot_calibration_scan_raw.csv, without touching
the robot or cameras. Drop this next to calibrate_all_safe_grid.py.

Usage:
    python build_calibration_from_csv.py
    python build_calibration_from_csv.py path/to/some_other_scan.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

# Reuse everything from the main calibration script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calibrate_all_safe_grid_xyz_models import (  # type: ignore
    ScanRow,
    OUT_RAW_CSV,
    OUT_BUNDLE,
    OUT_REPORT,
    SCAN_MODE,
    SOFT_LIMITS_PATH,
    DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM,
    L1_TO_L2_HEIGHT_DIFFERENCE_MM,
    J3_ZERO_TO_EE_DROP_MM,
    STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM,
    STEREO_CAM_X_OFFSET_MM,
    STEREO_CAM_Y_OFFSET_MM,
    STEREO_CAM_Z_OFFSET_MM,
    MIN_STEREO_XYZ_FIT_SAMPLES,
    WARN_STEREO_XYZ_FIT_RMSE_MM,
    MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM,
    MAX_RUNTIME_HOMOGRAPHY_RMS_MM,
    robot_z_to_floor_height_mm,
    build_homography_layers,
    fit_stereo_models,
    build_visibility_arrays,
    build_support_arrays,
    build_reference_arrays,
    load_overhead_intrinsics,
)
from config.camera_config import (  # type: ignore
    EE_TAG_ID,
    OVERHEAD_INDEX,
    STEREO_INDEX,
    OVERHEAD_INTRINSICS_PATH,
)


# ---------------------------------------------------------------------------
# CSV -> ScanRow parsing
# ---------------------------------------------------------------------------

_STR_FIELDS = {"grid_name", "reason"}
_BOOL_FIELDS = {
    "is_reference_pose",
    "reachable_ik",
    "target_safe",
    "path_safe",
    "move_attempted",
    "move_ok",
    "overhead_visible",
    "overhead_used_for_homography",
    "stereo_visible",
}
_INT_FIELDS = {"overhead_samples", "stereo_samples"}
_REQUIRED_FLOAT_FIELDS = {
    "z_level_mm",
    "x_cmd_mm",
    "y_cmd_mm",
    "z_cmd_mm",
    "floor_height_cmd_mm",
}


def _parse_optional_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() == "none" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bool(s):
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() in ("true", "1", "yes")


def _parse_int(s):
    if s is None or s == "":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def load_rows_from_csv(csv_path: Path) -> list[ScanRow]:
    fields = list(ScanRow.__dataclass_fields__.keys())
    rows: list[ScanRow] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in fields if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(
                f"CSV {csv_path} is missing required columns: {missing}"
            )

        for raw in reader:
            kwargs = {}
            for field in fields:
                val = raw.get(field, "")
                if field in _STR_FIELDS:
                    kwargs[field] = val if val is not None else ""
                elif field in _BOOL_FIELDS:
                    kwargs[field] = _parse_bool(val)
                elif field in _INT_FIELDS:
                    kwargs[field] = _parse_int(val)
                elif field in _REQUIRED_FLOAT_FIELDS:
                    parsed = _parse_optional_float(val)
                    if parsed is None:
                        raise ValueError(
                            f"Required float field {field!r} missing in row: {raw}"
                        )
                    kwargs[field] = parsed
                else:
                    # Everything else is Optional[float].
                    kwargs[field] = _parse_optional_float(val)
            rows.append(ScanRow(**kwargs))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = OUT_RAW_CSV if len(sys.argv) < 2 else Path(sys.argv[1])
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    print(f"[LOAD] Reading rows from {csv_path.resolve()}")
    rows = load_rows_from_csv(csv_path)
    print(f"[LOAD] {len(rows)} rows loaded")
    n_move_ok = sum(r.move_ok for r in rows)
    n_ov_vis = sum(r.overhead_visible for r in rows)
    n_st_vis = sum(r.stereo_visible for r in rows)
    print(f"       move_ok={n_move_ok}  overhead_visible={n_ov_vis}  stereo_visible={n_st_vis}")

    if n_ov_vis == 0:
        print("[ABORT] No overhead-visible samples in CSV. Nothing to fit.")
        return

    # Derive the actual scanned grid from the CSV so the metadata matches
    # what we actually have, not what SCAN_PRESETS says today.
    z_levels = sorted({r.z_level_mm for r in rows if not r.is_reference_pose})
    x_grid = sorted({r.x_cmd_mm for r in rows if not r.is_reference_pose})
    y_grid = sorted({r.y_cmd_mm for r in rows if not r.is_reference_pose})

    print(f"[GRID] z_levels(robot_z) = {z_levels}")
    print(f"[GRID] x_grid            = {x_grid}")
    print(f"[GRID] y_grid            = {y_grid}")

    # Overhead intrinsics (file-only, no camera open).
    K_overhead, dist_overhead, used_overhead_intrinsics = load_overhead_intrinsics(
        OVERHEAD_INTRINSICS_PATH
    )

    # Soft-limits JSON for provenance.
    if SOFT_LIMITS_PATH.exists():
        soft_limits_json_text = SOFT_LIMITS_PATH.read_text(encoding="utf-8")
    else:
        print(f"[WARN] {SOFT_LIMITS_PATH} not found; saving empty soft-limits JSON.")
        soft_limits_json_text = ""

    # Fit models. These are pure post-processing.
    H_bundle, H_report = build_homography_layers(rows)
    stereo_bundle, stereo_report = fit_stereo_models(rows)

    # Report
    report_lines: list[str] = []
    report_lines.append("ROBOT CALIBRATION BUNDLE REPORT (rebuilt from CSV)")
    report_lines.append("=" * 72)
    report_lines.append(f"Generated:    {time.ctime()}")
    report_lines.append(f"Source CSV:   {csv_path.resolve()}")
    report_lines.append(f"Scan mode (per current config): {SCAN_MODE}")
    report_lines.append("Floor-plane height convention:")
    report_lines.append(f"  DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM={DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM}")
    report_lines.append(f"  L1_TO_L2_HEIGHT_DIFFERENCE_MM={L1_TO_L2_HEIGHT_DIFFERENCE_MM}")
    report_lines.append(f"  J3_ZERO_TO_EE_DROP_MM={J3_ZERO_TO_EE_DROP_MM}")
    report_lines.append(f"  EE_floor_height = {robot_z_to_floor_height_mm(0.0):.1f} + robot_z")
    report_lines.append(f"Rows in CSV:            {len(rows)}")
    report_lines.append(f"Moves OK:               {n_move_ok}")
    report_lines.append(f"Overhead visible:       {n_ov_vis}")
    report_lines.append(f"Stereo visible:         {n_st_vis}")
    report_lines.append(
        f"Reference poses visible: {sum(r.is_reference_pose and r.overhead_visible for r in rows)}"
    )
    report_lines.append("")
    report_lines.append("Runtime confidence metadata:")
    report_lines.append(f"  MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM={MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM}")
    report_lines.append(f"  MAX_RUNTIME_HOMOGRAPHY_RMS_MM={MAX_RUNTIME_HOMOGRAPHY_RMS_MM}")
    report_lines.append("")
    report_lines.extend(H_report)
    report_lines.extend(stereo_report)
    report_lines.append("")
    report_lines.append("Soft limits JSON used at rebuild time:")
    report_lines.append(soft_limits_json_text)

    # Build the save dict identically to save_bundle() in the original script.
    visibility_table, visibility_cols = build_visibility_arrays(rows)

    save_dict = {
        "created_unix_time": np.asarray([time.time()], dtype=np.float64),
        "scan_mode": np.asarray([SCAN_MODE]),
        "x_grid_mm": np.asarray(x_grid, dtype=np.float64),
        "y_grid_mm": np.asarray(y_grid, dtype=np.float64),
        "requested_z_levels_mm": np.asarray(z_levels, dtype=np.float64),
        "requested_floor_heights_mm": np.asarray(
            [robot_z_to_floor_height_mm(z) for z in z_levels], dtype=np.float64
        ),
        "distance_from_origin_to_floor_plane_mm": np.asarray(
            [DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM], dtype=np.float64
        ),
        "l1_to_l2_height_difference_mm": np.asarray(
            [L1_TO_L2_HEIGHT_DIFFERENCE_MM], dtype=np.float64
        ),
        "j3_zero_to_ee_drop_mm": np.asarray([J3_ZERO_TO_EE_DROP_MM], dtype=np.float64),
        "robot_z_zero_floor_height_mm": np.asarray(
            [robot_z_to_floor_height_mm(0.0)], dtype=np.float64
        ),
        "stereo_ee_tag_to_tool_offset_robot_mm": STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM,
        "stereo_cam_offsets_mm": np.asarray(
            [STEREO_CAM_X_OFFSET_MM, STEREO_CAM_Y_OFFSET_MM, STEREO_CAM_Z_OFFSET_MM],
            dtype=np.float64,
        ),
        "min_stereo_xyz_fit_samples": np.asarray([MIN_STEREO_XYZ_FIT_SAMPLES], dtype=np.int32),
        "warn_stereo_xyz_fit_rmse_mm": np.asarray([WARN_STEREO_XYZ_FIT_RMSE_MM], dtype=np.float64),
        "ee_tag_id": np.asarray([EE_TAG_ID], dtype=np.int32),
        "overhead_index": np.asarray([OVERHEAD_INDEX], dtype=np.int32),
        "stereo_index": np.asarray([STEREO_INDEX], dtype=np.int32),
        "used_overhead_intrinsics": np.asarray([bool(used_overhead_intrinsics)]),
        "overhead_camera_matrix": np.asarray(
            K_overhead if K_overhead is not None else np.eye(3), dtype=np.float64
        ),
        "overhead_dist_coeffs": np.asarray(
            dist_overhead if dist_overhead is not None else np.zeros((1, 5)),
            dtype=np.float64,
        ),
        "visibility_table": visibility_table,
        "visibility_columns": visibility_cols,
        "soft_limits_json": np.asarray([soft_limits_json_text]),
        "max_runtime_nearest_sample_dist_mm": np.asarray(
            [MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM], dtype=np.float64
        ),
        "max_runtime_homography_rms_mm": np.asarray(
            [MAX_RUNTIME_HOMOGRAPHY_RMS_MM], dtype=np.float64
        ),
    }

    save_dict.update(build_support_arrays(rows))
    save_dict.update(build_reference_arrays(rows))

    if H_bundle is not None:
        save_dict.update(H_bundle)
    if stereo_bundle is not None:
        save_dict.update(stereo_bundle)

    np.savez(OUT_BUNDLE, **save_dict)
    print(f"[SAVE] Bundle -> {OUT_BUNDLE.resolve()}")

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[SAVE] Report -> {OUT_REPORT.resolve()}")

    print("\n[DONE] Bundle rebuilt from CSV. Cameras/robot were not touched.")


if __name__ == "__main__":
    main()