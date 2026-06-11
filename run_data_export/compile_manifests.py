"""
Compiles all manifest.json files from data/run_snapshots into master datasets.
Outputs:
  run_summary.csv / run_summary.json      — one row per run
  object_detail.csv / object_detail.json  — one row per picked object
  master_dataset.json                     — combined hierarchical JSON
  audit_report.txt                        — failure + watchdog anomaly log
"""

import json
import csv
import os
import glob
import math
from datetime import datetime
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOTS_DIR = os.path.join(ROOT, "data", "run_snapshots")
OUT_DIR = os.path.join(ROOT, "run_data_export")

# ── Physical constants (from calibration / paper) ──────────────────────────
NOMINAL_GRASP_OFFSET_MM = 127.5   # pick_grasp_z - pick_z expected value
GRASP_OFFSET_TOLERANCE = 10.0    # flag if deviation > this
STAGING_X_BOUNDS = (35.0, 345.0) # robot-frame staging platform X range
STAGING_Y_BOUNDS = (60.0, 500.0) # robot-frame staging platform Y range
BAG_X_BOUNDS = (100.0, 360.0)    # bag region X
BAG_Y_BOUNDS = (700.0, 860.0)    # bag region Y
DEGENERATE_DIM_MM = 5.0          # flag if any AABB dimension < this
TALL_PICK_Z_MM = 90.0            # flag pick_z above this (item stacked high)
CONFIDENCE_MISS_THRESHOLD = 0.97 # flag miss when confidence was above this


def parse_timestamp(ts: str):
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def dist2d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def load_all_manifests():
    manifests = []
    paths = sorted(glob.glob(os.path.join(SNAPSHOTS_DIR, "**/manifest.json"), recursive=True))
    for path in paths:
        run_dir = os.path.dirname(path)
        folder_name = os.path.basename(run_dir)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_folder"] = folder_name
        data["_path"] = path
        manifests.append(data)
    print(f"Loaded {len(manifests)} manifests.")
    return manifests


# ── Watchdog checks ─────────────────────────────────────────────────────────

def watchdog_grasp_offset(objects, ts):
    """WD-1: pick_grasp_z_mm should equal pick_z_mm + ~127.5mm."""
    flags = []
    for o in objects:
        pz = o.get("pick_z_mm")
        gz = o.get("pick_grasp_z_mm")
        if pz is None or gz is None:
            continue
        offset = gz - pz
        if abs(offset - NOMINAL_GRASP_OFFSET_MM) > GRASP_OFFSET_TOLERANCE:
            flags.append({
                "run": ts, "watchdog": "WD-1:grasp_offset",
                "object_i": o.get("object_i"), "class": o.get("class_name"),
                "detail": f"offset={offset:.1f}mm (nominal {NOMINAL_GRASP_OFFSET_MM}mm, "
                          f"deviation={offset - NOMINAL_GRASP_OFFSET_MM:+.1f}mm)"
            })
    return flags


def watchdog_bagstate_mismatch(objects, placed_boxes, ts):
    """WD-2: placed_boxes count should equal placed objects count.
    A large gap suggests bag state tracking reset or didn't accumulate recovery picks."""
    placed_count = sum(1 for o in objects if o.get("place_result") == "placed")
    pb_count = len(placed_boxes)
    flags = []
    if placed_count != pb_count:
        flags.append({
            "run": ts, "watchdog": "WD-2:bagstate_mismatch",
            "object_i": None, "class": None,
            "detail": f"placed_count={placed_count} but placed_boxes={pb_count} "
                      f"(delta={placed_count - pb_count}). "
                      "Possible bag-state reset or recovery pass not reflected in placed_boxes."
        })
    return flags


def watchdog_class_name_mismatch(objects, ts):
    """WD-3: class_name should always equal detection_class."""
    flags = []
    for o in objects:
        cn = o.get("class_name", "")
        dc = o.get("detection_class", "")
        if cn and dc and cn != dc:
            flags.append({
                "run": ts, "watchdog": "WD-3:class_mismatch",
                "object_i": o.get("object_i"), "class": cn,
                "detail": f"class_name='{cn}' != detection_class='{dc}'"
            })
    return flags


def watchdog_pick_out_of_bounds(objects, ts):
    """WD-4: Pick XY should be within the known staging platform footprint."""
    flags = []
    for o in objects:
        xy = o.get("pick_xy_mm") or []
        if len(xy) < 2:
            continue
        x, y = xy
        if not (STAGING_X_BOUNDS[0] <= x <= STAGING_X_BOUNDS[1] and
                STAGING_Y_BOUNDS[0] <= y <= STAGING_Y_BOUNDS[1]):
            flags.append({
                "run": ts, "watchdog": "WD-4:pick_out_of_bounds",
                "object_i": o.get("object_i"), "class": o.get("class_name"),
                "detail": f"pick_xy=({x:.1f}, {y:.1f}) outside staging bounds "
                          f"X{STAGING_X_BOUNDS} Y{STAGING_Y_BOUNDS}"
            })
    return flags


def watchdog_place_out_of_bounds(objects, ts):
    """WD-5: Place XY should land inside the bag footprint."""
    flags = []
    for o in objects:
        xy = o.get("place_xy_mm") or []
        if len(xy) < 2:
            continue
        x, y = xy
        if not (BAG_X_BOUNDS[0] <= x <= BAG_X_BOUNDS[1] and
                BAG_Y_BOUNDS[0] <= y <= BAG_Y_BOUNDS[1]):
            flags.append({
                "run": ts, "watchdog": "WD-5:place_out_of_bag",
                "object_i": o.get("object_i"), "class": o.get("class_name"),
                "detail": f"place_xy=({x:.1f}, {y:.1f}) outside bag bounds "
                          f"X{BAG_X_BOUNDS} Y{BAG_Y_BOUNDS}"
            })
    return flags


def watchdog_degenerate_bbox(objects, ts):
    """WD-6: Any AABB dimension < DEGENERATE_DIM_MM suggests bad stereo reconstruction."""
    flags = []
    for o in objects:
        sz = o.get("raw_box_size_xyz_mm") or []
        if not sz:
            continue
        bad_dims = [i for i, v in enumerate(sz) if v is not None and v < DEGENERATE_DIM_MM]
        if bad_dims:
            dims_str = f"[{', '.join(f'{v:.2f}' for v in sz)}]"
            flags.append({
                "run": ts, "watchdog": "WD-6:degenerate_bbox",
                "object_i": o.get("object_i"), "class": o.get("class_name"),
                "detail": f"raw_box_size={dims_str}mm — axis {bad_dims} below {DEGENERATE_DIM_MM}mm threshold"
            })
    return flags


def watchdog_high_confidence_miss(objects, ts):
    """WD-7: A miss on a very high-confidence detection suggests a reliable detection
    followed by a physical grasp failure — worth highlighting separately."""
    flags = []
    for o in objects:
        if o.get("place_result") == "miss":
            conf = o.get("detection_confidence") or 0.0
            if conf >= CONFIDENCE_MISS_THRESHOLD:
                flags.append({
                    "run": ts, "watchdog": "WD-7:high_conf_miss",
                    "object_i": o.get("object_i"), "class": o.get("class_name"),
                    "detail": f"MISS with detection_confidence={conf:.4f} >= {CONFIDENCE_MISS_THRESHOLD} "
                              "(grasp failed despite confident detection)"
                })
    return flags


def watchdog_duplicate_pick_site(objects, ts):
    """WD-8: Two objects in the same run with the same class picked within 30mm of each other
    may indicate a double-detection of the same physical item."""
    flags = []
    entries = [(o.get("object_i"), o.get("class_name", ""), o.get("pick_xy_mm") or [])
               for o in objects]
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            oi, ci, xyi = entries[i]
            oj, cj, xyj = entries[j]
            if ci != cj or len(xyi) < 2 or len(xyj) < 2:
                continue
            d = dist2d(xyi, xyj)
            if d < 30.0:
                flags.append({
                    "run": ts, "watchdog": "WD-8:duplicate_pick_site",
                    "object_i": f"{oi}&{oj}", "class": ci,
                    "detail": f"object_{oi} and object_{oj} both '{ci}' picked "
                              f"{d:.1f}mm apart (possible double-detection)"
                })
    return flags


def watchdog_recovery_sequence(objects, ts):
    """WD-9: Detect the missed-pick recovery pattern — object_i resets back toward 1
    after reaching a higher value, indicating the robot re-surveyed and retried."""
    flags = []
    ois = [o.get("object_i") for o in objects if o.get("object_i") is not None]
    resets = []
    for k in range(1, len(ois)):
        if ois[k] < ois[k - 1] and ois[k] <= 3:
            resets.append((k, ois[k - 1], ois[k]))
    if resets:
        desc = "; ".join(f"seq[{k}]: {a}→{b}" for k, a, b in resets)
        flags.append({
            "run": ts, "watchdog": "WD-9:recovery_sequence_detected",
            "object_i": None, "class": None,
            "detail": f"object_i reset(s) detected — recovery pass triggered. Transitions: {desc}"
        })
    return flags


def run_all_watchdogs(manifests):
    all_flags = []
    for m in manifests:
        ts = m.get("run_timestamp", m["_folder"])
        objects = m.get("objects", [])
        placed_boxes = m.get("placed_boxes", [])
        all_flags += watchdog_grasp_offset(objects, ts)
        all_flags += watchdog_bagstate_mismatch(objects, placed_boxes, ts)
        all_flags += watchdog_class_name_mismatch(objects, ts)
        all_flags += watchdog_pick_out_of_bounds(objects, ts)
        all_flags += watchdog_place_out_of_bounds(objects, ts)
        all_flags += watchdog_degenerate_bbox(objects, ts)
        all_flags += watchdog_high_confidence_miss(objects, ts)
        all_flags += watchdog_duplicate_pick_site(objects, ts)
        all_flags += watchdog_recovery_sequence(objects, ts)
    return all_flags


# ── Data builders ────────────────────────────────────────────────────────────

def build_run_summary(manifests, watchdog_flags):
    wd_by_run = defaultdict(list)
    for f in watchdog_flags:
        wd_by_run[f["run"]].append(f["watchdog"].split(":")[0])

    rows = []
    for m in manifests:
        ts_str = m.get("run_timestamp", m["_folder"])
        dt = parse_timestamp(ts_str)
        objects = m.get("objects", [])
        placed_boxes = m.get("placed_boxes", [])

        total_objects = len(objects)
        placed_count = sum(1 for o in objects if o.get("place_result") == "placed")
        miss_count = sum(1 for o in objects if o.get("place_result") == "miss")
        pending_count = sum(1 for o in objects if o.get("place_result") == "pending")
        success_rate = round(placed_count / total_objects, 4) if total_objects else None

        unique_classes = sorted(set(o.get("class_name", "") for o in objects))
        confidences = [o.get("detection_confidence") for o in objects
                       if o.get("detection_confidence") is not None]
        mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else None

        place_positions = set(tuple(o["place_xy_mm"]) for o in objects
                              if o.get("place_xy_mm") is not None)

        # Detect recovery pass
        ois = [o.get("object_i") for o in objects if o.get("object_i") is not None]
        has_recovery = any(ois[k] < ois[k - 1] and ois[k] <= 3 for k in range(1, len(ois)))

        grasp_offsets = [o.get("pick_grasp_z_mm", 0) - o.get("pick_z_mm", 0)
                         for o in objects
                         if o.get("pick_grasp_z_mm") is not None and o.get("pick_z_mm") is not None]
        mean_grasp_offset = round(sum(grasp_offsets) / len(grasp_offsets), 2) if grasp_offsets else None

        run_wds = list(set(wd_by_run.get(ts_str, [])))

        row = {
            "run_timestamp": ts_str,
            "folder": m["_folder"],
            "datetime": dt.isoformat() if dt else None,
            "date": dt.date().isoformat() if dt else None,
            "time": dt.strftime("%H:%M:%S") if dt else None,
            "script": m.get("script", ""),
            "place_scene": m.get("place_planning_sequence") or m.get("place_scene_name", ""),
            "total_objects": total_objects,
            "placed_count": placed_count,
            "miss_count": miss_count,
            "pending_count": pending_count,
            "success_rate": success_rate,
            "early_termination": pending_count > 0,
            "has_recovery_pass": has_recovery,
            "placed_boxes_in_bag": len(placed_boxes),
            "bagstate_delta": placed_count - len(placed_boxes),
            "unique_classes": "; ".join(unique_classes),
            "num_unique_classes": len(unique_classes),
            "mean_detection_confidence": mean_confidence,
            "mean_grasp_offset_mm": mean_grasp_offset,
            "num_place_positions": len(place_positions),
            "watchdog_flags": "; ".join(sorted(run_wds)) if run_wds else "",
        }
        rows.append(row)
    return rows


def build_object_detail(manifests, watchdog_flags):
    wd_by_run_obj = defaultdict(list)
    for f in watchdog_flags:
        key = (f["run"], str(f["object_i"]))
        wd_by_run_obj[key].append(f["watchdog"].split(":")[0])

    rows = []
    for m in manifests:
        ts_str = m.get("run_timestamp", m["_folder"])
        dt = parse_timestamp(ts_str)
        scene = m.get("place_planning_sequence") or m.get("place_scene_name", "")
        objects = m.get("objects", [])
        placed_boxes = m.get("placed_boxes", [])
        placed_labels = {pb.get("label", ""): pb for pb in placed_boxes}

        for idx, obj in enumerate(objects):
            oi = obj.get("object_i")
            place_result = obj.get("place_result", "")
            label_key = f"object{oi}_placed" if place_result == "placed" else f"object{oi}_stacked"
            placed_box = placed_labels.get(label_key)

            pick_xy = obj.get("pick_xy_mm") or [None, None]
            place_xy = obj.get("place_xy_mm") or [None, None]
            raw_size = obj.get("raw_box_size_xyz_mm") or [None, None, None]
            bbox = obj.get("detection_bbox_xyxy") or [None, None, None, None]

            pz = obj.get("pick_z_mm")
            gz = obj.get("pick_grasp_z_mm")
            grasp_offset = round(gz - pz, 3) if pz is not None and gz is not None else None

            obj_wds = list(set(wd_by_run_obj.get((ts_str, str(oi)), [])))

            row = {
                "run_timestamp": ts_str,
                "folder": m["_folder"],
                "datetime": dt.isoformat() if dt else None,
                "date": dt.date().isoformat() if dt else None,
                "script": m.get("script", ""),
                "place_scene": scene,
                "sequence_index": idx,
                "object_i": oi,
                "class_name": obj.get("class_name", ""),
                "detection_class": obj.get("detection_class", ""),
                "detection_confidence": obj.get("detection_confidence"),
                "detection_bbox_x1": bbox[0] if len(bbox) > 0 else None,
                "detection_bbox_y1": bbox[1] if len(bbox) > 1 else None,
                "detection_bbox_x2": bbox[2] if len(bbox) > 2 else None,
                "detection_bbox_y2": bbox[3] if len(bbox) > 3 else None,
                "pick_x_mm": pick_xy[0] if len(pick_xy) > 0 else None,
                "pick_y_mm": pick_xy[1] if len(pick_xy) > 1 else None,
                "pick_z_mm": pz,
                "pick_grasp_z_mm": gz,
                "grasp_offset_mm": grasp_offset,
                "pick_phi_deg": obj.get("pick_phi_deg"),
                "place_x_mm": place_xy[0] if len(place_xy) > 0 else None,
                "place_y_mm": place_xy[1] if len(place_xy) > 1 else None,
                "place_phi_deg": obj.get("place_phi_deg"),
                "destination_surface_z_mm": obj.get("destination_surface_z_mm"),
                "place_result": place_result,
                "raw_size_x_mm": raw_size[0] if len(raw_size) > 0 else None,
                "raw_size_y_mm": raw_size[1] if len(raw_size) > 1 else None,
                "raw_size_z_mm": raw_size[2] if len(raw_size) > 2 else None,
                "placed_box_label": label_key if placed_box else None,
                "watchdog_flags": "; ".join(sorted(obj_wds)) if obj_wds else "",
            }
            rows.append(row)
    return rows


# ── Report builder ───────────────────────────────────────────────────────────

def build_audit_report(run_summary, object_detail, watchdog_flags):
    lines = []
    sep = "=" * 72

    lines += [sep,
              "AUTONOMOUS GROCERY ROBOT — RUN AUDIT REPORT",
              f"Generated: {datetime.now().isoformat()}",
              f"Total runs analyzed: {len(run_summary)}",
              sep]

    # ── Overall stats ────────────────────────────────────────────────────────
    total_objects = sum(r["total_objects"] for r in run_summary)
    total_placed = sum(r["placed_count"] for r in run_summary)
    total_miss = sum(r["miss_count"] for r in run_summary)
    total_pending = sum(r["pending_count"] for r in run_summary)
    overall_rate = round(total_placed / total_objects, 4) if total_objects else 0

    lines += ["",
              "── OVERALL STATISTICS ──────────────────────────────────────────────",
              f"  Total pick attempts:         {total_objects}",
              f"  Total placed (success):      {total_placed}  ({overall_rate*100:.1f}%)",
              f"  Total misses:                {total_miss}  ({total_miss/total_objects*100:.1f}%)",
              f"  Total pending (incomplete):  {total_pending}  ({total_pending/total_objects*100:.1f}%)"]

    # ── Runs by date ─────────────────────────────────────────────────────────
    by_date = defaultdict(list)
    for r in run_summary:
        by_date[r.get("date", "unknown")].append(r)

    lines += ["", "── RUNS BY DATE ────────────────────────────────────────────────────"]
    for date in sorted(by_date):
        dr = by_date[date]
        dp = sum(r["placed_count"] for r in dr)
        dt_ = sum(r["total_objects"] for r in dr)
        dm = sum(r["miss_count"] for r in dr)
        drate = round(dp / dt_, 4) if dt_ else 0
        lines.append(f"  {date}: {len(dr):2d} runs | {dt_:3d} picks | "
                     f"{dp:3d} placed ({drate*100:.1f}%) | {dm} misses")

    # ── Per-class performance ─────────────────────────────────────────────────
    class_placed = defaultdict(int)
    class_miss = defaultdict(int)
    class_pending = defaultdict(int)
    class_conf = defaultdict(list)
    class_sizes = defaultdict(list)

    for o in object_detail:
        cls = o.get("class_name", "unknown")
        result = o.get("place_result", "")
        if result == "placed":
            class_placed[cls] += 1
        elif result == "miss":
            class_miss[cls] += 1
        elif result == "pending":
            class_pending[cls] += 1
        if o.get("detection_confidence") is not None:
            class_conf[cls].append(o["detection_confidence"])
        sz = [o.get("raw_size_x_mm"), o.get("raw_size_y_mm"), o.get("raw_size_z_mm")]
        if all(v is not None for v in sz):
            class_sizes[cls].append(sz)

    all_classes = sorted(set(list(class_placed) + list(class_miss) + list(class_pending)))
    lines += ["", "── PER-CLASS PERFORMANCE ──────────────────────────────────────────",
              f"  {'Class':<22} {'Placed':>7} {'Miss':>6} {'Pend':>6} {'Total':>6} "
              f"{'Success%':>9} {'AvgConf%':>9}",
              "  " + "─" * 68]
    for cls in all_classes:
        p = class_placed[cls]; ms = class_miss[cls]; pd_ = class_pending[cls]
        tot = p + ms + pd_
        rate = p / tot * 100 if tot else 0
        confs = class_conf[cls]
        avg_conf = sum(confs) / len(confs) * 100 if confs else 0
        lines.append(f"  {cls:<22} {p:>7} {ms:>6} {pd_:>6} {tot:>6} "
                     f"{rate:>8.1f}% {avg_conf:>8.1f}%")

    # ── Failure detail ───────────────────────────────────────────────────────
    failed_runs = [r for r in run_summary if r["miss_count"] > 0 or r["pending_count"] > 0]
    lines += ["", f"── RUNS WITH FAILURES ({len(failed_runs)} of {len(run_summary)}) ─"
              "─────────────────────────────"]
    for r in failed_runs:
        flags = []
        if r["miss_count"] > 0: flags.append(f"{r['miss_count']} miss(es)")
        if r["pending_count"] > 0: flags.append(f"{r['pending_count']} pending")
        if r["early_termination"]: flags.append("EARLY TERMINATION")
        lines.append(f"  [{r['run_timestamp']}]  {', '.join(flags)}")
        run_objs = [o for o in object_detail if o["run_timestamp"] == r["run_timestamp"]]
        for bo in run_objs:
            if bo["place_result"] in ("miss", "pending"):
                lines.append(f"    → object_{bo['object_i']} [{bo['class_name']}]: "
                              f"{bo['place_result'].upper()}"
                              f"  conf={bo['detection_confidence']:.3f}")

    # ── Early termination ─────────────────────────────────────────────────────
    early_runs = [r for r in run_summary if r["early_termination"]]
    lines += ["", f"── EARLY TERMINATION RUNS ({len(early_runs)}) ──────────────────────────────"]
    for r in early_runs:
        lines.append(f"  [{r['run_timestamp']}] {r['total_objects']} objects, "
                     f"{r['pending_count']} pending")
    if not early_runs:
        lines.append("  None.")

    # ── Timing ───────────────────────────────────────────────────────────────
    dts = [r["datetime"] for r in run_summary if r["datetime"]]
    if len(dts) >= 2:
        dts_parsed = sorted(datetime.fromisoformat(d) for d in dts)
        gaps = [(dts_parsed[i+1] - dts_parsed[i]).total_seconds() / 60
                for i in range(len(dts_parsed) - 1)]
        intra = [g for g in gaps if g < 60]
        lines += ["", "── RUN TIMING ──────────────────────────────────────────────────────",
                  f"  Earliest: {dts_parsed[0].isoformat()}",
                  f"  Latest:   {dts_parsed[-1].isoformat()}"]
        if intra:
            lines.append(f"  Mean gap between runs (intra-session <60min): "
                         f"{sum(intra)/len(intra):.1f} min")
            lines.append(f"  Min: {min(intra):.1f} min   Max: {max(intra):.1f} min")
        lines.append("  Note: no within-run timing is stored in manifests.")

    # ── Objects per run ───────────────────────────────────────────────────────
    obj_counts = [r["total_objects"] for r in run_summary]
    lines += ["", "── OBJECTS PER RUN ─────────────────────────────────────────────────"]
    for n, cnt in sorted(Counter(obj_counts).items()):
        lines.append(f"  {n:2d} objects: {cnt} run(s)")
    lines.append(f"  Mean: {sum(obj_counts)/len(obj_counts):.2f}  "
                 f"Min: {min(obj_counts)}  Max: {max(obj_counts)}")

    # ── Place scene ───────────────────────────────────────────────────────────
    scene_counter = Counter(r["place_scene"] for r in run_summary)
    lines += ["", "── PLACE SCENE BREAKDOWN ───────────────────────────────────────────"]
    for scene, cnt in scene_counter.most_common():
        lines.append(f"  {scene}: {cnt} run(s)")

    # ── WATCHDOG REPORT ───────────────────────────────────────────────────────
    wd_labels = {
        "WD-1": "Grasp offset anomaly (pick_grasp_z - pick_z far from 127.5mm)",
        "WD-2": "Bag-state mismatch (placed_count != placed_boxes count)",
        "WD-3": "class_name / detection_class mismatch",
        "WD-4": "Pick XY outside staging platform bounds",
        "WD-5": "Place XY outside bag bounds",
        "WD-6": "Degenerate AABB (<5mm dimension — bad stereo reconstruction)",
        "WD-7": "High-confidence miss (>97% conf, still missed — grasp failure)",
        "WD-8": "Duplicate pick site (same class, <30mm apart in same run)",
        "WD-9": "Recovery sequence detected (object_i reset mid-run)",
    }
    lines += ["", sep,
              "WATCHDOG ANOMALY REPORT",
              sep]
    for wd_code, wd_desc in wd_labels.items():
        wd_flags = [f for f in watchdog_flags if f["watchdog"].startswith(wd_code + ":")]
        lines += ["", f"  {wd_code}: {wd_desc}",
                  f"  Flagged: {len(wd_flags)} instance(s)"]
        if wd_flags:
            for f in wd_flags:
                obj_str = f"  obj_{f['object_i']}" if f["object_i"] is not None else ""
                lines.append(f"    [{f['run']}]{obj_str} [{f['class'] or ''}]  {f['detail']}")

    total_wd = len(watchdog_flags)
    clean_runs = sum(1 for r in run_summary if not r["watchdog_flags"])
    lines += ["",
              f"  Total watchdog flags: {total_wd}",
              f"  Runs with no flags:   {clean_runs} of {len(run_summary)}",
              "", sep, "END OF REPORT", sep]

    return "\n".join(lines)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifests = load_all_manifests()

    print("Running watchdog checks...")
    watchdog_flags = run_all_watchdogs(manifests)
    print(f"  {len(watchdog_flags)} total watchdog flags raised.")

    run_summary = build_run_summary(manifests, watchdog_flags)
    object_detail = build_object_detail(manifests, watchdog_flags)

    run_summary.sort(key=lambda r: r["run_timestamp"])
    object_detail.sort(key=lambda o: (o["run_timestamp"], o["sequence_index"]))

    write_csv(run_summary, os.path.join(OUT_DIR, "run_summary.csv"))
    write_csv(object_detail, os.path.join(OUT_DIR, "object_detail.csv"))
    print("Wrote run_summary.csv and object_detail.csv")

    with open(os.path.join(OUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "object_detail.json"), "w", encoding="utf-8") as f:
        json.dump(object_detail, f, indent=2)

    master = {
        "generated": datetime.now().isoformat(),
        "total_runs": len(run_summary),
        "total_objects": len(object_detail),
        "total_watchdog_flags": len(watchdog_flags),
        "run_summary": run_summary,
        "object_detail": object_detail,
        "watchdog_flags": watchdog_flags,
        "raw_manifests": [
            {k: v for k, v in m.items() if not k.startswith("_")}
            for m in manifests
        ],
    }
    with open(os.path.join(OUT_DIR, "master_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)
    print("Wrote master_dataset.json")

    audit = build_audit_report(run_summary, object_detail, watchdog_flags)
    with open(os.path.join(OUT_DIR, "audit_report.txt"), "w", encoding="utf-8") as f:
        f.write(audit)
    print("Wrote audit_report.txt\n")
    print(audit.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
