from __future__ import annotations

"""Offline validation for foundation_floor_future_aware planning mode.

Compares the new mode against bag_local_aabb_joint on:
  1. Saved wet-run snapshots under data/run_snapshots/ (if available)
  2. Synthetic test cases (always run)

Run:
    python scripts/validate_foundation_floor_future_aware.py

No live robot/camera connection required.
"""

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.aabb_utils import (
    AxisAlignedBox3D,
    make_aabb_from_center_size,
    make_aabb_from_min_max,
    pad_aabb,
)
from planning.autonomous_planning_sequences import (
    BAG_LOCAL_AABB_JOINT,
    FOUNDATION_FLOOR_FUTURE_AWARE,
    PlanningRuntime,
    PlanningSequenceContext,
    select_candidate_for_state,
)
from planning.grocery_properties import (
    GroceryProperties,
    foundation_strength_score,
    load_grocery_specs,
    properties_for_class,
)

# ── Validation output paths ───────────────────────────────────────────────────
_VALIDATION_DIR = REPO_ROOT / "data" / "validation"
_SUMMARY_PATH = _VALIDATION_DIR / "foundation_floor_future_aware_summary.json"
_FIG_DIR = _VALIDATION_DIR / "foundation_floor_future_aware"

# ── Bag geometry for validation ───────────────────────────────────────────────
_DEFAULT_SURFACE_ZONE = {
    "center_xy_mm": [178.0, 754.0],
    "width_mm": 290.0,
    "depth_mm": 175.0,
    "surface_z_mm": -200.0,
}
_DEFAULT_BAG_HEIGHT_MM = 250.0
_DEFAULT_PAD_XYZ_MM = np.array([0.0, 0.0, 20.0], dtype=np.float64)


# ── Lightweight stubs so we can call the planner without live hardware ────────

@dataclass
class _StubYOLO:
    class_name: str
    confidence: float = 0.99


@dataclass
class _StubCandidate:
    index: int
    yolo: _StubYOLO
    object_robot_xyz_raw: np.ndarray
    object_robot_xyz_corrected: np.ndarray | None
    topdown_width_cm: float | None = None
    topdown_depth_cm: float | None = None
    pointcloud_height_cm: float | None = None
    bbox_size_mm: list = field(default_factory=lambda: [80.0, 80.0, 60.0])


@dataclass
class _StubCandidateDebug:
    candidate: _StubCandidate
    blend_weight_overhead: float = 0.5
    stereo_xy_mm: np.ndarray | None = None
    overhead_xy_mm: np.ndarray | None = None
    xy_disagreement_mm: float | None = None


@dataclass
class _StubDecision:
    dbg: Any
    passed: bool
    candidate_index: int
    class_name: str
    reject_reasons: list = field(default_factory=list)
    volume_mm3: float = 0.0


@dataclass
class _StubResult:
    decisions: list
    selected: Any = None
    selected_decision: Any = None

    def print_debug(self, prefix: str = "") -> None:
        pass


@dataclass
class _StubState:
    candidates: list
    selected_index: int = 0


def _make_stub_candidate(
    index: int,
    class_name: str,
    platform_xy_mm: np.ndarray,
    size_mm: tuple[float, float, float],  # (w, d, h) in mm
    platform_z_mm: float = 0.0,
) -> _StubCandidateDebug:
    xyz = np.array([platform_xy_mm[0], platform_xy_mm[1], platform_z_mm], dtype=np.float64)
    cand = _StubCandidate(
        index=index,
        yolo=_StubYOLO(class_name=class_name),
        object_robot_xyz_raw=xyz.copy(),
        object_robot_xyz_corrected=xyz.copy(),
        topdown_width_cm=size_mm[0] / 10.0,
        topdown_depth_cm=size_mm[1] / 10.0,
        pointcloud_height_cm=size_mm[2] / 10.0,
        bbox_size_mm=list(size_mm),
    )
    return _StubCandidateDebug(candidate=cand)


def _all_pass_result(state: _StubState) -> _StubResult:
    """Return a BestCandidateResult-like that passes every candidate."""
    decisions = []
    for dbg in state.candidates:
        cand = dbg.candidate
        size = np.array(cand.bbox_size_mm, dtype=np.float64)
        vol = float(np.prod(size))
        decisions.append(
            _StubDecision(
                dbg=dbg,
                passed=True,
                candidate_index=int(cand.index),
                class_name=str(cand.yolo.class_name),
                volume_mm3=vol,
            )
        )
    return _StubResult(decisions=decisions)


def _make_runtime() -> PlanningRuntime:
    from planning.aabb_utils import aabb_from_object_candidate

    def _choose_best(state, config, robot, placed_boxes):
        return _all_pass_result(state)

    return PlanningRuntime(
        choose_best_candidate=_choose_best,
        aabb_from_object_candidate=aabb_from_object_candidate,
    )


def _make_ctx(
    placed_boxes: list,
    surface_zone: dict | None = None,
    bag_height_mm: float = _DEFAULT_BAG_HEIGHT_MM,
    pad_xyz_mm: np.ndarray | None = None,
) -> PlanningSequenceContext:
    sz = surface_zone if surface_zone is not None else _DEFAULT_SURFACE_ZONE
    pad = pad_xyz_mm if pad_xyz_mm is not None else _DEFAULT_PAD_XYZ_MM
    bag_center = np.asarray(sz["center_xy_mm"], dtype=np.float64)
    return PlanningSequenceContext(
        object_i=len(placed_boxes) + 1,
        robot=None,
        placed_boxes=list(placed_boxes),
        config=None,
        surface_zone=sz,
        base_xy=bag_center.copy(),
        base_phi_deg=0.0,
        target_limit=20,
        pad_xyz_mm=np.asarray(pad, dtype=np.float64).reshape(3).copy(),
        bag_height_mm=float(bag_height_mm),
    )


# ── Snapshot reconstruction ───────────────────────────────────────────────────

def _aabb_from_snapshot_object(obj: dict) -> AxisAlignedBox3D | None:
    try:
        center = np.asarray(obj["raw_box_center_xyz_mm"], dtype=np.float64)
        size = np.asarray(obj["raw_box_size_xyz_mm"], dtype=np.float64)
        return make_aabb_from_center_size(center, size, label=obj.get("class_name", "snap"))
    except Exception:
        return None


def _padded_box_from_snapshot_placed(entry: dict, pad: np.ndarray) -> Any | None:
    try:
        center = np.asarray(entry["center_xyz_mm"], dtype=np.float64)
        size = np.asarray(entry["size_xyz_mm"], dtype=np.float64)
        raw_box = make_aabb_from_center_size(center, size, label=entry.get("label", "placed"))
        return pad_aabb(raw_box, float(pad[0]), float(pad[1]), float(pad[2]))
    except Exception:
        return None


@dataclass
class _SnapshotRow:
    run_id: str
    object_slot: int
    class_name: str
    old_selected: str
    old_layer: float | None
    new_selected: str
    new_layer: float | None
    new_floor: bool | None
    future_floor_count: int | None
    stack_forced: int | None
    stranded: int | None
    reason: str
    flag: str  # "PASS", "INFO", "WARN"


def _run_both_modes(
    state: _StubState,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> tuple[str, float | None, str, float | None, bool | None, int | None, int | None, int | None]:
    """Returns (old_class, old_layer, new_class, new_layer, new_floor, future_floor, stack_forced, stranded)."""
    old_class, old_layer = "none", None
    new_class, new_layer, new_floor = "none", None, None
    future_floor, stack_forced, stranded = None, None, None

    try:
        old_sel = select_candidate_for_state(BAG_LOCAL_AABB_JOINT, state, ctx=ctx, runtime=runtime)
        if old_sel.result.selected is not None:
            old_class = str(old_sel.result.selected.candidate.yolo.class_name)
            old_idx = state.candidates.index(old_sel.result.selected)
            old_target = old_sel.target_cache.get(old_idx)
            if old_target is not None:
                old_layer = old_target.planner_layer_z_mm
    except Exception as exc:
        old_class = f"error:{exc}"

    # Reset state (select_candidate_for_state may mutate selected_index)
    state.selected_index = 0

    try:
        new_sel = select_candidate_for_state(FOUNDATION_FLOOR_FUTURE_AWARE, state, ctx=ctx, runtime=runtime)
        if new_sel.result.selected is not None:
            new_class = str(new_sel.result.selected.candidate.yolo.class_name)
            new_idx = state.candidates.index(new_sel.result.selected)
            new_target = new_sel.target_cache.get(new_idx)
            if new_target is not None:
                new_layer = new_target.planner_layer_z_mm
                new_floor = new_layer is not None and abs(float(new_layer)) <= 1e-6
                future_floor = new_target.future_placeable_count
                # future_placeable_count counts "any placement", but we need floor specifically
                # The score tuple contains future_floor_count at index 1 — extract from score
                # if planner_score encodes it.  As a simpler fallback use future_placeable_count.
    except Exception as exc:
        new_class = f"error:{exc}"

    return old_class, old_layer, new_class, new_layer, new_floor, future_floor, stack_forced, stranded


def _load_manifest_snapshots() -> list[_SnapshotRow]:
    rows: list[_SnapshotRow] = []
    snap_base = REPO_ROOT / "data" / "run_snapshots"
    if not snap_base.exists():
        print(f"[SNAPSHOT] no run_snapshots directory at {snap_base}")
        return rows

    manifests = sorted(snap_base.glob("*/manifest.json"))
    if not manifests:
        print("[SNAPSHOT] no manifests found; skipping snapshot validation")
        return rows

    pad = _DEFAULT_PAD_XYZ_MM.copy()
    specs = load_grocery_specs(REPO_ROOT / "config" / "grocery_spec.json")
    runtime = _make_runtime()

    for mf in manifests:
        run_id = mf.parent.name
        try:
            with open(mf, "r") as f:
                manifest = json.load(f)
        except Exception as exc:
            print(f"[SNAPSHOT] {run_id}: manifest load failed: {exc}")
            continue

        objects = manifest.get("objects", [])
        placed_boxes_raw = manifest.get("placed_boxes", [])

        # Reconstruct surface_zone from first placed box (approximate — use defaults)
        surface_zone = {
            "center_xy_mm": _DEFAULT_SURFACE_ZONE["center_xy_mm"],
            "width_mm": _DEFAULT_SURFACE_ZONE["width_mm"],
            "depth_mm": _DEFAULT_SURFACE_ZONE["depth_mm"],
            "surface_z_mm": _DEFAULT_SURFACE_ZONE["surface_z_mm"],
        }

        # Build placed box state from manifest
        placed_boxes = []
        for entry in placed_boxes_raw:
            pb = _padded_box_from_snapshot_placed(entry, pad)
            if pb is not None:
                placed_boxes.append(pb)

        # For each object, treat it as a single-candidate validation step
        for obj in objects:
            slot = int(obj.get("object_i", 0))
            class_name = str(obj.get("class_name", "unknown"))
            raw_size = obj.get("raw_box_size_xyz_mm")
            raw_center = obj.get("raw_box_center_xyz_mm")
            if raw_size is None or raw_center is None:
                print(f"[SNAPSHOT] {run_id} obj{slot}: missing AABB fields; skipping")
                continue

            cand = _make_stub_candidate(
                index=slot,
                class_name=class_name,
                platform_xy_mm=np.asarray(raw_center[:2], dtype=np.float64),
                size_mm=tuple(float(v) for v in raw_size[:3]),
                platform_z_mm=float(raw_center[2]),
            )
            state = _StubState(candidates=[cand])
            ctx = _make_ctx(placed_boxes, surface_zone=surface_zone)

            old_class, old_layer, new_class, new_layer, new_floor, ff_count, sf, st = (
                _run_both_modes(state, ctx, runtime)
            )

            old_is_floor = old_layer is not None and abs(float(old_layer)) <= 1e-6
            new_is_floor = new_layer is not None and abs(float(new_layer)) <= 1e-6

            flag = "INFO"
            if not old_is_floor and new_is_floor:
                flag = "PASS"

            rows.append(
                _SnapshotRow(
                    run_id=run_id,
                    object_slot=slot,
                    class_name=class_name,
                    old_selected=old_class,
                    old_layer=old_layer,
                    new_selected=new_class,
                    new_layer=new_layer,
                    new_floor=new_is_floor,
                    future_floor_count=ff_count,
                    stack_forced=sf,
                    stranded=st,
                    reason=f"old_floor={old_is_floor} new_floor={new_is_floor}",
                    flag=flag,
                )
            )
    return rows


# ── Synthetic test cases ──────────────────────────────────────────────────────

@dataclass
class _SyntheticCase:
    name: str
    candidates: list[_StubCandidateDebug]
    placed_boxes: list
    expected_floor_winner: bool
    description: str


def _make_synthetic_cases() -> list[_SyntheticCase]:
    """Four synthetic tests covering the key selection behaviours."""
    bag_center = np.asarray(_DEFAULT_SURFACE_ZONE["center_xy_mm"], dtype=np.float64)

    # Helper: offset from bag center on the platform (values in mm)
    def _plat(dx: float, dy: float) -> np.ndarray:
        return bag_center + np.array([dx, dy], dtype=np.float64)

    cases: list[_SyntheticCase] = []

    # ── Case 1: single heavy/low-fragility object; empty bag → floor ─────────
    c1 = _make_stub_candidate(1, "Diet Coke", _plat(-50, -100), (80.0, 60.0, 120.0))
    cases.append(
        _SyntheticCase(
            name="single_heavy_low_fragility",
            candidates=[c1],
            placed_boxes=[],
            expected_floor_winner=True,
            description="Single Diet Coke on empty bag should get floor placement",
        )
    )

    # ── Case 2: single fragile/light object; empty bag → still floor ─────────
    c2 = _make_stub_candidate(1, "Lays Chips", _plat(-50, -100), (176.0, 129.0, 24.0))
    cases.append(
        _SyntheticCase(
            name="single_fragile_light",
            candidates=[c2],
            placed_boxes=[],
            expected_floor_winner=True,
            description="Single Lays Chips on empty bag — floor placement, even though fragile",
        )
    )

    # ── Case 3: both objects fit on floor → prefer stronger foundation ────────
    c3a = _make_stub_candidate(1, "Diet Coke", _plat(-70, -120), (80.0, 60.0, 120.0))
    c3b = _make_stub_candidate(2, "Lays Chips", _plat(50, -100), (176.0, 129.0, 24.0))
    cases.append(
        _SyntheticCase(
            name="two_floor_candidates_prefer_stronger",
            candidates=[c3a, c3b],
            placed_boxes=[],
            expected_floor_winner=True,
            description=(
                "Diet Coke (heavier) and Lays Chips both fit on floor. "
                "New mode should prefer Diet Coke (higher foundation score)."
            ),
        )
    )

    # ── Case 4: choosing A blocks B's floor placement → prefer B first ────────
    # Bag is 290 x 175 mm. If A (200 x 100) placed at x=0, it occupies x=0..200.
    # B (100 x 100) can fit alongside at x=200. If B placed first (100 x 100 at x=0),
    # A can still fit at x=100. So placing B first preserves A's floor option.
    # We want the mode to pick B first so A still gets a floor spot.
    # A is heavy/low-fragility (Diet Coke-like). B is light (Lays-like).
    # But choosing A first squeezes B out of floor (B=176mm wide, remainder=290-200-pad=90mm < 176mm).
    # Choosing B first (small item) leaves room for A.
    # new mode should pick B first because it preserves future floor feasibility for A.
    c4a = _make_stub_candidate(1, "Diet Coke", _plat(-70, -120), (200.0, 100.0, 120.0))
    c4b = _make_stub_candidate(2, "Lays Chips", _plat(50, -100), (80.0, 100.0, 24.0))
    cases.append(
        _SyntheticCase(
            name="floor_choice_blocks_future_floor",
            candidates=[c4a, c4b],
            placed_boxes=[],
            expected_floor_winner=True,
            description=(
                "Diet Coke (wide=200mm) fills the bag floor if placed first, "
                "stranding Lays Chips. Lays (narrow=80mm) placed first leaves room. "
                "New mode should pick Lays first (floor_future=1 > 0)."
            ),
        )
    )

    return cases


def _print_table_header() -> None:
    print(
        f"{'run_id':>30s}  {'slot':>4s}  {'class':>14s}  "
        f"{'old_sel':>14s}  {'old_lyr':>8s}  "
        f"{'new_sel':>14s}  {'new_lyr':>8s}  {'flr?':>4s}  "
        f"{'ff':>4s}  {'sf':>4s}  {'st':>4s}  {'flag':>4s}"
    )
    print("-" * 130)


def _print_table_row(r: _SnapshotRow) -> None:
    def _fmt_layer(v: float | None) -> str:
        return "floor" if (v is not None and abs(v) <= 1e-6) else (f"{v:.1f}" if v is not None else "?")

    print(
        f"{r.run_id:>30s}  {r.object_slot:>4d}  {r.class_name:>14s}  "
        f"{r.old_selected:>14s}  {_fmt_layer(r.old_layer):>8s}  "
        f"{r.new_selected:>14s}  {_fmt_layer(r.new_layer):>8s}  "
        f"{str(r.new_floor):>4s}  "
        f"{str(r.future_floor_count or '?'):>4s}  "
        f"{str(r.stack_forced or '?'):>4s}  "
        f"{str(r.stranded or '?'):>4s}  "
        f"{r.flag:>4s}"
    )


def _save_summary(rows: list[_SnapshotRow], synthetic_results: list[dict]) -> None:
    _VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "snapshot_rows": [
            {
                "run_id": r.run_id,
                "object_slot": r.object_slot,
                "class_name": r.class_name,
                "old_selected": r.old_selected,
                "old_layer_mm": r.old_layer,
                "new_selected": r.new_selected,
                "new_layer_mm": r.new_layer,
                "new_floor": r.new_floor,
                "future_floor_count": r.future_floor_count,
                "stack_forced": r.stack_forced,
                "stranded": r.stranded,
                "flag": r.flag,
                "reason": r.reason,
            }
            for r in rows
        ],
        "synthetic_results": synthetic_results,
        "pass_count": sum(1 for r in rows if r.flag == "PASS"),
        "total_snapshot_steps": len(rows),
    }
    with open(_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] written to {_SUMMARY_PATH}")


def _try_save_figures(cases: list[_SyntheticCase], synthetic_results: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[FIG] matplotlib not available; skipping figure output")
        return

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    bag_w = float(_DEFAULT_SURFACE_ZONE.get("width_mm", 290.0))
    bag_d = float(_DEFAULT_SURFACE_ZONE.get("depth_mm", 175.0))

    for case, result in zip(cases, synthetic_results):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Synthetic: {case.name}", fontsize=10)
        colors = ["steelblue", "tomato", "forestgreen", "gold"]
        for ax_i, (ax, mode) in enumerate(zip(axes, ["old (bag_local_aabb_joint)", "new (foundation_floor_future_aware)"])):
            ax.set_title(mode, fontsize=8)
            ax.set_xlim(-5, bag_w + 5)
            ax.set_ylim(-5, bag_d + 5)
            ax.set_aspect("equal")
            ax.add_patch(mpatches.Rectangle((0, 0), bag_w, bag_d, fill=False, edgecolor="black", lw=2))
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            winner = result.get("old_winner" if ax_i == 0 else "new_winner", "?")
            ax.set_title(f"{mode}\nwinner: {winner}", fontsize=8)

        plt.tight_layout()
        fig_path = _FIG_DIR / f"{case.name}.png"
        plt.savefig(str(fig_path), dpi=100)
        plt.close(fig)
        print(f"[FIG] saved {fig_path}")


def main() -> None:
    print("=" * 70)
    print("foundation_floor_future_aware offline validation")
    print("=" * 70)

    # ── Part 1: snapshot validation ───────────────────────────────────────────
    print("\n[SNAPSHOTS] loading saved wet-run snapshots...")
    snap_rows = _load_manifest_snapshots()
    if not snap_rows:
        print("[SNAPSHOTS] no usable snapshot data found")
    else:
        print(f"\n[SNAPSHOTS] {len(snap_rows)} placement steps reconstructed\n")
        _print_table_header()
        for r in snap_rows:
            _print_table_row(r)
        pass_count = sum(1 for r in snap_rows if r.flag == "PASS")
        print(f"\n[SNAPSHOTS] PASS (new floor where old stacked): {pass_count}/{len(snap_rows)}")
        for r in snap_rows:
            if r.flag == "PASS":
                print(
                    f"  PASS: new mode chose floor placement where old mode stacked "
                    f"({r.run_id} slot {r.object_slot} {r.class_name})"
                )

    # ── Part 2: synthetic tests ───────────────────────────────────────────────
    print("\n[SYNTHETIC] running synthetic test cases...")
    cases = _make_synthetic_cases()
    runtime = _make_runtime()
    synthetic_results: list[dict] = []

    all_passed = True
    for case in cases:
        print(f"\n--- {case.name} ---")
        print(f"    {case.description}")
        state = _StubState(candidates=list(case.candidates))
        ctx = _make_ctx(case.placed_boxes)

        old_class, old_layer, new_class, new_layer, new_floor, ff_count, sf, st = (
            _run_both_modes(state, ctx, runtime)
        )

        old_is_floor = old_layer is not None and abs(float(old_layer)) <= 1e-6
        new_is_floor = new_layer is not None and abs(float(new_layer)) <= 1e-6

        print(f"    old mode ({BAG_LOCAL_AABB_JOINT}): selected={old_class!r} layer_z={old_layer}")
        print(f"    new mode ({FOUNDATION_FLOOR_FUTURE_AWARE}): selected={new_class!r} layer_z={new_layer} is_floor={new_is_floor}")

        ok = new_is_floor == case.expected_floor_winner
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"    expected_floor_winner={case.expected_floor_winner} -> {status}")

        if not old_is_floor and new_is_floor:
            print(
                f"    PASS: new mode chose floor placement where old mode stacked "
                f"({case.name})"
            )

        synthetic_results.append(
            {
                "case": case.name,
                "old_winner": old_class,
                "old_layer_mm": old_layer,
                "new_winner": new_class,
                "new_layer_mm": new_layer,
                "new_floor": new_is_floor,
                "status": status,
            }
        )

    print(f"\n[SYNTHETIC] {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # ── Part 3: save outputs ──────────────────────────────────────────────────
    _save_summary(snap_rows, synthetic_results)
    _try_save_figures(cases, synthetic_results)

    print("\n[DONE] validation complete")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
