"""Trace the BLB state step-by-step and save debug PNGs of bag layout."""
import sys, os, io
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2, numpy as np
from pathlib import Path
from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from test_calibration_bundle_live_stereo_z_pickplace import load_bundle
from planning.aabb_utils import aabb_from_object_candidate, pad_aabb
from planning.bag_state import BagState
from planning.grocery_item import GroceryItem
from planning.planner_2d_blb import choose_placement_spot_2d, generate_blb_candidate_spots

import scripts.capstone.interactive_packing_demo as demo

demo.IMAGES_ALREADY_RECTIFIED = True
calib  = {k: np.asarray(v) for k, v in np.load("stereo_calibration.npz", allow_pickle=False).items()}
bundle = load_bundle(Path("robot_calibration_bundle.npz"))
device = select_torch_device(use_cuda=True, use_half=True, print_info=False)

yolo = YOLOSegmenter(weights_path="yolo_weights/full_data.pt", device_info=device,
                     imgsz=640, conf=0.35, iou=0.5, retina_masks=True, min_mask_area_px=500)
yolo.warmup()
raft = demo.RAFTStereoRunner(raft_root="RAFT-Stereo",
                             checkpoint_path="RAFT-Stereo/models/raftstereo-middlebury.pth",
                             device_info=device, valid_iters=16)
raft.warmup()

left  = cv2.imread("Training_Images/Stereo_Left_0001.jpg")
right = cv2.imread("Training_Images/Stereo_Right_0001.jpg")

objects = demo._build_objects(left, right, yolo=yolo, raft=raft,
                              rectifier=demo.StereoRectifier(calib),
                              stereo_calib=calib, bundle=bundle)
print(f"\n{len(objects)} objects detected\n")

OUT = Path("outputs/capstone/_debug")
OUT.mkdir(parents=True, exist_ok=True)

BAG_OX = demo.BAG_CENTER_XY_MM[0] - demo.BAG_WIDTH_MM / 2   # 45mm
BAG_OY = demo.BAG_CENTER_XY_MM[1] - demo.BAG_DEPTH_MM / 2   # 182.5mm
BAG_W  = demo.BAG_WIDTH_MM   # 290mm
BAG_D  = demo.BAG_DEPTH_MM   # 175mm

COLOURS = ["#2e8bcb","#e8882a","#2dc96e","#d83f4a","#8854cc","#e8c025","#4abbc9","#e06090","#ffffff"]

def save_bag_debug(state, step, note=""):
    """Top-down view + side-view of bag packing state."""
    fig, (ax_td, ax_sv) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")
    for ax in (ax_td, ax_sv): ax.set_facecolor("#0d0d1a")

    # ── Top-down (XY) ─────────────────────────────────────────────────
    ax_td.set_title(f"Step {step}: {note}\nBag top-down (X=width, Y=depth)", color="#aaddff", fontsize=9)
    ax_td.add_patch(mpatches.FancyBboxPatch((0,0), BAG_W, BAG_D,
                    boxstyle="round,pad=2", edgecolor="white", facecolor="none", lw=2))
    ax_td.set_xlim(-15, BAG_W+15); ax_td.set_ylim(-15, BAG_D+15)
    ax_td.set_xlabel("Bag X (mm)"); ax_td.set_ylabel("Bag Y (mm)")

    for i, p in enumerate(state.placed):
        w = p.raw_box.size_xyz_mm[0]; d = p.raw_box.size_xyz_mm[1]
        bx = p.target_xy[0] - BAG_OX - w/2
        by = p.target_xy[1] - BAG_OY - d/2
        col = COLOURS[(p.object_i-1) % len(COLOURS)]
        ax_td.add_patch(mpatches.Rectangle((bx, by), w, d,
                        facecolor=col, edgecolor="white", alpha=0.65, lw=1))
        ax_td.text(bx+w/2, by+d/2,
                   f"#{p.object_i} {p.info.class_name[:6]}\nz={p.target_z_mm:.0f}",
                   ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        # Show padded box dashed
        pw = p.padded_box.size_xyz_mm[0]; pd2 = p.padded_box.size_xyz_mm[1]
        pbx = p.target_xy[0] - BAG_OX - pw/2; pby = p.target_xy[1] - BAG_OY - pd2/2
        ax_td.add_patch(mpatches.Rectangle((pbx, pby), pw, pd2,
                        facecolor="none", edgecolor=col, alpha=0.4, lw=0.8, ls="--"))
    ax_td.set_aspect("equal"); ax_td.tick_params(colors="#666", labelsize=7)

    # ── Side view (XZ) ────────────────────────────────────────────────
    max_z = max((p.target_z_mm + p.raw_box.size_xyz_mm[2] for p in state.placed), default=200)
    ax_sv.set_title("Side view (X × Z height)", color="#aaddff", fontsize=9)
    ax_sv.set_xlim(-15, BAG_W+15); ax_sv.set_ylim(-20, max(max_z+40, 200))
    ax_sv.add_patch(mpatches.Rectangle((0,-5), BAG_W, 5, facecolor="#334", edgecolor="white", lw=1))
    ax_sv.set_xlabel("Bag X (mm)"); ax_sv.set_ylabel("Z height (mm)")

    for p in state.placed:
        w = p.raw_box.size_xyz_mm[0]; h = p.raw_box.size_xyz_mm[2]
        bx = p.target_xy[0] - BAG_OX - w/2
        col = COLOURS[(p.object_i-1) % len(COLOURS)]
        ax_sv.add_patch(mpatches.Rectangle((bx, p.target_z_mm), w, h,
                        facecolor=col, edgecolor="white", alpha=0.65, lw=1))
        ax_sv.text(bx+w/2, p.target_z_mm+h/2,
                   f"#{p.object_i}\n{p.info.class_name[:5]}", ha="center", va="center",
                   fontsize=5, color="white")
    ax_sv.tick_params(colors="#666", labelsize=7)

    plt.tight_layout()
    fname = OUT / f"step{step:02d}.png"
    fig.savefig(str(fname), dpi=90, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PNG] {fname.name}")


def trace_blb(state, cand_dbg):
    """Trace exactly what BLB does for this candidate."""
    bag_w_cm = BAG_W / 10; bag_d_cm = BAG_D / 10

    def bl(xy_mm, w_cm, d_cm):
        x = (xy_mm[0] - BAG_OX) / 10.0 - w_cm / 2
        y = (xy_mm[1] - BAG_OY) / 10.0 - d_cm / 2
        return x, y   # RAW (no clamp)

    def bl_clamped(xy_mm, w_cm, d_cm):
        x, y = bl(xy_mm, w_cm, d_cm)
        return max(0.0, x), max(0.0, y)

    class_name = getattr(cand_dbg.candidate.yolo, "class_name", "Unknown")
    raw_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=class_name)
    iw = max(1.0, min(raw_box.size_xyz_mm[0]/10, bag_w_cm*0.99))
    id_ = max(1.0, min(raw_box.size_xyz_mm[1]/10, bag_d_cm*0.99))
    ih  = max(0.5, raw_box.size_xyz_mm[2]/10)

    def gi(w, d, h):
        return GroceryItem(id=class_name, class_name=class_name, confidence=0.9,
                           grasp_xyz_robot_mm=None, grasp_xyz_overhead=None,
                           grasp_xyz_stereo_cam_mm=None, cross_sectional_area_cm2=w*d,
                           height_cm=h, weight_score=0.3, fragility_score=0.3,
                           padded_rect_xy_cm=(w,d), padded_box_xyz_cm=(w,d,h))

    new_item = gi(iw, id_, ih)

    # Build base-layer bag state (UNCLAMPED)
    bag_unc = BagState(bag_w_cm, bag_d_cm)
    bag_cl  = BagState(bag_w_cm, bag_d_cm)
    base = [p for p in state.placed if p.target_z_mm < 1.0]

    print(f"\n  BLB trace for {class_name} ({iw:.2f}×{id_:.2f}cm):")
    print(f"  Base-layer items: {len(base)}")
    for p in base:
        pw = max(1.0, p.raw_box.size_xyz_mm[0]/10); pd = max(1.0, p.raw_box.size_xyz_mm[1]/10)
        ph = max(0.5, p.raw_box.size_xyz_mm[2]/10)
        ux, uy = bl(p.target_xy, pw, pd)
        cx, cy = bl_clamped(p.target_xy, pw, pd)
        print(f"    {p.info.class_name}: unclamped bl=({ux:.4f},{uy:.4f})  clamped=({cx:.4f},{cy:.4f})")
        bag_unc.add(gi(pw,pd,ph), ux, uy)
        bag_cl.add(gi(pw,pd,ph), cx, cy)

    spots_unc = generate_blb_candidate_spots(bag_unc)
    spots_cl  = generate_blb_candidate_spots(bag_cl)
    print(f"  BLB spots (unclamped): {[(round(s.x,3),round(s.y,3)) for s in spots_unc]}")
    print(f"  BLB spots (clamped):   {[(round(s.x,3),round(s.y,3)) for s in spots_cl]}")

    # Try each spot
    for label, spots, bag in [("UNCLAMPED", spots_unc, bag_unc), ("CLAMPED", spots_cl, bag_cl)]:
        result = None
        for s in spots:
            fits = bag.can_fit_2d(new_item, s.x, s.y)
            if fits:
                result = s
                break
        print(f"  {label} chosen spot: {(round(result.x,3),round(result.y,3)) if result else None}")


# ── Run simulation with full trace ─────────────────────────────────────────
state = demo.DemoState(objects=objects)
state.next_target = demo._select_next(state)

for step_i in range(12):
    nxt = state.next_target
    if nxt is None:
        print(f"\nDone at step {step_i}: {len(state.placed)}/9 placed")
        break

    info = next(o for o in objects if o.cand_dbg is nxt)
    cname = info.class_name
    print(f"\n--- Step {step_i+1}: Placing {cname} ---")

    # Trace BLB before placement
    with io.StringIO() as sink, __import__("contextlib").redirect_stdout(sink):
        trace_blb(state, nxt)
    trace_blb(state, nxt)  # prints directly

    # Actually place it
    with io.StringIO() as sink, __import__("contextlib").redirect_stdout(sink):
        xy, z = demo._compute_place_target(state, nxt)
        raw = aabb_from_object_candidate(nxt.candidate, default_label=cname)
        pad = pad_aabb(raw, demo.PAD_X_MM, demo.PAD_Y_MM, demo.PAD_Z_MM)

    layer = "BASE" if z < 1 else f"STACK z={z:.0f}mm"
    print(f"  → Placed at robot_xy=({xy[0]:.1f},{xy[1]:.1f})  {layer}")
    print(f"     bag_xy=({(xy[0]-BAG_OX):.1f},{(xy[1]-BAG_OY):.1f})  size=({raw.size_xyz_mm[0]:.0f}×{raw.size_xyz_mm[1]:.0f}mm)")

    placed = demo.PlacedObject(info=info, target_xy=xy, raw_box=raw, padded_box=pad.padded_box,
                               object_i=step_i+1, target_z_mm=float(z))
    state.placed.append(placed)
    save_bag_debug(state, step_i+1, f"{cname} ({layer})")
    state.next_target = demo._select_next(state)

print(f"\nTotal: {len(state.placed)}/9 placed")
print(f"Debug PNGs saved to: {OUT}")
