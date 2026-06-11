"""
Generates paper-ready figures from the compiled run_data_export datasets.
All figures use a clean, publication-style theme (no seaborn required).
Output: paper_plots/*.pdf  (vector, LaTeX-ready) and *.png (preview)
"""

import json
import os
import math
import sys
from collections import defaultdict, Counter

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
except ImportError:
    sys.exit("matplotlib not found. Run: pip install matplotlib")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "run_data_export")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load compiled data ───────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "master_dataset.json"), encoding="utf-8") as f:
    master = json.load(f)

runs = master["run_summary"]
objs = master["object_detail"]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

PALETTE = {
    "placed":  "#2ecc71",
    "miss":    "#e74c3c",
    "pending": "#f39c12",
    "neutral": "#3498db",
}
CLASSES = ["Lays Chips", "Bandaid", "Pringles", "Dove Deodorant",
           "Carmex Lip Balm", "Burts Bees", "Chex Mix"]
CLASS_SHORT = {
    "Lays Chips":    "Lays Chips",
    "Bandaid":       "Band-Aid",
    "Pringles":      "Pringles",
    "Dove Deodorant":"Dove Deod.",
    "Carmex Lip Balm":"Carmex",
    "Burts Bees":    "Burt's Bees",
    "Chex Mix":      "Chex Mix",
}


def save(fig, name):
    png = os.path.join(OUT_DIR, f"{name}.png")
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


# ────────────────────────────────────────────────────────────────────────────
# FIG 1 — Per-class success rate (horizontal bar, sorted)
# ────────────────────────────────────────────────────────────────────────────
def fig_class_success_rate():
    placed = defaultdict(int); miss = defaultdict(int); pend = defaultdict(int)
    for o in objs:
        cls = o["class_name"]
        r = o["place_result"]
        if r == "placed": placed[cls] += 1
        elif r == "miss": miss[cls] += 1
        elif r == "pending": pend[cls] += 1

    classes = sorted(CLASSES, key=lambda c: placed[c] / (placed[c] + miss[c] + pend[c]))
    labels = [CLASS_SHORT[c] for c in classes]
    p_rates = [placed[c] / (placed[c] + miss[c] + pend[c]) * 100 for c in classes]
    m_rates = [miss[c] / (placed[c] + miss[c] + pend[c]) * 100 for c in classes]
    pd_rates = [pend[c] / (placed[c] + miss[c] + pend[c]) * 100 for c in classes]
    totals = [placed[c] + miss[c] + pend[c] for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = range(len(classes))
    ax.barh(y, p_rates, color=PALETTE["placed"], label="Placed")
    ax.barh(y, m_rates, left=p_rates, color=PALETTE["miss"], label="Miss")
    ax.barh(y, pd_rates, left=[p + m for p, m in zip(p_rates, m_rates)],
            color=PALETTE["pending"], label="Pending")

    for i, (pr, tot) in enumerate(zip(p_rates, totals)):
        ax.text(pr - 1.5, i, f"{pr:.0f}%", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")
        ax.text(101, i, f"n={tot}", va="center", fontsize=8, color="#555")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percentage of pick attempts (%)")
    ax.set_title("Pick outcome by object class")
    ax.set_xlim(0, 112)
    ax.axvline(89.0, color="gray", linestyle=":", lw=1.2, label="Overall avg (89%)")
    ax.legend(loc="lower right", framealpha=0.8)
    fig.tight_layout()
    save(fig, "fig1_class_success_rate")


# ────────────────────────────────────────────────────────────────────────────
# FIG 2 — Overall outcome donut
# ────────────────────────────────────────────────────────────────────────────
def fig_overall_outcome_donut():
    placed = sum(1 for o in objs if o["place_result"] == "placed")
    miss   = sum(1 for o in objs if o["place_result"] == "miss")
    pend   = sum(1 for o in objs if o["place_result"] == "pending")
    total  = placed + miss + pend

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    sizes = [placed, miss, pend]
    labels = [f"Placed\n{placed} ({placed/total*100:.1f}%)",
              f"Miss\n{miss} ({miss/total*100:.1f}%)",
              f"Pending\n{pend} ({pend/total*100:.1f}%)"]
    colors = [PALETTE["placed"], PALETTE["miss"], PALETTE["pending"]]
    wedges, _ = ax.pie(sizes, labels=None, colors=colors,
                       startangle=90, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    ax.legend(wedges, labels, loc="center", framealpha=0, fontsize=9)
    ax.set_title(f"Pick outcome distribution\n(n={total} attempts, {len(runs)} runs)")
    save(fig, "fig2_outcome_donut")


# ────────────────────────────────────────────────────────────────────────────
# FIG 3 — Detection confidence by class (box plot)
# ────────────────────────────────────────────────────────────────────────────
def fig_confidence_boxplot():
    conf_by_class = defaultdict(list)
    for o in objs:
        c = o.get("detection_confidence")
        if c is not None:
            conf_by_class[o["class_name"]].append(c * 100)

    classes_sorted = sorted(CLASSES, key=lambda c: -sum(conf_by_class[c]) / max(len(conf_by_class[c]), 1))
    data = [conf_by_class[c] for c in classes_sorted]
    labels = [CLASS_SHORT[c] for c in classes_sorted]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#555"),
                    capprops=dict(color="#555"),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5, color="#aaa"))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["neutral"])
        patch.set_alpha(0.75)

    ax.set_ylabel("YOLO detection confidence (%)")
    ax.set_title("Detection confidence distribution by class")
    ax.set_ylim(50, 102)
    ax.axhline(97, color=PALETTE["miss"], linestyle="--", lw=1, label="High-conf miss threshold (97%)")
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    save(fig, "fig3_confidence_boxplot")


# ────────────────────────────────────────────────────────────────────────────
# FIG 4 — Pick position scatter colored by result
# ────────────────────────────────────────────────────────────────────────────
def fig_pick_position_scatter():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for result, color, label, zorder, size in [
        ("placed",  PALETTE["placed"],  "Placed",  2, 18),
        ("miss",    PALETTE["miss"],    "Miss",    4, 45),
        ("pending", PALETTE["pending"], "Pending", 3, 35),
    ]:
        xs = [o["pick_x_mm"] for o in objs if o["place_result"] == result and o["pick_x_mm"]]
        ys = [o["pick_y_mm"] for o in objs if o["place_result"] == result and o["pick_y_mm"]]
        ax.scatter(xs, ys, c=color, s=size, alpha=0.65, label=f"{label} (n={len(xs)})",
                   zorder=zorder, edgecolors="none")

    # Staging platform outline (approximate)
    from matplotlib.patches import FancyBboxPatch
    platform = FancyBboxPatch((35, 60), 310, 440, linewidth=1.2,
                               edgecolor="#888", facecolor="none",
                               linestyle="--", boxstyle="round,pad=2")
    ax.add_patch(platform)
    ax.text(40, 505, "Staging platform", fontsize=8, color="#888")

    ax.set_xlabel("Robot X (mm)")
    ax.set_ylabel("Robot Y (mm)")
    ax.set_title("Pick target positions on staging platform\n(colored by pick outcome)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig4_pick_position_scatter")


# ────────────────────────────────────────────────────────────────────────────
# FIG 5 — Place position scatter (bag packing density)
# ────────────────────────────────────────────────────────────────────────────
def fig_place_position_scatter():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    xs = [o["place_x_mm"] for o in objs if o["place_result"] == "placed" and o["place_x_mm"]]
    ys = [o["place_y_mm"] for o in objs if o["place_result"] == "placed" and o["place_y_mm"]]
    ax.scatter(xs, ys, c=PALETTE["placed"], s=14, alpha=0.4, edgecolors="none")

    # Bag outline from paper Fig 4 (100–350 x, 675–850 y)
    from matplotlib.patches import FancyBboxPatch
    bag = FancyBboxPatch((100, 675), 250, 175, linewidth=1.5,
                          edgecolor="#e74c3c", facecolor="none",
                          linestyle="--", boxstyle="round,pad=2")
    ax.add_patch(bag)
    ax.text(102, 857, "Bag region", fontsize=8, color="#e74c3c")

    ax.set_xlabel("Robot X (mm)")
    ax.set_ylabel("Robot Y (mm)")
    ax.set_title(f"Planned placement positions in bag\n(n={len(xs)} placed items)")
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig5_place_position_scatter")


# ────────────────────────────────────────────────────────────────────────────
# FIG 6 — Objects per run histogram
# ────────────────────────────────────────────────────────────────────────────
def fig_objects_per_run():
    counts = [r["total_objects"] for r in runs]
    mean_c = sum(counts) / len(counts)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bins = range(1, max(counts) + 2)
    ax.hist(counts, bins=bins, color=PALETTE["neutral"], alpha=0.8, edgecolor="white", linewidth=0.8)
    ax.axvline(mean_c, color="#e74c3c", linestyle="--", lw=1.5, label=f"Mean = {mean_c:.1f}")
    ax.set_xlabel("Objects per run")
    ax.set_ylabel("Number of runs")
    ax.set_title(f"Distribution of run sizes (n={len(runs)} runs)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    save(fig, "fig6_objects_per_run")


# ────────────────────────────────────────────────────────────────────────────
# FIG 7 — Recovery sequence frequency per run (bar)
# ────────────────────────────────────────────────────────────────────────────
def fig_recovery_frequency():
    recovery_counts = []
    for r in runs:
        wd = r.get("watchdog_flags", "")
        count = wd.count("WD-9")
        recovery_counts.append(count)

    has_recovery = sum(1 for c in recovery_counts if c > 0)
    no_recovery = sum(1 for c in recovery_counts if c == 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left: pie showing recovery presence
    ax1.pie([has_recovery, no_recovery],
            labels=[f"Recovery triggered\n({has_recovery} runs)",
                    f"No recovery\n({no_recovery} runs)"],
            colors=[PALETTE["pending"], PALETTE["placed"]],
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
    ax1.set_title("Runs with recovery behavior")

    # Right: bar showing success rate for runs with/without recovery
    for_runs = [r for r, c in zip(runs, recovery_counts) if c > 0]
    no_runs  = [r for r, c in zip(runs, recovery_counts) if c == 0]
    rates_for = [r["success_rate"] * 100 for r in for_runs if r["success_rate"] is not None]
    rates_no  = [r["success_rate"] * 100 for r in no_runs  if r["success_rate"] is not None]

    means = [sum(rates_for) / len(rates_for), sum(rates_no) / len(rates_no)]
    stds  = [
        math.sqrt(sum((x - means[0])**2 for x in rates_for) / len(rates_for)),
        math.sqrt(sum((x - means[1])**2 for x in rates_no) / len(rates_no)),
    ]
    bars = ax2.bar(["Recovery\ntriggered", "No recovery"], means,
                   color=[PALETTE["pending"], PALETTE["placed"]],
                   yerr=stds, capsize=5, edgecolor="white")
    ax2.set_ylabel("Mean success rate (%)")
    ax2.set_title("Success rate vs. recovery presence")
    ax2.set_ylim(0, 105)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig7_recovery_frequency")


# ────────────────────────────────────────────────────────────────────────────
# FIG 8 — Failure mode breakdown (miss vs pending, by class)
# ────────────────────────────────────────────────────────────────────────────
def fig_failure_breakdown():
    miss_by_class = defaultdict(int)
    pend_by_class = defaultdict(int)
    for o in objs:
        cls = o["class_name"]
        if o["place_result"] == "miss":    miss_by_class[cls] += 1
        elif o["place_result"] == "pending": pend_by_class[cls] += 1

    classes = sorted(CLASSES, key=lambda c: -(miss_by_class[c] + pend_by_class[c]))
    labels = [CLASS_SHORT[c] for c in classes]
    misses  = [miss_by_class[c] for c in classes]
    pendings = [pend_by_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = range(len(classes))
    w = 0.38
    ax.bar([xi - w/2 for xi in x], misses,  width=w, color=PALETTE["miss"],    label="Miss (grasp failure)")
    ax.bar([xi + w/2 for xi in x], pendings, width=w, color=PALETTE["pending"], label="Pending (early termination)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Failure count by class and type")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    save(fig, "fig8_failure_breakdown")


# ────────────────────────────────────────────────────────────────────────────
# FIG 9 — Grasp offset distribution (WD-1 validation)
# ────────────────────────────────────────────────────────────────────────────
def fig_grasp_offset():
    offsets = [o["grasp_offset_mm"] for o in objs if o.get("grasp_offset_mm") is not None]
    nominal = 127.5

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(offsets, bins=30, color=PALETTE["neutral"], alpha=0.8, edgecolor="white")
    ax.axvline(nominal, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Nominal offset ({nominal} mm)")
    ax.axvline(nominal + 10, color=PALETTE["pending"], linestyle=":", lw=1.2,
               label=f"WD-1 threshold (±10 mm)")
    ax.axvline(nominal - 10, color=PALETTE["pending"], linestyle=":", lw=1.2)
    ax.set_xlabel("Grasp offset (pick_grasp_z − pick_z) [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Grasp height offset consistency\n(IK command validation)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig9_grasp_offset")


# ────────────────────────────────────────────────────────────────────────────
# FIG 10 — Session-level success rate over time (run index)
# ────────────────────────────────────────────────────────────────────────────
def fig_success_over_time():
    sorted_runs = sorted(runs, key=lambda r: r["run_timestamp"])
    rates = [r["success_rate"] * 100 for r in sorted_runs if r["success_rate"] is not None]
    idxs  = list(range(1, len(rates) + 1))

    # Rolling mean (window=5)
    window = 5
    rolling = []
    for i in range(len(rates)):
        window_data = rates[max(0, i - window + 1): i + 1]
        rolling.append(sum(window_data) / len(window_data))

    # Color by date
    dates = [r["date"] for r in sorted_runs if r["success_rate"] is not None]
    unique_dates = sorted(set(dates))
    date_colors = {"2026-05-31": PALETTE["neutral"], "2026-06-08": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for d in unique_dates:
        d_idxs = [i + 1 for i, dt in enumerate(dates) if dt == d]
        d_rates = [rates[i - 1] for i in d_idxs]
        ax.scatter(d_idxs, d_rates, color=date_colors.get(d, "gray"),
                   s=20, alpha=0.7, label=d, zorder=3)

    ax.plot(idxs, rolling, color="#e74c3c", lw=1.8, label=f"Rolling mean (w={window})", zorder=4)
    ax.axhline(89.0, color="gray", linestyle="--", lw=1, label="Overall mean (89%)")
    ax.set_xlabel("Run index (chronological)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Pick success rate over time")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    save(fig, "fig10_success_over_time")


# ────────────────────────────────────────────────────────────────────────────
# FIG 11 — Place scene comparison (scene vs success rate box plot)
# ────────────────────────────────────────────────────────────────────────────
def fig_scene_comparison():
    by_scene = defaultdict(list)
    for r in runs:
        sr = r.get("success_rate")
        if sr is not None:
            by_scene[r["place_scene"]].append(sr * 100)

    scenes = sorted(by_scene.keys())
    scene_labels = {
        "foundation_floor_future_aware": "Foundation-floor\nfuture-aware",
        "bag_local_aabb_joint": "Bag-local\nAABB-joint",
    }
    labels = [scene_labels.get(s, s) for s in scenes]
    data = [by_scene[s] for s in scenes]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color="#555"),
                    capprops=dict(color="#555"),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    colors = [PALETTE["neutral"], "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, d in enumerate(data):
        mean = sum(d) / len(d)
        ax.text(i + 1, mean + 1.5, f"μ={mean:.1f}%", ha="center", fontsize=8, color="#333")
        ax.text(i + 1, -4, f"n={len(d)}", ha="center", fontsize=8, color="#777")

    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate by place planning scene")
    ax.set_ylim(-8, 105)
    fig.tight_layout()
    save(fig, "fig11_scene_comparison")


# ────────────────────────────────────────────────────────────────────────────
# FIG 12 — AABB size distribution by class (scatter: W vs H, sized by depth)
# ────────────────────────────────────────────────────────────────────────────
def fig_aabb_size_scatter():
    class_colors = {
        "Lays Chips":     "#e67e22",
        "Bandaid":        "#1abc9c",
        "Pringles":       "#8e44ad",
        "Dove Deodorant": "#2980b9",
        "Carmex Lip Balm":"#c0392b",
        "Burts Bees":     "#27ae60",
        "Chex Mix":       "#d35400",
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    legend_handles = []

    for cls in CLASSES:
        cls_objs = [o for o in objs if o["class_name"] == cls and
                    o.get("raw_size_x_mm") and o.get("raw_size_y_mm") and o.get("raw_size_z_mm")]
        if not cls_objs:
            continue
        ws = [o["raw_size_x_mm"] for o in cls_objs]
        ds = [o["raw_size_y_mm"] for o in cls_objs]
        hs = [o["raw_size_z_mm"] for o in cls_objs]
        # Use height as marker size
        sizes = [max(10, min(h * 2, 80)) for h in hs]
        sc = ax.scatter(ws, ds, s=sizes, color=class_colors[cls], alpha=0.55,
                        edgecolors="none", label=CLASS_SHORT[cls])
        legend_handles.append(sc)

    ax.set_xlabel("AABB width (X dimension) [mm]")
    ax.set_ylabel("AABB depth (Y dimension) [mm]")
    ax.set_title("Raw AABB footprint by class\n(marker size ∝ AABB height)")
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    save(fig, "fig12_aabb_size_scatter")


# ────────────────────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating plots -> {OUT_DIR}\n")
    fig_class_success_rate()
    fig_overall_outcome_donut()
    fig_confidence_boxplot()
    fig_pick_position_scatter()
    fig_place_position_scatter()
    fig_objects_per_run()
    fig_recovery_frequency()
    fig_failure_breakdown()
    fig_grasp_offset()
    fig_success_over_time()
    fig_scene_comparison()
    fig_aabb_size_scatter()
    print(f"\nDone. {len([f for f in os.listdir(OUT_DIR) if f.endswith('.png')])} PNG files in paper_plots/")
