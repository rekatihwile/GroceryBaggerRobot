# Capstone Publication Visualizers

The capstone scripts write into one publication source tree:

```text
paper_figure_sources/
  runs/<run_name>/
    perception_steps/
    planner_steps/
    packing/
    storyboards/
    animations/
    diagnostics/
    manifest.json
    placement_reconciliation.csv
    run_audit.json
    run_source_index.csv
  global/calibration/
  paper_figures/
```

## Build The Archive

Export every June 8 run, source image variants, disparity data, photo-colored
point clouds, packing views, reconciliation audits, and the six curated paper
figure folders:

```powershell
.venv\Scripts\python.exe scripts\capstone\build_paper_figure_sources.py
```

The default static export resolution is 600 DPI. Plots are written as PNG and
PDF. Camera images are retained as native PNG plus high-quality JPG and PDF.
Plot source data is written as CSV where applicable.

## Execution Ground Truth

The run `manifest.json` is authoritative for what the robot executed. Offline
planner replay is diagnostic only. `visualize_pickplace_planner.py` compares the
current replay with the executed target and records one of:

- `planner_replay_matches_manifest`
- `manifest_fallback_no_replay_target`
- `manifest_override_replay_mismatch`

The executed placement remains in the visuals even when the current planner
rejects it or chooses a different target.

## Point-Cloud Color

Saved point-cloud NPZ files often contain XYZ without RGB or UV. Every capstone
point-cloud renderer recovers color from the original rectified-left photograph:

1. Use stored UV coordinates when present.
2. Otherwise project saved camera-frame XYZ through `stereo_calibration.npz`.
3. Sample RGB from the matching `Stereo_Left_XXXX.png`.

The master exporter also writes:

- `08_point_cloud_colorized.npz` with camera XYZ, robot XYZ, UV, and RGB.
- `07_point_cloud_coordinates_sample.csv` with XYZ, UV, and RGB columns.
- `point_cloud_color_audit.json` describing the color source.

An uncolorable point cloud is omitted rather than shown with synthetic class
colors.

## Main Scripts

- `build_paper_figure_sources.py`: batch packaging and paper-figure curation.
- `run_snapshot_manifest_survey_story.py`: five clean perception panels per row.
- `visualize_pickplace_planner.py`: manifest-reconciled planner diagnostics.
- `run_snapshot_packing_visualizer.py`: executed packing views and animations.
- `run_snapshot_storyboard.py`: perception and planning storyboards.
- `make_run_snapshot_figures.py`: compact top-down and 3D packing exports.
- `run_snapshot_placement_gif.py`: pick-to-place animation.
- `calibration_grid_visualizer.py`: camera-space and robot-space calibration.
- `pipeline_steps_viewer.py`, `aabb_placement_viewer.py`,
  `autonomous_system_wrapper.py`, and `interactive_packing_demo.py`: saved-image
  or interactive diagnostics under the same output tree.

Static exporters are clean by default. Use `--annotated` where available to keep
titles, labels, and explanatory text.
