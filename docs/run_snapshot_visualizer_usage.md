# Run Snapshot Visualizer Usage

`scripts/capstone/run_snapshot_packing_visualizer.py` creates offline presentation figures from saved wet-run snapshot folders in `data/run_snapshots/`.

It reads `manifest.json`, saved images, saved disparity NPZ files, and optional saved point-cloud NPZ files. It does not connect to cameras, serial, Teensy, or robot hardware, and it does not rerun YOLO or RAFT.

## What It Produces

Core outputs:

- `bag_aabb_topdown.png`
- `bag_aabb_topdown.svg`
- `bag_aabb_3d.png`
- `bag_aabb_3d.svg`
- `bag_aabb_sequence.gif`
- `bag_aabb_descent.gif`
- `run_snapshot_visualizer_summary.txt`

Optional outputs:

- `frames_sequence/` when `--save-frames` is set.
- `frames_descent/` when `--save-frames` is set.
- `perception_panel.png` and `perception_panel.svg` when `--make-perception-panel` is set.
- `missed_pending_audit.txt` when manifest rows are skipped because they are `miss`, `pending`, or incomplete.

## Example: `run_20260531_184606`

```powershell
python scripts\capstone\run_snapshot_packing_visualizer.py `
  --run-dir "data\run_snapshots\run_20260531_184606" `
  --out-dir "presentation_outputs\run_20260531_184606" `
  --theme slide_dark `
  --make-topdown `
  --make-3d `
  --make-sequence-gif `
  --make-descent-gif `
  --dpi 300
```

With point-cloud overlays:

```powershell
python scripts\capstone\run_snapshot_packing_visualizer.py `
  --run-dir "data\run_snapshots\run_20260531_184606" `
  --out-dir "presentation_outputs\run_20260531_184606_pointcloud" `
  --theme slide_dark `
  --make-topdown `
  --make-3d `
  --make-sequence-gif `
  --make-descent-gif `
  --include-pointcloud `
  --dpi 300
```

## Example: `run_20260531_185839`

```powershell
python scripts\capstone\run_snapshot_packing_visualizer.py `
  --run-dir "data\run_snapshots\run_20260531_185839" `
  --out-dir "presentation_outputs\run_20260531_185839" `
  --theme slide_dark `
  --make-topdown `
  --make-3d `
  --make-sequence-gif `
  --make-descent-gif `
  --highlight-last `
  --dpi 300
```

For a report-style white background:

```powershell
python scripts\capstone\run_snapshot_packing_visualizer.py `
  --run-dir "data\run_snapshots\run_20260531_185839" `
  --out-dir "presentation_outputs\run_20260531_185839_paper" `
  --theme paper `
  --make-topdown `
  --make-3d `
  --make-sequence-gif `
  --dpi 300
```

## Perception Panel

The perception panel uses the selected row's saved `stereo_left`, `left_overlay`, `disparity`, and optional point-cloud file.

```powershell
python scripts\capstone\run_snapshot_packing_visualizer.py `
  --run-dir "data\run_snapshots\run_20260531_184606" `
  --out-dir "presentation_outputs\run_20260531_184606_panel" `
  --theme slide_dark `
  --make-perception-panel `
  --perception-row 5 `
  --include-pointcloud `
  --dpi 300
```

If `--perception-row` is omitted, the last placed row is used.

## Best Figures For The Final Presentation

Use `bag_aabb_topdown.png` for the clearest explanation of bag-local placement. It shows the bag footprint, raw AABBs, padded AABBs, placement order, and class labels.

Use `bag_aabb_sequence.gif` to explain how the bag state grows over the run.

Use `bag_aabb_descent.gif` to communicate that the chosen object is planned to descend vertically into a specific cell rather than being dropped arbitrarily.

Use `bag_aabb_3d.png` as a backup technical slide for the 3D AABB stack and bag volume.

Use `perception_panel.png` only if you want one slide connecting saved camera evidence to the reconstructed AABB.

## Caveats

Manifest `object_i` values are not guaranteed to be unique. Some runs repeat object indices or append rows after continued attempts. The visualizer therefore uses manifest row order as the displayed placement sequence.

Some manifests store `raw_box_center_xyz_mm` at the original pick-platform pose while `padded_box_center_xyz_mm` and `place_xy_mm` are in the bag pose. For bag-state presentation figures, the script keeps the raw AABB size but normalizes the raw AABB center to the placed bag pose when the raw center is outside the active bag and the padded center is inside it. The summary file lists any such normalization.

Rows with `place_result` equal to `miss` or `pending` are not included in the final bag-state visualization. They are listed in `run_snapshot_visualizer_summary.txt` and, when present, `missed_pending_audit.txt`.
