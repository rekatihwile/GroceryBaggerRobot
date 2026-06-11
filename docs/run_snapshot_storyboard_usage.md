# Run Snapshot Storyboard Usage

`scripts/capstone/run_snapshot_storyboard.py` builds presentation-ready storyboard figures directly from saved wet-run snapshot folders under `data/run_snapshots/...`.

It is offline-only:

- no live cameras
- no serial or Teensy
- no robot motion
- no YOLO rerun
- no RAFT rerun

The script reads `manifest.json` plus any saved images, disparity maps, point clouds, and optional calibration bundle files that already exist on disk.

## Modes

### 1. Perception mode

Build a multi-panel perception storyboard for one selected manifest row.

Outputs:

- `perception_storyboard.png`
- `perception_storyboard.svg`

Example:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode perception `
  --row 5 `
  --out-dir presentation_outputs/run_20260531_184606/perception_row5 `
  --include-pointcloud `
  --dpi 300
```

You can also select by saved `object_i`:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_185839 `
  --mode perception `
  --object-index 7 `
  --out-dir presentation_outputs/run_20260531_185839/perception_object7
```

### 2. Planning mode

Build a two-panel planning storyboard:

- left: saved survey-context image with the chosen next object called out
- right: bag-local 3D packing state with raw AABBs, padded AABBs, and optional shifted point clouds

Outputs:

- `planning_storyboard.png`
- `planning_storyboard.svg`

Example:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode planning `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/planning_row5 `
  --include-pointcloud `
  --dpi 300
```

`--placement-step` is accepted as an alias for `--up-to-row`.

### 3. Tune-view mode

Open an interactive 3D planning view so you can rotate the camera and save a reusable view JSON for later exports.

Example:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode tune-view `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/tune_view `
  --include-pointcloud `
  --view-name capstone_row5_paper
```

Keyboard controls inside the interactive window:

- `v` save current view config JSON
- `p` toggle point clouds
- `a` toggle raw AABBs
- `d` toggle padded AABBs
- `l` toggle labels
- `n` swap theme
- `+` / `-` adjust point size
- `q` or `Esc` close

Saved configs go to `presentation_outputs/view_configs/` unless you pass `--view-config` directly.

## Reusing a saved view config

After saving a view JSON in `tune-view`, reuse it for a planning export:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode planning `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/planning_row5_saved_view `
  --include-pointcloud `
  --view-config presentation_outputs/view_configs/capstone_row5_paper.json `
  --dpi 300
```

## Optional outputs

### Planning GIF

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode planning `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/planning_row5_gif `
  --make-gif `
  --include-pointcloud
```

This writes `planning_storyboard.gif`.

### Descent GIF

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode planning `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/planning_row5_descent `
  --make-descent-gif
```

This writes `planning_descent.gif`.

### Panel crops

Use `--save-panel-crops` to save per-panel PNG crops next to the combined storyboard figure.

### Saved frame directories

Use `--save-frames` to keep intermediate GIF frames instead of writing them to a temporary directory.

## Themes and export polish

Available themes:

- `slide_dark`
- `paper`

For presentation slides with white backgrounds:

```powershell
python scripts/capstone/run_snapshot_storyboard.py `
  --run-dir data/run_snapshots/run_20260531_184606 `
  --mode planning `
  --up-to-row 5 `
  --out-dir presentation_outputs/run_20260531_184606/planning_row5_paper `
  --theme paper `
  --dpi 300
```

The storyboard writes both PNG and SVG, so the SVG can be opened directly in Illustrator for final annotation and layout work.

## Notes

- Only rows with `place_result == "placed"` and valid box fields are included in bag-state reconstruction.
- If the manifest contains duplicate `object_i` values, displayed placement numbers come from manifest row order.
- If point-cloud conversion or placement alignment fails, the script prints a warning and still exports AABB-only figures.
- If `config/surface_zones.json` contains `New Bag Test`, that bag footprint is used; otherwise bounds are inferred from placed boxes.
