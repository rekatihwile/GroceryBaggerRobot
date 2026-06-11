from __future__ import annotations

"""Shared paths and export helpers for capstone paper figures."""

from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SNAPSHOTS_ROOT = REPO_ROOT / "data" / "run_snapshots"
PUBLICATION_OUTPUT_ROOT = REPO_ROOT / "paper_figure_sources"
PUBLICATION_RUNS_ROOT = PUBLICATION_OUTPUT_ROOT / "runs"
PUBLICATION_GLOBAL_ROOT = PUBLICATION_OUTPUT_ROOT / "global"
PUBLICATION_FIGURES_ROOT = PUBLICATION_OUTPUT_ROOT / "paper_figures"
DEFAULT_PUBLICATION_DPI = 600


def newest_run_dir(runs_root: Path = RUN_SNAPSHOTS_ROOT) -> Path:
    candidates = [path for path in runs_root.glob("run_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run_* snapshot folders found under {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_dir(value: Path | str | None) -> Path:
    path = Path(value).expanduser().resolve() if value else newest_run_dir().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {path}")
    return path


def run_output_dir(run_dir: Path, output_root: Path | str | None = None) -> Path:
    if output_root is None:
        return (PUBLICATION_RUNS_ROOT / run_dir.name).resolve()
    root = Path(output_root).expanduser().resolve()
    if root.name == run_dir.name:
        return root
    if root.name == "runs":
        return root / run_dir.name
    return root


def output_dir_for_images(images_dir: Path, output_root: Path | str | None = None) -> Path:
    images_dir = images_dir.expanduser().resolve()
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    if images_dir.name.startswith("run_"):
        return run_output_dir(images_dir)
    return PUBLICATION_GLOBAL_ROOT / "legacy_pair_visualizers"


def safe_name(value: object, fallback: str = "item") -> str:
    text = str(value or fallback).strip().replace(" ", "_")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-"))
    return cleaned or fallback


def save_figure_bundle(
    fig,
    stem: Path,
    *,
    dpi: int = DEFAULT_PUBLICATION_DPI,
    formats: Iterable[str] = ("png", "pdf"),
    facecolor=None,
    bbox_inches: str | None = "tight",
) -> list[Path]:
    stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        path = stem.with_suffix(f".{suffix}")
        kwargs = {
            "bbox_inches": bbox_inches,
            "facecolor": facecolor if facecolor is not None else fig.get_facecolor(),
        }
        if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = max(80, int(dpi))
        fig.savefig(path, **kwargs)
        written.append(path)
    return written

