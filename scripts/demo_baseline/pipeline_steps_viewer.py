from __future__ import annotations

try:
    from ._bootstrap import ensure_repo_root_on_path
except ImportError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.capstone.pipeline_steps_viewer import main


if __name__ == "__main__":
    raise SystemExit(main())
