from __future__ import annotations

import argparse
from typing import Callable

try:
    from ._bootstrap import ensure_repo_root_on_path
except ImportError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.capstone.aabb_placement_viewer import main as aabb_main
from scripts.capstone.autonomous_system_wrapper import main as system_main
from scripts.capstone.interactive_packing_demo import main as packing_main
from scripts.capstone.pipeline_steps_viewer import main as pipeline_main


_DEMO_MODES: dict[str, Callable[[list[str] | None], int]] = {
    "packing": packing_main,
    "system": system_main,
    "pipeline": pipeline_main,
    "aabb": aabb_main,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Baseline capstone demo launcher.",
    )
    parser.add_argument(
        "mode",
        choices=sorted(_DEMO_MODES),
        nargs="?",
        default="system",
        help="Which baseline demo to run.",
    )
    parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected demo.",
    )
    args = parser.parse_args(argv)
    return _DEMO_MODES[args.mode](args.demo_args)


if __name__ == "__main__":
    raise SystemExit(main())
