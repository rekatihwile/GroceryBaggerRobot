from __future__ import annotations

"""Runtime context and optional local overrides.

This keeps "wet robot", "saved-photo validation", and "demo" behaviors easy to
switch without scattering ad-hoc booleans across scripts.
"""

from dataclasses import dataclass
import importlib
import os
from typing import Any


LOCAL_OVERRIDE_MODULE = "config.local_runtime_overrides"
ENABLE_LOCAL_CONFIG_OVERRIDES = True

ENV_RUNTIME_CONTEXT = "GB_RUNTIME_CONTEXT"
ENV_PLACE_SCENE_NAME = "GB_PLACE_SCENE_NAME"
ENV_WORKSPACE_PROFILE = "GB_WORKSPACE_PROFILE"


@dataclass(frozen=True)
class RuntimeContext:
    name: str
    workspace_profile_name: str
    notes: str = ""


KNOWN_RUNTIME_CONTEXTS: dict[str, RuntimeContext] = {
    "wet_run": RuntimeContext(
        name="wet_run",
        workspace_profile_name="wet_run",
        notes="Use real platform/workspace filters and live-robot conservative defaults.",
    ),
    "saved_photo_test": RuntimeContext(
        name="saved_photo_test",
        workspace_profile_name="saved_photo_test",
        notes="Disable platform/workspace bounds for older saved photos taken before the platform existed.",
    ),
    "demo": RuntimeContext(
        name="demo",
        workspace_profile_name="saved_photo_test",
        notes="Use bag/demo geometry with no platform workspace filtering.",
    ),
}


def _load_local_override_module() -> Any | None:
    if not ENABLE_LOCAL_CONFIG_OVERRIDES:
        return None
    try:
        return importlib.import_module(LOCAL_OVERRIDE_MODULE)
    except Exception:
        return None


def _local_override_value(name: str) -> Any | None:
    module = _load_local_override_module()
    if module is None or not hasattr(module, name):
        return None
    return getattr(module, name)


def _resolve_override(env_var: str, local_name: str, default_value: str) -> str:
    env_value = os.environ.get(env_var)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    local_value = _local_override_value(local_name)
    if local_value is not None and str(local_value).strip():
        return str(local_value).strip()
    return str(default_value).strip()


def resolve_runtime_context_name(default_name: str = "wet_run") -> str:
    return _resolve_override(ENV_RUNTIME_CONTEXT, "ACTIVE_RUNTIME_CONTEXT", default_name).lower()


def resolve_runtime_context(default_name: str = "wet_run") -> RuntimeContext:
    name = resolve_runtime_context_name(default_name)
    if name not in KNOWN_RUNTIME_CONTEXTS:
        known = ", ".join(sorted(KNOWN_RUNTIME_CONTEXTS))
        raise ValueError(f"unknown runtime context {name!r}; expected one of {known}")
    return KNOWN_RUNTIME_CONTEXTS[name]


def resolve_place_scene_name(default_name: str) -> str:
    env_value = os.environ.get(ENV_PLACE_SCENE_NAME)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    local_value = _local_override_value("PLACE_SCENE_NAME_OVERRIDE")
    if local_value is not None and str(local_value).strip():
        return str(local_value).strip()
    return str(default_name).strip()


def resolve_workspace_profile_name(default_name: str) -> str:
    return _resolve_override(ENV_WORKSPACE_PROFILE, "WORKSPACE_PROFILE_OVERRIDE", default_name).lower()
