# config

Robot, camera, scene, and safety configuration modules.

Where to change things:

- Bag pose and bag footprint for wet run, demos, and saved-image validation:
  `config/surface_zones.json`
  Update the `New Bag Test` entry when the grocery bag moves in XY, changes size, or needs a new default place yaw.
  `config/place/place_config.py` treats this as the canonical `PLACE_SCENE_*` source.

- Negative-bin bag floor / release policy:
  `config/place/place_config.py`
  This is where the shared place-Z policy and bag-local planner padding live.

- Workspace filtering profiles:
  `config/workspace/workspace_config.py`
  `wet_run` keeps real platform bounds on.
  `saved_photo_test` disables those platform bounds for older training photos taken before the platform existed.

- Runtime context switching:
  `config/runtime_context.py`
  Scripts can default to `wet_run`, `saved_photo_test`, or `demo`.
  For quick overrides you can use `GB_PLACE_SCENE_NAME`, `GB_RUNTIME_CONTEXT`, and `GB_WORKSPACE_PROFILE`.

- Local-only experiments:
  Copy `config/local_runtime_overrides.example.py` to `config/local_runtime_overrides.py` and set temporary overrides there.
