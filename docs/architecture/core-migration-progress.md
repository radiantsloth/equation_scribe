# Core Migration Progress

## Purpose

This document tracks work completed against the core migration plan and records
the next recommended step. Update this file after each migration step or any
meaningful partial progress.

Primary plan:

- [core-migration-plan.md](C:/Data/repos/equation_scribe/docs/architecture/core-migration-plan.md)

## Current Status

- Overall phase: migration complete
- Last updated: 2026-07-09
- Immediate next step: none
- Recommended next step: no further migration step pending

## Files Touched So Far

The following files were created or modified so far:

- `pyproject.toml`
- `setup.cfg`
- `packages/core/pyproject.toml`
- `packages/core/README.md`
- `packages/core/src/equation_scribe_core/__init__.py`
- `packages/core/src/equation_scribe_core/config/__init__.py`
- `packages/core/src/equation_scribe_core/config/constants.py`
- `packages/core/src/equation_scribe_core/config/settings.py`
- `packages/core/src/equation_scribe_core/io/__init__.py`
- `packages/core/src/equation_scribe_core/io/jsonl.py`
- `packages/core/src/equation_scribe_core/io/index_store.py`
- `packages/core/src/equation_scribe_core/io/profile_store.py`
- `packages/core/src/equation_scribe_core/models/__init__.py`
- `packages/core/src/equation_scribe_core/models/paper_profiles.py`
- `packages/core/src/equation_scribe_core/models/index.py`
- `packages/core/src/equation_scribe_core/paths/__init__.py`
- `packages/core/src/equation_scribe_core/paths/roots.py`
- `apps/web/backend/main.py`
- `conftest.py`
- `equation_scribe/autodetect_equations.py`
- `equation_scribe/config.py`
- `equation_scribe/detector/data_prep.py`
- `equation_scribe/detector/data_prep_coco.py`
- `equation_scribe/detector/split_recognition_pairs.py`
- `equation_scribe/profile_index.py`
- `equation_scribe/store.py`
- `equation_scribe/ui_gradio.py`
- `tests/test_autodetect_equations.py`
- `tests/test_backend_main.py`
- `tests/test_core_jsonl.py`
- `tests/test_core_constants.py`
- `tests/test_core_paths.py`
- `tests/test_core_settings.py`
- `tests/test_detector_data_prep.py`
- `tests/test_profile_store.py`
- `docs/architecture/core-migration-progress.md`

Shared domain logic now lives in core. Runtime settings, stable constants,
storage helpers, index helpers, and profile persistence now live in core.
Web/backend, active root consumers, and low-risk detector utility consumers
have been migrated. Cleanup and test consolidation are complete.

## Known Repo Findings

These findings motivated the current migration order and should be treated as
established context unless later code changes invalidate them.

### Duplication hotspots

- Shared domain models, JSONL helpers, profile persistence, index management,
  runtime settings, and stable constants have been centralized into core.
- The remaining compatibility shims are limited to legacy root import paths:
  - `equation_scribe/store.py`
  - `equation_scribe/profile_index.py`
- `equation_scribe/ui_gradio.py` remains deprecated legacy code and is no
  longer part of the active application path after the shift to the npm-based
  frontend.

### Recommended migration order rationale

- Move shared models early because storage and index helpers depend on stable
  shared shapes.
- Move JSONL/path/storage helpers after models so their signatures can use the
  shared types cleanly.
- Migrate app consumers only after the core interfaces exist.
- With shared JSONL helpers in place, the next low-risk move is to consolidate
  profile persistence around those primitives instead of duplicating update and
  delete behavior in app/root modules.
- With shared profile persistence now in core, the next clean cut is to migrate
  application consumers to direct core imports and shrink compatibility layers.
- After the backend consumer migration, the remaining high-value duplication is
  in root-package wrappers and direct JSONL reads outside core.
- After active root consumers are moved, the remaining generic duplication is
  concentrated in detector utility scripts and runtime settings.
- With low-risk detector utility scripts migrated, the remaining architectural
  gap is the shared runtime settings layer and the final cleanup/testing pass.
- With shared runtime settings in place, the remaining work was final constant
  consolidation and cleanup of compatibility layers and tests.

## Known Environment Quirks

### Python interpreter behavior

- `python.exe` from the shell environment failed with:
  - `A specified logon session does not exist. It may already have been terminated`
- The repo virtual environment interpreter is available at:
  - `C:\Data\repos\equation_scribe\.venv\Scripts\python.exe`

### Import verification behavior

- `equation_scribe_core` is discoverable by setuptools.
- Plain import from repo root fails unless `packages/core/src` is added to
  `PYTHONPATH` or the package is installed into the environment.
- Import verification succeeded with:
  - `PYTHONPATH=packages/core/src`
  - `.venv\Scripts\python.exe`

Implication:

- Future verification steps involving `equation_scribe_core` should use the
  virtualenv interpreter and account for the current source-path requirement.

### Git worktree behavior

- `git status` currently fails in this environment with Git's dubious ownership
  protection for `C:/Data/repos/equation_scribe`.

Implication:

- Repo-state verification should not rely on Git commands here unless the safe
  directory setting is addressed outside this migration work.

## Progress Summary

### Completed

#### Step 1: Bootstrap `packages/core`

Status:

- Completed

Work performed:

- Created the standalone core package scaffold under
  `packages/core/src/equation_scribe_core/`.
- Added empty subpackages for:
  - `config`
  - `io`
  - `models`
  - `paths`
- Added `packages/core/pyproject.toml`.
- Added `packages/core/README.md`.
- Updated repo packaging discovery so setuptools searches `packages/core/src`.

Verification performed:

- Verified setuptools discovery sees:
  - `equation_scribe_core`
  - `equation_scribe_core.config`
  - `equation_scribe_core.io`
  - `equation_scribe_core.models`
  - `equation_scribe_core.paths`
- Verified Python import succeeds when `packages/core/src` is on `PYTHONPATH`.

Notes:

- Plain `import equation_scribe_core` from repo root still fails in the virtual
  environment unless `packages/core` is installed or `packages/core/src` is
  added to `PYTHONPATH`.
- This is an environment ergonomics gap, not a package-structure failure.

#### Step 4: Centralize shared domain models

Status:

- Completed

Work performed:

- Added `equation_scribe_core.models.paper_profiles` with shared paper-profile
  domain models:
  - `BBox`
  - `Box`
  - `EquationRecord`
- Added `equation_scribe_core.models.index` with typed index models:
  - `PaperIndexEntry`
  - `PaperIndex`
- Updated `equation_scribe_core.models.__init__` to export those shared models.
- Converted `apps/web/backend/schemas.py` into a compatibility wrapper that
  re-exports the core-owned models.

Verification performed:

- Verified core model imports succeed with:
  - `PYTHONPATH=packages/core/src`
  - `.venv\Scripts\python.exe`
- Verified the backend compatibility wrapper initially failed when
  `equation_scribe_core` was not on `sys.path`, which exposed a real source-tree
  import gap.
- Added a temporary source-tree fallback in `apps/web/backend/schemas.py` so the
  wrapper can locate `packages/core/src` during the transition.

Notes:

- Shared domain models are now owned by core.
- Backend callers can continue importing from `apps/web/backend/schemas.py`
  during the transition, which reduces migration risk.
- Direct imports from `equation_scribe_core` still depend on the current source
  path or installation setup described in the environment quirks section.
- The backend compatibility wrapper now compensates for that gap temporarily.

#### Step 8: Centralize index management

Status:

- Completed

Work performed:

- Added `equation_scribe_core.io.index_store` as the shared typed
  implementation for `index.json` load/save/register behavior.
- Updated `equation_scribe_core.io.__init__` to export the shared index
  helpers.
- Replaced `equation_scribe/profile_index.py` with a compatibility wrapper that
  delegates to core while preserving the legacy dictionary-based API.
- Updated `apps/web/backend/main.py` to use the shared index loader instead of
  reimplementing `index.json` reads locally.

Verification performed:

- Ran the existing regression test suite for index behavior:
  - `.venv\Scripts\python.exe -m pytest tests\test_writeindex.py`
- Result:
  - `2 passed`

Notes:

- Core now owns the shared index implementation.
- The legacy root import path remains stable for existing callers.
- The web backend no longer duplicates index loading logic.

#### Step 5: Centralize path utilities

Status:

- Completed

Work performed:

- Added `equation_scribe_core.paths.roots` as the shared home for small
  profile-layout helpers.
- Added and exported:
  - `ensure_dir`
  - `paper_dir`
  - `equations_path`
  - `glossary_path`
- Updated `equation_scribe/store.py` to consume the shared path helpers.
- Updated `apps/web/backend/storage.py` to consume the shared `equations_path`
  helper.

Verification performed:

- Verified the core path helpers directly with:
  - `PYTHONPATH=packages/core/src`
  - `.venv\Scripts\python.exe`
- Verified `equation_scribe/store.py` resolves the shared `paper_dir` helper.
- Verified `apps/web/backend/storage.py` resolves the shared `equations_path`
  helper when imported in package context.

Notes:

- Core now owns the paper-profile path layout primitives used by storage code.
- The repo still relies on temporary source-tree fallbacks until the package
  import workflow is cleaned up more broadly.

#### Step 6: Centralize JSONL helpers

Status:

- Completed

Work performed:

- Added `equation_scribe_core.io.jsonl` as the shared home for JSONL helpers.
- Added and exported:
  - `JsonlEntry`
  - `read_jsonl`
  - `read_jsonl_entries`
  - `append_jsonl`
  - `write_jsonl`
  - `rewrite_jsonl`
- Implemented shared serialization support for plain JSON values and Pydantic
  models.
- Documented the default malformed-line behavior:
  - `read_jsonl` skips malformed lines by default
  - `read_jsonl_entries` preserves malformed raw lines for lossless rewrites
- Updated `equation_scribe/store.py` to delegate JSONL appends to the shared
  core helper.
- Updated `apps/web/backend/storage.py` to use the shared core JSONL read,
  append, and rewrite helpers while preserving malformed lines during updates
  and deletes.
- Updated `equation_scribe/detector/split_recognition_pairs.py` to consume the
  shared JSONL read/write helpers instead of keeping local duplicates.
- Added focused tests for round-trip behavior, malformed-line handling,
  Pydantic serialization, and rewrite preservation.

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_writeindex.py tests\test_core_jsonl.py`
- Result:
  - `6 passed`

Notes:

- Core now owns the shared JSONL helper layer used by root, backend, and one
  detector utility.
- Full paper-profile persistence is still split across `equation_scribe/store.py`
  and `apps/web/backend/storage.py`, which is why Step 7 remains the next
  consolidation target.

#### Step 7: Centralize profile persistence

Status:

- Completed

Work performed:

- Added `equation_scribe_core.io.profile_store` as the shared home for equation
  profile persistence behavior.
- Added and exported:
  - `backup_profile_file`
  - `read_equations`
  - `append_equation`
  - `update_equation`
  - `delete_equation`
- Centralized history-backup behavior for `equations.jsonl` rewrites.
- Kept update/delete behavior compatible with existing semantics:
  - append on missing file during update
  - preserve malformed lines during rewrites
  - return `False` on delete when `eq_uid` is not found
- Updated `apps/web/backend/storage.py` into a thin compatibility layer over
  the shared core store.
- Updated `equation_scribe/store.py` so `save_equation` delegates to the shared
  core append helper while preserving its legacy `(root, paper_id, record)`
  signature.
- Left `save_symbol` in the root package because glossary persistence is still a
  small root-only helper and was not part of this migration step.
- Added focused tests for:
  - core append/read round-trip
  - update semantics and backup creation
  - delete semantics and backup creation
  - legacy root wrapper compatibility

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_writeindex.py tests\test_core_jsonl.py tests\test_profile_store.py`
- Result:
  - `10 passed`

Notes:

- Core now owns the shared equation-profile persistence layer used by backend
  and root wrappers.
- The web backend still imports profile helpers through `apps/web/backend/storage.py`
  rather than directly from core, so Step 9 remains the next consumer cleanup.
- The root package still exposes wrapper functions and some direct JSONL reads
  outside core, which will be addressed in later consumer migration work.

#### Step 9: Migrate web backend consumers

Status:

- Completed

Work performed:

- Updated `apps/web/backend/main.py` to import shared core primitives directly
  instead of going through backend-local compatibility wrappers for:
  - `EquationRecord`
  - `read_equations`
  - `append_equation`
  - `update_equation`
  - `delete_equation`
  - `load_index`
- Preserved the current source-tree fallback so backend imports still work
  before the repo adopts a cleaner installed-package workflow.
- Restored the `/papers/index` endpoint's direct-call return shape by converting
  the shared `PaperIndex` model back to the legacy JSON-ready dict shape at the
  endpoint boundary.
- Added focused backend regression tests that exercise:
  - profile index listing
  - equation listing
  - save/update/delete equation flows
  - missing-delete error handling
- Used import-time stubs in the backend tests for heavy ML modules so endpoint
  behavior can be verified in this environment without `torch`.

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_writeindex.py tests\test_core_jsonl.py tests\test_profile_store.py tests\test_backend_main.py`
- Result:
  - `14 passed`

Notes:

- The backend now consumes core directly for the shared profile and index
  primitives that were migrated in earlier steps.
- `apps/web/backend/schemas.py` and `apps/web/backend/storage.py` still exist as
  compatibility layers, but `main.py` no longer depends on them for the shared
  backend flows covered here.
- Importing `apps/web/backend/main.py` in this environment still requires test
  stubs for ML-heavy modules because `equation_scribe.recognition.inference`
  depends on `torch`.

#### Step 10: Migrate root package consumers

Status:

- Completed

Work performed:

- Updated `equation_scribe/autodetect_equations.py` to import shared core
  storage and index primitives directly instead of routing JSONL writes and
  index registration through root compatibility wrappers.
- Added `write_detected_equations(...)` so the active root CLI write path now
  uses:
  - `equation_scribe_core.io.write_jsonl`
  - `equation_scribe_core.io.register_paper`
- Preserved the existing CLI overwrite behavior for autodetect output:
  - do not overwrite `equations.jsonl` without `--force`
  - rotate the old file to `equations.jsonl.bak.<timestamp>` when `--force`
- Marked `equation_scribe/ui_gradio.py` as a deprecated legacy surface because
  the application has moved away from Gradio.
- Updated the deprecated Gradio module's profile read path to use the shared
  core `read_equations(...)` helper instead of opening and parsing
  `equations.jsonl` directly.
- Left `equation_scribe/store.py` and `equation_scribe/profile_index.py` in
  place as compatibility shims for older callers that may still import them.
- Added focused regression tests for the active root autodetect write path:
  - write-and-index behavior
  - `force=False` no-overwrite behavior
  - `force=True` backup-and-rewrite behavior

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_writeindex.py tests\test_core_jsonl.py tests\test_profile_store.py tests\test_backend_main.py tests\test_autodetect_equations.py -q`
- Result:
  - `16 passed`
- Ran:
  - `.venv\Scripts\python.exe -m pytest -q`
- Result:
  - `30 passed`

Notes:

- The active root CLI write path now consumes core directly for shared storage
  and index behavior.
- `equation_scribe/ui_gradio.py` is deprecated rather than a primary migration
  target, but its remaining profile read path now uses the shared core helper.
- Root wrapper modules still exist for compatibility, but active consumers no
  longer need to own JSONL parsing or index registration logic directly.

#### Step 11: Migrate detector utility consumers

Status:

- Completed

Work performed:

- Updated `equation_scribe/detector/data_prep.py` to consume the shared core
  `read_jsonl(...)` helper instead of opening and parsing JSONL lines directly.
- Updated `equation_scribe/detector/data_prep_coco.py` to consume the shared
  core `read_jsonl(...)` helper instead of duplicating line-by-line JSONL reads.
- Kept detector-specific conversion, rendering, and bbox logic local to the
  detector utilities.
- Added focused regression tests covering:
  - converting from a single `equations.jsonl` file
  - discovering profile files under a profiles root
  - converting a profiles-root directory to COCO output
- Updated the repo-level pytest temp-root strategy to use a unique repo-local
  base temp directory per test run, which was necessary to keep repeated full
  suite runs stable on Windows during this migration work.

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_detector_data_prep.py tests\test_writeindex.py tests\test_core_jsonl.py tests\test_profile_store.py tests\test_backend_main.py tests\test_autodetect_equations.py -q`
- Result:
  - `18 passed`
- Ran:
  - `.venv\Scripts\python.exe -m pytest -q`
- Result:
  - `32 passed`

Notes:

- This step intentionally covered only low-risk detector utilities that mostly
  duplicated generic JSONL handling.
- Detector inference and recognition logic remain local, which matches the
  migration plan's boundary for this phase.
- The only remaining full-suite warnings are unrelated Pillow deprecations in
  `test_recognition_preprocess.py`.

#### Step 3: Centralize runtime settings

Status:

- Completed

Work performed:

- Added `equation_scribe_core.config.settings` as the shared home for
  environment-derived runtime path resolution.
- Added and exported:
  - `RuntimeSettings`
  - `get_runtime_settings`
  - `PROFILES_ROOT_ENV`
  - `PAPERS_ROOT_ENV`
  - `DEFAULT_PROFILES_ROOT`
  - `DEFAULT_PAPERS_ROOT`
- Centralized the backend's runtime root resolution and directory creation into
  that shared settings layer.
- Updated `apps/web/backend/main.py` to consume the shared settings instead of
  reading `os.getenv(...)` directly for:
  - `PROFILES_ROOT`
  - `PAPERS_ROOT`
- Added focused tests for:
  - default runtime path resolution
  - environment override behavior
  - backend import behavior through the shared settings layer

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_core_settings.py tests\test_backend_main.py tests\test_writeindex.py tests\test_core_jsonl.py tests\test_profile_store.py tests\test_autodetect_equations.py tests\test_detector_data_prep.py -q`
- Result:
  - `21 passed`
- Ran:
  - `.venv\Scripts\python.exe -m pytest -q`
- Result:
  - `35 passed`

Notes:

- The web backend no longer owns its own environment-derived root-path logic.
- Shared runtime path resolution is now available for future consumers beyond
  the backend if the project needs them later in the cleanup phase.

#### Step 2: Centralize stable constants

Status:

- Completed

Work performed:

- Added `equation_scribe_core.config.constants` as the shared home for durable
  numeric detector and synthetic-data constants.
- Added and exported:
  - `DEFAULT_DPI`
  - `PAGE_WIDTH_IN`
  - `PAGE_HEIGHT_IN`
  - `DEFAULT_PAGE_W_PX`
  - `DEFAULT_PAGE_H_PX`
  - `ROTATION_AUG_MAX_ANGLE`
  - `DESKEW_THRESHOLD_DEG`
  - `MAX_EQ_WIDTH_FRAC`
  - `MAX_EQ_HEIGHT_FRAC`
  - `NON_OVERLAP_IOU`
  - `MAX_PLACEMENT_ATTEMPTS`
- Updated `equation_scribe_core.config.__init__` so the shared constants are
  available from the package-level config namespace.
- Replaced `equation_scribe/config.py` with a thin compatibility wrapper that
  re-exports the core-owned constants and preserves the existing source-tree
  fallback import behavior.
- Added focused tests for:
  - direct core constant exports
  - legacy `equation_scribe.config` wrapper compatibility

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_core_constants.py tests\test_core_settings.py tests\test_detector_data_prep.py -q`
- Result:
  - `6 passed`

Notes:

- Active detector modules still import constants through
  `equation_scribe.config`, so the compatibility wrapper remains useful for the
  cleanup phase.
- There are still no dedicated tests for `equation_scribe/detector/preprocess.py`
  or `equation_scribe/detector/synthetic_coco.py`; the new constant tests cover
  the shared constant layer rather than those full consumers.

#### Step 12: Test Consolidation And Cleanup

Status:

- Completed

Work performed:

- Removed obsolete backend compatibility wrappers:
  - `apps/web/backend/schemas.py`
  - `apps/web/backend/storage.py`
- Shifted index-focused tests to import the shared core helpers directly instead
  of relying on the legacy `equation_scribe.profile_index` wrapper.
- Shifted the autodetect regression test to assert against the shared core
  index model directly.
- Added focused core-layer path tests covering:
  - `ensure_dir`
  - `paper_dir`
  - `equations_path`
  - `glossary_path`
- Cleaned the legacy root storage shim by removing migration-era debug output
  and unused imports while preserving its compatibility behavior.
- Kept the root compatibility wrappers in place for now:
  - `equation_scribe/store.py`
  - `equation_scribe/profile_index.py`
- Kept `equation_scribe/ui_gradio.py` out of active migration paths while
  treating it as deprecated legacy code rather than an actively supported UI.

Verification performed:

- Ran:
  - `.venv\Scripts\python.exe -m pytest tests\test_core_paths.py tests\test_writeindex.py tests\test_autodetect_equations.py tests\test_profile_store.py tests\test_backend_main.py tests\test_core_jsonl.py tests\test_core_constants.py tests\test_core_settings.py tests\test_detector_data_prep.py -q`
- Result:
  - `26 passed`
- Ran:
  - `.venv\Scripts\python.exe -m pytest -q`
- Result:
  - `40 passed, 2 warnings`

Notes:

- Shared logic is now tested primarily at the core layer, with only narrow
  legacy-wrapper coverage retained where compatibility shims still exist.
- The remaining full-suite warnings are unrelated Pillow deprecations in
  `equation_scribe/detector/tests/test_recognition_preprocess.py`.

### In Progress

- None

### Not Started

- None

## Recommended Next Step

Migration steps complete.

Reasoning:

- Shared models, storage, paths, index handling, detector utility reads,
  runtime settings, and stable constants are now centralized in core.
- The planned migration steps have been completed.
- Remaining work, if any, is ordinary follow-on maintenance rather than part of
  this migration plan.

Execution target:

- None.

Definition of done:

- Achieved.

## Handoff Notes For The Next Agent

Start by reading:

- [core-migration-plan.md](C:/Data/repos/equation_scribe/docs/architecture/core-migration-plan.md)
- [core-migration-progress.md](C:/Data/repos/equation_scribe/docs/architecture/core-migration-progress.md)

Then:

1. Verify the current branch/worktree state without reverting unrelated changes.
2. Treat the migration as complete unless the user explicitly asks for further
   cleanup beyond the original plan.
3. Before editing, inspect:
   - `apps/web/backend/main.py`
   - `equation_scribe/profile_index.py`
   - `equation_scribe/store.py`
4. When verifying imports or tests involving core, prefer the virtualenv
   interpreter and remember that `packages/core/src` is not yet automatically on
   `sys.path`.

## Update Instructions

When updating this document:

- Record the date.
- Mark the step status clearly.
- Summarize the work actually performed, not the intended work.
- Note any verification that was run.
- Call out blockers or environment issues explicitly.
- End with one clear next recommended step.
