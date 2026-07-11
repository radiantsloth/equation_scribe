# Core Migration Plan

## Purpose

This document is the authoritative execution plan for centralizing shared backend,
detector, and future reusable code into `packages/core`.

The target outcome is a clean shared Python package named
`equation_scribe_core` that owns common domain models, storage primitives,
configuration helpers, and path utilities. Application-specific orchestration,
framework wiring, and UI code should remain outside core unless they become
truly shared.

This plan is written so an LLM or human engineer can execute the migration
incrementally with low risk.

## Primary Goals

- Centralize paper profile schemas.
- Centralize JSONL read/write helpers.
- Centralize index management.
- Centralize configuration and environment handling.
- Centralize common path utilities.
- Centralize shared DTOs and Pydantic models.
- Reduce duplicate implementations across:
  - `equation_scribe/`
  - `apps/web/backend/`
  - `equation_scribe/detector/`

## Non-Goals For This Migration Phase

- Do not move FastAPI route wiring into core.
- Do not move PDF rendering, OCR, detector inference, or recognition pipelines
  into core unless they become stable shared primitives later.
- Do not redesign the on-disk schema for `equations.jsonl` or `index.json`
  during this phase.
- Do not mix framework concerns with domain/storage concerns.

## Engineering Standards

Every migration step must follow these standards:

- Preserve behavior unless the step explicitly includes a tested behavior
  change.
- Keep public APIs small and explicit.
- Prefer typed interfaces over raw dictionaries when the structure is durable.
- Add clear docstrings to public modules, classes, and functions.
- Add concise explanatory comments where the design or control flow is not
  obvious.
- Write comments for understanding, not narration.
- Keep code approachable for a Python learner:
  - use readable names
  - separate concerns cleanly
  - avoid unnecessary indirection
  - explain why a helper exists when that is not obvious
- Add or update tests with each migration step that changes behavior or moves
  critical logic.
- Maintain backwards-compatible wrappers temporarily when that reduces risk.

## Target Package Layout

```text
packages/core/
  pyproject.toml
  README.md
  src/equation_scribe_core/
    __init__.py
    config/
      __init__.py
      constants.py
      settings.py
    io/
      __init__.py
      jsonl.py
      profile_store.py
      index_store.py
    models/
      __init__.py
      paper_profiles.py
      index.py
    paths/
      __init__.py
      roots.py
      utils.py
```

## Source Mapping

These are the main current sources that should be absorbed or wrapped by core:

- `equation_scribe/config.py`
- `equation_scribe/store.py`
- `equation_scribe/profile_index.py`
- `apps/web/backend/schemas.py`
- `apps/web/backend/storage.py`
- selected JSONL/path helpers in detector scripts

These should remain outside core for now:

- `apps/web/backend/main.py`
- `apps/web/backend/services/`
- `equation_scribe/pdf_ingest.py`
- `equation_scribe/detector/inference.py`
- `equation_scribe/recognition/`

## Current Duplication Map

This section summarizes the key duplication and centralization seams observed in
the repository at planning time. A new agent should use this as the initial
orientation map before making further changes.

### Shared schemas and DTOs

- `apps/web/backend/schemas.py` currently owns shared domain models such as:
  - `Box`
  - `EquationRecord`
- These models are backend-specific in location, but not in purpose.
- They should move into core before wider storage/index migration continues.

### JSONL helpers and profile persistence

- `equation_scribe/store.py` has generic JSONL append helpers and directory
  helpers.
- `apps/web/backend/storage.py` has its own read/append/update/delete JSONL
  logic for equation profiles.
- `equation_scribe/detector/split_recognition_pairs.py` defines local
  `read_jsonl` and `write_jsonl`.
- Additional detector scripts perform direct JSONL writes inline.

### Index management

- `equation_scribe/profile_index.py` already centralizes most index logic.
- `apps/web/backend/main.py` duplicates index loading instead of importing the
  shared helper.

### Runtime settings and environment handling

- `apps/web/backend/main.py` reads environment variables directly for:
  - `PROFILES_ROOT`
  - `PAPERS_ROOT`
- Root creation is handled locally there instead of through a shared settings
  layer.

### Path helpers

- `equation_scribe/store.py` defines `paper_dir`.
- `apps/web/backend/storage.py` defines `equations_path`.
- Detector scripts repeatedly create directories and resolve common data paths
  inline.

### Constants vs settings

- `equation_scribe/config.py` contains stable constants.
- It does not currently represent runtime settings, which should remain a
  separate concern in core.

## Current Repo Packaging State

- `packages/core` now exists as a standalone package scaffold.
- Repo packaging discovery was updated to search `packages/core/src`.
- The package can be discovered by setuptools.
- Plain imports from repo root still require either:
  - installing `packages/core` into the active environment, or
  - adding `packages/core/src` to `PYTHONPATH`

This environment ergonomics gap should be remembered during future verification
work.

## Execution Rules

Follow these rules during implementation:

1. Make one logical migration step at a time.
2. Keep each step small enough to review and test independently.
3. Prefer introducing core code first, then switching consumers, then deleting
   duplicates.
4. Leave compatibility wrappers in place until all known consumers are moved.
5. When moving a shared concept, migrate the tests with it or add new tests
   before deleting the old implementation.
6. Update the progress document after every completed step.
7. When a step reveals a new repo constraint or environment quirk, record it in
   the progress document so the next agent does not have to rediscover it.

## Migration Steps

### Step 1: Bootstrap `packages/core`

Objective:
Create the installable package skeleton for `equation_scribe_core` and wire the
repo packaging so the package can be discovered and imported.

Tasks:

- Create `packages/core/pyproject.toml`.
- Create `packages/core/README.md`.
- Create `src/equation_scribe_core/` with empty subpackages:
  - `config`
  - `io`
  - `models`
  - `paths`
- Update repo packaging discovery to include `packages/core/src`.
- Verify setuptools can discover the package.
- Verify Python import behavior and document any remaining environment setup gap.

Completion criteria:

- `equation_scribe_core` exists as a valid Python package.
- Package discovery succeeds.
- Import behavior is verified and documented.

### Step 2: Centralize Stable Constants

Objective:
Move reusable numeric and static constants into `core.config.constants`.

Tasks:

- Move shared constants from `equation_scribe/config.py` into
  `equation_scribe_core.config.constants`.
- Keep constant names stable unless there is a compelling reason to rename.
- Leave a thin compatibility wrapper in `equation_scribe/config.py` if needed.
- Add module and symbol docstrings explaining what the constants control.

Completion criteria:

- Shared constants are owned by core.
- Existing imports continue to work, either directly or via a wrapper.

### Step 3: Centralize Runtime Settings

Objective:
Create one clear place for environment-derived runtime paths and settings.

Tasks:

- Create `equation_scribe_core.config.settings`.
- Define a settings object or helper functions for:
  - profiles root
  - papers root
  - future model roots if needed
- Ensure directory creation behavior is explicit and documented.
- Replace direct `os.getenv(...)` access in `apps/web/backend/main.py` with the
  shared settings layer.

Completion criteria:

- Runtime env/path resolution is centralized.
- Web/backend no longer owns its own root-path logic.

### Step 4: Centralize Shared Domain Models

Objective:
Move durable paper-profile and index schemas into core.

Tasks:

- Create `equation_scribe_core.models.paper_profiles`.
- Move shared models such as:
  - `Box`
  - `EquationRecord`
- Create `equation_scribe_core.models.index`.
- Add typed index models such as:
  - `PaperIndexEntry`
  - `PaperIndex`
- Keep API-only request/response models out of core unless reused elsewhere.
- Add docstrings that explain how each model fits into the system.

Completion criteria:

- Shared schemas live in core.
- Web/backend imports shared domain models from core instead of local schema
  modules.

### Step 5: Centralize Path Utilities

Objective:
Provide shared helpers for common filesystem layout operations.

Tasks:

- Create `equation_scribe_core.paths.roots`.
- Create helpers for:
  - `paper_dir`
  - `equations_path`
  - root directory creation
- Create `equation_scribe_core.paths.utils` for small generic path helpers.
- Keep behavior simple and explicit.
- Document path assumptions and directory layout in docstrings.

Completion criteria:

- Shared path logic is owned by core.
- Duplicate path helpers in app/root code are removed or wrapped.

### Step 6: Centralize JSONL Helpers

Objective:
Provide one shared implementation for reading and writing JSONL files.

Tasks:

- Create `equation_scribe_core.io.jsonl`.
- Implement shared helpers such as:
  - `read_jsonl`
  - `write_jsonl`
  - `append_jsonl`
  - one rewrite helper for full-file updates
- Support both raw dictionaries and Pydantic model serialization where useful.
- Document malformed-line handling decisions explicitly.
- Add unit tests for round-trip behavior and edge cases.

Completion criteria:

- JSONL logic is no longer duplicated across web/backend, root package, and
  detector helpers.

### Step 7: Centralize Profile Persistence

Objective:
Unify paper-profile storage behavior in one module.

Tasks:

- Create `equation_scribe_core.io.profile_store`.
- Move shared paper-profile persistence logic from:
  - `apps/web/backend/storage.py`
  - `equation_scribe/store.py`
- Include operations for:
  - read equations
  - append equation
  - update by `eq_uid`
  - delete by `eq_uid`
  - optional backup/history behavior
- Add clear function docstrings that explain expected file layout and write
  semantics.

Completion criteria:

- Shared profile persistence logic is owned by core.
- App/root storage modules are thin wrappers or direct import consumers.

### Step 8: Centralize Index Management

Objective:
Move index management into core while preserving locking and on-disk format.

Tasks:

- Create `equation_scribe_core.io.index_store`.
- Move or wrap logic from `equation_scribe/profile_index.py`.
- Preserve locking behavior and index file compatibility.
- Use typed models where practical without forcing a risky schema rewrite.
- Update web/backend to consume the shared index loader instead of reimplementing
  it.

Completion criteria:

- Index management is owned by core.
- Duplicate index loading logic is removed.

### Step 9: Migrate Web Backend Consumers

Objective:
Make the FastAPI backend consume core primitives instead of local duplicates.

Tasks:

- Replace imports from local schema/storage helpers with core imports.
- Remove duplicate `load_profiles_index` logic from `apps/web/backend/main.py`.
- Keep API request/response models local if they are not shared.
- Run focused regression tests for profile listing, save/update/delete, and
  index endpoints.

Completion criteria:

- Web/backend uses core for shared logic.
- Behavior remains stable.

### Step 10: Migrate Root Package Consumers

Objective:
Make `equation_scribe/` consume core primitives instead of owning duplicate
implementations.

Tasks:

- Update `equation_scribe/store.py` to wrap or import core storage helpers.
- Update `equation_scribe/profile_index.py` to wrap or import core index helpers.
- Update callers such as `equation_scribe/autodetect_equations.py`.
- Preserve existing CLI-facing behavior during the transition.

Completion criteria:

- Root package duplicate logic is removed or reduced to compatibility shims.

### Step 11: Migrate Detector Utility Consumers

Objective:
Adopt shared JSONL/path helpers in detector scripts where it is low risk.

Tasks:

- Start with scripts that already duplicate JSONL helpers, such as:
  - `equation_scribe/detector/split_recognition_pairs.py`
  - `equation_scribe/detector/make_pairs.py`
  - `equation_scribe/detector/make_recognition_pairs.py`
  - `equation_scribe/detector/synctex_extractor.py`
- Keep detector algorithm logic local.
- Use shared JSONL/path helpers only where they reduce duplication cleanly.

Completion criteria:

- Detector utility scripts stop duplicating generic persistence helpers.

### Step 12: Test Consolidation And Cleanup

Objective:
Finish the migration by consolidating tests and removing obsolete duplicate code.

Tasks:

- Add core-layer tests for:
  - JSONL read/write behavior
  - index round-trip behavior
  - overwrite/force rules
  - path helper behavior
  - settings resolution behavior
- Migrate or adapt existing tests such as `tests/test_writeindex.py`.
- Remove obsolete duplicate implementations after consumers are switched.
- Confirm docs and progress tracker are up to date.

Completion criteria:

- Shared logic is tested at the core layer.
- Duplicate helper implementations are deleted.
- Progress document reflects the final state.

## Review Checklist For Each Step

Before marking any step complete, verify:

- The code remains readable to a Python learner.
- Public functions/classes have docstrings.
- Comments explain design decisions where needed.
- No app-specific framework concern leaked into core.
- Imports moved in the smallest reasonable scope.
- Tests were added or updated when appropriate.
- The progress document was updated.

## Status Reference

The live status of this migration belongs in:

- `docs/architecture/core-migration-progress.md`

Do not treat this plan document as the live progress tracker.

## Handoff Instructions For A New Agent

If continuing this effort in a new chat or with a different LLM:

1. Read this document fully.
2. Read `docs/architecture/core-migration-progress.md` fully.
3. Inspect the files referenced in the progress document before editing.
4. Start with the single recommended next step from the progress document unless
   the user explicitly changes priority.
5. After completing a step:
   - update the progress document
   - note verification performed
   - record any environment or packaging issues discovered
