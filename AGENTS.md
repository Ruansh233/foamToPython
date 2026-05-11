# Codex Workspace Instructions

## Project Overview

`foamToPython` is a Python package for reading and writing OpenFOAM field/list
data and running POD-related workflows. The main package lives in
`src/foamToPython/`, with POD helpers in `src/PODopenFOAM/`.

Prioritize correctness for OpenFOAM file formats, NumPy array behavior, and
backward-compatible public APIs.

## Environment

- Use `uv` for dependency and virtual environment management.
- Do not create separate `venv` or conda environments for this workspace.
- Prefer `uv run <command>` when running Python tools.
- Python support starts at 3.8, so avoid syntax or typing features that require
  newer versions unless the project metadata is updated intentionally.

Common commands:

```bash
uv pip install -e .[dev]
uv run pytest
uv run pytest tests/test_of_field.py
```

Optional example dependencies:

```bash
uv pip install -e .[examples]
```

## Repository Structure

- `src/foamToPython/readOFField.py`: OpenFOAM field reading API.
- `src/foamToPython/readOFList.py`: OpenFOAM list/listList parsing API.
- `src/foamToPython/_field_parser.py`: Shared parsing helpers.
- `src/foamToPython/_field_writer.py`: Field writing helpers.
- `src/foamToPython/_mp_utils.py`: Multiprocessing utilities.
- `src/PODopenFOAM/`: POD analysis helpers.
- `tests/`: Pytest suite.
- `examples/`: Example and validation scripts.

## Coding Guidelines

- Keep changes focused and small; avoid unrelated refactors.
- Preserve existing public names and call patterns unless the user explicitly
  asks for an API change.
- Prefer simple, explicit parsing and writing logic over clever abstractions;
  OpenFOAM syntax edge cases should remain easy to reason about.
- Maintain NumPy array shape and dtype behavior carefully, especially for scalar
  versus vector fields and serial versus parallel cases.
- Use ASCII for new text unless an existing file already uses non-ASCII content.
- Add comments only when they clarify non-obvious OpenFOAM format handling or
  performance-sensitive logic.

## Testing Expectations

- Run `uv run pytest` after code changes when feasible.
- For targeted changes, run the nearest focused test first, then the full suite
  if the change is not trivial.
- The cavity validation example may require OpenFOAM and `foamlib`; do not assume
  those are installed or sourced in every environment.
- If a test cannot run because an external OpenFOAM dependency is unavailable,
  report that clearly and include the command attempted.

## Dependency Policy

- Keep runtime dependencies minimal. The core package currently depends on
  `numpy`.
- Add new runtime dependencies only when they are clearly justified.
- Put test-only or example-only dependencies in the appropriate optional
  dependency group in `pyproject.toml`.

## Git And Workspace Safety

- Check `git status --short` before editing.
- Do not revert or overwrite user changes unless explicitly asked.
- Do not commit, amend, merge, or push unless the user asks.
- Do not remove generated or data files unless the user asks or the task clearly
  requires it and the action is safe.

## Documentation

- Update `README.md` when behavior, user-facing APIs, installation, or example
  commands change.
- Keep examples runnable and aligned with the current API.
