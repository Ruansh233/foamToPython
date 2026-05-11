import importlib.util
import os
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "validate_cavity_serial_parallel.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("validate_cavity_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_latest_numeric_time_ignores_non_time_directories(tmp_path: Path) -> None:
    for name in ["0", "0.5", "constant", "processor0", "postProcessing", "2"]:
        (tmp_path / name).mkdir()

    module = _load_example_module()

    assert module.latest_numeric_time(tmp_path) == "2"


def test_ensure_openfoam_available_passes_with_foam_appbin(monkeypatch) -> None:
    module = _load_example_module()
    monkeypatch.setenv("FOAM_APPBIN", "/tmp/fake-openfoam-bin")
    monkeypatch.setattr(module.shell_utils, "which", lambda _name: None)

    module.ensure_openfoam_available()


def test_default_tutorial_case_points_inside_repo() -> None:
    module = _load_example_module()

    expected = ROOT / "examples" / "cavity"
    assert module.DEFAULT_TUTORIAL_CASE == expected
    assert module.DEFAULT_TUTORIAL_CASE.is_dir()


def test_ensure_openfoam_available_passes_with_commands_on_path(monkeypatch) -> None:
    module = _load_example_module()
    monkeypatch.delenv("FOAM_APPBIN", raising=False)
    monkeypatch.setattr(module.shell_utils, "which", lambda _name: "/usr/bin/fake")

    module.ensure_openfoam_available()


def test_ensure_openfoam_available_raises_when_missing(monkeypatch) -> None:
    module = _load_example_module()
    monkeypatch.delenv("FOAM_APPBIN", raising=False)
    monkeypatch.setattr(module.shell_utils, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="OpenFOAM is not available"):
        module.ensure_openfoam_available()
