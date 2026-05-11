#!/usr/bin/env python3
"""Validate foamToPython with the vendored OpenFOAM v2312 cavity tutorial.

The example copies the in-repo `icoFoam/cavity` tutorial, activates OpenFOAM,
runs the case through foamlib, then validates serial and decomposed field I/O.
"""

from __future__ import annotations

import argparse
import os
import shutil as shell_utils
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foamToPython import OFField  # noqa: E402


DEFAULT_TUTORIAL_CASE = ROOT / "examples" / "cavity"


def latest_numeric_time(case_dir: Path) -> str:
    numeric_times: List[Tuple[float, str]] = []
    for path in case_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            numeric_times.append((float(path.name), path.name))
        except ValueError:
            continue

    if not numeric_times:
        raise FileNotFoundError(f"No numeric time directories found in {case_dir}")
    return max(numeric_times, key=lambda item: item[0])[1]


def ensure_openfoam_available() -> None:
    required_bins = ["blockMesh", "icoFoam", "decomposePar"]
    missing_bins = [name for name in required_bins if shell_utils.which(name) is None]
    if missing_bins and "FOAM_APPBIN" not in os.environ:
        missing_text = ", ".join(missing_bins)
        raise RuntimeError(
            "OpenFOAM is not available in the current environment. "
            "Please install OpenFOAM and source its environment (for example via "
            "`source .../etc/bashrc` or your shell activation command), then rerun. "
            f"Missing commands: {missing_text}"
        )


def copy_case(tutorial_case: Path, work_dir: Path) -> Path:
    case_dir = work_dir / "cavity"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(tutorial_case, case_dir)
    return case_dir


def validate_read_write(
    case_dir: Path,
    time_name: str,
    field_name: str,
    data_type: str,
    *,
    parallel: bool,
    output_time: str,
) -> OFField:
    source = case_dir / time_name / field_name
    field = OFField(str(source), data_type, read_data=True, parallel=parallel)

    if field.dimensions.shape != (7,):
        raise AssertionError(f"{field_name}: expected 7 dimensions, got {field.dimensions}")
    if not field.boundaryField:
        raise AssertionError(f"{field_name}: boundaryField is empty")
    if parallel:
        if not isinstance(field.internalField, list) or not field.internalField:
            raise AssertionError(f"{field_name}: expected processor internal fields")
    else:
        if getattr(field.internalField, "size", 0) == 0:
            raise AssertionError(f"{field_name}: internalField is empty")

    target = case_dir / output_time / field_name
    field.writeField(str(target), precision=8)

    reread = OFField(str(target), data_type, read_data=True, parallel=parallel)
    if parallel:
        if len(reread.internalField) != len(field.internalField):
            raise AssertionError(f"{field_name}: processor count changed after write")
    else:
        if reread.internalField.shape != field.internalField.shape:
            raise AssertionError(f"{field_name}: shape changed after write")
    return field


def run_validation(parsed_args: argparse.Namespace) -> Path:
    try:
        from foamlib import FoamCase
    except ImportError as exc:
        raise ImportError(
            "foamlib is required for this example. Install it with `pip install foamlib`."
        ) from exc

    parsed_args.tutorial_case = parsed_args.tutorial_case.expanduser().resolve()
    parsed_args.work_dir = parsed_args.work_dir.expanduser().resolve()

    if not parsed_args.tutorial_case.is_dir():
        raise FileNotFoundError(f"Tutorial case not found: {parsed_args.tutorial_case}")
    parsed_args.work_dir.mkdir(parents=True, exist_ok=True)

    ensure_openfoam_available()

    case_dir = copy_case(parsed_args.tutorial_case, parsed_args.work_dir)
    case = FoamCase(case_dir)
    with case.control_dict as control_dict:
        control_dict["writeFormat"] = "ascii"
        control_dict["writePrecision"] = 8

    print("$ foamlib: blockMesh")
    case.block_mesh(log=True)
    print("$ foamlib: icoFoam")
    case.run(["icoFoam"], check=True, log=True)

    latest_time = latest_numeric_time(case_dir)
    print(f"Latest serial time: {latest_time}")

    # Access fields through foamlib for additional parser-level validation.
    _ = case[latest_time]["U"].internal_field
    _ = case[latest_time]["p"].internal_field

    validate_read_write(
        case_dir,
        latest_time,
        "U",
        "vector",
        parallel=False,
        output_time="foamToPython_serial",
    )
    validate_read_write(
        case_dir,
        latest_time,
        "p",
        "scalar",
        parallel=False,
        output_time="foamToPython_serial",
    )

    print("$ foamlib: decomposePar")
    case.run(["decomposePar", "-latestTime", "-force"], check=True, log=True)

    validate_read_write(
        case_dir,
        latest_time,
        "U",
        "vector",
        parallel=True,
        output_time="foamToPython_parallel",
    )
    validate_read_write(
        case_dir,
        latest_time,
        "p",
        "scalar",
        parallel=True,
        output_time="foamToPython_parallel",
    )

    print(f"Validated serial and parallel cavity fields in {case_dir}")
    return case_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OpenFOAM cavity tutorial and validate foamToPython serial/parallel field I/O."
    )
    parser.add_argument("--tutorial-case", type=Path, default=DEFAULT_TUTORIAL_CASE)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "foamToPython-cavity-validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
