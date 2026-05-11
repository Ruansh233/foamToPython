import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from foamToPython.readOFField import OFField, find_patches


SCALAR_FIELD = """FoamFile
{
}
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<scalar>
3
(
1
2
3
)
;

boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform 1.5;
    }
    outlet
    {
        type zeroGradient;
    }
}
"""


VECTOR_FIELD = """FoamFile
{
}
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
2
(
(1 0 0)
(0 1 0)
)
;

boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform (1 0 0);
    }
    outlet
    {
        type zeroGradient;
    }
}
"""


def test_of_field_lazy_loading(tmp_path: Path) -> None:
    field_path = tmp_path / "0" / "U"
    field_path.parent.mkdir(parents=True)
    field_path.write_text(VECTOR_FIELD, encoding="utf-8")

    field = OFField(str(field_path), "vector", read_data=False)
    assert field._field_loaded is False

    values = field.internalField
    assert field._field_loaded is True
    assert values.shape == (2, 3)


def test_of_field_serial_read_parses_dimensions_and_boundary(tmp_path: Path) -> None:
    field_path = tmp_path / "0" / "p"
    field_path.parent.mkdir(parents=True)
    field_path.write_text(SCALAR_FIELD, encoding="utf-8")

    field = OFField(str(field_path), "scalar", read_data=True)
    assert np.array_equal(field.dimensions, np.array([0, 1, -1, 0, 0, 0, 0]))
    assert np.array_equal(field.internalField, np.array([1.0, 2.0, 3.0]))
    assert field.boundaryField["inlet"]["type"] == "fixedValue"
    assert field.boundaryField["inlet"]["value"] == 1.5


def test_write_field_accepts_three_arg_form(tmp_path: Path) -> None:
    source = tmp_path / "0" / "U"
    source.parent.mkdir(parents=True)
    source.write_text(VECTOR_FIELD, encoding="utf-8")

    field = OFField(str(source), "vector", read_data=True)
    field.writeField(str(tmp_path / "case"), timeDir=2, fieldName="U", precision=6)

    target = tmp_path / "case" / "2" / "U"
    assert target.exists()
    assert "volVectorField" in target.read_text(encoding="utf-8")


def test_write_field_accepts_single_path_form(tmp_path: Path) -> None:
    source = tmp_path / "0" / "U"
    source.parent.mkdir(parents=True)
    source.write_text(VECTOR_FIELD, encoding="utf-8")

    field = OFField(str(source), "vector", read_data=True)
    target = tmp_path / "2" / "U"
    field.writeField(str(target), precision=6)

    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "internalField   nonuniform List<vector>" in content


def test_parallel_write_forwards_precision(tmp_path: Path) -> None:
    field = OFField(data_type="scalar", parallel=True, read_data=False)
    field._field_loaded = True
    field._dimensions = np.array([0, 1, -1, 0, 0, 0, 0])
    field.internal_field_type = "nonuniform"
    field._internalField = [np.array([1.234567]), np.array([9.876543])]
    field._boundaryField = [
        {"wall": {"type": "fixedValue", "value": np.array([1.0])}},
        {"wall": {"type": "fixedValue", "value": np.array([2.0])}},
    ]

    case_path = tmp_path / "case"
    field.writeField(str(case_path), timeDir=1, fieldName="p", precision=4)

    first_proc = case_path / "processor0" / "1" / "p"
    assert first_proc.exists()
    assert "1.235" in first_proc.read_text(encoding="utf-8")


def test_invalid_data_type_raises_value_error(tmp_path: Path) -> None:
    field_path = tmp_path / "0" / "p"
    field_path.parent.mkdir(parents=True)
    field_path.write_text(SCALAR_FIELD, encoding="utf-8")

    field = OFField(str(field_path), "invalid", read_data=False)
    with pytest.raises(ValueError):
        _ = field.internalField


def test_import_does_not_call_set_start_method(monkeypatch) -> None:
    import multiprocessing

    module_name = "foamToPython.readOFField"
    sys.modules.pop(module_name, None)

    def _fail(*_args, **_kwargs):
        raise AssertionError("set_start_method should not be called at import time")

    monkeypatch.setattr(multiprocessing, "set_start_method", _fail)
    importlib.import_module(module_name)


def test_find_patches_requires_boundary_field_content() -> None:
    with pytest.raises(ValueError, match="boundaryField"):
        list(find_patches(SCALAR_FIELD.splitlines()))


def test_find_patches_extracts_patches_from_boundary_field_content() -> None:
    lines = SCALAR_FIELD.splitlines()
    boundary_start = next(i for i, line in enumerate(lines) if "boundaryField" in line)
    boundary_lines = lines[boundary_start:]

    patches = list(find_patches(boundary_lines))

    assert patches == [
        ["inlet", "type fixedValue;", "value uniform 1.5;"],
        ["outlet", "type zeroGradient;"],
    ]
