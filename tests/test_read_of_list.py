from pathlib import Path

import numpy as np
import pytest

from foamToPython.readOFList import readList, readListList


def test_read_list_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        readList(str(missing), "scalar")


def test_read_list_invalid_data_type_raises_value_error(tmp_path: Path) -> None:
    data_file = tmp_path / "values"
    data_file.write_text("3\n(\n1\n2\n3\n)\n", encoding="utf-8")
    with pytest.raises(ValueError):
        readList(str(data_file), "bad_type")


def test_read_list_uniform_scalar_full_length_and_single_value(tmp_path: Path) -> None:
    data_file = tmp_path / "uniform"
    data_file.write_text("4{2.5}", encoding="utf-8")

    expanded = readList(str(data_file), "scalar", fullLength=True)
    single = readList(str(data_file), "scalar", fullLength=False)

    assert np.array_equal(expanded, np.array([2.5, 2.5, 2.5, 2.5]))
    assert np.array_equal(single, np.array(2.5))


def test_read_list_list_vector(tmp_path: Path) -> None:
    data_file = tmp_path / "list_list"
    data_file.write_text(
        "2\n(\n2\n(\n(1 0 0)\n(0 1 0)\n)\n1\n(\n(0 0 1)\n)\n)\n",
        encoding="utf-8",
    )

    data = readListList(str(data_file), "vector")
    assert data.shape == (2,)
    assert np.array_equal(data[0], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert np.array_equal(data[1], np.array([[0.0, 0.0, 1.0]]))


def test_read_list_nonuniform_label(tmp_path: Path) -> None:
    data_file = tmp_path / "labels"
    data_file.write_text("4\n(\n1\n2\n10\n20\n)\n", encoding="utf-8")
    data = readList(str(data_file), "label")
    assert np.array_equal(data, np.array([1, 2, 10, 20]))
