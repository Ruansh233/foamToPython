import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._field_parser import find_patches, read_field_file, validate_data_type
from ._field_writer import (
    normalize_write_target,
    write_field_parallel,
    write_field_serial,
)
from ._mp_utils import get_spawn_pool


class OFField:
    filename: str
    fieldName: str
    timeName: str
    data_type: str
    read_data: bool
    parallel: bool
    reconstructPar: bool
    caseDir: str
    num_batch: int
    _field_loaded: bool
    _dimensions: np.ndarray
    _internalField: Union[float, np.ndarray, List[np.ndarray]]
    internal_field_type: Optional[str]
    _boundaryField: Union[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]
    _num_processors: int

    def __init__(
        self,
        filename: str = None,
        data_type: str = None,
        read_data: bool = False,
        parallel: bool = False,
        reconstructPar: bool = False,
        num_batch: int = 8,
    ) -> None:
        if filename is not None:
            self.filename = filename
            self.caseDir = "/".join(filename.split("/")[:-2])
            self.fieldName = filename.split("/")[-1]
            self.timeName = filename.split("/")[-2]
        else:
            self.filename = ""
            self.caseDir = ""
            self.fieldName = ""
            self.timeName = ""

        self.parallel = parallel
        self.reconstructPar = reconstructPar
        self.num_batch = num_batch
        self._num_processors = 1

        if not self.parallel and self.reconstructPar:
            raise ValueError("reconstructPar can only be True if parallel is True.")

        self.data_type = data_type
        self.read_data = read_data
        self.internal_field_type = None
        self._dimensions = np.array([])
        self._internalField = np.array([])
        self._boundaryField = {}
        self._field_loaded = False

        if self.read_data:
            (
                self._dimensions,
                self._internalField,
                self._boundaryField,
                self.internal_field_type,
            ) = self.readField()
            self._field_loaded = True

    @classmethod
    def from_OFField(cls, other: "OFField") -> "OFField":
        new_field = cls()
        new_field.filename = other.filename
        new_field.caseDir = other.caseDir
        new_field.fieldName = other.fieldName
        new_field.timeName = other.timeName
        new_field.data_type = other.data_type
        new_field.read_data = other.read_data
        new_field.parallel = other.parallel
        new_field.reconstructPar = other.reconstructPar
        new_field.num_batch = other.num_batch
        new_field.internal_field_type = other.internal_field_type
        new_field._num_processors = other._num_processors

        new_field._dimensions = copy.deepcopy(other._dimensions)
        new_field._internalField = copy.deepcopy(other._internalField)
        new_field._boundaryField = copy.deepcopy(other._boundaryField)
        new_field._field_loaded = other._field_loaded
        return new_field

    @property
    def dimensions(self):
        if not self._field_loaded:
            self._load_field()
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = value

    @property
    def internalField(self):
        if not self._field_loaded:
            self._load_field()
        return self._internalField

    @internalField.setter
    def internalField(self, value):
        self._internalField = value

    @property
    def boundaryField(self):
        if not self._field_loaded:
            self._load_field()
        return self._boundaryField

    @boundaryField.setter
    def boundaryField(self, value):
        self._boundaryField = value

    def _load_field(self) -> None:
        (
            self._dimensions,
            self._internalField,
            self._boundaryField,
            self.internal_field_type,
        ) = self.readField()
        self._field_loaded = True

    def readField(self):
        if self.data_type is None:
            raise ValueError("data_type must be specified before reading a field.")
        validate_data_type(self.data_type)
        if self.parallel:
            return self._readField_parallel()
        return self._readField(self.filename, self.data_type)

    @staticmethod
    def _readField(
        filename: str, data_type: str, parallel: bool = False
    ) -> Tuple[np.ndarray, Union[float, np.ndarray], Dict[str, Dict[str, Any]], str]:
        return read_field_file(filename, data_type, parallel)

    def _readField_parallel(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str, Dict[str, Any]]], str]:
        case_dir = self.caseDir
        processor_dirs = sorted(
            [d for d in os.listdir(case_dir) if d.startswith("processor")],
            key=lambda x: int(x.replace("processor", "")),
        )
        if not processor_dirs:
            raise FileNotFoundError("No processor directories found.")

        proc_paths = [
            os.path.join(case_dir, proc_dir, self.timeName, self.fieldName)
            for proc_dir in processor_dirs
        ]
        for proc_path in proc_paths:
            if not os.path.isfile(proc_path):
                raise FileNotFoundError(f"Field file not found in {proc_path}")

        with get_spawn_pool(processes=self.num_batch) as pool:
            results = pool.starmap(
                read_field_file,
                [(proc_path, self.data_type, True) for proc_path in proc_paths],
            )

        dimensions = results[0][0]
        internal_field: List[np.ndarray] = []
        boundary_field: List[Dict[str, Dict[str, Any]]] = []
        internal_field_types: List[str] = []

        for dim, internal, boundary, field_type in results:
            if not np.array_equal(dim, dimensions):
                raise ValueError("Inconsistent field dimensions across processors.")
            internal_field.append(internal)
            boundary_field.append(boundary)
            internal_field_types.append(field_type)

        if all("nonuniform" in ft for ft in internal_field_types):
            merged_internal_field_type = "nonuniform"
        else:
            merged_internal_field_type = "uniform"

        self._num_processors = len(results)
        return dimensions, internal_field, boundary_field, merged_internal_field_type

    def writeField(
        self,
        casePath: str,
        timeDir: Union[int, str, None] = None,
        fieldName: Optional[str] = None,
        precision: int = 10,
    ) -> None:
        if self.data_type is None:
            raise ValueError("data_type must be specified before writing a field.")
        validate_data_type(self.data_type)
        if not self._field_loaded:
            self._load_field()

        case_path, time_dir, field_name = normalize_write_target(
            casePath, timeDir, fieldName
        )

        if self.parallel:
            self._writeField_parallel(
                case_path,
                timeDir=time_dir,
                fieldName=field_name,
                precision=precision,
            )
        else:
            self._writeField_serial(
                case_path,
                internalField=self.internalField,
                boundaryField=self.boundaryField,
                timeDir=time_dir,
                fieldName=field_name,
                precision=precision,
            )

    def _writeField_serial(
        self,
        casePath: str,
        internalField: Union[float, np.ndarray],
        boundaryField: Dict[str, Dict[str, Any]],
        timeDir: Union[int, str],
        fieldName: str,
        precision: int = 10,
    ) -> None:
        if self.internal_field_type not in {"uniform", "nonuniform"}:
            raise ValueError("internal_field_type should be 'uniform' or 'nonuniform'")

        write_field_serial(
            casePath,
            internalField,
            boundaryField,
            timeDir,
            fieldName,
            self.data_type,
            self._dimensions,
            self.internal_field_type,
            precision,
        )

    def _writeField_parallel(
        self,
        casePath: str,
        timeDir: Union[int, str],
        fieldName: str,
        precision: int = 10,
    ) -> None:
        write_field_parallel(
            casePath,
            self._internalField,
            self._boundaryField,
            timeDir,
            fieldName,
            self.data_type,
            self._dimensions,
            self.internal_field_type,
            precision,
            self.num_batch,
        )


__all__ = ["OFField", "find_patches"]
