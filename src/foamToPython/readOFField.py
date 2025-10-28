import numpy as np
import sys
import os
import re
import mmap
import multiprocessing
import copy
from typing import List, Dict, Any, Optional, Tuple, Union
from .headerEnd import *


class OFField:
    """
    A class for reading and writing OpenFOAM field files supporting both scalar and vector fields.

    This class provides functionality to read OpenFOAM field files in both serial and parallel
    formats, parse internal and boundary field data, and write field data back to OpenFOAM format.

    For serial fields, the internal field is a ndarray (for nonuniform) or a float (for uniform scalar),
    and boundary fields are stored in a dictionary.
    For parallel fields, the internal field is a list of ndarrays (one per processor), and boundary fields
    are stored in a list of dictionaries (one per processor).

    Attributes
    ----------
    filename : str
        Path to the OpenFOAM field file.
    fieldName : str
        Name of the field.
    timeName : str
        Time directory name.
    data_type : str, optional
        Type of field ('scalar' or 'vector').
    read_data : bool
        Whether to read field data upon initialization.
    parallel : bool
        Whether the field uses parallel processing.
    internal_field_type : str, optional
        Type of internal field ('uniform', 'nonuniform', or 'nonuniformZero').
    dimensions : np.ndarray
        Physical dimensions of the field [kg m s K mol A cd].
    internalField : Union[float, np.ndarray]
        Internal field data.
    boundaryField : Dict[str, Dict[str, Any]]
        Boundary field data organized by patch names.

    Examples
    --------
    Reading a scalar field file:

    >>> field = OFField('case/0/p', data_type='scalar', read_data=True)
    >>> pressure_values = field.internalField

    Reading a vector field file:

    >>> velocity = OFField('case/0/U', data_type='vector', read_data=True)
    >>> u_components = velocity.internalField  # Shape: (n_cells, 3)
    """

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
    internal_field_type: str
    _boundaryField: Dict[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]

    def __init__(
        self,
        filename: str = None,
        data_type: str = None,
        read_data: bool = False,
        parallel: bool = False,
        reconstructPar: bool = False,
        num_batch: int = 8,
    ) -> None:
        """
        Initialize OFField object.

        Parameters
        ----------
        filename : str, optional
            Path to the OpenFOAM field file, by default None.
        data_type : str, optional
            Type of field ('scalar' or 'vector'), by default None.
        read_data : bool, optional
            If True, read the field file upon initialization, by default False.
        parallel : bool, optional
            If True, enable parallel processing for multi-processor cases, by default False.
        reconstructPar : bool, optional
            If True, reconstruct the parallel field into a single field, by default False.
        num_batch : int, optional
            Number of processors to use for parallel reading, by default 8.

        Notes
        -----
        If filename is provided, the object will automatically extract caseDir, fieldName,
        and timeName from the path. If read_data is True, the field data will be loaded
        immediately upon initialization.
        """
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

    # a initial constructor copy from another OFField object
    @classmethod
    def from_OFField(cls, other: "OFField") -> "OFField":
        """
        Create a new OFField instance by copying another OFField object.

        Parameters
        ----------
        other : OFField
            The source OFField object to copy from.

        Returns
        -------
        OFField
            A new OFField instance with copied attributes from the source object.

        Notes
        -----
        This method performs a deep copy of arrays and dictionaries to ensure
        the new instance is independent of the original object.
        """
        new_field = cls()

        new_field.filename = other.filename
        new_field.caseDir = other.caseDir
        new_field.fieldName = other.fieldName
        new_field.timeName = other.timeName

        # simple attributes
        new_field.data_type = other.data_type
        new_field.read_data = other.read_data
        new_field.parallel = other.parallel
        new_field.reconstructPar = other.reconstructPar
        new_field.num_batch = other.num_batch

        new_field._dimensions = other._dimensions.copy()

        # internalField: handle ndarray, list-of-ndarrays (parallel case)
        if isinstance(other._internalField, list):
            new_field._internalField = [
                arr.copy() if isinstance(arr, np.ndarray) else copy.deepcopy(arr)
                for arr in other._internalField
            ]
        elif isinstance(other._internalField, np.ndarray):
            new_field._internalField = other._internalField.copy()
        else:
            raise ValueError(
                "Unsupported type for internalField. It should be ndarray or list of ndarrays."
            )

        # boundaryField: handle Dict[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]] (parallel case)
        if isinstance(other._boundaryField, list):
            new_field._boundaryField = [
                copy.deepcopy(bc) for bc in other._boundaryField
            ]
        elif isinstance(other._boundaryField, dict):
            new_field._boundaryField = copy.deepcopy(other._boundaryField)
        else:
            raise ValueError(
                "Unsupported type for boundaryField. It should be dict or list of dicts."
            )

        new_field._field_loaded = other._field_loaded
        return new_field

    @property
    def dimensions(self):
        if not self._field_loaded:
            (
                self._dimensions,
                self._internalField,
                self._boundaryField,
                self.internal_field_type,
            ) = self.readField()
            self._field_loaded = True
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = value

    @property
    def internalField(self):
        """
        Get the internal field data.

        Returns
        -------
        Union[float, np.ndarray, List[np.ndarray]]
            For serial fields, this returns a float (for uniform scalar), a single array (for nonuniform), or value.
            For parallel fields, this returns a list of arrays.
        """
        if not self._field_loaded:
            (
                self._dimensions,
                self._internalField,
                self._boundaryField,
                self.internal_field_type,
            ) = self.readField()
            self._field_loaded = True
        return self._internalField

    @internalField.setter
    def internalField(self, value):
        self._internalField = value

    @property
    def boundaryField(self):
        """
        Get the boundary field data.

        Returns
        -------
        Union[Dict, List[Dict]]
            For serial fields, this returns a dictionary of boundary field properties.
            For parallel fields, this returns a list of dictionaries.
        """
        if not self._field_loaded:
            (
                self._dimensions,
                self._internalField,
                self._boundaryField,
                self.internal_field_type,
            ) = self.readField()
            self._field_loaded = True
        return self._boundaryField

    @boundaryField.setter
    def boundaryField(self, value):
        self._boundaryField = value

    def readField(self):
        if self.parallel:
            return self._readField_parallel()
        else:
            return self._readField(self.filename, self.data_type)

    @staticmethod
    def _readField(filename: str, data_type: str, parallel: bool = False):
        """
        Read the field file and parse internal and boundary fields.

        Parameters
        ----------
        filename : str
            Path to the OpenFOAM field file.
        data_type : str
            Type of field ('scalar' or 'vector').
        parallel : bool, optional
            If True, indicates parallel processing context, by default False.

        Returns
        -------
        tuple
            A tuple containing:
            - _dimensions : np.ndarray
                Physical dimensions of the field.
            - _internalField : Union[float, np.ndarray]
                Internal field data.
            - _boundaryField : Dict[str, Dict[str, Any]]
                Boundary field data organized by patch names.
            - internal_field_type : str
                Type of internal field ('uniform', 'nonuniform', or 'nonuniformZero').

        Raises
        ------
        ValueError
            If internal field type is invalid or file format is incorrect.
        SystemExit
            If unknown data_type is encountered.
        """
        with open(f"{filename}", "rb") as f:
            # For very large files, use memory mapping
            file_size = os.path.getsize(filename)
            if (
                file_size > 50 * 1024 * 1024
            ):  # 50MB threshold (lower for multiprocessing)
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    content = mmapped_file.read().splitlines()
            else:
                content = f.readlines()

            data_idx, boundary_start_idx, dim_idx, data_size, internal_field_type = (
                OFField._num_field(content)
            )

            _dimensions = _process_dimensions(content[dim_idx].decode("utf-8"))

            if internal_field_type == "uniform":
                _internalField = OFField._process_uniform(
                    content[data_idx].decode("utf-8"), data_type
                )
            elif internal_field_type == "nonuniform":
                data_start_idx = data_idx + 2
                # Extract relevant lines containing coordinates
                _internalField = OFField._process_field(
                    content[data_start_idx : data_start_idx + data_size],
                    data_size,
                    data_type,
                )
            elif internal_field_type == "nonuniformZero":
                if data_type == "scalar":
                    _internalField = np.array([])
                elif data_type == "vector":
                    _internalField = np.empty((0, 3))
                else:
                    sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")
            else:
                raise ValueError(
                    "internal_field_type should be 'uniform' or 'nonuniform'"
                )

            _boundaryField = OFField._process_boundary(
                content[boundary_start_idx:], data_type, parallel
            )

        return _dimensions, _internalField, _boundaryField, internal_field_type

    def _readField_parallel(self):
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

        with multiprocessing.Pool(processes=self.num_batch) as pool:
            # Use the optimized reading function
            results = pool.starmap(
                self._readField,
                [(proc_path, self.data_type, True) for proc_path in proc_paths],
            )

        # Unpack results
        _dimensions = results[0][0]
        _internalField = []
        _boundaryField = []
        internal_field_types = []

        for dim, internal, boundary, field_type in results:  # Added _ for filepath
            if not np.array_equal(dim, _dimensions):
                raise ValueError("Inconsistent field dimensions across processors.")
            _internalField.append(internal)
            _boundaryField.append(boundary)
            internal_field_types.append(field_type)

        if all("nonuniform" in ft for ft in internal_field_types):
            self.internal_field_type = "nonuniform"
        else:
            self.internal_field_type = "uniform"
        self._num_processors = len(results)

        return _dimensions, _internalField, _boundaryField, self.internal_field_type

    # def _reconstruct_fields_optimized(self, _internalField, _boundaryField, internal_field_types):
    #     """
    #     Efficiently reconstruct parallel fields as a single field for large datasets.
    #     """
    #     if self.internal_field_type == "uniform":
    #         reconstructed_internal = _internalField[0]
    #     elif self.internal_field_type == "nonuniform":
    #         if self.data_type == "scalar":
    #             # Use numpy's concatenate for efficiency
    #             reconstructed_internal = np.concatenate(_internalField)
    #         elif self.data_type == "vector":
    #             # Use vstack for vectors
    #             reconstructed_internal = np.vstack(_internalField)
    #         else:
    #             sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")
    #     else:
    #         raise ValueError(
    #             "internal_field_type should be 'uniform' or 'nonuniform'"
    #         )

    #     # Merge boundary fields efficiently
    #     merged_boundary = {}
    #     for boundary in _boundaryField:
    #         for patch, props in boundary.items():
    #             if "procBoundary" in patch:
    #                 continue  # Skip processor boundary patches
    #             if patch not in merged_boundary:
    #                 merged_boundary[patch] = props.copy()
    #             else:
    #                 # For overlapping patches, merge arrays if they exist
    #                 for key, value in props.items():
    #                     if key in merged_boundary[patch]:
    #                         if isinstance(value, np.ndarray) and isinstance(merged_boundary[patch][key], np.ndarray):
    #                             # Concatenate arrays
    #                             if value.ndim == 1 and merged_boundary[patch][key].ndim == 1:
    #                                 merged_boundary[patch][key] = np.concatenate([merged_boundary[patch][key], value])
    #                             elif value.ndim == 2 and merged_boundary[patch][key].ndim == 2:
    #                                 merged_boundary[patch][key] = np.vstack([merged_boundary[patch][key], value])

    #     return reconstructed_internal, merged_boundary

    @staticmethod
    def _process_uniform(line: str, data_type: str):
        """
        Process uniform internal field value.

        Parameters
        ----------
        line : str
            Line containing the uniform value.
        data_type : str
            Type of field ('scalar' or 'vector').

        Returns
        -------
        Union[float, np.ndarray]
            For scalar fields: float value.
            For vector fields: numpy array with shape (3,).

        Raises
        ------
        ValueError
            If uniform field format is invalid for the specified data_type.
        """
        if data_type == "scalar":
            # Extract the scalar value after 'uniform'
            match = re.search(r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\d+)", line)
            if match:
                _internalField = float(match.group(1))
            else:
                raise ValueError("Invalid uniform scalar format")

        elif data_type == "vector":
            # Extract the vector value after 'uniform', the element may be integer or float
            # for example: uniform (0 0 1.0) or uniform (1 0 0)
            match = re.search(
                r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                line,
            )
            if match:
                vec_str = match.group(1)
                _internalField = np.array([float(x) for x in vec_str.split()])
            else:
                raise ValueError("Invalid uniform vector format")

        return _internalField

    @staticmethod
    def _process_field(string_coords: List[bytes], data_size: int, data_type: str):
        """
        Process nonuniform internal field values.

        Parameters
        ----------
        string_coords : List[bytes]
            List of byte strings containing field values.
        data_size : int
            Number of data points expected.
        data_type : str
            Type of field ('scalar' or 'vector').

        Returns
        -------
        np.ndarray
            For scalar fields: 1D array with shape (data_size,).
            For vector fields: 2D array with shape (data_size, 3).

        Raises
        ------
        SystemExit
            If unknown data_type is encountered.
        """
        if data_type == "scalar":
            # Join all lines and replace unwanted characters once
            joined_coords = b" ".join(string_coords).replace(b"\n", b"")
            # Convert to a numpy array in one go
            _internalField = np.fromstring(
                joined_coords.decode("utf-8"), sep=" ", dtype=np.float64
            )

            if len(_internalField) != data_size:
                raise ValueError(
                    f"Expected {data_size} data points, but got {len(_internalField)}."
                )

        elif data_type == "vector":
            # Join all lines and replace unwanted characters once
            joined_coords = (
                b" ".join(string_coords)
                .replace(b")", b"")
                .replace(b"(", b"")
                .replace(b"\n", b"")
            )
            # Convert to a numpy array
            arr = np.fromstring(
                joined_coords.decode("utf-8"), sep=" ", dtype=np.float64
            )
            try:
                _internalField = arr.reshape(data_size, 3)
            except ValueError:
                raise ValueError(
                    f"Cannot reshape internal field of length {arr.size} to shape ({data_size}, 3)."
                )

        else:
            sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")

        return _internalField

    @staticmethod
    def _process_boundary(
        lines: List[Union[str, bytes]], data_type: str, parallel: bool
    ):
        """
        Process boundaryField section and extract patch properties.

        Parameters
        ----------
        lines : List[Union[str, bytes]]
            List of lines (bytes or str) for boundaryField section.
        data_type : str
            Type of field ('scalar' or 'vector').
        parallel : bool
            If True, indicates parallel processing context.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing boundary field properties organized by patch names.
            Each patch contains properties like 'type', 'value', etc.

        Raises
        ------
        ValueError
            If file format is incorrect or boundary field parsing fails.
        """
        # decode bytes to string if necessary
        if isinstance(lines[0], bytes):
            lines = [line.decode("utf-8") for line in lines]

        bc_dict = {}
        i = 0
        n = len(lines)

        def skip_empty_and_comments(idx):
            while idx < n:
                line = lines[idx].strip()
                if line == "" or line.startswith("//"):
                    idx += 1
                else:
                    break
            return idx

        i = skip_empty_and_comments(i)

        # Expect "boundaryField {"
        if not lines[i].strip().startswith("boundaryField"):
            raise ValueError("File does not start with boundaryField")
        i += 1
        i = skip_empty_and_comments(i)
        if lines[i].strip() != "{":
            raise ValueError("Expected '{' after boundaryField")
        i += 1

        # Parse patches
        while i < n:
            i = skip_empty_and_comments(i)
            line = lines[i].strip()
            if line == "}":  # end of boundaryField
                break

            patch_name = line
            i += 1
            i = skip_empty_and_comments(i)
            if lines[i].strip() != "{":
                raise ValueError(f"Expected '{{' after {patch_name}")
            i += 1

            # Parse patch properties and save all lines in a list, i.e., prop_lines
            props = {}
            brace_count = 1
            prop_lines = []
            while i < n and brace_count > 0:
                l = lines[i].strip()
                if "}" in l:
                    brace_count -= l.count("}")
                prop_lines.append(l)
                i += 1

            # Remove last closing brace
            prop_lines = prop_lines[:-1]

            # Combine multi-line values into single strings
            key = None
            value_lines = []
            for l in prop_lines:
                if ";" in l:
                    parts = l.split(None, 1)
                    # Handle single-line key-value pairs (e.g., type fixedValue;)
                    if len(parts) == 2:
                        key, value = parts
                        value_lines.append(value)
                    if key:
                        value_str = " ".join(value_lines).replace(";", "").strip()
                        # convert value string to np.array if it contains numeric data
                        if value_str.startswith("uniform"):
                            if data_type == "scalar":
                                # read scalar
                                scalar_match = re.match(
                                    r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                                    value_str,
                                )
                                if scalar_match:
                                    props[key] = float(scalar_match.group(1))
                                else:
                                    raise ValueError(
                                        f"Invalid scalar format: {value_str}"
                                    )
                            elif data_type == "vector":
                                # read vector
                                vec_match = re.match(
                                    r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                                    value_str,
                                )
                                if vec_match:
                                    props[key] = _parse_vector_string(
                                        vec_match.group(1)
                                    )
                                else:
                                    raise ValueError(
                                        f"Invalid vector format: {value_str}"
                                    )
                        elif value_str.startswith("nonuniform List<vector>"):
                            # read scalar list
                            if data_type == "scalar":
                                value_str = value_str.split("(", 1)[1].rsplit(")", 1)[0]
                                scalar_match = re.findall(
                                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                    value_str,
                                )
                                if scalar_match:
                                    props[key] = np.array(
                                        [float(x) for x in scalar_match]
                                    )
                                # read "value           nonuniform List<vector> 0();" for parallel case
                                elif parallel:
                                    props[key] = value_str
                                else:
                                    raise ValueError(
                                        f"Invalid scalar list format: {value_str}"
                                    )
                            elif data_type == "vector":
                                # read vector list
                                vecs = re.findall(
                                    r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                                    value_str,
                                )
                                if vecs:
                                    props[key] = np.array(
                                        [[float(x) for x in v.split()] for v in vecs]
                                    )
                                # read "value           nonuniform List<vector> 0();" for parallel case
                                elif parallel:
                                    props[key] = value_str
                                else:
                                    raise ValueError(
                                        f"Invalid vector list format: {value_str}"
                                    )
                        else:
                            props[key] = value_str
                        key = None
                        value_lines = []
                else:
                    # Handle multi-line values. This line does not end with ;
                    if key is None:
                        key = l.split()[0]
                        value_lines = l.split()[1:]
                    else:
                        value_lines.append(l)

            bc_dict[patch_name] = props

        return bc_dict

    @staticmethod
    def _num_field(
        subcontent: List[bytes],
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Find indices for dimensions, internalField, and boundaryField sections.

        Parameters
        ----------
        subcontent : List[bytes]
            List of file lines as bytes.

        Returns
        -------
        Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[str]]
            A tuple containing:
            - data_idx : int or None
                Index of the internalField data.
            - boundary_idx : int or None
                Index where boundaryField section starts.
            - dim_idx : int or None
                Index of the dimensions line.
            - data_size : int or None
                Number of data points in nonuniform fields.
            - internal_field_type : str or None
                Type of internal field ('uniform', 'nonuniform', or 'nonuniformZero').

        Raises
        ------
        ValueError
            If internalField is not found in the file.
        """
        dim_idx = None
        data_size = None
        data_idx = None
        boundary_idx = None
        idx = 0
        internal_field_type = None

        while idx < len(subcontent):
            if b"dimensions" in subcontent[idx]:
                dim_idx = idx
            if b"internalField" in subcontent[idx]:
                if b"nonuniform" in subcontent[idx]:
                    if b"0()" in subcontent[idx]:
                        data_idx = idx
                        internal_field_type = "nonuniformZero"
                    else:
                        internal_field_type = "nonuniform"
                else:
                    internal_field_type = "uniform"
                    data_idx = idx
            if data_size is None and internal_field_type == "nonuniform":
                try:
                    data_size = int(subcontent[idx])
                    data_idx = idx
                    idx = data_idx + data_size + 1
                except ValueError:
                    pass
            if b"boundaryField" in subcontent[idx]:
                boundary_idx = idx
                break
            idx += 1

        if internal_field_type is None:
            raise ValueError("internalField not found in the file.")
        return data_idx, boundary_idx, dim_idx, data_size, internal_field_type

    def writeField(
        self,
        fieldDir: str,
        timeDir: Optional[str] = None,
        fieldName: Optional[str] = None,
    ) -> None:
        """
        Write field data to a file in OpenFOAM format.

        Parameters
        ----------
        fieldDir : str
            Path to output file or directory.
        timeDir : str, optional
            Time directory name, by default None. If None, extracted from fieldDir.
        fieldName : str, optional
            Field name, by default None. If None, extracted from fieldDir.

        Returns
        -------
        None

        Notes
        -----
        Automatically handles both serial and parallel field writing based on the
        parallel attribute of the object.
        """
        if self.parallel:
            self._writeField_parallel(fieldDir, timeDir=timeDir, fieldName=fieldName)
        else:
            self._writeField_serial(
                fieldDir,
                internalField=self.internalField,
                boundaryField=self.boundaryField,
                timeDir=timeDir,
                fieldName=fieldName,
            )

    def _writeField_serial(
        self,
        fieldDir: str,
        internalField: Union[float, np.ndarray],
        boundaryField: Dict[str, Dict[str, Any]],
        timeDir: Optional[int] = None,
        fieldName: Optional[str] = None,
    ) -> None:
        """
        Write field data to a file in OpenFOAM format (serial version).

        Parameters
        ----------
        fieldDir : str
            Path to output file.
        internalField : Union[float, np.ndarray]
            Internal field data to write.
        boundaryField : Dict[str, Dict[str, Any]]
            Boundary field data organized by patch names.
        timeDir : str, optional
            Time directory name, by default None.
        fieldName : str, optional
            Field name, by default None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If internal_field_type is invalid.
        SystemExit
            If fieldDir format is incorrect.
        """
        _timeDir = timeDir if timeDir is not None else fieldDir.split("/")[-2]
        _fieldName = fieldName if fieldName is not None else fieldDir.split("/")[-1]

        _fieldDir = (
            "/".join(fieldDir.split("/")[:-2]) + f"/{_timeDir}" + f"/{_fieldName}"
        )

        try:
            int(_timeDir)
        except ValueError:
            sys.exit(
                "The fieldDir should be like '.../0/U' or '.../100/p'. "
                "You can provide <timeDir> and <fieldName> to use other formats."
            )

        with open(_fieldDir, "w") as f:
            # write header
            thisHeader = header.replace(
                "className;", f"vol{self.data_type.capitalize()}Field;"
            )
            thisHeader = thisHeader.replace("timeDir;", f"{_timeDir};")
            thisHeader = thisHeader.replace(
                "object      data;", f"object      {_fieldName};"
            )
            f.write(thisHeader + "\n\n")

            # write dimensions as "dimensions      [0 1 -1 0 0 0 0];"
            f.write(
                f"dimensions      [{ ' '.join(str(d) for d in self._dimensions) }];\n\n"
            )

            # write internalField for scalar or vector
            if self.data_type == "scalar":
                if self.internal_field_type == "uniform":
                    f.write(f"internalField   uniform {internalField:.8g};\n\n")
                elif self.internal_field_type == "nonuniform":
                    f.write(f"internalField   nonuniform List<scalar>\n")
                    f.write(f"{internalField.shape[0]}\n")
                    f.write("(\n")
                    for point in internalField:
                        f.write(f"{point:.8g}\n")
                    f.write(")\n;\n")
                else:
                    raise ValueError(
                        "internal_field_type should be 'uniform' or 'nonuniform'"
                    )
            elif self.data_type == "vector":
                if self.internal_field_type == "uniform":
                    f.write(
                        f"internalField   uniform ({internalField[0]:.8g} {internalField[1]:.8g} {internalField[2]:.8g});\n\n"
                    )
                elif self.internal_field_type == "nonuniform":
                    f.write(f"internalField   nonuniform List<vector>\n")
                    f.write(f"{internalField.shape[0]}\n")
                    f.write("(\n")
                    for point in internalField:
                        f.write(f"({point[0]:.8g} {point[1]:.8g} {point[2]:.8g})\n")
                    f.write(")\n;\n")
                else:
                    raise ValueError(
                        "internal_field_type should be 'uniform' or 'nonuniform'"
                    )

            # write boundaryField
            f.write("boundaryField\n")
            f.write("{\n")
            for patch, props in boundaryField.items():
                f.write(f"    {patch}\n")
                f.write("    {\n")
                for key, value in props.items():
                    if isinstance(value, np.ndarray):
                        if value.ndim == 0:
                            # scalar
                            f.write(f"        {key} uniform {value:.8g};\n")
                        elif value.ndim == 1 and value.shape[0] == 3:
                            # vector
                            f.write(
                                f"        {key} uniform ({value[0]:.8g} {value[1]:.8g} {value[2]:.8g});\n"
                            )
                        elif value.ndim == 1:
                            # scalar list
                            f.write(f"        {key} nonuniform List<scalar>\n")
                            f.write(f"{value.shape[0]}\n")
                            f.write("(\n")
                            for v in value:
                                f.write(f"{v:.8g}\n")
                            f.write(");\n")
                        elif (
                            value.ndim == 2
                            and value.shape[0] == 1
                            and value.shape[1] != 3
                        ):
                            # a scalar list in 2D array with shape (1, N)
                            f.write(f"        {key} nonuniform List<scalar>\n")
                            f.write(f"{value.shape[1]}\n")
                            f.write("(\n")
                            for v in value.T:
                                f.write(f"{v:.8g}\n")
                            f.write(");\n")
                        elif (
                            value.ndim == 2
                            and value.shape[0] == 1
                            and value.shape[1] == 3
                        ):
                            # a single vector in 2D array
                            f.write(
                                f"        {key} uniform ({value[0,0]:.8g} {value[0,1]:.8g} {value[0,2]:.8g});\n"
                            )
                        elif value.ndim == 2 and value.shape[1] == 3:
                            # vector list
                            f.write(f"{key} nonuniform List<vector>\n")
                            f.write(f"{value.shape[0]}\n")
                            f.write("(\n")
                            for v in value:
                                f.write(f"({v[0]:.8g} {v[1]:.8g} {v[2]:.8g})\n")
                            f.write(");\n")
                    else:
                        # assume it's a string or other simple type
                        f.write(f"        {key} {value};\n")
                f.write("    }\n")
            f.write("}\n\n")

            # write ender
            f.write(ender)

    def _writeField_parallel(
        self,
        fieldDir: str,
        timeDir: Optional[str] = None,
        fieldName: Optional[str] = None,
    ) -> None:
        """
        Write field data to processor directories in OpenFOAM format.

        Parameters
        ----------
        fieldDir : str
            Path to the field directory (e.g., '.../case/1/U').
        timeDir : str, optional
            Time directory name, by default None.
        fieldName : str, optional
            Field name, by default None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If internalField and boundaryField are not lists for parallel writing.
        SystemExit
            If fieldDir format is incorrect.
        """
        _timeDir = timeDir if timeDir is not None else fieldDir.split("/")[-2]
        _fieldName = fieldName if fieldName is not None else fieldDir.split("/")[-1]

        _fieldDir = (
            "/".join(fieldDir.split("/")[:-2])
            + "/processor"
            + f"/{_timeDir}"
            + f"/{_fieldName}"
        )

        try:
            int(_timeDir)
        except ValueError:
            sys.exit(
                "The fieldDir should be like '.../0/U' or '.../100/p'. "
                "You can provide <timeDir> and <fieldName> to use other formats."
            )

        if not isinstance(self._internalField, list) or not isinstance(
            self._boundaryField, list
        ):
            raise ValueError(
                "For parallel writing, internalField and boundaryField should be lists."
            )

        num_processors = len(self._internalField)

        proc_field_path = [
            _fieldDir.replace("processor", f"processor{idx}")
            for idx in range(num_processors)
        ]

        with multiprocessing.Pool(processes=self.num_batch) as pool:
            list(
                pool.imap(
                    self._writeField_wrapper,
                    [
                        (
                            proc_path,
                            self._internalField[idx],
                            self._boundaryField[idx],
                            _timeDir,
                            _fieldName,
                        )
                        for idx, proc_path in enumerate(proc_field_path)
                    ],
                )
            )

    def _writeField_wrapper(self, args):
        # args: (fieldDir, internalField, boundaryField, timeDir, fieldName)
        return self._writeField_serial(*args)


def _parse_vector_string(s: str) -> np.ndarray:
    """
    Parse a single vector string like '(0 0 1.0)' into a NumPy array.

    Parameters
    ----------
    s : str
        Vector string with format '(x y z)' or 'x y z'.

    Returns
    -------
    np.ndarray
        1D array with shape (3,) containing the parsed vector components.

    Examples
    --------
    >>> _parse_vector_string("0 0 1.0")
    array([0., 0., 1.])
    >>> _parse_vector_string("(1.5 -2.0 3.14)")
    array([ 1.5, -2.0,  3.14])
    """
    s = s.strip("()")
    return np.array([float(x) for x in s.split()])


def _process_dimensions(line: str) -> np.ndarray:
    """
    Parse dimensions line from OpenFOAM file.

    Parameters
    ----------
    line : str
        Line containing dimensions in format '[kg m s K mol A cd]'.

    Returns
    -------
    np.ndarray
        Array of 7 integers representing physical dimensions in SI base units:
        [mass, length, time, temperature, amount, current, luminous_intensity].

    Raises
    ------
    ValueError
        If dimensions format is invalid or cannot be parsed.

    Examples
    --------
    >>> _process_dimensions("dimensions      [0 1 -1 0 0 0 0];")
    array([ 0,  1, -1,  0,  0,  0,  0])
    """
    match = re.search(
        r"\[\s*-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s*\]\s*", line
    )
    if match:
        dims = match.group(0).strip("[]").split()
        return np.array([int(d) for d in dims])
    else:
        raise ValueError("Invalid dimensions format")


def find_patches(text: List[str]) -> Any:
    """
    Generator that yields complete patch blocks from OpenFOAM boundaryField file.

    Parameters
    ----------
    text : List[str]
        Lines of the boundaryField file as strings.

    Yields
    ------
    List[str]
        Full text content of a single patch block as list of lines.
        Each yielded item contains all lines belonging to one patch definition.

    Notes
    -----
    This function parses the hierarchical structure of OpenFOAM boundaryField
    files, correctly handling nested braces and extracting complete patch
    definitions including all properties and values.

    Examples
    --------
    >>> with open('0/U') as f:
    ...     lines = f.readlines()
    >>> for patch_lines in find_patches(lines):
    ...     print(f"Patch: {patch_lines[0]}")  # First line is patch name
    """
    in_boundary = False
    start_boundary = False
    in_patch = False
    start_patch = False
    brace_level = 0
    current_patch_lines = []

    for line in text:
        # skip empty lines and comments
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("//"):
            continue

        if "boundaryField" in line:
            in_boundary = True
            continue

        if in_boundary and not start_boundary and "{" in line:
            start_boundary = True
            continue

        # Look for a patch name (simple check: not starting with whitespace, not just braces)
        stripped_line = line.strip()
        if (
            not in_patch
            and stripped_line
            and not stripped_line.startswith("{")
            and not stripped_line.startswith("}")
        ):
            in_patch = True
            current_patch_lines.append(stripped_line)

        if in_patch:
            if brace_level == 0 and "{" in stripped_line:
                start_patch = True
                brace_level += stripped_line.count("{")
                continue
            if not start_patch:
                continue
            if "}" in stripped_line:
                brace_level -= stripped_line.count("}")
            else:
                current_patch_lines.append(stripped_line)

            # If brace_level is 0, we have found the end of the patch
            if brace_level == 0:
                yield current_patch_lines
                # Reset for the next patch
                in_patch = False
                start_patch = False
                current_patch_lines = []
                continue

        # If we find the final closing brace of boundaryField, we can stop
        if stripped_line.startswith("}") and brace_level == 0 and not in_patch:
            break
