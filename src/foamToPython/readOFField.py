import numpy as np
import sys
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from .headerEnd import *
from .readOFList import _check_data_type


class OFField:
    """
    Class to read and write OpenFOAM field files (scalar/vector).
    """

    filename: str
    fieldName: str
    timeDir: str
    data_type: str
    dimensions: np.ndarray
    internalField: Union[float, np.ndarray]
    internal_field_type: Optional[str]
    num_data_: Optional[int]
    boundaryField: Dict[str, Dict[str, Any]]

    def __init__(self, filename: str, data_type: str) -> None:
        """
        Initialize OFField object.
        Args:
            filename (str): Path to the OpenFOAM field file.
            data_type (str): Type of field ('scalar' or 'vector').
        """
        self.filename = filename
        self.fieldName = filename.split("/")[-1]
        self.timeDir = filename.split("/")[-2]
        self.data_type = data_type
        self._readField()

    def _readField(self) -> None:
        """
        Read the field file and parse internal and boundary fields.
        """
        with open(f"{self.filename}", "rb") as f:
            content = f.readlines()
            data_idx, boundary_start_idx, dim_idx = self._num_field(content)

            self.dimensions = _process_dimensions(content[dim_idx].decode("utf-8"))

            if self.internal_field_type == "uniform":
                self._process_uniform(content[data_idx].decode("utf-8"))
            elif self.internal_field_type == "nonuniform":
                data_start_idx = data_idx + 2
                # Extract relevant lines containing coordinates
                internal_string = content[
                    data_start_idx : data_start_idx + self.num_data_
                ]
                self._process_field(internal_string)

            boundary_string = content[boundary_start_idx:]
            self._process_boundary(boundary_string, self.data_type)

    def _process_uniform(self, line: str) -> None:
        """
        Process uniform internal field value.
        Args:
            line (str): Line containing the uniform value.
        """
        if self.data_type == "scalar":
            # Extract the scalar value after 'uniform'
            match = re.search(r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\d+)", line)
            if match:
                self.internalField = float(match.group(1))
            else:
                raise ValueError("Invalid uniform scalar format")

        elif self.data_type == "vector":
            # Extract the vector value after 'uniform', the element may be integer or float
            # for example: uniform (0 0 1.0) or uniform (1 0 0)
            match = re.search(
                r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                line,
            )
            if match:
                vec_str = match.group(1)
                self.internalField = np.array([float(x) for x in vec_str.split()])
            else:
                raise ValueError("Invalid uniform vector format")

    def _process_field(self, string_coords: List[bytes]) -> None:
        """
        Process nonuniform internal field values.
        Args:
            string_coords (list): List of byte strings containing field values.
        """
        if self.data_type == "scalar":
            # Join all lines and replace unwanted characters once
            joined_coords = b" ".join(string_coords).replace(b"\n", b"")
            # Convert to a numpy array in one go
            self.internalField = np.fromstring(joined_coords, sep=" ", dtype=float)

        elif self.data_type == "vector":
            # Join all lines and replace unwanted characters once
            joined_coords = (
                b" ".join(string_coords)
                .replace(b")", b"")
                .replace(b"(", b"")
                .replace(b"\n", b"")
            )
            # Convert to a numpy array in one go
            self.internalField = np.fromstring(
                joined_coords, sep=" ", dtype=float
            ).reshape(self.num_data_, 3)

        else:
            sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")

    def _process_boundary(self, lines: List[Union[str, bytes]], data_type: str) -> None:
        """
        Process boundaryField section and extract patch properties.
        Args:
            lines (list): List of lines (bytes or str) for boundaryField.
            data_type (str): Type of field ('scalar' or 'vector').
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

            # Parse patch properties
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
                                scalar_match = re.findall(
                                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                    value_str,
                                )
                                if scalar_match:
                                    props[key] = np.array(
                                        [float(x) for x in scalar_match]
                                    )
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
                                else:
                                    raise ValueError(
                                        f"Invalid vector list format: {value_str}"
                                    )
                        else:
                            props[key] = value_str
                        key = None
                        value_lines = []
                    else:
                        parts = l.split(None, 1)
                        if len(parts) == 2:
                            k, v = parts
                            # simple scalar value
                            props[k] = v.replace(";", "").strip()
                else:
                    if key is None:
                        key = l.split()[0]
                        value_lines = l.split()[1:]
                    else:
                        value_lines.append(l)

            bc_dict[patch_name] = props

        self.boundaryField = bc_dict

    def _num_field(
        self, subcontent: List[bytes]
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Find indices for dimensions, internalField, and boundaryField sections.
        Args:
            subcontent (list): List of file lines (bytes).
        Returns:
            tuple: (data_idx, boundary_idx, dim_idx)
        """
        dim_idx = None
        data_size = None
        data_idx = None
        boundary_idx = None
        idx = 0
        self.internal_field_type = None

        while idx < len(subcontent):
            if b"dimensions" in subcontent[idx]:
                dim_idx = idx
            if b"internalField" in subcontent[idx]:
                if b"nonuniform" in subcontent[idx]:
                    self.internal_field_type = "nonuniform"
                    idx += 1
                else:
                    self.internal_field_type = "uniform"
                    data_idx = idx
                    idx += 1
            if data_size is None and self.internal_field_type == "nonuniform":
                try:
                    data_size = int(subcontent[idx])
                    data_idx = idx
                    idx = data_idx + data_size + 1
                except ValueError:
                    idx += 1
            if b"boundaryField" in subcontent[idx]:
                boundary_idx = idx
                break
            idx += 1

        if self.internal_field_type is None:
            raise ValueError("internalField not found in the file.")
        self.num_data_ = data_size
        return data_idx, boundary_idx, dim_idx

    def writeField(
        self,
        fieldPath: str,
        timeDir: Optional[str] = None,
        fieldName: Optional[str] = None,
    ) -> None:
        """
        Write field data to a file in OpenFOAM format.
        Args:
            fieldPath (str): Path to output file.
            timeDir (str, optional): Time directory name.
            fieldName (str, optional): Field name.
        """
        _timeDir = timeDir if timeDir is not None else fieldPath.split("/")[-2]
        _fieldName = fieldName if fieldName is not None else fieldPath.split("/")[-1]

        try:
            int(_timeDir)
        except ValueError:
            sys.exit(
                "The fieldPath should be like '.../0/U' or '.../100/p'. "
                "You can provide <timeDir> and <fieldName> to use other formats."
            )

        with open(fieldPath, "w") as f:
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
                f"dimensions      [{ ' '.join(str(d) for d in self.dimensions) }];\n\n"
            )

            # write internalField for scalar or vector
            if self.data_type == "scalar":
                if self.internal_field_type == "uniform":
                    f.write(f"internalField   uniform {self.internalField:.8g};\n\n")
                elif self.internal_field_type == "nonuniform":
                    f.write(f"internalField   nonuniform List<scalar>\n")
                    f.write(f"{self.internalField.shape[0]}\n")
                    f.write("(\n")
                    for point in self.internalField:
                        f.write(f"{point:.8g}\n")
                    f.write(")\n;\n")
            elif self.data_type == "vector":
                if self.internal_field_type == "uniform":
                    f.write(
                        f"internalField   uniform ({self.internalField[0]:.8g} {self.internalField[1]:.8g} {self.internalField[2]:.8g});\n\n"
                    )
                elif self.internal_field_type == "nonuniform":
                    f.write(f"internalField   nonuniform List<vector>\n")
                    f.write(f"{self.internalField.shape[0]}\n")
                    f.write("(\n")
                    for point in self.internalField:
                        f.write(f"({point[0]:.8g} {point[1]:.8g} {point[2]:.8g})\n")
                    f.write(")\n;\n")

            # write boundaryField
            f.write("boundaryField\n")
            f.write("{\n")
            for patch, props in self.boundaryField.items():
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


def _parse_vector_string(s: str) -> np.ndarray:
    """
    Parse a single vector string like '(0 0 1.0)' into a NumPy array.
    Args:
        s (str): Vector string.
    Returns:
        np.ndarray: Parsed vector.
    """
    s = s.strip("()")
    return np.array([float(x) for x in s.split()])


def _process_dimensions(line: str) -> np.ndarray:
    """
    Parse dimensions line from OpenFOAM file.
    Args:
        line (str): Line containing dimensions.
    Returns:
        np.ndarray: Array of dimensions.
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
    Args:
        text (list): Lines of the boundaryField file.
    Yields:
        str: Full text content of a single patch block.
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
