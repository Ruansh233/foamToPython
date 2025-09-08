import numpy as np
import sys
import re
from readOFData.headerEnd import *
from readOFData.readOFList import _check_data_type


class OFField:
    def __init__(self, filename, data_type=None):
        self.filename = filename
        self.data_type = data_type
        self._readField()

    def _readField(self):
        with open(f"{self.filename}", "rb") as f:
            content = f.readlines()
            self.num_data_, data_idx, boundary_start_idx = _num_field(content)
            data_start_idx = data_idx + 2
            # Extract relevant lines containing coordinates
            internal_string = content[data_start_idx : data_start_idx + self.num_data_]

            self._process_field(internal_string, self.data_type)

            boundary_string = content[boundary_start_idx:]
            self.boundaryField = _process_boundary(boundary_string, self.data_type)

    def _process_field(self, string_coords, data_type):
        if data_type is None:
            self.data_type = _check_data_type(string_coords[0])

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

    def writeField(data, data_type, filename, object_name="None"):
        """
        Write data to a file
        :param data: data to write
        :param data_type: data type
        :param filename: file name
        :return: None
        """

        with open(filename, "w") as f:
            output = []

            thisHeader = header.replace("className;", f"{data_type}Field;")
            if object_name != "None":
                thisHeader = thisHeader.replace(
                    "object      data;", f"object      {object_name};"
                )
                output.append(thisHeader + "\n\n")
            else:
                output.append(thisHeader + "\n\n")

            output.append(f"{data.shape[0]}\n")
            output.append("(\n")
            if data_type == "label":
                for point in data:
                    output.append(f"{point:d}\n")
            elif data_type == "scalar":
                for point in data:
                    output.append(f"{point:.8e}\n")
            elif data_type == "vector":
                for point in data:
                    output.append(f"({point[0]:.8e} {point[1]:.8e} {point[2]:.8e})\n")
            else:
                sys.exit("Unknown data_type. please use 'label', 'scalar' or 'vector'.")
            output.append(")\n")
            output.append(ender)
            f.write("".join(output))


def _num_field(subcontent):
    data_size = None
    data_idx = None
    boundary_idx = None
    for idx, line in enumerate(subcontent):
        if data_size is None:
            try:
                data_size = int(line)
                data_idx = idx
            except ValueError:
                pass
        if b"boundaryField" in line:
            boundary_idx = idx
            break
    return data_size, data_idx, boundary_idx


def _parse_vector_string(s):
    """Parse a single vector string like '(0 0 1.0)' into a NumPy array."""
    s = s.strip("()")
    return np.array([float(x) for x in s.split()])


def _process_boundary(lines, data_type):
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
                                r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", value_str
                            )
                            if scalar_match:
                                props[key] = float(scalar_match.group(1))
                            else:
                                raise ValueError(f"Invalid scalar format: {value_str}")
                        elif data_type == "vector":
                            # read vector
                            vec_match = re.match(
                                r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)", value_str
                            )
                            if vec_match:
                                props[key] = _parse_vector_string(vec_match.group(1))
                            else:
                                raise ValueError(f"Invalid vector format: {value_str}")
                    elif value_str.startswith("nonuniform List<vector>"):
                        # read scalar list
                        if data_type == "scalar":
                            scalar_match = re.findall(
                                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                value_str,
                            )
                            if scalar_match:
                                props[key] = np.array([float(x) for x in scalar_match])
                            else:
                                raise ValueError(f"Invalid scalar list format: {value_str}")
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
                                raise ValueError(f"Invalid vector list format: {value_str}")
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

    return bc_dict
