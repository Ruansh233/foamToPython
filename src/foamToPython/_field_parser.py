import mmap
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


ALLOWED_DATA_TYPES = {"scalar", "vector"}


def validate_data_type(data_type: str) -> None:
    if data_type not in ALLOWED_DATA_TYPES:
        raise ValueError("Unknown data_type. please use 'scalar' or 'vector'.")


def parse_vector_string(value: str) -> np.ndarray:
    return np.array([float(x) for x in value.strip().strip("()").split()])


def process_dimensions(line: str) -> np.ndarray:
    match = re.search(
        r"\[\s*-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s+-?\d+\s*\]\s*",
        line,
    )
    if not match:
        raise ValueError("Invalid dimensions format")
    dims = match.group(0).strip("[]").split()
    return np.array([int(d) for d in dims])


def num_field(
    subcontent: List[bytes],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[str]]:
    dim_idx = None
    data_size = None
    data_idx = None
    boundary_idx = None
    internal_field_type = None
    searching_for_data_size = False

    idx = 0
    n_lines = len(subcontent)
    while idx < n_lines:
        line = subcontent[idx]

        if dim_idx is None and b"dimensions" in line:
            dim_idx = idx
            idx += 1
            continue

        if internal_field_type is None and b"internalField" in line:
            if b"nonuniform" in line:
                if b"0()" in line:
                    data_idx = idx
                    internal_field_type = "nonuniformZero"
                    data_size = 0
                else:
                    internal_field_type = "nonuniform"
                    searching_for_data_size = True
            else:
                internal_field_type = "uniform"
                data_idx = idx
                data_size = None

        elif searching_for_data_size:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith(b"//"):
                try:
                    data_size = int(stripped_line)
                    data_idx = idx
                    searching_for_data_size = False
                    idx = data_idx + data_size + 1
                    continue
                except ValueError:
                    pass

        if b"boundaryField" in line:
            boundary_idx = idx
            break

        idx += 1

    if internal_field_type is None:
        raise ValueError("internalField not found in the file.")
    if dim_idx is None:
        raise ValueError("dimensions not found in the file.")
    if boundary_idx is None:
        raise ValueError("boundaryField not found in the file.")

    return data_idx, boundary_idx, dim_idx, data_size, internal_field_type


def process_uniform(line: str, data_type: str) -> Union[float, np.ndarray]:
    validate_data_type(data_type)
    if data_type == "scalar":
        match = re.search(r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\d+)", line)
        if not match:
            raise ValueError("Invalid uniform scalar format")
        return float(match.group(1))

    match = re.search(
        r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
        line,
    )
    if not match:
        raise ValueError("Invalid uniform vector format")
    return np.array([float(x) for x in match.group(1).split()])


def process_field(string_coords: List[bytes], data_size: int, data_type: str) -> np.ndarray:
    validate_data_type(data_type)
    if data_type == "scalar":
        joined = b" ".join(string_coords).replace(b"\n", b"")
        internal = np.fromstring(joined.decode("utf-8"), sep=" ", dtype=np.float64)
        if len(internal) != data_size:
            raise ValueError(
                f"Expected {data_size} data points, but got {len(internal)}."
            )
        return internal

    joined = (
        b" ".join(string_coords)
        .replace(b")", b"")
        .replace(b"(", b"")
        .replace(b"\n", b"")
    )
    arr = np.fromstring(joined.decode("utf-8"), sep=" ", dtype=np.float64)
    try:
        return arr.reshape(data_size, 3)
    except ValueError as exc:
        raise ValueError(
            f"Cannot reshape internal field of length {arr.size} to shape ({data_size}, 3)."
        ) from exc


def process_boundary(
    lines: List[Union[str, bytes]], data_type: str, parallel: bool
) -> Dict[str, Dict[str, Any]]:
    validate_data_type(data_type)
    if not lines:
        return {}
    if isinstance(lines[0], bytes):
        lines = [line.decode("utf-8") for line in lines]

    bc_dict: Dict[str, Dict[str, Any]] = {}
    i = 0
    n = len(lines)

    def skip_empty_and_comments(idx: int) -> int:
        while idx < n:
            line = lines[idx].strip()
            if line == "" or line.startswith("//"):
                idx += 1
                continue
            return idx
        return idx

    i = skip_empty_and_comments(i)
    if i >= n or not lines[i].strip().startswith("boundaryField"):
        raise ValueError("File does not start with boundaryField")
    i += 1
    i = skip_empty_and_comments(i)
    if i >= n or lines[i].strip() != "{":
        raise ValueError("Expected '{' after boundaryField")
    i += 1

    while i < n:
        i = skip_empty_and_comments(i)
        if i >= n:
            break
        line = lines[i].strip()
        if line == "}":
            break

        patch_name = line
        i += 1
        i = skip_empty_and_comments(i)
        if i >= n or lines[i].strip() != "{":
            raise ValueError(f"Expected '{{' after {patch_name}")
        i += 1

        props: Dict[str, Any] = {}
        brace_count = 1
        prop_lines: List[str] = []
        while i < n and brace_count > 0:
            l = lines[i].strip()
            if "}" in l:
                brace_count -= l.count("}")
            prop_lines.append(l)
            i += 1
        prop_lines = prop_lines[:-1]

        key = None
        value_lines: List[str] = []
        for l in prop_lines:
            if ";" in l:
                parts = l.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    value_lines.append(value)
                if key:
                    value_str = " ".join(value_lines).replace(";", "").strip()
                    if value_str.startswith("uniform"):
                        if data_type == "scalar":
                            scalar_match = re.match(
                                r"uniform\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                                value_str,
                            )
                            if not scalar_match:
                                raise ValueError(f"Invalid scalar format: {value_str}")
                            props[key] = float(scalar_match.group(1))
                        else:
                            vec_match = re.match(
                                r"uniform\s+\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                                value_str,
                            )
                            if not vec_match:
                                raise ValueError(f"Invalid vector format: {value_str}")
                            props[key] = parse_vector_string(vec_match.group(1))
                    elif value_str.startswith("nonuniform"):
                        if data_type == "scalar":
                            raw = value_str.split("(", 1)[1].rsplit(")", 1)[0]
                            scalar_match = re.findall(
                                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                                raw,
                            )
                            if scalar_match:
                                props[key] = np.array([float(x) for x in scalar_match])
                            elif parallel:
                                props[key] = value_str
                            else:
                                raise ValueError(f"Invalid scalar list format: {raw}")
                        else:
                            vecs = re.findall(
                                r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)\)",
                                value_str,
                            )
                            if vecs:
                                props[key] = np.array(
                                    [[float(x) for x in v.split()] for v in vecs]
                                )
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
                if key is None:
                    parts = l.split()
                    if not parts:
                        continue
                    key = parts[0]
                    value_lines = parts[1:]
                else:
                    value_lines.append(l)

        bc_dict[patch_name] = props

    return bc_dict


def read_field_file(
    filename: str, data_type: str, parallel: bool = False
) -> Tuple[np.ndarray, Union[float, np.ndarray], Dict[str, Dict[str, Any]], str]:
    validate_data_type(data_type)
    with open(filename, "rb") as f:
        file_size = os.path.getsize(filename)
        if file_size > 50 * 1024 * 1024:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                content = mmapped.read().splitlines()
        else:
            content = f.readlines()

    data_idx, boundary_idx, dim_idx, data_size, internal_field_type = num_field(content)
    dimensions = process_dimensions(content[dim_idx].decode("utf-8"))

    if internal_field_type == "uniform":
        internal = process_uniform(content[data_idx].decode("utf-8"), data_type)
    elif internal_field_type == "nonuniform":
        data_start_idx = data_idx + 2
        internal = process_field(
            content[data_start_idx : data_start_idx + data_size],
            data_size,
            data_type,
        )
    elif internal_field_type == "nonuniformZero":
        internal = np.array([]) if data_type == "scalar" else np.empty((0, 3))
    else:
        raise ValueError("internal_field_type should be 'uniform' or 'nonuniform'")

    boundary = process_boundary(content[boundary_idx:], data_type, parallel)
    return dimensions, internal, boundary, internal_field_type


def find_patches(text: List[str]):
    """Yield patch blocks from boundaryField content.

    The first non-empty, non-comment line must be ``boundaryField``. This helper
    is intentionally scoped to boundary sections, not complete OpenFOAM field
    files.
    """
    in_boundary = False
    start_boundary = False
    in_patch = False
    start_patch = False
    brace_level = 0
    current_patch_lines: List[str] = []
    found_content = False

    for line in text:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("//"):
            continue

        found_content = True
        if not in_boundary:
            if not stripped_line.startswith("boundaryField"):
                raise ValueError(
                    "find_patches expects boundaryField content, not a full field file."
                )
            in_boundary = True
            if "{" in stripped_line:
                start_boundary = True
            continue

        if in_boundary and not start_boundary and "{" in line:
            start_boundary = True
            continue
        if in_boundary and not start_boundary:
            raise ValueError("Expected '{' after boundaryField")

        if (
            not in_patch
            and start_boundary
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
            if brace_level == 0:
                yield current_patch_lines
                in_patch = False
                start_patch = False
                current_patch_lines = []
                continue
        if stripped_line.startswith("}") and brace_level == 0 and not in_patch:
            break

    if not found_content:
        raise ValueError("find_patches expects boundaryField content.")
