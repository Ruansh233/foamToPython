import numpy as np
import sys
import re
from .headerEnd import *


def _num_list(subcontent):
    for idx, line in enumerate(subcontent):
        try:
            return int(line), idx
        except ValueError:
            continue


def _check_data_type(data):
    data = data.decode("utf-8").strip()
    if re.match(r"^\(\s*-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s*\)$", data):
        return "vector"
    elif re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", data):
        return "scalar"
    elif re.match(r"^\d+$", data):
        return "label"
    else:
        sys.exit("Unknown data_type. please use 'label', 'scalar' or 'vector'.")


def _extractList(content, data_type):
    num_data_, data_idx = _num_list(content)
    data_start_idx = data_idx + 2
    # Extract relevant lines containing coordinates
    string_coords = content[data_start_idx : data_start_idx + num_data_]

    if data_type is None:
        data_type = _check_data_type(string_coords[0])

    if data_type == "label":
        # Join all lines and replace unwanted characters once
        joined_coords = b" ".join(string_coords).replace(b"\n", b"")
        # Convert to a numpy array in one go
        data_array = np.fromstring(joined_coords, sep=" ", dtype=int)

    elif data_type == "scalar":
        # Join all lines and replace unwanted characters once
        joined_coords = b" ".join(string_coords).replace(b"\n", b"")
        # Convert to a numpy array in one go
        data_array = np.fromstring(joined_coords, sep=" ", dtype=float)

    elif data_type == "vector":
        # Join all lines and replace unwanted characters once
        joined_coords = (
            b" ".join(string_coords)
            .replace(b")", b"")
            .replace(b"(", b"")
            .replace(b"\n", b"")
        )
        # Convert to a numpy array in one go
        data_array = np.fromstring(joined_coords, sep=" ", dtype=float).reshape(
            num_data_, 3
        )

    else:
        sys.exit("Unknown data_type. please use 'label', 'scalar' or 'vector'.")

    return data_array


def _extractListList(content, data_type):
    num_list_, start_idx = _num_list(content)
    # print(f"The number of list is: {num_list_}")
    subcontent = content[start_idx + 1 :]
    data_list = []
    for i in range(num_list_):
        num_data_, data_idx = _num_list(subcontent)
        data_start_idx = data_idx + 2
        # Extract relevant lines containing coordinates
        string_coords = subcontent[data_start_idx : data_start_idx + num_data_]

        if data_type == "label":
            # Join all lines and replace unwanted characters once
            joined_coords = b" ".join(string_coords).replace(b"\n", b"")
            # Convert to a numpy array in one go
            data_list.append(np.fromstring(joined_coords, sep=" ", dtype=int))

        elif data_type == "scalar":
            # Join all lines and replace unwanted characters once
            joined_coords = b" ".join(string_coords).replace(b"\n", b"")
            # Convert to a numpy array in one go
            data_list.append(np.fromstring(joined_coords, sep=" ", dtype=float))

        elif data_type == "vector":
            # Join all lines and replace unwanted characters once
            joined_coords = (
                b" ".join(string_coords)
                .replace(b")", b"")
                .replace(b"(", b"")
                .replace(b"\n", b"")
            )
            # Convert to a numpy array in one go
            data_list.append(
                np.fromstring(joined_coords, sep=" ", dtype=float).reshape(num_data_, 3)
            )
        else:
            sys.exit("Unknown data_type. please use 'label', 'scalar' or 'vector'.")

        subcontent = subcontent[data_start_idx + num_data_ + 1 :]

    return np.array(data_list)


def readList(filename, data_type):
    try:
        with open(filename, "rb") as f:
            pass
    except FileNotFoundError:
        sys.exit(f"File {filename} not found. Please check the file path.")
    with open(f"{filename}", "rb") as f:
        return _extractList(f.readlines(), data_type)


def readListList(filename, data_type):
    try:
        with open(filename, "rb") as f:
            pass
    except FileNotFoundError:
        sys.exit(f"File {filename} not found. Please check the file path.")
    with open(f"{filename}", "rb") as f:
        return _extractListList(f.readlines(), data_type)


def writeList(data, data_type, filename, object_name="None"):
    """
    Write data to a file
    :param data: data to write
    :param data_type: data type
    :param filename: file name
    :return: None
    """

    with open(filename, "w") as f:
        output = []

        thisHeader = header.replace("className;", f"{data_type}List;")
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
