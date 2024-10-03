import numpy as np
import sys


def num_list(subcontent):
    for idx, line in enumerate(subcontent):
        try:
            return int(line), idx
        except ValueError:
            continue


def readList(filename, data_type):
    with open(f"{filename}", "rb") as f:
        return extractList(f.readlines(), data_type)


def readListList(filename, data_type):
    with open(f"{filename}", "rb") as f:
        return extractListList(f.readlines(), data_type)


def extractList(content, data_type):
    num_data_, data_idx = num_list(content)
    data_start_idx = data_idx + 2
    # Extract relevant lines containing coordinates
    string_coords = content[data_start_idx : data_start_idx + num_data_]

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


def extractListList(content, data_type):
    num_list_, start_idx = num_list(content)
    # print(f"The number of list is: {num_list_}")
    subcontent = content[start_idx + 1 :]
    data_list = []
    for i in range(num_list_):
        num_data_, data_idx = num_list(subcontent)
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
