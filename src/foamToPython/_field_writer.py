import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._mp_utils import get_spawn_pool
from .headerEnd import ender, header


def normalize_write_target(
    case_path: str,
    time_dir: Optional[Union[int, str]] = None,
    field_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    if time_dir is None and field_name is None:
        normalized = os.path.normpath(case_path)
        derived_case, derived_field = os.path.split(normalized)
        derived_case, derived_time = os.path.split(derived_case)
        if not derived_case or not derived_time or not derived_field:
            raise ValueError(
                "Invalid field path. Expected format like 'case/2/U' for single-path writeField."
            )
        return derived_case, str(derived_time), str(derived_field)

    if time_dir is None or field_name is None:
        raise TypeError(
            "writeField requires either (casePath, timeDir, fieldName) or a single field path."
        )
    return case_path, str(time_dir), str(field_name)


def _write_boundary_line(file_obj, key: str, value: Any, precision: int) -> None:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            file_obj.write(f"        {key} uniform {float(value):.{precision}g};\n")
        elif value.ndim == 1 and value.shape[0] == 3:
            file_obj.write(
                f"        {key} uniform ({value[0]:.{precision}g} {value[1]:.{precision}g} {value[2]:.{precision}g});\n"
            )
        elif value.ndim == 1:
            file_obj.write(f"        {key} nonuniform List<scalar>\n")
            file_obj.write(f"{value.shape[0]}\n")
            file_obj.write("(\n")
            for entry in value:
                file_obj.write(f"{entry:.{precision}g}\n")
            file_obj.write(");\n")
        elif value.ndim == 2 and value.shape[0] == 1 and value.shape[1] != 3:
            file_obj.write(f"        {key} nonuniform List<scalar>\n")
            file_obj.write(f"{value.shape[1]}\n")
            file_obj.write("(\n")
            for entry in value.ravel():
                file_obj.write(f"{entry:.{precision}g}\n")
            file_obj.write(");\n")
        elif value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == 3:
            file_obj.write(
                f"        {key} uniform ({value[0,0]:.{precision}g} {value[0,1]:.{precision}g} {value[0,2]:.{precision}g});\n"
            )
        elif value.ndim == 2 and value.shape[1] == 3:
            file_obj.write(f"        {key} nonuniform List<vector>\n")
            file_obj.write(f"{value.shape[0]}\n")
            file_obj.write("(\n")
            for entry in value:
                file_obj.write(
                    f"({entry[0]:.{precision}g} {entry[1]:.{precision}g} {entry[2]:.{precision}g})\n"
                )
            file_obj.write(");\n")
        else:
            raise ValueError(
                f"Unsupported boundary array shape for key '{key}': {value.shape}"
            )
    else:
        file_obj.write(f"        {key} {value};\n")


def write_field_serial(
    case_path: str,
    internal_field: Union[float, np.ndarray],
    boundary_field: Dict[str, Dict[str, Any]],
    time_dir: Union[int, str],
    field_name: str,
    data_type: str,
    dimensions: np.ndarray,
    internal_field_type: str,
    precision: int = 10,
) -> None:
    field_dir = os.path.join(case_path, str(time_dir), field_name)
    os.makedirs(os.path.dirname(field_dir), exist_ok=True)

    with open(field_dir, "w", encoding="utf-8") as f:
        this_header = header.replace("className;", f"vol{data_type.capitalize()}Field;")
        this_header = this_header.replace("timeDir;", f"{time_dir};")
        this_header = this_header.replace("object      data;", f"object      {field_name};")
        f.write(this_header + "\n\n")

        f.write(f"dimensions      [{ ' '.join(str(d) for d in dimensions) }];\n\n")

        if data_type == "scalar":
            if internal_field_type == "uniform":
                f.write(f"internalField   uniform {float(internal_field):.{precision}g};\n\n")
            elif internal_field_type == "nonuniform":
                f.write("internalField   nonuniform List<scalar>\n")
                f.write(f"{internal_field.shape[0]}\n")
                f.write("(\n")
                for point in internal_field:
                    f.write(f"{point:.{precision}g}\n")
                f.write(")\n;\n")
            else:
                raise ValueError(
                    "internal_field_type should be 'uniform' or 'nonuniform'"
                )
        elif data_type == "vector":
            if internal_field_type == "uniform":
                f.write(
                    f"internalField   uniform ({internal_field[0]:.{precision}g} {internal_field[1]:.{precision}g} {internal_field[2]:.{precision}g});\n\n"
                )
            elif internal_field_type == "nonuniform":
                f.write("internalField   nonuniform List<vector>\n")
                f.write(f"{internal_field.shape[0]}\n")
                f.write("(\n")
                for point in internal_field:
                    f.write(
                        f"({point[0]:.{precision}g} {point[1]:.{precision}g} {point[2]:.{precision}g})\n"
                    )
                f.write(")\n;\n")
            else:
                raise ValueError(
                    "internal_field_type should be 'uniform' or 'nonuniform'"
                )
        else:
            raise ValueError("Unknown data_type. please use 'scalar' or 'vector'.")

        f.write("boundaryField\n")
        f.write("{\n")
        for patch, props in boundary_field.items():
            f.write(f"    {patch}\n")
            f.write("    {\n")
            for key, value in props.items():
                _write_boundary_line(f, key, value, precision)
            f.write("    }\n")
        f.write("}\n\n")
        f.write(ender)


def _write_parallel_worker(args: Tuple[Any, ...]) -> None:
    (
        proc_path,
        internal_field,
        boundary_field,
        time_dir,
        field_name,
        data_type,
        dimensions,
        internal_field_type,
        precision,
    ) = args
    write_field_serial(
        proc_path,
        internal_field,
        boundary_field,
        time_dir,
        field_name,
        data_type,
        dimensions,
        internal_field_type,
        precision,
    )


def write_field_parallel(
    case_path: str,
    internal_field: List[np.ndarray],
    boundary_field: List[Dict[str, Dict[str, Any]]],
    time_dir: Union[int, str],
    field_name: str,
    data_type: str,
    dimensions: np.ndarray,
    internal_field_type: str,
    precision: int,
    num_batch: int,
) -> None:
    if not isinstance(internal_field, list) or not isinstance(boundary_field, list):
        raise ValueError(
            "For parallel writing, internalField and boundaryField should be lists."
        )
    num_processors = len(internal_field)
    proc_paths = [os.path.join(case_path, f"processor{idx}") for idx in range(num_processors)]

    tasks = [
        (
            proc_path,
            internal_field[idx],
            boundary_field[idx],
            time_dir,
            field_name,
            data_type,
            dimensions,
            internal_field_type,
            precision,
        )
        for idx, proc_path in enumerate(proc_paths)
    ]
    with get_spawn_pool(processes=num_batch) as pool:
        pool.map(_write_parallel_worker, tasks)
