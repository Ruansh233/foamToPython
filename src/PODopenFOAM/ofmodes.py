import sys
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Union

import multiprocessing

import numpy as np
from scipy.linalg import svd, lu_factor, lu_solve, LinAlgError

from foamToPython.readOFField import OFField

import time


# Class for computing POD modes from OpenFOAM field data
class PODmodes:
    def __init__(
        self,
        fieldList: list,
        POD_algo: str = "eigen",
        rank: int = 10,
        run: bool = True,
    ) -> None:
        """
        Initialize the PODmodes object for computing POD modes from OpenFOAM field data.

        Parameters
        ----------
        fieldList : List[OFField]
            List of OpenFOAM field objects.
        POD_algo : str, optional
            Algorithm for POD ('eigen' or 'svd'), by default 'eigen'.
        rank : int, optional
            Number of modes to compute, by default 10.
        run : bool, optional
            Whether to immediately compute the modes, by default True.

        Raises
        ------
        ValueError
            If rank is greater than the number of fields.
        """
        self.fieldList: List[OFField] = fieldList
        self.POD_algo: str = POD_algo
        if rank > len(fieldList):
            raise ValueError("Rank cannot be greater than the number of fields.")
        self._rank: int = rank
        self._num_processors: int = (
            fieldList[0]._num_processors if fieldList[0].parallel else 1
        )
        self._mode_convert: bool = True

        self.start_time = time.time()
        self.parallel = fieldList[0].parallel

        self.run: bool = run
        if self.run:
            self.getModes()

        if fieldList[0].data_type != "scalar" and fieldList[0].data_type != "vector":
            raise TypeError("Unknown data_type. please use 'scalar' or 'vector'.")

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        if value > len(self.fieldList):
            raise ValueError("Rank cannot be greater than the number of fields.")
        self._rank = value

    @property
    def modes(self):
        if not hasattr(self, "_modes"):
            self.getModes()
        if self.fieldList[0].parallel:
            if not self._mode_convert:
                # Flatten the list of lists into a single list
                return [self._modes[i][: self._rank] for i in range(len(self._modes))]
            else:
                return self._convert_mode_list(self._modes)[: self._rank]
        else:
            return self._modes[: self._rank]

    @property
    def coeffs(self):
        if not hasattr(self, "_coeffs"):
            self._performPOD()
        return self._coeffs[:, : self._rank]

    def getModes(self) -> None:
        """
        Compute and store the POD modes from the field list.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If POD has not been performed yet or if fieldList is empty.
        """
        # Cache frequently accessed properties
        first_field = self.fieldList[0]
        is_parallel = first_field.parallel
        data_type = first_field.data_type

        if not hasattr(self, "data_matrix"):
            if is_parallel:
                self.data_matrix: np.ndarray = self._field2ndarray_parallel(
                    self.fieldList
                )
            else:
                self.data_matrix: np.ndarray = self._field2ndarray_serial(
                    self.fieldList
                )

        print(
            "Convert field to ndarray at time: {:.3f} s".format(
                time.time() - self.start_time
            )
        )

        self._performPOD(self.data_matrix)
        self.truncation_error, self.projection_error = self._truncation_error()

        print("Perform POD at time: {:.3f} s".format(time.time() - self.start_time))

        if is_parallel:
            procN_idx = self._cal_procN_len(first_field, self._num_processors)
            boundary_field = first_field.boundaryField
            dimensions = first_field.dimensions

            # Pre-slice cellModes for each processor to avoid repeated slicing
            cell_modes_slices = [
                self.cellModes[:, procN_idx[procN] : procN_idx[procN + 1]]
                for procN in range(self._num_processors)
            ]

            # Use single pool for both boundary and mode computations
            with multiprocessing.Pool() as pool:
                # Prepare boundary computation tasks
                boundary_tasks = [
                    (
                        [f.boundaryField[procN] for f in self.fieldList],
                        self._coeffs,
                        data_type,
                    )
                    for procN in range(self._num_processors)
                ]
                self.boundaryValues = pool.starmap(
                    self._computeBoundary, boundary_tasks
                )

                print(
                    "Compute boundary values at time: {:.3f} s".format(
                        time.time() - self.start_time
                    )
                )

                # Prepare mode creation tasks
                mode_tasks = [
                    (
                        boundary_field[procN],
                        cell_modes_slices[procN],
                        self.boundaryValues[procN],
                        data_type,
                        dimensions,
                        is_parallel,
                    )
                    for procN in range(self._num_processors)
                ]
                self._modes = pool.starmap(self._createModes, mode_tasks)
        else:
            self.boundaryValues = self._computeBoundary(
                [f.boundaryField for f in self.fieldList],
                self._coeffs,
                data_type,
            )

            print(
                "Compute boundary values at time: {:.3f} s".format(
                    time.time() - self.start_time
                )
            )

            self._modes = self._createModes(
                first_field.boundaryField,
                self.cellModes,
                self.boundaryValues,
                data_type,
                first_field.dimensions,
                is_parallel,
            )

        print("Create modes at time: {:.3f} s".format(time.time() - self.start_time))

    @staticmethod
    def _cal_procN_len(field, num_processors) -> np.ndarray:
        """
        Calculate the number of data points in each processor for parallel fields.

        Parameters
        ----------
        field : OFField
            An OpenFOAM field object.
        Returns
        -------
        List[int]
            A list containing the number of data points in each processor.
        """
        if field.data_type == "scalar":
            procN_len = [
                field.internalField[procN].shape[0] for procN in range(num_processors)
            ]
        elif field.data_type == "vector":
            procN_len = [
                field.internalField[procN].shape[0] * 3
                for procN in range(num_processors)
            ]

        return np.cumsum([0] + procN_len)

    @staticmethod
    def _field2ndarray_serial(fieldList: list) -> np.ndarray:
        """
        Convert a list of OpenFOAM field files to a 2D NumPy array (serial version).

        Parameters
        ----------
        fieldList : list
            List of OpenFOAM field objects.

        Returns
        -------
        np.ndarray
            A 2D array where each row is a flattened field.

        Raises
        ------
        ValueError
            If fieldList is empty or field data is invalid.
        SystemExit
            If uniform fields are encountered or unknown data_type is used.
        """
        if not fieldList:
            raise ValueError("fieldList is empty.")

        if fieldList[0].internal_field_type == "uniform":
            sys.exit("Uniform field is not supported yet.")

        num_fields: int = len(fieldList)
        data_type: str = fieldList[0].data_type

        # Calculate num_data based on data type
        if data_type == "scalar":
            num_data: int = fieldList[0].internalField.size
        elif data_type == "vector":
            if (
                fieldList[0].internalField.ndim != 2
                or fieldList[0].internalField.shape[1] != 3
            ):
                raise ValueError(
                    "Vector field internalField must be a 2D array with shape (num_points, 3)."
                )
            num_data: int = fieldList[0].internalField.shape[0] * 3
        else:
            raise TypeError("Unknown data_type. please use 'scalar' or 'vector'.")

        # Preallocate matrix
        data_matrix: np.ndarray = np.zeros((num_fields, num_data))

        # Process fields based on type (check once, not per iteration)
        if data_type == "scalar":
            for i, field in enumerate(fieldList):
                # ravel() creates a view when possible, faster than flatten()
                data_matrix[i, :] = field.internalField.ravel()
        elif data_type == "vector":
            for i, field in enumerate(fieldList):
                # Use ravel() instead of flatten() for better performance
                data_matrix[i, :] = field.internalField.T.ravel()

        return data_matrix

    @staticmethod
    def _field2ndarray_parallel(fieldList: list) -> np.ndarray:
        """
        Convert a list of OpenFOAM field files to a 2D NumPy array (parallel version).

        Parameters
        ----------
        fieldList : list
            List of OpenFOAM field objects.

        Returns
        -------
        np.ndarray
            A 2D array where each row is a flattened field.

        Raises
        ------
        ValueError
            If fieldList is empty or field data is invalid.
        SystemExit
            If uniform fields are encountered or unknown data_type is used.
        """
        if not fieldList:
            raise ValueError("fieldList is empty.")

        if fieldList[0].internal_field_type == "uniform":
            raise SystemExit("Uniform field is not supported yet.")

        num_fields: int = len(fieldList)
        data_type: str = fieldList[0].data_type

        # Calculate num_data more efficiently
        if data_type == "scalar":
            num_data: int = sum(field.shape[0] for field in fieldList[0].internalField)
        elif data_type == "vector":
            if (
                fieldList[0].internalField[0].ndim != 2
                or fieldList[0].internalField[0].shape[1] != 3
            ):
                raise ValueError(
                    "Vector field internalField must be a 2D array with shape (num_points, 3)."
                )
            num_data: int = (
                sum(field.shape[0] for field in fieldList[0].internalField) * 3
            )
        else:
            raise SystemExit("Unknown data_type. please use 'scalar' or 'vector'.")

        # Preallocate matrix
        data_matrix: np.ndarray = np.zeros((num_fields, num_data))

        # Process fields based on type (check once, not per iteration)
        if data_type == "scalar":
            for i, field in enumerate(fieldList):
                # np.concatenate is faster than np.hstack for 1D arrays
                data_matrix[i, :] = np.concatenate(field.internalField)
        elif data_type == "vector":
            for i, field in enumerate(fieldList):
                # Avoid redundant .flatten() - concatenate already returns 1D
                data_matrix[i, :] = np.concatenate(
                    [f.T.ravel() for f in field.internalField]
                )

        return data_matrix

    @staticmethod
    def _computeBoundary(
        boundaryFields: list, coeffs: np.ndarray, data_type: str
    ) -> dict:
        """
        Compute boundary modes for each patch.

        Parameters
        ----------
        boundaryFields : list
            List of boundary field dictionaries in one processor.
        coeffs : np.ndarray
            The coefficients of the POD modes.
        data_type : str
            The data type ('scalar' or 'vector').

        Returns
        -------
        dict
            Dictionary with patch names as keys and boundary mode values as values.

        Raises
        ------
        ValueError
            If boundary field value type is unknown or coeffs matrix is singular.
        """
        supported_types = ("fixedValue", "fixedGradient", "processor", "calculated")
        boundaryValues: dict = {}

        # Pre-compute inverse once (with error handling)
        try:
            coeffs_inv = np.linalg.inv(coeffs)
        except np.linalg.LinAlgError:
            raise ValueError("Coefficient matrix is singular and cannot be inverted.")

        num_fields = len(boundaryFields)
        first_boundary = boundaryFields[0]

        for patch in first_boundary.keys():
            patch_dict = first_boundary[patch]

            # Skip unsupported types
            if patch_dict["type"] not in supported_types:
                continue

            value_type = next((key for key in patch_dict.keys() if key != "type"), None)
            if value_type is None:
                continue
            first_value = patch_dict[value_type]

            # Skip string values
            if isinstance(first_value, str):
                continue

            # Process only numpy arrays
            if not isinstance(first_value, np.ndarray):
                continue

            # Get value length and type info
            value_len, patch_value_type, uniform_indices = PODmodes._get_value_len(
                boundaryFields, patch, value_type, data_type
            )

            # Pre-allocate the boundary values array
            if data_type == "scalar":
                boundaryValues[patch] = np.zeros((num_fields, value_len))
                # NumPy broadcasting handles both scalar and array assignments
                for i, field in enumerate(boundaryFields):
                    boundaryValues[patch][i, :] = field[patch][value_type]

            elif data_type == "vector":
                boundaryValues[patch] = np.zeros((num_fields, value_len))
                uniform_set = set(uniform_indices)

                for i, field in enumerate(boundaryFields):
                    field_value = field[patch][value_type]
                    if i in uniform_set:
                        # Uniform: 1D array (3,) - tile if needed for mixed, else assign directly
                        if patch_value_type == "uniform":
                            boundaryValues[patch][i, :] = field_value
                        else:  # mixed case
                            boundaryValues[patch][i, :] = np.tile(
                                field_value, (value_len // 3, 1)
                            ).T.ravel()
                    else:
                        # Non-uniform: 2D array (n, 3) - transpose and ravel
                        boundaryValues[patch][i, :] = field_value.T.ravel()

            # Apply coefficient inverse transformation
            boundaryValues[patch] = coeffs_inv @ boundaryValues[patch]

        return boundaryValues

    @staticmethod
    def _get_value_len(
        boundaryFields: list, patch: str, value_type: str, data_type: str
    ) -> Tuple[str, int, list]:
        """
        Determine the value length and type for a boundary patch.

        Parameters
        ----------
        boundaryFields : list
            List of boundary field dictionaries.
        patch : str
            The patch name.
        value_type : str
            The value type (e.g., 'value', 'gradient').
        data_type : str
            The data type ('scalar' or 'vector').

        Returns
        -------
        tuple
            (patch_value_type, value_len, uniform_indices)
            - patch_value_type: 'uniform', 'nonuniform', or 'mixed'
            - value_len: Length of the value array
            - uniform_indices: List of indices with uniform values
        """
        uniform_indices = []

        # Single pass through boundary fields to identify uniform values
        for i, field in enumerate(boundaryFields):
            field_value = field[patch][value_type]
            if data_type == "scalar":
                if field_value.size == 1:
                    uniform_indices.append(i)
            elif data_type == "vector":
                if field_value.ndim == 1:
                    uniform_indices.append(i)

        num_uniform = len(uniform_indices)
        num_total = len(boundaryFields)

        # Determine patch type and value length
        if num_uniform == num_total:
            # All uniform
            patch_value_type = "uniform"
            value_len = 1 if data_type == "scalar" else 3
        elif num_uniform == 0:
            # All non-uniform
            patch_value_type = "nonuniform"
            value_len = boundaryFields[0][patch][value_type].size
        else:
            # Mixed: some uniform, some non-uniform
            patch_value_type = "mixed"
            # Find first non-uniform index to get correct size
            non_uniform_idx = next(
                i for i in range(num_total) if i not in uniform_indices
            )
            value_len = boundaryFields[non_uniform_idx][patch][value_type].size

        return value_len, patch_value_type, uniform_indices

    @staticmethod
    def _createModes(
        bField: dict,
        cellModes: np.ndarray,
        boundaryValues: dict,
        data_type: str,
        dimensions: list,
        parallel: bool,
    ) -> list:
        """
        Assemble the POD modes into OpenFOAM field objects with cellModes and boundaryValues.

        Parameters
        ----------
        bField : dict
            The boundary field dictionary.
        cellModes : np.ndarray
            The cell modes array.
        boundaryValues : dict
            The boundary values dictionary.
        data_type : str
            The data type ('scalar' or 'vector').
        dimensions : list
            The physical dimensions of the field.
        parallel : bool
            Whether the field is parallel or not.

        Returns
        -------
        list
            List of OpenFOAM field objects representing the POD modes.

        Raises
        ------
        ValueError
            If boundary field value type is unknown.
        """
        _modes = []
        internal_field_type = "nonuniform"
        supported_types = {"fixedValue", "fixedGradient", "processor", "calculated"}

        num_modes = cellModes.shape[0]

        for i in range(num_modes):
            mode = OFField()
            mode.data_type = data_type
            mode.dimensions = dimensions
            mode.internal_field_type = internal_field_type
            mode._field_loaded = True
            mode.parallel = parallel

            # Set internal field based on data type
            if data_type == "scalar":
                mode.internalField = cellModes[i, :]
            elif data_type == "vector":
                num_points = cellModes.shape[1] // 3
                mode.internalField = cellModes[i, :].reshape((3, num_points)).T

            # Process boundary fields
            mode.boundaryField = {}
            for patch, patch_dict in bField.items():
                patch_type = patch_dict["type"]
                mode.boundaryField[patch] = {"type": patch_type}

                # Handle supported boundary types
                if patch_type in supported_types:
                    # Get value type (last key that's not 'type')
                    value_type = next(
                        (k for k in reversed(patch_dict.keys()) if k != "type"), None
                    )

                    if value_type is None:
                        continue

                    patch_value = patch_dict[value_type]

                    # Handle string values (e.g., "uniform (0 0 0)")
                    if isinstance(patch_value, str):
                        mode.boundaryField[patch][value_type] = patch_value

                    # Handle numpy array values
                    elif isinstance(patch_value, np.ndarray):
                        if patch not in boundaryValues:
                            continue

                        if data_type == "scalar":
                            mode.boundaryField[patch][value_type] = boundaryValues[
                                patch
                            ][i, :]
                        elif data_type == "vector":
                            mode.boundaryField[patch][value_type] = (
                                boundaryValues[patch][i, :].reshape(3, -1).T
                            )
                    else:
                        raise ValueError(
                            f"Unknown boundary field value type for patch '{patch}'. "
                            f"Expected str or np.ndarray, got {type(patch_value)}."
                        )

                elif len(patch_dict) > 1:
                    # Patch has additional keys beyond 'type' but isn't a supported type
                    raise ValueError(
                        f"Unsupported boundary type '{patch_type}' for patch '{patch}' with additional fields. "
                        f"Supported types are: {', '.join(supported_types)}."
                    )

            _modes.append(mode)

        return _modes

    def writeModes(self, outputDir: str, fieldName: str = "PODmode") -> None:
        """
        Write the POD modes to files. Should be called after `getModes`.

        Parameters
        ----------
        outputDir : str
            The directory where the mode files will be saved, e.g., case folder.
        fieldName : str, optional
            The base name for the mode files, by default 'PODmode'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If POD modes have not been computed yet.
        """
        if not hasattr(self, "modes"):
            raise ValueError(
                "POD modes have not been computed yet. Call getModes() first."
            )
        if self.fieldList[0].parallel:
            tasks = [
                (procN, j + 1, mode, outputDir, fieldName)
                for procN, modeList in enumerate(self._modes)
                for j, mode in enumerate(modeList[: self._rank])
            ]
            with multiprocessing.Pool() as pool:
                pool.map(write_mode_worker, tasks)
        else:
            tasks = [
                (i + 1, mode, outputDir, fieldName)
                for i, mode in enumerate(self._modes[: self._rank])
            ]
            with multiprocessing.Pool() as pool:
                pool.map(write_single_mode, tasks)

    def _performPOD(self, y: np.ndarray) -> None:
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data.

        Parameters
        ----------
        y : np.ndarray
            The training data for which POD is to be performed. Should be a 2D array where each row is a flattened field.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the rank is greater than the number of modes.
        """
        if not hasattr(self, "cellModes"):
            self.cellModes, self.s_all, self._coeffs = self.reduction(
                y, POD_algo=self.POD_algo
            )
        if self._rank > self.cellModes.shape[0]:
            raise ValueError("Rank is greater than the number of modes.")

    @staticmethod
    def reduction(y: np.ndarray, POD_algo: str) -> tuple:
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data using the specified method.

        Parameters
        ----------
        y : np.ndarray
            The training data for which POD is to be performed. Should be a 2D array where each row is a flattened field.
        POD_algo : str
            The algorithm to use ('svd' or 'eigen').

        Returns
        -------
        cellModes : np.ndarray
            The POD modes.
        s_all : np.ndarray
            The singular values or eigenvalues.
        _coeffs : np.ndarray
            The coefficients of the POD modes.

        Raises
        ------
        ValueError
            If POD_algo is not 'svd' or 'eigen'.
        """
        if POD_algo == "svd":
            # SVD-based POD
            u, s_all, cellModes = svd(y, full_matrices=False)
            _coeffs = u @ np.diag(s_all)
            print(f"POD_SVD reduction completed.")
        elif POD_algo == "eigen":
            # Eigenvalue-based POD
            N, M = y.shape

            C: np.ndarray = y @ y.T
            eigenvalues, U = np.linalg.eigh(C)

            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            U = U[:, sorted_indices]

            s_all = np.sqrt(sorted_eigenvalues)
            _coeffs = U @ np.diag(s_all)

            cellModes = np.zeros((N, M))
            for i in range(N):
                u_i = U[:, i]
                cellModes[i, :] = (1 / s_all[i]) * (u_i.T @ y)
            print("POD_eigen reduction completed.")
        else:
            raise ValueError("POD_algo must be 'svd' or 'eigen'.")

        return cellModes, s_all, _coeffs

    def _truncation_error(self) -> np.ndarray:
        """
        Compute the truncation error for each mode.

        Returns
        -------
        tuple of np.ndarray
            truncation_error : np.ndarray
                The truncation error for each mode.
            projection_error : np.ndarray
                The projection error for each mode.
        """
        total_energy: float = np.sum(self.s_all**2)
        cumulative_energy: np.ndarray = np.cumsum(self.s_all**2)
        truncation_error: np.ndarray = 1 - cumulative_energy / total_energy
        numerical_noise_indices = np.where(truncation_error < 0)
        truncation_error[numerical_noise_indices] = 0.0
        projection_error: np.ndarray = np.sqrt(truncation_error)
        return truncation_error, projection_error

    def split_cellData(self) -> np.ndarray:
        """
        Split the cellModes into segments corresponding to each processor.
        Not needed for current implementation but may be useful for future extensions.

        Returns
        -------
        np.ndarray
            Segmented cellModes for each processor.

        Raises
        ------
        ValueError
            If cellModes are not computed or field is not parallel.
        """
        if not hasattr(self, "cellModes"):
            raise ValueError("cellModes not computed yet. Call _performPOD first.")

        if not self.fieldList[0].parallel:
            raise ValueError(
                "Field is not parallel. split_cellModes is not applicable."
            )

        _length = [0]
        for i in range(self.fieldList[0]._num_processors):
            _length.append(_length[-1] + self.fieldList[0].internalField[i].shape[0])
        total_len = sum([field.shape[0] for field in self.fieldList[0].internalField])
        if total_len != _length[-1]:
            raise ValueError("Mismatch in total number of cells across processors.")

        proc_idx = [
            list(range(_length[i], _length[i + 1])) for i in range(len(_length) - 1)
        ]

        cellModes = []
        if self.fieldList[0].data_type == "scalar":
            for procI in proc_idx:
                proc_cellModes = []
                for i in range(self.cellModes.shape[0]):
                    modes = self.cellModes[i, procI]
                    proc_cellModes.append(modes)
                cellModes.append(np.array(proc_cellModes))
        elif self.fieldList[0].data_type == "vector":
            for procI in proc_idx:
                proc_cellModes = []
                idx_x = procI
                idx_y = [i + total_len for i in procI]
                idx_z = [i + 2 * total_len for i in procI]
                idx = idx_x + idx_y + idx_z
                for i in range(self.cellModes.shape[0]):
                    modes = self.cellModes[i, idx]
                    proc_cellModes.append(modes)
                cellModes.append(np.array(proc_cellModes))

        return cellModes

    @staticmethod
    def _reconstructField_parallel(
        _modes: List[List[OFField]], coeffs: np.ndarray, _num_processors: int
    ):
        """
        Reconstruct the original field from the POD modes and coefficients (parallel version).

        Parameters
        ----------
        _modes : List[List[OFField]]
            The list of POD mode OpenFOAM field objects for each processor.
        coeffs : np.ndarray
            The coefficients for reconstructing the field. Shape should be (rank,).
        _num_processors : int
            The number of processors.

        Returns
        -------
        list
            List of OpenFOAM field objects representing the reconstructed field for each processor.

        Raises
        ------
        ValueError
            If rank is greater than the number of modes or coeffs is not 1D.
        """
        rank = coeffs.shape[0]
        if rank > len(_modes[0]):
            raise ValueError("Rank cannot be greater than the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        # Define supported boundary types
        supported_types = {"fixedValue", "fixedGradient", "processor", "calculated"}

        recOFFieldList = []
        for procN in range(_num_processors):
            # Initialize reconstructed field
            recOFField = OFField.from_OFField(_modes[procN][0])
            recOFField.internalField = np.zeros(_modes[procN][0].internalField.shape)

            # Reconstruct internal field using vectorized operation
            for i in range(rank):
                recOFField.internalField += coeffs[i] * _modes[procN][i].internalField

            # Reconstruct boundary field
            first_boundary = _modes[procN][0].boundaryField
            for patch, patch_dict in first_boundary.items():
                patch_type = patch_dict["type"]

                if patch_type in supported_types:
                    # Get value type (last key that's not 'type')
                    value_type = next(
                        (k for k in reversed(patch_dict.keys()) if k != "type"), None
                    )

                    if value_type is None:
                        continue

                    patch_value = patch_dict[value_type]

                    # Handle string values (e.g., "uniform (0 0 0)")
                    if isinstance(patch_value, str):
                        recOFField.boundaryField[patch][value_type] = patch_value

                    # Handle numpy array values
                    elif isinstance(patch_value, np.ndarray):
                        # Initialize with zeros
                        recOFField.boundaryField[patch][value_type] = np.zeros(
                            patch_value.shape
                        )

                        # Accumulate contributions from all modes
                        for i in range(rank):
                            recOFField.boundaryField[patch][value_type] += (
                                coeffs[i]
                                * _modes[procN][i].boundaryField[patch][value_type]
                            )
                    else:
                        raise ValueError(
                            f"Unknown boundary field value type for patch '{patch}'. "
                            f"Expected str or np.ndarray, got {type(patch_value)}."
                        )
                else:
                    # For unsupported types, just copy the type
                    recOFField.boundaryField[patch]["type"] = patch_type

            recOFFieldList.append(recOFField)
        return recOFFieldList

    @staticmethod
    def _reconstructField_serial(_modes: List[OFField], coeffs: np.ndarray):
        """
        Reconstruct the field using the given coefficients and rank (serial version).

        Parameters
        ----------
        _modes : List[OFField]
            The list of POD mode OpenFOAM field objects.
        coeffs : np.ndarray
            The coefficients for reconstructing the field. Shape should be (rank,).

        Returns
        -------
        OFField
            The reconstructed OpenFOAM field object.

        Raises
        ------
        ValueError
            If rank is greater than the number of modes or coeffs is not 1D.
        """
        rank = coeffs.shape[0]
        if rank > len(_modes):
            raise ValueError("Rank cannot be greater than the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        # Initialize reconstructed field
        recOFField = OFField.from_OFField(_modes[0])
        recOFField.internalField = np.zeros(_modes[0].internalField.shape)

        # Reconstruct internal field using vectorized operation
        for i in range(rank):
            recOFField.internalField += coeffs[i] * _modes[i].internalField

        # Define supported boundary types
        supported_types = {"fixedValue", "fixedGradient", "calculated"}

        # Reconstruct boundary field
        first_boundary = _modes[0].boundaryField
        for patch, patch_dict in first_boundary.items():
            patch_type = patch_dict["type"]

            if patch_type in supported_types:
                # Get value type (last key that's not 'type')
                value_type = next(
                    (k for k in reversed(patch_dict.keys()) if k != "type"), None
                )

                if value_type is None:
                    continue

                patch_value = patch_dict[value_type]

                # Handle string values (e.g., "uniform (0 0 0)")
                if isinstance(patch_value, str):
                    recOFField.boundaryField[patch][value_type] = patch_value

                # Handle numpy array values
                elif isinstance(patch_value, np.ndarray):
                    # Initialize with zeros
                    recOFField.boundaryField[patch][value_type] = np.zeros(
                        patch_value.shape
                    )

                    # Accumulate contributions from all modes
                    for i in range(rank):
                        recOFField.boundaryField[patch][value_type] += (
                            coeffs[i] * _modes[i].boundaryField[patch][value_type]
                        )
                else:
                    raise ValueError(
                        f"Unknown boundary field value type for patch '{patch}'. "
                        f"Expected str or np.ndarray, got {type(patch_value)}."
                    )
            else:
                # For unsupported types, just copy the type
                recOFField.boundaryField[patch]["type"] = patch_type

        return recOFField

    def writeRecField(
        self,
        coeffs: np.ndarray,
        outputDir: str,
        timeDir: int,
        fieldName: str = "recField",
    ):
        """
        Write the reconstructed field to files.

        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients for reconstructing the field.
        outputDir : str
            The case directory where the reconstructed field files will be saved.
        timeDir : int
            The time directory for the reconstructed field files.
        fieldName : str, optional
            The base name for the reconstructed field files, by default 'recField'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the reconstructed field format is incorrect.
        """
        if self.parallel:
            recOFField = self._reconstructField_parallel(
                self._modes, coeffs, self._num_processors
            )
            if (
                not isinstance(recOFField, list)
                or len(recOFField) != self._num_processors
            ):
                raise ValueError(
                    "For parallel fields, recOFFields should be a list with length equal to the number of processors."
                )
            tasks = [
                (procN, timeDir, recOFField[procN], outputDir, fieldName)
                for procN in range(self._num_processors)
            ]
            with multiprocessing.Pool() as pool:
                pool.map(write_mode_worker, tasks)
        else:
            recOFField = self._reconstructField_serial(self._modes, coeffs)
            if not isinstance(recOFField, OFField):
                raise ValueError(
                    "For non-parallel fields, recOFFields should be a single OFField object."
                )
            recOFField.writeField(outputDir, timeDir, fieldName)

    @staticmethod
    def _convert_mode_list(mode_list: List[List[OFField]]) -> List[OFField]:
        """
        Convert processor-wise mode list to mode-wise list for parallel fields.

        Transforms mode_list[processor][mode] structure into a list where each mode
        contains data from all processors, enabling parallel field reconstruction.

        Parameters
        ----------
        mode_list : List[List[OFField]]
            Nested list where mode_list[procN][modeN] contains the mode data for
            processor procN and mode modeN. All sublists must have the same length.

        Returns
        -------
        List[OFField]
            List of merged OFField objects, where each mode aggregates data from
            all processors with parallel=True flag set.

        Raises
        ------
        ValueError
            If mode_list is empty or sublists have inconsistent lengths.
        TypeError
            If mode_list structure is invalid or contains non-OFField objects.

        Examples
        --------
        >>> # mode_list[0] = [mode0_proc0, mode1_proc0]
        >>> # mode_list[1] = [mode0_proc1, mode1_proc1]
        >>> converted = _convert_mode_list(mode_list)
        >>> # converted[0] contains mode0 data from all processors
        >>> # converted[1] contains mode1 data from all processors
        """
        # Validate input
        if not mode_list:
            raise ValueError("mode_list cannot be empty.")

        if not isinstance(mode_list, list) or not all(
            isinstance(sublist, list) for sublist in mode_list
        ):
            raise TypeError("mode_list must be a list of lists.")

        num_processors = len(mode_list)
        num_modes = len(mode_list[0])

        # Check consistency across processors
        if not all(len(sublist) == num_modes for sublist in mode_list):
            raise ValueError(
                f"Inconsistent mode counts across processors. "
                f"Expected {num_modes} modes in all sublists, but got varying lengths."
            )

        # Validate that all elements are OFField objects
        for procN, sublist in enumerate(mode_list):
            for modeN, field in enumerate(sublist):
                if not isinstance(field, OFField):
                    raise TypeError(
                        f"mode_list[{procN}][{modeN}] is not an OFField object. "
                        f"Got {type(field).__name__} instead."
                    )

        # Pre-allocate list for better performance
        converted_modes = []

        # Convert modes: transpose the structure from [proc][mode] to [mode][proc]
        for j in range(num_modes):
            # Clone structure from first processor's mode
            mode = OFField.from_OFField(mode_list[0][j])

            # Pre-allocate lists with known size
            mode.internalField = [None] * num_processors
            mode.boundaryField = [None] * num_processors

            # Gather data from all processors for this mode
            for procN in range(num_processors):
                mode.internalField[procN] = mode_list[procN][j].internalField
                mode.boundaryField[procN] = mode_list[procN][j].boundaryField

            mode.parallel = True
            converted_modes.append(mode)

        return converted_modes


def write_mode_worker(args):
    """
    Worker function to write a single POD mode to disk for parallel fields.

    Parameters
    ----------
    args : tuple
        Tuple containing (procN, j, mode, outputDir, fieldName):
            procN (int): Processor number.
            j (int): Mode index.
            mode (OFField): The POD mode object to write.
            outputDir (str): Output directory path.
            fieldName (str): Name for the output field file.

    raises
    ------
    FileNotFoundError
        If the parallel directory does not exist.
    """
    procN, j, mode, outputDir, fieldName = args
    mode.parallel = False
    output_path = f"{outputDir}/processor{procN}"
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Processor directory {output_path} does not exist.")
    mode.writeField(output_path, j, fieldName)
    mode.parallel = True


def write_single_mode(args):
    """
    Worker function to write a single POD mode to disk for serial fields.

    Parameters
    ----------
    args : tuple
        Tuple containing (i, mode, outputDir, fieldName):
            i (int): Mode index.
            mode (OFField): The POD mode object to write.
            outputDir (str): Output directory path.
            fieldName (str): Name for the output field file.
    """
    i, mode, outputDir, fieldName = args
    mode.writeField(outputDir, i, fieldName)
