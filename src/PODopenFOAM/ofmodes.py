import sys
import re
import os
from typing import List, Dict, Any, Optional, Tuple, Union

import multiprocessing

import numpy as np
from scipy.linalg import svd

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
        fieldList : list
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
        self.fieldList: list = fieldList
        self.POD_algo: str = POD_algo
        if rank > len(fieldList):
            raise ValueError("Rank cannot be greater than the number of fields.")
        self._rank: int = rank
        self._num_processors: int = (
            fieldList[0]._num_processors if fieldList[0].parallel else 1
        )

        self.start_time = time.time()
        self.parallel = fieldList[0].parallel

        self.run: bool = run
        if self.run:
            self.getModes()

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
            # Flatten the list of lists into a single list
            return [self._modes[i][: self._rank] for i in range(len(self._modes))]
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
        if not hasattr(self, "data_matrix"):
            if self.fieldList[0].parallel:
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

        if self.fieldList[0].parallel:
            with multiprocessing.Pool() as pool:
                self.boundaryValues = pool.starmap(
                    self._computeBoundary,
                    [
                        (
                            [f.boundaryField[procN] for f in self.fieldList],
                            self._coeffs,
                            self.fieldList[0].data_type,
                        )
                        for procN in range(self._num_processors)
                    ],
                )

                if self.fieldList[0].data_type == "scalar":
                    procN_len = [
                        self.fieldList[0].internalField[procN].shape[0]
                        for procN in range(self._num_processors)
                    ]
                elif self.fieldList[0].data_type == "vector":
                    procN_len = [
                        self.fieldList[0].internalField[procN].shape[0] * 3
                        for procN in range(self._num_processors)
                    ]

                procN_idx = np.cumsum([0] + procN_len)

                self._modes = pool.starmap(
                    self._createModes,
                    [
                        (
                            self.fieldList[0].boundaryField[procN],
                            self.cellModes[:, procN_idx[procN] : procN_idx[procN + 1]],
                            self.boundaryValues[procN],
                            self.fieldList[0].data_type,
                            self.fieldList[0].dimensions,
                            self.fieldList[0].parallel,
                        )
                        for procN in range(self._num_processors)
                    ],
                )
        else:
            self.boundaryValues = self._computeBoundary(
                [f.boundaryField for f in self.fieldList],
                self._coeffs,
                self.fieldList[0].data_type,
            )

            self._modes = self._createModes(
                self.fieldList[0].boundaryField,
                self.cellModes,
                self.boundaryValues,
                self.fieldList[0].data_type,
                self.fieldList[0].dimensions,
                self.fieldList[0].parallel,
            )

        print("Create modes at time: {:.3f} s".format(time.time() - self.start_time))

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
        if fieldList[0].internal_field_type == "uniform":
            sys.exit("Uniform field is not supported yet.")
        num_fields: int = len(fieldList)
        if num_fields == 0:
            raise ValueError("fieldList is empty.")

        if fieldList[0].data_type == "vector":
            if (
                fieldList[0].internalField.ndim != 2
                or fieldList[0].internalField.shape[1] != 3
            ):
                raise ValueError(
                    "Vector field internalField must be a 2D array with shape (num_points, 3)."
                )

        num_data: int = (
            fieldList[0].internalField.size
            if fieldList[0].data_type == "scalar"
            else fieldList[0].internalField.shape[0] * 3
        )
        data_matrix: np.ndarray = np.zeros((num_fields, num_data))
        for i, field in enumerate(fieldList):
            if field.data_type == "scalar":
                data_matrix[i, :] = field.internalField.flatten()
            elif field.data_type == "vector":
                # Flatten vector field by swapping axes
                data_matrix[i, :] = field.internalField.T.flatten()
            else:
                sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")

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
        if fieldList[0].internal_field_type == "uniform":
            sys.exit("Uniform field is not supported yet.")
        num_fields: int = len(fieldList)
        if num_fields == 0:
            raise ValueError("fieldList is empty.")

        if fieldList[0].data_type == "vector":
            if (
                fieldList[0].internalField[0].ndim != 2
                or fieldList[0].internalField[0].shape[1] != 3
            ):
                raise ValueError(
                    "Vector field internalField must be a 2D array with shape (num_points, 3)."
                )

        num_data: int = np.sum([field.shape[0] for field in fieldList[0].internalField])
        num_data: int = num_data if fieldList[0].data_type == "scalar" else num_data * 3

        data_matrix: np.ndarray = np.zeros((num_fields, num_data))
        for i, field in enumerate(fieldList):
            if field.data_type == "scalar":
                data_matrix[i, :] = np.hstack(
                    [f for f in field.internalField]
                ).flatten()
            elif field.data_type == "vector":
                # Flatten vector field by swapping axes
                data_matrix[i, :] = np.hstack(
                    [f.T.flatten() for f in field.internalField]
                ).flatten()
            else:
                sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")

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
            List of boundary field dictionaries.
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
            If boundary field value type is unknown.
        """
        boundaryValues: dict = {}
        for patch in boundaryFields[0].keys():
            if (
                boundaryFields[0][patch]["type"] == "fixedValue"
                or boundaryFields[0][patch]["type"] == "fixedGradient"
                or boundaryFields[0][patch]["type"] == "processor"
                or boundaryFields[0][patch]["type"] == "calculated"
            ):
                value_type = list(boundaryFields[0][patch].keys())[-1]
                if isinstance(boundaryFields[0][patch][value_type], str):
                    continue
                elif isinstance(boundaryFields[0][patch][value_type], np.ndarray):
                    # loop over all boundaryFields to get the value_len
                    value_len = 0
                    uniform_indices = []
                    patch_value_type = None
                    for i, field in enumerate(boundaryFields):
                        if (
                            data_type == "scalar"
                            and boundaryFields[i][patch][value_type].size == 1
                        ):
                            uniform_indices.append(i)
                        if (
                            data_type == "vector"
                            and boundaryFields[i][patch][value_type].ndim == 1
                        ):
                            uniform_indices.append(i)

                    if len(uniform_indices) == len(boundaryFields):
                        patch_value_type = "uniform"
                        if data_type == "scalar":
                            value_len = 1
                        elif data_type == "vector":
                            value_len = 3
                    elif len(uniform_indices) == 0:
                        patch_value_type = "nonuniform"
                        value_len = boundaryFields[0][patch][value_type].size
                    else:
                        patch_value_type = "mixed"
                        one_index = list(
                            set(range(len(boundaryFields))) - set(uniform_indices)
                        )
                        value_len = boundaryFields[one_index[0]][patch][value_type].size

                if data_type == "scalar":
                    if (
                        patch_value_type == "uniform"
                        or patch_value_type == "nonuniform"
                    ):
                        boundaryValues[patch] = np.zeros(
                            (len(boundaryFields), value_len)
                        )
                        for i, field in enumerate(boundaryFields):
                            boundaryValues[patch][i, :] = field[patch][value_type]
                    elif patch_value_type == "mixed":
                        boundaryValues[patch] = np.zeros(
                            (len(boundaryFields), value_len)
                        )
                        for i, field in enumerate(boundaryFields):
                            if i in uniform_indices:
                                boundaryValues[patch][i, :] = np.full(
                                    (value_len,), field[patch][value_type]
                                )
                            else:
                                boundaryValues[patch][i, :] = field[patch][value_type]

                elif data_type == "vector":
                    if patch_value_type == "uniform":
                        boundaryValues[patch] = np.zeros((len(boundaryFields), 3))
                        for i, field in enumerate(boundaryFields):
                            boundaryValues[patch][i, :] = field[patch][value_type]
                    elif patch_value_type == "nonuniform":
                        boundaryValues[patch] = np.zeros(
                            (len(boundaryFields), value_len)
                        )
                        for i, field in enumerate(boundaryFields):
                            boundaryValues[patch][i, :] = field[patch][
                                value_type
                            ].T.flatten()
                    elif patch_value_type == "mixed":
                        boundaryValues[patch] = np.zeros(
                            (len(boundaryFields), value_len)
                        )
                        for i, field in enumerate(boundaryFields):
                            if i in uniform_indices:
                                boundaryValues[patch][i, :] = np.tile(
                                    field[patch][value_type], value_len // 3
                                ).T.flatten()
                            else:
                                boundaryValues[patch][i, :] = field[patch][
                                    value_type
                                ].T.flatten()

                # The mode equals self.coeffs.inverse() @ self.boundaryModes[patch]
                boundaryValues[patch] = np.linalg.inv(coeffs) @ boundaryValues[patch]

        return boundaryValues

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
        for i in range(0, cellModes.shape[0]):
            mode = OFField()
            mode.data_type = data_type
            mode.dimensions = dimensions
            mode.internal_field_type = internal_field_type
            mode._field_loaded = True
            mode.parallel = parallel

            if data_type == "scalar":
                mode.internalField = cellModes[i, :]
            elif data_type == "vector":
                num_points: int = cellModes[i, :].shape[0] // 3
                mode.internalField = cellModes[i, :].reshape((3, num_points)).T

            mode.boundaryField = {}
            for patch in bField.keys():
                mode.boundaryField[patch] = {}
                if (
                    bField[patch]["type"] == "fixedValue"
                    or bField[patch]["type"] == "fixedGradient"
                    or bField[patch]["type"] == "processor"
                    or bField[patch]["type"] == "calculated"
                ):
                    mode.boundaryField[patch]["type"] = bField[patch]["type"]
                    value_type = list(bField[patch].keys())[-1]
                    if isinstance(bField[patch][value_type], str):
                        mode.boundaryField[patch][value_type] = bField[patch][
                            value_type
                        ]
                    elif isinstance(bField[patch][value_type], np.ndarray):
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
                            "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                        )
                elif len(bField[patch].keys()) == 1:
                    mode.boundaryField[patch]["type"] = bField[patch]["type"]
                else:
                    raise ValueError(
                        """Unknown boundary field value type for patch with single type.
                        Supported types are fixedValue, fixedGradient, processor, calculated.
                        or patch only with type."""
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
                (procN, j, mode, outputDir, fieldName)
                for procN, modeList in enumerate(self._modes)
                for j, mode in enumerate(modeList[: self._rank])
            ]
            with multiprocessing.Pool() as pool:
                pool.map(write_mode_worker, tasks)
        else:
            tasks = [
                (i, mode, outputDir, fieldName)
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
            If rank is greater than the number of modes.
        """
        rank = coeffs.shape[0]
        if rank > len(_modes[0]):
            raise ValueError("Rank cannot be greater than the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        recOFFieldList = []
        for procN in range(_num_processors):
            # Reconstruct internal field
            recOFField = OFField.from_OFField(_modes[procN][0])
            recOFField.internalField = np.zeros(_modes[procN][0].internalField.shape)
            for i in range(rank):
                recOFField.internalField += coeffs[i] * _modes[procN][i].internalField

            # Reconstruct boundary field
            for patch in _modes[procN][0].boundaryField.keys():
                patch_type = _modes[procN][0].boundaryField[patch]["type"]
                if (
                    patch_type == "fixedValue"
                    or patch_type == "fixedGradient"
                    or patch_type == "processor"
                    or patch_type == "calculated"
                ):
                    value_type = list(_modes[procN][0].boundaryField[patch].keys())[-1]
                    if isinstance(
                        _modes[procN][0].boundaryField[patch][value_type],
                        str,
                    ):
                        recOFField.boundaryField[patch][value_type] = _modes[procN][
                            0
                        ].boundaryField[patch][value_type]
                    elif isinstance(
                        _modes[procN][0].boundaryField[patch][value_type],
                        np.ndarray,
                    ):
                        recOFField.boundaryField[patch][value_type] = np.zeros(
                            _modes[procN][0].boundaryField[patch][value_type].shape
                        )
                        for i in range(rank):
                            recOFField.boundaryField[patch][value_type] += (
                                coeffs[i]
                                * _modes[procN][i].boundaryField[patch][value_type]
                            )
                    else:
                        raise ValueError(
                            "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                        )
                else:
                    recOFField.boundaryField[patch]["type"] = _modes[procN][
                        0
                    ].boundaryField[patch]["type"]

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
            If rank is greater than the number of modes.
        """
        rank = coeffs.shape[0]
        if rank != len(_modes):
            raise ValueError("Rank must match the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        recOFField = OFField.from_OFField(_modes[0])
        recOFField.internalField = np.zeros(_modes[0].internalField.shape)
        for i in range(rank):
            recOFField.internalField += coeffs[i] * _modes[i].internalField

        # Reconstruct boundary field
        for patch in _modes[0].boundaryField.keys():
            patch_type = _modes[0].boundaryField[patch]["type"]
            if (
                patch_type == "fixedValue"
                or patch_type == "fixedGradient"
                or patch_type == "calculated"
            ):
                value_type = list(_modes[0].boundaryField[patch].keys())[-1]
                if isinstance(_modes[0].boundaryField[patch][value_type], str):
                    recOFField.boundaryField[patch][value_type] = _modes[
                        0
                    ].boundaryField[patch][value_type]
                elif isinstance(_modes[0].boundaryField[patch][value_type], np.ndarray):
                    recOFField.boundaryField[patch][value_type] = np.zeros(
                        _modes[0].boundaryField[patch][value_type].shape
                    )
                    for i in range(rank):
                        recOFField.boundaryField[patch][value_type] += (
                            coeffs[i] * _modes[i].boundaryField[patch][value_type]
                        )
                else:
                    raise ValueError(
                        "Unknown boundary field value type for fixedValue, fixedGradient, or calculated."
                    )
            else:
                recOFField.boundaryField[patch]["type"] = _modes[0].boundaryField[
                    patch
                ]["type"]

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
                (procN, timeDir - 1, recOFField[procN], outputDir, fieldName)
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
            os.makedirs(f"{outputDir}/", exist_ok=True)
            recOFField.writeField(f"{outputDir}/{timeDir}/{fieldName}")


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
    """
    procN, j, mode, outputDir, fieldName = args
    mode.parallel = False
    output_path = f"{outputDir}/processor{procN}/{j+1}"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    mode.writeField(f"{output_path}/{fieldName}")
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
    output_path = f"{outputDir}/{i+1}"
    os.makedirs(output_path, exist_ok=True)
    mode.writeField(f"{output_path}/{fieldName}")
