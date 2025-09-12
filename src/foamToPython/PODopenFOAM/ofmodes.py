import sys
import re
import os

import multiprocessing

import numpy as np
from scipy.linalg import svd

from foamToPython.readOFField import OFField


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
        Args:
            fieldList (list): List of OpenFOAM field objects.
            POD_algo (str): Algorithm for POD ('eigen' or 'svd').
            rank (int): Number of modes to compute.
        """
        self.fieldList: list = fieldList
        self.POD_algo: str = POD_algo
        if rank > len(fieldList):
            raise ValueError("Rank cannot be greater than the number of fields.")
        self._rank: int = rank

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
        return self._coeffs[: self._rank]

    def getModes(self) -> None:
        """
        Get the POD modes. Should be called after `performPOD`.
        Returns:
            np.ndarray: The POD modes.
        Raises:
            ValueError: If POD has not been performed yet.
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

        self._performPOD(self.data_matrix)
        self.truncation_error, self.projection_error = self._truncation_error()

        if self.fieldList[0].parallel:
            self.cellModes = self.split_cellData()

            with multiprocessing.Pool() as pool:
                self.boundaryValues = pool.map(
                    self.boundary_worker, range(self.fieldList[0]._num_processors)
                )
                self._modes = pool.map(
                    self.create_modes_worker, range(self.fieldList[0]._num_processors)
                )

        else:
            self.boundaryValues = self._computeBoundary(
                [f.boundaryField for f in self.fieldList]
            )
            self._modes = self._createModes(
                self.fieldList[0],
                self.fieldList[0].boundaryField,
                self.cellModes,
                self.boundaryValues,
            )

    def _field2ndarray_serial(self, fieldList: list) -> np.ndarray:
        """
        Convert a list of OpenFOAM field files to a 2D NumPy array.
        Each column in the array corresponds to a flattened field from a file.
        Returns:
            np.ndarray: A 2D array where each column is a flattened field.
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

    def _field2ndarray_parallel(self, fieldList: list) -> np.ndarray:
        """
        Convert a list of OpenFOAM field files to a 2D NumPy array.
        Each column in the array corresponds to a flattened field from a file.
        Returns:
            np.ndarray: A 2D array where each column is a flattened field.
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
                    [f.T for f in field.internalField]
                ).flatten()
            else:
                sys.exit("Unknown data_type. please use 'scalar' or 'vector'.")

        return data_matrix

    def _computeBoundary(self, boundaryFields: list) -> None:
        """
        Compute boundary modes for each patch.
        Args:
            boundaryFields (list): A list of boundary field objects.
        """
        data_type = self.fieldList[0].data_type
        boundaryValues: dict = {}
        for patch in boundaryFields[0].keys():
            if (
                boundaryFields[0][patch]["type"] == "fixedValue"
                or boundaryFields[0][patch]["type"] == "fixedGradient"
                or boundaryFields[0][patch]["type"] == "processor"
            ):
                value_type = list(boundaryFields[0][patch].keys())[-1]
                if type(boundaryFields[0][patch][value_type]) is str:
                    continue
                elif type(boundaryFields[0][patch][value_type]) is np.ndarray:
                    value_shape = boundaryFields[0][patch][value_type].ndim
                else:
                    raise ValueError(
                        "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                    )
                if value_shape == 1:
                    num_components: int = 1 if data_type == "scalar" else 3
                    boundaryValues[patch] = np.zeros(
                        (len(boundaryFields), num_components)
                    )

                    for i, field in enumerate(boundaryFields):
                        if data_type == "scalar":
                            boundaryValues[patch][i, :] = field[patch][value_type]
                        elif data_type == "vector":
                            boundaryValues[patch][i, :] = field[patch][value_type]

                elif value_shape == 2:
                    num_points: int = boundaryFields[0][patch][value_type].shape[0]
                    num_components: int = 1 if data_type == "scalar" else 3
                    boundaryValues[patch] = np.zeros(
                        (len(boundaryFields), num_points * num_components)
                    )

                    for i, field in enumerate(boundaryFields):
                        if data_type == "scalar":
                            boundaryValues[patch][i, :] = field[patch][
                                value_type
                            ].flatten()
                        elif data_type == "vector":
                            boundaryValues[patch][i, :] = field[patch][
                                value_type
                            ].T.flatten()

                # The mode equals self.coeffs.inverse() @ self.boundaryModes[patch]
                boundaryValues[patch] = (
                    np.linalg.inv(self._coeffs) @ boundaryValues[patch]
                )

        return boundaryValues

    def _createModes(self, field, bField, cellModes, boundaryValues) -> None:
        """
        Assemble the POD modes into OpenFOAM field objects with cellModes and boundaryValues.
        Args:
            field (OFField): OpenFOAM field objects to access metadata.
            cellModes (np.ndarray): The POD modes in array form.
            boundaryValues (dict): The boundary values for each patch.
        """
        _modes = []
        data_type = field.data_type
        dimensions = field.dimensions
        internal_field_type = "nonuniform"
        parallel = field.parallel
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
                ):
                    mode.boundaryField[patch]["type"] = bField[patch]["type"]
                    value_type = list(bField[patch].keys())[-1]
                    if type(bField[patch][value_type]) is str:
                        continue
                    elif type(bField[patch][value_type]) is np.ndarray:
                        value_shape = bField[patch][value_type].ndim
                    else:
                        raise ValueError(
                            "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                        )
                    if value_shape == 1:
                        if data_type == "scalar":
                            mode.boundaryField[patch][value_type] = boundaryValues[
                                patch
                            ][i, :]
                        elif data_type == "vector":
                            mode.boundaryField[patch][value_type] = boundaryValues[
                                patch
                            ][i, :]
                    elif value_shape == 2:
                        num_points: int = bField[patch][value_type].shape[0]
                        if data_type == "scalar":
                            mode.boundaryField[patch][value_type] = boundaryValues[
                                patch
                            ][i, :]
                        elif data_type == "vector":
                            mode.boundaryField[patch][value_type] = (
                                boundaryValues[patch][i, :].reshape((3, num_points)).T
                            )
                else:
                    mode.boundaryField[patch]["type"] = bField[patch]["type"]

            _modes.append(mode)

        return _modes

    def writeModes(self, outputDir: str, fieldName: str = "PODmode") -> None:
        """
        Write the POD modes to files. Should be called after `getModes`.
        Args:
            outputDir (str): The directory where the mode files will be saved.
            fieldName (str): The base name for the mode files.
        """
        if not hasattr(self, "modes"):
            raise ValueError(
                "POD modes have not been computed yet. Call getModes() first."
            )
        if self.fieldList[0].parallel:
            for procN, modeList in enumerate(self._modes):
                for j, mode in enumerate(modeList):
                    mode.parallel = False
                    os.makedirs(f"{outputDir}/processor{procN}/{j+1}", exist_ok=True)
                    mode.writeField(f"{outputDir}/processor{procN}/{j+1}/{fieldName}")
                    mode.parallel = True
        else:
            for i, mode in enumerate(self._modes[: self._rank]):
                os.makedirs(f"{outputDir}/{i+1}", exist_ok=True)
                mode.writeField(f"{outputDir}/{i+1}/{fieldName}")

    def _performPOD(self, y: np.ndarray) -> np.ndarray:
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data.
        Args:
            y (np.ndarray): The training data for which POD is to be performed.
            The training data should be a 2D array where each row is a flattened field.
        Returns:
            np.ndarray: The coefficients of the POD modes.
        Raises:
            ValueError: If the rank is greater than the number of modes.
        """
        if not hasattr(self, "cellModes"):
            self._reduction(y)
        if self._rank > self.cellModes.shape[0]:
            raise ValueError("Rank is greater than the number of modes.")

    def _reduction(self, y: np.ndarray) -> None:
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data using
        the specified method (SVD or eigenvalue decomposition).
        Args:
            y (np.ndarray): The training data for which POD is to be performed.
            The training data should be a 2D array where each row is a flattened field.
        Returns:
            np.ndarray: The coefficients of the POD modes.
        """
        if self.POD_algo == "svd":
            # SVD-based POD
            u, self.s_all, self.cellModes = svd(y, full_matrices=False)
            self._coeffs = u @ np.diag(self.s_all)
            print(f"POD_SVD reduction completed.")
        elif self.POD_algo == "eigen":
            # Eigenvalue-based POD
            N, M = y.shape

            C: np.ndarray = y @ y.T
            eigenvalues, U = np.linalg.eigh(C)

            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            U = U[:, sorted_indices]

            self.s_all = np.sqrt(sorted_eigenvalues)
            self._coeffs = U @ np.diag(self.s_all)

            self.cellModes = np.zeros((N, M))
            tolerance: float = 1e-10
            for i in range(N):
                if self.s_all[i] > tolerance:
                    u_i = U[:, i]
                    self.cellModes[i, :] = (1 / self.s_all[i]) * (u_i.T @ y)
            print("POD_eigen reduction completed.")
        else:
            raise ValueError("Invalid POD method.")

    def _truncation_error(self) -> np.ndarray:
        """
        Compute the truncation error for each mode.
        Returns:
            np.ndarray: The truncation error for each mode.
        """
        total_energy: float = np.sum(self.s_all**2)
        cumulative_energy: np.ndarray = np.cumsum(self.s_all**2)
        truncation_error: np.ndarray = 1 - cumulative_energy / total_energy
        projection_error: np.ndarray = np.sqrt(1 - cumulative_energy / total_energy)
        return truncation_error, projection_error

    def split_cellData(self) -> None:
        """
        Split the cellModes into segments corresponding to each processor.
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
            for procN in proc_idx:
                proc_cellModes = []
                for i in range(self.cellModes.shape[0]):
                    modes = self.cellModes[i, procN]
                    proc_cellModes.append(modes)
                cellModes.append(np.array(proc_cellModes))
        elif self.fieldList[0].data_type == "vector":
            for procN in proc_idx:
                proc_cellModes = []
                idx_x = procN
                idx_y = [i + total_len for i in procN]
                idx_z = [i + 2 * total_len for i in procN]
                idx = idx_x + idx_y + idx_z
                for i in range(self.cellModes.shape[0]):
                    modes = self.cellModes[i, idx]
                    proc_cellModes.append(modes)
                cellModes.append(np.array(proc_cellModes))

        return cellModes

    def boundary_worker(self, i):
        return self._computeBoundary([f.boundaryField[i] for f in self.fieldList])

    def create_modes_worker(self, i):
        return self._createModes(
            self.fieldList[0],
            self.fieldList[0].boundaryField[i],
            self.cellModes[i],
            self.boundaryValues[i],
        )
