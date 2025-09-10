import sys
import re
import os

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
        compute_or_not: bool = True,
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

        self.compute_or_not: bool = compute_or_not
        if self.compute_or_not:
            self.getModes()

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        if value > len(self.fieldList):
            raise ValueError("Rank cannot be greater than the number of fields.")
        if value > self._rank:
            self._rank = value
            self.getModes()
        else:
            self._rank = value

    @property
    def modes(self):
        if not hasattr(self, "_modes"):
            self.getModes()
        return self._modes

    @property
    def coeffs(self):
        if not hasattr(self, "_coeffs"):
            self._performPOD()
        return self._coeffs

    def _field2ndarray(self, fieldList: list) -> np.ndarray:
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

        first_field = fieldList[0]
        num_data: int = (
            first_field.internalField.size
            if first_field.data_type == "scalar"
            else first_field.internalField.shape[0] * first_field.internalField.shape[1]
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

    def getModes(self) -> np.ndarray:
        """
        Get the POD modes. Should be called after `performPOD`.
        Returns:
            np.ndarray: The POD modes.
        Raises:
            ValueError: If POD has not been performed yet.
        """
        data_matrix: np.ndarray = self._field2ndarray(self.fieldList)
        self._performPOD(data_matrix)
        self.truncation_error, self.projection_error = self._truncation_error()
        self._computeBoundaryValues(self.fieldList)

        # If self._modes exists, append new modes; else, initialize
        if hasattr(self, "_modes"):
            start_idx = len(self._modes)
        else:
            self._modes = []
            start_idx = 0
        for i in range(start_idx, self._rank):
            mode = OFField()
            mode._field_loaded = True
            mode.data_type = self.fieldList[0].data_type
            mode.dimensions = self.fieldList[0].dimensions
            mode.internal_field_type = "nonuniform"

            if self.fieldList[0].data_type == "scalar":
                mode.internalField = self.cellModes[i, :]
            elif self.fieldList[0].data_type == "vector":
                num_points: int = self.fieldList[0].internalField.shape[0]
                mode.internalField = self.cellModes[i, :].reshape((3, num_points)).T

            mode.boundaryField = {}
            for patch in self.fieldList[0].boundaryField.keys():
                mode.boundaryField[patch] = {}
                if (
                    self.fieldList[0].boundaryField[patch]["type"] == "fixedValue"
                    or self.fieldList[0].boundaryField[patch]["type"] == "fixedGradient"
                ):
                    value_type = list(self.fieldList[0].boundaryField[patch].keys())[-1]
                    value_shape = len(
                        self.fieldList[0].boundaryField[patch][value_type].shape
                    )
                    if value_shape == 1:
                        if self.fieldList[0].data_type == "scalar":
                            mode.boundaryField[patch][value_type] = self.boundaryValues[
                                patch
                            ][i, :]
                        elif self.fieldList[0].data_type == "vector":
                            mode.boundaryField[patch][value_type] = self.boundaryValues[
                                patch
                            ][i, :]
                    elif value_shape == 2:
                        num_points: int = (
                            self.fieldList[0].boundaryField[patch][value_type].shape[0]
                        )
                        if self.fieldList[0].data_type == "scalar":
                            mode.boundaryField[patch][value_type] = self.boundaryValues[
                                patch
                            ][i, :]
                        elif self.fieldList[0].data_type == "vector":
                            mode.boundaryField[patch][value_type] = (
                                self.boundaryValues[patch][i, :]
                                .reshape((3, num_points))
                                .T
                            )
                else:
                    mode.boundaryField[patch]["type"] = self.fieldList[0].boundaryField[
                        patch
                    ]["type"]
            self._modes.append(mode)

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
        for i, mode in enumerate(self._modes[: self._rank]):
            os.makedirs(f"{outputDir}/{i+1}", exist_ok=True)
            mode.writeField(f"{outputDir}/{i+1}/{fieldName}")

    def _computeBoundaryValues(self, fieldList: list) -> None:
        """
        Compute boundary modes for each patch.
        Args:
            fieldList (list): List of OpenFOAM field objects.
        """
        self.boundaryValues: dict = {}
        for patch in self.fieldList[0].boundaryField.keys():
            if (
                self.fieldList[0].boundaryField[patch]["type"] == "fixedValue"
                or self.fieldList[0].boundaryField[patch]["type"] == "fixedGradient"
            ):
                value_type = list(self.fieldList[0].boundaryField[patch].keys())[-1]
                value_shape = len(
                    self.fieldList[0].boundaryField[patch][value_type].shape
                )
                if value_shape == 1:
                    num_components: int = (
                        1 if self.fieldList[0].data_type == "scalar" else 3
                    )
                    self.boundaryValues[patch] = np.zeros(
                        (len(fieldList), num_components)
                    )

                    for i, field in enumerate(fieldList):
                        if field.data_type == "scalar":
                            self.boundaryValues[patch][i, :] = field.boundaryField[
                                patch
                            ][value_type]
                        elif field.data_type == "vector":
                            self.boundaryValues[patch][i, :] = field.boundaryField[
                                patch
                            ][value_type]

                elif value_shape == 2:
                    num_points: int = (
                        self.fieldList[0].boundaryField[patch][value_type].shape[0]
                    )
                    num_components: int = (
                        1 if self.fieldList[0].data_type == "scalar" else 3
                    )
                    self.boundaryValues[patch] = np.zeros(
                        (len(fieldList), num_points * num_components)
                    )

                    for i, field in enumerate(fieldList):
                        if field.data_type == "scalar":
                            self.boundaryValues[patch][i, :] = field.boundaryField[
                                patch
                            ][value_type].flatten()
                        elif field.data_type == "vector":
                            self.boundaryValues[patch][i, :] = field.boundaryField[
                                patch
                            ][value_type].T.flatten()

                # The mode equals self.coeffs.inverse() @ self.boundaryModes[patch]
                self.boundaryValues[patch] = (
                    np.linalg.inv(self._coeffs) @ self.boundaryValues[patch]
                )

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
