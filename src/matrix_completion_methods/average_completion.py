"""Average matrix completion method"""

from .abstract_method import MatrixCompletionMethod
import numpy as np


class AverageCompletion(MatrixCompletionMethod):
    """
    Average completion method - replaces NaN values with global average.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matrix_average = None
        self.initial_matrix = None

    def fit(self, initial_matrix: np.ndarray) -> None:
        """
        Fit the average completion method.

        Args:
            initial_matrix: Input matrix with NaN values

        Returns:
        """
        self.initial_matrix = initial_matrix
        self.matrix_average = np.nanmean(self.initial_matrix)
        self.is_fitted = True

    def complete(self) -> np.ndarray:
        """
        Complete the matrix using average completion.

        Returns:
            np.ndarray: Completed matrix
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before completing matrix")

        return np.nan_to_num(self.initial_matrix, nan=self.matrix_average)
