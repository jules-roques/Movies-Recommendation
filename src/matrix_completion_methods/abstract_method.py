"""Abstract class for methods"""

from abc import ABC, abstractmethod
import numpy as np


class MatrixCompletionMethod(ABC):
    """
    Abstract base class for matrix completion methods.

    This class provides a common interface for different matrix completion algorithms
    while ensuring consistency with the project's requirements.
    """

    def __init__(self, **kwargs):
        """
        Initialize the matrix completion method.

        Args:
            **kwargs: Method-specific parameters
        """
        self.is_fitted = False
        self.original_shape = None
        self.params = kwargs

    @abstractmethod
    def fit(self, initial_matrix: np.ndarray) -> None:
        """
        Fit the matrix completion method to the data.

        Args:
            initial_matrix: Input matrix with NaN values for missing entries

        """
        pass

    @abstractmethod
    def complete(self) -> np.ndarray:
        """
        Complete the missing values in the matrix used for fitting.

        Returns:
            np.ndarray: Completed matrix with no NaN values
        """
        pass
