import numpy as np

from .abstract_method import MatrixCompletionMethod
from src.preprocessing import DataPreprocessor
from src.metrics import print_metrics


class IterativePCA(MatrixCompletionMethod):
    """
    Implements an iterative Principal Component Analysis (PCA) for matrix completion.

    Handles missing data by iteratively imputing values and applying SVD.
    Refactored to match the structure of IterativeKernelPCA.
    """

    def __init__(
        self,
        k: int = 20,
        **kwargs,
    ):
        """
        Args:
            k (int): The number of principal components (rank) to keep.
        """
        super().__init__(**kwargs)

        self.data_preprocessor = DataPreprocessor()

        self.k = k

    def fit(
        self,
        raw_train_matrix: np.ndarray,
        raw_valid_matrix: np.ndarray = None,
        n_iter: int = 20,
        verbose: bool = False,
    ) -> None:
        """
        Fits the Iterative PCA model to the training data.

        Args:
            raw_train_matrix (np.ndarray): The raw training matrix with NaN for missing values.
            raw_valid_matrix (np.ndarray): The raw validation matrix with NaN.
            n_iter (int): Number of iterations for the imputation loop.
            verbose (bool): If True, print metrics during training.
        """
        # Normalize the raw training data at the beginning
        # This normalized matrix contains NaNs where original data was missing
        initial_normalized_matrix = self.data_preprocessor.normalize(raw_train_matrix)

        # Create the initial dense matrix for iteration by filling NaNs with 0
        # (assuming normalization includes centering, so 0 is the mean)
        X = self._compute_iterative_pca_initial_matrix(initial_normalized_matrix)

        for i in range(n_iter):
            # Step 1: Perform SVD on the current dense matrix X
            U, s, Vt = self._perform_svd(X)
            X_hat = self._perform_low_rank_approx(U, s, Vt)
            X = self._update_known_values(X_hat, initial_normalized_matrix)

            if verbose:
                X_hat_denorm = self.data_preprocessor.denormalize(X_hat)
                print_metrics(
                    prefix=f"Iter {i+1}/{n_iter}",
                    pred_matrix=np.rint(X_hat_denorm),
                    validation_matrix=raw_valid_matrix,
                    train_matrix=raw_train_matrix,
                )

        self.completed_normalized_matrix = X
        self.is_fitted = True

    def complete(self) -> np.ndarray:
        """
        Returns the final completed matrix after denormalization and rounding.
        """
        if not self.is_fitted:
            raise RuntimeError("You must run fit() before calling complete().")

        return np.rint(
            self.data_preprocessor.denormalize(self.completed_normalized_matrix)
        )

    def _compute_iterative_pca_initial_matrix(
        self, initial_normalized_matrix: np.ndarray
    ) -> np.ndarray:
        X_init = initial_normalized_matrix.copy()
        missing_values_mask = np.isnan(initial_normalized_matrix)
        X_init[missing_values_mask] = 0
        return X_init

    def _perform_svd(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return U, s, Vt

    def _perform_low_rank_approx(
        self, U: np.ndarray, s: np.ndarray, Vt: np.ndarray
    ) -> np.ndarray:
        X_hat = U[:, : self.k] @ np.diag(s[: self.k]) @ Vt[: self.k, :]
        return X_hat

    def _update_known_values(
        self,
        X_hat: np.ndarray,
        initial_normalized_matrix: np.ndarray,
    ) -> np.ndarray:

        known_values_mask = ~np.isnan(initial_normalized_matrix)
        X_next = X_hat.copy()
        X_next[known_values_mask] = initial_normalized_matrix[known_values_mask]

        return X_next
