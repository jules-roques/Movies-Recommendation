import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge


from .abstract_method import MatrixCompletionMethod
from src.preprocessing import DataPreprocessor

from src.metrics import print_metrics


class IterativeKernelPCA(MatrixCompletionMethod):
    """
    Implements an iterative Kernel PCA for matrix completion.

    This method handles missing data by iteratively imputing values and applying
    Kernel PCA. The key steps are:
    1.  Initialize missing values (e.g., with mean).
    2.  Apply Kernel PCA to get non-linear latent features.
    3.  Solve the "pre-image" problem by training a regression model (Ridge)
        to map the latent features back to the original data space.
    4.  Update the imputed values with the new predictions.
    5.  Repeat until convergence.
    """

    def __init__(
        self,
        k: int = 20,
        gamma: float = 0.1,
        alpha: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            valid_matrix (np.ndarray): A validation matrix for tracking performance.
            k (int): The number of principal components to keep.
            gamma (float): The gamma parameter for the RBF kernel.
            alpha (float): The regularization strength for the Ridge regression pre-image model.
        """
        super().__init__(**kwargs)

        self.data_preprocessor = DataPreprocessor()

        self.k = k
        self.gamma = gamma
        self.alpha = alpha

    def fit(
        self,
        raw_train_matrix: np.ndarray,
        raw_valid_matrix: np.ndarray = None,
        n_iter: int = 20,
        verbose: bool = False,
    ) -> None:

        initial_normalized_matrix = self.data_preprocessor.normalize(raw_train_matrix)
        X = self._compute_iterative_pca_initial_matrix(initial_normalized_matrix)

        for i in range(n_iter):

            X_kpca_features = self._perform_kpca(X)
            X_hat = self._perform_preimage_reconstruction(X_kpca_features, X)
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

        return self

    def complete(self) -> np.ndarray:
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

    def _perform_kpca(self, X: np.ndarray) -> np.ndarray:
        kpca = KernelPCA(
            n_components=self.k,
            kernel="rbf",
            gamma=self.gamma,
        )
        X_kpca_features = kpca.fit_transform(X)
        return X_kpca_features

    def _perform_preimage_reconstruction(
        self, X_kpca_features: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        pre_image_model = Ridge(alpha=self.alpha)
        pre_image_model.fit(X_kpca_features, X)
        X_hat = pre_image_model.predict(X_kpca_features)
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
