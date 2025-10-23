"""Matrix factorisation method using Alternating Least Squares (ALS)"""

from .abstract_method import MatrixCompletionMethod
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm
from src.preprocessing import DataPreprocessor
from typing import Optional


class MatrixFactorisation(MatrixCompletionMethod):
    """
    Matrix factorisation method using either Alternating Least Squares (ALS) or Gradient Descent.
    """

    def __init__(
        self,
        fitting_algorithm: str = "gd",
        init_method: str = "user_mean",
        k: int = 20,
        lambda_reg: float = 0.1,
        mu_reg: float = 0.1,
        n_iter: int = 20,
        learning_rate_I: float = 0.01,
        learning_rate_U: float = 0.01,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize the Matrix Factorisation method.

        Args:
            method: Method to use ('als' for Alternating Least Squares or 'gd' for Gradient Descent)
            k: Latent dimension (number of factors)
            lambda_reg: Regularization parameter for item factors
            mu_reg: Regularization parameter for user factors
            n_iter: Number of iterations
            learning_rate_I: Learning rate for item factors (ignored for ALS)
            learning_rate_U: Learning rate for user factors (ignored for ALS)
            seed: Random seed for initialization
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        if fitting_algorithm not in ["als", "gd"]:
            raise ValueError("Method must be either 'als' or 'gd'")
        self.fitting_algorithm = fitting_algorithm
        self.k = k
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.n_iter = n_iter
        self.learning_rate_I = learning_rate_I
        self.learning_rate_U = learning_rate_U
        self.seed = seed
        self.n_items = 0
        self.n_users = 0
        self.U = None  # User factors
        self.I = None  # Item factors
        self.raw_train_matrix = None
        self.mask = None  # Mask for observed entries
        self.valid_mask = None  # Mask for validation entries
        self.normalize = False

        self.data_preprocessor = DataPreprocessor(method=init_method)
        self.historic = []
        self.acc_tolerance: float = kwargs.pop("acc_tolerance", 0.5)

    def fit(
        self,
        raw_train_matrix: np.ndarray,
        normalize: bool,
        valid_matrix: Optional[np.ndarray] = None,
        init_from_pca: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Fit the matrix factorisation method using the specified algorithm.

        Args:
            raw_train_matrix: Input matrix with NaN values for missing entries
        """
        self.n_users = raw_train_matrix.shape[0]
        self.n_items = raw_train_matrix.shape[1]

        self.normalize = normalize

        self.ratings_train = raw_train_matrix.copy()
        self.ratings_val = valid_matrix.copy() if valid_matrix is not None else None

        self.raw_train_matrix = raw_train_matrix
        if normalize:
            self.raw_train_matrix = self.data_preprocessor.normalize(self.ratings_train)

        self.mask = ~np.isnan(raw_train_matrix)
        self.valid_mask = (
            (~np.isnan(valid_matrix)) if valid_matrix is not None else None
        )

        # Initialize factors
        self.U, self.I = self._initialize_factors(init_from_pca=init_from_pca)
        # Choose algorithm based on method
        if self.fitting_algorithm == "als":
            self._fit_als(raw_train_matrix=self.raw_train_matrix)
        elif self.fitting_algorithm == "gd":
            self._fit_gd(raw_train_matrix=self.raw_train_matrix, verbose=verbose)
        else:
            raise ValueError(f"Unknown method: {self.fitting_algorithm}")

        # self.is_fitted = True

    def _initialize_factors(
        self, init_from_pca: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize user and item factor matrices.

        Args:
            n_items: Number of items
            n_users: Number of users

        Returns:
            Tuple of (U, I) factor matrices
        """
        if init_from_pca is not None:
            U_pca, s_pca, Vt_pca = init_from_pca
            k = self.k

            S_root = np.diag(np.sqrt(s_pca[:k]))
            U = U_pca[:, :k] @ S_root
            I = Vt_pca[:k, :].T @ S_root
            print(f"Init from PCA: U={U.shape}, I={I.shape}")
            return U, I
        else:
            rng = np.random.default_rng(self.seed)
            U = 0.01 * rng.standard_normal((self.n_users, self.k))
            I = 0.01 * rng.standard_normal((self.n_items, self.k))
            print(f"Random init: U={U.shape}, I={I.shape}")
            return U, I

    def _fit_als(self, raw_train_matrix: np.ndarray, verbose=True) -> None:
        """
        Fit using Alternating Least Squares (ALS).

        Args:
            raw_train_matrix: Input matrix with NaN values for missing entries
        """
        # Convert NaN to 0 for sparse matrix operations
        R_filled: np.ndarray = np.nan_to_num(raw_train_matrix, nan=0)
        R_csr: sp.csr_matrix = sp.csr_matrix(R_filled)
        R_csc: sp.csc_matrix = sp.csc_matrix(R_filled)

        # Ensure factors are initialized (narrow types for type checkers)
        assert (
            self.U is not None and self.I is not None
        ), "U and I must be initialized before running ALS"

        self.historic = []
        # Perform ALS iterations
        for it in tqdm(range(self.n_iter), desc="ALS iterations"):
            hold_I = self.I
            self.I = self._update_item_factors(R_csc, self.U)
            self.U = self._update_user_factors(R_csr, hold_I)

            self._compute_metrics(iteration=it, verbose=verbose)

    def _update_item_factors(self, R_csc: sp.csc_matrix, U: np.ndarray) -> np.ndarray:
        """
        Update item matrix by solving independent ridge regressions for each item.

        Args:
            R_csc: Sparse item-by-user rating matrix
            U: Current user latent factors (shape (n_users, k))

        Returns:
            Updated item latent factors
        """
        I = np.zeros((self.n_items, self.k))
        lambdaI = self.lambda_reg * np.eye(self.k)

        for i in range(self.n_items):
            start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
            if start == end:
                continue
            user_idx = R_csc.indices[start:end]
            ratings = R_csc.data[start:end]

            U_i = U[user_idx, :]
            A = U_i.T @ U_i + lambdaI
            b = U_i.T @ ratings
            I[i, :] = self._solve_ridge(A, b)

        return I

    def _update_user_factors(self, R_csr: sp.csr_matrix, I: np.ndarray) -> np.ndarray:
        """
        Update user latent factors by solving independent ridge regressions for each user.

        Args:
            R_csc: Sparse user-by-item rating matrix (transpose of item-by-user)
            I: Current item latent factors (shape (n_users, k))

        Returns:
            Updated user latent factors
        """

        muI = self.mu_reg * np.eye(self.k)
        U = np.zeros((self.n_users, self.k))

        for u in range(self.n_users):
            start, end = R_csr.indptr[u], R_csr.indptr[u + 1]
            if start == end:
                continue
            item_idx = R_csr.indices[start:end]
            ratings = R_csr.data[start:end]

            I_i = I[item_idx, :]
            A = I_i.T @ I_i + muI
            b = I_i.T @ ratings
            U[u, :] = self._solve_ridge(A, b)

        return U

    def _solve_ridge(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the regularized least squares system A x = b
        using Cholesky decomposition (fast and stable).

        Args:
            A: Symmetric positive-definite system matrix
            b: Right-hand side vector

        Returns:
            Solution vector
        """
        facto = cho_factor(A)
        return cho_solve(facto, b)

    def _fit_gd(self, raw_train_matrix: np.ndarray, verbose: bool = True) -> None:
        """
        Fit using Gradient Descent.

        Args:
            raw_train_matrix: Input matrix with NaN values for missing entries
        """
        # Fill NaN with 0 for computation
        R_filled = np.nan_to_num(raw_train_matrix, nan=0)

        # Ensure factors are initialized (narrow types for type checkers)
        assert (
            self.U is not None and self.I is not None
        ), "U and I must be initialized before running ALS"

        self.historic = []

        # Perform gradient descent iterations
        for it in tqdm(range(self.n_iter), desc="Gradient Descent iterations"):
            # Compute gradients

            # Compute residual only for observed entries
            E = self.mask * (R_filled - self.U @ self.I.T)

            # Gradients
            grad_U = -2 * (E @ self.I) + 2 * self.mu_reg * self.U
            grad_I = -2 * (E.T @ self.U) + 2 * self.lambda_reg * self.I

            # Update
            self.U -= self.learning_rate_U * grad_U
            self.I -= self.learning_rate_I * grad_I

            self._compute_metrics(iteration=it, verbose=verbose)

    def _compute_metrics(self, iteration: int, verbose: bool = True) -> None:
        """Compute and log metrics for a given iteration."""

        # Reconstruction
        reconstruction = self.complete()

        # Train metrics on observed entries
        train_loss, train_rmse, train_acc = self._compute_split_metrics(
            self.ratings_train, self.mask, reconstruction
        )

        # Validation metrics if provided
        val_loss = val_rmse = val_acc = None
        if self.ratings_val is not None and self.valid_mask is not None:
            val_loss, val_rmse, val_acc = self._compute_split_metrics(
                self.ratings_val, self.valid_mask, reconstruction
            )

        # Log per-epoch metrics
        self.historic.append(
            {
                "epoch": iteration + 1,
                "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
                "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
            }
        )

        if verbose:
            loss_obj = self._compute_loss(
                R=np.nan_to_num(self.ratings_train, nan=0.0), mask=self.mask
            )
            tqdm.write(
                f"Iter {iteration + 1}/{self.n_iter} | "
                f"Train RMSE: {train_rmse:.4f}, Acc: {train_acc:.3f}, Loss: {train_loss:.2f}, Obj: {loss_obj:.2f}"
                + (
                    f" | Val RMSE: {val_rmse:.4f}, Acc: {val_acc:.3f}, Loss: {val_loss:.2f}"
                    if val_rmse is not None
                    else ""
                )
            )

    def _compute_loss(self, R: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute the cost function:
            C(U, I) = ||R - U I^T||_S^2 + mu||U||_F^2 + lambda||I||_F^2
        where ||.||_S^2 is over observed entries only.

        Args:
            R : np.ndarray
                Rating matrix (zeros for missing)
            mask : np.ndarray of bool
                True where rating is observed
            U, I : np.ndarray
                Latent factor matrices
            lambda_, mu : float
                Regularization parameters

        Returns:
            Total loss (reconstruction error + regularization)
        """
        # Predicted ratings
        R_pred = self.U @ self.I.T

        # Residual only for observed entries
        diff = mask * (R - R_pred)

        # Squared error on observed entries
        mse = np.sum(diff**2)

        # Regularization
        reg = self.mu_reg * np.sum(self.U**2) + self.lambda_reg * np.sum(self.I**2)

        return mse + reg

    # --- Metrics helpers ---
    def _compute_split_metrics(
        self, R: np.ndarray, mask: np.ndarray, recon: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute reconstruction loss (sum of squared error), RMSE, and tolerance accuracy on a split.
        R may contain NaN; mask must select valid observed entries.
        """
        if mask is None or not np.any(mask):
            return np.nan, np.nan, np.nan
        y_true = R[mask]
        y_pred = recon[mask]
        diff = y_true - y_pred
        se = diff**2
        loss = float(np.sum(se))
        rmse = float(np.sqrt(np.mean(se)))
        acc = float(np.mean(np.abs(diff) <= self.acc_tolerance))
        return loss, rmse, acc

    def complete(self) -> np.ndarray:
        """
        Complete the matrix using the learned factorization.

        Returns:
            Completed matrix with no NaN values
        """
        # if not self.is_fitted:
        #     raise ValueError("Method must be fitted before completing matrix")

        # Reconstruct the matrix using U @ I.T
        completed_matrix = self.U @ self.I.T
        if self.normalize:
            completed_matrix = self.data_preprocessor.denormalize(
                matrix_standardized=completed_matrix
            )
        return completed_matrix
