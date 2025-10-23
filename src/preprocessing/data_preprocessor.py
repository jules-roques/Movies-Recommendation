"""Data preprocessing utilities for matrix completion"""

import numpy as np


class DataPreprocessor:
    """
    Data preprocessor for matrix completion tasks.

    """

    def __init__(self, method: str = "user_mean", **kwargs):
        """
        Initialize the data preprocessor.

        Args:
            **kwargs: Additional parameters for preprocessing
        """
        self.params = kwargs
        self.user_means = None
        self.user_std = None
        self.random_state = None

        self.method = method
        if self.method == "user_mean":
            self.axis = 1  # Operations per row
        elif self.method == "movie_mean":
            self.axis = 0
        else:
            raise ValueError("Incorrect or unknown method")

    def fusion(self, train_matrix: np.ndarray, test_matrix: np.ndarray):
        """
        Combine train and test rating matrices into a single matrix.

        Args:
            train_matrix (np.ndarray): Training rating matrix with NaNs for missing values.
            test_matrix (np.ndarray): Test rating matrix with NaNs for missing values.
        """
        mask_train = ~np.isnan(train_matrix)
        mask_test = ~np.isnan(test_matrix)
        ratings = np.copy(train_matrix)
        intersection = np.sum(mask_train & mask_test)
        if intersection != 0:
            raise RuntimeError(
                f"Train/Test fusion error: {intersection} overlapping ratings detected."
            )

        ratings[mask_test] = test_matrix[mask_test]

        return ratings

    def normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Standardize ratings by centering and scaling each user's ratings.

        Args:
            matrix (np.ndarray): Matrix with NaN values for missing entries.

        Returns:
            np.ndarray: User-centered normalized matrix.
        """
        standardized = matrix.copy().astype(float)

        # Compute mean and std
        self.means = np.nanmean(standardized, axis=self.axis, keepdims=True)
        self.stds = np.nanstd(standardized, axis=self.axis, keepdims=True)

        # Replace NaN means with the mean of the other means
        overall_mean = np.nanmean(self.means)
        self.means[np.isnan(self.means)] = overall_mean

        # For users/items with no ratings or zero std, set std to mean of stds (ignoring NaNs and zeros)
        mean_std = np.nanmean(self.stds[self.stds != 0])
        self.stds[np.isnan(self.stds) | (self.stds == 0)] = mean_std

        assert not np.isnan(self.means).any()
        assert not np.isnan(self.stds).any()

        return (standardized - self.means) / self.stds

    def denormalize(self, matrix_standardized: np.ndarray) -> np.ndarray:
        """
        Restore absolute rating scale by adding back user means.

        Args:
            matrix_centered (np.ndarray): Matrix centered by user means.

        Returns:
            np.ndarray: Reconstructed matrix in original rating scale.
        """
        if self.means is None:
            raise ValueError(
                "User means and stds are not computed. Run normalize_by_user() first."
            )

        return matrix_standardized * self.stds + self.means

    def filter_by_threshold(
        self, matrix: np.ndarray, min_ratings_user: int = 5, min_ratings_movies: int = 5
    ):
        """
        Remove users and/or movies with fewer ratings than the threshold.

        Args:
            matrix (np.ndarray): User-item rating matrix with NaNs.
            min_ratings_user (int): Minimum number of ratings required per user.
            min_ratings_item (int): Minimum number of ratings required per item.

        Returns:
            np.ndarray: Filtered matrix.
            tuple: (kept_user_indices, kept_item_indices)
        """
        user_counts = np.sum(~np.isnan(matrix), axis=1)
        movies_counts = np.sum(~np.isnan(matrix), axis=0)

        keep_users = user_counts >= min_ratings_user
        keep_movies = movies_counts >= min_ratings_movies

        filtered_matrix = matrix[np.ix_(keep_users, keep_movies)]

        return filtered_matrix, np.where(keep_users)[0], np.where(keep_movies)[0]

    def preprocess(
        self, matrix: np.ndarray, min_ratings_user=0, min_ratings_item=0
    ) -> np.ndarray:
        """
        Initialize the data preprocessor.

        Args:
            matrix: Input matrix with NaN values for missing entries

        Returns:
            np.ndarray: Preprocessed matrix
        """

        matrix, _, _ = self.filter_by_threshold(
            matrix, min_ratings_user, min_ratings_item
        )

        matrix = self.normalize_by_user(matrix)

        return matrix

    def split(self, ratings: np.ndarray, test_size: float = 0.2):
        """
        Split the rating matrix into train and test matrices.

        Args:
            ratings (np.ndarray): Original rating matrix (with NaN for missing values)

        Returns:
            train_matrix (np.ndarray), test_matrix (np.ndarray)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Récupérer les positions non-NaN
        users, items = np.where(~np.isnan(ratings))
        n_ratings = len(users)
        if n_ratings == 0:
            raise ValueError("The input matrix has no ratings to split.")

        # Mélanger aléatoirement les indices
        indices = np.arange(n_ratings)
        np.random.shuffle(indices)

        # Déterminer la taille du test
        n_test = int(np.floor(test_size * n_ratings))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        # Construire les matrices vides
        train_matrix = np.full_like(ratings, np.nan, dtype=float)
        test_matrix = np.full_like(ratings, np.nan, dtype=float)

        # Remplir train
        train_users = users[train_idx]
        train_items = items[train_idx]
        train_values = ratings[train_users, train_items]
        train_matrix[train_users, train_items] = train_values

        # Remplir test
        test_users = users[test_idx]
        test_items = items[test_idx]
        test_values = ratings[test_users, test_items]
        test_matrix[test_users, test_items] = test_values

        # Vérification intersection
        overlap = np.sum(~np.isnan(train_matrix) & ~np.isnan(test_matrix))
        if overlap != 0:
            raise RuntimeError(
                f"Train/Test split error: {overlap} overlapping ratings detected."
            )

        n, m = train_matrix.shape[0], train_matrix.shape[1]
        mask_train = ~np.isnan(train_matrix)
        mask_test = ~np.isnan(test_matrix)
        print(
            f"Proportion of train ratings : {(np.sum(mask_train) / (n * m))*100:.2f}%"
        )
        print(f"Proportion of test ratings : {(np.sum(mask_test) / (n * m))*100:.2f}%")

        return train_matrix, test_matrix
