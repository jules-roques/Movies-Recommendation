from src.matrix_completion_methods.iterative_pca import IterativePCA
import numpy as np


def main():

    train_matrix = np.load("data/ratings_train.npy")
    valid_matrix = np.load("data/ratings_test.npy")

    method = IterativePCA(k = 1)

    method.fit(
        n_iter=50,
        verbose=True,
        raw_train_matrix=train_matrix,
        raw_valid_matrix=valid_matrix,
    )


if __name__ == "__main__":
    main()
