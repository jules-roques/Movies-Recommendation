from src.matrix_completion_methods.matrix_factorisation import MatrixFactorisation
import numpy as np


def main():

    train_matrix = np.load("data/ratings_train.npy")
    valid_matrix = np.load("data/ratings_test.npy")

    method = MatrixFactorisation(n_iter=50)

    method.fit(
        raw_train_matrix=train_matrix,
        valid_matrix=valid_matrix,
        verbose=True,
        normalize=True,
    )


if __name__ == "__main__":
    main()
