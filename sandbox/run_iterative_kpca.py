from src.matrix_completion_methods.iterative_kernel_pca import IterativeKernelPCA
import numpy as np


def main():

    train_matrix = np.load("data/ratings_train.npy")
    valid_matrix = np.load("data/ratings_test.npy")

    method = IterativeKernelPCA()

    method.fit(
        n_iter=50,
        verbose=True,
        raw_train_matrix=train_matrix,
        raw_valid_matrix=valid_matrix,
    )


if __name__ == "__main__":
    main()
