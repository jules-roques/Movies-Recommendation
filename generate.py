"""Main file of the project"""

import argparse

import numpy as np
from src.matrix_completion_methods import MatrixFactorisation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a completed ratings table.")
    parser.add_argument(
        "--name",
        type=str,
        default="ratings_eval.npy",
        help="Name of the npy of the ratings table to complete",
    )

    args = parser.parse_args()

    # Open Ratings table
    print("Ratings loading...")
    table = np.load(args.name)  # DO NOT CHANGE THIS LINE
    print("Ratings Loaded.")

    # Any method you want
    method = MatrixFactorisation(
        k=50,
        n_iter=125,
        learning_rate_I=0.001,
        learning_rate_U=0.001,
        lambda_reg=0.1,
        mu_reg=0.1,
        seed=42,
    )

    method.fit(initial_matrix=table, normalize=True)
    completed_table = method.complete()

    # Round the completed table to integer values
    completed_table = np.rint(completed_table)

    # Save the completed table
    out_name = "output.npy"
    np.save(out_name, completed_table)  # DO NOT CHANGE THIS LINE
    print(f'Completed ratings table saved to "{out_name}".')
