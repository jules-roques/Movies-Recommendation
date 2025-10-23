import numpy as np


def rmse(pred_matrix: np.ndarray, true_sparse_matrix: np.ndarray) -> float:
    """
    Compute RMSE between predicted and true rating matrices
    over the non-NaN entries of the true matrix.
    """
    mask = ~np.isnan(true_sparse_matrix)
    diff = pred_matrix[mask] - true_sparse_matrix[mask]
    return np.sqrt(np.mean(diff**2))


def accuracy(pred_matrix: np.ndarray, true_sparse_matrix: np.ndarray) -> float:
    """
    Compute Accuracy between predicted and true rating matrices
    over the non-NaN entries of the true matrix.
    """
    mask = ~np.isnan(true_sparse_matrix)
    pred_values = pred_matrix[mask]
    true_values = true_sparse_matrix[mask]
    correct = np.sum(pred_values == true_values)
    return correct / len(true_values)


def print_metrics(
    prefix: str,
    pred_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_matrix: np.ndarray,
) -> None:
    """
    Print RMSE and Accuracy for training and validation sets.
    """
    train_rmse = rmse(pred_matrix, train_matrix)
    train_acc = accuracy(pred_matrix, train_matrix)

    msg = f"{prefix} | Train RMSE: {train_rmse:.4f} | Train Acc: {train_acc:.4f}"

    if validation_matrix is not None:
        valid_rmse = rmse(pred_matrix, validation_matrix)
        valid_acc = accuracy(pred_matrix, validation_matrix)
        msg += f" | Valid RMSE: {valid_rmse:.4f} | Valid Acc: {valid_acc:.4f}"

    print(msg)
