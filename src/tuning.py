import numpy as np
from preprocessing import DataPreprocessor
from matrix_completion_methods import MatrixFactorisation
from metrics import rmse
from sklearn.model_selection import KFold
from itertools import product
import os, csv


# ---------- Utilities ----------

def matrix_kfold(ratings, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.argwhere(~np.isnan(ratings))
    rng.shuffle(mask)

    
    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, val_idx in kf.split(mask):
        R_train = ratings.copy()
        R_val = ratings.copy()
        for i, j in mask[val_idx]:
            R_train[i, j] = np.nan
        for i, j in mask[train_idx]:
            R_val[i, j] = np.nan
        yield R_train, R_val

def _params_slug(algo: str, params: dict) -> str:
    def norm(v):
        if isinstance(v, float):
            return f"{v:.6g}".replace(".", "p")
        return str(v)
    keys = sorted(k for k in params.keys() if k != "fitting_algorithm")
    parts = [f"algo={algo}"] + [f"{k}={norm(params[k])}" for k in keys]
    return "__".join(parts)

def _save_historic_csv(log_dir: str, algo: str, params: dict, fold_id: int, historic: list[dict]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    slug = _params_slug(algo, params)
    path = os.path.join(log_dir, f"{slug}.csv")  # append all folds into one file
    write_header = not os.path.exists(path)
    hp_keys = sorted(params.keys())
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["fold", "epoch", "train_loss", "train_rmse", "train_acc", "val_loss", "val_rmse", "val_acc"]
                + [f"hp_{k}" for k in hp_keys]
            )
        for e in historic:
            train = e.get("train", {})
            val = e.get("val", {})
            row = [
                fold_id,
                e.get("epoch"),
                train.get("loss"),
                train.get("rmse"),
                train.get("acc"),
                val.get("loss"),
                val.get("rmse"),
                val.get("acc"),
            ] + [params[k] for k in hp_keys]
            w.writerow(row)

def cross_val_rmse(matrix, model_params, metric=rmse, n_splits=5, seed=42, verbose=False, log_dir="outputs/try_normalisation"):
    """Evaluate hyperparameters via KFold cross-validation."""
    scores = []
    for fold_id, (R_train, R_val) in enumerate(matrix_kfold(matrix, n_splits=n_splits, seed=seed), start=1):
        model = MatrixFactorisation(**model_params)
        model.fit(initial_matrix=R_train, valid_matrix=R_val, normalize=True, verbose=False)
        # save per-epoch curves
        _save_historic_csv(
            log_dir=log_dir,
            algo=model_params.get("fitting_algorithm", "gd"),
            params=model_params,
            fold_id=fold_id,
            historic=model.historic,
        )
        R_pred = model.complete()
        score = metric(R_pred, R_val)
        scores.append(score)
        if verbose:
            print(f"  Fold {fold_id}: RMSE = {score:.4f}")
    return float(np.mean(scores)), float(np.std(scores))


def grid_search_matrix_factorisation(
    matrix,
    param_grid,
    metric=rmse,
    n_splits=3,
    seed=42,
    verbose=True
):
    """
    Grid search + K-Fold CV for MatrixFactorisation over both ALS and GD.

    param_grid shape:
    {
      "als": {"k": [...], "lambda_reg": [...], "mu_reg": [...], "n_iter": [...]},
      "gd":  {"k": [...], "lambda_reg": [...], "mu_reg": [...], "n_iter": [...],
              "learning_rate_I": [...], "learning_rate_U": [...]}
    }
    """
    results = []
    best = None

    for algo, grid in param_grid.items():
        keys = list(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            algo_params = {k: v for k, v in zip(keys, values)}
            params = {
                "fitting_algorithm": algo,
                "seed": seed,
                **algo_params,
            }
            if verbose:
                print(f"\nTesting ({algo}): {algo_params}")
            mean_rmse, std_rmse = cross_val_rmse(
                matrix, params, metric=metric, n_splits=n_splits, seed=seed, verbose=False
            )
            results.append((algo, algo_params, mean_rmse, std_rmse))
            if verbose:
                print(f"→ Mean RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
            if best is None or mean_rmse < best[2]:
                best = (algo, params, mean_rmse)

    best_algo, best_params, best_rmse = best
    if verbose:
        print("\nBest configuration:")
        print(f"  algorithm: {best_algo}")
        for k, v in best_params.items():
            if k in ("fitting_algorithm", "seed"):
                continue
            print(f"  {k}: {v}")
        print(f"  RMSE = {best_rmse:.4f}")

    return best_algo, best_params, best_rmse, results


if __name__ == '__main__':
    # Open ratings
    print('Ratings loading...')
    ratings_train = np.load("data/ratings_train.npy")
    ratings_test = np.load("data/ratings_test.npy")
    print('Ratings Loaded.')

    # Settings
    test_size = 0.2
    n_splits = 3
    seed = 42

    # Pre-processing
    data_preprocessor = DataPreprocessor(method='user_mean')
    ratings = data_preprocessor.fusion(train_matrix=ratings_train, test_matrix=ratings_test)
    
    param_grid = {
        "gd": {
            "k": [1, 2, 5, 10, 20, 50],
            "lambda_reg": [0.1],
            "mu_reg": [0.1],
            "n_iter": [200],
            "learning_rate_I": [0.001],
            "learning_rate_U": [0.001],
        },
    }

    best_algo, best_params, best_rmse, all_results = grid_search_matrix_factorisation(
        matrix=ratings,
        param_grid=param_grid,
        metric=rmse,
        n_splits=n_splits,
        seed=seed,
        verbose=True,
    )

    print("\nBest params returned:", {"algorithm": best_algo, **best_params})