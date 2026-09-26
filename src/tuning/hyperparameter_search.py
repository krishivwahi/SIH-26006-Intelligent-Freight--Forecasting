"""
Bayesian hyperparameter optimization for the LightGBM quantile forecaster.

Researcher 1 - Day 1
Uses Optuna TPE + expanding-window time-series CV
and minimizes average pinball loss across P10/P50/P90.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.feature_engineering import build_feature_matrix
from src.data.load_real_data import load_and_merge_all


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "merged_market_data.csv"
OUTPUT_PATH = ROOT / "models" / "best_params.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HORIZON = 30
ALPHAS = [0.1, 0.5, 0.9]

DEFAULT_TRIALS = 200
DEFAULT_SPLITS = 5

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Search space required by the Day 1 roadmap
# ---------------------------------------------------------------------

def suggest_parameters(trial: optuna.Trial) -> dict:
    """Return one LightGBM hyperparameter configuration."""

    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 1000
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 127
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 5, 100
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.5, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "lambda_l1": trial.suggest_float(
            "lambda_l1", 0.0, 10.0
        ),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 0.0, 10.0
        ),
    }


# ---------------------------------------------------------------------
# Pinball loss
# ---------------------------------------------------------------------

def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """Calculate quantile pinball loss."""

    error = y_true - y_pred

    return float(
        np.mean(
            np.maximum(
                alpha * error,
                (alpha - 1.0) * error,
            )
        )
    )


# ---------------------------------------------------------------------
# Prepare direct multi-step training data
# ---------------------------------------------------------------------

def prepare_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Match the existing QuantileForecaster training design.

    Each row is expanded across forecast horizons 1..30.
    """

    X_parts = []
    y_parts = []

    for h in range(1, horizon + 1):

        n_valid = len(X) - h

        if n_valid <= 0:
            continue

        X_h = X.iloc[:n_valid].copy()
        X_h = X_h.reset_index(drop=True)

        X_h["horizon_step"] = h

        y_h = (
            y.iloc[h:h + n_valid]
            .reset_index(drop=True)
        )

        X_parts.append(X_h)
        y_parts.append(y_h)

    if not X_parts:
        raise ValueError(
            "Not enough data to construct the "
            f"{horizon}-day forecasting dataset."
        )

    X_expanded = pd.concat(
        X_parts,
        ignore_index=True,
    )

    y_expanded = pd.concat(
        y_parts,
        ignore_index=True,
    )

    return X_expanded, y_expanded


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:

    data_path = DATA_PATH

    if not data_path.exists():

        print(
            "Processed dataset not found."
        )

        print(
            "Running load_and_merge_all()..."
        )

        load_and_merge_all(
            output_path=str(data_path)
        )

    df = pd.read_csv(data_path)

    if "date" not in df.columns:
        raise ValueError(
            "Dataset does not contain a 'date' column."
        )

    if "freight_rate" not in df.columns:
        raise ValueError(
            "Dataset does not contain "
            "'freight_rate'."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    X, y = build_feature_matrix(
        df,
        target_column="freight_rate",
        date_column="date",
    )

    # Remove rows containing invalid values.
    valid_mask = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid_mask].reset_index(
        drop=True
    )

    y = y.loc[valid_mask].reset_index(
        drop=True
    )

    print(
        f"Dataset loaded: {len(X)} rows"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    return X, y


# ---------------------------------------------------------------------
# Train one quantile model
# ---------------------------------------------------------------------

def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    params: dict,
) -> lgb.LGBMRegressor:

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,

        n_estimators=params[
            "n_estimators"
        ],

        learning_rate=params[
            "learning_rate"
        ],

        num_leaves=params[
            "num_leaves"
        ],

        min_child_samples=params[
            "min_child_samples"
        ],

        subsample=params[
            "subsample"
        ],

        bagging_freq=1,

        colsample_bytree=params[
            "colsample_bytree"
        ],

        reg_alpha=params[
            "lambda_l1"
        ],

        reg_lambda=params[
            "lambda_l2"
        ],

        random_state=RANDOM_STATE,

        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------

def create_objective(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
):

    tscv = TimeSeriesSplit(
        n_splits=n_splits
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:

        params = suggest_parameters(
            trial
        )

        fold_losses = []

        for fold, (
            train_idx,
            validation_idx,
        ) in enumerate(
            tscv.split(X),
            start=1,
        ):

            X_train = X.iloc[
                train_idx
            ].copy()

            y_train = y.iloc[
                train_idx
            ].copy()

            X_val = X.iloc[
                validation_idx
            ].copy()

            y_val = y.iloc[
                validation_idx
            ].copy()

            # We need at least one future observation
            # because horizon=1 predicts the next day.
            if len(X_val) < 2:
                continue

            # Match the project's direct multi-step
            # QuantileForecaster training approach.
            X_train_exp, y_train_exp = (
                prepare_training_data(
                    X_train,
                    y_train,
                    HORIZON,
                )
            )

            # Train P10, P50 and P90 models.
            models = {}

            for alpha in ALPHAS:

                models[alpha] = (
                    train_quantile_model(
                        X_train_exp,
                        y_train_exp,
                        alpha,
                        params,
                    )
                )

            # ---------------------------------------------------------
            # One-step-ahead validation
            #
            # X_val[i] predicts y_val[i+1]
            # ---------------------------------------------------------

            X_val_current = X_val.iloc[
                :-1
            ].copy()

            y_val_future = y_val.iloc[
                1:
            ].to_numpy()

            X_val_current[
                "horizon_step"
            ] = 1

            X_val_current = X_val_current[
                X_train_exp.columns
            ]

            for alpha in ALPHAS:

                predictions = (
                    models[alpha].predict(
                        X_val_current
                    )
                )

                loss = pinball_loss(
                    y_val_future,
                    predictions,
                    alpha,
                )

                fold_losses.append(
                    loss
                )

            trial.report(
                float(
                    np.mean(
                        fold_losses
                    )
                ),
                fold,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_losses:
            raise optuna.TrialPruned()

        return float(
            np.mean(
                fold_losses
            )
        )

    return objective


# ---------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------

def run_search(
    n_trials: int = DEFAULT_TRIALS,
    n_splits: int = DEFAULT_SPLITS,
) -> None:

    print("=" * 70)
    print(
        "DAY 1 - BAYESIAN HYPERPARAMETER OPTIMIZATION"
    )
    print("=" * 70)

    print(
        f"Trials: {n_trials}"
    )

    print(
        f"CV splits: {n_splits}"
    )

    print(
        "Sampler: Optuna TPE"
    )

    print(
        "Metric: Pinball Loss"
    )

    print(
        "Quantiles: P10 / P50 / P90"
    )

    print("=" * 70)

    X, y = load_dataset()

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="freight_rate_lgbm_bayesian_search",
    )

    objective = create_objective(
        X,
        y,
        n_splits,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # -------------------------------------------------------------
    # Save best parameters
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "study_name": study.study_name,
        "best_value": float(
            study.best_value
        ),
        "best_params": study.best_params,
        "n_trials": len(
            study.trials
        ),
        "cv_splits": n_splits,
        "quantiles": ALPHAS,
        "horizon": HORIZON,
        "random_state": RANDOM_STATE,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Best pinball loss: "
        f"{study.best_value:.6f}"
    )

    print()
    print("Best parameters:")

    for key, value in (
        study.best_params.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Bayesian optimization for "
            "LightGBM freight forecasting."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of Optuna trials.",
    )

    parser.add_argument(
        "--splits",
        type=int,
        default=DEFAULT_SPLITS,
        help="Number of expanding time-series CV splits.",
    )

    args = parser.parse_args()

    run_search(
        n_trials=args.trials,
        n_splits=args.splits,
    )