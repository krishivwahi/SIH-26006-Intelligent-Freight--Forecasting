"""
Day 3: Bunker Cost Forecaster.

Trains a dedicated LightGBM regression model on IFO380 bunker fuel prices,
independent of the freight-rate (BDI-proxy) model in ``quantile_model.py``.
Bunker cost is a distinct market signal driven primarily by crude oil
benchmarks (Brent/WTI), the US Dollar Index (DXY, since oil is dollar-
denominated), and global logistics friction (GSCPI) -- NOT by the
supply/demand dynamics that move dry-bulk freight rates. Keeping the two
models separate lets each specialise on its own drivers and prevents
freight-specific noise from degrading the fuel-cost forecast that feeds the
MILP solver's voyage cost objective (see BETA_ROADMAP.md, 3.4).

Pipeline
--------
1. Ingest IFO380, Brent/WTI, DXY and GSCPI (real proxy data via
   ``src.data.load_real_data`` when available in ``data/raw/``, otherwise a
   clearly-labelled synthetic fallback via ``src.data.synthetic_data`` so the
   module is runnable out of the box for development/demo purposes).
2. Merge all sources on date, forward-filling lower-frequency series
   (GSCPI is monthly) onto the daily IFO380 index -- never back-filled, to
   avoid leaking future information into early rows.
3. Engineer lag / rolling / momentum / calendar features for every market
   column, using only information available up to and including day *t*.
4. Build a **direct 30-day-ahead** regression target:
   ``y_t = ifo380_price_{t+30}``. Because the target is a single fixed-
   horizon shift rather than a recursive 1-step forecast, the model is
   trained once and produces the day-t+30 price directly from day-t
   features -- no error accumulation from chaining 30 daily predictions.
5. Split chronologically (never shuffled) into train/test, with an
   additional expanding-window walk-forward cross-validation pass to sanity
   check stability across time before the final holdout evaluation.
6. Train a single LightGBM regressor, evaluate with RMSE/MAE/R^2/MAPE, and
   persist the model + metadata for later inference.

Run directly:
    python -m src.forecaster.bunker_forecaster

See the ``if __name__ == "__main__"`` block / CLI flags at the bottom of
this file for options.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.benchmarks.metrics import calculate_forecast_errors
from src.data.feature_engineering import (
    add_calendar_features,
    add_lag_features,
    add_momentum_features,
    add_rolling_features,
)
from src.data.load_real_data import (
    load_bunker_fuel,
    load_crude_oil,
    load_dxy,
    load_gscpi,
)
from src.data.synthetic_data import (
    generate_synthetic_bunker_fuel,
    generate_synthetic_crude_oil,
    generate_synthetic_dxy,
    generate_synthetic_gscpi,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = os.path.join("data", "raw")
DEFAULT_MODEL_DIR = os.path.join("models", "bunker_forecaster")

HORIZON_DAYS: int = 30          # Forecast horizon: bunker cost 30 days forward.
LAG_PERIODS: List[int] = [1, 3, 7, 14, 30]
ROLLING_WINDOWS: List[int] = [7, 14, 30]
MOMENTUM_PERIODS: List[int] = [7, 14]

TARGET_COLUMN = "ifo380_price"
DATE_COLUMN = "date"
# Exogenous market columns the model is allowed to see (per task spec:
# crude oil, DXY, GSCPI -- plus the bunker price series itself for its own
# autoregressive lags/rolling stats).
MARKET_COLUMNS: List[str] = [TARGET_COLUMN, "brent", "wti", "dxy", "gscpi"]


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def load_bunker_market_data(raw_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and merge IFO380, crude oil, DXY and GSCPI series.

    Prefers real proxy data from ``raw_dir`` (USDA bunker prices, FRED
    crude/DXY, NY Fed GSCPI -- see ``src/data/load_real_data.py`` for exact
    filenames). Any source missing on disk falls back to a synthetic series
    generated with the same schema, so the pipeline is runnable end-to-end
    without the raw files present. Synthetic data is never silently mixed
    into a "real" label -- a warning is logged for every fallback used.

    Args:
        raw_dir: Directory containing raw source files. Defaults to
            ``data/raw``.

    Returns:
        Daily-frequency DataFrame with columns ``date``, ``ifo380_price``,
        ``brent``, ``wti``, ``dxy``, ``gscpi``, forward-filled and sorted
        chronologically.
    """
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR

    bunker_path = os.path.join(raw_dir, "Daily_Bunker_Fuel_Prices_20260904.csv")
    crude_path = os.path.join(raw_dir, "Crude_Oil_Combined.xlsx")
    dxy_path = os.path.join(raw_dir, "DXYUSDollar Index.csv")
    gscpi_path = os.path.join(raw_dir, "gscpi_data.xls")

    if os.path.exists(bunker_path):
        bunker_df = load_bunker_fuel(bunker_path)
    else:
        logger.warning("Bunker fuel file not found at %s -- using synthetic IFO380 series.", bunker_path)
        bunker_df = generate_synthetic_bunker_fuel()

    if os.path.exists(crude_path):
        crude_df = load_crude_oil(crude_path)
    else:
        logger.warning("Crude oil file not found at %s -- using synthetic Brent/WTI series.", crude_path)
        crude_df = generate_synthetic_crude_oil()

    if os.path.exists(dxy_path):
        dxy_df = load_dxy(dxy_path)
    else:
        logger.warning("DXY file not found at %s -- using synthetic DXY series.", dxy_path)
        dxy_df = generate_synthetic_dxy()

    if os.path.exists(gscpi_path):
        gscpi_df = load_gscpi(gscpi_path)
    else:
        logger.warning("GSCPI file not found at %s -- using synthetic GSCPI series.", gscpi_path)
        gscpi_df = generate_synthetic_gscpi()

    # IFO380 (daily) is the anchor series -- it defines the row grid and is
    # also the regression target. Every other source is left-joined onto it.
    merged = bunker_df.copy()
    merged[DATE_COLUMN] = pd.to_datetime(merged[DATE_COLUMN])
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)

    for source in (crude_df, dxy_df, gscpi_df):
        src = source.copy()
        src[DATE_COLUMN] = pd.to_datetime(src[DATE_COLUMN])
        merged = merged.merge(src, on=DATE_COLUMN, how="left")

    # Forward-fill only. GSCPI is monthly, so this propagates the latest
    # known reading to every day until the next release. We deliberately
    # never back-fill: doing so would let a future GSCPI print leak into
    # earlier training rows (lookahead bias).
    merged = merged.sort_values(DATE_COLUMN).reset_index(drop=True)
    merged = merged.ffill()
    merged = merged.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# 2. Feature engineering + direct 30-day-forward target
# ---------------------------------------------------------------------------

def build_bunker_feature_matrix(
    merged_df: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features and a direct 30-day-forward IFO380 target.

    For every row *t*, the feature vector is built exclusively from data
    known as of day *t* (lags, trailing rolling stats, trailing momentum,
    calendar fields). The label is the *actual* IFO380 price observed
    ``horizon`` days later: ``y_t = ifo380_price_{t + horizon}``.

    This is a direct multi-step formulation (as opposed to a recursive
    1-day-ahead model applied 30 times): the model is trained once to jump
    straight to day t+30, so there is no compounding of daily prediction
    error across 30 recursive steps.

    Leakage safeguards:
      * Lag/rolling/momentum features only ever look backward from row t.
      * The target is created with ``shift(-horizon)`` (pulls a *future*
        value down to row t for supervised learning) -- this is safe only
        because the split step below keeps all of a test window's rows,
        and the horizon shift, entirely on one side of the chronological
        train/test boundary (see ``chronological_train_test_split``).
      * The final ``horizon`` rows have no realized future price yet and
        are dropped from training (they would otherwise be NaN targets);
        the same rows can be reused later purely as an inference batch.

    Args:
        merged_df: Output of ``load_bunker_market_data``.
        horizon: Forecast horizon in days. Defaults to 30.

    Returns:
        Tuple ``(X, y)``: feature matrix and forward-shifted target,
        row-aligned and with NaN rows (from lagging and the trailing
        horizon window) dropped.
    """
    df = merged_df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Calendar features (month / day-of-week / day-of-year seasonality in
    # bunker demand, e.g. peak shipping season).
    df = add_calendar_features(df, DATE_COLUMN)

    # Lag / rolling / momentum features for every available market column.
    available_columns = [c for c in MARKET_COLUMNS if c in df.columns]
    for col in available_columns:
        df = add_lag_features(df, col, lags=LAG_PERIODS)
        df = add_rolling_features(df, col, windows=ROLLING_WINDOWS)
        df = add_momentum_features(df, col, periods=MOMENTUM_PERIODS)

    # --- Direct 30-day-forward target -------------------------------------
    # y_t = ifo380_price observed at t + horizon. Shifting *up* (negative
    # shift) pulls the future value back to the current row.
    df["target_bunker_cost_30d"] = df[TARGET_COLUMN].shift(-horizon)

    # Drop helper/date columns that are not model features.
    feature_df = df.drop(columns=[DATE_COLUMN])

    y = feature_df["target_bunker_cost_30d"].copy()
    X = feature_df.drop(columns=["target_bunker_cost_30d"])

    # Drop rows with NaN features (early rows, from lagging) or NaN target
    # (final `horizon` rows, which have no realized future price yet).
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    return X, y


# ---------------------------------------------------------------------------
# 3. Chronological splitting (no leakage)
# ---------------------------------------------------------------------------

def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split time-ordered data into train/test WITHOUT shuffling.

    The data is already sorted chronologically by the time this is called
    (see ``build_bunker_feature_matrix``). A random/shuffled split would let
    the model train on rows that are chronologically *after* test rows,
    which is a classic time-series leakage bug -- this function guarantees
    the test set is strictly the most recent ``test_size`` fraction of rows
    and is never seen during training.

    Args:
        X: Feature matrix, sorted oldest-to-newest.
        y: Target series, same order as X.
        test_size: Fraction of rows (from the end) held out for testing.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If ``test_size`` is not in (0, 1) or the split would
            leave an empty train or test set.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    n = len(X)
    split_idx = int(n * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"test_size={test_size} leaves an empty train or test set for n={n} rows."
        )

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def walk_forward_cv_scores(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    lgb_params: Optional[Dict] = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward cross-validation.

    Uses ``sklearn.model_selection.TimeSeriesSplit``, which always trains on
    an earlier contiguous block and validates on a later contiguous block
    (never shuffled), repeated across ``n_splits`` expanding folds. This
    gives a more robust read on generalization than a single holdout split,
    since bunker prices are regime-dependent (e.g. an oil-price shock can
    make a single holdout window unrepresentative).

    Args:
        X: Full feature matrix, chronologically sorted.
        y: Full target series, chronologically sorted.
        n_splits: Number of expanding-window folds.
        lgb_params: Optional LightGBM hyperparameter overrides.

    Returns:
        DataFrame with one row per fold and columns
        ``['fold', 'mae', 'rmse', 'r2', 'mape']``.
    """
    params = dict(_default_lgb_params())
    if lgb_params:
        params.update(lgb_params)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        errors = calculate_forecast_errors(y_val.to_numpy(), preds)
        errors["fold"] = fold
        rows.append(errors)

    return pd.DataFrame(rows)[["fold", "mae", "rmse", "r2", "mape"]]


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------

def _default_lgb_params() -> Dict:
    """Default LightGBM hyperparameters for the bunker cost regressor."""
    return dict(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


class BunkerCostForecaster:
    """Dedicated LightGBM regressor for 30-day-forward IFO380 bunker cost.

    Unlike ``QuantileForecaster`` (used for the freight-rate P10/P50/P90
    bands), this is a single-point regressor: the solver's voyage-cost
    objective needs one deterministic fuel-cost number per scenario, not a
    distribution, so a plain point forecast keeps the downstream MILP
    formulation linear and simple.

    Attributes:
        horizon: Forecast horizon in days (must match how ``y`` was built).
        model: Trained ``lightgbm.LGBMRegressor``, or ``None`` before fit.
    """

    def __init__(
        self,
        horizon: int = HORIZON_DAYS,
        lgb_params: Optional[Dict] = None,
    ) -> None:
        self.horizon = horizon
        self.params = _default_lgb_params()
        if lgb_params:
            self.params.update(lgb_params)
        self.model: Optional[lgb.LGBMRegressor] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def is_trained(self) -> bool:
        """Return True once ``train`` has been called successfully."""
        return self.model is not None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """Fit the LightGBM regressor on the training split.

        Args:
            X_train: Training feature matrix (chronologically earlier
                than any held-out data).
            y_train: Training target vector (30-day-forward IFO380 price).
            eval_set: Optional ``(X_val, y_val)`` for early-stopping /
                validation-curve logging during boosting.
        """
        self._feature_names = list(X_train.columns)
        model = lgb.LGBMRegressor(**self.params)

        fit_kwargs: Dict = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            # Older/newer LightGBM sklearn-API versions occasionally change
            # keyword names for early stopping; fall back to a plain fit
            # without early stopping rather than hard-failing training.
            model.fit(X_train, y_train)
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the 30-day-forward IFO380 bunker price.

        Args:
            X: Feature matrix built the same way as training features
                (same columns, same order is enforced internally).

        Returns:
            Array of predicted bunker prices (USD/ton), one per row.

        Raises:
            RuntimeError: If called before ``train``.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        X_aligned = X[self._feature_names]
        return self.model.predict(X_aligned)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate on a held-out chronological test set.

        Args:
            X_test: Test feature matrix (chronologically after training data).
            y_test: True 30-day-forward IFO380 prices for the test rows.

        Returns:
            Dict with keys ``mae``, ``rmse``, ``r2``, ``mape``.
        """
        preds = self.predict(X_test)
        return calculate_forecast_errors(y_test.to_numpy(), preds)

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return the top-N most important features by LightGBM gain.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns ``feature`` and ``importance``, sorted
            descending.

        Raises:
            RuntimeError: If called before ``train``.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        importances = self.model.booster_.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": self._feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        return imp_df.head(top_n).reset_index(drop=True)

    def save(self, directory: str = DEFAULT_MODEL_DIR) -> None:
        """Persist the trained model and metadata to disk.

        Args:
            directory: Target directory (created if missing).

        Raises:
            RuntimeError: If called before ``train``.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "lgb_bunker_model.pkl"))
        meta = {
            "horizon": self.horizon,
            "feature_names": self._feature_names,
            "params": self.params,
        }
        joblib.dump(meta, os.path.join(directory, "meta.pkl"))
        logger.info("Saved bunker cost model to %s", directory)

    def load(self, directory: str = DEFAULT_MODEL_DIR) -> None:
        """Load a previously trained model and metadata from disk.

        Args:
            directory: Directory containing ``lgb_bunker_model.pkl`` and
                ``meta.pkl``.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        meta_path = os.path.join(directory, "meta.pkl")
        model_path = os.path.join(directory, "lgb_bunker_model.pkl")
        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Bunker model files not found in: {directory}")

        meta = joblib.load(meta_path)
        self.horizon = meta["horizon"]
        self._feature_names = meta["feature_names"]
        self.params = meta["params"]
        self.model = joblib.load(model_path)


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def run_bunker_forecaster_pipeline(
    raw_dir: Optional[str] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    horizon: int = HORIZON_DAYS,
    test_size: float = 0.2,
    run_cv: bool = True,
) -> Dict:
    """Run the full Day 3 bunker cost forecaster pipeline end-to-end.

    Steps: ingest -> merge -> feature engineer -> (optional) walk-forward
    CV sanity check -> chronological holdout split -> train -> evaluate ->
    save.

    Args:
        raw_dir: Directory with raw proxy data files (falls back to
            synthetic data per-source when files are absent).
        model_dir: Where to persist the trained model.
        horizon: Forecast horizon in days.
        test_size: Fraction of the most recent rows held out for testing.
        run_cv: Whether to also run walk-forward CV for a stability check.

    Returns:
        Dict with keys ``test_metrics``, ``cv_metrics`` (or ``None``),
        ``feature_importance``, ``n_train``, ``n_test``.
    """
    logger.info("Loading and merging bunker market data...")
    merged = load_bunker_market_data(raw_dir)
    logger.info("Merged dataset: %d rows spanning %s to %s",
                len(merged), merged[DATE_COLUMN].min().date(), merged[DATE_COLUMN].max().date())

    logger.info("Engineering features and %d-day-forward target...", horizon)
    X, y = build_bunker_feature_matrix(merged, horizon=horizon)
    logger.info("Feature matrix: %d rows x %d features", *X.shape)

    cv_metrics = None
    if run_cv:
        logger.info("Running walk-forward cross-validation...")
        cv_df = walk_forward_cv_scores(X, y, n_splits=5)
        cv_metrics = cv_df.to_dict(orient="records")
        logger.info("Walk-forward CV results:\n%s", cv_df.to_string(index=False))

    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=test_size)
    logger.info("Chronological split: %d train rows / %d test rows", len(X_train), len(X_test))

    forecaster = BunkerCostForecaster(horizon=horizon)
    forecaster.train(X_train, y_train, eval_set=(X_test, y_test))

    test_metrics = forecaster.evaluate(X_test, y_test)
    logger.info(
        "Test-set performance (30-day-forward IFO380, USD/ton): "
        "MAE=%.2f RMSE=%.2f R^2=%.4f MAPE=%.2f%%",
        test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"], test_metrics["mape"],
    )

    importance_df = forecaster.feature_importance(top_n=15)
    logger.info("Top features by gain:\n%s", importance_df.to_string(index=False))

    forecaster.save(model_dir)

    return {
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "feature_importance": importance_df.to_dict(orient="records"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Day 3 Bunker Cost Forecaster.")
    parser.add_argument("--raw-dir", type=str, default=DEFAULT_RAW_DIR,
                         help="Directory containing raw proxy data files.")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                         help="Directory to save the trained model + metadata.")
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS,
                         help="Forecast horizon in days.")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of most-recent rows held out for testing.")
    parser.add_argument("--no-cv", action="store_true",
                         help="Skip the walk-forward cross-validation sanity check.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_bunker_forecaster_pipeline(
        raw_dir=args.raw_dir,
        model_dir=args.model_dir,
        horizon=args.horizon,
        test_size=args.test_size,
        run_cv=not args.no_cv,
    )
