"""
End-to-end model training, explainability, and forecast generation pipeline.

Orchestrates:
1. Ingestion and merging of real external proxy datasets.
2. Feature engineering (lags, rolling stats, momentum, seasonal calendar).
3. AutoARIMA statistical baseline fitting (Paper 1).
4. LightGBM multi-step direct quantile regression training (Paper 2 & 4).
5. SHAP TreeExplainer feature attribution export (Paper 2).
6. Forward-looking 30-day forecast generation matching CONTRACT.md (Option A).
"""
from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.feature_engineering import build_feature_matrix
from src.data.load_real_data import load_and_merge_all
from src.benchmarks.metrics import (
    calculate_directional_accuracy,
    calculate_forecast_errors,
)
from src.forecaster.baseline_arima import AutoARIMABaseline
from src.forecaster.explainability import export_shap_summary
from src.forecaster.quantile_model import QuantileForecaster
from src.forecaster.write_forecast import (
    BDI_TO_USD_PER_MT_SCALE,
    create_forecast_records,
    write_forecast_csv,
    write_forecast_excel,
    write_forecast_json,
)

DEFAULT_DATA_PATH = os.path.join("data", "processed", "merged_market_data.csv")
DEFAULT_MODEL_DIR = "models"
DEFAULT_FORECAST_PATH = os.path.join("data", "interim", "freight_forecast_30d.json")


def run_training_pipeline(
    data_path: Optional[str] = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    forecast_path: str = DEFAULT_FORECAST_PATH,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute the complete training, explainability, and forecast export pipeline.

    Args:
        data_path: Path to merged market CSV. If missing, runs load_and_merge_all().
        model_dir: Directory to save serialized model artifacts.
        forecast_path: Destination for contract JSON.
        n_estimators: LightGBM trees per quantile model.
        learning_rate: LightGBM learning rate.
        random_state: Reproducibility seed.

    Returns:
        Summary dict containing paths, metrics, and data shapes.
    """
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Ensure processed dataset exists ──────────────────────────────────
    if data_path is None or not os.path.exists(data_path):
        data_path = DEFAULT_DATA_PATH
        if not os.path.exists(data_path):
            print("Processed market dataset not found. Running load_and_merge_all()...")
            load_and_merge_all(output_path=data_path)

    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # ── 2. Build base feature matrix (calendar, seasonal, lags, rolling, momentum) ──
    X, y = build_feature_matrix(df_raw, target_column="freight_rate", date_column="date")

    # ── 3. Temporal 80/20 train / test split ────────────────────────────────
    # Splitting by time (not random) to avoid lookahead leakage.
    # The model trains on the first 80% of the dataset and is evaluated on
    # the final 20% it has never seen — giving out-of-sample (OOS) metrics.
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    print(f"Train: {len(X_train)} rows | Test (OOS): {len(X_test)} rows")

    # ── 4. Fit AutoARIMA baseline on TRAIN only, inject as bridge feature ───
    print("Fitting AutoARIMA baseline on train split...")
    arima_model = AutoARIMABaseline(max_p=2, max_q=2, max_d=1, random_state=random_state)
    arima_model.fit(y_train)
    arima_fitted_train = arima_model.predict_in_sample()

    # Inject in-sample ARIMA predictions as a bridge feature for LightGBM
    X_train["arima_pred"] = arima_fitted_train[-len(X_train):]

    # For the test set, produce ARIMA out-of-sample forecasts
    try:
        arima_oos = arima_model.model_.predict(n_periods=len(X_test))
        X_test["arima_pred"] = arima_oos
    except Exception:
        # Fallback: use the last known fitted value
        X_test["arima_pred"] = arima_fitted_train[-1] if len(arima_fitted_train) else 0.0

    arima_path = os.path.join(model_dir, "arima_baseline.joblib")
    arima_model.save(arima_path)

    # ── 5. Train LightGBM Multi-Step Quantile Regressors on TRAIN only ──────
    print(f"Training QuantileForecaster on {len(X_train)} samples with {X_train.shape[1]} features...")
    forecaster = QuantileForecaster(
        horizon=30,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    forecaster.train(X_train, y_train)

    model_save_path = os.path.join(model_dir, "quantile_forecaster.joblib")
    forecaster.save(model_save_path)

    # ── 5b. Compute Out-of-Sample (OOS) Evaluation Metrics ──────────────────
    print("Computing out-of-sample evaluation metrics on held-out 20%...")
    oos_preds_df = forecaster.predict(X_test.iloc[[0]])  # 1-step-ahead from test start
    # For OOS directional accuracy: use 1-step-ahead P50 predictions across the test set
    oos_p50_list = []
    step = max(1, len(X_test) // 30)  # sample ~30 points across test window
    for idx in range(0, len(X_test), step):
        row_pred = forecaster.predict(X_test.iloc[[idx]])
        p50_h1 = float(row_pred[row_pred["horizon_step"] == 1]["p50"].iloc[0])
        oos_p50_list.append(p50_h1)

    # Align true values to the sampled indices
    sampled_indices = list(range(0, len(X_test), step))
    y_test_sampled = [float(y_test.iloc[i]) for i in sampled_indices]

    oos_metrics: dict = {"mae": None, "rmse": None, "r2": None, "mape": None, "directional_accuracy": None}
    if len(oos_p50_list) >= 2 and len(y_test_sampled) >= 2:
        n = min(len(oos_p50_list), len(y_test_sampled))
        oos_metrics = calculate_forecast_errors(y_test_sampled[:n], oos_p50_list[:n])
        da = calculate_directional_accuracy(y_test_sampled[:n], oos_p50_list[:n])
        oos_metrics["directional_accuracy"] = round(da, 2)
        print(f"OOS MAE: {oos_metrics['mae']:.4f} | RMSE: {oos_metrics['rmse']:.4f} | DA: {da:.1f}%")

    # ── 6. Compute and export SHAP explainability artifact ─────────────────
    shap_json_path = os.path.join(model_dir, "shap_summary.json")
    latest_row = X_train.iloc[[-1]]  # use last training row for SHAP baseline

    # Extract the median (alpha=0.5) model for representative SHAP attributions
    median_model = forecaster.models.get(0.5)
    if median_model is not None:
        print("Computing SHAP feature attributions...")
        background_X = X_train.iloc[:200].copy()
        background_X["horizon_step"] = 1
        eval_row = latest_row.copy()
        eval_row["horizon_step"] = 1
        export_shap_summary(
            model=median_model,
            X=background_X,
            latest_row=eval_row,
            output_path=shap_json_path,
        )

    # ── 7. Generate 30-day forward predictions & write contract JSON ────────
    print("Generating 30-day forward quantile forecasts...")
    predictions = forecaster.predict(latest_row)

    # Convert BDI proxy points (>100) to $/MT spot freight rate if needed
    mean_p50 = float(predictions["p50"].mean()) if "p50" in predictions.columns else 1.0
    scale = BDI_TO_USD_PER_MT_SCALE if mean_p50 > 100.0 else 1.0
    if scale != 1.0:
        print(f"Applying BDI-to-USD/MT calibration factor: {scale} (mean raw index: {mean_p50:.1f})")

    # Base date: today's date for live operations, or latest observed date in dataset
    run_base_date = date.today()
    records = create_forecast_records(predictions, start_date=run_base_date, rate_scale=scale)

    written_path = write_forecast_json(records, filepath=forecast_path)
    excel_path = write_forecast_excel(records)
    csv_path = write_forecast_csv(records)
    print(f"Contract JSON successfully written to: {written_path}")
    print(f"Enriched Excel workbook written to: {excel_path}")
    print(f"CSV export written to: {csv_path}")

    return {
        "status": "Success",
        "model_path": os.path.abspath(model_save_path),
        "arima_path": os.path.abspath(arima_path),
        "shap_path": os.path.abspath(shap_json_path),
        "forecast_path": os.path.abspath(written_path),
        "excel_path": os.path.abspath(excel_path),
        "csv_path": os.path.abspath(csv_path),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_train.shape[1],
        "horizon_days": 30,
        "base_date": run_base_date.isoformat(),
        "rate_scale": scale,
        "oos_metrics": oos_metrics,
    }


if __name__ == "__main__":
    run_training_pipeline()
