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
import json
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

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

def _next_version(model_dir: str) -> str:
    """Auto-increment model version: v001, v002, ..."""
    existing = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("v")
    ]
    if existing:
        latest_num = max(int(v[1:]) for v in existing)
        return f"v{latest_num + 1:03d}"
    return "v001"


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

    #  1. Ensure processed dataset exists 
    if data_path is None or not os.path.exists(data_path):
        data_path = DEFAULT_DATA_PATH
        if not os.path.exists(data_path):
            print("Processed market dataset not found. Running load_and_merge_all()...")
            load_and_merge_all(output_path=data_path)

    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    #  2. Build base feature matrix (calendar, seasonal, lags, rolling, momentum) 
    X, y = build_feature_matrix(df_raw, target_column="freight_rate", date_column="date")

    #  3. Walk-Forward Cross-Validation 
    # Replacing the naive 80/20 split with rigorous 3-fold Walk-Forward CV.
    print("Starting 3-fold Walk-Forward Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    cv_metrics = {"mae": [], "rmse": [], "da": [], "coverage": []}
    
    X_train_final = None
    X_test_final = None
    y_train_final = None
    y_test_final = None
    final_forecaster = None
    final_arima = None
    
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_tr, X_te = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_tr, y_te = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        
        # Fit ARIMA
        arima_m = AutoARIMABaseline(max_p=2, max_q=2, max_d=1, random_state=random_state)
        arima_m.fit(y_tr)
        X_tr["arima_pred"] = arima_m.predict_in_sample()[-len(X_tr):]
        try:
            X_te["arima_pred"] = arima_m.model_.predict(n_periods=len(X_te))
        except Exception:
            X_te["arima_pred"] = X_tr["arima_pred"].iloc[-1] if len(X_tr) else 0.0

        # Load optimized hyperparameters if available
        best_params_path = os.path.join(model_dir, "best_params.json")
        optuna_kwargs = {}
        if os.path.exists(best_params_path):
            with open(best_params_path, "r") as f:
                bp_data = json.load(f)
                if "best_params" in bp_data:
                    optuna_kwargs = bp_data["best_params"]
                    n_est = optuna_kwargs.pop("n_estimators", n_estimators)
                    lr = optuna_kwargs.pop("learning_rate", learning_rate)
        else:
            n_est = n_estimators
            lr = learning_rate

        # Train QuantileForecaster
        f_model = QuantileForecaster(
            horizon=30, n_estimators=n_est, learning_rate=lr, random_state=random_state, **optuna_kwargs
        )
        f_model.train(X_tr, y_tr)

        # Evaluate OOS metrics for the fold
        oos_p50_list = []
        oos_p10_list = []
        oos_p90_list = []
        step = max(1, len(X_te) // 30)
        for idx in range(0, len(X_te), step):
            row_pred = f_model.predict(X_te.iloc[[idx]])
            p50_h1 = float(row_pred[row_pred["horizon_step"] == 1]["p50"].iloc[0])
            p10_h1 = float(row_pred[row_pred["horizon_step"] == 1]["p10"].iloc[0])
            p90_h1 = float(row_pred[row_pred["horizon_step"] == 1]["p90"].iloc[0])
            oos_p50_list.append(p50_h1)
            oos_p10_list.append(p10_h1)
            oos_p90_list.append(p90_h1)

        sampled_indices = list(range(0, len(X_te), step))
        y_te_sampled = [float(y_te.iloc[i]) for i in sampled_indices]
        
        if len(oos_p50_list) >= 2:
            n = min(len(oos_p50_list), len(y_te_sampled))
            y_true = y_te_sampled[:n]
            preds = oos_p50_list[:n]
            err = calculate_forecast_errors(y_true, preds)
            da = calculate_directional_accuracy(y_true, preds)
            
            # Calibration check: Does P10-P90 contain the true value?
            in_bound = sum(1 for i in range(n) if oos_p10_list[i] <= y_true[i] <= oos_p90_list[i])
            coverage = in_bound / n * 100
            
            cv_metrics["mae"].append(err["mae"])
            cv_metrics["rmse"].append(err["rmse"])
            cv_metrics["da"].append(da)
            cv_metrics["coverage"].append(coverage)
            print(f"Fold {fold} - MAE: {err['mae']:.2f}, DA: {da:.1f}%, Coverage: {coverage:.1f}%")
        
        # Keep last fold as final for serialization
        if fold == tscv.get_n_splits():
            X_train_final, X_test_final = X_tr, X_te
            y_train_final, y_test_final = y_tr, y_te
            final_forecaster = f_model
            final_arima = arima_m
            
        fold += 1

    avg_mae = np.mean(cv_metrics["mae"])
    avg_rmse = np.mean(cv_metrics["rmse"])
    avg_da = np.mean(cv_metrics["da"])
    avg_cov = np.mean(cv_metrics["coverage"])
    print(f"\n--- Walk-Forward CV Averages ---")
    print(f"Avg MAE: {avg_mae:.2f} | Avg RMSE: {avg_rmse:.2f} | Avg DA: {avg_da:.1f}% | Avg P10-P90 Coverage: {avg_cov:.1f}%\n")
    
    oos_metrics = {
        "mae": round(avg_mae, 4),
        "rmse": round(avg_rmse, 4),
        "directional_accuracy": round(avg_da, 2),
        "coverage": round(avg_cov, 2)
    }

    version = _next_version(model_dir)
    versioned_dir = os.path.join(model_dir, version)
    os.makedirs(versioned_dir, exist_ok=True)

    arima_path = os.path.join(versioned_dir, "arima_baseline.joblib")
    final_arima.save(arima_path)

    model_save_path = os.path.join(versioned_dir, "meta.pkl")
    final_forecaster.save(versioned_dir)

    # Use the final fold's training/test sets for SHAP and forward forecasting
    X_train, X_test = X_train_final, X_test_final
    forecaster = final_forecaster

    # ── 6. Compute and export SHAP explainability artifact ─────────────────
    shap_json_path = os.path.join(versioned_dir, "shap_summary.json")
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

    #  7. Generate 30-day forward predictions & write contract JSON 
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

    # Write metadata for audit trail
    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_train.shape[1],
        "oos_metrics": oos_metrics,
        "arima_order": list(final_arima.get_order()),
        "lgb_params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        },
    }
    with open(os.path.join(versioned_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(model_dir, "latest_version.txt"), "w") as f:
        f.write(version)

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
