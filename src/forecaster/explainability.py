"""
SHAP explainability module for LightGBM dry bulk freight rate models.

Grounded in Paper 2 (feature importance, financial market drivers, and model auditability).
Provides:
1. TreeExplainer computation for LightGBM models.
2. Global feature importance ranking (mean |SHAP| values).
3. Local per-prediction feature contributions.
4. JSON export for fast, offline rendering in the Streamlit UI.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap


def compute_feature_importance_shap(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
) -> List[Tuple[str, float]]:
    """Compute global feature importance using mean absolute SHAP values.

    Args:
        model: Trained LightGBM regressor.
        X: Feature matrix DataFrame.
        max_samples: Maximum number of background samples for speed.

    Returns:
        Sorted list of tuples: (feature_name, mean_abs_shap_value) descending.
    """
    sample_X = X.iloc[:max_samples] if len(X) > max_samples else X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    features = list(X.columns)

    importance = [(feat, round(float(val), 4)) for feat, val in zip(features, mean_abs)]
    importance.sort(key=lambda item: item[1], reverse=True)
    return importance


def compute_prediction_attribution(
    model: Any,
    X_single_row: pd.DataFrame,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Compute local SHAP attribution for a single prediction row.

    Args:
        model: Trained LightGBM regressor.
        X_single_row: 1-row DataFrame of features.
        top_n: Number of top contributing features to return.

    Returns:
        Dict with base_value, prediction, and feature_contributions.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_val = float(explainer.expected_value)
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[0])

    contributions = []
    features = list(X_single_row.columns)
    row_values = shap_values[0]

    for feat, val in zip(features, row_values):
        feat_val = float(X_single_row[feat].iloc[0])
        contributions.append({
            "feature": feat,
            "feature_value": round(feat_val, 4),
            "shap_value": round(float(val), 4),
            "abs_shap": abs(float(val)),
        })

    contributions.sort(key=lambda c: c["abs_shap"], reverse=True)
    top_contribs = contributions[:top_n]

    pred_val = base_val + sum(c["shap_value"] for c in contributions)

    return {
        "base_value": round(base_val, 4),
        "predicted_value": round(pred_val, 4),
        "top_contributions": top_contribs,
    }


def export_shap_summary(
    model: Any,
    X: pd.DataFrame,
    latest_row: pd.DataFrame,
    output_path: str = "models/shap_summary.json",
    top_n: int = 10,
) -> str:
    """Generate and export a complete SHAP explainability artifact for the UI.

    Args:
        model: Trained LightGBM model.
        X: Background feature matrix.
        latest_row: Latest feature row used for current forecast.
        output_path: Destination JSON path.
        top_n: Number of top features to retain.

    Returns:
        Absolute path to the exported JSON.
    """
    global_imp = compute_feature_importance_shap(model, X, max_samples=200)
    local_attr = compute_prediction_attribution(model, latest_row, top_n=top_n)

    summary = {
        "global_importance": [
            {"feature": feat, "importance": score} for feat, score in global_imp[:top_n]
        ],
        "local_attribution": local_attr,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return os.path.abspath(output_path)
