"""
Evaluation script to compute final test metrics, generate ROC/PR curves,
calibration diagrams, and feature importance rankings.
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from bac3.src.metrics import (
    calculate_classification_metrics,
    plot_calibration,
    plot_feature_importance,
)


def evaluate_on_test(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_error",
    results_dir: str = "bac3/results"
) -> Dict[str, float]:
    """
    Evaluates model on unseen test split and saves diagnostic artifacts.
    """
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)

    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy()

    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_prob)

    # Save metrics table
    metrics_df = pd.DataFrame([metrics])
    metrics_csv_path = os.path.join(results_dir, "tables", "test_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Calibration plot
    calib_plot_path = os.path.join(results_dir, "plots", "calibration_curve.png")
    plot_calibration(y_test, y_prob, save_path=calib_plot_path)

    # Feature importance plot
    importances = model.feature_importances_
    feat_plot_path = os.path.join(results_dir, "plots", "feature_importance.png")
    feat_df = plot_feature_importance(feature_cols, importances, save_path=feat_plot_path)
    feat_df.to_csv(os.path.join(results_dir, "tables", "feature_importance.csv"), index=False)

    return metrics
