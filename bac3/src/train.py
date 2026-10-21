"""
Model training pipeline for XGBoost blunder prediction.
Supports hyperparameter loading, group-based validation, class balancing, and artifact saving.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from bac3.src.dataset_schema import (
    POSITION_FEATURES,
    META_FEATURES,
    ALL_REQUIRED_FEATURES,
    validate_dataset_schema,
    split_by_games,
)
from bac3.src.metrics import calculate_classification_metrics


def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "is_error",
    model_params: Optional[Dict] = None
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Trains an XGBoost classifier with early stopping on validation split.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Calculate scale_pos_weight dynamically if not provided
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    default_spw = float(neg_count / max(1, pos_count))

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": default_spw,
        "eval_metric": "auc",
        "random_state": 42,
    }
    if model_params:
        params.update(model_params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    metrics = calculate_classification_metrics(y_val.to_numpy(), val_probs)

    return model, metrics


def save_model_artifacts(
    model: xgb.XGBClassifier,
    output_dir: str,
    feature_cols: List[str],
    hyperparams: Dict
) -> str:
    """
    Saves the trained model to JSON and records features and parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.json")
    model.save_model(model_path)

    metadata = {
        "features": feature_cols,
        "hyperparameters": hyperparams,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model_path
