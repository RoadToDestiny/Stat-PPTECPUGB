"""
Unit and integration tests for Backend 3 (ML pipeline, validation, target checks).
"""

import os
import pytest
import numpy as np
import pandas as pd

from bac3.src.dataset_schema import (
    validate_dataset_schema,
    verify_group_split_integrity,
    split_by_games,
    POSITION_FEATURES,
    ALL_REQUIRED_FEATURES,
)
from bac3.src.eval_target_checker import (
    compute_eval_loss_from_white_perspective,
    assign_error_label,
    filter_mate_positions,
    audit_features_for_leakage,
    verify_evaluation_feasibility,
)
from bac3.src.metrics import calculate_classification_metrics
from bac3.src.train import train_xgboost_model, save_model_artifacts
from bac3.src.evaluate import evaluate_on_test
from bac3.data.sample.generate_dummy_dataset import generate_synthetic_chess_data


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    return generate_synthetic_chess_data(num_games=20, random_seed=42)


def test_dataset_schema_validation(synthetic_data):
    is_valid, errors = validate_dataset_schema(synthetic_data)
    assert is_valid, f"Validation failed with errors: {errors}"


def test_schema_missing_column(synthetic_data):
    df_broken = synthetic_data.drop(columns=["eval_before"])
    is_valid, errors = validate_dataset_schema(df_broken)
    assert not is_valid
    assert any("eval_before" in err for err in errors)


def test_group_split_no_leakage(synthetic_data):
    train_df, val_df, test_df = split_by_games(synthetic_data, 0.7, 0.15, 0.15)
    is_clean, errors = verify_group_split_integrity(train_df, val_df, test_df)
    assert is_clean, f"Group split leaked games: {errors}"
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_loss_calculation_perspective():
    # White moves: eval drops from +150 to -60 -> loss is 210 cp
    loss_white = compute_eval_loss_from_white_perspective(150.0, -60.0, color='w')
    assert loss_white == 210.0

    # Black moves: eval from white POV changes from -50 to +180 -> White improved, so Black lost 230 cp
    loss_black = compute_eval_loss_from_white_perspective(-50.0, 180.0, color='b')
    assert loss_black == 230.0

    # White makes good move: eval increases from +100 to +200 -> loss is -100
    loss_white_good = compute_eval_loss_from_white_perspective(100.0, 200.0, color='w')
    assert loss_white_good == -100.0


def test_error_threshold_labeling():
    assert assign_error_label(199.0, threshold_cp=200.0) == 0
    assert assign_error_label(200.0, threshold_cp=200.0) == 1
    assert assign_error_label(350.0, threshold_cp=200.0) == 1
    # Sensitivity thresholds
    assert assign_error_label(160.0, threshold_cp=150.0) == 1
    assert assign_error_label(160.0, threshold_cp=200.0) == 0


def test_leakage_detector():
    clean_features = ["eval_before", "material_balance", "player_rating"]
    is_clean, violations = audit_features_for_leakage(clean_features)
    assert is_clean
    assert len(violations) == 0

    leaky_features = ["eval_before", "eval_loss", "is_error"]
    is_clean, violations = audit_features_for_leakage(leaky_features)
    assert not is_clean
    assert len(violations) == 2


def test_mate_filtering():
    df = pd.DataFrame({
        "game_id": ["g1", "g1", "g2"],
        "is_mate_eval": [False, True, False],
        "eval_loss": [10.0, 9999.0, 50.0]
    })
    filtered, stats = filter_mate_positions(df)
    assert len(filtered) == 2
    assert stats["excluded_mate_positions"] == 1


def test_smoke_pipeline_training_and_evaluation(synthetic_data, tmp_path):
    train_df, val_df, test_df = split_by_games(synthetic_data, 0.7, 0.15, 0.15)
    feature_cols = POSITION_FEATURES

    model, val_metrics = train_xgboost_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        model_params={"n_estimators": 10, "max_depth": 3}
    )

    assert "roc_auc" in val_metrics
    assert 0.0 <= val_metrics["roc_auc"] <= 1.0

    model_dir = str(tmp_path / "models")
    save_model_artifacts(model, model_dir, feature_cols, {"n_estimators": 10})
    assert os.path.exists(os.path.join(model_dir, "model.json"))

    results_dir = str(tmp_path / "results")
    test_metrics = evaluate_on_test(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        results_dir=results_dir
    )
    assert "roc_auc" in test_metrics
    assert os.path.exists(os.path.join(results_dir, "tables", "test_metrics.csv"))
    assert os.path.exists(os.path.join(results_dir, "plots", "calibration_curve.png"))
