"""
Module for verifying and computing evaluation metrics, loss calculations,
and target variable generation from Stockfish evaluations.
Ensures no target leakage into features and handles edge cases like mate scores.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


LEAKAGE_TARGET_PATTERNS = [
    "eval_after",
    "eval_loss",
    "is_error",
    "target",
    "label",
    "blunder",
    "blunder_threshold",
    "cp_loss",
]


def compute_eval_loss_from_white_perspective(
    eval_before_white: float,
    eval_after_white: float,
    color: Union[str, int]
) -> float:
    """
    Computes centipawn loss from the perspective of the player making the move,
    when raw Stockfish evaluations are provided relative to White.

    Parameters:
    - eval_before_white: Stockfish centipawn score before the move (from White's POV)
    - eval_after_white: Stockfish centipawn score after the move (from White's POV)
    - color: 'w' / 1 for White, 'b' / 0 for Black

    Returns:
    - loss in centipawns (>= 0 means position worsened for the player who made the move).
    """
    is_white = (color in ['w', 'W', 1, 'white', 'White'])

    if is_white:
        # For White: a good move maintains or increases eval_white.
        # Loss = eval_before - eval_after
        loss = eval_before_white - eval_after_white
    else:
        # For Black: a good move maintains or decreases eval_white (more negative is better for Black).
        # Loss = eval_after - eval_before
        loss = eval_after_white - eval_before_white

    return float(loss)


def assign_error_label(
    eval_loss: float,
    threshold_cp: float = 200.0
) -> int:
    """
    Assigns binary error label:
    1 if centipawn loss >= threshold_cp, otherwise 0.
    """
    if pd.isna(eval_loss):
        return 0
    return 1 if eval_loss >= threshold_cp else 0


def filter_mate_positions(
    df: pd.DataFrame,
    mate_col: str = "is_mate_eval"
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """
    Filters out positions with forced checkmate evaluations as specified in the research plan.
    Returns filtered DataFrame and exclusion statistics.
    """
    total_rows = len(df)
    if mate_col not in df.columns:
        return df.copy(), {"total_positions": total_rows, "excluded_mate_positions": 0, "mate_percentage": 0.0}

    filtered_df = df[~df[mate_col].fillna(False)].copy()
    excluded = total_rows - len(filtered_df)
    pct = (excluded / total_rows * 100.0) if total_rows > 0 else 0.0

    stats = {
        "total_positions": total_rows,
        "excluded_mate_positions": excluded,
        "mate_percentage": round(pct, 2),
    }
    return filtered_df, stats


def audit_features_for_leakage(feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Verifies that no target-related columns or post-move values leaked into feature space.
    """
    violations: List[str] = []
    for feat in feature_names:
        feat_lower = feat.lower()
        for pattern in LEAKAGE_TARGET_PATTERNS:
            if pattern in feat_lower:
                violations.append(
                    f"Feature '{feat}' matches forbidden pattern '{pattern}'! Potential target leakage."
                )
    return len(violations) == 0, violations


def verify_evaluation_feasibility(df: pd.DataFrame, threshold_cp: float = 200.0) -> Dict[str, object]:
    """
    Comprehensive verification to ensure Stockfish evaluations can be reliably used
    for target formation (is_error) and input features (eval_before, eval_diff_prev_move).
    """
    report = {
        "status": "PASS",
        "checks": {},
        "target_distribution": {},
        "warnings": [],
        "recommendations": []
    }

    # Check 1: eval_before existence and stats
    if "eval_before" not in df.columns:
        report["status"] = "FAIL"
        report["warnings"].append("Missing 'eval_before' column.")
    else:
        report["checks"]["eval_before_finite"] = bool(np.isfinite(df["eval_before"]).all())
        report["checks"]["eval_before_min"] = float(df["eval_before"].min())
        report["checks"]["eval_before_max"] = float(df["eval_before"].max())
        report["checks"]["eval_before_mean"] = float(df["eval_before"].mean())

    # Check 2: eval_loss existence and stats
    if "eval_loss" not in df.columns:
        report["status"] = "FAIL"
        report["warnings"].append("Missing 'eval_loss' column.")
    else:
        report["checks"]["eval_loss_finite"] = bool(np.isfinite(df["eval_loss"]).all())
        report["checks"]["eval_loss_min"] = float(df["eval_loss"].min())
        report["checks"]["eval_loss_max"] = float(df["eval_loss"].max())
        report["checks"]["eval_loss_median"] = float(df["eval_loss"].median())

    # Check 3: Target variable formation
    if "is_error" in df.columns:
        pos_count = int((df["is_error"] == 1).sum())
        neg_count = int((df["is_error"] == 0).sum())
        total = pos_count + neg_count
        pos_ratio = (pos_count / total) if total > 0 else 0.0

        report["target_distribution"] = {
            "error_count_pos": pos_count,
            "non_error_count_neg": neg_count,
            "error_ratio_pos": round(pos_ratio, 4),
            "suggested_scale_pos_weight": round((neg_count / pos_count), 2) if pos_count > 0 else 1.0,
        }

        # Check expected imbalance
        if pos_ratio < 0.01:
            report["warnings"].append(
                f"Error ratio is very low ({pos_ratio:.1%}). Model may suffer extreme class imbalance."
            )
        elif pos_ratio > 0.40:
            report["warnings"].append(
                f"Error ratio is unusually high ({pos_ratio:.1%}) for blunder threshold {threshold_cp} cp."
            )

    # Check 4: Data leakage in feature list
    candidate_features = [col for col in df.columns if col not in ["game_id", "ply", "move_number", "color", "is_error", "eval_after", "eval_loss"]]
    no_leakage, violations = audit_features_for_leakage(candidate_features)
    report["checks"]["no_target_leakage"] = no_leakage
    if not no_leakage:
        report["status"] = "FAIL"
        report["warnings"].extend(violations)

    return report
