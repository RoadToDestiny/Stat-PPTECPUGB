"""
Data contract and validation schema for the chess blunder prediction dataset.
Defines required features, targets, and group-based splitting rules.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "game_id",
    "ply",
    "move_number",
    "color",
]

POSITION_FEATURES = [
    "eval_before",
    "eval_diff_prev_move",
    "material_balance",
    "piece_count",
    "pawn_count",
    "legal_checks_count",
    "legal_captures_count",
    "attacked_pieces_count",
    "game_phase",
]

META_FEATURES = [
    "player_rating",
    "opponent_rating",
    "rating_diff",
    "time_control_base",
    "time_control_inc",
]

TARGET_COLUMNS = [
    "eval_after",
    "eval_loss",
    "is_error",
]

OPTIONAL_COLUMNS = [
    "fen",
    "clock_remaining",
    "is_mate_eval",
]

ALL_REQUIRED_FEATURES = POSITION_FEATURES + META_FEATURES


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether the provided DataFrame complies with the required schema.
    Returns (is_valid, error_messages).
    """
    errors: List[str] = []

    # Check required columns
    all_required = METADATA_COLUMNS + POSITION_FEATURES + META_FEATURES + TARGET_COLUMNS
    missing_cols = [col for col in all_required if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if errors:
        return False, errors

    # Check null values in critical columns
    critical_cols = ["game_id", "ply", "is_error", "eval_before", "eval_loss"]
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null/NaN values.")

    # Check target binary validity
    unique_targets = set(df["is_error"].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        errors.append(f"Target column 'is_error' contains invalid values: {unique_targets}. Must be 0 or 1.")

    # Check logical ranges
    if (df["pawn_count"] < 0).any() or (df["pawn_count"] > 16).any():
        errors.append("Column 'pawn_count' has values outside the valid range [0, 16].")

    if (df["piece_count"] < 2).any() or (df["piece_count"] > 32).any():
        errors.append("Column 'piece_count' has values outside the valid range [2, 32].")

    if (df["ply"] < 1).any():
        errors.append("Column 'ply' must contain positive integers >= 1.")

    return len(errors) == 0, errors


def verify_group_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "game_id"
) -> Tuple[bool, List[str]]:
    """
    Ensures no game_id leaks between train, val, and test splits.
    """
    errors: List[str] = []
    train_games = set(train_df[group_col].unique())
    val_games = set(val_df[group_col].unique())
    test_games = set(test_df[group_col].unique())

    train_val_overlap = train_games.intersection(val_games)
    train_test_overlap = train_games.intersection(test_games)
    val_test_overlap = val_games.intersection(test_games)

    if train_val_overlap:
        errors.append(f"Data leakage detected! {len(train_val_overlap)} game_ids overlap between train and val.")
    if train_test_overlap:
        errors.append(f"Data leakage detected! {len(train_test_overlap)} game_ids overlap between train and test.")
    if val_test_overlap:
        errors.append(f"Data leakage detected! {len(val_test_overlap)} game_ids overlap between val and test.")

    return len(errors) == 0, errors


def split_by_games(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs deterministic group-based split by game_id.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    unique_games = df["game_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_seed)
    shuffled_games = rng.permutation(unique_games)

    n_total = len(shuffled_games)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(shuffled_games[:n_train])
    val_ids = set(shuffled_games[n_train:n_train + n_val])
    test_ids = set(shuffled_games[n_train + n_val:])

    train_df = df[df["game_id"].isin(train_ids)].copy()
    val_df = df[df["game_id"].isin(val_ids)].copy()
    test_df = df[df["game_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df
