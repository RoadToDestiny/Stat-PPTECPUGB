"""
Tests for Stage 2 & Stage 3: Dataset schema specification, data contract, and target integrity.
"""

from chess_error_prediction.schemas import (
    METADATA_COLUMNS,
    POSITION_FEATURES,
    META_FEATURES,
    TARGET_COLUMNS,
    ALL_FEATURES,
    ALL_REQUIRED_COLUMNS,
    PositionRecord,
)
from chess_error_prediction.labeling.error_labeler import audit_features_for_leakage


def test_schema_definitions():
    assert "game_id" in METADATA_COLUMNS
    assert "ply" in METADATA_COLUMNS
    assert "eval_before" in POSITION_FEATURES
    assert "player_rating" in META_FEATURES
    assert "eval_loss" in TARGET_COLUMNS
    assert "is_error" in TARGET_COLUMNS

    assert len(ALL_FEATURES) == len(POSITION_FEATURES) + len(META_FEATURES)
    assert len(ALL_REQUIRED_COLUMNS) == len(METADATA_COLUMNS) + len(ALL_FEATURES) + len(TARGET_COLUMNS)


def test_no_leakage_in_features():
    is_clean, violations = audit_features_for_leakage(ALL_FEATURES)
    assert is_clean, f"Target leakage detected in feature space: {violations}"


def test_position_record_structure():
    rec = PositionRecord(
        game_id="g1",
        ply=1,
        move_number=1,
        color="w",
        eval_before=15.0,
        eval_diff_prev_move=0.0,
        material_balance=0.0,
        piece_count=32,
        pawn_count=16,
        legal_checks_count=0,
        legal_captures_count=0,
        attacked_pieces_count=0,
        game_phase=1,
        player_rating=1800.0,
        opponent_rating=1820.0,
        rating_diff=-20.0,
        time_control_base=600.0,
        time_control_inc=0.0,
        eval_after=10.0,
        eval_loss=5.0,
        is_error=0,
    )
    d = rec.to_dict()
    assert d["game_id"] == "g1"
    assert d["is_error"] == 0
    assert d["eval_loss"] == 5.0
