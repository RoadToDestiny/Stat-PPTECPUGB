"""
Tests for Stockfish configuration, lifecycle, and evaluation normalization.
"""

import pytest
import chess
from chess_error_prediction.engine.stockfish_client import StockfishConfig, StockfishEngine, StockfishError
from chess_error_prediction.engine.evaluation import normalize_evaluation, EvaluationResult


def test_stockfish_config_defaults():
    cfg = StockfishConfig()
    assert cfg.depth == 18
    assert cfg.executable == "stockfish"
    assert cfg.timeout == 30.0
    assert cfg.mate_score_cp == 100000


def test_stockfish_config_validation():
    with pytest.raises(ValueError):
        StockfishConfig(depth=0)
    with pytest.raises(ValueError):
        StockfishConfig(timeout=-5.0)
    with pytest.raises(ValueError):
        StockfishConfig(mate_score_cp=0)


def test_evaluation_normalization():
    raw_eval = {"score_cp": 125.0, "is_mate": False, "mate_in": None, "depth": 18}
    norm = normalize_evaluation(raw_eval)
    assert isinstance(norm, EvaluationResult)
    assert norm.score_cp == 125.0
    assert not norm.is_mate
    assert norm.is_valid_numeric

    mate_eval = {"score_cp": 100000.0, "is_mate": True, "mate_in": 2, "depth": 18}
    norm_mate = normalize_evaluation(mate_eval)
    assert norm_mate.is_mate
    assert not norm_mate.is_valid_numeric
    assert norm_mate.mate_in == 2
