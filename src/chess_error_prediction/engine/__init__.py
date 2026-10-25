"""
Stockfish engine management and evaluation normalization.
"""

from chess_error_prediction.engine.stockfish_client import StockfishConfig, StockfishEngine, StockfishError
from chess_error_prediction.engine.evaluation import EvaluationResult, normalize_evaluation

__all__ = [
    "StockfishConfig",
    "StockfishEngine",
    "StockfishError",
    "EvaluationResult",
    "normalize_evaluation",
]
