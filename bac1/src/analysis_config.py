"""Single source of truth for Stockfish analysis parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters required by the dataset analysis contract."""

    stockfish_version: str
    depth: int = 18
    mate_score_cp: int = 100000

    def __post_init__(self) -> None:
        if not self.stockfish_version.strip():
            raise ValueError("Stockfish version must be recorded.")
        if self.depth < 1:
            raise ValueError("Analysis depth must be at least 1.")
        if self.mate_score_cp <= 0:
            raise ValueError("Mate score must be positive.")


DEFAULT_ANALYSIS_CONFIG = AnalysisConfig(
    stockfish_version="Stockfish 17",
    depth=18,
    mate_score_cp=100000,
)
