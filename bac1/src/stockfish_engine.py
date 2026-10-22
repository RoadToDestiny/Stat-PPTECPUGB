"""Stockfish lifecycle and position evaluation helpers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import chess
import chess.engine


class StockfishError(RuntimeError):
    """Raised when Stockfish cannot be started or queried."""


@dataclass(frozen=True)
class StockfishConfig:
    """Parameters shared by all Stockfish analyses."""

    executable: str = "stockfish"
    depth: int = 18
    timeout: float = 30.0
    threads: Optional[int] = None
    hash_mb: Optional[int] = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("Stockfish depth must be at least 1.")
        if self.timeout <= 0:
            raise ValueError("Stockfish timeout must be positive.")


class StockfishEngine:
    """A small context-managed wrapper with a stable white-perspective API."""

    def __init__(self, config: Optional[StockfishConfig] = None) -> None:
        self.config = config or StockfishConfig()
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self) -> "StockfishEngine":
        if self._engine is not None:
            return self
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(
                self.config.executable,
                timeout=self.config.timeout,
            )
            options: Dict[str, Any] = {}
            if self.config.threads is not None:
                options["Threads"] = self.config.threads
            if self.config.hash_mb is not None:
                options["Hash"] = self.config.hash_mb
            if options:
                self._engine.configure(options)
        except (OSError, chess.engine.EngineError) as exc:
            self._engine = None
            raise StockfishError(
                f"Unable to start Stockfish executable '{self.config.executable}'. "
                "Install Stockfish or pass its executable path."
            ) from exc
        return self

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "StockfishEngine":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @property
    def engine_id(self) -> Dict[str, str]:
        """Return the UCI engine identification after startup."""
        if self._engine is None:
            raise StockfishError("Stockfish is not started.")
        return dict(self._engine.id)

    def analyze(self, board: chess.Board) -> Dict[str, Any]:
        """Analyze a position and return a score in centipawns from White's POV."""
        if self._engine is None:
            raise StockfishError("Stockfish is not started.")
        try:
            info = self._engine.analyse(
                board,
                chess.engine.Limit(depth=self.config.depth),
            )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            raise StockfishError("Stockfish failed while analyzing a position.") from exc

        score = info["score"].pov(chess.WHITE)
        score_cp = score.score(mate_score=100000)
        return {
            "score_cp": float(score_cp) if score_cp is not None else None,
            "is_mate": score.is_mate(),
            "mate_in": score.mate(),
            "depth": info.get("depth", self.config.depth),
        }

    def verify(self) -> Dict[str, str]:
        """Start Stockfish, analyze the initial position, and return its identity."""
        self.start()
        self.analyze(chess.Board())
        return self.engine_id
