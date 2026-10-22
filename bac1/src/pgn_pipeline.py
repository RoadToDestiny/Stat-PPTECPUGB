"""Download, validate, and filter Lichess PGN games."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator, Optional, TextIO, Tuple
import gzip
from io import TextIOWrapper

import chess
import chess.pgn
import requests
import zstandard


REQUIRED_HEADERS = (
    "Event",
    "Site",
    "Date",
    "Round",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
)
_TIME_CONTROL = re.compile(r"^(?P<base>\d+)(?:\+(?P<increment>\d+))?$")


@dataclass(frozen=True)
class FilterCriteria:
    """Selection criteria for the research dataset."""

    min_elo: Optional[int] = None
    max_elo: Optional[int] = None
    min_base_seconds: Optional[int] = 600
    allowed_time_controls: Optional[Tuple[str, ...]] = None
    min_full_moves: Optional[int] = None
    max_full_moves: Optional[int] = None


@dataclass(frozen=True)
class GameFilterStats:
    read: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0


def parse_time_control(value: str) -> Tuple[int, int]:
    """Parse a Lichess ``base+increment`` time control into seconds."""
    match = _TIME_CONTROL.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported time control '{value}'. Expected '<base>+<increment>'.")
    return int(match.group("base")), int(match.group("increment") or 0)


def _elo(game: chess.pgn.Game, color: str) -> Optional[int]:
    value = game.headers.get(f"{color}Elo")
    if not value or not value.isdigit():
        return None
    return int(value)


def _is_valid_game(game: chess.pgn.Game) -> bool:
    if any(not game.headers.get(header) for header in REQUIRED_HEADERS):
        return False
    if game.errors:
        return False
    board = game.board()
    try:
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                return False
            board.push(move)
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
        return False
    return True


def game_matches(game: chess.pgn.Game, criteria: FilterCriteria) -> bool:
    """Return whether a validated game satisfies the configured filters."""
    time_control = game.headers.get("TimeControl", "")
    if criteria.allowed_time_controls is not None and time_control not in criteria.allowed_time_controls:
        return False
    try:
        base_seconds, _ = parse_time_control(time_control)
    except ValueError:
        return False
    if criteria.min_base_seconds is not None and base_seconds < criteria.min_base_seconds:
        return False

    ratings = [_elo(game, "White"), _elo(game, "Black")]
    if criteria.min_elo is not None and any(rating is None or rating < criteria.min_elo for rating in ratings):
        return False
    if criteria.max_elo is not None and any(rating is None or rating > criteria.max_elo for rating in ratings):
        return False

    full_moves = game.end().board().fullmove_number
    if criteria.min_full_moves is not None and full_moves < criteria.min_full_moves:
        return False
    if criteria.max_full_moves is not None and full_moves > criteria.max_full_moves:
        return False
    return True


def read_games(handle: TextIO) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN stream, preserving parser errors for filtering."""
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        yield game


def _open_pgn(path: Path) -> TextIO:
    """Open plain, gzip, or Lichess zstandard-compressed PGN as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".zst":
        binary = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(binary)
        return TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def filter_pgn(input_path: Path, output_path: Path, criteria: Optional[FilterCriteria] = None) -> GameFilterStats:
    """Validate and write matching games, returning auditable counts."""
    criteria = criteria or FilterCriteria()
    stats = GameFilterStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_games = []
    with _open_pgn(input_path) as source:
        for game in read_games(source):
            stats = GameFilterStats(stats.read + 1, stats.accepted, stats.rejected, stats.invalid)
            if not _is_valid_game(game):
                stats = GameFilterStats(stats.read, stats.accepted, stats.rejected, stats.invalid + 1)
            elif game_matches(game, criteria):
                accepted_games.append(game)
                stats = GameFilterStats(stats.read, stats.accepted + 1, stats.rejected, stats.invalid)
            else:
                stats = GameFilterStats(stats.read, stats.accepted, stats.rejected + 1, stats.invalid)
    with output_path.open("w", encoding="utf-8") as destination:
        for game in accepted_games:
            print(game, file=destination, end="\n\n")
    return stats


def download_pgn(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Stream a PGN (or compressed PGN response) to disk with explicit failures."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        output.write(chunk)
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(f"Unable to download PGN from '{url}'.") from exc
    return destination
