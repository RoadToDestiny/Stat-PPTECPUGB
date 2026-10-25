"""
Game filtering based on time control, player ratings, and game length criteria.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import chess.pgn

from chess_error_prediction.data.pgn_reader import open_pgn_file, read_games, parse_time_control
from chess_error_prediction.data.validation import validate_game


@dataclass(frozen=True)
class FilterCriteria:
    """Selection criteria for filtering the research dataset."""
    min_elo: Optional[int] = 1600
    max_elo: Optional[int] = 2800
    min_base_seconds: Optional[int] = 600
    allowed_time_controls: Optional[Tuple[str, ...]] = None
    min_full_moves: Optional[int] = 10
    max_full_moves: Optional[int] = 120


@dataclass(frozen=True)
class GameFilterStats:
    """Audit statistics for PGN filtering."""
    read: int = 0
    accepted: int = 0
    rejected: int = 0
    invalid: int = 0


def _get_elo(game: chess.pgn.Game, color_key: str) -> Optional[int]:
    val = game.headers.get(color_key)
    if val and val.isdigit():
        return int(val)
    return None


def game_matches(game: chess.pgn.Game, criteria: FilterCriteria) -> bool:
    """
    Checks if a valid chess game matches all specified criteria.
    """
    time_control = game.headers.get("TimeControl", "")
    if criteria.allowed_time_controls is not None and time_control not in criteria.allowed_time_controls:
        return False

    try:
        base_sec, _ = parse_time_control(time_control)
    except ValueError:
        return False

    if criteria.min_base_seconds is not None and base_sec < criteria.min_base_seconds:
        return False

    white_elo = _get_elo(game, "WhiteElo")
    black_elo = _get_elo(game, "BlackElo")

    if criteria.min_elo is not None:
        if white_elo is None or white_elo < criteria.min_elo:
            return False
        if black_elo is None or black_elo < criteria.min_elo:
            return False

    if criteria.max_elo is not None:
        if white_elo is not None and white_elo > criteria.max_elo:
            return False
        if black_elo is not None and black_elo > criteria.max_elo:
            return False

    full_moves = game.end().board().fullmove_number
    if criteria.min_full_moves is not None and full_moves < criteria.min_full_moves:
        return False
    if criteria.max_full_moves is not None and full_moves > criteria.max_full_moves:
        return False

    return True


def filter_pgn(
    input_path: Path,
    output_path: Path,
    criteria: Optional[FilterCriteria] = None,
    max_games_to_accept: Optional[int] = None,
) -> GameFilterStats:
    """
    Filters raw PGN games into filtered_games.pgn and returns filter statistics.
    """
    criteria = criteria or FilterCriteria()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    accepted_count = 0
    rejected_count = 0
    invalid_count = 0

    with open_pgn_file(input_path) as source, output_path.open("w", encoding="utf-8") as dest:
        for game in read_games(source):
            read_count += 1
            is_valid, _ = validate_game(game)
            if not is_valid:
                invalid_count += 1
            elif game_matches(game, criteria):
                accepted_count += 1
                print(game, file=dest, end="\n\n")
                if max_games_to_accept and accepted_count >= max_games_to_accept:
                    break
            else:
                rejected_count += 1

    return GameFilterStats(
        read=read_count,
        accepted=accepted_count,
        rejected=rejected_count,
        invalid=invalid_count,
    )
