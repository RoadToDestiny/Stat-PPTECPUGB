"""
PGN file reading, decompression, and metadata parsing.
"""

from io import TextIOWrapper
from pathlib import Path
import gzip
import re
from typing import Iterator, Optional, TextIO, Tuple
import chess.pgn
import zstandard


_TIME_CONTROL_PATTERN = re.compile(r"^(?P<base>\d+)(?:\+(?P<increment>\d+))?$")


def parse_time_control(value: str) -> Tuple[int, int]:
    """
    Parses Lichess time control string (e.g. '600+0' or '900+10') into (base_seconds, inc_seconds).
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"Invalid time control: '{value}'")

    match = _TIME_CONTROL_PATTERN.fullmatch(value.strip())
    if not match:
        # Handle cases like '-' or '?' or unparsed format
        raise ValueError(f"Unsupported time control format: '{value}'. Expected '<base>+<inc>'.")

    base = int(match.group("base"))
    inc = int(match.group("increment") or 0)
    return base, inc


def open_pgn_file(path: Path) -> TextIO:
    """
    Opens plain .pgn, gzip .gz, or zstandard .zst archives transparently as text streams.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PGN file not found: {path}")

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".zst":
        binary = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(binary)
        return TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_games(handle: TextIO) -> Iterator[chess.pgn.Game]:
    """
    Generator yielding chess.pgn.Game objects from an open PGN text handle.
    """
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game
