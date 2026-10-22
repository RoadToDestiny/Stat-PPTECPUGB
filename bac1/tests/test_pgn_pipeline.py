from io import StringIO

import chess.pgn

from bac1.src.pgn_pipeline import FilterCriteria, game_matches, parse_time_control, read_games


PGN = """[Event "Rated Rapid game"]
[Site "https://lichess.org/test"]
[Date "2026.01.01"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1700"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def test_parse_time_control():
    assert parse_time_control("600+5") == (600, 5)


def test_game_matches_research_filters():
    game = next(read_games(StringIO(PGN)))
    assert game_matches(game, FilterCriteria(min_elo=1600, min_base_seconds=600))
    assert not game_matches(game, FilterCriteria(min_elo=1900))
