from bac1.src.analysis_config import DEFAULT_ANALYSIS_CONFIG, AnalysisConfig


def test_default_analysis_contract():
    assert DEFAULT_ANALYSIS_CONFIG.depth == 18
    assert DEFAULT_ANALYSIS_CONFIG.mate_score_cp == 100000


def test_analysis_config_requires_version():
    try:
        AnalysisConfig(stockfish_version="")
    except ValueError as exc:
        assert "version" in str(exc).lower()
    else:
        raise AssertionError("An empty Stockfish version must be rejected.")
