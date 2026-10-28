"""Regression tests for segmentation.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdf_review.segmentation import is_heading, windows


def test_numbered_top_level_heading_detected():
    assert is_heading("2. Methods") is True


def test_named_heading_detected_case_insensitive():
    assert is_heading("Introduction") is True
    assert is_heading("LIMITATIONS") is True


def test_figure_caption_not_treated_as_heading():
    assert is_heading("Figure 3: Overview of the pipeline") is False


def test_affiliation_line_not_treated_as_heading():
    assert is_heading("1 Department of Computer Science") is False


def test_ordinary_body_text_not_a_heading():
    assert is_heading("The results show a clear improvement over baselines.") is False


def test_windows_covers_full_token_range_with_overlap():
    tokens = list(range(10))
    spans = list(windows(tokens, size=4, overlap=1))
    assert spans[0] == (0, 4, [0, 1, 2, 3])
    assert spans[-1][1] == len(tokens)  # last window reaches the end


def test_windows_single_pass_when_shorter_than_size():
    tokens = list(range(3))
    spans = list(windows(tokens, size=10, overlap=2))
    assert len(spans) == 1
    assert spans[0] == (0, 3, tokens)
