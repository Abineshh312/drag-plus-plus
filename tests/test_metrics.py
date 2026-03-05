"""Unit tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import MetricsCalculator


class TestMetrics:
    """Test DRAG++ metrics."""

    def test_exact_match_identical(self):
        pred = "The answer is yes"
        ref = "The answer is yes"
        assert MetricsCalculator.exact_match(pred, ref) == 1.0

    def test_exact_match_different(self):
        pred = "The answer is yes"
        ref = "The answer is no"
        assert MetricsCalculator.exact_match(pred, ref) == 0.0

    def test_exact_match_case_insensitive(self):
        pred = "THE ANSWER IS YES"
        ref = "the answer is yes"
        assert MetricsCalculator.exact_match(pred, ref) == 1.0

    def test_f1_score_perfect(self):
        pred = "hello world"
        ref = "hello world"
        assert MetricsCalculator.f1_score(pred, ref) == 1.0

    def test_f1_score_partial(self):
        pred = "hello world"
        ref = "hello earth"
        f1 = MetricsCalculator.f1_score(pred, ref)
        assert 0 < f1 < 1

    def test_f1_score_empty(self):
        pred = ""
        ref = ""
        assert MetricsCalculator.f1_score(pred, ref) == 1.0
