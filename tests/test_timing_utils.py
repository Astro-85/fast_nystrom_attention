import math

from experiments.llava.eval.timing_utils import (
    average_context_length,
    context_normalization_factor,
    normalize_latency,
)


def test_average_context_length_handles_generation_growth():
    assert average_context_length(10, 0) is None
    assert math.isclose(average_context_length(10, 1), 10.0)
    assert math.isclose(average_context_length(10, 4), 11.5)


def test_context_normalization_factor_matches_average_length():
    factor = context_normalization_factor(10, 4)
    assert factor is not None
    assert math.isclose(factor, 11.5 / 10.0)


def test_normalize_latency_scales_by_factor():
    latency = 2.0
    normalized = normalize_latency(latency, 10, 4)
    expected = latency / (11.5 / 10.0)
    assert math.isclose(normalized, expected)


def test_normalize_latency_handles_missing_latency():
    assert normalize_latency(None, 10, 4) is None
    assert normalize_latency(1.0, 10, 0) is None
