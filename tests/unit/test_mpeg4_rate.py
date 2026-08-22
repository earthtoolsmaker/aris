"""Unit tests for mpeg4-compatible frame-rate Fractions."""

from fractions import Fraction

import pytest

from aris.pyARIS.pyARIS import _mpeg4_compatible_rate

# Mpeg4 encoder rejects timebases whose denominator exceeds this.
_MPEG4_MAX_TIMEBASE_DEN = 65535


def _timebase_denominator(rate: Fraction) -> int:
    """Mpeg4 timebase is 1/rate → denominator is the rate numerator."""
    return rate.numerator


@pytest.mark.parametrize(
    "fps",
    [
        15.000149726867676,  # typical ARIS instantaneous rate
        5.607141017913818,  # also seen in local recordings
        15.0,
        24.0,
        29.97,
        59.94,
    ],
)
def test_mpeg4_compatible_rate_timebase_within_limit(fps: float) -> None:
    rate = _mpeg4_compatible_rate(fps)
    assert isinstance(rate, Fraction)
    assert _timebase_denominator(rate) <= _MPEG4_MAX_TIMEBASE_DEN
    assert abs(float(rate) - fps) < 1e-3


def test_aris_near_15_matches_ffmpeg_reduction() -> None:
    """ffmpeg -r 15.000149… lands on 65521/4368; match that."""
    rate = _mpeg4_compatible_rate(15.000149726867676)
    assert rate == Fraction(65521, 4368)


def test_naive_limit_denominator_is_illegal_for_aris_rate() -> None:
    """Document why Fraction(fps).limit_denominator() is insufficient."""
    fps = 15.000149726867676
    naive = Fraction(fps).limit_denominator()
    assert _timebase_denominator(naive) > _MPEG4_MAX_TIMEBASE_DEN
