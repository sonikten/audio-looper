# Copyright 2026 SONIK TEN PTY LTD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the duration parsing utility (STORY-010, STORY-011 acceptance criteria)."""

from __future__ import annotations

import pytest

from audioloop.duration import parse_duration, parse_loop_length
from audioloop.exceptions import DurationParseError


class TestParseSecondsFormats:
    """Tests for seconds-based input formats."""

    def test_parse_seconds_suffix(self) -> None:
        """'30s' must parse to 30.0 seconds."""
        assert parse_duration("30s") == pytest.approx(30.0)

    def test_parse_bare_number(self) -> None:
        """'30' (no suffix) must parse to 30.0 seconds."""
        assert parse_duration("30") == pytest.approx(30.0)

    def test_parse_fractional(self) -> None:
        """'2.5s' must parse to 2.5 seconds."""
        assert parse_duration("2.5s") == pytest.approx(2.5)

    def test_parse_fractional_bare(self) -> None:
        """'2.5' (no suffix) must parse to 2.5 seconds."""
        assert parse_duration("2.5") == pytest.approx(2.5)

    def test_parse_zero_seconds(self) -> None:
        """'0s' must parse to 0.0 seconds."""
        assert parse_duration("0s") == pytest.approx(0.0)


class TestParseMinutesFormats:
    """Tests for minutes-based input formats."""

    def test_parse_minutes_seconds(self) -> None:
        """'1:30' must parse to 90.0 seconds."""
        assert parse_duration("1:30") == pytest.approx(90.0)

    def test_parse_minutes_suffix(self) -> None:
        """'2m' must parse to 120.0 seconds."""
        assert parse_duration("2m") == pytest.approx(120.0)

    def test_parse_minutes_min_suffix(self) -> None:
        """'2min' must parse to 120.0 seconds."""
        assert parse_duration("2min") == pytest.approx(120.0)

    def test_parse_minutes_fractional(self) -> None:
        """'1.5m' must parse to 90.0 seconds."""
        assert parse_duration("1.5m") == pytest.approx(90.0)

    def test_parse_mm_ss_with_fractional_seconds(self) -> None:
        """'1:30.5' must parse to 90.5 seconds."""
        assert parse_duration("1:30.5") == pytest.approx(90.5)

    def test_parse_zero_minutes_colon(self) -> None:
        """'0:30' must parse to 30.0 seconds."""
        assert parse_duration("0:30") == pytest.approx(30.0)


class TestParseMilliseconds:
    """Tests for millisecond-suffix format."""

    def test_parse_milliseconds(self) -> None:
        """'500ms' must parse to 0.5 seconds."""
        assert parse_duration("500ms") == pytest.approx(0.5)

    def test_parse_milliseconds_whole(self) -> None:
        """'1000ms' must parse to 1.0 seconds."""
        assert parse_duration("1000ms") == pytest.approx(1.0)


class TestParseHours:
    """Tests for hours-suffix format."""

    def test_parse_hours_suffix(self) -> None:
        """'1h' must parse to 3600.0 seconds."""
        assert parse_duration("1h") == pytest.approx(3600.0)


class TestParseInvalidInput:
    """Tests for invalid / unparseable duration strings."""

    def test_parse_invalid_raises_error(self) -> None:
        """'abc' must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_duration("abc")

    def test_parse_empty_string_raises_error(self) -> None:
        """An empty string must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_duration("")

    def test_parse_only_whitespace_raises_error(self) -> None:
        """A whitespace-only string must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_duration("   ")

    def test_parse_letters_only_raises_error(self) -> None:
        """A string of letters must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_duration("xyz")

    def test_parse_invalid_with_valid_suffix_raises_error(self) -> None:
        """A malformed value like 's30' must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_duration("s30")

    def test_parse_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace around a valid value must be tolerated."""
        assert parse_duration("  30s  ") == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# STORY-011: parse_loop_length — bar-based and time-based parsing
# ---------------------------------------------------------------------------


class TestParseLoopLengthBars:
    """Tests for bar-based loop length parsing via parse_loop_length."""

    def test_parse_loop_length_bars(self) -> None:
        """'4bars' with 120 BPM must return 8.0 seconds (4 * 4 / 120 * 60)."""
        result = parse_loop_length("4bars", bpm=120.0)
        assert result == pytest.approx(8.0)

    def test_parse_loop_length_bar_singular(self) -> None:
        """'1bar' with 120 BPM must return 2.0 seconds (1 * 4 / 120 * 60)."""
        result = parse_loop_length("1bar", bpm=120.0)
        assert result == pytest.approx(2.0)

    def test_parse_loop_length_bars_case_insensitive(self) -> None:
        """'4BARS' must parse identically to '4bars'."""
        result = parse_loop_length("4BARS", bpm=120.0)
        assert result == pytest.approx(8.0)

    def test_parse_loop_length_bars_different_bpm(self) -> None:
        """'8bars' with 90 BPM must return the correct second value."""
        # 8 * 4 / 90 * 60 = 21.333...
        result = parse_loop_length("8bars", bpm=90.0)
        assert result == pytest.approx(8 * 4 / 90 * 60)

    def test_parse_loop_length_bars_no_bpm_raises(self) -> None:
        """'4bars' with bpm=None must raise DurationParseError."""
        with pytest.raises(DurationParseError, match="tempo"):
            parse_loop_length("4bars", bpm=None)

    def test_parse_loop_length_bar_no_bpm_raises(self) -> None:
        """'1bar' with bpm=None must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_loop_length("1bar", bpm=None)

    def test_parse_loop_length_bars_error_message_mentions_tempo(self) -> None:
        """DurationParseError for missing BPM must mention 'tempo' in its message."""
        with pytest.raises(DurationParseError, match="tempo"):
            parse_loop_length("4bars")


class TestParseLoopLengthTimeBased:
    """Tests for time-based delegation in parse_loop_length."""

    def test_parse_loop_length_seconds(self) -> None:
        """'8s' must delegate to parse_duration and return 8.0."""
        result = parse_loop_length("8s")
        assert result == pytest.approx(8.0)

    def test_parse_loop_length_minutes(self) -> None:
        """'2m' must delegate to parse_duration and return 120.0."""
        result = parse_loop_length("2m")
        assert result == pytest.approx(120.0)

    def test_parse_loop_length_colon_format(self) -> None:
        """'1:30' must delegate to parse_duration and return 90.0."""
        result = parse_loop_length("1:30")
        assert result == pytest.approx(90.0)

    def test_parse_loop_length_time_ignores_bpm(self) -> None:
        """Time-based values must ignore the bpm argument."""
        result = parse_loop_length("8s", bpm=120.0)
        assert result == pytest.approx(8.0)

    def test_parse_loop_length_invalid_raises(self) -> None:
        """An unparseable string must raise DurationParseError."""
        with pytest.raises(DurationParseError):
            parse_loop_length("notavalue")


# ---------------------------------------------------------------------------
# STORY-022: parse_loop_length — non-4/4 time signature support
# ---------------------------------------------------------------------------


class TestParseLoopLengthTimeSig:
    """Tests for beats_per_bar parameter in parse_loop_length (STORY-022)."""

    def test_parse_loop_length_3_4_time(self) -> None:
        """'4bars' with 120 BPM and beats_per_bar=3 must return 6.0s (4*3/120*60)."""
        result = parse_loop_length("4bars", bpm=120.0, beats_per_bar=3)
        assert result == pytest.approx(4 * 3 / 120 * 60)

    def test_parse_loop_length_6_8_time(self) -> None:
        """'2bars' with 120 BPM and beats_per_bar=6 must return 6.0s (2*6/120*60)."""
        result = parse_loop_length("2bars", bpm=120.0, beats_per_bar=6)
        assert result == pytest.approx(2 * 6 / 120 * 60)

    def test_parse_loop_length_default_4_4(self) -> None:
        """'4bars' with 120 BPM and default beats_per_bar must return 8.0s (unchanged)."""
        result = parse_loop_length("4bars", bpm=120.0)
        assert result == pytest.approx(8.0)
