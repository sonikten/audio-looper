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

"""Unit tests for Reporter class in reporting.py (STORY-014 acceptance criteria)."""

from __future__ import annotations

import numpy as np
import pytest

from audioloop.models import AudioData
from audioloop.reporting import Reporter, VerbosityLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(
    *,
    channels: int = 2,
    sample_rate: int = 48000,
    bit_depth: int = 24,
    num_samples: int = 4800,
) -> AudioData:
    """Return a synthetic AudioData instance for testing."""
    shape = (num_samples, channels) if channels > 1 else (num_samples,)
    samples = np.zeros(shape, dtype="float64")
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
    )


# ---------------------------------------------------------------------------
# Reporter.info()
# ---------------------------------------------------------------------------


class TestReporterInfo:
    """Reporter.info() respects verbosity levels."""

    def test_info_prints_at_normal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() must write to stdout when verbosity is NORMAL."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        reporter.info("hello normal")
        captured = capsys.readouterr()
        assert "hello normal" in captured.out

    def test_info_prints_at_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() must write to stdout when verbosity is VERBOSE."""
        reporter = Reporter(VerbosityLevel.VERBOSE)
        reporter.info("hello verbose")
        captured = capsys.readouterr()
        assert "hello verbose" in captured.out

    def test_info_prints_at_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() must write to stdout when verbosity is DEBUG."""
        reporter = Reporter(VerbosityLevel.DEBUG)
        reporter.info("hello debug")
        captured = capsys.readouterr()
        assert "hello debug" in captured.out

    def test_info_suppressed_in_quiet(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() must produce no stdout output when verbosity is QUIET."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.info("should be silent")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_info_writes_to_stdout_not_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info() must write to stdout, not stderr."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        reporter.info("stdout only")
        captured = capsys.readouterr()
        assert "stdout only" in captured.out
        assert captured.err == ""


# ---------------------------------------------------------------------------
# Reporter.verbose()
# ---------------------------------------------------------------------------


class TestReporterVerbose:
    """Reporter.verbose() only outputs at VERBOSE or higher."""

    def test_verbose_suppressed_at_quiet(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose() must produce no output when verbosity is QUIET."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.verbose("quiet check")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_verbose_suppressed_at_normal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose() must produce no output when verbosity is NORMAL."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        reporter.verbose("normal check")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_verbose_prints_at_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose() must write to stderr when verbosity is VERBOSE."""
        reporter = Reporter(VerbosityLevel.VERBOSE)
        reporter.verbose("verbose detail")
        captured = capsys.readouterr()
        assert "verbose detail" in captured.err

    def test_verbose_prints_at_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose() must write to stderr when verbosity is DEBUG."""
        reporter = Reporter(VerbosityLevel.DEBUG)
        reporter.verbose("debug also sees verbose")
        captured = capsys.readouterr()
        assert "debug also sees verbose" in captured.err

    def test_verbose_writes_to_stderr_not_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """verbose() must write to stderr, not stdout."""
        reporter = Reporter(VerbosityLevel.VERBOSE)
        reporter.verbose("stderr only")
        captured = capsys.readouterr()
        assert "stderr only" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Reporter.debug()
# ---------------------------------------------------------------------------


class TestReporterDebug:
    """Reporter.debug() only outputs at DEBUG."""

    def test_debug_suppressed_at_quiet(self, capsys: pytest.CaptureFixture[str]) -> None:
        """debug() must produce no output when verbosity is QUIET."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.debug("quiet debug")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_debug_suppressed_at_normal(self, capsys: pytest.CaptureFixture[str]) -> None:
        """debug() must produce no output when verbosity is NORMAL."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        reporter.debug("normal debug")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_debug_suppressed_at_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """debug() must produce no output when verbosity is VERBOSE."""
        reporter = Reporter(VerbosityLevel.VERBOSE)
        reporter.debug("verbose debug")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_debug_prints_at_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        """debug() must write to stderr when verbosity is DEBUG."""
        reporter = Reporter(VerbosityLevel.DEBUG)
        reporter.debug("full debug output")
        captured = capsys.readouterr()
        assert "full debug output" in captured.err

    def test_debug_writes_to_stderr_not_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """debug() must write to stderr, not stdout."""
        reporter = Reporter(VerbosityLevel.DEBUG)
        reporter.debug("debug stderr")
        captured = capsys.readouterr()
        assert "debug stderr" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Reporter.error()
# ---------------------------------------------------------------------------


class TestReporterError:
    """Reporter.error() always outputs to stderr regardless of verbosity."""

    @pytest.mark.parametrize(
        "level",
        [VerbosityLevel.QUIET, VerbosityLevel.NORMAL, VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG],
    )
    def test_error_always_prints(
        self, capsys: pytest.CaptureFixture[str], level: VerbosityLevel
    ) -> None:
        """error() must write to stderr at every verbosity level."""
        reporter = Reporter(level)
        reporter.error("something went wrong")
        captured = capsys.readouterr()
        assert "something went wrong" in captured.err

    def test_error_writes_to_stderr_not_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """error() must write to stderr, not stdout."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.error("error to stderr")
        captured = capsys.readouterr()
        assert "error to stderr" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# Quiet mode — comprehensive suppression check
# ---------------------------------------------------------------------------


class TestQuietModeSuppression:
    """QUIET mode suppresses info/verbose/debug but not error."""

    def test_quiet_suppresses_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """In QUIET mode, info() must produce no output."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.info("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_quiet_suppresses_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """In QUIET mode, verbose() must produce no output."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.verbose("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_quiet_suppresses_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        """In QUIET mode, debug() must produce no output."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.debug("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_quiet_does_not_suppress_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """In QUIET mode, error() must still write to stderr."""
        reporter = Reporter(VerbosityLevel.QUIET)
        reporter.error("critical failure")
        captured = capsys.readouterr()
        assert "critical failure" in captured.err


# ---------------------------------------------------------------------------
# Reporter.format_metadata() and print_metadata()
# ---------------------------------------------------------------------------


class TestReporterFormatMetadata:
    """Reporter.format_metadata() and print_metadata() delegate to the module-level function."""

    def test_format_metadata_contains_filename(self) -> None:
        """format_metadata() output must include the filename."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio()
        result = reporter.format_metadata("track.wav", audio)
        assert "track.wav" in result

    def test_format_metadata_contains_bit_depth(self) -> None:
        """format_metadata() output must include the bit depth."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio(bit_depth=24)
        result = reporter.format_metadata("track.wav", audio)
        assert "24-bit" in result

    def test_format_metadata_contains_sample_rate(self) -> None:
        """format_metadata() output must include the sample rate in kHz."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio(sample_rate=48000)
        result = reporter.format_metadata("track.wav", audio)
        assert "48kHz" in result

    def test_format_metadata_stereo_label(self) -> None:
        """format_metadata() must report 'stereo' for a 2-channel file."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio(channels=2)
        result = reporter.format_metadata("track.wav", audio)
        assert "stereo" in result

    def test_format_metadata_mono_label(self) -> None:
        """format_metadata() must report 'mono' for a 1-channel file."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio(channels=1)
        result = reporter.format_metadata("track.wav", audio)
        assert "mono" in result

    def test_print_metadata_writes_to_stdout_at_normal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """print_metadata() must write to stdout when verbosity is NORMAL."""
        reporter = Reporter(VerbosityLevel.NORMAL)
        audio = _make_audio()
        reporter.print_metadata("track.wav", audio)
        captured = capsys.readouterr()
        assert "track.wav" in captured.out

    def test_print_metadata_suppressed_in_quiet(self, capsys: pytest.CaptureFixture[str]) -> None:
        """print_metadata() must produce no output when verbosity is QUIET."""
        reporter = Reporter(VerbosityLevel.QUIET)
        audio = _make_audio()
        reporter.print_metadata("track.wav", audio)
        captured = capsys.readouterr()
        assert captured.out == ""
