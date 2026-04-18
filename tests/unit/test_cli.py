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

"""Unit tests for the audioloop CLI (STORY-001, STORY-002, and STORY-003 acceptance criteria)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from audioloop.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click CliRunner for invoking the CLI in tests."""
    return CliRunner()


@pytest.fixture()
def stereo_wav(tmp_path: pytest.TempPathFactory) -> str:
    """Create a synthetic stereo 24-bit/48kHz WAV file for CLI tests.

    Returns:
        Absolute path to the WAV file as a string.
    """
    path = str(tmp_path / "test_stereo.wav")  # type: ignore[operator]
    samples = np.zeros((4800, 2), dtype="float64")  # 0.1 s of silence
    sf.write(path, samples, 48000, subtype="PCM_24")
    return path


@pytest.fixture()
def loopable_wav(tmp_path: pytest.TempPathFactory) -> str:
    """Create a synthetic stereo 24-bit/48kHz WAV suitable for loop detection.

    The file contains 5 seconds of a sustained 440 Hz sine tone.  Because the
    spectral content is identical throughout, MFCC cosine similarity between
    any two 1-second windows will be ~1.0, ensuring ``detect_loop_points``
    succeeds.

    Returns:
        Absolute path to the WAV file as a string.
    """
    sr = 48000
    duration = 5.0
    path = str(tmp_path / "test_loopable.wav")  # type: ignore[operator]
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")
    return path


class TestHelpOption:
    """Tests for --help / -h behaviour."""

    def test_help_exits_zero(self, runner: CliRunner) -> None:
        """--help must exit with code 0."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_input_argument(self, runner: CliRunner) -> None:
        """--help output must mention the INPUT argument."""
        result = runner.invoke(main, ["--help"])
        assert "INPUT" in result.output

    def test_help_lists_output_option(self, runner: CliRunner) -> None:
        """--help output must list the --output / -o option."""
        result = runner.invoke(main, ["--help"])
        assert "--output" in result.output

    def test_help_lists_loop_length_option(self, runner: CliRunner) -> None:
        """--help output must list the --loop-length option."""
        result = runner.invoke(main, ["--help"])
        assert "--loop-length" in result.output

    def test_help_lists_count_option(self, runner: CliRunner) -> None:
        """--help output must list the --count / -n option."""
        result = runner.invoke(main, ["--help"])
        assert "--count" in result.output

    def test_help_lists_duration_option(self, runner: CliRunner) -> None:
        """--help output must list the --duration / -d option."""
        result = runner.invoke(main, ["--help"])
        assert "--duration" in result.output

    def test_help_lists_crossfade_option(self, runner: CliRunner) -> None:
        """--help output must list the --crossfade / -x option."""
        result = runner.invoke(main, ["--help"])
        assert "--crossfade" in result.output

    def test_help_lists_batch_option(self, runner: CliRunner) -> None:
        """--help output must list the --batch / -b flag."""
        result = runner.invoke(main, ["--help"])
        assert "--batch" in result.output

    def test_help_lists_analyze_only_option(self, runner: CliRunner) -> None:
        """--help output must list the --analyze-only flag."""
        result = runner.invoke(main, ["--help"])
        assert "--analyze-only" in result.output

    def test_help_lists_verbose_option(self, runner: CliRunner) -> None:
        """--help output must list the --verbose / -v flag."""
        result = runner.invoke(main, ["--help"])
        assert "--verbose" in result.output

    def test_help_lists_quiet_option(self, runner: CliRunner) -> None:
        """--help output must list the --quiet / -q flag."""
        result = runner.invoke(main, ["--help"])
        assert "--quiet" in result.output

    def test_short_h_alias_works(self, runner: CliRunner) -> None:
        """-h must produce the same help output as --help."""
        result = runner.invoke(main, ["-h"])
        assert result.exit_code == 0
        assert "INPUT" in result.output


class TestVersionOption:
    """Tests for --version behaviour."""

    def test_version_exits_zero(self, runner: CliRunner) -> None:
        """--version must exit with code 0."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0

    def test_version_shows_audioloop_prefix(self, runner: CliRunner) -> None:
        """--version output must contain the program name 'audioloop'."""
        result = runner.invoke(main, ["--version"])
        assert "audioloop" in result.output

    def test_version_shows_correct_version_number(self, runner: CliRunner) -> None:
        """--version output must contain '0.3.1'."""
        result = runner.invoke(main, ["--version"])
        assert "0.3.1" in result.output

    def test_version_exact_format(self, runner: CliRunner) -> None:
        """--version output must match the exact string 'audioloop, version 0.3.1'."""
        result = runner.invoke(main, ["--version"])
        assert "audioloop, version 0.3.1" in result.output or "audioloop 0.3.1" in result.output


class TestNoArguments:
    """Tests for missing-argument error behaviour (AC-3)."""

    def test_no_args_exits_with_code_2(self, runner: CliRunner) -> None:
        """No arguments must produce exit code 2."""
        result = runner.invoke(main, [])
        assert result.exit_code == 2

    def test_no_args_prints_error_to_stderr(self, runner: CliRunner) -> None:
        """No arguments must print an error message."""
        result = runner.invoke(main, [])
        # Click combines stdout and stderr in result.output when using CliRunner.
        assert "error" in result.output.lower() or "missing" in result.output.lower()


class TestUnknownOption:
    """Tests for unrecognised option error behaviour (AC-4)."""

    def test_unknown_option_exits_with_code_2(self, runner: CliRunner) -> None:
        """An unrecognised option must produce exit code 2."""
        result = runner.invoke(main, ["--nonsense"])
        assert result.exit_code == 2

    def test_unknown_option_reports_error(self, runner: CliRunner) -> None:
        """An unrecognised option must print an error message."""
        result = runner.invoke(main, ["--nonsense"])
        assert "error" in result.output.lower() or "no such option" in result.output.lower()


class TestMutualExclusivity:
    """Tests for mutually exclusive option pairs."""

    def test_count_and_duration_are_mutually_exclusive(
        self, runner: CliRunner, stereo_wav: str
    ) -> None:
        """--count and --duration together must produce an error."""
        result = runner.invoke(main, [stereo_wav, "--count", "4", "--duration", "30s"])
        assert result.exit_code != 0

    def test_verbose_and_quiet_are_mutually_exclusive(
        self, runner: CliRunner, stereo_wav: str
    ) -> None:
        """--verbose and --quiet together must produce an error."""
        result = runner.invoke(main, [stereo_wav, "--verbose", "--quiet"])
        assert result.exit_code != 0


class TestNormalInvocation:
    """Tests for successful invocation with a valid input file."""

    def test_processes_existing_file(self, runner: CliRunner, loopable_wav: str) -> None:
        """A valid input file must produce exit code 0 and print a processing message."""
        result = runner.invoke(main, [loopable_wav])
        assert result.exit_code == 0
        assert "Processing:" in result.output


class TestAnalyzeOnly:
    """Tests for --analyze-only flag (STORY-002 acceptance criteria)."""

    def test_analyze_only_exits_zero(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only with a valid WAV must exit with code 0."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert result.exit_code == 0

    def test_analyze_only_shows_analyzing_header(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only output must begin with 'Analyzing: <filename>'."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "Analyzing:" in result.output

    def test_analyze_only_shows_sample_rate(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only output must include the sample rate in kHz."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "48kHz" in result.output

    def test_analyze_only_shows_bit_depth(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only output must include the bit depth."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "24-bit" in result.output

    def test_analyze_only_shows_channel_label(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only output must report 'stereo' for a 2-channel file."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "stereo" in result.output

    def test_analyze_only_shows_duration(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only output must include the duration."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "0.1s" in result.output

    def test_analyze_only_mono_shows_mono_label(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--analyze-only must report 'mono' for a single-channel file."""
        path = str(tmp_path / "test_mono.wav")  # type: ignore[operator]
        samples = np.zeros(4800, dtype="float64")
        sf.write(path, samples, 48000, subtype="PCM_24")
        result = runner.invoke(main, [path, "--analyze-only"])
        assert result.exit_code == 0
        assert "mono" in result.output

    def test_analyze_only_does_not_write_output(self, runner: CliRunner, stereo_wav: str) -> None:
        """--analyze-only must not print a 'Processing:' line."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only"])
        assert "Processing:" not in result.output


# ---------------------------------------------------------------------------
# STORY-007: Analyze-only mode (enhanced) — loop candidates and no file output
# ---------------------------------------------------------------------------


@pytest.fixture()
def loopable_wav_analyze(tmp_path: pytest.TempPathFactory) -> str:
    """Create a 5-second loopable stereo WAV for STORY-007 analyze-only tests.

    Uses a 440 Hz sine tone so MFCC similarity is high throughout, guaranteeing
    loop candidates are found.

    Returns:
        Absolute path to the WAV file as a string.
    """
    sr = 48000
    duration = 5.0
    path = str(tmp_path / "loopable_analyze.wav")  # type: ignore[operator]
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")
    return path


class TestAnalyzeOnlyEnhanced:
    """Tests for enhanced --analyze-only output (STORY-007 acceptance criteria)."""

    def test_analyze_only_shows_loop_candidates(
        self, runner: CliRunner, loopable_wav_analyze: str
    ) -> None:
        """--analyze-only must print 'Loop candidates:' with scored entries."""
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only"])
        assert result.exit_code == 0
        assert "Loop candidates:" in result.output
        # At least one candidate line contains a similarity score.
        assert "similarity:" in result.output

    def test_analyze_only_shows_content_type(
        self, runner: CliRunner, loopable_wav_analyze: str
    ) -> None:
        """--analyze-only must include 'Content type:' in output."""
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only"])
        assert result.exit_code == 0
        assert "Content type:" in result.output

    def test_analyze_only_no_output_file_created(
        self, runner: CliRunner, loopable_wav_analyze: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--analyze-only must not create any output WAV file."""
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only"])
        assert result.exit_code == 0
        # The default output would be loopable_analyze_loop.wav in the same dir.
        import pathlib

        input_dir = pathlib.Path(loopable_wav_analyze).parent
        loop_files = list(input_dir.glob("*_loop.wav"))
        assert loop_files == [], f"Unexpected output files created: {loop_files}"

    def test_analyze_only_with_output_flag_ignored(
        self, runner: CliRunner, loopable_wav_analyze: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--analyze-only with -o must ignore -o and not write any file."""
        out = str(tmp_path / "should_not_exist.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only", "-o", out])
        assert result.exit_code == 0
        import pathlib

        assert not pathlib.Path(out).exists(), "Output file was created despite --analyze-only"

    def test_analyze_only_exit_code_zero(
        self, runner: CliRunner, loopable_wav_analyze: str
    ) -> None:
        """--analyze-only must exit with code 0 (AC-2)."""
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only"])
        assert result.exit_code == 0

    def test_analyze_only_shows_suggested_commands(
        self, runner: CliRunner, loopable_wav_analyze: str
    ) -> None:
        """--analyze-only output must include a 'Suggested' section."""
        result = runner.invoke(main, [loopable_wav_analyze, "--analyze-only"])
        assert result.exit_code == 0
        assert "Suggested" in result.output


class TestFileNotFound:
    """Tests for missing file error handling (STORY-002 AC-2)."""

    def test_nonexistent_file_exits_1(self, runner: CliRunner) -> None:
        """A path that does not exist must produce exit code 1."""
        result = runner.invoke(main, ["/no/such/file.wav"])
        assert result.exit_code == 1

    def test_nonexistent_file_error_message(self, runner: CliRunner) -> None:
        """A missing file must print 'file was not found' (stderr is mixed into output)."""
        result = runner.invoke(main, ["/no/such/file.wav"])
        assert "file was not found" in result.output


class TestInvalidFile:
    """Tests for corrupt/non-audio file error handling (STORY-002 AC-3)."""

    def test_invalid_file_exits_1(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """A text file renamed to .wav must produce exit code 1."""
        bad_wav = tmp_path / "bad.wav"  # type: ignore[operator]
        bad_wav.write_text("this is not audio")
        result = runner.invoke(main, [str(bad_wav)])
        assert result.exit_code == 1

    def test_invalid_file_error_message(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """A corrupt file must print 'could not be read as audio' (stderr mixed into output)."""
        bad_wav = tmp_path / "bad.wav"  # type: ignore[operator]
        bad_wav.write_text("this is not audio")
        result = runner.invoke(main, [str(bad_wav)])
        assert "could not be read as audio" in result.output


# ---------------------------------------------------------------------------
# STORY-003: export / output path tests
# ---------------------------------------------------------------------------


class TestOutputFlag:
    """Tests for -o / --output flag (STORY-003 AC-1)."""

    def test_output_flag_writes_to_specified_path(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When -o is given, the output file must be written at that path."""
        out = str(tmp_path / "explicit_out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        assert info.samplerate == 48000

    def test_output_flag_preserves_sample_rate(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The output file written via -o must preserve the 48kHz sample rate."""
        out = str(tmp_path / "sr_out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav, "-o", out])
        info = sf.info(out)
        assert info.samplerate == 48000

    def test_output_flag_preserves_bit_depth(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The output file written via -o must preserve the 24-bit depth."""
        out = str(tmp_path / "bd_out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav, "-o", out])
        info = sf.info(out)
        assert info.subtype == "PCM_24"

    def test_output_flag_preserves_channels(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The output file written via -o must preserve the stereo channel count."""
        out = str(tmp_path / "ch_out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav, "-o", out])
        info = sf.info(out)
        assert info.channels == 2

    def test_output_path_printed_to_stdout(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """The resolved output path must appear in CLI stdout output."""
        out = str(tmp_path / "printed_out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "-o", out])
        assert out in result.output


class TestDefaultOutputPath:
    """Tests for default output path naming (STORY-003 AC-2)."""

    def _make_loopable_samples(self) -> np.ndarray:
        """Return 5 seconds of stereo 440 Hz sine at 48kHz as float64."""
        sr = 48000
        duration = 5.0
        num_samples = int(sr * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        return np.stack([mono, mono], axis=1)

    def test_default_output_has_loop_suffix(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When no -o is given, the output file name must include the '_loop' suffix."""
        input_path = tmp_path / "track01.wav"  # type: ignore[operator]
        sf.write(str(input_path), self._make_loopable_samples(), 48000, subtype="PCM_24")
        result = runner.invoke(main, [str(input_path)])
        assert result.exit_code == 0
        assert "track01_loop.wav" in result.output

    def test_default_output_written_to_input_directory(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When no -o is given, the output file must exist in the same dir as input."""
        input_path = tmp_path / "track01.wav"  # type: ignore[operator]
        sf.write(str(input_path), self._make_loopable_samples(), 48000, subtype="PCM_24")
        runner.invoke(main, [str(input_path)])
        expected = tmp_path / "track01_loop.wav"  # type: ignore[operator]
        assert expected.exists()


class TestOverwriteExistingOutput:
    """Tests for overwrite behaviour (STORY-003 AC-3, updated by STORY-015)."""

    def test_overwrite_existing_output_with_flag(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Writing to an existing output path with --overwrite must succeed."""
        out = tmp_path / "existing.wav"  # type: ignore[operator]
        out.write_bytes(b"placeholder")  # type: ignore[union-attr]
        result = runner.invoke(main, [loopable_wav, "-o", str(out), "--overwrite"])
        assert result.exit_code == 0
        # After overwrite the file should be a valid WAV, not the placeholder bytes.
        info = sf.info(str(out))
        assert info.samplerate == 48000

    def test_overwrite_existing_output_without_flag_exits_1(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Writing to an existing output path without --overwrite must exit 1."""
        out = tmp_path / "existing.wav"  # type: ignore[operator]
        out.write_bytes(b"placeholder")  # type: ignore[union-attr]
        result = runner.invoke(main, [loopable_wav, "-o", str(out)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# STORY-014: Verbosity / quiet output mode tests
# ---------------------------------------------------------------------------


class TestQuietModeOutput:
    """Tests for --quiet flag suppression (STORY-014 AC-4)."""

    def test_quiet_mode_no_stdout_on_success(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--quiet must produce no stdout output on success."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "-o", out, "--quiet"])
        assert result.exit_code == 0
        assert result.output == ""

    def test_quiet_mode_error_still_shown_on_missing_file(self, runner: CliRunner) -> None:
        """--quiet must still show error messages on failure."""
        result = runner.invoke(main, ["/no/such/file.wav", "--quiet"])
        assert result.exit_code == 1
        # CliRunner mixes stderr into result.output
        assert "file was not found" in result.output

    def test_quiet_mode_no_stdout_on_analyze_only(self, runner: CliRunner, stereo_wav: str) -> None:
        """--quiet with --analyze-only must produce no stdout output."""
        result = runner.invoke(main, [stereo_wav, "--analyze-only", "--quiet"])
        assert result.exit_code == 0
        assert result.output == ""


class TestVerboseFlagAccepted:
    """Tests that -v and -vv are accepted without error (STORY-014 AC-2/3)."""

    def test_single_verbose_flag_accepted(self, runner: CliRunner, loopable_wav: str) -> None:
        """-v must not cause an error exit."""
        result = runner.invoke(main, [loopable_wav, "-v"])
        assert result.exit_code == 0

    def test_double_verbose_flag_accepted(self, runner: CliRunner, loopable_wav: str) -> None:
        """-vv must not cause an error exit."""
        result = runner.invoke(main, [loopable_wav, "-vv"])
        assert result.exit_code == 0

    def test_verbose_long_form_accepted(self, runner: CliRunner, loopable_wav: str) -> None:
        """--verbose must not cause an error exit."""
        result = runner.invoke(main, [loopable_wav, "--verbose"])
        assert result.exit_code == 0


class TestVerboseQuietMutualExclusion:
    """Re-verify --verbose and --quiet remain mutually exclusive (STORY-014 AC-5)."""

    def test_verbose_and_quiet_exit_nonzero(self, runner: CliRunner, stereo_wav: str) -> None:
        """--verbose and --quiet together must produce a non-zero exit code."""
        result = runner.invoke(main, [stereo_wav, "--verbose", "--quiet"])
        assert result.exit_code != 0

    def test_verbose_and_quiet_error_message_mentions_mutual_exclusion(
        self, runner: CliRunner, stereo_wav: str
    ) -> None:
        """--verbose and --quiet together must mention mutual exclusivity."""
        result = runner.invoke(main, [stereo_wav, "--verbose", "--quiet"])
        assert "mutually exclusive" in result.output.lower()

    def test_quiet_and_verbose_order_does_not_matter(
        self, runner: CliRunner, stereo_wav: str
    ) -> None:
        """Passing --quiet before --verbose must also fail with mutual exclusion."""
        result = runner.invoke(main, [stereo_wav, "--quiet", "--verbose"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# STORY-010: Manual loop point overrides
# ---------------------------------------------------------------------------


@pytest.fixture()
def loopable_wav_5s(tmp_path: pytest.TempPathFactory) -> str:
    """Create a 5-second loopable WAV file for manual override tests.

    Returns:
        Absolute path to the WAV file as a string.
    """
    sr = 48000
    duration = 5.0
    path = str(tmp_path / "loopable_5s.wav")  # type: ignore[operator]
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")
    return path


class TestManualLoopPointOverrides:
    """Tests for --start and --end manual loop point overrides (STORY-010)."""

    def test_start_and_end_override_exits_zero(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--start and --end together must produce a successful result."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--start", "1s", "--end", "3s", "-o", out])
        assert result.exit_code == 0

    def test_start_and_end_override_writes_output(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--start and --end together must write a valid WAV output file."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav_5s, "--start", "1s", "--end", "3s", "-o", out])
        info = sf.info(out)
        assert info.samplerate == 48000

    def test_start_and_end_override_bypasses_auto_detection(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When both --start and --end are given, output must show 'manual' in loop region line."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--start", "1s", "--end", "3s", "-o", out])
        assert "manual" in result.output.lower()

    def test_start_and_end_minutes_seconds_format(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--start 0:30 --end 1:30 format must be accepted (AC-1 format support)."""
        # File is only 5s so use values within range.
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(
            main, [loopable_wav_5s, "--start", "0:01", "--end", "0:03", "-o", out]
        )
        assert result.exit_code == 0

    def test_start_only_partial_override_exits_zero(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--start without --end must still complete successfully (AC-2)."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--start", "1s", "-o", out])
        assert result.exit_code == 0

    def test_start_only_partial_override_writes_output(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--start without --end must produce a valid output WAV (AC-2)."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav_5s, "--start", "1s", "-o", out])
        info = sf.info(out)
        assert info.samplerate == 48000

    def test_start_after_end_exits_2(self, runner: CliRunner, loopable_wav_5s: str) -> None:
        """--start >= --end must exit with code 2 (AC-3)."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "60s", "--end", "30s"])
        assert result.exit_code == 2

    def test_start_after_end_error_message(self, runner: CliRunner, loopable_wav_5s: str) -> None:
        """--start >= --end must print 'start must be before end' (AC-3)."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "3s", "--end", "1s"])
        assert "start must be before end" in result.output

    def test_start_equal_to_end_exits_2(self, runner: CliRunner, loopable_wav_5s: str) -> None:
        """--start == --end must exit with code 2 (AC-3)."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "2s", "--end", "2s"])
        assert result.exit_code == 2

    def test_position_beyond_duration_start_exits_2(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--start beyond file duration must exit with code 2 (AC-4)."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "9999s"])
        assert result.exit_code == 2

    def test_position_beyond_duration_end_exits_2(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--end beyond file duration must exit with code 2 (AC-4)."""
        result = runner.invoke(main, [loopable_wav_5s, "--end", "9999s"])
        assert result.exit_code == 2

    def test_position_beyond_duration_error_message(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--start beyond file duration must mention 'beyond end of file' (AC-4)."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "9999s"])
        assert "beyond end of file" in result.output

    def test_invalid_start_format_exits_2(self, runner: CliRunner, loopable_wav_5s: str) -> None:
        """An unparseable --start value must exit with code 2."""
        result = runner.invoke(main, [loopable_wav_5s, "--start", "abc"])
        assert result.exit_code == 2

    def test_invalid_end_format_exits_2(self, runner: CliRunner, loopable_wav_5s: str) -> None:
        """An unparseable --end value must exit with code 2."""
        result = runner.invoke(main, [loopable_wav_5s, "--end", "xyz"])
        assert result.exit_code == 2

    def test_plain_number_accepted_for_start(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """A plain number (e.g. '1') must be accepted as seconds for --start."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--start", "1", "--end", "3", "-o", out])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# STORY-012: Loop repetition and target duration output
# ---------------------------------------------------------------------------


@pytest.fixture()
def loopable_wav_long(tmp_path: pytest.TempPathFactory) -> str:
    """Create a 10-second loopable stereo WAV file for STORY-012 tests.

    Returns:
        Absolute path to the WAV file as a string.
    """
    sr = 48000
    duration = 10.0
    path = str(tmp_path / "loopable_10s.wav")  # type: ignore[operator]
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")
    return path


class TestCountFlag:
    """Tests for --count flag behaviour (STORY-012 AC-1, AC-4)."""

    def test_count_flag_exits_zero(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--count N must complete successfully."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "--count", "4", "-o", out])
        assert result.exit_code == 0

    def test_count_flag_produces_repeated_output(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--count 4 must produce an output longer than a single iteration."""
        out_single = str(tmp_path / "single.wav")  # type: ignore[operator]
        out_quad = str(tmp_path / "quad.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav, "--count", "1", "-o", out_single])
        runner.invoke(main, [loopable_wav, "--count", "4", "-o", out_quad])
        single_info = sf.info(out_single)
        quad_info = sf.info(out_quad)
        # 4 reps must be longer than 1 rep (at least 3x to account for crossfade shrinkage).
        assert quad_info.frames > single_info.frames * 3

    def test_count_flag_n_alias_works(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """-n must be accepted as an alias for --count."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "-n", "2", "-o", out])
        assert result.exit_code == 0

    def test_default_count_is_4(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Default (no --count) must produce the same output as --count 4 (AC-4)."""
        out_default = str(tmp_path / "default.wav")  # type: ignore[operator]
        out_count4 = str(tmp_path / "count4.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav, "-o", out_default])
        runner.invoke(main, [loopable_wav, "--count", "4", "-o", out_count4])
        default_info = sf.info(out_default)
        count4_info = sf.info(out_count4)
        assert default_info.frames == count4_info.frames

    def test_count_output_duration_reported(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """CLI output must report 'Output duration:' after processing."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "--count", "2", "-o", out])
        assert result.exit_code == 0
        assert "Output duration:" in result.output


class TestDurationFlag:
    """Tests for --duration flag behaviour (STORY-012 AC-2)."""

    def test_duration_flag_exits_zero(
        self, runner: CliRunner, loopable_wav_long: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--duration must complete successfully."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_long, "--duration", "30s", "-o", out])
        assert result.exit_code == 0

    def test_duration_flag_produces_target_length(
        self, runner: CliRunner, loopable_wav_long: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--duration 30s must produce output approximately 30 seconds long."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_long, "--duration", "30s", "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        output_duration = info.frames / info.samplerate
        # Allow +-loop-length tolerance (loop is < 10s for a 10s file).
        assert abs(output_duration - 30.0) < 11.0

    def test_duration_flag_output_does_not_exceed_target(
        self, runner: CliRunner, loopable_wav_long: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--duration target must not be exceeded in output length."""
        target = 30.0
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav_long, "--duration", f"{target}s", "-o", out])
        info = sf.info(out)
        output_duration = info.frames / info.samplerate
        assert output_duration <= target + 0.05

    def test_duration_flag_d_alias_works(
        self, runner: CliRunner, loopable_wav_long: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """-d must be accepted as an alias for --duration."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_long, "-d", "20s", "-o", out])
        assert result.exit_code == 0

    def test_duration_output_duration_reported(
        self, runner: CliRunner, loopable_wav_long: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """CLI output must report 'Output duration:' after processing."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_long, "--duration", "20s", "-o", out])
        assert result.exit_code == 0
        assert "Output duration:" in result.output

    def test_duration_invalid_format_exits_2(
        self, runner: CliRunner, loopable_wav_long: str
    ) -> None:
        """An unparseable --duration value must exit with code 2."""
        result = runner.invoke(main, [loopable_wav_long, "--duration", "notaduration"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# STORY-011: --loop-length hint (time-based and bar-based)
# ---------------------------------------------------------------------------


class TestLoopLengthOption:
    """Tests for --loop-length hint (STORY-011 acceptance criteria).

    Uses the existing ``loopable_wav_5s`` fixture (440 Hz sine, 5 s), which
    classifies as AMBIENT and produces no detectable tempo — ideal for both
    the time-based success path and the bar-based no-tempo error path.
    """

    def test_loop_length_time_based_exits_zero(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--loop-length 3s on a loopable file must exit with code 0."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "3s", "-o", out])
        assert result.exit_code == 0

    def test_loop_length_time_based_produces_output(
        self, runner: CliRunner, loopable_wav_5s: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--loop-length 3s must produce 'Processing:' in output."""
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "3s", "-o", out])
        assert "Processing:" in result.output

    def test_loop_length_exceeds_duration_exits_2(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--loop-length larger than file duration must exit with code 2."""
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "999s"])
        assert result.exit_code == 2

    def test_loop_length_exceeds_duration_error_message(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--loop-length larger than file duration must print an error mentioning 'duration'."""
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "999s"])
        assert "duration" in result.output.lower() or "exceeds" in result.output.lower()

    def test_loop_length_bars_no_tempo_exits_1(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--loop-length 4bars on ambient audio (no tempo) must exit with code 1."""
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "4bars"])
        assert result.exit_code == 1

    def test_loop_length_bars_no_tempo_error_mentions_tempo(
        self, runner: CliRunner, loopable_wav_5s: str
    ) -> None:
        """--loop-length 4bars with no tempo must print an error mentioning 'tempo'."""
        result = runner.invoke(main, [loopable_wav_5s, "--loop-length", "4bars"])
        assert "tempo" in result.output.lower()

    def test_help_mentions_bar_syntax(self, runner: CliRunner) -> None:
        """--help output must mention 'bars' or 'bar' in the loop-length description."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "bar" in result.output.lower()

    def test_help_mentions_4_4_assumption(self, runner: CliRunner) -> None:
        """--help output for --loop-length must mention '4/4' time assumption."""
        result = runner.invoke(main, ["--help"])
        assert "4/4" in result.output


# ---------------------------------------------------------------------------
# STORY-016: --max-file-size CLI flag
# ---------------------------------------------------------------------------


class TestMaxFileSizeFlag:
    """Tests for the --max-file-size option (STORY-016 AC-1, AC-2, AC-4)."""

    def test_help_lists_max_file_size_option(self, runner: CliRunner) -> None:
        """--help output must list the --max-file-size option."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--max-file-size" in result.output

    def test_max_file_size_flag_accepted(self, runner: CliRunner, loopable_wav: str) -> None:
        """Passing --max-file-size with a large value must not cause a CLI error."""
        result = runner.invoke(main, [loopable_wav, "--max-file-size", "4096"])
        # Should succeed (exit 0) — the file is well under 4 GB.
        assert result.exit_code == 0

    def test_max_file_size_zero_rejects_any_file(
        self, runner: CliRunner, loopable_wav: str
    ) -> None:
        """--max-file-size 0 must reject even a tiny WAV file with exit code 1."""
        result = runner.invoke(main, [loopable_wav, "--max-file-size", "0"])
        assert result.exit_code == 1

    def test_max_file_size_zero_prints_error_message(
        self, runner: CliRunner, loopable_wav: str
    ) -> None:
        """--max-file-size 0 must print an error mentioning 'exceeds' or 'maximum size'."""
        result = runner.invoke(main, [loopable_wav, "--max-file-size", "0"])
        assert "exceeds" in result.output.lower() or "maximum size" in result.output.lower()

    def test_max_file_size_default_shown_in_help(self, runner: CliRunner) -> None:
        """--help output must show the default value for --max-file-size."""
        result = runner.invoke(main, ["--help"])
        assert "2048" in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_temp_wav(runner: CliRunner) -> str:
    """Create a temporary empty file (legacy helper, kept for reference).

    Returns:
        Absolute path to the temporary file as a string.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# STORY-017: --count upper bound validation
# ---------------------------------------------------------------------------


class TestCountUpperBound:
    """Tests for --count upper bound validation (STORY-017 acceptance criteria)."""

    def test_count_10001_exits_2(self, runner: CliRunner, loopable_wav: str) -> None:
        """--count 10001 must exit with code 2 (exceeds maximum)."""
        result = runner.invoke(main, [loopable_wav, "--count", "10001"])
        assert result.exit_code == 2

    def test_count_10001_error_mentions_maximum(self, runner: CliRunner, loopable_wav: str) -> None:
        """--count 10001 error message must mention 'exceeds maximum'."""
        result = runner.invoke(main, [loopable_wav, "--count", "10001"])
        assert "exceeds maximum" in result.output.lower() or "exceeds" in result.output

    def test_count_10000_accepted(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--count 10000 must proceed without a bounds error (exit code not 2 from bounds check).

        The test confirms the CLI does not reject the value at the validation gate.
        Runtime may succeed or fail for other reasons but must not exit 2 due to
        the 'exceeds maximum' guard.
        """
        out = str(tmp_path / "out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "--count", "10000", "-o", out])
        # Must not exit 2 with "exceeds maximum" message.
        if result.exit_code == 2:
            assert "exceeds maximum" not in result.output, (
                "--count 10000 must not be rejected by the upper-bound guard"
            )

    def test_count_0_exits_2(self, runner: CliRunner, loopable_wav: str) -> None:
        """--count 0 must exit with code 2."""
        result = runner.invoke(main, [loopable_wav, "--count", "0"])
        assert result.exit_code == 2

    def test_count_0_error_mentions_at_least_1(self, runner: CliRunner, loopable_wav: str) -> None:
        """--count 0 error message must mention 'at least 1'."""
        result = runner.invoke(main, [loopable_wav, "--count", "0"])
        assert "at least 1" in result.output

    def test_count_negative_exits_2(self, runner: CliRunner, loopable_wav: str) -> None:
        """--count -1 must exit with code 2."""
        result = runner.invoke(main, [loopable_wav, "--count", "-1"])
        assert result.exit_code == 2

    def test_count_negative_error_mentions_at_least_1(
        self, runner: CliRunner, loopable_wav: str
    ) -> None:
        """--count -1 error message must mention 'at least 1'."""
        result = runner.invoke(main, [loopable_wav, "--count", "-1"])
        assert "at least 1" in result.output


# ---------------------------------------------------------------------------
# STORY-015: Output file overwrite protection
# ---------------------------------------------------------------------------


class TestOverwriteFlag:
    """Tests for --overwrite flag (STORY-015 acceptance criteria)."""

    def test_single_file_refuses_overwrite_default(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Without --overwrite, writing to an existing output must exit 1 (AC-1)."""
        out = tmp_path / "out.wav"  # type: ignore[operator]
        out.write_bytes(b"placeholder")  # type: ignore[union-attr]
        result = runner.invoke(main, [loopable_wav, "-o", str(out)])
        assert result.exit_code == 1

    def test_single_file_refuses_overwrite_error_message(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Without --overwrite, error message must say 'output file already exists' (AC-1)."""
        out = tmp_path / "out.wav"  # type: ignore[operator]
        out.write_bytes(b"placeholder")  # type: ignore[union-attr]
        result = runner.invoke(main, [loopable_wav, "-o", str(out)])
        assert "output file already exists" in result.output

    def test_single_file_overwrites_with_flag(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """With --overwrite, an existing output must be replaced and exit 0 (AC-2)."""
        out = tmp_path / "out.wav"  # type: ignore[operator]
        out.write_bytes(b"placeholder")  # type: ignore[union-attr]
        result = runner.invoke(main, [loopable_wav, "-o", str(out), "--overwrite"])
        assert result.exit_code == 0
        info = sf.info(str(out))
        assert info.samplerate == 48000

    def test_overwrite_flag_in_help(self, runner: CliRunner) -> None:
        """--help output must list the --overwrite flag."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--overwrite" in result.output

    def test_no_existing_output_succeeds_without_flag(
        self, runner: CliRunner, loopable_wav: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When the output does not yet exist, the default (no --overwrite) must succeed."""
        out = str(tmp_path / "new_out.wav")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav, "-o", out])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# STORY-018: Edge-case CLI tests
# ---------------------------------------------------------------------------


class TestEdgeCaseCLI:
    """Tests for CLI behaviour on edge-case audio fixtures (STORY-018)."""

    def test_empty_wav_exits_1(self, runner: CliRunner, empty_wav: Path) -> None:
        """The CLI must exit 1 when the input WAV contains no audio frames."""
        result = runner.invoke(main, [str(empty_wav)])
        assert result.exit_code == 1

    def test_empty_wav_error_message(self, runner: CliRunner, empty_wav: Path) -> None:
        """The CLI must print an error message for an empty WAV."""
        result = runner.invoke(main, [str(empty_wav)])
        assert "no audio data" in result.output

    def test_malformed_wav_exits_1(self, runner: CliRunner, malformed_wav: Path) -> None:
        """The CLI must exit 1 when the input file is not a valid WAV."""
        result = runner.invoke(main, [str(malformed_wav)])
        assert result.exit_code == 1

    def test_very_short_wav_exits_nonzero(self, runner: CliRunner, very_short_wav: Path) -> None:
        """The CLI must exit non-zero for a 0.1 s WAV (too short for loop detection)."""
        result = runner.invoke(main, [str(very_short_wav)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# STORY-022: --time-signature flag
# ---------------------------------------------------------------------------


class TestTimeSignatureFlag:
    """Tests for --time-signature flag (STORY-022 acceptance criteria)."""

    def test_time_signature_flag_accepted(self, runner: CliRunner, loopable_wav: str) -> None:
        """--time-signature 3/4 must be accepted without error."""
        result = runner.invoke(main, [loopable_wav, "--time-signature", "3/4"])
        assert result.exit_code == 0

    def test_time_signature_invalid_exits_2(self, runner: CliRunner, loopable_wav: str) -> None:
        """--time-signature abc must exit with code 2 (invalid format)."""
        result = runner.invoke(main, [loopable_wav, "--time-signature", "abc"])
        assert result.exit_code == 2

    def test_time_signature_zero_numerator_exits_2(
        self, runner: CliRunner, loopable_wav: str
    ) -> None:
        """--time-signature 0/4 must exit with code 2 (numerator must be >= 1)."""
        result = runner.invoke(main, [loopable_wav, "--time-signature", "0/4"])
        assert result.exit_code == 2

    def test_time_signature_bad_denominator_exits_2(
        self, runner: CliRunner, loopable_wav: str
    ) -> None:
        """--time-signature 4/3 must exit with code 2 (denominator not in 2,4,8,16)."""
        result = runner.invoke(main, [loopable_wav, "--time-signature", "4/3"])
        assert result.exit_code == 2

    def test_time_signature_in_help(self, runner: CliRunner) -> None:
        """--help output must list the --time-signature option."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--time-signature" in result.output


# ---------------------------------------------------------------------------
# STORY-021: Multi-format write support with --format flag
# ---------------------------------------------------------------------------


@pytest.fixture()
def loopable_wav_format(tmp_path: pytest.TempPathFactory) -> str:
    """Create a 5-second loopable stereo 24-bit/48kHz WAV for format tests.

    Returns:
        Absolute path to the WAV file as a string.
    """
    sr = 48000
    duration = 5.0
    path = str(tmp_path / "loopable_format.wav")  # type: ignore[operator]
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_24")
    return path


class TestFormatFlagFlac:
    """AC-1: audioloop input.wav --format flac produces a valid FLAC output."""

    def test_format_flac_exit_code_zero(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--format flac must exit with code 0."""
        out = str(tmp_path / "out.flac")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_format, "--format", "flac", "-o", out])
        assert result.exit_code == 0

    def test_format_flac_produces_flac_file(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--format flac must write a FLAC-format file."""
        out = str(tmp_path / "out.flac")  # type: ignore[operator]
        runner.invoke(main, [loopable_wav_format, "--format", "flac", "-o", out])
        info = sf.info(out)
        assert info.format == "FLAC"

    def test_format_flac_default_output_name_has_flac_extension(
        self, runner: CliRunner, loopable_wav_format: str
    ) -> None:
        """--format flac without -o must produce a file named *_loop.flac."""
        result = runner.invoke(main, [loopable_wav_format, "--format", "flac"])
        assert result.exit_code == 0
        import pathlib

        out = pathlib.Path(loopable_wav_format).parent / "loopable_format_loop.flac"
        assert out.exists(), f"Expected {out} to exist"
        info = sf.info(str(out))
        assert info.format == "FLAC"

    def test_format_flac_in_help(self, runner: CliRunner) -> None:
        """--help output must list the --format option."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "--format" in result.output


class TestFormatFromExtension:
    """AC-2: audioloop input.wav -o output.ogg infers OGG format from extension."""

    def test_ogg_extension_produces_ogg(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """-o output.ogg must produce an OGG-format file."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        out = str(tmp_path / "output.ogg")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_format, "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        assert info.format == "OGG"

    def test_flac_extension_produces_flac(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """-o output.flac must produce a FLAC-format file."""
        out = str(tmp_path / "output.flac")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_format, "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        assert info.format == "FLAC"


class TestDefaultFormatWav:
    """AC-3: audioloop input.flac (no --format, no -o) defaults to *_loop.wav."""

    def test_flac_input_default_output_is_wav(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """A FLAC input with no --format must produce a .wav output named *_loop.wav."""
        # Create a loopable FLAC input file.
        sr = 48000
        duration = 5.0
        flac_path = tmp_path / "input.flac"  # type: ignore[operator]
        num_samples = int(sr * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        stereo = np.stack([mono, mono], axis=1)
        sf.write(str(flac_path), stereo, sr, format="FLAC", subtype="PCM_24")

        result = runner.invoke(main, [str(flac_path)])
        assert result.exit_code == 0
        expected = tmp_path / "input_loop.wav"  # type: ignore[operator]
        assert expected.exists(), f"Expected {expected} to exist"
        info = sf.info(str(expected))
        assert info.format == "WAV"

    def test_wav_input_no_format_produces_wav(
        self, runner: CliRunner, loopable_wav_format: str
    ) -> None:
        """A WAV input with no --format must produce a .wav output."""
        result = runner.invoke(main, [loopable_wav_format])
        assert result.exit_code == 0
        import pathlib

        out = pathlib.Path(loopable_wav_format).parent / "loopable_format_loop.wav"
        assert out.exists()
        info = sf.info(str(out))
        assert info.format == "WAV"


class TestFormatFlagOverridesExtension:
    """AC-4: --format ogg with -o output.flac writes OGG (--format takes precedence)."""

    def test_format_flag_overrides_output_extension(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--format ogg with -o output.flac must write an OGG file, not FLAC."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        out = str(tmp_path / "output.flac")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_format, "--format", "ogg", "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        assert info.format == "OGG"

    def test_format_wav_overrides_ogg_extension(
        self, runner: CliRunner, loopable_wav_format: str, tmp_path: pytest.TempPathFactory
    ) -> None:
        """--format wav with -o output.ogg must write a WAV file, not OGG."""
        out = str(tmp_path / "output.ogg")  # type: ignore[operator]
        result = runner.invoke(main, [loopable_wav_format, "--format", "wav", "-o", out])
        assert result.exit_code == 0
        info = sf.info(out)
        assert info.format == "WAV"

    def test_invalid_format_choice_exits_2(
        self, runner: CliRunner, loopable_wav_format: str
    ) -> None:
        """An invalid --format value must produce exit code 2."""
        result = runner.invoke(main, [loopable_wav_format, "--format", "mp3"])
        assert result.exit_code == 2
