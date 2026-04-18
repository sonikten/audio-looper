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

"""Unit and integration tests for batch processing (STORY-013, STORY-020)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from audioloop.batch import (
    SUPPORTED_EXTENSIONS,
    discover_audio_files,
    discover_wav_files,
    run_batch,
)
from audioloop.cli import main
from audioloop.exceptions import BatchError
from audioloop.reporting import Reporter, VerbosityLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_loopable_wav(path: Path, duration: float = 5.0, sr: int = 22050) -> Path:
    """Write a mono sine-wave WAV to *path*.

    A sine wave has no transients, so MFCC cosine similarity between any two
    windows is ~1.0 — ``detect_loop_points`` will always succeed.  Duration
    must be > 2 s (MIN_LOOP_SECONDS) for the loop detector to find candidates.
    """
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(str(path), mono, sr, subtype="PCM_16")
    return path


def make_batch_dir(tmp_path: Path, n: int = 3, duration: float = 5.0) -> Path:
    """Create a directory containing *n* loopable WAV files."""
    batch_dir = tmp_path / "batch_input"
    batch_dir.mkdir()
    for i in range(n):
        make_loopable_wav(batch_dir / f"track_{i:02d}.wav", duration=duration)
    return batch_dir


# ---------------------------------------------------------------------------
# discover_wav_files
# ---------------------------------------------------------------------------


class TestDiscoverWavFiles:
    """Unit tests for discover_wav_files."""

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """Files are returned in sorted order."""
        d = tmp_path / "wavs"
        d.mkdir()
        for name in ("c.wav", "a.wav", "b.wav"):
            make_loopable_wav(d / name)
        result = discover_wav_files(d)
        assert [f.name for f in result] == ["a.wav", "b.wav", "c.wav"]

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Files with .WAV or .Wav extensions are included."""
        d = tmp_path / "wavs"
        d.mkdir()
        make_loopable_wav(d / "lower.wav")
        # Copy file and rename to uppercase extension.
        upper = d / "upper.WAV"
        upper.write_bytes((d / "lower.wav").read_bytes())
        result = discover_wav_files(d)
        names = [f.name for f in result]
        assert "lower.wav" in names
        assert "upper.WAV" in names

    def test_skips_non_wav_files(self, tmp_path: Path) -> None:
        """Non-WAV files in the directory are ignored."""
        d = tmp_path / "mixed"
        d.mkdir()
        make_loopable_wav(d / "audio.wav")
        (d / "notes.txt").write_text("not audio")
        (d / "image.png").write_bytes(b"\x89PNG")
        result = discover_wav_files(d)
        assert len(result) == 1
        assert result[0].name == "audio.wav"

    def test_nonexistent_directory_raises_batch_error(self, tmp_path: Path) -> None:
        """BatchError is raised when the directory does not exist."""
        missing = tmp_path / "nonexistent"
        with pytest.raises(BatchError, match="Directory not found"):
            discover_wav_files(missing)

    def test_empty_directory_raises_batch_error(self, tmp_path: Path) -> None:
        """BatchError is raised when the directory contains no WAV files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(BatchError, match="No WAV files found"):
            discover_wav_files(empty)

    def test_directory_with_only_non_wav_raises_batch_error(self, tmp_path: Path) -> None:
        """BatchError is raised when files exist but none are WAV."""
        d = tmp_path / "no_wav"
        d.mkdir()
        (d / "readme.txt").write_text("hello")
        with pytest.raises(BatchError, match="No WAV files found"):
            discover_wav_files(d)


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


class TestRunBatchAllSucceed:
    """AC-3: all files succeed -> BatchResult.successful == total, failed == 0."""

    def test_all_files_processed(self, tmp_path: Path) -> None:
        """run_batch returns a result with successful == 3 when all files process cleanly."""
        batch_dir = make_batch_dir(tmp_path, n=3)
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.results) == 3

    def test_output_files_exist(self, tmp_path: Path) -> None:
        """Each successful FileResult has an output_path that exists on disk."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        for fr in result.results:
            assert fr.success
            assert fr.output_path is not None
            assert fr.output_path.exists()

    def test_output_filenames_have_loop_suffix(self, tmp_path: Path) -> None:
        """Output file stems end with _loop."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        for fr in result.results:
            assert fr.output_path is not None
            assert fr.output_path.stem.endswith("_loop")

    def test_output_placed_alongside_input_by_default(self, tmp_path: Path) -> None:
        """Without output_dir, outputs land next to the source files."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        for fr in result.results:
            assert fr.output_path is not None
            assert fr.output_path.parent == batch_dir


class TestRunBatchWithOutputDir:
    """AC-5: -o output_dir writes outputs to the specified directory."""

    def test_outputs_written_to_output_dir(self, tmp_path: Path) -> None:
        """All outputs are placed in the specified output directory."""
        batch_dir = make_batch_dir(tmp_path, n=3)
        out_dir = tmp_path / "out"
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, output_dir=out_dir, reporter=reporter)
        assert result.successful == 3
        for fr in result.results:
            assert fr.output_path is not None
            assert fr.output_path.parent == out_dir

    def test_output_dir_created_when_missing(self, tmp_path: Path) -> None:
        """run_batch creates output_dir if it does not already exist."""
        batch_dir = make_batch_dir(tmp_path, n=1)
        out_dir = tmp_path / "new_dir" / "nested"
        reporter = Reporter(VerbosityLevel.QUIET)
        run_batch(batch_dir, output_dir=out_dir, reporter=reporter)
        assert out_dir.exists()

    def test_original_filename_preserved_in_output_dir(self, tmp_path: Path) -> None:
        """Output file name uses original stem + _loop suffix."""
        batch_dir = make_batch_dir(tmp_path, n=1)
        # Get name of the single file created.
        source_name = next(batch_dir.iterdir()).stem
        out_dir = tmp_path / "outputs"
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, output_dir=out_dir, reporter=reporter)
        fr = result.results[0]
        assert fr.output_path is not None
        assert fr.output_path.stem == f"{source_name}_loop"


class TestRunBatchPartialFailure:
    """AC-2: 3 succeed, 2 fail -> failed count == 2, results list shows each outcome."""

    def test_partial_failure_counts(self, tmp_path: Path) -> None:
        """Failed files are recorded without stopping the rest of the batch."""
        batch_dir = tmp_path / "partial"
        batch_dir.mkdir()
        # 3 valid loopable WAVs (named so they sort after the bad files).
        for i in range(3):
            make_loopable_wav(batch_dir / f"good_{i}.wav")
        # 2 corrupt WAVs (zero-length files are unreadable by soundfile).
        for i in range(2):
            (batch_dir / f"bad_{i}.wav").write_bytes(b"")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)

        assert result.successful == 3
        assert result.failed == 2
        assert len(result.results) == 5

    def test_failed_results_have_error_message(self, tmp_path: Path) -> None:
        """Each failed FileResult carries a non-empty error string."""
        batch_dir = tmp_path / "with_bad"
        batch_dir.mkdir()
        make_loopable_wav(batch_dir / "good.wav")
        (batch_dir / "bad.wav").write_bytes(b"")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        failed = [r for r in result.results if not r.success]
        assert len(failed) == 1
        assert failed[0].error  # non-empty string
        assert failed[0].output_path is None

    def test_successful_results_still_written(self, tmp_path: Path) -> None:
        """When some files fail, the good files are still written to disk."""
        batch_dir = tmp_path / "mixed"
        batch_dir.mkdir()
        make_loopable_wav(batch_dir / "good.wav")
        (batch_dir / "bad.wav").write_bytes(b"")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        succeeded = [r for r in result.results if r.success]
        assert len(succeeded) == 1
        assert succeeded[0].output_path is not None
        assert succeeded[0].output_path.exists()


class TestRunBatchAllFail:
    """All files are corrupt -> successful == 0, failed == count."""

    def test_all_fail(self, tmp_path: Path) -> None:
        """BatchResult.failed == total when every file is unreadable."""
        batch_dir = tmp_path / "all_bad"
        batch_dir.mkdir()
        for i in range(2):
            (batch_dir / f"bad_{i}.wav").write_bytes(b"")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        assert result.successful == 0
        assert result.failed == 2


class TestRunBatchErrors:
    """AC-4: nonexistent or empty directory -> BatchError."""

    def test_nonexistent_dir_raises(self, tmp_path: Path) -> None:
        """BatchError is raised immediately when the input directory is missing."""
        missing = tmp_path / "missing"
        reporter = Reporter(VerbosityLevel.QUIET)
        with pytest.raises(BatchError, match="Directory not found"):
            run_batch(missing, reporter=reporter)

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        """BatchError is raised immediately when the directory has no WAV files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        reporter = Reporter(VerbosityLevel.QUIET)
        with pytest.raises(BatchError, match="No WAV files found"):
            run_batch(empty, reporter=reporter)

    def test_dir_with_only_non_wav_raises(self, tmp_path: Path) -> None:
        """BatchError is raised when files exist but none are WAV."""
        d = tmp_path / "no_wav"
        d.mkdir()
        (d / "readme.txt").write_text("hello")
        reporter = Reporter(VerbosityLevel.QUIET)
        with pytest.raises(BatchError, match="No WAV files found"):
            run_batch(d, reporter=reporter)


class TestRunBatchSkipsNonWav:
    """Non-WAV files are silently skipped (not counted as failures)."""

    def test_non_wav_files_not_in_results(self, tmp_path: Path) -> None:
        """A .txt file in the batch directory is not included in results."""
        batch_dir = tmp_path / "with_txt"
        batch_dir.mkdir()
        make_loopable_wav(batch_dir / "audio.wav")
        (batch_dir / "notes.txt").write_text("ignore me")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        assert len(result.results) == 1
        assert result.results[0].input_path.name == "audio.wav"

    def test_non_wav_not_counted_as_failure(self, tmp_path: Path) -> None:
        """Non-WAV files do not inflate the failed count."""
        batch_dir = tmp_path / "skip_non_wav"
        batch_dir.mkdir()
        make_loopable_wav(batch_dir / "audio.wav")
        (batch_dir / "data.json").write_text("{}")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter)
        assert result.failed == 0
        assert result.successful == 1


# ---------------------------------------------------------------------------
# CLI integration: --batch flag
# ---------------------------------------------------------------------------


class TestCliBatchAllSucceed:
    """AC-1 + AC-3: batch mode all succeed -> exit code 0 + summary."""

    def test_exit_code_zero(self, tmp_path: Path) -> None:
        """Exit code is 0 when all files succeed."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        assert result.exit_code == 0

    def test_summary_shows_processed_count(self, tmp_path: Path) -> None:
        """Output contains a summary line with the processed count."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        assert "2/2" in result.output

    def test_output_files_written(self, tmp_path: Path) -> None:
        """All output files are written to disk."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        runner.invoke(main, [str(batch_dir), "--batch"])
        loop_files = list(batch_dir.glob("*_loop.wav"))
        assert len(loop_files) == 2


class TestCliBatchPartialFailure:
    """AC-2: partial failures -> exit code 3 + summary lists failed files."""

    def _make_partial_dir(self, tmp_path: Path) -> Path:
        batch_dir = tmp_path / "partial"
        batch_dir.mkdir()
        make_loopable_wav(batch_dir / "good_00.wav")
        make_loopable_wav(batch_dir / "good_01.wav")
        (batch_dir / "bad_00.wav").write_bytes(b"")
        return batch_dir

    def test_exit_code_three(self, tmp_path: Path) -> None:
        """Exit code is 3 for partial success."""
        batch_dir = self._make_partial_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        assert result.exit_code == 3

    def test_summary_mentions_failed_file(self, tmp_path: Path) -> None:
        """Output summary lists the name of the failed file."""
        batch_dir = self._make_partial_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        # The failed filename should appear somewhere in the combined output.
        combined = result.output + (result.stderr_bytes.decode() if result.stderr_bytes else "")
        assert "bad_00.wav" in combined


class TestCliBatchNonexistentDir:
    """AC-4: nonexistent directory -> exit code 1."""

    def test_exit_code_one_missing_dir(self, tmp_path: Path) -> None:
        """Exit code is 1 when the directory does not exist."""
        missing = str(tmp_path / "nope")
        runner = CliRunner()
        result = runner.invoke(main, [missing, "--batch"])
        assert result.exit_code == 1

    def test_exit_code_one_empty_dir(self, tmp_path: Path) -> None:
        """Exit code is 1 when the directory has no WAV files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, [str(empty), "--batch"])
        assert result.exit_code == 1


class TestCliBatchOutputDir:
    """AC-5: -o output_dir -> outputs written to specified directory."""

    def test_output_dir_used(self, tmp_path: Path) -> None:
        """Files are written to the -o directory."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        out_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(main, [str(batch_dir), "--batch", "-o", str(out_dir)])
        assert result.exit_code == 0
        assert out_dir.exists()
        loop_files = list(out_dir.glob("*_loop.wav"))
        assert len(loop_files) == 2

    def test_original_names_with_loop_suffix_in_output_dir(self, tmp_path: Path) -> None:
        """Original file names are preserved with _loop suffix in output dir."""
        batch_dir = make_batch_dir(tmp_path, n=1)
        source_stem = next(batch_dir.glob("*.wav")).stem
        out_dir = tmp_path / "out"
        runner = CliRunner()
        runner.invoke(main, [str(batch_dir), "--batch", "-o", str(out_dir)])
        expected = out_dir / f"{source_stem}_loop.wav"
        assert expected.exists()


# ---------------------------------------------------------------------------
# STORY-015: Overwrite protection in batch mode
# ---------------------------------------------------------------------------


class TestBatchSkipExistingOutputs:
    """AC-3/4: batch mode with/without --overwrite when outputs already exist."""

    def test_batch_skips_existing_outputs_without_overwrite(self, tmp_path: Path) -> None:
        """When outputs exist and --overwrite is absent, they are skipped (AC-3)."""
        batch_dir = make_batch_dir(tmp_path, n=5)
        out_dir = tmp_path / "out"
        reporter = Reporter(VerbosityLevel.QUIET)
        # First run writes all 5 outputs into a separate output directory.
        result1 = run_batch(batch_dir, output_dir=out_dir, reporter=reporter, overwrite=True)
        assert result1.successful == 5

        # Second run without overwrite — all 5 outputs already exist.
        result2 = run_batch(batch_dir, output_dir=out_dir, reporter=reporter, overwrite=False)
        assert result2.skipped == 5
        assert result2.successful == 0
        assert result2.failed == 0

    def test_batch_skips_three_of_five(self, tmp_path: Path) -> None:
        """3 of 5 outputs exist without --overwrite -> 3 skipped, 2 processed (AC-3)."""
        batch_dir = make_batch_dir(tmp_path, n=5)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        files = sorted(batch_dir.glob("*.wav"))
        reporter = Reporter(VerbosityLevel.QUIET)

        # Pre-create loop outputs for the first 3 files only.
        for f in files[:3]:
            (out_dir / f"{f.stem}_loop{f.suffix}").write_bytes(b"placeholder")

        result = run_batch(batch_dir, output_dir=out_dir, reporter=reporter, overwrite=False)
        assert result.skipped == 3
        assert result.successful == 2
        assert result.failed == 0
        assert len(result.results) == 5

    def test_batch_overwrite_reprocesses_all(self, tmp_path: Path) -> None:
        """With --overwrite, all outputs are reprocessed even when they already exist (AC-4)."""
        batch_dir = make_batch_dir(tmp_path, n=3)
        out_dir = tmp_path / "out"
        reporter = Reporter(VerbosityLevel.QUIET)
        # First run.
        run_batch(batch_dir, output_dir=out_dir, reporter=reporter)
        # Second run with overwrite=True.
        result = run_batch(batch_dir, output_dir=out_dir, reporter=reporter, overwrite=True)
        assert result.successful == 3
        assert result.skipped == 0
        assert result.failed == 0

    def test_skipped_file_result_has_skipped_flag(self, tmp_path: Path) -> None:
        """A skipped FileResult must have skipped=True and success=False."""
        batch_dir = make_batch_dir(tmp_path, n=1)
        source = next(batch_dir.glob("*.wav"))
        # Pre-create the output.
        out = batch_dir / f"{source.stem}_loop{source.suffix}"
        out.write_bytes(b"placeholder")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter, overwrite=False)
        fr = result.results[0]
        assert fr.skipped is True
        assert fr.success is False

    def test_skipped_file_result_error_message(self, tmp_path: Path) -> None:
        """A skipped FileResult error string must say 'skipped: output exists'."""
        batch_dir = make_batch_dir(tmp_path, n=1)
        source = next(batch_dir.glob("*.wav"))
        out = batch_dir / f"{source.stem}_loop{source.suffix}"
        out.write_bytes(b"placeholder")

        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(batch_dir, reporter=reporter, overwrite=False)
        assert result.results[0].error == "skipped: output exists"

    def test_batch_summary_shows_skip_count_in_cli(self, tmp_path: Path) -> None:
        """CLI summary must mention 'Skipped' when files were skipped (AC-3)."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        # First pass writes outputs.
        runner.invoke(main, [str(batch_dir), "--batch"])
        # Second pass without --overwrite should skip both.
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        assert result.exit_code == 0
        assert "Skipped" in result.output

    def test_batch_cli_overwrite_flag_reprocesses_all(self, tmp_path: Path) -> None:
        """--overwrite with --batch must reprocess all files, exit 0 (AC-4)."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        # Write outputs first.
        runner.invoke(main, [str(batch_dir), "--batch"])
        # Reprocess with --overwrite.
        result = runner.invoke(main, [str(batch_dir), "--batch", "--overwrite"])
        assert result.exit_code == 0
        assert "Skipped" not in result.output

    def test_batch_skipped_not_counted_as_failure(self, tmp_path: Path) -> None:
        """Skipped files must not inflate the failed count or trigger exit 3."""
        batch_dir = make_batch_dir(tmp_path, n=2)
        runner = CliRunner()
        # Write outputs.
        runner.invoke(main, [str(batch_dir), "--batch"])
        # Re-run without overwrite; skipped should NOT cause exit 3.
        result = runner.invoke(main, [str(batch_dir), "--batch"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# STORY-020: FLAC and OGG read support in batch mode
# ---------------------------------------------------------------------------


def make_loopable_flac(path: Path, duration: float = 5.0, sr: int = 22050) -> Path:
    """Write a mono sine-wave FLAC file to *path*."""
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(str(path), mono, sr, format="FLAC", subtype="PCM_16")
    return path


def make_loopable_ogg(path: Path, duration: float = 5.0, sr: int = 22050) -> Path:
    """Write a mono sine-wave OGG Vorbis file to *path*.

    Skips when OGG Vorbis is not available.
    """
    if "VORBIS" not in sf.available_subtypes("OGG"):
        pytest.skip("OGG Vorbis not available on this system")
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    sf.write(str(path), mono, sr, format="OGG", subtype="VORBIS")
    return path


class TestSupportedExtensionsConstant:
    """SUPPORTED_EXTENSIONS contains all required formats."""

    def test_wav_included(self) -> None:
        """'.wav' must be in SUPPORTED_EXTENSIONS."""
        assert ".wav" in SUPPORTED_EXTENSIONS

    def test_flac_included(self) -> None:
        """'.flac' must be in SUPPORTED_EXTENSIONS."""
        assert ".flac" in SUPPORTED_EXTENSIONS

    def test_ogg_included(self) -> None:
        """'.ogg' must be in SUPPORTED_EXTENSIONS."""
        assert ".ogg" in SUPPORTED_EXTENSIONS


class TestDiscoverAudioFiles:
    """AC-3: discover_audio_files finds WAV, FLAC, and OGG files."""

    def test_discovers_wav_files(self, tmp_path: Path) -> None:
        """discover_audio_files returns .wav files."""
        d = tmp_path / "audio"
        d.mkdir()
        make_loopable_wav(d / "track.wav")
        result = discover_audio_files(d)
        assert any(f.suffix == ".wav" for f in result)

    def test_discovers_flac_files(self, tmp_path: Path) -> None:
        """discover_audio_files returns .flac files."""
        d = tmp_path / "audio"
        d.mkdir()
        make_loopable_flac(d / "track.flac")
        result = discover_audio_files(d)
        assert any(f.suffix == ".flac" for f in result)

    def test_discovers_ogg_files(self, tmp_path: Path) -> None:
        """discover_audio_files returns .ogg files."""
        d = tmp_path / "audio"
        d.mkdir()
        make_loopable_ogg(d / "track.ogg")
        result = discover_audio_files(d)
        assert any(f.suffix == ".ogg" for f in result)

    def test_discovers_mixed_formats(self, tmp_path: Path) -> None:
        """AC-3: a directory with .wav, .flac, and .ogg files returns all three."""
        d = tmp_path / "mixed"
        d.mkdir()
        make_loopable_wav(d / "a.wav")
        make_loopable_flac(d / "b.flac")
        make_loopable_ogg(d / "c.ogg")
        result = discover_audio_files(d)
        suffixes = {f.suffix for f in result}
        assert ".wav" in suffixes
        assert ".flac" in suffixes
        assert ".ogg" in suffixes
        assert len(result) == 3

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        """discover_audio_files returns files in sorted order."""
        d = tmp_path / "sorted"
        d.mkdir()
        make_loopable_flac(d / "z.flac")
        make_loopable_wav(d / "a.wav")
        make_loopable_flac(d / "m.flac")
        result = discover_audio_files(d)
        assert [f.name for f in result] == sorted(f.name for f in result)

    def test_skips_non_audio_files(self, tmp_path: Path) -> None:
        """discover_audio_files must not include .txt or .png files."""
        d = tmp_path / "skip"
        d.mkdir()
        make_loopable_wav(d / "audio.wav")
        (d / "readme.txt").write_text("ignore")
        (d / "cover.png").write_bytes(b"\x89PNG")
        result = discover_audio_files(d)
        assert len(result) == 1
        assert result[0].name == "audio.wav"

    def test_empty_dir_raises_batch_error(self, tmp_path: Path) -> None:
        """BatchError is raised when the directory contains no audio files."""
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(BatchError, match="No WAV files found"):
            discover_audio_files(d)

    def test_nonexistent_dir_raises_batch_error(self, tmp_path: Path) -> None:
        """BatchError is raised when the directory does not exist."""
        missing = tmp_path / "nonexistent"
        with pytest.raises(BatchError, match="Directory not found"):
            discover_audio_files(missing)

    def test_discover_wav_files_alias_is_discover_audio_files(self) -> None:
        """discover_wav_files must be the same callable as discover_audio_files."""
        assert discover_wav_files is discover_audio_files


class TestBatchProcessFlacFiles:
    """AC-1 via batch: FLAC files are processed successfully in run_batch."""

    def test_flac_files_processed_successfully(self, tmp_path: Path) -> None:
        """run_batch must process FLAC files and report success."""
        d = tmp_path / "flac_batch"
        d.mkdir()
        make_loopable_flac(d / "track_00.flac")
        make_loopable_flac(d / "track_01.flac")
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(d, reporter=reporter)
        assert result.successful == 2
        assert result.failed == 0

    def test_flac_output_files_exist(self, tmp_path: Path) -> None:
        """Output loop files are written for each FLAC input."""
        d = tmp_path / "flac_out"
        d.mkdir()
        make_loopable_flac(d / "track.flac")
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(d, reporter=reporter)
        assert result.results[0].output_path is not None
        assert result.results[0].output_path.exists()


class TestBatchProcessOggFiles:
    """AC-2 via batch: OGG files are processed successfully in run_batch."""

    def test_ogg_files_processed_successfully(self, tmp_path: Path) -> None:
        """run_batch must process OGG files and report success."""
        d = tmp_path / "ogg_batch"
        d.mkdir()
        make_loopable_ogg(d / "track.ogg")
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(d, reporter=reporter)
        assert result.successful == 1
        assert result.failed == 0

    def test_ogg_output_file_exists(self, tmp_path: Path) -> None:
        """Output loop file is written for an OGG input."""
        d = tmp_path / "ogg_out"
        d.mkdir()
        make_loopable_ogg(d / "track.ogg")
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(d, reporter=reporter)
        assert result.results[0].output_path is not None
        assert result.results[0].output_path.exists()


class TestBatchMixedFormats:
    """AC-3: batch mode with mixed .wav/.flac/.ogg discovers and processes all."""

    def test_mixed_format_batch_all_succeed(self, tmp_path: Path) -> None:
        """All three format types are discovered and processed in a single batch run."""
        d = tmp_path / "mixed"
        d.mkdir()
        make_loopable_wav(d / "a.wav")
        make_loopable_flac(d / "b.flac")
        make_loopable_ogg(d / "c.ogg")
        reporter = Reporter(VerbosityLevel.QUIET)
        result = run_batch(d, reporter=reporter)
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.results) == 3
