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

"""Batch processing of multiple audio files (WAV, FLAC, OGG)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from audioloop.analysis import classify_content, detect_loop_points
from audioloop.crossfade import align_loop_to_zero_crossings
from audioloop.exceptions import AudioLoopError, BatchError
from audioloop.io import read_audio, resolve_output_path, write_wav
from audioloop.looper import create_loop
from audioloop.reporting import Reporter, VerbosityLevel

logger = logging.getLogger(__name__)

# File extensions discovered by discover_audio_files (case-insensitive).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg"})


@dataclass
class FileResult:
    """Result of processing a single file in a batch run.

    Attributes:
        input_path: Path to the source WAV file.
        output_path: Path where the output was written, or ``None`` on failure
            or skip.
        success: ``True`` when the file was processed without error.
        error: Human-readable error message when ``success`` is ``False``.
        skipped: ``True`` when the file was intentionally skipped (e.g. output
            already exists and ``overwrite=False``).  A skipped file is counted
            as neither successful nor failed.
    """

    input_path: Path
    output_path: Path | None
    success: bool
    error: str | None = None
    skipped: bool = False


@dataclass
class BatchResult:
    """Aggregate result for a batch processing run.

    Attributes:
        results: Ordered list of per-file results.
    """

    results: list[FileResult] = field(default_factory=list)

    @property
    def successful(self) -> int:
        """Number of files processed without error (excludes skipped files)."""
        return sum(1 for r in self.results if r.success and not r.skipped)

    @property
    def failed(self) -> int:
        """Number of files that encountered an error (excludes skipped files)."""
        return sum(1 for r in self.results if not r.success and not r.skipped)

    @property
    def skipped(self) -> int:
        """Number of files skipped because the output already existed."""
        return sum(1 for r in self.results if r.skipped)


def discover_audio_files(directory: Path) -> list[Path]:
    """Find all supported audio files in *directory* (case-insensitive, non-recursive).

    Supported extensions are ``.wav``, ``.flac``, and ``.ogg``.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of :class:`pathlib.Path` objects for each supported audio
        file found.

    Raises:
        BatchError: If *directory* does not exist or contains no supported
            audio files.
    """
    if not directory.is_dir():
        raise BatchError(f"Directory not found: {directory}")
    files = sorted(
        [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()]
    )
    if not files:
        raise BatchError(f"No WAV files found in: {directory}")
    return files


# Backward-compatibility alias — existing callers using discover_wav_files continue to work.
discover_wav_files = discover_audio_files


def process_single_file(
    input_path: Path,
    output_dir: Path | None,
    crossfade_ms: int | None,
    count: int,
    reporter: Reporter,
    max_file_size_mb: int = 2048,
    overwrite: bool = True,
) -> FileResult:
    """Process a single audio file in batch mode.

    Loads the file, classifies content, detects loop points, aligns to zero
    crossings, applies crossfade, and writes the output.  Any exception is
    caught and returned as a failed :class:`FileResult` so batch processing
    continues with the remaining files.

    Args:
        input_path: Path to the source audio file.
        output_dir: Directory to write output to, or ``None`` to write
            alongside the input file with a ``_loop`` suffix.
        crossfade_ms: Crossfade duration in milliseconds, or ``None`` for auto.
        count: Number of loop repetitions to render.
        reporter: Reporter instance used for progress messages.
        max_file_size_mb: Maximum allowed file size in megabytes.  Defaults
            to 2048 MB (2 GB).
        overwrite: When ``False``, skip files whose output already exists
            instead of overwriting them.  Defaults to ``True``.

    Returns:
        :class:`FileResult` describing whether the file succeeded, failed, or
        was skipped.
    """
    # Determine output path first so we can check existence before any work.
    if output_dir is not None:
        out_path = output_dir / f"{input_path.stem}_loop{input_path.suffix}"
    else:
        out_path = resolve_output_path(input_path, None)

    if out_path.exists() and not overwrite:
        reporter.info("  skipped: output exists")
        return FileResult(
            input_path=input_path,
            output_path=out_path,
            success=False,
            error="skipped: output exists",
            skipped=True,
        )

    try:
        audio = read_audio(input_path, max_file_size_mb=max_file_size_mb)
        result = classify_content(audio)
        loop_region = detect_loop_points(audio, result.content_type)
        loop_region = align_loop_to_zero_crossings(audio, loop_region)
        loop_audio = create_loop(
            audio=audio,
            region=loop_region,
            content_type=result.content_type,
            crossfade_ms=crossfade_ms,
            count=count,
        )

        write_wav(out_path, loop_audio)
        return FileResult(input_path=input_path, output_path=out_path, success=True)
    except AudioLoopError as exc:
        return FileResult(input_path=input_path, output_path=None, success=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return FileResult(input_path=input_path, output_path=None, success=False, error=str(exc))


def run_batch(
    input_dir: Path,
    output_dir: Path | None = None,
    crossfade_ms: int | None = None,
    count: int = 4,
    reporter: Reporter | None = None,
    max_file_size_mb: int = 2048,
    overwrite: bool = True,
) -> BatchResult:
    """Process all supported audio files found in *input_dir*.

    Files are discovered via :func:`discover_audio_files` and processed one at
    a time by :func:`process_single_file`.  A failure on one file does not stop
    processing of subsequent files.  Supported formats are WAV, FLAC, and OGG.

    Args:
        input_dir: Directory containing audio files to process.
        output_dir: Directory to write outputs to.  Created if it does not
            exist.  Pass ``None`` to write output alongside each source file.
        crossfade_ms: Crossfade duration in milliseconds, or ``None`` for auto.
        count: Number of loop repetitions to render per file (default 4).
        reporter: Reporter for progress output.  A new NORMAL-verbosity
            reporter is created when ``None`` is passed.
        max_file_size_mb: Maximum allowed file size in megabytes.  Defaults
            to 2048 MB (2 GB).
        overwrite: When ``False``, files whose output already exists are
            skipped and counted in :attr:`BatchResult.skipped`.  Defaults to
            ``True``.

    Returns:
        :class:`BatchResult` with one :class:`FileResult` per discovered file.

    Raises:
        BatchError: If *input_dir* does not exist or contains no supported
            audio files.
    """
    if reporter is None:
        reporter = Reporter(VerbosityLevel.NORMAL)

    files = discover_audio_files(input_dir)

    if output_dir is not None and not output_dir.exists():
        output_dir.mkdir(parents=True)

    results: list[FileResult] = []
    for i, filepath in enumerate(files, 1):
        reporter.info(f"[{i}/{len(files)}] {filepath.name}...")
        file_result = process_single_file(
            filepath,
            output_dir,
            crossfade_ms,
            count,
            reporter,
            max_file_size_mb,
            overwrite=overwrite,
        )
        if file_result.skipped:
            pass  # progress message already printed by process_single_file
        elif file_result.success:
            reporter.info(f"  -> {file_result.output_path}")
        else:
            reporter.error(f"  FAILED: {file_result.error}")
        results.append(file_result)

    return BatchResult(results=results)
