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

"""Command-line interface for audioloop."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import soundfile as sf

from audioloop import __version__
from audioloop.analysis import (
    LOOP_LENGTH_TOLERANCE,
    classify_content,
    detect_beats,
    detect_loop_points,
    find_loop_candidates,
)
from audioloop.batch import run_batch
from audioloop.crossfade import align_loop_to_zero_crossings
from audioloop.duration import parse_duration, parse_loop_length
from audioloop.exceptions import AudioLoopError, BatchError, DurationParseError, LoopDetectionError
from audioloop.io import read_audio, resolve_output_path, write_audio
from audioloop.looper import MAX_LOOP_COUNT, STREAMING_THRESHOLD, create_loop, create_loop_streaming
from audioloop.models import ContentType, LoopRegion
from audioloop.reporting import Reporter, VerbosityLevel

logger = logging.getLogger(__name__)


class _MutuallyExclusiveOption(click.Option):
    """Click Option subclass that enforces mutual exclusivity with a named peer."""

    def __init__(self, *args: object, mutually_exclusive: list[str], **kwargs: object) -> None:
        self._mutually_exclusive: list[str] = mutually_exclusive
        help_text = kwargs.pop("help", "")  # type: ignore[assignment]
        exclusion_note = f"[mutually exclusive with: {', '.join(mutually_exclusive)}]"
        kwargs["help"] = f"{help_text}  {exclusion_note}".strip()
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def handle_parse_result(
        self,
        ctx: click.Context,
        opts: dict[str, object],
        args: list[str],
    ) -> tuple[object, list[str]]:
        """Raise UsageError if both this option and its exclusive peer are present."""
        current_present = self.name in opts
        for exclusive_with in self._mutually_exclusive:
            if exclusive_with in opts and current_present:
                raise click.UsageError(
                    f"Options '{self.name}' and '{exclusive_with}' are mutually exclusive."
                )
        return super().handle_parse_result(ctx, opts, args)


def _parse_time_signature(value: str | None) -> tuple[int, int]:
    """Parse 'N/D' time signature. Returns (numerator, denominator).

    Args:
        value: A time signature string such as ``'3/4'`` or ``'6/8'``, or
            ``None`` to use the default 4/4.

    Returns:
        A ``(numerator, denominator)`` tuple.

    Raises:
        click.UsageError: If the string is not in the expected ``N/D`` format,
            the numerator is less than 1, or the denominator is not one of
            2, 4, 8, or 16.
    """
    if value is None:
        return (4, 4)
    import re

    m = re.match(r"^(\d+)/(\d+)$", value.strip())
    if not m:
        raise click.UsageError(
            f"Invalid time signature '{value}'. Expected format: N/D (e.g., '3/4', '6/8')"
        )
    num, den = int(m.group(1)), int(m.group(2))
    if num < 1:
        raise click.UsageError(f"Time signature numerator must be >= 1, got {num}")
    if den not in (2, 4, 8, 16):
        raise click.UsageError(f"Time signature denominator must be 2, 4, 8, or 16, got {den}")
    return (num, den)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="audioloop")
@click.argument("input", type=click.Path(), required=True)
@click.option("-o", "--output", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["wav", "flac", "ogg"], case_sensitive=False),
    default=None,
    help="Output format (default: wav, or inferred from -o extension). --format takes precedence.",
)
@click.option(
    "--loop-length",
    default=None,
    help=(
        "Target loop length hint (e.g. '8s', '2m', '4bars'). "
        "Bar values require a detectable tempo; use --time-signature for non-4/4 time."
    ),
)
@click.option(
    "--time-signature",
    default=None,
    help="Time signature for bar-based calculations (e.g., '3/4', '6/8'). Default: 4/4.",
)
@click.option("--start", default=None, help="Loop start offset (e.g. '1.5s', '0:01').")
@click.option("--end", default=None, help="Loop end offset (e.g. '10s', '0:10').")
@click.option(
    "--count",
    "-n",
    default=4,
    show_default=True,
    cls=_MutuallyExclusiveOption,
    mutually_exclusive=["duration"],
    help="Number of loop repetitions to render.",
)
@click.option(
    "--duration",
    "-d",
    default=None,
    cls=_MutuallyExclusiveOption,
    mutually_exclusive=["count"],
    help="Target total output duration (e.g. '30s', '2m').",
)
@click.option("--crossfade", "-x", default=None, type=int, help="Crossfade duration in ms.")
@click.option(
    "--max-file-size",
    default=2048,
    type=int,
    show_default=True,
    help="Maximum input file size in MB.",
)
@click.option("--batch", "-b", is_flag=True, default=False, help="Process a directory of files.")
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output files.",
)
@click.option(
    "--analyze-only", is_flag=True, default=False, help="Analyse and report without writing output."
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    cls=_MutuallyExclusiveOption,
    mutually_exclusive=["quiet"],
    help="Increase verbosity (repeat for more detail).",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    cls=_MutuallyExclusiveOption,
    mutually_exclusive=["verbose"],
    help="Suppress all non-error output.",
)
def main(
    input: str,
    output: str | None,
    output_format: str | None,
    loop_length: str | None,
    time_signature: str | None,
    start: str | None,
    end: str | None,
    count: int,
    duration: str | None,
    crossfade: int | None,
    max_file_size: int,
    batch: bool,
    overwrite: bool,
    analyze_only: bool,
    verbose: int,
    quiet: bool,
) -> None:
    """Auto-detect and create seamless audio loops from audio files.

    INPUT is the path to an audio file (WAV, FLAC, or OGG) or a directory
    when --batch is used.
    """
    _configure_logging(verbose, quiet)

    verbosity = _determine_verbosity(verbose, quiet)
    reporter = Reporter(verbosity)

    # ------------------------------------------------------------------
    # Validate --count bounds (STORY-017)
    # ------------------------------------------------------------------
    if count is not None and count < 1:
        raise click.UsageError(f"--count must be at least 1, got {count}")
    if count is not None and count > MAX_LOOP_COUNT:
        raise click.UsageError(f"--count {count} exceeds maximum of {MAX_LOOP_COUNT}")

    # ------------------------------------------------------------------
    # Parse --time-signature flag (STORY-022)
    # ------------------------------------------------------------------
    ts_num, ts_den = _parse_time_signature(time_signature)

    # ------------------------------------------------------------------
    # Batch mode: process an entire directory of WAV files (STORY-013)
    # ------------------------------------------------------------------
    if batch:
        input_dir = Path(input)
        output_dir = Path(output) if output else None
        try:
            batch_result = run_batch(
                input_dir,
                output_dir,
                crossfade,
                count,
                reporter,
                max_file_size,
                overwrite=overwrite,
            )
        except BatchError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        total = len(batch_result.results)
        reporter.info(f"\nBatch complete: {batch_result.successful}/{total} files processed")
        if batch_result.skipped > 0:
            reporter.info(f"  Skipped: {batch_result.skipped} (output exists)")
        if batch_result.failed > 0:
            reporter.info(f"  Failed: {batch_result.failed}")
            for r in batch_result.results:
                if not r.success and not r.skipped:
                    reporter.info(f"    - {r.input_path.name}: {r.error}")
            sys.exit(3)
        sys.exit(0)

    input_path = Path(input)

    # ------------------------------------------------------------------
    # Overwrite check — bail out early before doing any analysis work
    # (STORY-015)
    # ------------------------------------------------------------------
    if not analyze_only:
        _output_path_check = resolve_output_path(input_path, output, output_format)
        if _output_path_check.exists() and not overwrite:
            reporter.error(f"output file already exists: {_output_path_check}")
            sys.exit(1)

    try:
        audio = read_audio(input_path, max_file_size_mb=max_file_size)
    except AudioLoopError as exc:
        reporter.error(str(exc))
        sys.exit(1)

    result = classify_content(audio)
    reporter.info(f"Content type: {result.content_type.value}")
    reporter.verbose(f"  onset strength variance:  {result.onset_variance:.6f}")
    reporter.verbose(f"  percussive energy ratio:  {result.percussive_ratio:.6f}")
    reporter.verbose(f"  mean spectral flatness:   {result.spectral_flatness:.6f}")
    reporter.verbose(
        f"  thresholds: onset_high={result.thresholds['onset_variance_high']}"
        f" onset_low={result.thresholds['onset_variance_low']}"
        f" perc_high={result.thresholds['percussive_ratio_high']}"
        f" perc_low={result.thresholds['percussive_ratio_low']}"
    )

    # ------------------------------------------------------------------
    # Beat and tempo detection (STORY-005)
    # ------------------------------------------------------------------
    beat_info = None
    if result.content_type in (ContentType.RHYTHMIC, ContentType.MIXED):
        beat_info = detect_beats(audio)
        if beat_info is not None:
            reporter.info(
                f"Tempo: {beat_info.tempo:.1f} BPM ({len(beat_info.beat_positions)} beats)"
            )
            reporter.verbose(f"  beat confidence: {beat_info.confidence:.3f}")
        else:
            reporter.info("No tempo detected")
    else:
        reporter.info("No tempo detected")

    # ------------------------------------------------------------------
    # Parse --loop-length hint (STORY-011, STORY-022)
    # ------------------------------------------------------------------
    if time_signature is not None:
        reporter.verbose(f"  time signature: {ts_num}/{ts_den}")

    target_length_seconds: float | None = None
    if loop_length is not None:
        detected_bpm = beat_info.tempo if beat_info is not None else None
        try:
            target_length_seconds = parse_loop_length(
                loop_length, bpm=detected_bpm, beats_per_bar=ts_num
            )
        except DurationParseError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        if target_length_seconds > audio.duration:
            reporter.error(
                f"--loop-length {target_length_seconds:.3f}s exceeds file duration"
                f" ({audio.duration:.3f}s)"
            )
            sys.exit(2)

    # ------------------------------------------------------------------
    # Parse and validate manual loop point overrides (STORY-010)
    # ------------------------------------------------------------------
    start_seconds: float | None = None
    end_seconds: float | None = None

    if not analyze_only:
        if start is not None:
            try:
                start_seconds = parse_duration(start)
            except DurationParseError as exc:
                reporter.error(str(exc))
                sys.exit(2)

        if end is not None:
            try:
                end_seconds = parse_duration(end)
            except DurationParseError as exc:
                reporter.error(str(exc))
                sys.exit(2)

        # Validate positions are within the file.
        if start_seconds is not None and start_seconds >= audio.duration:
            reporter.error(
                f"--start position {start_seconds}s is beyond end of file ({audio.duration:.3f}s)"
            )
            sys.exit(2)

        if end_seconds is not None and end_seconds > audio.duration:
            reporter.error(
                f"--end position {end_seconds}s is beyond end of file ({audio.duration:.3f}s)"
            )
            sys.exit(2)

        # Validate ordering when both are given.
        if start_seconds is not None and end_seconds is not None and start_seconds >= end_seconds:
            reporter.error("start must be before end")
            sys.exit(2)

    # Build loop region from manual override when both endpoints are given.
    if not analyze_only and start_seconds is not None and end_seconds is not None:
        start_sample = int(start_seconds * audio.sample_rate)
        end_sample = int(end_seconds * audio.sample_rate)
        loop_region: LoopRegion = LoopRegion(
            start_sample=start_sample,
            end_sample=end_sample,
            confidence=1.0,
            content_type=result.content_type,
            crossfade_samples=0,
        )
        reporter.info(f"Loop region (manual): {start_seconds:.3f}s – {end_seconds:.3f}s")
    else:
        # ------------------------------------------------------------------
        # Spectral similarity loop-point detection
        # ------------------------------------------------------------------
        length_tolerance = LOOP_LENGTH_TOLERANCE[result.content_type]
        all_candidates = find_loop_candidates(
            audio,
            target_length_seconds=target_length_seconds,
            tolerance=length_tolerance,
        )

        if analyze_only:
            # ------------------------------------------------------------------
            # Analyze-only mode: print metadata, candidates and suggested
            # commands then return without writing any output file (STORY-007).
            # ------------------------------------------------------------------
            reporter.print_metadata(input_path.name, audio)
            top3 = all_candidates[:3]
            if top3:
                reporter.info("")
                reporter.info("  Loop candidates:")
                for i, cand in enumerate(top3, start=1):
                    reporter.info(
                        f"    {i}. {cand.start_seconds:.3f}s"
                        f" \u2013 {cand.end_seconds:.3f}s"
                        f" (similarity: {cand.similarity:.2f})"
                    )
            reporter.info("")
            reporter.info("  Suggested commands:")
            quoted = f'"{input_path.name}"'
            reporter.info(f"    audioloop {quoted} --count 4")
            if top3:
                best = top3[0]
                reporter.info(
                    f"    audioloop {quoted}"
                    f" --start {best.start_seconds:.1f}"
                    f" --end {best.end_seconds:.1f}"
                )
            return

        if verbose >= 1:
            # Print top-3 candidates to stderr before committing to the best.
            top3 = all_candidates[:3]
            for i, cand in enumerate(top3, start=1):
                reporter.verbose(
                    f"  loop candidate {i}: start={cand.start_seconds:.3f}s"
                    f" end={cand.end_seconds:.3f}s"
                    f" similarity={cand.similarity:.4f}"
                )

        try:
            loop_region = detect_loop_points(audio, result.content_type)
        except LoopDetectionError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        reporter.info(
            f"Loop region: {loop_region.start_sample / audio.sample_rate:.3f}s"
            f" – {loop_region.end_sample / audio.sample_rate:.3f}s"
            f" (similarity={loop_region.confidence:.4f})"
        )

    reporter.info(f"Processing: {input_path.name}...")

    output_path = resolve_output_path(input_path, output, output_format)

    # ------------------------------------------------------------------
    # Parse --duration when provided (STORY-012)
    # ------------------------------------------------------------------
    target_duration_seconds: float | None = None
    if duration is not None:
        try:
            target_duration_seconds = parse_duration(duration)
        except DurationParseError as exc:
            reporter.error(str(exc))
            sys.exit(2)

    # ------------------------------------------------------------------
    # Zero-crossing alignment (STORY-009)
    # ------------------------------------------------------------------
    original_start_s = loop_region.start_sample / audio.sample_rate
    original_end_s = loop_region.end_sample / audio.sample_rate
    loop_region = align_loop_to_zero_crossings(audio, loop_region)
    adjusted_start_s = loop_region.start_sample / audio.sample_rate
    adjusted_end_s = loop_region.end_sample / audio.sample_rate
    reporter.debug(
        f"  zero-crossing alignment:"
        f" start {original_start_s:.6f}s -> {adjusted_start_s:.6f}s"
        f" end {original_end_s:.6f}s -> {adjusted_end_s:.6f}s"
    )

    # ------------------------------------------------------------------
    # Apply crossfade and tile repetitions (STORY-012, STORY-023)
    # Use the streaming path when count > STREAMING_THRESHOLD or when
    # target_duration_seconds is provided (arbitrary length output).
    # ------------------------------------------------------------------
    use_streaming = target_duration_seconds is not None or (
        count is not None and count > STREAMING_THRESHOLD
    )

    if use_streaming:
        try:
            create_loop_streaming(
                audio=audio,
                region=loop_region,
                content_type=result.content_type,
                output_path=output_path,
                crossfade_ms=crossfade,
                count=count if target_duration_seconds is None else None,
                target_duration_seconds=target_duration_seconds,
                overwrite=overwrite,
                output_format=output_format,
            )
        except AudioLoopError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        _info = sf.info(str(output_path))
        output_duration = _info.frames / _info.samplerate
        reporter.verbose(f"  output: {_info.frames} samples ({output_duration:.3f}s)")
        reporter.info(f"Output duration: {output_duration:.3f}s")
        reporter.info(str(output_path))
    else:
        try:
            loop_audio = create_loop(
                audio=audio,
                region=loop_region,
                content_type=result.content_type,
                crossfade_ms=crossfade,
                count=count,
                target_duration_seconds=None,
            )
        except AudioLoopError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        n_samples = loop_audio.samples.shape[0]
        output_duration = n_samples / loop_audio.sample_rate
        reporter.verbose(f"  output: {n_samples} samples ({output_duration:.3f}s)")

        try:
            write_audio(output_path, loop_audio, overwrite=overwrite, output_format=output_format)
        except AudioLoopError as exc:
            reporter.error(str(exc))
            sys.exit(1)

        reporter.info(f"Output duration: {output_duration:.3f}s")
        reporter.info(str(output_path))


def _determine_verbosity(verbose: int, quiet: bool) -> VerbosityLevel:
    """Map CLI flags to a VerbosityLevel value.

    Args:
        verbose: Count of -v flags (0 = normal, 1 = verbose, 2+ = debug).
        quiet: If True, suppress all non-error output.

    Returns:
        The appropriate VerbosityLevel for the given flags.
    """
    if quiet:
        return VerbosityLevel.QUIET
    if verbose >= 2:
        return VerbosityLevel.DEBUG
    if verbose == 1:
        return VerbosityLevel.VERBOSE
    return VerbosityLevel.NORMAL


def _configure_logging(verbose: int, quiet: bool) -> None:
    """Configure the root logger based on verbosity flags.

    Args:
        verbose: Count of -v flags supplied (0 = normal, 1 = verbose, 2+ = debug).
        quiet: If True, set level to WARNING to suppress informational output.
    """
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)
