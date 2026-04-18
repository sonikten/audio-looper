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

"""Core audio looping pipeline."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np

from audioloop.crossfade import apply_crossfade
from audioloop.io import write_audio_streaming
from audioloop.models import AudioData, ContentType, LoopRegion

logger = logging.getLogger(__name__)

#: Duration of the linear fade-out applied to a partial final repetition (ms).
_PARTIAL_REP_FADEOUT_MS: int = 100

#: Maximum number of loop repetitions accepted by :func:`create_loop`.
MAX_LOOP_COUNT: int = 10000

#: Loop repetition count above which :func:`create_loop_streaming` is preferred
#: over the in-memory :func:`create_loop` path.
STREAMING_THRESHOLD: int = 10


def create_loop(
    audio: AudioData,
    region: LoopRegion,
    content_type: ContentType,
    crossfade_ms: int | None = None,
    count: int | None = None,
    target_duration_seconds: float | None = None,
) -> AudioData:
    """Apply crossfade and tile the loop segment to the requested length.

    Exactly one of *count* or *target_duration_seconds* must control the
    output length.  When neither is provided, *count* defaults to 4.

    Steps:

    1. Apply equal-power crossfade to produce a single seamless loop iteration.
    2a. If *count* is used: tile that iteration *count* times.
    2b. If *target_duration_seconds* is used: tile enough full repetitions to
        fill the target, then append a partial repetition (if needed) with a
        short linear fade-out to avoid a hard cut.
    3. Return the combined ``AudioData``.

    Args:
        audio: Source audio containing the loop region.
        region: Loop region specifying start/end sample boundaries.
        content_type: Classified content type for crossfade duration selection.
        crossfade_ms: Crossfade duration in milliseconds; ``None`` for auto.
        count: Number of times to repeat the loop iteration (must be >= 1).
            Defaults to 4 when neither *count* nor *target_duration_seconds*
            is provided.
        target_duration_seconds: Target total output duration in seconds.
            A partial final repetition is appended when the target is not an
            exact multiple of the loop iteration length; that partial rep
            receives a linear fade-out over :data:`_PARTIAL_REP_FADEOUT_MS` ms.

    Returns:
        New :class:`~audioloop.models.AudioData` with the requested length.

    Raises:
        CrossfadeError: If crossfade application fails.
        ValueError: If *count* < 1, if *target_duration_seconds* <= 0, or if
            both *count* and *target_duration_seconds* are supplied.
    """
    if count is not None and target_duration_seconds is not None:
        raise ValueError("Provide count or target_duration_seconds, not both.")

    if count is not None and count < 1:
        raise ValueError(f"count must be at least 1, got {count}")

    if count is not None and count > MAX_LOOP_COUNT:
        raise ValueError(f"count exceeds maximum of {MAX_LOOP_COUNT}, got {count}")

    if target_duration_seconds is not None and target_duration_seconds <= 0:
        raise ValueError(f"target_duration_seconds must be > 0, got {target_duration_seconds}")

    # Default: repeat 4 times when no mode is specified.
    if count is None and target_duration_seconds is None:
        count = 4

    loop_iter = apply_crossfade(
        audio=audio,
        region=region,
        content_type=content_type,
        crossfade_override_ms=crossfade_ms,
    )

    # ------------------------------------------------------------------
    # Count-based tiling
    # ------------------------------------------------------------------
    if count is not None:
        if count == 1:
            return loop_iter

        tiled = np.concatenate([loop_iter.samples] * count, axis=0)
        return AudioData(
            samples=tiled,
            sample_rate=loop_iter.sample_rate,
            channels=loop_iter.channels,
            bit_depth=loop_iter.bit_depth,
        )

    # ------------------------------------------------------------------
    # Duration-based tiling
    # ------------------------------------------------------------------
    assert target_duration_seconds is not None  # mypy narrowing

    loop_samples = loop_iter.samples.shape[0]
    sr = loop_iter.sample_rate
    loop_seconds = loop_samples / sr
    target_samples = int(round(target_duration_seconds * sr))

    full_reps = int(target_duration_seconds / loop_seconds)
    remainder_samples = target_samples - full_reps * loop_samples

    if remainder_samples <= 0:
        # Exact fit — no partial repetition needed.
        if full_reps == 1:
            return loop_iter
        tiled = np.concatenate([loop_iter.samples] * full_reps, axis=0)
        return AudioData(
            samples=tiled,
            sample_rate=sr,
            channels=loop_iter.channels,
            bit_depth=loop_iter.bit_depth,
        )

    # Build partial repetition with a linear fade-out.
    partial_raw = loop_iter.samples[:remainder_samples]
    fadeout_samples = min(
        int(round(_PARTIAL_REP_FADEOUT_MS / 1000.0 * sr)),
        remainder_samples,
    )
    ramp = np.linspace(1.0, 0.0, fadeout_samples, endpoint=True)

    partial = partial_raw.copy()
    if partial.ndim == 1:
        partial[-fadeout_samples:] *= ramp
    else:
        # Shape: (num_samples, channels) — broadcast ramp over channel axis.
        partial[-fadeout_samples:] *= ramp[:, np.newaxis]

    # Concatenate full repetitions + partial.
    segments = [partial] if full_reps == 0 else [loop_iter.samples] * full_reps + [partial]

    tiled = np.concatenate(segments, axis=0)
    return AudioData(
        samples=tiled,
        sample_rate=sr,
        channels=loop_iter.channels,
        bit_depth=loop_iter.bit_depth,
    )


def create_loop_streaming(
    audio: AudioData,
    region: LoopRegion,
    content_type: ContentType,
    output_path: Path,
    crossfade_ms: int | None = None,
    count: int | None = None,
    target_duration_seconds: float | None = None,
    overwrite: bool = True,
    output_format: str | None = None,
) -> None:
    """Create a loop and write it directly to disk without holding the full output in memory.

    Exactly one of *count* or *target_duration_seconds* controls output length.
    When neither is provided the function raises :exc:`ValueError` — callers
    must be explicit about length when using the streaming path.

    The crossfade is applied once in memory (a single loop iteration is always
    small) and then the resulting block is yielded *count* times (or for enough
    repetitions to fill *target_duration_seconds*) via :func:`write_audio_streaming`.

    For duration-based output a partial final repetition is appended with the
    same linear fade-out as :func:`create_loop`.

    Args:
        audio: Source audio containing the loop region.
        region: Loop region specifying start/end sample boundaries.
        content_type: Classified content type for crossfade duration selection.
        output_path: Destination path for the written audio file.
        crossfade_ms: Crossfade duration in milliseconds; ``None`` for auto.
        count: Number of times to repeat the loop iteration (must be >= 1).
        target_duration_seconds: Target total output duration in seconds.
        overwrite: Passed through to :func:`write_audio_streaming`.
        output_format: Passed through to :func:`write_audio_streaming`.

    Raises:
        ValueError: If neither *count* nor *target_duration_seconds* is given,
            if both are given, or if *count* < 1 or *target_duration_seconds* <= 0.
        AudioFormatError: If the output file cannot be written.
        CrossfadeError: If crossfade application fails.
    """
    if count is not None and target_duration_seconds is not None:
        raise ValueError("Provide count or target_duration_seconds, not both.")
    if count is None and target_duration_seconds is None:
        raise ValueError("Provide either count or target_duration_seconds.")
    if count is not None and count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if count is not None and count > MAX_LOOP_COUNT:
        raise ValueError(f"count exceeds maximum of {MAX_LOOP_COUNT}, got {count}")
    if target_duration_seconds is not None and target_duration_seconds <= 0:
        raise ValueError(f"target_duration_seconds must be > 0, got {target_duration_seconds}")

    # Compute one crossfaded iteration — this stays in memory throughout.
    loop_iter = apply_crossfade(
        audio=audio,
        region=region,
        content_type=content_type,
        crossfade_override_ms=crossfade_ms,
    )

    def _chunk_generator() -> Generator[np.ndarray, None, None]:
        if count is not None:
            for _ in range(count):
                yield loop_iter.samples
        else:
            assert target_duration_seconds is not None  # mypy narrowing
            sr = loop_iter.sample_rate
            loop_samples = loop_iter.samples.shape[0]
            loop_seconds = loop_samples / sr
            target_samples = int(round(target_duration_seconds * sr))

            full_reps = int(target_duration_seconds / loop_seconds)
            remainder_samples = target_samples - full_reps * loop_samples

            for _ in range(full_reps):
                yield loop_iter.samples

            if remainder_samples > 0:
                partial = loop_iter.samples[:remainder_samples].copy()
                fadeout_samples = min(
                    int(round(_PARTIAL_REP_FADEOUT_MS / 1000.0 * sr)),
                    remainder_samples,
                )
                if fadeout_samples > 0:
                    ramp = np.linspace(1.0, 0.0, fadeout_samples, endpoint=True)
                    if partial.ndim == 1:
                        partial[-fadeout_samples:] *= ramp
                    else:
                        partial[-fadeout_samples:] *= ramp[:, np.newaxis]
                yield partial

    write_audio_streaming(
        path=output_path,
        sample_rate=loop_iter.sample_rate,
        channels=loop_iter.channels,
        bit_depth=loop_iter.bit_depth,
        chunks=_chunk_generator(),
        overwrite=overwrite,
        output_format=output_format,
    )


def run_pipeline(audio: AudioData, region: LoopRegion, content_type: ContentType) -> AudioData:
    """Execute the full loop-creation pipeline for a single input file.

    Steps: crossfade -> repeat -> return.

    Args:
        audio: Loaded source audio.
        region: Detected loop region.
        content_type: Classified content type.

    Returns:
        The final processed AudioData (before writing to disk).

    Raises:
        CrossfadeError: If crossfade application fails.
    """
    return create_loop(
        audio=audio,
        region=region,
        content_type=content_type,
        crossfade_ms=None,
        count=4,
    )
