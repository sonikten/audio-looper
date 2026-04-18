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

"""Crossfade generation and application utilities."""

from __future__ import annotations

import logging
import sys

import numpy as np

from audioloop.exceptions import CrossfadeError
from audioloop.models import AudioData, ContentType, LoopRegion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default crossfade durations by content type (milliseconds)
# ---------------------------------------------------------------------------

#: Default crossfade durations (ms) per content type.
#: Ambient and mixed use longer crossfades to spread the spectral blend over
#: time, making the transition imperceptible in atmospheric/textural content.
CROSSFADE_DEFAULTS_MS: dict[ContentType, int] = {
    ContentType.RHYTHMIC: 20,
    ContentType.AMBIENT: 500,
    ContentType.MIXED: 250,
}

#: Per-content-type minimum crossfade (ms).
CROSSFADE_MIN_BY_TYPE_MS: dict[ContentType, int] = {
    ContentType.RHYTHMIC: 5,
    ContentType.AMBIENT: 100,
    ContentType.MIXED: 50,
}

#: Per-content-type maximum crossfade (ms).
CROSSFADE_MAX_BY_TYPE_MS: dict[ContentType, int] = {
    ContentType.RHYTHMIC: 50,
    ContentType.AMBIENT: 2000,
    ContentType.MIXED: 1000,
}

#: Absolute minimum crossfade — no degenerately short crossfades.
CROSSFADE_MIN_MS: int = 2


def build_equal_power_curve(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build equal-power fade-out and fade-in curves.

    The curves satisfy the equal-power constraint:
    ``fade_out[i]**2 + fade_in[i]**2 == 1`` at every point, which preserves
    perceived loudness across the crossfade transition.

    Args:
        length: Number of samples in the fade.  Must be >= 1.

    Returns:
        A ``(fade_out, fade_in)`` tuple of float64 arrays, each of shape
        ``(length,)``.  ``fade_out = cos(t * pi/2)``,
        ``fade_in = sin(t * pi/2)`` where ``t`` runs from 0 to 1 (inclusive).

    Raises:
        CrossfadeError: If *length* is less than 1.
    """
    if length < 1:
        raise CrossfadeError(f"Crossfade length must be >= 1, got {length}")
    t = np.linspace(0.0, 1.0, length, endpoint=True)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)
    return fade_out, fade_in


def get_crossfade_samples(
    content_type: ContentType,
    sample_rate: int,
    loop_length_samples: int,
    override_ms: int | None = None,
) -> tuple[int, bool]:
    """Determine crossfade duration in samples.

    If *override_ms* is provided, it is used directly (subject to clamping).
    Otherwise the content-type default is used and clamped to the per-type
    [min, max] range.  In both cases the result is also clamped to
    ``[CROSSFADE_MIN_MS, loop_length_samples / 2]``.

    Args:
        content_type: Classified content type driving default selection.
        sample_rate: Samples per second (e.g. 44100 or 48000).
        loop_length_samples: Total number of samples in the loop region.
        override_ms: If not ``None``, force this crossfade duration (ms).

    Returns:
        A ``(crossfade_samples, clamped)`` tuple.  *clamped* is ``True``
        when the requested duration was clipped to the half-loop limit.

    Raises:
        CrossfadeError: If *loop_length_samples* is less than 1.
    """
    if loop_length_samples < 1:
        raise CrossfadeError(f"loop_length_samples must be >= 1, got {loop_length_samples}")

    clamped = False

    if override_ms is not None:
        requested_ms = override_ms
    else:
        # Content-type default, clamped to per-type [min, max].
        default_ms = CROSSFADE_DEFAULTS_MS[content_type]
        type_min_ms = CROSSFADE_MIN_BY_TYPE_MS[content_type]
        type_max_ms = CROSSFADE_MAX_BY_TYPE_MS[content_type]
        requested_ms = max(type_min_ms, min(default_ms, type_max_ms))

    # Absolute minimum.
    requested_ms = max(requested_ms, CROSSFADE_MIN_MS)

    # Convert to samples.
    requested_samples = int(round(requested_ms * sample_rate / 1000.0))
    requested_samples = max(requested_samples, 1)

    # Clamp to half the loop length.
    half_loop = loop_length_samples // 2
    if requested_samples > half_loop:
        clamped = True
        requested_samples = max(half_loop, 1)

    return requested_samples, clamped


def find_zero_crossing(
    samples: np.ndarray,
    target_sample: int,
    sample_rate: int,
    max_shift_ms: float = 5.0,
) -> int:
    """Find the nearest zero-crossing to *target_sample*.

    For multi-channel audio the sum of all channels (mono sum) is used to
    determine crossing positions.  The same position is applied to every
    channel.

    The search window is ``[target_sample - max_shift, target_sample + max_shift]``
    (clamped to array bounds).  A zero-crossing is defined as any adjacent pair
    of samples whose sign differs (i.e. ``sign(s[i]) != sign(s[i+1])``).  The
    returned index is the *left* sample of the crossing pair — i.e. the last
    sample before the signal crosses zero.

    If no zero-crossing exists within the window, *target_sample* is returned
    unchanged.

    Args:
        samples: Audio array of shape ``(N,)`` for mono or ``(N, channels)``
            for multi-channel audio.
        target_sample: The nominal splice point to search around.
        sample_rate: Audio sample rate in Hz (e.g. 44100 or 48000).
        max_shift_ms: Maximum search radius in milliseconds.  Defaults to 5.0.

    Returns:
        Sample index of the nearest zero-crossing within the search window, or
        *target_sample* if none is found.
    """
    max_shift = int(sample_rate * max_shift_ms / 1000.0)

    # Build mono sum for crossing detection.
    mono = np.sum(samples, axis=1) if samples.ndim == 2 else samples

    n = len(mono)

    # Clamp search window to valid indices.
    start = max(0, target_sample - max_shift)
    end = min(n - 1, target_sample + max_shift)

    if start >= end:
        return target_sample

    window = mono[start : end + 1]
    signs = np.sign(window)
    # np.diff gives sign[i+1] - sign[i]; non-zero means a sign change.
    crossings = np.where(np.diff(signs) != 0)[0] + start

    if len(crossings) == 0:
        return target_sample

    distances = np.abs(crossings - target_sample)
    nearest_idx = int(np.argmin(distances))
    return int(crossings[nearest_idx])


def align_loop_to_zero_crossings(
    audio: AudioData,
    region: LoopRegion,
    max_shift_ms: float = 5.0,
) -> LoopRegion:
    """Adjust loop region start and end to the nearest zero-crossings.

    Both the start and end splice points are independently shifted by at most
    *max_shift_ms* milliseconds to land on a zero-crossing, reducing audible
    discontinuity when the loop is played seamlessly.  Zero-crossing detection
    uses the mono sum of all channels so that both channels are aligned to the
    same position.

    Args:
        audio: Source audio containing the loop region.
        region: Loop region whose boundaries will be adjusted.
        max_shift_ms: Maximum allowed shift (ms) for each splice point.
            Defaults to 5.0.

    Returns:
        A new :class:`~audioloop.models.LoopRegion` with adjusted
        ``start_sample`` and ``end_sample`` values.  All other fields are
        copied from *region*.
    """
    new_start = find_zero_crossing(
        audio.samples,
        region.start_sample,
        audio.sample_rate,
        max_shift_ms=max_shift_ms,
    )
    new_end = find_zero_crossing(
        audio.samples,
        region.end_sample,
        audio.sample_rate,
        max_shift_ms=max_shift_ms,
    )
    return LoopRegion(
        start_sample=new_start,
        end_sample=new_end,
        confidence=region.confidence,
        content_type=region.content_type,
        crossfade_samples=region.crossfade_samples,
    )


def apply_crossfade(
    audio: AudioData,
    region: LoopRegion,
    content_type: ContentType,
    crossfade_override_ms: int | None = None,
) -> AudioData:
    """Apply equal-power crossfade and create the looped segment.

    Creates a single seamless loop iteration using a circular crossfade:

    1. Extract the loop segment (``region.start_sample`` to
       ``region.end_sample``), length *L*.
    2. Compute crossfade length *N* via :func:`get_crossfade_samples`.
    3. Build equal-power fade curves (cos/sin, endpoint-inclusive).
    4. Build a crossfaded head: the first *N* samples of the loop (fading in)
       are summed with the last *N* samples of the loop (fading out).
    5. Append the unmodified body (samples *N* to *L - N*).
    6. Output length is *L - N* samples.

    When this output is tiled end-to-end, the last body sample
    (``loop_seg[L-N-1]``) is followed by the crossfade head's first sample
    (``loop_seg[L-N]``), which are adjacent in the original audio — seamless.

    Args:
        audio: Source audio containing the loop region.
        region: Loop region specifying start/end sample boundaries.
        content_type: Classified content type for default crossfade selection.
        crossfade_override_ms: If not ``None``, force this crossfade (ms).

    Returns:
        New :class:`~audioloop.models.AudioData` containing one iteration
        of the crossfaded loop.

    Raises:
        CrossfadeError: If the loop region is too short for a valid crossfade.
    """
    start_s = region.start_sample
    end_s = region.end_sample
    loop_length = end_s - start_s

    if loop_length < 2:
        raise CrossfadeError(
            f"Loop region is too short for crossfade: {loop_length} samples (need at least 2)"
        )

    crossfade_n, clamped = get_crossfade_samples(
        content_type=content_type,
        sample_rate=audio.sample_rate,
        loop_length_samples=loop_length,
        override_ms=crossfade_override_ms,
    )

    if clamped:
        requested_ms = crossfade_override_ms if crossfade_override_ms is not None else None
        half_ms = (loop_length // 2) * 1000.0 / audio.sample_rate
        msg = (
            f"Crossfade duration{f' ({requested_ms}ms)' if requested_ms is not None else ''} "
            f"exceeds half the loop length; clamped to {half_ms:.1f}ms"
        )
        print(f"WARNING: {msg}", file=sys.stderr)
        logger.warning(msg)

    # Extract loop segment — preserve channel layout.
    is_multichannel = audio.samples.ndim > 1
    if is_multichannel:
        loop_seg = audio.samples[start_s:end_s, :].astype(np.float64)
    else:
        loop_seg = audio.samples[start_s:end_s].astype(np.float64)

    fade_out, fade_in = build_equal_power_curve(crossfade_n)

    # Build crossfaded HEAD: the first N samples of the loop fade in while
    # the last N samples of the loop fade out.  They are summed to create a
    # smooth circular splice.  The output is then [crossfaded_head | body],
    # with total length = loop_length - crossfade_n.
    #
    # When this segment is tiled (concatenated with copies of itself), the
    # boundary between tiles is:
    #   tile[-1] = loop_seg[L-N-1]  (last body sample)
    #   next_tile[0] = loop_seg[L-N]*fade_out[0] + loop_seg[0]*fade_in[0]
    #                = loop_seg[L-N]*1.0 + loop_seg[0]*0.0 = loop_seg[L-N]
    # which is the adjacent sample in the original audio — seamless.
    if is_multichannel:
        fade_out_2d = fade_out[:, np.newaxis]
        fade_in_2d = fade_in[:, np.newaxis]
        head_end = loop_seg[-crossfade_n:] * fade_out_2d
        head_start = loop_seg[:crossfade_n] * fade_in_2d
    else:
        head_end = loop_seg[-crossfade_n:] * fade_out
        head_start = loop_seg[:crossfade_n] * fade_in

    crossfaded_head = head_end + head_start

    # --- Tapered RMS normalisation of crossfade zone ---
    # When two correlated signals are summed with equal-power curves, the
    # resulting energy can exceed both inputs (constructive interference).
    # Apply a tapered correction: full correction in the middle of the
    # crossfade, tapering to 1.0 (no correction) at the edges.  This
    # preserves sample adjacency at both boundaries while reducing the
    # energy bump in the centre.
    body_start = crossfade_n
    body_end = loop_length - crossfade_n
    body = loop_seg[body_start:body_end, :] if is_multichannel else loop_seg[body_start:body_end]

    if len(body) > 0 and crossfade_n > 2:
        # Measure RMS in small blocks across the crossfade zone and the
        # adjacent body reference, then compute a smooth correction envelope.
        ref_len = min(crossfade_n, len(body))
        ref_region = body[:ref_len]
        ref_rms = float(np.sqrt(np.mean(ref_region**2)))
        head_rms = float(np.sqrt(np.mean(crossfaded_head**2)))

        if head_rms > 0 and ref_rms > 0:
            raw_scale = ref_rms / head_rms
            # Taper: Hann-shaped window that is 1.0 at the middle and 0.0
            # at both edges.  The correction factor at each sample is
            # 1.0 + (raw_scale - 1.0) * taper[i], so edges stay at 1.0
            # (no change) and the centre gets the full correction.
            taper = np.sin(np.linspace(0, np.pi, crossfade_n)) ** 2
            correction = 1.0 + (raw_scale - 1.0) * taper
            if is_multichannel:
                crossfaded_head = crossfaded_head * correction[:, np.newaxis]
            else:
                crossfaded_head = crossfaded_head * correction

    out_samples = np.concatenate([crossfaded_head, body], axis=0)

    return AudioData(
        samples=out_samples,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        bit_depth=audio.bit_depth,
    )
