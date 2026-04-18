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

"""Audio content classification and loop-point detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa  # only this module may import librosa (ADR-001)
import numpy as np

from audioloop.exceptions import LoopDetectionError
from audioloop.models import AudioData, ContentType, LoopRegion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named threshold constants — calibrated from synthetic test fixtures.
# Rhythmic fixture (click track at 120 BPM): onset_var ≈ 27, perc_ratio ≈ 1.0
# Ambient fixture (440 Hz sine):             onset_var ≈ 0.0002, perc_ratio ≈ 0.0002
# Mixed fixture (equal-RMS clicks + tone):   onset_var ≈ 16, perc_ratio ≈ 0.02
# ---------------------------------------------------------------------------

#: Onset-strength variance above which the signal is considered "highly rhythmic".
ONSET_VARIANCE_HIGH: float = 5.0

#: Onset-strength variance below which the signal is considered "ambient-like".
ONSET_VARIANCE_LOW: float = 0.01

#: Percussive-energy ratio (from HPSS) above which rhythmic character is confirmed.
PERCUSSIVE_RATIO_HIGH: float = 0.35

#: Percussive-energy ratio (from HPSS) below which the signal is considered tonal.
PERCUSSIVE_RATIO_LOW: float = 0.15


@dataclass
class ClassificationResult:
    """Detailed result of audio content classification.

    Attributes:
        content_type: The determined ContentType (RHYTHMIC, AMBIENT, or MIXED).
        onset_variance: Variance of the onset-strength envelope (primary feature).
        percussive_ratio: Ratio of percussive energy to total energy from HPSS (secondary).
        spectral_flatness: Mean spectral flatness across the signal (tertiary feature).
        thresholds: Mapping of threshold names to the values used in classification.
    """

    content_type: ContentType
    onset_variance: float
    percussive_ratio: float
    spectral_flatness: float
    thresholds: dict[str, float] = field(default_factory=dict)


def classify_content(audio: AudioData) -> ClassificationResult:
    """Classify audio content as rhythmic, ambient, or mixed.

    Uses three features in priority order:

    1. Onset-strength variance (primary) — high values indicate rhythmic content.
    2. Percussive-energy ratio from HPSS (secondary) — values above
       ``PERCUSSIVE_RATIO_HIGH`` confirm rhythmic character.
    3. Mean spectral flatness (tertiary) — reported in diagnostics.

    Classification rules
    --------------------
    * ``onset_var > ONSET_VARIANCE_HIGH`` **and** ``perc_ratio > PERCUSSIVE_RATIO_HIGH``
      → ``ContentType.RHYTHMIC``
    * ``onset_var < ONSET_VARIANCE_LOW`` **and** ``perc_ratio < PERCUSSIVE_RATIO_LOW``
      → ``ContentType.AMBIENT``
    * Otherwise → ``ContentType.MIXED``

    Args:
        audio: The audio data to analyse.

    Returns:
        A :class:`ClassificationResult` with the content type and all diagnostic values.
    """
    mono = audio.mono  # 1-D float64 array

    # ------------------------------------------------------------------
    # Feature 1: Onset-strength variance
    # ------------------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=mono, sr=audio.sample_rate)
    onset_var = float(np.var(onset_env))

    # ------------------------------------------------------------------
    # Feature 2: HPSS percussive-energy ratio
    # ------------------------------------------------------------------
    _harmonic, percussive = librosa.effects.hpss(mono)
    perc_energy = float(np.sum(percussive**2))
    total_energy = float(np.sum(mono**2))
    perc_ratio = perc_energy / total_energy if total_energy > 0.0 else 0.0

    # ------------------------------------------------------------------
    # Feature 3: Mean spectral flatness
    # ------------------------------------------------------------------
    flatness_frames = librosa.feature.spectral_flatness(y=mono)
    mean_flatness = float(np.mean(flatness_frames))

    # ------------------------------------------------------------------
    # Classification decision
    # ------------------------------------------------------------------
    thresholds: dict[str, float] = {
        "onset_variance_high": ONSET_VARIANCE_HIGH,
        "onset_variance_low": ONSET_VARIANCE_LOW,
        "percussive_ratio_high": PERCUSSIVE_RATIO_HIGH,
        "percussive_ratio_low": PERCUSSIVE_RATIO_LOW,
    }

    if onset_var > ONSET_VARIANCE_HIGH and perc_ratio > PERCUSSIVE_RATIO_HIGH:
        content_type = ContentType.RHYTHMIC
    elif onset_var < ONSET_VARIANCE_LOW and perc_ratio < PERCUSSIVE_RATIO_LOW:
        content_type = ContentType.AMBIENT
    else:
        content_type = ContentType.MIXED

    logger.debug(
        "classify_content: onset_var=%.4f perc_ratio=%.4f flatness=%.6f -> %s",
        onset_var,
        perc_ratio,
        mean_flatness,
        content_type.value,
    )

    return ClassificationResult(
        content_type=content_type,
        onset_variance=onset_var,
        percussive_ratio=perc_ratio,
        spectral_flatness=mean_flatness,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Loop length tolerance constants (STORY-011)
# ---------------------------------------------------------------------------

#: Per-content-type fractional tolerance used when filtering candidates by a
#: target loop length.  Values are ±fraction of the target length.
LOOP_LENGTH_TOLERANCE: dict[ContentType, float] = {
    ContentType.RHYTHMIC: 0.05,
    ContentType.AMBIENT: 0.10,
    ContentType.MIXED: 0.075,
}

# ---------------------------------------------------------------------------
# Spectral similarity constants (ADR-003, STORY-006)
# ---------------------------------------------------------------------------

#: Number of MFCC coefficients used for spectral comparison.
MFCC_N_COEFF: int = 13

#: Minimum cosine-similarity score for a loop region to be accepted.
SIMILARITY_THRESHOLD: float = 0.85

#: Preferred minimum loop length in seconds for long tracks (>30s).
#: For shorter tracks, the minimum is scaled down proportionally.
MIN_LOOP_SECONDS: float = 8.0

#: Absolute minimum loop length — never go below this regardless of track length.
_ABSOLUTE_MIN_LOOP_SECONDS: float = 2.0

#: Maximum loop length in seconds; longer loops are excluded from search.
MAX_LOOP_SECONDS: float = 60.0

#: Duration (seconds) of the MFCC analysis window centred on each candidate.
ANALYSIS_WINDOW_SECONDS: float = 1.0

#: Step size (seconds) between successive loop-end candidate positions.
SEARCH_STEP_SECONDS: float = 0.5

#: Number of start positions to sample across the track body.
_NUM_START_PROBES: int = 20

#: Fraction of the track to skip at the start (avoid fade-in).
_SKIP_INTRO_FRACTION: float = 0.05

#: Fraction of the track to skip at the end (avoid fade-out).
_SKIP_OUTRO_FRACTION: float = 0.05

#: Weight for RMS envelope consistency in candidate scoring [0, 1].
_RMS_CONSISTENCY_WEIGHT: float = 0.3

#: Minimum mean RMS for a candidate region — reject near-silent loops.
_MIN_REGION_RMS: float = 0.005

#: librosa MFCC hop length (samples).
_HOP_LENGTH: int = 512

#: Minimum internal score for a candidate to be collected (pre-threshold filter).
_MIN_COLLECT_SCORE: float = 0.0


@dataclass
class LoopCandidate:
    """A candidate loop region with its spectral-similarity score.

    Attributes:
        start_seconds: Time in seconds at which the loop begins.
        end_seconds: Time in seconds at which the loop ends.
        similarity: Cosine similarity of the mean MFCC vectors at start and end
            windows.  Range [0.0, 1.0].
    """

    start_seconds: float
    end_seconds: float
    similarity: float


def _compute_mfcc_features(mono: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCC features for the full audio.

    Args:
        mono: 1-D float64 array of audio samples.
        sr: Sample rate in Hz.

    Returns:
        MFCC matrix of shape ``(n_mfcc, n_frames)``.
    """
    return librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=MFCC_N_COEFF, hop_length=_HOP_LENGTH)


def _mean_mfcc_window(mfcc: np.ndarray, center_frame: int, window_frames: int) -> np.ndarray:
    """Extract the mean MFCC vector over a window centred on *center_frame*.

    The window is clamped to valid frame indices so that frames near the edges
    of the audio still produce a valid (though shorter) average.

    Args:
        mfcc: MFCC matrix of shape ``(n_mfcc, n_frames)``.
        center_frame: Frame index around which the window is centred.
        window_frames: Total number of frames in the analysis window.

    Returns:
        1-D array of shape ``(n_mfcc,)`` containing the mean MFCC vector.
    """
    half = window_frames // 2
    start = max(0, center_frame - half)
    end = min(mfcc.shape[1], center_frame + half)
    return np.mean(mfcc[:, start:end], axis=1)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]; returns 0.0 when either vector has
        zero magnitude.
    """
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def find_loop_candidates(
    audio: AudioData,
    target_length_seconds: float | None = None,
    tolerance: float = 0.10,
) -> list[LoopCandidate]:
    """Find candidate loop regions using MFCC spectral similarity.

    The search fixes a set of loop-start positions (1 s to 10 s in 1-second
    steps) and, for each start, slides the loop-end position from
    ``start + MIN_LOOP_SECONDS`` up to ``start + min(MAX_LOOP_SECONDS,
    half_file_duration)`` in steps of ``SEARCH_STEP_SECONDS``.  For each
    (start, end) pair the cosine similarity of the mean MFCC vectors in a
    1-second window around each point is computed.

    When *target_length_seconds* is provided, only candidates whose loop
    duration is within *tolerance* (fractional) of the target are returned.
    If no candidates fall within tolerance the full unfiltered list is
    returned with a warning so that the caller always gets usable results.

    Args:
        audio: The loaded audio to search for loop regions.
        target_length_seconds: Desired loop length in seconds, or ``None`` to
            skip length filtering.
        tolerance: Fractional tolerance around *target_length_seconds* (e.g.
            ``0.05`` means ±5 %).  Ignored when *target_length_seconds* is
            ``None``.

    Returns:
        List of :class:`LoopCandidate` instances sorted by ``similarity``
        descending.  May be empty for very short audio.
    """
    mono = audio.mono
    sr = audio.sample_rate
    duration = audio.duration

    mfcc = _compute_mfcc_features(mono, sr)
    frames_per_second = sr / _HOP_LENGTH  # e.g. 48000/512 ≈ 93.75

    window_frames = max(1, int(round(ANALYSIS_WINDOW_SECONDS * frames_per_second)))
    step_frames = max(1, int(round(SEARCH_STEP_SECONDS * frames_per_second)))

    # Adaptive minimum loop length: prefer MIN_LOOP_SECONDS for long tracks,
    # but scale down for short audio so that loop detection still works.
    if duration > 30.0:
        effective_min_loop = MIN_LOOP_SECONDS
    else:
        effective_min_loop = max(
            _ABSOLUTE_MIN_LOOP_SECONDS,
            min(MIN_LOOP_SECONDS, duration / 4.0),
        )
    min_loop_frames = int(round(effective_min_loop * frames_per_second))

    # Maximum loop length: 60 s or half the file, whichever is shorter.
    effective_max_loop_seconds = min(MAX_LOOP_SECONDS, duration / 2.0)
    max_loop_frames = int(round(effective_max_loop_seconds * frames_per_second))

    total_frames = mfcc.shape[1]

    # --- Compute RMS envelope for consistency scoring ---
    rms_block_frames = max(1, int(round(0.5 * frames_per_second)))  # 500ms blocks
    n_rms_blocks = total_frames // rms_block_frames
    rms_envelope = (
        np.array(
            [
                float(np.sqrt(np.mean(mono[i * int(0.5 * sr) : (i + 1) * int(0.5 * sr)] ** 2)))
                for i in range(n_rms_blocks)
            ]
        )
        if n_rms_blocks > 0
        else np.array([0.0])
    )

    # --- Determine search range: skip intro/outro fade regions ---
    skip_start_s = max(duration * _SKIP_INTRO_FRACTION, ANALYSIS_WINDOW_SECONDS)
    skip_end_s = duration - max(duration * _SKIP_OUTRO_FRACTION, ANALYSIS_WINDOW_SECONDS)
    usable_duration = skip_end_s - skip_start_s

    if usable_duration < effective_min_loop + ANALYSIS_WINDOW_SECONDS:
        # Track too short for skip — use the whole thing
        skip_start_s = ANALYSIS_WINDOW_SECONDS
        skip_end_s = duration - ANALYSIS_WINDOW_SECONDS
        usable_duration = skip_end_s - skip_start_s

    # --- Sample start positions across the usable range ---
    n_probes = min(_NUM_START_PROBES, max(1, int(usable_duration / 2.0)))
    start_times_s = np.linspace(skip_start_s, skip_end_s - effective_min_loop, n_probes)

    candidates: list[LoopCandidate] = []

    for start_s in start_times_s:
        start_frame = int(round(start_s * frames_per_second))
        if start_frame >= total_frames:
            break

        start_vec = _mean_mfcc_window(mfcc, start_frame, window_frames)

        end_frame_min = start_frame + min_loop_frames
        # Constrain end position to usable range (before outro skip region).
        usable_end_frame = int(round(skip_end_s * frames_per_second))
        end_frame_max = min(usable_end_frame, start_frame + max_loop_frames)

        if end_frame_min > end_frame_max:
            break

        for end_frame in range(end_frame_min, end_frame_max + 1, step_frames):
            end_vec = _mean_mfcc_window(mfcc, end_frame, window_frames)
            spectral_score = _cosine_similarity(start_vec, end_vec)

            if spectral_score > _MIN_COLLECT_SCORE:
                start_seconds = start_frame / frames_per_second
                end_seconds = end_frame / frames_per_second

                # --- RMS envelope consistency scoring ---
                # Penalise loops where the RMS varies a lot (pulsing when looped).
                # Also reject near-silent regions entirely.
                rms_start_block = int(start_seconds / 0.5)
                rms_end_block = int(end_seconds / 0.5)
                if rms_start_block < len(rms_envelope) and rms_end_block <= len(rms_envelope):
                    region_rms = rms_envelope[rms_start_block:rms_end_block]
                    mean_rms = float(np.mean(region_rms)) if len(region_rms) > 0 else 0.0
                    # Skip near-silent regions — they produce trivially high
                    # MFCC similarity but result in silence loops.
                    if mean_rms < _MIN_REGION_RMS:
                        continue
                    if len(region_rms) > 1 and mean_rms > 0:
                        # Coefficient of variation: std/mean. Low = consistent energy.
                        cv = float(np.std(region_rms) / mean_rms)
                        rms_consistency = max(0.0, 1.0 - cv)
                    else:
                        rms_consistency = 1.0
                else:
                    rms_consistency = 1.0

                # Combined score: weighted blend of spectral similarity and
                # RMS consistency.
                combined = (
                    1.0 - _RMS_CONSISTENCY_WEIGHT
                ) * spectral_score + _RMS_CONSISTENCY_WEIGHT * rms_consistency

                candidates.append(
                    LoopCandidate(
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        similarity=combined,
                    )
                )

    candidates.sort(key=lambda c: c.similarity, reverse=True)

    logger.debug(
        "find_loop_candidates: %d candidates found (best=%.4f)",
        len(candidates),
        candidates[0].similarity if candidates else 0.0,
    )

    # ------------------------------------------------------------------
    # Optional: filter by target loop length (STORY-011)
    # ------------------------------------------------------------------
    if target_length_seconds is not None:
        filtered = [
            c
            for c in candidates
            if abs((c.end_seconds - c.start_seconds) - target_length_seconds)
            / target_length_seconds
            <= tolerance
        ]
        if filtered:
            logger.debug(
                "find_loop_candidates: %d candidates within %.1f%% of %.3fs target",
                len(filtered),
                tolerance * 100,
                target_length_seconds,
            )
            return filtered
        logger.warning(
            "No loop candidates found within %.1f%% of %.3fs target length; "
            "returning all %d candidates",
            tolerance * 100,
            target_length_seconds,
            len(candidates),
        )

    return candidates


@dataclass
class BeatInfo:
    """Result of beat and tempo detection.

    Attributes:
        tempo: Detected tempo in beats per minute (BPM).
        beat_positions: Beat times in seconds, sorted ascending.
        confidence: Estimate of detection reliability in [0.0, 1.0].
            Computed as the ratio of detected beats to expected beats given
            the tempo and audio duration.  Clamped to 1.0.
    """

    tempo: float
    beat_positions: list[float]
    confidence: float


def detect_beats(audio: AudioData) -> BeatInfo | None:
    """Detect tempo and beat positions in audio.

    Uses :func:`librosa.beat.beat_track` to estimate a global tempo and
    locate individual beat frames.

    The function returns ``None`` when the detected tempo falls outside the
    range [40, 240] BPM or fewer than 4 beats are found, indicating that a
    reliable tempo estimate could not be made.

    Args:
        audio: The audio data to analyse.

    Returns:
        A :class:`BeatInfo` instance when a reliable tempo is detected,
        or ``None`` for ambient / non-rhythmic content.
    """
    mono = audio.mono

    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=audio.sample_rate)

    # librosa may return a 1-element array in newer versions — extract scalar.
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    # Convert beat frame indices to time in seconds.
    beat_times = librosa.frames_to_time(beat_frames, sr=audio.sample_rate)

    # Reject implausible tempos or sparse beat detection.
    if tempo < 40.0 or tempo > 240.0 or len(beat_times) < 4:
        return None

    # Confidence: ratio of detected beats to expected beats at the estimated tempo.
    expected_beats = audio.duration * tempo / 60.0
    confidence = min(1.0, len(beat_times) / expected_beats) if expected_beats > 0 else 0.0

    logger.debug(
        "detect_beats: tempo=%.2f bpm, beats=%d, confidence=%.3f",
        tempo,
        len(beat_times),
        confidence,
    )

    return BeatInfo(
        tempo=float(tempo),
        beat_positions=[float(t) for t in beat_times],
        confidence=confidence,
    )


def detect_loop_points(audio: AudioData, content_type: ContentType) -> LoopRegion:
    """Detect optimal loop start and end points for the given audio.

    Uses MFCC spectral similarity to find a region where the audio at the
    loop-end sounds spectrally similar to the audio at the loop-start.

    Args:
        audio: The audio data to analyse.
        content_type: Pre-classified content type that drives detection strategy.

    Returns:
        A :class:`~audioloop.models.LoopRegion` describing the best loop
        boundaries found.

    Raises:
        LoopDetectionError: If no candidate reaches the minimum similarity
            threshold of :data:`SIMILARITY_THRESHOLD`.  The exception message
            suggests using ``--start`` / ``--end`` to specify manual overrides.
    """
    candidates = find_loop_candidates(audio)

    best = candidates[0] if candidates else None

    if best is None or best.similarity < SIMILARITY_THRESHOLD:
        best_score = best.similarity if best else 0.0
        raise LoopDetectionError(
            f"No loop region found with similarity >= {SIMILARITY_THRESHOLD:.2f} "
            f"(best score: {best_score:.4f}). "
            "Try specifying manual boundaries with --start / --end."
        )

    start_sample = int(round(best.start_seconds * audio.sample_rate))
    end_sample = int(round(best.end_seconds * audio.sample_rate))

    logger.debug(
        "detect_loop_points: loop=%.3f–%.3fs similarity=%.4f",
        best.start_seconds,
        best.end_seconds,
        best.similarity,
    )

    return LoopRegion(
        start_sample=start_sample,
        end_sample=end_sample,
        confidence=best.similarity,
        content_type=content_type,
        crossfade_samples=0,
    )
