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

"""Unit tests for analysis.py — STORY-004, STORY-005, STORY-006, and STORY-011."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# conftest.py lives one level up; add it to the path so we can call its
# plain helper functions (not just pytest fixtures) with custom arguments.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_rhythmic_wav  # noqa: E402

from audioloop.analysis import (
    LOOP_LENGTH_TOLERANCE,
    BeatInfo,
    ClassificationResult,
    LoopCandidate,
    _cosine_similarity,
    classify_content,
    detect_beats,
    detect_loop_points,
    find_loop_candidates,
)
from audioloop.exceptions import LoopDetectionError
from audioloop.io import read_wav
from audioloop.models import AudioData, ContentType, LoopRegion

# Re-use the shared synthetic WAV generators from conftest.py.
# The pytest fixtures rhythmic_wav, ambient_wav, and mixed_wav are automatically
# available in this module through conftest.py.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> AudioData:
    """Load a WAV file into AudioData."""
    return read_wav(path)


# ---------------------------------------------------------------------------
# AC-1: Rhythmic classification
# ---------------------------------------------------------------------------


class TestClassifyRhythmic:
    """GIVEN a WAV with strong, consistent percussive elements
    WHEN classify_content is called
    THEN content_type is RHYTHMIC.
    """

    def test_classify_rhythmic(self, rhythmic_wav: Path) -> None:
        """A click-track WAV must classify as RHYTHMIC."""
        audio = _load(rhythmic_wav)
        result = classify_content(audio)
        assert result.content_type is ContentType.RHYTHMIC


# ---------------------------------------------------------------------------
# AC-2: Ambient classification
# ---------------------------------------------------------------------------


class TestClassifyAmbient:
    """GIVEN a WAV with no percussive elements and slowly evolving tonal content
    WHEN classify_content is called
    THEN content_type is AMBIENT.
    """

    def test_classify_ambient(self, ambient_wav: Path) -> None:
        """A sustained-sine-tone WAV must classify as AMBIENT."""
        audio = _load(ambient_wav)
        result = classify_content(audio)
        assert result.content_type is ContentType.AMBIENT


# ---------------------------------------------------------------------------
# AC-3: Mixed classification
# ---------------------------------------------------------------------------


class TestClassifyMixed:
    """GIVEN a WAV with both percussive and sustained tonal elements
    WHEN classify_content is called
    THEN content_type is MIXED.
    """

    def test_classify_mixed(self, mixed_wav: Path) -> None:
        """A mixed (clicks + sine) WAV must classify as MIXED."""
        audio = _load(mixed_wav)
        result = classify_content(audio)
        assert result.content_type is ContentType.MIXED


# ---------------------------------------------------------------------------
# AC-4: Diagnostic values are populated in the result
# ---------------------------------------------------------------------------


class TestClassificationDiagnostics:
    """GIVEN any classified WAV
    WHEN classify_content returns a ClassificationResult
    THEN all diagnostic numeric fields are populated with finite values.
    """

    def test_classification_result_has_onset_variance(self, rhythmic_wav: Path) -> None:
        """onset_variance must be a finite float."""
        result = classify_content(_load(rhythmic_wav))
        assert isinstance(result.onset_variance, float)
        assert np.isfinite(result.onset_variance)

    def test_classification_result_has_percussive_ratio(self, rhythmic_wav: Path) -> None:
        """percussive_ratio must be a finite float in [0, 1]."""
        result = classify_content(_load(rhythmic_wav))
        assert isinstance(result.percussive_ratio, float)
        assert 0.0 <= result.percussive_ratio <= 1.0

    def test_classification_result_has_spectral_flatness(self, ambient_wav: Path) -> None:
        """spectral_flatness must be a finite float >= 0."""
        result = classify_content(_load(ambient_wav))
        assert isinstance(result.spectral_flatness, float)
        assert result.spectral_flatness >= 0.0

    def test_classification_result_has_thresholds(self, rhythmic_wav: Path) -> None:
        """thresholds dict must contain the four named threshold keys."""
        result = classify_content(_load(rhythmic_wav))
        assert isinstance(result.thresholds, dict)
        assert "onset_variance_high" in result.thresholds
        assert "onset_variance_low" in result.thresholds
        assert "percussive_ratio_high" in result.thresholds
        assert "percussive_ratio_low" in result.thresholds

    def test_classification_result_thresholds_are_positive_floats(self, rhythmic_wav: Path) -> None:
        """All threshold values must be positive finite floats."""
        result = classify_content(_load(rhythmic_wav))
        for key, value in result.thresholds.items():
            assert isinstance(value, float), f"threshold '{key}' is not a float"
            assert value > 0.0, f"threshold '{key}' is not positive"


# ---------------------------------------------------------------------------
# Edge case: silence must not crash
# ---------------------------------------------------------------------------


class TestSilenceClassified:
    """GIVEN a silent WAV file
    WHEN classify_content is called
    THEN it must return a ClassificationResult without raising.
    """

    def test_silence_classified_without_error(self, tmp_path: Path) -> None:
        """A silent WAV must classify (as AMBIENT or MIXED) without raising."""
        path = tmp_path / "silence.wav"
        samples = np.zeros((48000 * 3, 2), dtype="float64")  # 3 s of silence, stereo
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = _load(path)
        result = classify_content(audio)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.content_type, ContentType)

    def test_silence_percussive_ratio_is_zero(self, tmp_path: Path) -> None:
        """A silent WAV must have percussive_ratio == 0.0 (no energy)."""
        path = tmp_path / "silence_ratio.wav"
        samples = np.zeros((48000 * 3, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = _load(path)
        result = classify_content(audio)
        assert result.percussive_ratio == 0.0


# ---------------------------------------------------------------------------
# Mono mixdown property
# ---------------------------------------------------------------------------


class TestMonoProperty:
    """Tests for AudioData.mono property (added in STORY-004 to support analysis)."""

    def test_mono_from_stereo_returns_1d(self) -> None:
        """AudioData.mono must return a 1-D array for stereo input."""
        samples = np.ones((4800, 2), dtype="float64")
        audio = AudioData(samples=samples, sample_rate=48000, channels=2, bit_depth=24)
        assert audio.mono.ndim == 1
        assert audio.mono.shape == (4800,)

    def test_mono_from_stereo_averages_channels(self) -> None:
        """AudioData.mono must average both channels for stereo input."""
        left = np.ones(4800, dtype="float64")
        right = np.zeros(4800, dtype="float64")
        samples = np.stack([left, right], axis=1)
        audio = AudioData(samples=samples, sample_rate=48000, channels=2, bit_depth=24)
        np.testing.assert_allclose(audio.mono, 0.5)

    def test_mono_from_mono_returns_same_array(self) -> None:
        """AudioData.mono must return the original array unchanged for mono input."""
        samples = np.ones(4800, dtype="float64")
        audio = AudioData(samples=samples, sample_rate=48000, channels=1, bit_depth=24)
        assert audio.mono is samples


# ---------------------------------------------------------------------------
# STORY-006: Spectral similarity loop-point detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers for STORY-006 tests
# ---------------------------------------------------------------------------


def _make_repeating_audio(sr: int = 22050, period: float = 1.0, repetitions: int = 5) -> AudioData:
    """Create mono AudioData of a single frequency sine repeated *repetitions* times.

    Because the signal is a pure, stationary sine wave the MFCC vector is
    identical at every 1-second window, so cosine similarity between any two
    windows will be very close to 1.0.

    Args:
        sr: Sample rate in Hz (kept low to make tests faster).
        period: Period of each repetition in seconds.
        repetitions: Number of full periods in the output.

    Returns:
        AudioData containing the synthetic repeating signal.
    """
    duration = period * repetitions
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float64)
    return AudioData(samples=mono, sample_rate=sr, channels=1, bit_depth=16)


def _make_short_noise_audio(sr: int = 22050, duration: float = 4.0) -> AudioData:
    """Create mono AudioData of white noise.

    White noise produces variable MFCC frames; cosine similarity between
    windows is unpredictable but typically well below 0.85.

    Args:
        sr: Sample rate in Hz.
        duration: Duration in seconds.

    Returns:
        AudioData containing white noise.
    """
    rng = np.random.default_rng(seed=0)
    mono = rng.uniform(-0.5, 0.5, int(sr * duration)).astype(np.float64)
    return AudioData(samples=mono, sample_rate=sr, channels=1, bit_depth=16)


# ---------------------------------------------------------------------------
# AC-1 / AC-2: find_loop_candidates on a repeating signal
# ---------------------------------------------------------------------------


class TestFindLoopCandidatesRepeatingSignal:
    """GIVEN audio with a repeating tonal pattern
    WHEN find_loop_candidates is called
    THEN at least one candidate is identified and all are sorted descending.
    """

    def test_find_candidates_on_repeating_signal(self) -> None:
        """A repeating sine tone must produce at least one loop candidate."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        assert len(candidates) >= 1

    def test_find_candidates_returns_sorted_by_score(self) -> None:
        """Candidates must be sorted with the highest similarity first."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        assert len(candidates) >= 2
        for i in range(len(candidates) - 1):
            assert candidates[i].similarity >= candidates[i + 1].similarity

    def test_each_candidate_has_similarity_in_unit_range(self) -> None:
        """Every candidate's similarity must be in [0.0, 1.0]."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        for cand in candidates:
            assert 0.0 <= cand.similarity <= 1.0, (
                f"Candidate similarity {cand.similarity} out of [0, 1]"
            )

    def test_each_candidate_is_loop_candidate_instance(self) -> None:
        """find_loop_candidates must return LoopCandidate objects."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        for cand in candidates:
            assert isinstance(cand, LoopCandidate)

    def test_candidate_end_greater_than_start(self) -> None:
        """Each candidate's end_seconds must be strictly greater than start_seconds."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        for cand in candidates:
            assert cand.end_seconds > cand.start_seconds


# ---------------------------------------------------------------------------
# AC-2: Repeating signal produces high-similarity candidates
# ---------------------------------------------------------------------------


class TestFindLoopCandidatesSimilarityValues:
    """GIVEN a repeating tonal audio
    WHEN candidates are ranked
    THEN the best candidate has similarity close to 1.0.
    """

    def test_best_candidate_similarity_near_one_for_repeating_signal(self) -> None:
        """A pure repeating sine tone must yield a best similarity >= 0.9."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        candidates = find_loop_candidates(audio)
        assert candidates, "Expected at least one candidate for a repeating signal"
        assert candidates[0].similarity >= 0.9, (
            f"Best similarity was {candidates[0].similarity:.4f}, expected >= 0.9"
        )


# ---------------------------------------------------------------------------
# _cosine_similarity unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Unit tests for the internal _cosine_similarity helper."""

    def test_cosine_similarity_identical_vectors_equals_1(self) -> None:
        """Cosine similarity of a vector with itself must be 1.0."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        result = _cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_vectors_equals_0(self) -> None:
        """Cosine similarity of orthogonal vectors must be 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = _cosine_similarity(a, b)
        assert abs(result) < 1e-9

    def test_cosine_similarity_zero_vector_returns_0(self) -> None:
        """Cosine similarity with a zero vector must return 0.0 (no division by zero)."""
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        result = _cosine_similarity(a, z)
        assert result == 0.0

    def test_cosine_similarity_returns_float(self) -> None:
        """_cosine_similarity must always return a Python float."""
        a = np.array([1.0, 0.5])
        b = np.array([0.5, 1.0])
        result = _cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_cosine_similarity_symmetric(self) -> None:
        """Cosine similarity must be symmetric: sim(a, b) == sim(b, a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-12


# ---------------------------------------------------------------------------
# AC-4: No candidate >= threshold raises LoopDetectionError
# ---------------------------------------------------------------------------


class TestNoHighSimilarityCandidateRaisesError:
    """GIVEN audio where no candidate reaches the 0.85 similarity threshold
    WHEN detect_loop_points is called
    THEN a LoopDetectionError is raised with a suggestion for manual overrides.
    """

    def test_no_candidates_above_threshold_raises_error(self, tmp_path: Path) -> None:
        """Very short audio must raise LoopDetectionError (file too short for any candidates)."""
        path = tmp_path / "short.wav"
        samples = np.zeros((22050, 2), dtype="float64")  # 1 s silence — too short
        sf.write(str(path), samples, 22050, subtype="PCM_16")
        audio = read_wav(path)
        with pytest.raises(LoopDetectionError):
            detect_loop_points(audio, ContentType.AMBIENT)

    def test_loop_detection_error_mentions_manual_override(self, tmp_path: Path) -> None:
        """LoopDetectionError message must mention --start / --end."""
        path = tmp_path / "short2.wav"
        samples = np.zeros((22050, 2), dtype="float64")  # 1 s silence
        sf.write(str(path), samples, 22050, subtype="PCM_16")
        audio = read_wav(path)
        with pytest.raises(LoopDetectionError, match="--start"):
            detect_loop_points(audio, ContentType.AMBIENT)


# ---------------------------------------------------------------------------
# detect_loop_points success path
# ---------------------------------------------------------------------------


class TestDetectLoopPointsSuccess:
    """GIVEN audio where a candidate achieves >= 0.85 similarity
    WHEN detect_loop_points is called
    THEN it returns a LoopRegion with valid start/end samples.
    """

    def test_detect_loop_points_returns_loop_region(self) -> None:
        """detect_loop_points must return a LoopRegion for a repeating sine tone."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        region = detect_loop_points(audio, ContentType.AMBIENT)
        assert isinstance(region, LoopRegion)

    def test_detect_loop_points_start_less_than_end(self) -> None:
        """LoopRegion.start_sample must be less than end_sample."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        region = detect_loop_points(audio, ContentType.AMBIENT)
        assert region.start_sample < region.end_sample

    def test_detect_loop_points_confidence_in_unit_range(self) -> None:
        """LoopRegion.confidence must be in [0.0, 1.0]."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        region = detect_loop_points(audio, ContentType.AMBIENT)
        assert 0.0 <= region.confidence <= 1.0

    def test_detect_loop_points_preserves_content_type(self) -> None:
        """LoopRegion.content_type must match the argument passed in."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=5)
        region = detect_loop_points(audio, ContentType.RHYTHMIC)
        assert region.content_type is ContentType.RHYTHMIC


# ---------------------------------------------------------------------------
# AC-3: --verbose shows top-3 candidates
# ---------------------------------------------------------------------------


class TestVerboseShowsCandidates:
    """GIVEN loop detection completes
    WHEN --verbose is passed
    THEN the top 3 candidates (start, end, score) are printed to stderr.
    """

    def _make_loopable_wav(self, tmp_path: Path, filename: str) -> Path:
        """Write a 5-second 440 Hz stereo sine WAV and return its path."""
        sr = 22050
        duration = 5.0
        num_samples = int(sr * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        stereo = np.stack([mono, mono], axis=1)
        wav_path = tmp_path / filename
        sf.write(str(wav_path), stereo, sr, subtype="PCM_16")
        return wav_path

    def test_verbose_shows_candidates_in_output(self, tmp_path: Path) -> None:
        """With -v the CLI must print 'loop candidate' lines to output (stderr is mixed in)."""
        from click.testing import CliRunner

        from audioloop.cli import main

        wav_path = self._make_loopable_wav(tmp_path, "loop_test.wav")
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_path), "-v"])
        assert "loop candidate" in result.output.lower()

    def test_verbose_shows_at_most_three_candidates(self, tmp_path: Path) -> None:
        """With -v the CLI must show no more than 3 loop candidates."""
        from click.testing import CliRunner

        from audioloop.cli import main

        wav_path = self._make_loopable_wav(tmp_path, "loop_test2.wav")
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_path), "-v"])
        count = result.output.lower().count("loop candidate")
        assert count <= 3


# ---------------------------------------------------------------------------
# STORY-005: Beat and tempo detection
# ---------------------------------------------------------------------------


class TestDetectBeatsRhythmic:
    """GIVEN a WAV classified as rhythmic with a steady tempo
    WHEN detect_beats is called
    THEN BPM is within ±2 of the known ground-truth tempo.
    """

    def test_detect_beats_rhythmic_fixture_bpm_within_tolerance(self, rhythmic_wav: Path) -> None:
        """Detected BPM for a 120 BPM click track must be within ±2 BPM."""
        audio = read_wav(rhythmic_wav)
        info = detect_beats(audio)
        assert info is not None, "detect_beats returned None for rhythmic fixture"
        assert abs(info.tempo - 120.0) <= 2.0, (
            f"Detected BPM {info.tempo:.2f} is not within ±2 of 120"
        )

    def test_detect_beats_returns_beat_info_instance(self, rhythmic_wav: Path) -> None:
        """detect_beats must return a BeatInfo instance for rhythmic audio."""
        audio = read_wav(rhythmic_wav)
        info = detect_beats(audio)
        assert isinstance(info, BeatInfo)

    def test_detect_beats_returns_beat_positions(self, rhythmic_wav: Path) -> None:
        """beat_positions must be a non-empty list of floats."""
        audio = read_wav(rhythmic_wav)
        info = detect_beats(audio)
        assert info is not None
        assert len(info.beat_positions) > 0
        for pos in info.beat_positions:
            assert isinstance(pos, float)

    def test_beat_positions_are_monotonically_increasing(self, rhythmic_wav: Path) -> None:
        """Each beat position must be strictly greater than the previous one."""
        audio = read_wav(rhythmic_wav)
        info = detect_beats(audio)
        assert info is not None
        positions = info.beat_positions
        assert len(positions) >= 2, "Need at least two beats to test ordering"
        for i in range(len(positions) - 1):
            assert positions[i + 1] > positions[i], (
                f"Beat positions not monotonically increasing at index {i}: "
                f"{positions[i]:.4f} >= {positions[i + 1]:.4f}"
            )

    def test_detect_beats_confidence_in_unit_range(self, rhythmic_wav: Path) -> None:
        """BeatInfo.confidence must be in [0.0, 1.0]."""
        audio = read_wav(rhythmic_wav)
        info = detect_beats(audio)
        assert info is not None
        assert 0.0 <= info.confidence <= 1.0


class TestDetectBeatsAmbient:
    """GIVEN a WAV classified as ambient (no beat)
    WHEN detect_beats is called
    THEN None is returned (no reliable tempo).
    """

    def test_detect_beats_ambient_returns_none(self, ambient_wav: Path) -> None:
        """detect_beats must return None for ambient (sine tone) audio."""
        audio = read_wav(ambient_wav)
        result = detect_beats(audio)
        assert result is None, f"Expected None for ambient audio, got {result}"


class TestDetectBeatsEdgeCases:
    """Edge-case tests for detect_beats: very short audio, graceful handling."""

    def test_detect_beats_very_short_audio_does_not_raise(self, tmp_path: Path) -> None:
        """detect_beats must not raise for very short audio (< 1 s)."""
        path = tmp_path / "short.wav"
        samples = np.zeros((4800, 2), dtype="float64")  # 0.1 s at 48 kHz
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = read_wav(path)
        result = detect_beats(audio)
        # Very short audio should return None (not enough beats), not raise.
        assert result is None or isinstance(result, BeatInfo)

    def test_detect_beats_custom_bpm_within_tolerance(self, tmp_path: Path) -> None:
        """Detected BPM for a 90 BPM click track must be within ±2 BPM."""
        path = make_rhythmic_wav(tmp_path, bpm=90.0, duration=8.0)
        audio = read_wav(path)
        info = detect_beats(audio)
        assert info is not None, "detect_beats returned None for 90 BPM click track"
        assert abs(info.tempo - 90.0) <= 2.0, (
            f"Detected BPM {info.tempo:.2f} is not within ±2 of 90"
        )


class TestDetectBeatsCLIVerbose:
    """GIVEN beat detection completed on a rhythmic file
    WHEN --verbose is passed
    THEN detected BPM and beat count are printed.
    """

    def _make_rhythmic_wav(self, tmp_path: Path) -> Path:
        """Write a 120 BPM stereo click-track WAV and return its path."""
        return make_rhythmic_wav(tmp_path, sr=48000, duration=4.0, bpm=120.0)

    def test_verbose_shows_tempo_info(self, tmp_path: Path) -> None:
        """With -v the CLI must print BPM and beat count for rhythmic content."""
        from click.testing import CliRunner

        from audioloop.cli import main

        wav_path = self._make_rhythmic_wav(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_path), "-v"])
        assert "bpm" in result.output.lower(), f"Expected 'BPM' in output. Got:\n{result.output}"

    def test_normal_mode_shows_tempo_line(self, tmp_path: Path) -> None:
        """Without --verbose the CLI must still print a 'Tempo:' line for rhythmic content."""
        from click.testing import CliRunner

        from audioloop.cli import main

        wav_path = self._make_rhythmic_wav(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, [str(wav_path)])
        assert "tempo" in result.output.lower() or "bpm" in result.output.lower(), (
            f"Expected tempo info in output. Got:\n{result.output}"
        )


# ---------------------------------------------------------------------------
# STORY-011: find_loop_candidates with target_length_seconds filtering
# ---------------------------------------------------------------------------


class TestFindLoopCandidatesTargetLength:
    """GIVEN audio with multiple loop candidates
    WHEN find_loop_candidates is called with a target_length_seconds
    THEN only candidates within the tolerance band are returned.
    """

    def test_find_candidates_with_target_length_filters_near_target(self) -> None:
        """Candidates returned when target set must all be within the tolerance."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=6)
        # Most candidates will be ~1-5 s long; target 3 s with 50% tolerance
        # to ensure at least some pass, then check they are near the target.
        target = 3.0
        tol = 0.5
        candidates = find_loop_candidates(audio, target_length_seconds=target, tolerance=tol)
        for cand in candidates:
            length = cand.end_seconds - cand.start_seconds
            assert abs(length - target) / target <= tol, (
                f"Candidate length {length:.3f}s is not within {tol * 100:.0f}% of {target}s"
            )

    def test_find_candidates_no_target_returns_all(self) -> None:
        """Without a target, find_loop_candidates must return the full candidate list."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=6)
        all_cands = find_loop_candidates(audio)
        no_target_cands = find_loop_candidates(audio, target_length_seconds=None)
        assert len(all_cands) == len(no_target_cands)

    def test_find_candidates_fallback_when_no_match(self) -> None:
        """When no candidate falls within tolerance, all candidates must be returned as fallback."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=6)
        all_cands = find_loop_candidates(audio)
        # Use an unreachably long target — no candidate will be near 1000 s.
        filtered = find_loop_candidates(audio, target_length_seconds=1000.0, tolerance=0.001)
        # Fallback: must return all candidates, not zero.
        assert len(filtered) == len(all_cands)

    def test_find_candidates_sorted_descending_after_filter(self) -> None:
        """Filtered results must still be sorted by similarity descending."""
        audio = _make_repeating_audio(sr=22050, period=1.0, repetitions=6)
        candidates = find_loop_candidates(audio, target_length_seconds=3.0, tolerance=0.5)
        for i in range(len(candidates) - 1):
            assert candidates[i].similarity >= candidates[i + 1].similarity


class TestLoopLengthToleranceConstants:
    """Tests for LOOP_LENGTH_TOLERANCE content-type constants."""

    def test_rhythmic_tolerance_is_5_percent(self) -> None:
        """Rhythmic tolerance must be 5% (0.05)."""
        assert LOOP_LENGTH_TOLERANCE[ContentType.RHYTHMIC] == pytest.approx(0.05)

    def test_ambient_tolerance_is_10_percent(self) -> None:
        """Ambient tolerance must be 10% (0.10)."""
        assert LOOP_LENGTH_TOLERANCE[ContentType.AMBIENT] == pytest.approx(0.10)

    def test_mixed_tolerance_is_7_point_5_percent(self) -> None:
        """Mixed tolerance must be 7.5% (0.075)."""
        assert LOOP_LENGTH_TOLERANCE[ContentType.MIXED] == pytest.approx(0.075)

    def test_all_content_types_have_tolerance(self) -> None:
        """Every ContentType enum member must have a tolerance entry."""
        for ct in ContentType:
            assert ct in LOOP_LENGTH_TOLERANCE, f"{ct} missing from LOOP_LENGTH_TOLERANCE"


# ---------------------------------------------------------------------------
# STORY-018: Edge case analysis tests
# ---------------------------------------------------------------------------


class TestEdgeCaseAnalysis:
    """Tests for analysis behaviour on edge-case audio fixtures (STORY-018)."""

    def test_very_short_wav_classify_returns_valid_type(self, very_short_wav: Path) -> None:
        """classify_content on a 0.1 s WAV must return a valid ContentType without crashing."""
        audio = _load(very_short_wav)
        result = classify_content(audio)
        assert result.content_type in (
            ContentType.RHYTHMIC,
            ContentType.AMBIENT,
            ContentType.MIXED,
        )

    def test_very_short_wav_classify_result_has_finite_onset_variance(
        self, very_short_wav: Path
    ) -> None:
        """classify_content on a 0.1 s WAV must populate onset_variance with a finite value."""
        audio = _load(very_short_wav)
        result = classify_content(audio)
        assert np.isfinite(result.onset_variance)

    def test_very_short_wav_detect_loop_raises_loop_detection_error(
        self, very_short_wav: Path
    ) -> None:
        """detect_loop_points on a 0.1 s WAV must raise LoopDetectionError (not crash)."""
        audio = _load(very_short_wav)
        result = classify_content(audio)
        with pytest.raises(LoopDetectionError):
            detect_loop_points(audio, result.content_type)
