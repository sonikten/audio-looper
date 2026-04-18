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

"""Unit tests for looper.py — STORY-012 and STORY-023 acceptance criteria.

Tests cover:
  - create_loop with count (default and explicit)
  - create_loop with target_duration_seconds (exact fit, partial rep, fade-out)
  - create_loop format preservation
  - Error cases (invalid count, invalid duration, both supplied)
  - create_loop_streaming: count-based and duration-based streaming paths
  - Bit-exact comparison between in-memory and streaming paths
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioloop.io import read_audio
from audioloop.looper import (
    _PARTIAL_REP_FADEOUT_MS,
    MAX_LOOP_COUNT,
    STREAMING_THRESHOLD,
    create_loop,
    create_loop_streaming,
)
from audioloop.models import AudioData, ContentType, LoopRegion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ambient_audio() -> AudioData:
    """Return 4 seconds of stereo 48 kHz 16-bit sine audio for testing."""
    sr = 48000
    duration = 4.0
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([mono, mono], axis=1)
    return AudioData(samples=stereo, sample_rate=sr, channels=2, bit_depth=16)


@pytest.fixture()
def ambient_region(ambient_audio: AudioData) -> LoopRegion:
    """Return a LoopRegion covering the first 2 seconds of ambient_audio."""
    sr = ambient_audio.sample_rate
    return LoopRegion(
        start_sample=0,
        end_sample=int(sr * 2.0),
        confidence=1.0,
        content_type=ContentType.AMBIENT,
        crossfade_samples=0,
    )


def _loop_iter_length(audio: AudioData, region: LoopRegion) -> int:
    """Return the number of samples in one crossfaded loop iteration."""
    single = create_loop(
        audio=audio,
        region=region,
        content_type=ContentType.AMBIENT,
        count=1,
    )
    return single.samples.shape[0]


# ---------------------------------------------------------------------------
# TASK-3: create_loop count-based tests
# ---------------------------------------------------------------------------


class TestCreateLoopCount:
    """Tests for count-based repetition."""

    def test_create_loop_count_1(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=1 must produce a single iteration."""
        single = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=1,
        )
        assert single.samples.shape[0] > 0

    def test_create_loop_count_4(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=4 must produce output approximately 4x the single iteration length."""
        iter_len = _loop_iter_length(ambient_audio, ambient_region)
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=4,
        )
        assert result.samples.shape[0] == iter_len * 4

    def test_create_loop_default_count_is_4(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """When neither count nor target_duration_seconds is given, default is 4 reps."""
        iter_len = _loop_iter_length(ambient_audio, ambient_region)
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
        )
        assert result.samples.shape[0] == iter_len * 4

    def test_create_loop_count_2(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=2 must produce output exactly 2x the single iteration length."""
        iter_len = _loop_iter_length(ambient_audio, ambient_region)
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=2,
        )
        assert result.samples.shape[0] == iter_len * 2

    def test_create_loop_count_less_than_1_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=0 must raise ValueError."""
        with pytest.raises(ValueError, match="count must be at least 1"):
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                count=0,
            )

    def test_create_loop_both_count_and_duration_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Providing both count and target_duration_seconds must raise ValueError."""
        with pytest.raises(ValueError, match="not both"):
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                count=2,
                target_duration_seconds=10.0,
            )

    def test_create_loop_zero_duration_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """target_duration_seconds=0.0 must raise ValueError."""
        with pytest.raises(ValueError, match="target_duration_seconds must be > 0"):
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                target_duration_seconds=0.0,
            )


# ---------------------------------------------------------------------------
# TASK-3: create_loop target_duration_seconds tests
# ---------------------------------------------------------------------------


class TestCreateLoopTargetDuration:
    """Tests for duration-based repetition."""

    def test_create_loop_target_duration_approximate_length(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """target_duration_seconds should produce output close to the target.

        3s loop, 10s target -> 3 full reps + ~1s partial.
        """
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=10.0,
        )
        output_duration = result.samples.shape[0] / result.sample_rate
        # Allow up to one loop-iteration's tolerance (approx 2s loop).
        assert abs(output_duration - 10.0) < 2.5

    def test_create_loop_target_duration_does_not_exceed_target(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Output must not substantially exceed the target duration."""
        target = 10.0
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=target,
        )
        output_duration = result.samples.shape[0] / result.sample_rate
        # Partial rep is truncated, so it should not exceed target.
        assert output_duration <= target + 0.01

    def test_create_loop_target_duration_exact_fit(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """When target is an exact multiple of the loop length, there is no partial rep."""
        iter_len = _loop_iter_length(ambient_audio, ambient_region)
        sr = ambient_audio.sample_rate
        # Use exactly 2 loop iterations as target.
        target_s = (iter_len * 2) / sr
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=target_s,
        )
        assert result.samples.shape[0] == iter_len * 2

    def test_create_loop_target_duration_less_than_one_loop(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Target shorter than one loop iteration produces a short partial rep."""
        iter_len = _loop_iter_length(ambient_audio, ambient_region)
        sr = ambient_audio.sample_rate
        # Target = half a loop iteration.
        target_s = (iter_len * 0.5) / sr
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=target_s,
        )
        # Output should be approximately half the loop length.
        assert result.samples.shape[0] < iter_len

    def test_create_loop_partial_rep_has_fadeout(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """The last samples of a partial repetition must fade toward zero.

        We use a target that forces a partial repetition and then verify the
        final sample amplitude is significantly lower than the peak amplitude
        in the output, confirming that the linear fade-out was applied.
        """
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=10.0,
        )
        # Extract last _PARTIAL_REP_FADEOUT_MS samples (where the fade occurs).
        sr = result.sample_rate
        fadeout_n = int(round(_PARTIAL_REP_FADEOUT_MS / 1000.0 * sr))
        tail = result.samples[-fadeout_n:]

        # The final sample should be (close to) zero.
        if tail.ndim == 1:
            last_sample_amp = float(np.abs(tail[-1]))
        else:
            last_sample_amp = float(np.max(np.abs(tail[-1])))

        # The peak amplitude in the bulk of the signal should be non-trivial.
        bulk = result.samples[:-fadeout_n]
        peak_amp = float(np.max(np.abs(bulk)))

        assert peak_amp > 0.0, "Test audio must have non-zero amplitude"
        # Last sample should be much quieter than peak.
        assert last_sample_amp < peak_amp * 0.1


# ---------------------------------------------------------------------------
# TASK-3: create_loop format preservation
# ---------------------------------------------------------------------------


class TestCreateLoopPreservesFormat:
    """Tests that output AudioData matches source format metadata."""

    def test_preserves_sample_rate(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Output sample_rate must equal the input sample_rate."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=2,
        )
        assert result.sample_rate == ambient_audio.sample_rate

    def test_preserves_channels(self, ambient_audio: AudioData, ambient_region: LoopRegion) -> None:
        """Output channels must equal the input channels."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=2,
        )
        assert result.channels == ambient_audio.channels

    def test_preserves_bit_depth(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Output bit_depth must equal the input bit_depth."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=2,
        )
        assert result.bit_depth == ambient_audio.bit_depth

    def test_preserves_format_with_duration_mode(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Format metadata must be preserved in duration-based mode too."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            target_duration_seconds=8.0,
        )
        assert result.sample_rate == ambient_audio.sample_rate
        assert result.channels == ambient_audio.channels
        assert result.bit_depth == ambient_audio.bit_depth

    def test_samples_are_2d_for_stereo_input(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Output samples must remain 2-D for stereo input."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=3,
        )
        assert result.samples.ndim == 2

    def test_samples_second_dim_is_channels(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """Samples shape[1] must equal channels count for stereo output."""
        result = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=2,
        )
        assert result.samples.shape[1] == ambient_audio.channels


# ---------------------------------------------------------------------------
# STORY-017: MAX_LOOP_COUNT upper bound tests
# ---------------------------------------------------------------------------


class TestMaxLoopCount:
    """Tests for MAX_LOOP_COUNT constant and upper bound validation (STORY-017)."""

    def test_max_loop_count_constant(self) -> None:
        """MAX_LOOP_COUNT must equal 10000."""
        assert MAX_LOOP_COUNT == 10000

    def test_create_loop_count_exceeds_max_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=10001 must raise ValueError mentioning 'exceeds maximum'."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                count=10001,
            )

    def test_create_loop_count_at_max_does_not_raise(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=10000 must not raise a ValueError for bounds checking."""
        # We only confirm no bounds error is raised by create_loop itself;
        # the call intentionally raises CrossfadeError/etc. for unrelated reasons
        # but ValueError("exceeds maximum") must not occur.
        try:
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                count=MAX_LOOP_COUNT,
            )
        except ValueError as exc:
            assert "exceeds maximum" not in str(exc), (
                f"count={MAX_LOOP_COUNT} should not raise a bounds ValueError, got: {exc}"
            )

    def test_create_loop_count_negative_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion
    ) -> None:
        """count=-1 must raise ValueError mentioning 'at least 1'."""
        with pytest.raises(ValueError, match="at least 1"):
            create_loop(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                count=-1,
            )


# ---------------------------------------------------------------------------
# STORY-023: create_loop_streaming tests
# ---------------------------------------------------------------------------


class TestStreamingThreshold:
    """STREAMING_THRESHOLD constant must be defined and correct."""

    def test_streaming_threshold_is_10(self) -> None:
        """STREAMING_THRESHOLD must equal 10."""
        assert STREAMING_THRESHOLD == 10


class TestCreateLoopStreamingCount:
    """create_loop_streaming with count-based output."""

    def test_streaming_count_20_produces_correct_length(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """count=20 (above threshold) must produce a file with 20x the single iteration length."""
        # Determine single iteration length via in-memory path.
        single = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=1,
        )
        iter_len = single.samples.shape[0]

        out = tmp_path / "stream20.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=20,
        )
        assert out.exists()
        info = sf.info(str(out))
        assert info.frames == iter_len * 20

    def test_streaming_count_produces_output_file(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """create_loop_streaming must create the output file."""
        out = tmp_path / "streaming_out.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=15,
        )
        assert out.exists()

    def test_streaming_preserves_sample_rate(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Sample rate must be preserved in streaming output."""
        out = tmp_path / "stream_sr.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=12,
        )
        info = sf.info(str(out))
        assert info.samplerate == ambient_audio.sample_rate

    def test_streaming_preserves_channels(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Channel count must be preserved in streaming output."""
        out = tmp_path / "stream_ch.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=12,
        )
        info = sf.info(str(out))
        assert info.channels == ambient_audio.channels


class TestCreateLoopStreamingMatchesInMemory:
    """Streaming output must be bit-exact with the in-memory path for small counts.

    AC-3: Streaming output bit-exact match with in-memory output (count-based).
    """

    def test_streaming_matches_in_memory_count_5(self, tmp_path: Path) -> None:
        """count=5 via both paths must produce identical sample arrays.

        Both paths are compared after WAV round-trip so quantisation is the
        same for both.  We use 24-bit audio so the tolerance is tight.
        """
        # Build 24-bit stereo source audio directly (not using ambient_audio
        # fixture which is 16-bit) so PCM_24 tolerance applies to both paths.
        sr = 48000
        duration = 4.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        mono = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        stereo = np.stack([mono, mono], axis=1)
        audio_24 = AudioData(samples=stereo, sample_rate=sr, channels=2, bit_depth=24)

        region = LoopRegion(
            start_sample=0,
            end_sample=int(sr * 2.0),
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )

        # In-memory path — write to disk then read back to apply the same
        # PCM_24 quantisation as the streaming path.
        in_mem_audio = create_loop(
            audio=audio_24, region=region, content_type=ContentType.AMBIENT, count=5
        )
        from audioloop.io import write_audio

        ref_path = tmp_path / "ref.wav"
        write_audio(ref_path, in_mem_audio)
        ref = read_audio(ref_path)

        # Streaming path.
        out = tmp_path / "stream_cmp.wav"
        create_loop_streaming(
            audio=audio_24,
            region=region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=5,
        )
        streamed = read_audio(out)

        # After identical WAV PCM_24 round-trips both arrays must be bit-exact.
        np.testing.assert_array_equal(
            streamed.samples,
            ref.samples,
            err_msg=(
                "Streaming output must be bit-exact with in-memory output after PCM_24 round-trip"
            ),
        )

    def test_streaming_count_1_produces_correct_length(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """count=1 via streaming must produce exactly one loop iteration."""
        single = create_loop(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            count=1,
        )
        iter_len = single.samples.shape[0]

        out = tmp_path / "stream1.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            count=1,
        )
        info = sf.info(str(out))
        assert info.frames == iter_len


class TestCreateLoopStreamingDuration:
    """create_loop_streaming with target_duration_seconds."""

    def test_streaming_duration_produces_output(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Duration-based streaming must create a file at the output path."""
        out = tmp_path / "stream_dur.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            target_duration_seconds=10.0,
        )
        assert out.exists()

    def test_streaming_duration_does_not_exceed_target(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Duration-based streaming output must not substantially exceed the target."""
        target = 10.0
        out = tmp_path / "stream_dur_len.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            target_duration_seconds=target,
        )
        info = sf.info(str(out))
        output_duration = info.frames / info.samplerate
        assert output_duration <= target + 0.01

    def test_streaming_duration_approximate_length(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Duration-based streaming output length must be close to the target."""
        target = 10.0
        out = tmp_path / "stream_dur_approx.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            target_duration_seconds=target,
        )
        info = sf.info(str(out))
        output_duration = info.frames / info.samplerate
        # Allow tolerance of up to one loop iteration (approx 2 s).
        assert abs(output_duration - target) < 2.5

    def test_streaming_duration_partial_rep_has_fadeout(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Duration-based streaming partial rep must fade to near-zero at the tail.

        Mirrors the in-memory fade-out test to ensure the streaming path also
        applies a linear fade to the final partial repetition.
        """
        out = tmp_path / "stream_fadeout.wav"
        create_loop_streaming(
            audio=ambient_audio,
            region=ambient_region,
            content_type=ContentType.AMBIENT,
            output_path=out,
            target_duration_seconds=10.0,
        )
        result = read_audio(out)
        sr = result.sample_rate
        fadeout_n = int(round(_PARTIAL_REP_FADEOUT_MS / 1000.0 * sr))
        tail = result.samples[-fadeout_n:]

        if tail.ndim == 1:
            last_sample_amp = float(np.abs(tail[-1]))
        else:
            last_sample_amp = float(np.max(np.abs(tail[-1])))

        bulk = result.samples[:-fadeout_n]
        peak_amp = float(np.max(np.abs(bulk)))

        assert peak_amp > 0.0, "Test audio must have non-zero amplitude"
        assert last_sample_amp < peak_amp * 0.1


class TestCreateLoopStreamingErrors:
    """create_loop_streaming must raise errors for invalid arguments."""

    def test_neither_count_nor_duration_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Calling with neither count nor target_duration_seconds must raise ValueError."""
        out = tmp_path / "err.wav"
        with pytest.raises(ValueError):
            create_loop_streaming(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                output_path=out,
            )

    def test_both_count_and_duration_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """Providing both count and target_duration_seconds must raise ValueError."""
        out = tmp_path / "err.wav"
        with pytest.raises(ValueError, match="not both"):
            create_loop_streaming(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                output_path=out,
                count=5,
                target_duration_seconds=10.0,
            )

    def test_count_less_than_1_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """count=0 must raise ValueError."""
        out = tmp_path / "err.wav"
        with pytest.raises(ValueError, match="at least 1"):
            create_loop_streaming(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                output_path=out,
                count=0,
            )

    def test_zero_duration_raises(
        self, ambient_audio: AudioData, ambient_region: LoopRegion, tmp_path: Path
    ) -> None:
        """target_duration_seconds=0.0 must raise ValueError."""
        out = tmp_path / "err.wav"
        with pytest.raises(ValueError, match="target_duration_seconds must be > 0"):
            create_loop_streaming(
                audio=ambient_audio,
                region=ambient_region,
                content_type=ContentType.AMBIENT,
                output_path=out,
                target_duration_seconds=0.0,
            )
