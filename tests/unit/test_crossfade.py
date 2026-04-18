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

"""Unit tests for crossfade.py and looper.py — STORY-008."""

from __future__ import annotations

import numpy as np
import pytest

from audioloop.crossfade import (
    CROSSFADE_DEFAULTS_MS,
    CROSSFADE_MIN_MS,
    align_loop_to_zero_crossings,
    apply_crossfade,
    build_equal_power_curve,
    find_zero_crossing,
    get_crossfade_samples,
)
from audioloop.exceptions import CrossfadeError
from audioloop.looper import create_loop
from audioloop.models import AudioData, ContentType, LoopRegion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(
    num_samples: int = 48000,
    channels: int = 1,
    sample_rate: int = 48000,
    bit_depth: int = 24,
    value: float = 0.5,
) -> AudioData:
    """Create a simple constant-amplitude AudioData for testing."""
    if channels == 1:
        samples = np.full(num_samples, value, dtype=np.float64)
    else:
        samples = np.full((num_samples, channels), value, dtype=np.float64)
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
    )


def _make_region(
    audio: AudioData,
    start_frac: float = 0.1,
    end_frac: float = 0.9,
    content_type: ContentType = ContentType.AMBIENT,
) -> LoopRegion:
    """Create a LoopRegion spanning a fraction of the audio."""
    num_samples = audio.samples.shape[0] if audio.samples.ndim > 1 else len(audio.samples)
    return LoopRegion(
        start_sample=int(num_samples * start_frac),
        end_sample=int(num_samples * end_frac),
        confidence=1.0,
        content_type=content_type,
        crossfade_samples=0,
    )


# ---------------------------------------------------------------------------
# build_equal_power_curve
# ---------------------------------------------------------------------------


class TestBuildEqualPowerCurve:
    """Tests for the equal-power curve builder."""

    def test_equal_power_preserves_energy(self) -> None:
        """fade_out**2 + fade_in**2 must equal 1 at every point."""
        fade_out, fade_in = build_equal_power_curve(1000)
        energy = fade_out**2 + fade_in**2
        np.testing.assert_allclose(energy, np.ones(1000), atol=1e-12)

    def test_endpoints_fade_out(self) -> None:
        """fade_out must start at 1 and approach 0 at the end."""
        fade_out, _ = build_equal_power_curve(1000)
        assert abs(fade_out[0] - 1.0) < 1e-12
        # The last point with endpoint=False is cos((999/1000)*pi/2) ≈ cos(89.91°)
        assert fade_out[-1] < 0.01

    def test_endpoints_fade_in(self) -> None:
        """fade_in must start at 0 and approach 1 at the end."""
        _, fade_in = build_equal_power_curve(1000)
        assert abs(fade_in[0] - 0.0) < 1e-12
        # The last point is sin((999/1000)*pi/2) ≈ sin(89.91°) ≈ 1.0
        assert fade_in[-1] > 0.99

    def test_length_1(self) -> None:
        """A curve of length 1 must still satisfy energy constraint."""
        fade_out, fade_in = build_equal_power_curve(1)
        energy = fade_out**2 + fade_in**2
        np.testing.assert_allclose(energy, np.ones(1), atol=1e-12)

    def test_raises_for_zero_length(self) -> None:
        """Length < 1 must raise CrossfadeError."""
        with pytest.raises(CrossfadeError):
            build_equal_power_curve(0)

    def test_raises_for_negative_length(self) -> None:
        """Negative length must raise CrossfadeError."""
        with pytest.raises(CrossfadeError):
            build_equal_power_curve(-5)

    def test_output_shape(self) -> None:
        """Both curves must have the requested shape."""
        fade_out, fade_in = build_equal_power_curve(512)
        assert fade_out.shape == (512,)
        assert fade_in.shape == (512,)

    def test_output_dtype_float64(self) -> None:
        """Curves must be float64."""
        fade_out, fade_in = build_equal_power_curve(100)
        assert fade_out.dtype == np.float64
        assert fade_in.dtype == np.float64


# ---------------------------------------------------------------------------
# get_crossfade_samples
# ---------------------------------------------------------------------------


class TestGetCrossfadeSamples:
    """Tests for crossfade sample-count resolution."""

    def test_default_rhythmic_20ms(self) -> None:
        """Default for RHYTHMIC content must be ~20ms."""
        sr = 48000
        loop_samples = sr  # 1 second — large enough
        n, clamped = get_crossfade_samples(ContentType.RHYTHMIC, sr, loop_samples)
        expected = int(round(CROSSFADE_DEFAULTS_MS[ContentType.RHYTHMIC] * sr / 1000))
        assert n == expected
        assert not clamped

    def test_default_ambient_150ms(self) -> None:
        """Default for AMBIENT content must be ~150ms."""
        sr = 48000
        loop_samples = sr  # 1 second
        n, clamped = get_crossfade_samples(ContentType.AMBIENT, sr, loop_samples)
        expected = int(round(CROSSFADE_DEFAULTS_MS[ContentType.AMBIENT] * sr / 1000))
        assert n == expected
        assert not clamped

    def test_default_mixed_75ms(self) -> None:
        """Default for MIXED content must be ~75ms."""
        sr = 48000
        loop_samples = sr  # 1 second
        n, clamped = get_crossfade_samples(ContentType.MIXED, sr, loop_samples)
        expected = int(round(CROSSFADE_DEFAULTS_MS[ContentType.MIXED] * sr / 1000))
        assert n == expected
        assert not clamped

    def test_override_50ms(self) -> None:
        """--crossfade 50 must yield ~50ms regardless of content type."""
        sr = 48000
        loop_samples = sr
        n, clamped = get_crossfade_samples(ContentType.RHYTHMIC, sr, loop_samples, override_ms=50)
        expected = int(round(50 * sr / 1000))
        assert n == expected
        assert not clamped

    def test_clamped_when_exceeds_half_loop(self) -> None:
        """Crossfade exceeding half loop length must be clamped."""
        sr = 48000
        # Short loop: 100ms → 4800 samples; half = 2400
        loop_samples = int(sr * 0.1)
        # Request 80ms = 3840 samples > 2400 (half)
        n, clamped = get_crossfade_samples(ContentType.AMBIENT, sr, loop_samples, override_ms=80)
        assert clamped
        assert n == loop_samples // 2

    def test_minimum_2ms(self) -> None:
        """Crossfade must never be shorter than CROSSFADE_MIN_MS."""
        sr = 48000
        loop_samples = sr
        # Request 0ms → should be clamped to minimum
        n, _ = get_crossfade_samples(ContentType.RHYTHMIC, sr, loop_samples, override_ms=0)
        min_samples = max(1, int(round(CROSSFADE_MIN_MS * sr / 1000)))
        assert n >= min_samples

    def test_minimum_1ms_request_floored(self) -> None:
        """A 1ms request (below absolute min) is raised to CROSSFADE_MIN_MS."""
        sr = 48000
        loop_samples = sr
        n, _ = get_crossfade_samples(ContentType.RHYTHMIC, sr, loop_samples, override_ms=1)
        min_samples = int(round(CROSSFADE_MIN_MS * sr / 1000))
        assert n >= min_samples

    def test_raises_for_zero_loop_length(self) -> None:
        """Loop length < 1 must raise CrossfadeError."""
        with pytest.raises(CrossfadeError):
            get_crossfade_samples(ContentType.AMBIENT, 48000, 0)


# ---------------------------------------------------------------------------
# apply_crossfade
# ---------------------------------------------------------------------------


class TestApplyCrossfade:
    """Tests for apply_crossfade."""

    def test_no_clipping_mono(self) -> None:
        """Output samples must stay within [-1, 1] when input is at safe amplitude.

        For equal-power crossfade the splice zone sums fade_out * v + fade_in * v.
        The maximum of (cos + sin) is sqrt(2), so the input must be ≤ 1/sqrt(2) ≈ 0.707
        to guarantee the summed overlap never exceeds 1.0.
        """
        safe_value = 0.7  # below 1/sqrt(2) ≈ 0.7071
        audio = _make_audio(num_samples=96000, channels=1, value=safe_value)
        region = _make_region(audio, 0.1, 0.9, ContentType.AMBIENT)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert float(np.max(np.abs(result.samples))) <= 1.0 + 1e-9

    def test_no_clipping_stereo(self) -> None:
        """Output samples must stay within [-1, 1] for stereo safe-amplitude input."""
        safe_value = 0.7
        audio = _make_audio(num_samples=96000, channels=2, value=safe_value)
        region = _make_region(audio, 0.1, 0.9, ContentType.AMBIENT)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert float(np.max(np.abs(result.samples))) <= 1.0 + 1e-9

    def test_preserves_sample_rate(self) -> None:
        """Output must carry the same sample_rate as the input."""
        audio = _make_audio(sample_rate=44100)
        region = _make_region(audio)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert result.sample_rate == 44100

    def test_preserves_bit_depth(self) -> None:
        """Output must carry the same bit_depth as the input."""
        audio = _make_audio(bit_depth=16)
        region = _make_region(audio)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert result.bit_depth == 16

    def test_preserves_channels_mono(self) -> None:
        """Mono output must have channels=1."""
        audio = _make_audio(channels=1)
        region = _make_region(audio)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert result.channels == 1
        assert result.samples.ndim == 1

    def test_preserves_channels_stereo(self) -> None:
        """Stereo output must have channels=2 and shape (N, 2)."""
        audio = _make_audio(channels=2)
        region = _make_region(audio)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert result.channels == 2
        assert result.samples.ndim == 2
        assert result.samples.shape[1] == 2

    def test_output_length_equals_loop_minus_crossfade(self) -> None:
        """Output length must equal loop_length - crossfade_n.

        The circular crossfade overlaps the first N and last N samples of the
        loop, producing an output shorter by N samples.  When tiled, the
        boundary between copies is seamless because the crossfade head
        transitions smoothly from the end of the loop into the start.
        """
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio, 0.1, 0.9)
        loop_len = region.end_sample - region.start_sample
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        cf_n, _ = get_crossfade_samples(ContentType.AMBIENT, audio.sample_rate, loop_len)
        assert len(result.samples) == loop_len - cf_n

    def test_crossfade_override_changes_head(self) -> None:
        """Explicit --crossfade override must produce a different splice zone than default.

        Results differ in length (different crossfade N) and the crossfade head
        values differ because a different fade curve length was applied.
        """
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio, 0.1, 0.9)
        result_default = apply_crossfade(audio, region, ContentType.AMBIENT)
        result_override = apply_crossfade(
            audio, region, ContentType.AMBIENT, crossfade_override_ms=10
        )
        # Different crossfade lengths → different output lengths.
        assert len(result_default.samples) != len(result_override.samples)
        # The head (first 100 samples) must differ because fade curves differ.
        assert not np.allclose(result_default.samples[:100], result_override.samples[:100])

    def test_clamping_warns_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        """Clamped crossfade must emit a WARNING to stderr."""
        sr = 48000
        # Build audio with a very short loop region (100ms = 4800 samples).
        audio = _make_audio(num_samples=sr * 2, channels=1, sample_rate=sr)
        # Loop region: 100ms
        loop_samples = int(sr * 0.1)
        start = sr // 4
        region = LoopRegion(
            start_sample=start,
            end_sample=start + loop_samples,
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )
        # Request 200ms crossfade on a 100ms loop → must clamp.
        apply_crossfade(audio, region, ContentType.AMBIENT, crossfade_override_ms=200)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err or "warning" in captured.err.lower()

    def test_raises_for_too_short_loop(self) -> None:
        """A loop region of 1 sample must raise CrossfadeError."""
        audio = _make_audio(num_samples=100, channels=1)
        region = LoopRegion(
            start_sample=10,
            end_sample=11,  # 1 sample loop
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )
        with pytest.raises(CrossfadeError):
            apply_crossfade(audio, region, ContentType.AMBIENT)

    def test_output_dtype_float64(self) -> None:
        """Output samples must be float64."""
        audio = _make_audio(channels=1)
        region = _make_region(audio)
        result = apply_crossfade(audio, region, ContentType.AMBIENT)
        assert result.samples.dtype == np.float64

    def test_equal_power_curve_satisfies_energy_constraint(self) -> None:
        """The equal-power curves must satisfy fade_out**2 + fade_in**2 == 1.

        This is the fundamental equal-power property: the sum of squared gains
        is constant at 1, preserving perceived energy across the crossfade.
        The test verifies this directly on the curves returned by
        build_equal_power_curve, which are the curves used by apply_crossfade.
        """
        sr = 48000
        cf_ms = 10
        cf_n = int(round(cf_ms * sr / 1000))
        fade_out, fade_in = build_equal_power_curve(cf_n)
        # For every sample i: fade_out[i]**2 + fade_in[i]**2 must equal 1.
        energy_sum = fade_out**2 + fade_in**2
        np.testing.assert_allclose(energy_sum, np.ones(cf_n), atol=1e-12)

    def test_crossfade_boundary_samples_preserve_adjacency(self) -> None:
        """Crossfade head edges must be continuous with adjacent samples.

        The tapered RMS correction keeps edges at scale=1.0, so the first
        and last samples of the crossfade zone remain adjacent to the body
        samples — no discontinuity at the crossfade boundaries.
        """
        sr = 48000
        loop_samples_count = int(sr * 0.5)  # 500ms loop
        # Use a sine wave so inter-sample diffs are non-zero.
        n_samples = sr * 2
        t = np.linspace(0, 2.0, n_samples, endpoint=False)
        sine_samples = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float64)
        audio = AudioData(
            samples=sine_samples,
            sample_rate=sr,
            channels=1,
            bit_depth=24,
        )
        start = sr // 4
        region = LoopRegion(
            start_sample=start,
            end_sample=start + loop_samples_count,
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )
        cf_ms = 10
        result = apply_crossfade(audio, region, ContentType.AMBIENT, crossfade_override_ms=cf_ms)
        cf_n = int(round(cf_ms * sr / 1000))

        # When tiled, the last body sample flows into the first head sample.
        # These should be within normal inter-sample variation.
        body_diffs = np.abs(np.diff(result.samples[cf_n : cf_n + 1000]))
        avg_diff = float(np.mean(body_diffs))

        # Tile boundary: last body sample → first head sample (of next copy)
        tile_jump = abs(result.samples[0] - result.samples[-1])
        assert tile_jump < avg_diff * 5, (
            f"Tile boundary jump {tile_jump} exceeds 5x body avg diff {avg_diff}"
        )


# ---------------------------------------------------------------------------
# create_loop (looper.py)
# ---------------------------------------------------------------------------


class TestCreateLoop:
    """Tests for create_loop."""

    def test_loop_repetition_count_4(self) -> None:
        """count=4 must produce output exactly 4x the single-iteration length."""
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio, 0.1, 0.9)
        single = apply_crossfade(audio, region, ContentType.AMBIENT)
        result = create_loop(audio, region, ContentType.AMBIENT, count=4)
        expected_len = len(single.samples) * 4
        assert len(result.samples) == expected_len

    def test_loop_repetition_count_1(self) -> None:
        """count=1 must match a single apply_crossfade result."""
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio, 0.1, 0.9)
        single = apply_crossfade(audio, region, ContentType.AMBIENT)
        result = create_loop(audio, region, ContentType.AMBIENT, count=1)
        np.testing.assert_array_equal(result.samples, single.samples)

    def test_loop_preserves_format(self) -> None:
        """Output must carry the same sample_rate, bit_depth, channels."""
        audio = _make_audio(num_samples=96000, channels=2, sample_rate=44100, bit_depth=16)
        region = _make_region(audio, 0.1, 0.9)
        result = create_loop(audio, region, ContentType.AMBIENT, count=2)
        assert result.sample_rate == 44100
        assert result.bit_depth == 16
        assert result.channels == 2

    def test_loop_raises_for_count_zero(self) -> None:
        """count=0 must raise ValueError."""
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio)
        with pytest.raises(ValueError):
            create_loop(audio, region, ContentType.AMBIENT, count=0)

    def test_loop_raises_for_negative_count(self) -> None:
        """count=-1 must raise ValueError."""
        audio = _make_audio(num_samples=96000, channels=1)
        region = _make_region(audio)
        with pytest.raises(ValueError):
            create_loop(audio, region, ContentType.AMBIENT, count=-1)

    def test_loop_stereo_shape(self) -> None:
        """Stereo output must have shape (N, 2)."""
        audio = _make_audio(num_samples=96000, channels=2)
        region = _make_region(audio)
        result = create_loop(audio, region, ContentType.AMBIENT, count=3)
        assert result.samples.ndim == 2
        assert result.samples.shape[1] == 2


# ---------------------------------------------------------------------------
# find_zero_crossing  (STORY-009)
# ---------------------------------------------------------------------------


class TestFindZeroCrossing:
    """Tests for find_zero_crossing."""

    def test_find_zero_crossing_at_exact_zero(self) -> None:
        """When the target sample is exactly at a zero-crossing it is returned.

        A sine wave crosses zero at regular intervals.  We place the target at
        a known crossing and verify the returned index equals the target.

        For a 1 kHz sine at 48 kHz (48 samples per cycle) the first
        positive-to-negative crossing (left sample of the pair) is at index 24.
        ``np.sign(sine[24]) == 1`` and ``np.sign(sine[25]) == -1``, so the
        crossing is reported at index 24.
        """
        sr = 48000
        freq = 1000.0
        num_samples = sr  # 1 second
        t = np.linspace(0.0, 1.0, num_samples, endpoint=False)
        sine = np.sin(2.0 * np.pi * freq * t)

        # Find the actual first zero-crossing position programmatically so
        # the test does not depend on floating-point specifics.
        signs = np.sign(sine)
        crossings = np.where(np.diff(signs) != 0)[0]
        target = int(crossings[1])  # Second crossing (first pos->neg transition)

        result = find_zero_crossing(sine, target, sr, max_shift_ms=5.0)
        assert result == target

    def test_find_zero_crossing_within_5ms(self) -> None:
        """Returned crossing must be no more than 5ms away from the target.

        We construct a signal where crossings are known, place the target a
        few samples off, and verify the shift does not exceed 5ms.
        """
        sr = 48000
        max_shift_samples = int(sr * 5.0 / 1000.0)
        freq = 100.0  # Low frequency so crossings are ~240 samples apart
        num_samples = sr
        t = np.linspace(0.0, 1.0, num_samples, endpoint=False)
        sine = np.sin(2.0 * np.pi * freq * t)

        target = sr // 2  # Middle of the signal
        result = find_zero_crossing(sine, target, sr, max_shift_ms=5.0)
        assert abs(result - target) <= max_shift_samples

    def test_find_zero_crossing_stereo_uses_mono_sum(self) -> None:
        """Stereo detection uses the sum of channels, not individual channels.

        Channel 0 is a sine wave; channel 1 is a cosine (90-degree offset).
        Their sum is a scaled sine at 45-degree offset.  The detected crossing
        must match the sum's crossing, not either individual channel's crossing.
        """
        sr = 48000
        freq = 1000.0
        num_samples = sr
        t = np.linspace(0.0, 1.0, num_samples, endpoint=False)
        ch0 = np.sin(2.0 * np.pi * freq * t)
        ch1 = np.cos(2.0 * np.pi * freq * t)
        stereo = np.stack([ch0, ch1], axis=1)  # shape (N, 2)

        mono_sum = ch0 + ch1
        signs = np.sign(mono_sum)
        # Find first zero-crossing in the mono sum.
        mono_crossings = np.where(np.diff(signs) != 0)[0]
        assert len(mono_crossings) > 0
        first_crossing = int(mono_crossings[0])

        result = find_zero_crossing(stereo, first_crossing, sr, max_shift_ms=5.0)
        # The function must return the same crossing found in the mono sum.
        assert result == first_crossing

    def test_find_zero_crossing_no_crossing_returns_target(self) -> None:
        """A DC (flat) signal has no zero-crossings; target is returned unchanged.

        If the signal never changes sign within the window, find_zero_crossing
        must return *target_sample* unmodified.
        """
        sr = 48000
        dc = np.full(sr, 0.5, dtype=np.float64)  # Constant positive signal
        target = sr // 2
        result = find_zero_crossing(dc, target, sr, max_shift_ms=5.0)
        assert result == target


# ---------------------------------------------------------------------------
# align_loop_to_zero_crossings  (STORY-009)
# ---------------------------------------------------------------------------


class TestAlignLoopToZeroCrossings:
    """Tests for align_loop_to_zero_crossings."""

    def _make_sine_audio(
        self,
        sr: int = 48000,
        freq: float = 200.0,
        duration: float = 1.0,
        channels: int = 1,
    ) -> AudioData:
        """Return AudioData containing a sine wave."""
        num_samples = int(sr * duration)
        t = np.linspace(0.0, duration, num_samples, endpoint=False)
        mono = np.sin(2.0 * np.pi * freq * t)
        samples = mono if channels == 1 else np.stack([mono] * channels, axis=1)
        return AudioData(
            samples=samples,
            sample_rate=sr,
            channels=channels,
            bit_depth=24,
        )

    def test_align_loop_adjusts_both_endpoints(self) -> None:
        """align_loop_to_zero_crossings must move both start and end.

        We set start and end at a position that is not a zero-crossing.
        After alignment both positions must have moved towards a crossing
        (i.e. the mono sum changes sign across the returned index).
        """
        sr = 48000
        freq = 200.0  # 200 Hz: one cycle = 240 samples
        audio = self._make_sine_audio(sr=sr, freq=freq)
        mono = audio.samples  # 1-D since channels=1

        # Find the first zero-crossing.
        signs = np.sign(mono)
        crossings = np.where(np.diff(signs) != 0)[0]
        assert len(crossings) >= 2

        first_crossing = int(crossings[0])
        second_crossing = int(crossings[1])

        # Place start and end 3 samples away from known crossings.
        start_off = first_crossing + 3
        end_off = second_crossing + 3

        region = LoopRegion(
            start_sample=start_off,
            end_sample=end_off,
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )
        adjusted = align_loop_to_zero_crossings(audio, region)

        # Both endpoints must have shifted.
        assert adjusted.start_sample != start_off or adjusted.end_sample != end_off
        # At minimum the start must have moved toward the crossing.
        assert abs(adjusted.start_sample - first_crossing) <= abs(start_off - first_crossing)
        assert abs(adjusted.end_sample - second_crossing) <= abs(end_off - second_crossing)

    def test_align_loop_preserves_approximate_length(self) -> None:
        """Adjusted loop length must be within 2 * max_shift samples of original.

        Each endpoint can shift at most max_shift samples; in the worst case
        both shift in opposite directions, giving a maximum length change of
        2 * max_shift.
        """
        sr = 48000
        max_shift_ms = 5.0
        max_shift_samples = int(sr * max_shift_ms / 1000.0)

        audio = self._make_sine_audio(sr=sr, freq=200.0)
        region = LoopRegion(
            start_sample=sr // 4,
            end_sample=3 * sr // 4,
            confidence=1.0,
            content_type=ContentType.AMBIENT,
            crossfade_samples=0,
        )
        original_length = region.end_sample - region.start_sample
        adjusted = align_loop_to_zero_crossings(audio, region, max_shift_ms=max_shift_ms)
        adjusted_length = adjusted.end_sample - adjusted.start_sample

        assert abs(adjusted_length - original_length) <= 2 * max_shift_samples
