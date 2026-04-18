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

"""Unit tests for audioloop.io.

Covers STORY-002, STORY-003, STORY-016, STORY-020, STORY-021, and STORY-023.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from audioloop.exceptions import AudioFormatError
from audioloop.io import (
    DEFAULT_MAX_FILE_SIZE_MB,
    read_audio,
    read_wav,
    resolve_output_path,
    write_audio,
    write_audio_streaming,
    write_wav,
)
from audioloop.models import AudioData


class TestReadValidStereoWav:
    """Tests for loading a valid stereo WAV file."""

    def test_returns_audio_data(self, tmp_path: Path) -> None:
        """read_wav must return an AudioData instance."""
        path = tmp_path / "stereo.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert isinstance(result, AudioData)

    def test_sample_rate_correct(self, tmp_path: Path) -> None:
        """read_wav must capture the correct sample rate."""
        path = tmp_path / "stereo.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert result.sample_rate == 48000

    def test_channels_correct(self, tmp_path: Path) -> None:
        """read_wav must report 2 channels for a stereo file."""
        path = tmp_path / "stereo.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert result.channels == 2

    def test_bit_depth_pcm24(self, tmp_path: Path) -> None:
        """read_wav must map PCM_24 subtype to bit_depth 24."""
        path = tmp_path / "stereo24.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert result.bit_depth == 24

    def test_bit_depth_pcm16(self, tmp_path: Path) -> None:
        """read_wav must map PCM_16 subtype to bit_depth 16."""
        path = tmp_path / "stereo16.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 44100, subtype="PCM_16")
        result = read_wav(path)
        assert result.bit_depth == 16

    def test_bit_depth_pcm32(self, tmp_path: Path) -> None:
        """read_wav must map PCM_32 subtype to bit_depth 32."""
        path = tmp_path / "stereo32.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_32")
        result = read_wav(path)
        assert result.bit_depth == 32

    def test_is_not_mono(self, tmp_path: Path) -> None:
        """read_wav on a stereo file must set is_mono to False."""
        path = tmp_path / "stereo.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert not result.is_mono


class TestReadValidMonoWav:
    """Tests for loading a valid mono WAV file."""

    def test_channels_is_1(self, tmp_path: Path) -> None:
        """read_wav must report channels=1 for a mono file."""
        path = tmp_path / "mono.wav"
        samples = np.zeros(4800, dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert result.channels == 1

    def test_is_mono(self, tmp_path: Path) -> None:
        """read_wav on a mono file must set is_mono to True."""
        path = tmp_path / "mono.wav"
        samples = np.zeros(4800, dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert result.is_mono

    def test_returns_audio_data(self, tmp_path: Path) -> None:
        """read_wav must return an AudioData instance for mono files."""
        path = tmp_path / "mono.wav"
        samples = np.zeros(4800, dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert isinstance(result, AudioData)


class TestReadNonexistentFile:
    """Tests for missing file error handling."""

    def test_raises_audio_format_error(self, tmp_path: Path) -> None:
        """read_wav must raise AudioFormatError for a missing file."""
        path = tmp_path / "ghost.wav"
        with pytest.raises(AudioFormatError):
            read_wav(path)

    def test_error_message_mentions_not_found(self, tmp_path: Path) -> None:
        """AudioFormatError message must include 'file was not found'."""
        path = tmp_path / "ghost.wav"
        with pytest.raises(AudioFormatError, match="file was not found"):
            read_wav(path)


class TestReadInvalidFile:
    """Tests for corrupt/non-audio file error handling."""

    def test_raises_audio_format_error(self, tmp_path: Path) -> None:
        """read_wav must raise AudioFormatError for a text file named .wav."""
        path = tmp_path / "fake.wav"
        path.write_text("this is definitely not audio data")
        with pytest.raises(AudioFormatError):
            read_wav(path)

    def test_error_message_mentions_could_not_be_read(self, tmp_path: Path) -> None:
        """AudioFormatError message must include 'could not be read as audio'."""
        path = tmp_path / "fake.wav"
        path.write_text("this is definitely not audio data")
        with pytest.raises(AudioFormatError, match="could not be read as audio"):
            read_wav(path)


class TestAudioDataDuration:
    """Tests for AudioData.duration property."""

    def test_duration_stereo(self, tmp_path: Path) -> None:
        """duration must equal num_samples / sample_rate for stereo audio."""
        path = tmp_path / "dur_stereo.wav"
        sample_rate = 48000
        num_samples = 96000  # 2 seconds
        samples = np.zeros((num_samples, 2), dtype="float64")
        sf.write(str(path), samples, sample_rate, subtype="PCM_24")
        audio = read_wav(path)
        assert audio.duration == pytest.approx(2.0)

    def test_duration_mono(self, tmp_path: Path) -> None:
        """duration must equal num_samples / sample_rate for mono audio."""
        path = tmp_path / "dur_mono.wav"
        sample_rate = 44100
        num_samples = 44100  # 1 second
        samples = np.zeros(num_samples, dtype="float64")
        sf.write(str(path), samples, sample_rate, subtype="PCM_16")
        audio = read_wav(path)
        assert audio.duration == pytest.approx(1.0)

    def test_duration_fractional(self, tmp_path: Path) -> None:
        """duration must be accurate for fractional-second audio lengths."""
        path = tmp_path / "dur_frac.wav"
        sample_rate = 48000
        num_samples = 4800  # 0.1 seconds
        samples = np.zeros((num_samples, 2), dtype="float64")
        sf.write(str(path), samples, sample_rate, subtype="PCM_24")
        audio = read_wav(path)
        assert audio.duration == pytest.approx(0.1)


class TestAudioDataSamplesRange:
    """Tests that loaded samples are in the normalised float64 range."""

    def test_silence_samples_are_zero(self, tmp_path: Path) -> None:
        """Silence samples must all be exactly 0.0 after loading."""
        path = tmp_path / "silence.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = read_wav(path)
        assert np.all(audio.samples == 0.0)

    def test_samples_within_range(self, tmp_path: Path) -> None:
        """All loaded samples must be in [-1.0, 1.0]."""
        path = tmp_path / "signal.wav"
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=(4800, 2))
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = read_wav(path)
        assert np.all(audio.samples >= -1.0)
        assert np.all(audio.samples <= 1.0)

    def test_samples_dtype_is_float64(self, tmp_path: Path) -> None:
        """Loaded samples must have dtype float64."""
        path = tmp_path / "dtype.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        audio = read_wav(path)
        assert audio.samples.dtype == np.float64


# ---------------------------------------------------------------------------
# STORY-003: write_wav tests
# ---------------------------------------------------------------------------


def _make_audio(
    sample_rate: int = 48000,
    bit_depth: int = 24,
    channels: int = 2,
    num_samples: int = 4800,
) -> AudioData:
    """Build a synthetic AudioData instance for write_wav tests."""
    rng = np.random.default_rng(0)
    if channels == 1:
        samples = rng.uniform(-0.5, 0.5, size=num_samples)
    else:
        samples = rng.uniform(-0.5, 0.5, size=(num_samples, channels))
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
    )


class TestWriteWavPreservesSampleRate:
    """write_wav must write a file with the original sample rate."""

    def test_write_wav_preserves_sample_rate(self, tmp_path: Path) -> None:
        """The written WAV must have the same sample rate as the AudioData."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "out_sr.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.samplerate == 48000


class TestWriteWavPreservesBitDepth:
    """write_wav must write a file with the original bit depth."""

    def test_write_wav_preserves_bit_depth_24(self, tmp_path: Path) -> None:
        """Writing 24-bit audio must produce a PCM_24 WAV file."""
        audio = _make_audio(bit_depth=24)
        out = tmp_path / "out_24.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.subtype == "PCM_24"

    def test_write_wav_preserves_bit_depth_16(self, tmp_path: Path) -> None:
        """Writing 16-bit audio must produce a PCM_16 WAV file."""
        audio = _make_audio(sample_rate=44100, bit_depth=16)
        out = tmp_path / "out_16.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.subtype == "PCM_16"


class TestWriteWavPreservesChannels:
    """write_wav must write a file with the original channel count."""

    def test_write_wav_preserves_channels_stereo(self, tmp_path: Path) -> None:
        """Writing stereo audio must produce a 2-channel WAV file."""
        audio = _make_audio(channels=2)
        out = tmp_path / "out_stereo.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.channels == 2

    def test_write_wav_preserves_channels_mono(self, tmp_path: Path) -> None:
        """Writing mono audio must produce a 1-channel WAV file."""
        audio = _make_audio(channels=1)
        out = tmp_path / "out_mono.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.channels == 1


class TestWriteWavRoundTrip:
    """write_wav followed by read_wav must preserve format metadata."""

    def test_round_trip_sample_rate(self, tmp_path: Path) -> None:
        """Round-trip must preserve sample rate exactly."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "rt.wav"
        write_wav(out, audio)
        reloaded = read_wav(out)
        assert reloaded.sample_rate == audio.sample_rate

    def test_round_trip_bit_depth(self, tmp_path: Path) -> None:
        """Round-trip must preserve bit depth exactly."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "rt_bd.wav"
        write_wav(out, audio)
        reloaded = read_wav(out)
        assert reloaded.bit_depth == audio.bit_depth

    def test_round_trip_channels(self, tmp_path: Path) -> None:
        """Round-trip must preserve channel count exactly."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "rt_ch.wav"
        write_wav(out, audio)
        reloaded = read_wav(out)
        assert reloaded.channels == audio.channels

    def test_round_trip_sample_values_close(self, tmp_path: Path) -> None:
        """Round-trip sample values must match within PCM_24 quantisation tolerance."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "rt_val.wav"
        write_wav(out, audio)
        reloaded = read_wav(out)
        # 24-bit PCM has ~1 LSB of quantisation: 2 / 2^24 ≈ 1.19e-7
        np.testing.assert_allclose(reloaded.samples, audio.samples, atol=1.2e-7)

    def test_round_trip_stereo_48k_24bit(self, tmp_path: Path) -> None:
        """Full round-trip for stereo 24-bit/48kHz matches AC-1."""
        audio = _make_audio(sample_rate=48000, bit_depth=24, channels=2)
        out = tmp_path / "rt_ac1.wav"
        write_wav(out, audio)
        reloaded = read_wav(out)
        assert reloaded.sample_rate == 48000
        assert reloaded.bit_depth == 24
        assert reloaded.channels == 2


class TestWriteWavError:
    """write_wav must raise AudioFormatError on write failure."""

    def test_unwritable_path_raises_audio_format_error(self, tmp_path: Path) -> None:
        """Writing to a non-existent parent directory must raise AudioFormatError."""
        audio = _make_audio()
        bad_path = tmp_path / "nonexistent_dir" / "out.wav"
        with pytest.raises(AudioFormatError, match="could not write audio to"):
            write_wav(bad_path, audio)


# ---------------------------------------------------------------------------
# STORY-003: resolve_output_path tests
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    """resolve_output_path returns the correct path in all cases."""

    def test_explicit_output_returned_as_path(self, tmp_path: Path) -> None:
        """When output is provided, it must be returned as a Path."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, "/some/other/out.wav")
        assert result == Path("/some/other/out.wav")

    def test_default_output_appends_loop_suffix(self, tmp_path: Path) -> None:
        """When output is None, the stem must gain the '_loop' suffix."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, None)
        assert result.name == "track01_loop.wav"

    def test_default_output_same_directory(self, tmp_path: Path) -> None:
        """When output is None, the result must share the input's parent directory."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, None)
        assert result.parent == tmp_path

    def test_default_output_preserves_extension(self, tmp_path: Path) -> None:
        """Default output path must keep the original .wav extension."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, None)
        assert result.suffix == ".wav"

    def test_explicit_none_uses_default(self, tmp_path: Path) -> None:
        """Passing output=None explicitly uses the default naming rule."""
        input_path = tmp_path / "ambient_drone.wav"
        result = resolve_output_path(input_path, None)
        assert result.name == "ambient_drone_loop.wav"


# ---------------------------------------------------------------------------
# STORY-016: file size and sample rate validation
# ---------------------------------------------------------------------------


def _make_small_wav(tmp_path: Path, filename: str = "small.wav") -> Path:
    """Write a minimal valid WAV file and return its path."""
    path = tmp_path / filename
    samples = np.zeros((4800, 2), dtype="float64")
    sf.write(str(path), samples, 48000, subtype="PCM_16")
    return path


class TestReadWavFileSizeValidation:
    """Tests for file-size rejection in read_wav (STORY-016 AC-1, AC-2, AC-4)."""

    def test_rejects_file_exceeding_size_limit(self, tmp_path: Path) -> None:
        """read_wav must raise AudioFormatError when the file exceeds max_file_size_mb."""
        path = _make_small_wav(tmp_path)
        # Any real WAV file is well above 0 MB, so setting the limit to 0 must reject it.
        with pytest.raises(AudioFormatError, match="file exceeds the maximum size"):
            read_wav(path, max_file_size_mb=0)

    def test_error_message_includes_limit(self, tmp_path: Path) -> None:
        """AudioFormatError message must include the configured MB limit."""
        path = _make_small_wav(tmp_path)
        # Use a limit of 0 so any real file triggers the check.
        with pytest.raises(AudioFormatError, match="0 MB"):
            read_wav(path, max_file_size_mb=0)

    def test_accepts_file_under_default_limit(self, tmp_path: Path) -> None:
        """A small WAV file must pass through with the default size limit."""
        path = _make_small_wav(tmp_path)
        audio = read_wav(path)  # default limit is 2 GB — tiny file must pass
        assert audio.sample_rate == 48000

    def test_custom_limit_accepts_file_within_limit(self, tmp_path: Path) -> None:
        """A file smaller than a custom max_file_size_mb limit must be accepted."""
        path = _make_small_wav(tmp_path)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        # Set the limit generously above the actual file size.
        generous_limit = int(file_size_mb) + 10
        audio = read_wav(path, max_file_size_mb=generous_limit)
        assert audio.sample_rate == 48000

    def test_custom_limit_rejects_file_over_limit(self, tmp_path: Path) -> None:
        """read_wav must reject a file whose size exceeds a custom limit."""
        path = _make_small_wav(tmp_path)
        # File is at least a few bytes; a limit of 0 must always reject it.
        with pytest.raises(AudioFormatError):
            read_wav(path, max_file_size_mb=0)

    def test_default_max_file_size_constant_is_2048(self) -> None:
        """DEFAULT_MAX_FILE_SIZE_MB must be 2048 (2 GB)."""
        assert DEFAULT_MAX_FILE_SIZE_MB == 2048


class TestReadWavSampleRateValidation:
    """Tests for sample-rate rejection in read_wav (STORY-016 AC-3)."""

    def test_rejects_zero_sample_rate(self, tmp_path: Path) -> None:
        """read_wav must raise AudioFormatError when sf.info reports samplerate < 1."""
        path = _make_small_wav(tmp_path)

        # Patch sf.info to return an info object with samplerate = 0.
        mock_info = MagicMock()
        mock_info.samplerate = 0
        mock_info.subtype = "PCM_16"
        mock_info.channels = 2

        with (
            patch("audioloop.io.sf.info", return_value=mock_info),
            pytest.raises(AudioFormatError, match="sample rate is invalid"),
        ):
            read_wav(path)

    def test_error_message_includes_zero_rate(self, tmp_path: Path) -> None:
        """AudioFormatError for zero sample rate must mention the invalid value."""
        path = _make_small_wav(tmp_path)

        mock_info = MagicMock()
        mock_info.samplerate = 0
        mock_info.subtype = "PCM_16"
        mock_info.channels = 2

        with (
            patch("audioloop.io.sf.info", return_value=mock_info),
            pytest.raises(AudioFormatError, match=r"sample rate is invalid \(0\)"),
        ):
            read_wav(path)

    def test_rejects_negative_sample_rate(self, tmp_path: Path) -> None:
        """read_wav must raise AudioFormatError when samplerate is negative."""
        path = _make_small_wav(tmp_path)

        mock_info = MagicMock()
        mock_info.samplerate = -1
        mock_info.subtype = "PCM_16"
        mock_info.channels = 2

        with (
            patch("audioloop.io.sf.info", return_value=mock_info),
            pytest.raises(AudioFormatError, match="sample rate is invalid"),
        ):
            read_wav(path)

    def test_valid_sample_rate_passes(self, tmp_path: Path) -> None:
        """A file with a positive sample rate must not trigger the sample-rate check."""
        path = _make_small_wav(tmp_path)
        audio = read_wav(path)
        assert audio.sample_rate > 0


# ---------------------------------------------------------------------------
# STORY-015: write_wav overwrite protection
# ---------------------------------------------------------------------------


class TestWriteWavOverwriteProtection:
    """Tests for write_wav overwrite=False behaviour (STORY-015 TASK-1)."""

    def test_write_wav_refuses_overwrite_when_exists(self, tmp_path: Path) -> None:
        """write_wav must raise AudioFormatError when output exists and overwrite=False."""
        audio = _make_audio()
        out = tmp_path / "out.wav"
        out.write_bytes(b"placeholder")
        with pytest.raises(AudioFormatError, match="output file already exists"):
            write_wav(out, audio, overwrite=False)

    def test_write_wav_error_message_contains_path(self, tmp_path: Path) -> None:
        """AudioFormatError message must include the conflicting output path."""
        audio = _make_audio()
        out = tmp_path / "conflict.wav"
        out.write_bytes(b"placeholder")
        with pytest.raises(AudioFormatError, match="conflict.wav"):
            write_wav(out, audio, overwrite=False)

    def test_write_wav_allows_overwrite_when_flag_set(self, tmp_path: Path) -> None:
        """write_wav with overwrite=True must succeed even when the output already exists."""
        audio = _make_audio()
        out = tmp_path / "out.wav"
        out.write_bytes(b"placeholder")
        write_wav(out, audio, overwrite=True)
        info = sf.info(str(out))
        assert info.samplerate == 48000

    def test_write_wav_default_overwrite_true(self, tmp_path: Path) -> None:
        """write_wav with no overwrite argument (default) must silently overwrite."""
        audio = _make_audio()
        out = tmp_path / "out.wav"
        out.write_bytes(b"placeholder")
        write_wav(out, audio)  # overwrite defaults to True
        info = sf.info(str(out))
        assert info.samplerate == 48000

    def test_write_wav_no_error_when_output_does_not_exist(self, tmp_path: Path) -> None:
        """write_wav with overwrite=False must succeed when the output does not yet exist."""
        audio = _make_audio()
        out = tmp_path / "new_output.wav"
        write_wav(out, audio, overwrite=False)
        assert out.exists()

    def test_write_wav_existing_file_not_modified_on_error(self, tmp_path: Path) -> None:
        """When write_wav raises due to overwrite=False, the existing file must be intact."""
        audio = _make_audio()
        out = tmp_path / "protected.wav"
        original_content = b"do not touch"
        out.write_bytes(original_content)
        with pytest.raises(AudioFormatError):
            write_wav(out, audio, overwrite=False)
        assert out.read_bytes() == original_content


# ---------------------------------------------------------------------------
# STORY-018: Edge case format fixtures
# ---------------------------------------------------------------------------


class TestEdgeCaseFormats:
    """Tests for read_wav behaviour across uncommon but valid WAV formats (STORY-018)."""

    def test_load_16bit_mono_22050_bit_depth(self, mono_16bit_22050_wav: Path) -> None:
        """A 16-bit mono 22050 Hz WAV must report bit_depth 16."""
        audio = read_wav(mono_16bit_22050_wav)
        assert audio.bit_depth == 16

    def test_load_16bit_mono_22050_channels(self, mono_16bit_22050_wav: Path) -> None:
        """A 16-bit mono 22050 Hz WAV must report channels 1."""
        audio = read_wav(mono_16bit_22050_wav)
        assert audio.channels == 1

    def test_load_16bit_mono_22050_sample_rate(self, mono_16bit_22050_wav: Path) -> None:
        """A 16-bit mono 22050 Hz WAV must report sample_rate 22050."""
        audio = read_wav(mono_16bit_22050_wav)
        assert audio.sample_rate == 22050

    def test_load_16bit_stereo_44100_bit_depth(self, stereo_16bit_44100_wav: Path) -> None:
        """A 16-bit stereo 44100 Hz WAV must report bit_depth 16."""
        audio = read_wav(stereo_16bit_44100_wav)
        assert audio.bit_depth == 16

    def test_load_16bit_stereo_44100_channels(self, stereo_16bit_44100_wav: Path) -> None:
        """A 16-bit stereo 44100 Hz WAV must report channels 2."""
        audio = read_wav(stereo_16bit_44100_wav)
        assert audio.channels == 2

    def test_load_16bit_stereo_44100_sample_rate(self, stereo_16bit_44100_wav: Path) -> None:
        """A 16-bit stereo 44100 Hz WAV must report sample_rate 44100."""
        audio = read_wav(stereo_16bit_44100_wav)
        assert audio.sample_rate == 44100

    def test_load_24bit_mono_96000_bit_depth(self, mono_24bit_96000_wav: Path) -> None:
        """A 24-bit mono 96000 Hz WAV must report bit_depth 24."""
        audio = read_wav(mono_24bit_96000_wav)
        assert audio.bit_depth == 24

    def test_load_24bit_mono_96000_channels(self, mono_24bit_96000_wav: Path) -> None:
        """A 24-bit mono 96000 Hz WAV must report channels 1."""
        audio = read_wav(mono_24bit_96000_wav)
        assert audio.channels == 1

    def test_load_24bit_mono_96000_sample_rate(self, mono_24bit_96000_wav: Path) -> None:
        """A 24-bit mono 96000 Hz WAV must report sample_rate 96000."""
        audio = read_wav(mono_24bit_96000_wav)
        assert audio.sample_rate == 96000

    def test_empty_wav_raises_audio_format_error(self, empty_wav: Path) -> None:
        """read_wav on a 0-frame WAV must raise AudioFormatError."""
        with pytest.raises(AudioFormatError):
            read_wav(empty_wav)

    def test_empty_wav_error_message_no_audio_data(self, empty_wav: Path) -> None:
        """AudioFormatError for an empty WAV must include 'no audio data'."""
        with pytest.raises(AudioFormatError, match="no audio data"):
            read_wav(empty_wav)

    def test_malformed_wav_raises_audio_format_error(self, malformed_wav: Path) -> None:
        """read_wav on a file with random bytes must raise AudioFormatError."""
        with pytest.raises(AudioFormatError):
            read_wav(malformed_wav)


# ---------------------------------------------------------------------------
# STORY-020: FLAC and OGG read support
# ---------------------------------------------------------------------------


class TestReadAudioFlac:
    """AC-1: read_audio loads a FLAC file correctly."""

    def test_returns_audio_data(self, flac_file: Path) -> None:
        """read_audio on a FLAC file must return an AudioData instance."""
        result = read_audio(flac_file)
        assert isinstance(result, AudioData)

    def test_sample_rate(self, flac_file: Path) -> None:
        """read_audio must capture the correct sample rate from a FLAC file."""
        result = read_audio(flac_file)
        assert result.sample_rate == 48000

    def test_channels(self, flac_file: Path) -> None:
        """read_audio must report 2 channels for a stereo FLAC file."""
        result = read_audio(flac_file)
        assert result.channels == 2

    def test_source_format_is_flac(self, flac_file: Path) -> None:
        """read_audio must populate source_format='FLAC' for a FLAC input."""
        result = read_audio(flac_file)
        assert result.source_format == "FLAC"

    def test_samples_dtype_is_float64(self, flac_file: Path) -> None:
        """Loaded FLAC samples must have dtype float64."""
        result = read_audio(flac_file)
        assert result.samples.dtype == np.float64

    def test_samples_within_range(self, flac_file: Path) -> None:
        """All loaded FLAC samples must be in [-1.0, 1.0]."""
        result = read_audio(flac_file)
        assert np.all(result.samples >= -1.0)
        assert np.all(result.samples <= 1.0)

    def test_duration_approx_5s(self, flac_file: Path) -> None:
        """Duration of the 5-second FLAC fixture must be approximately 5.0 s."""
        result = read_audio(flac_file)
        assert result.duration == pytest.approx(5.0, abs=0.1)


class TestReadAudioOgg:
    """AC-2: read_audio loads an OGG Vorbis file correctly."""

    def test_returns_audio_data(self, ogg_file: Path) -> None:
        """read_audio on an OGG file must return an AudioData instance."""
        result = read_audio(ogg_file)
        assert isinstance(result, AudioData)

    def test_sample_rate(self, ogg_file: Path) -> None:
        """read_audio must capture the correct sample rate from an OGG file."""
        result = read_audio(ogg_file)
        assert result.sample_rate == 48000

    def test_channels(self, ogg_file: Path) -> None:
        """read_audio must report 2 channels for a stereo OGG file."""
        result = read_audio(ogg_file)
        assert result.channels == 2

    def test_source_format_is_ogg(self, ogg_file: Path) -> None:
        """read_audio must populate source_format='OGG' for an OGG input."""
        result = read_audio(ogg_file)
        assert result.source_format == "OGG"

    def test_samples_dtype_is_float64(self, ogg_file: Path) -> None:
        """Loaded OGG samples must have dtype float64."""
        result = read_audio(ogg_file)
        assert result.samples.dtype == np.float64

    def test_duration_approx_5s(self, ogg_file: Path) -> None:
        """Duration of the 5-second OGG fixture must be approximately 5.0 s."""
        result = read_audio(ogg_file)
        assert result.duration == pytest.approx(5.0, abs=0.1)


class TestReadAudioSourceFormatWav:
    """source_format is populated correctly for WAV inputs."""

    def test_source_format_wav(self, tmp_path: Path) -> None:
        """read_audio must set source_format='WAV' when loading a WAV file."""
        path = tmp_path / "test.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_audio(path)
        assert result.source_format == "WAV"

    def test_default_source_format_on_audio_data(self) -> None:
        """AudioData default source_format must be 'WAV'."""
        audio = AudioData(
            samples=np.zeros(1000, dtype="float64"),
            sample_rate=48000,
            channels=1,
            bit_depth=24,
        )
        assert audio.source_format == "WAV"


class TestReadAudioBackwardCompat:
    """read_wav alias still works (backward compatibility)."""

    def test_read_wav_alias_returns_audio_data(self, tmp_path: Path) -> None:
        """read_wav must still be callable and return an AudioData instance."""
        path = tmp_path / "compat.wav"
        samples = np.zeros((4800, 2), dtype="float64")
        sf.write(str(path), samples, 48000, subtype="PCM_24")
        result = read_wav(path)
        assert isinstance(result, AudioData)

    def test_read_wav_is_read_audio(self) -> None:
        """read_wav must be the same callable object as read_audio."""
        assert read_wav is read_audio


class TestReadAudioInvalidFlac:
    """AC-4: a non-audio file with .flac extension must produce a clear error."""

    def test_malformed_flac_raises_audio_format_error(self, tmp_path: Path) -> None:
        """A file with a .flac extension but random bytes must raise AudioFormatError."""
        path = tmp_path / "bad.flac"
        path.write_bytes(b"THIS IS NOT A FLAC FILE " * 10)
        with pytest.raises(AudioFormatError):
            read_audio(path)

    def test_malformed_flac_error_message(self, tmp_path: Path) -> None:
        """AudioFormatError message for a corrupt .flac must say 'could not be read as audio'."""
        path = tmp_path / "bad.flac"
        path.write_bytes(b"THIS IS NOT A FLAC FILE " * 10)
        with pytest.raises(AudioFormatError, match="could not be read as audio"):
            read_audio(path)


# ---------------------------------------------------------------------------
# STORY-021: Multi-format write support
# ---------------------------------------------------------------------------


class TestWriteAudioFlac:
    """write_audio with output_format='flac' must produce a readable FLAC file."""

    def test_write_audio_flac_produces_flac_format(self, tmp_path: Path) -> None:
        """write_audio must write a FLAC-format file when output_format='flac'."""
        audio = _make_audio(bit_depth=24)
        out = tmp_path / "out.flac"
        write_audio(out, audio, output_format="flac")
        info = sf.info(str(out))
        assert info.format == "FLAC"

    def test_write_audio_flac_is_readable(self, tmp_path: Path) -> None:
        """The FLAC file written by write_audio must be readable by read_audio."""
        audio = _make_audio(bit_depth=24)
        out = tmp_path / "out.flac"
        write_audio(out, audio, output_format="flac")
        reloaded = read_audio(out)
        assert isinstance(reloaded, AudioData)

    def test_write_audio_flac_preserves_sample_rate(self, tmp_path: Path) -> None:
        """The FLAC file must retain the original sample rate."""
        audio = _make_audio(sample_rate=48000, bit_depth=24)
        out = tmp_path / "out.flac"
        write_audio(out, audio, output_format="flac")
        reloaded = read_audio(out)
        assert reloaded.sample_rate == 48000

    def test_write_audio_flac_preserves_channels(self, tmp_path: Path) -> None:
        """The FLAC file must retain the original channel count."""
        audio = _make_audio(channels=2)
        out = tmp_path / "out.flac"
        write_audio(out, audio, output_format="flac")
        reloaded = read_audio(out)
        assert reloaded.channels == 2

    def test_write_audio_flac_uppercase_format_accepted(self, tmp_path: Path) -> None:
        """output_format='FLAC' (uppercase) must be accepted and produce a FLAC file."""
        audio = _make_audio()
        out = tmp_path / "out.flac"
        write_audio(out, audio, output_format="FLAC")
        info = sf.info(str(out))
        assert info.format == "FLAC"


class TestWriteAudioOgg:
    """write_audio with output_format='ogg' must produce a readable OGG file."""

    def test_write_audio_ogg_produces_ogg_format(self, tmp_path: Path) -> None:
        """write_audio must write an OGG-format file when output_format='ogg'."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        audio = _make_audio()
        out = tmp_path / "out.ogg"
        write_audio(out, audio, output_format="ogg")
        info = sf.info(str(out))
        assert info.format == "OGG"

    def test_write_audio_ogg_is_readable(self, tmp_path: Path) -> None:
        """The OGG file written by write_audio must be readable by read_audio."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        audio = _make_audio()
        out = tmp_path / "out.ogg"
        write_audio(out, audio, output_format="ogg")
        reloaded = read_audio(out)
        assert isinstance(reloaded, AudioData)

    def test_write_audio_ogg_preserves_sample_rate(self, tmp_path: Path) -> None:
        """The OGG file must retain the original sample rate."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        audio = _make_audio(sample_rate=48000)
        out = tmp_path / "out.ogg"
        write_audio(out, audio, output_format="ogg")
        reloaded = read_audio(out)
        assert reloaded.sample_rate == 48000


class TestWriteAudioFormatFromExtension:
    """write_audio must infer format from path extension when output_format is None."""

    def test_flac_extension_produces_flac(self, tmp_path: Path) -> None:
        """Writing to a .flac path without explicit output_format must produce FLAC."""
        audio = _make_audio()
        out = tmp_path / "inferred.flac"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.format == "FLAC"

    def test_ogg_extension_produces_ogg(self, tmp_path: Path) -> None:
        """Writing to a .ogg path without explicit output_format must produce OGG."""
        if "VORBIS" not in sf.available_subtypes("OGG"):
            pytest.skip("OGG Vorbis not available on this system")
        audio = _make_audio()
        out = tmp_path / "inferred.ogg"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.format == "OGG"

    def test_wav_extension_produces_wav(self, tmp_path: Path) -> None:
        """Writing to a .wav path without explicit output_format must produce WAV."""
        audio = _make_audio()
        out = tmp_path / "inferred.wav"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.format == "WAV"

    def test_unknown_extension_defaults_to_wav(self, tmp_path: Path) -> None:
        """Writing to an unrecognised extension without output_format must default to WAV."""
        audio = _make_audio()
        out = tmp_path / "inferred.mp3"  # not in _FORMAT_MAP, defaults to WAV
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.format == "WAV"


class TestWriteAudioDefaultWav:
    """write_audio with no output_format and a .wav path must produce WAV."""

    def test_default_output_format_is_wav(self, tmp_path: Path) -> None:
        """write_audio with no format argument must produce a WAV file."""
        audio = _make_audio()
        out = tmp_path / "default.wav"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.format == "WAV"

    def test_default_wav_preserves_bit_depth_24(self, tmp_path: Path) -> None:
        """Default WAV write must preserve 24-bit depth."""
        audio = _make_audio(bit_depth=24)
        out = tmp_path / "default_24.wav"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.subtype == "PCM_24"

    def test_default_wav_preserves_bit_depth_16(self, tmp_path: Path) -> None:
        """Default WAV write must preserve 16-bit depth."""
        audio = _make_audio(bit_depth=16)
        out = tmp_path / "default_16.wav"
        write_audio(out, audio)
        info = sf.info(str(out))
        assert info.subtype == "PCM_16"


class TestWriteAudioBackwardCompat:
    """write_wav alias must remain fully functional (backward compatibility)."""

    def test_write_wav_is_write_audio(self) -> None:
        """write_wav must be the same callable object as write_audio."""
        assert write_wav is write_audio

    def test_write_wav_alias_produces_wav(self, tmp_path: Path) -> None:
        """Calling write_wav must still produce a valid WAV file."""
        audio = _make_audio()
        out = tmp_path / "compat.wav"
        write_wav(out, audio)
        info = sf.info(str(out))
        assert info.format == "WAV"

    def test_write_wav_alias_overwrite_false_raises(self, tmp_path: Path) -> None:
        """write_wav with overwrite=False must raise AudioFormatError when file exists."""
        audio = _make_audio()
        out = tmp_path / "compat_overwrite.wav"
        out.write_bytes(b"placeholder")
        with pytest.raises(AudioFormatError, match="output file already exists"):
            write_wav(out, audio, overwrite=False)


class TestWriteAudioUnsupportedFormat:
    """write_audio must raise AudioFormatError for unknown format keys."""

    def test_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """Passing an unsupported output_format must raise AudioFormatError."""
        audio = _make_audio()
        out = tmp_path / "out.xyz"
        with pytest.raises(AudioFormatError, match="Unsupported output format"):
            write_audio(out, audio, output_format="xyz")


# ---------------------------------------------------------------------------
# STORY-021: resolve_output_path with output_format
# ---------------------------------------------------------------------------


class TestResolveOutputPathWithFormat:
    """resolve_output_path must use output_format to choose the extension."""

    def test_format_flac_gives_flac_extension(self, tmp_path: Path) -> None:
        """output_format='flac' must produce a path with .flac extension."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, None, output_format="flac")
        assert result.suffix == ".flac"
        assert result.name == "track01_loop.flac"

    def test_format_ogg_gives_ogg_extension(self, tmp_path: Path) -> None:
        """output_format='ogg' must produce a path with .ogg extension."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, None, output_format="ogg")
        assert result.suffix == ".ogg"
        assert result.name == "track01_loop.ogg"

    def test_format_wav_gives_wav_extension(self, tmp_path: Path) -> None:
        """output_format='wav' must produce a path with .wav extension."""
        input_path = tmp_path / "track01.flac"
        result = resolve_output_path(input_path, None, output_format="wav")
        assert result.suffix == ".wav"
        assert result.name == "track01_loop.wav"

    def test_no_format_defaults_to_wav(self, tmp_path: Path) -> None:
        """When output_format is None and output is None, extension defaults to .wav."""
        input_path = tmp_path / "track01.flac"
        result = resolve_output_path(input_path, None)
        assert result.suffix == ".wav"
        assert result.name == "track01_loop.wav"

    def test_explicit_output_path_ignores_format(self, tmp_path: Path) -> None:
        """When output is provided, output_format is ignored."""
        input_path = tmp_path / "track01.wav"
        result = resolve_output_path(input_path, "/out/explicit.ogg", output_format="flac")
        assert result == Path("/out/explicit.ogg")


# ---------------------------------------------------------------------------
# STORY-023: write_audio_streaming tests
# ---------------------------------------------------------------------------


def _make_chunks(
    num_chunks: int,
    chunk_samples: int = 1000,
    channels: int = 2,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return a list of float64 audio chunk arrays for streaming write tests."""
    rng = np.random.default_rng(seed)
    if channels == 1:
        return [rng.uniform(-0.5, 0.5, size=chunk_samples) for _ in range(num_chunks)]
    return [rng.uniform(-0.5, 0.5, size=(chunk_samples, channels)) for _ in range(num_chunks)]


class TestWriteAudioStreamingBasic:
    """write_audio_streaming must produce a valid, readable audio file."""

    def test_output_file_exists(self, tmp_path: Path) -> None:
        """write_audio_streaming must create a file at the given path."""
        out = tmp_path / "stream.wav"
        chunks = _make_chunks(3, chunk_samples=1024, channels=2)
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter(chunks))
        assert out.exists()

    def test_output_is_readable(self, tmp_path: Path) -> None:
        """The file produced by write_audio_streaming must be readable by read_audio."""
        out = tmp_path / "stream.wav"
        chunks = _make_chunks(3, chunk_samples=1024, channels=2)
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter(chunks))
        audio = read_audio(out)
        assert isinstance(audio, AudioData)

    def test_output_sample_rate_preserved(self, tmp_path: Path) -> None:
        """Sample rate must be preserved in the written file."""
        out = tmp_path / "stream.wav"
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(out, sample_rate=44100, channels=2, bit_depth=16, chunks=iter(chunks))
        info = sf.info(str(out))
        assert info.samplerate == 44100

    def test_output_channels_preserved(self, tmp_path: Path) -> None:
        """Channel count must be preserved in the written file."""
        out = tmp_path / "stream.wav"
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter(chunks))
        info = sf.info(str(out))
        assert info.channels == 2

    def test_output_bit_depth_24_produces_pcm24(self, tmp_path: Path) -> None:
        """bit_depth=24 must produce PCM_24 WAV subtype."""
        out = tmp_path / "stream24.wav"
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter(chunks))
        info = sf.info(str(out))
        assert info.subtype == "PCM_24"

    def test_output_bit_depth_16_produces_pcm16(self, tmp_path: Path) -> None:
        """bit_depth=16 must produce PCM_16 WAV subtype."""
        out = tmp_path / "stream16.wav"
        chunks = _make_chunks(2, chunk_samples=512, channels=1)
        write_audio_streaming(out, sample_rate=44100, channels=1, bit_depth=16, chunks=iter(chunks))
        info = sf.info(str(out))
        assert info.subtype == "PCM_16"

    def test_total_samples_equal_sum_of_chunks(self, tmp_path: Path) -> None:
        """Total samples in the output must equal the sum of all chunk lengths."""
        out = tmp_path / "stream_len.wav"
        chunk_samples = 1024
        num_chunks = 5
        chunks = _make_chunks(num_chunks, chunk_samples=chunk_samples, channels=2)
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter(chunks))
        audio = read_audio(out)
        assert audio.samples.shape[0] == chunk_samples * num_chunks

    def test_empty_chunks_produces_empty_file(self, tmp_path: Path) -> None:
        """Passing an empty iterable must produce a zero-length (but valid) audio file."""
        out = tmp_path / "stream_empty.wav"
        write_audio_streaming(out, sample_rate=48000, channels=2, bit_depth=24, chunks=iter([]))
        info = sf.info(str(out))
        assert info.frames == 0

    def test_single_chunk_mono(self, tmp_path: Path) -> None:
        """Streaming a single mono chunk must produce correct mono output."""
        out = tmp_path / "stream_mono.wav"
        chunk = np.linspace(-0.5, 0.5, 2048, dtype=np.float64)
        write_audio_streaming(
            out, sample_rate=48000, channels=1, bit_depth=24, chunks=iter([chunk])
        )
        info = sf.info(str(out))
        assert info.channels == 1
        assert info.frames == 2048

    def test_output_flac_format(self, tmp_path: Path) -> None:
        """write_audio_streaming with output_format='flac' must produce a FLAC file."""
        out = tmp_path / "stream.flac"
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(
            out,
            sample_rate=48000,
            channels=2,
            bit_depth=24,
            chunks=iter(chunks),
            output_format="flac",
        )
        info = sf.info(str(out))
        assert info.format == "FLAC"

    def test_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """Passing an unsupported output_format must raise AudioFormatError."""
        out = tmp_path / "stream.xyz"
        chunks = _make_chunks(1, chunk_samples=512, channels=2)
        with pytest.raises(AudioFormatError, match="Unsupported output format"):
            write_audio_streaming(
                out,
                sample_rate=48000,
                channels=2,
                bit_depth=24,
                chunks=iter(chunks),
                output_format="xyz",
            )


class TestWriteAudioStreamingOverwriteProtection:
    """write_audio_streaming must respect the overwrite flag."""

    def test_overwrite_false_raises_when_file_exists(self, tmp_path: Path) -> None:
        """AudioFormatError is raised when output already exists and overwrite=False."""
        out = tmp_path / "protected.wav"
        out.write_bytes(b"placeholder")
        chunks = _make_chunks(1, chunk_samples=512, channels=2)
        with pytest.raises(AudioFormatError, match="output file already exists"):
            write_audio_streaming(
                out,
                sample_rate=48000,
                channels=2,
                bit_depth=24,
                chunks=iter(chunks),
                overwrite=False,
            )

    def test_overwrite_false_does_not_modify_existing_file(self, tmp_path: Path) -> None:
        """When write_audio_streaming raises due to overwrite=False, the original file is intact."""
        out = tmp_path / "protected2.wav"
        original = b"original content"
        out.write_bytes(original)
        chunks = _make_chunks(1, chunk_samples=512, channels=2)
        with pytest.raises(AudioFormatError):
            write_audio_streaming(
                out,
                sample_rate=48000,
                channels=2,
                bit_depth=24,
                chunks=iter(chunks),
                overwrite=False,
            )
        assert out.read_bytes() == original

    def test_overwrite_true_replaces_existing_file(self, tmp_path: Path) -> None:
        """write_audio_streaming with overwrite=True must succeed when the file already exists."""
        out = tmp_path / "overwritable.wav"
        out.write_bytes(b"old content")
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(
            out,
            sample_rate=48000,
            channels=2,
            bit_depth=24,
            chunks=iter(chunks),
            overwrite=True,
        )
        info = sf.info(str(out))
        assert info.samplerate == 48000

    def test_overwrite_default_is_true(self, tmp_path: Path) -> None:
        """Default overwrite behaviour must allow writing when file exists."""
        out = tmp_path / "default_overwrite.wav"
        out.write_bytes(b"old content")
        chunks = _make_chunks(2, chunk_samples=512, channels=2)
        write_audio_streaming(
            out,
            sample_rate=48000,
            channels=2,
            bit_depth=24,
            chunks=iter(chunks),
        )
        info = sf.info(str(out))
        assert info.samplerate == 48000
