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

"""Shared pytest fixtures and configuration for the audioloop test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def make_rhythmic_wav(
    tmp_path: Path,
    sr: int = 48000,
    duration: float = 4.0,
    bpm: float = 120.0,
) -> Path:
    """Generate a stereo 24-bit 48 kHz WAV containing a click track.

    Clicks are short unit-amplitude impulses placed at every beat position
    determined by *bpm*.  The signal is otherwise silent, giving a very high
    onset-strength variance and high percussive-energy ratio.

    Args:
        tmp_path: Directory in which to write the WAV file.
        sr: Sample rate in Hz.
        duration: Duration of the generated audio in seconds.
        bpm: Beats per minute, controlling the click spacing.

    Returns:
        Path to the generated WAV file.
    """
    num_samples = int(sr * duration)
    # Build a mono click track.
    mono = np.zeros(num_samples, dtype="float64")
    beat_period = sr * 60.0 / bpm  # samples between beats
    click_width = int(sr * 0.002)  # 2 ms click
    click_positions = np.arange(0, num_samples, beat_period, dtype=float).astype(int)
    for pos in click_positions:
        end = min(pos + click_width, num_samples)
        mono[pos:end] = 1.0

    # Stereo: duplicate mono to both channels.
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "rhythmic.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_24")
    return path


def make_ambient_wav(
    tmp_path: Path,
    sr: int = 48000,
    duration: float = 4.0,
    freq: float = 440.0,
) -> Path:
    """Generate a stereo 24-bit 48 kHz WAV containing a sustained sine tone.

    The sine wave has no transients, giving a near-zero onset-strength
    variance and very low percussive-energy ratio.

    Args:
        tmp_path: Directory in which to write the WAV file.
        sr: Sample rate in Hz.
        duration: Duration of the generated audio in seconds.
        freq: Frequency of the sine tone in Hz.

    Returns:
        Path to the generated WAV file.
    """
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    mono = 0.5 * np.sin(2.0 * np.pi * freq * t)

    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "ambient.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_24")
    return path


def make_mixed_wav(
    tmp_path: Path,
    sr: int = 48000,
    duration: float = 4.0,
    bpm: float = 120.0,
    freq: float = 440.0,
) -> Path:
    """Generate a stereo 24-bit 48 kHz WAV mixing a click track with a sine tone.

    The click track and sine wave are normalised to equal RMS energy before
    summing, so neither element dominates.

    Args:
        tmp_path: Directory in which to write the WAV file.
        sr: Sample rate in Hz.
        duration: Duration of the generated audio in seconds.
        bpm: Beats per minute for the click track component.
        freq: Frequency of the sine tone component in Hz.

    Returns:
        Path to the generated WAV file.
    """
    num_samples = int(sr * duration)

    # Click track.
    clicks = np.zeros(num_samples, dtype="float64")
    beat_period = sr * 60.0 / bpm
    click_width = int(sr * 0.002)
    click_positions = np.arange(0, num_samples, beat_period, dtype=float).astype(int)
    for pos in click_positions:
        end = min(pos + click_width, num_samples)
        clicks[pos:end] = 1.0

    # Sine tone.
    t = np.linspace(0, duration, num_samples, endpoint=False)
    tone = 0.5 * np.sin(2.0 * np.pi * freq * t)

    # Normalise each to unit RMS then mix at equal weight.
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    clicks_rms = _rms(clicks)
    tone_rms = _rms(tone)
    clicks_norm = clicks / clicks_rms if clicks_rms > 0 else clicks
    tone_norm = tone / tone_rms if tone_rms > 0 else tone
    mono = 0.5 * clicks_norm + 0.5 * tone_norm
    # Clip to [-1, 1] to avoid clipping artefacts in the WAV writer.
    mono = np.clip(mono, -1.0, 1.0)

    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "mixed.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_24")
    return path


# ---------------------------------------------------------------------------
# Pytest fixtures wrapping the generator functions above
# ---------------------------------------------------------------------------


@pytest.fixture()
def rhythmic_wav(tmp_path: Path) -> Path:
    """Return a Path to a synthetic rhythmic (click track) WAV file."""
    return make_rhythmic_wav(tmp_path)


@pytest.fixture()
def ambient_wav(tmp_path: Path) -> Path:
    """Return a Path to a synthetic ambient (sustained sine) WAV file."""
    return make_ambient_wav(tmp_path)


@pytest.fixture()
def mixed_wav(tmp_path: Path) -> Path:
    """Return a Path to a synthetic mixed (clicks + sine) WAV file."""
    return make_mixed_wav(tmp_path)


# ---------------------------------------------------------------------------
# Edge-case fixture generator (STORY-018)
# ---------------------------------------------------------------------------


def make_wav(
    tmp_path: Path,
    sr: int = 48000,
    duration: float = 1.0,
    channels: int = 2,
    bit_depth: int = 24,
    freq: float = 440.0,
) -> Path:
    """Generate a WAV file with configurable format parameters.

    Produces a sine-tone signal at *freq* Hz.  The resulting file is written
    using soundfile with the PCM subtype that corresponds to *bit_depth*.

    Args:
        tmp_path: Directory in which to write the WAV file.
        sr: Sample rate in Hz.
        duration: Duration of the audio in seconds.
        channels: Number of channels (1 for mono, 2 for stereo).
        bit_depth: PCM bit depth — 16, 24, or 32.
        freq: Frequency of the sine tone in Hz.

    Returns:
        Path to the generated WAV file.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    mono = np.sin(2.0 * np.pi * freq * t).astype(np.float64)
    if channels == 2:
        samples: np.ndarray = np.column_stack([mono, mono])
    else:
        samples = mono

    subtype_map: dict[int, str] = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
    path = tmp_path / f"test_{sr}_{channels}ch_{bit_depth}bit.wav"
    sf.write(str(path), samples, sr, subtype=subtype_map[bit_depth])
    return path


# ---------------------------------------------------------------------------
# Specific edge-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mono_16bit_22050_wav(tmp_path: Path) -> Path:
    """Return a Path to a 16-bit mono 22050 Hz WAV file."""
    return make_wav(tmp_path, sr=22050, channels=1, bit_depth=16)


@pytest.fixture()
def stereo_16bit_44100_wav(tmp_path: Path) -> Path:
    """Return a Path to a 16-bit stereo 44100 Hz WAV file."""
    return make_wav(tmp_path, sr=44100, channels=2, bit_depth=16)


@pytest.fixture()
def mono_24bit_96000_wav(tmp_path: Path) -> Path:
    """Return a Path to a 24-bit mono 96000 Hz WAV file."""
    return make_wav(tmp_path, sr=96000, channels=1, bit_depth=24)


@pytest.fixture()
def very_short_wav(tmp_path: Path) -> Path:
    """Return a Path to a 0.1 s stereo 24-bit 48 kHz WAV file.

    At 0.1 s the signal is too short for reliable loop detection and should
    cause detect_loop_points to raise LoopDetectionError rather than crash.
    """
    return make_wav(tmp_path, sr=48000, duration=0.1, channels=2, bit_depth=24)


@pytest.fixture()
def empty_wav(tmp_path: Path) -> Path:
    """Return a Path to a 0-frame stereo 24-bit 48 kHz WAV file."""
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.array([]).reshape(0, 2), 48000, subtype="PCM_24")
    return path


@pytest.fixture()
def malformed_wav(tmp_path: Path) -> Path:
    """Return a Path to a file with a .wav extension but random binary content."""
    path = tmp_path / "malformed.wav"
    path.write_bytes(b"NOT A WAV FILE AT ALL " * 10)
    return path


# ---------------------------------------------------------------------------
# STORY-020: FLAC and OGG format fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def flac_file(tmp_path: Path) -> Path:
    """Return a Path to a valid stereo 24-bit 48 kHz FLAC file.

    Generates a 5-second 440 Hz sine tone encoded as FLAC PCM_24.
    """
    path = tmp_path / "test.flac"
    duration = 5.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float64)
    stereo = np.column_stack([mono, mono])
    sf.write(str(path), stereo, sr, format="FLAC", subtype="PCM_24")
    return path


@pytest.fixture()
def ogg_file(tmp_path: Path) -> Path:
    """Return a Path to a valid stereo OGG Vorbis file.

    Generates a 5-second 440 Hz sine tone encoded as OGG Vorbis.
    Skips automatically when OGG Vorbis is not available on this system.
    """
    if "VORBIS" not in sf.available_subtypes("OGG"):
        pytest.skip("OGG Vorbis not available on this system")
    path = tmp_path / "test.ogg"
    duration = 5.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float64)
    stereo = np.column_stack([mono, mono])
    sf.write(str(path), stereo, sr, format="OGG", subtype="VORBIS")
    return path
