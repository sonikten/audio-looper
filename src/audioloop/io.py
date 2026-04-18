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

"""Audio file I/O utilities (WAV, FLAC, OGG, and any format supported by libsndfile)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from audioloop.exceptions import AudioFormatError
from audioloop.models import AudioData

logger = logging.getLogger(__name__)

# Maps soundfile subtype strings to integer bit depths.
_SUBTYPE_TO_BIT_DEPTH: dict[str, int] = {
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}

# Maps integer bit depths back to soundfile PCM subtype strings.
_BIT_DEPTH_TO_SUBTYPE: dict[int, str] = {
    16: "PCM_16",
    24: "PCM_24",
    32: "PCM_32",
}

# Maps output format key to (soundfile format string, default soundfile subtype).
_FORMAT_MAP: dict[str, tuple[str, str]] = {
    "wav": ("WAV", "PCM_24"),
    "flac": ("FLAC", "PCM_24"),
    "ogg": ("OGG", "VORBIS"),
}

# Maps output format key to file extension.
_FORMAT_TO_EXT: dict[str, str] = {
    "wav": ".wav",
    "flac": ".flac",
    "ogg": ".ogg",
}

# Default maximum file size in megabytes (2 GB).
DEFAULT_MAX_FILE_SIZE_MB = 2048


def read_audio(path: Path, max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB) -> AudioData:
    """Read an audio file and return an AudioData instance.

    Supports any format recognised by libsndfile, including WAV, FLAC, and OGG
    Vorbis.  The ``source_format`` field of the returned :class:`AudioData` is
    populated from the container format reported by :func:`soundfile.info`.

    Args:
        path: Absolute path to the audio file to read.
        max_file_size_mb: Maximum allowed file size in megabytes.  Files
            larger than this threshold are rejected before any audio data is
            loaded into memory.  Defaults to 2048 MB (2 GB).

    Returns:
        AudioData containing the decoded PCM samples and metadata.
        Samples are float64 arrays shaped (num_samples, channels) for stereo
        or (num_samples,) for mono.

    Raises:
        AudioFormatError: If the file is missing, too large, has an invalid
            sample rate, is corrupt, or is an unsupported format.
    """
    if not path.exists():
        raise AudioFormatError(f"file was not found: {path}")

    # File size check — must happen before sf.read() to avoid loading huge files.
    file_size_bytes = path.stat().st_size
    max_bytes = max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise AudioFormatError(f"file exceeds the maximum size ({max_file_size_mb} MB): {path}")

    try:
        info = sf.info(str(path))
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioFormatError(f"could not be read as audio: {path}") from exc

    # Sample rate sanity check — a rate of 0 would cause division-by-zero later.
    if info.samplerate < 1:
        raise AudioFormatError(f"sample rate is invalid ({info.samplerate}): {path}")

    subtype = info.subtype  # e.g. "PCM_24", "VORBIS"
    bit_depth = _SUBTYPE_TO_BIT_DEPTH.get(subtype, 16)
    channels = info.channels
    source_format = info.format  # e.g. "WAV", "FLAC", "OGG"

    try:
        samples, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioFormatError(f"could not be read as audio: {path}") from exc

    if samples.shape[0] == 0:
        raise AudioFormatError(f"no audio data in file: {path}")

    logger.debug(
        "Loaded %s: %dHz %d-bit %d-ch %.2fs (format=%s)",
        path.name,
        sample_rate,
        bit_depth,
        channels,
        len(samples) / sample_rate if samples.ndim == 1 else samples.shape[0] / sample_rate,
        source_format,
    )

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        source_format=source_format,
    )


# Backward-compatibility alias — existing callers using read_wav continue to work.
read_wav = read_audio


def write_audio(
    path: Path,
    audio: AudioData,
    overwrite: bool = True,
    output_format: str | None = None,
) -> None:
    """Write an AudioData instance to an audio file.

    The output format is determined in priority order: explicit *output_format*
    argument, then extension of *path*, then WAV as the default.  The output
    file preserves the original sample rate and channel count.  For WAV output
    the PCM subtype is chosen from the audio bit depth; for FLAC PCM_24 is
    used; for OGG the VORBIS subtype is used.

    Args:
        path: Destination path for the output file.
        audio: AudioData to encode and write.
        overwrite: When ``False``, raise :exc:`AudioFormatError` if *path*
            already exists.  Defaults to ``True`` (silent overwrite, legacy
            behaviour).
        output_format: Explicit format key — one of ``'wav'``, ``'flac'``,
            ``'ogg'``.  When ``None`` the format is inferred from *path*'s
            extension; unknown extensions fall back to WAV.

    Raises:
        AudioFormatError: If *output_format* names an unsupported format, if
            *path* exists and *overwrite* is ``False``, or if the samples
            cannot be encoded or the path is not writable.
    """
    if path.exists() and not overwrite:
        raise AudioFormatError(f"output file already exists: {path}")

    # Determine format key: explicit arg > extension > WAV default.
    if output_format:
        fmt_key = output_format.lower()
    elif path.suffix.lower() in (".flac", ".ogg"):
        fmt_key = path.suffix.lower().lstrip(".")
    else:
        fmt_key = "wav"

    if fmt_key not in _FORMAT_MAP:
        raise AudioFormatError(f"Unsupported output format: {fmt_key}")

    sf_format, sf_subtype = _FORMAT_MAP[fmt_key]

    # For WAV, honour the source bit depth.
    if fmt_key == "wav":
        sf_subtype = _BIT_DEPTH_TO_SUBTYPE.get(audio.bit_depth, "PCM_16")

    try:
        sf.write(str(path), audio.samples, audio.sample_rate, format=sf_format, subtype=sf_subtype)
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise AudioFormatError(f"could not write audio to: {path}") from exc

    logger.debug(
        "Wrote %s: %dHz %d-bit %.2fs (format=%s)",
        path.name,
        audio.sample_rate,
        audio.bit_depth,
        audio.duration,
        sf_format,
    )


# Backward-compatibility alias — existing callers using write_wav continue to work.
write_wav = write_audio


def write_audio_streaming(
    path: Path,
    sample_rate: int,
    channels: int,
    bit_depth: int,
    chunks: Iterable[np.ndarray],
    overwrite: bool = True,
    output_format: str | None = None,
) -> None:
    """Write audio data from an iterable of chunks directly to disk.

    Uses :class:`soundfile.SoundFile` in write mode so the entire output need
    not be held in memory simultaneously.  This is suitable for generating very
    long outputs (e.g. hundreds of loop repetitions) where an in-memory
    concatenation would consume excessive RAM.

    The format/subtype selection follows the same rules as :func:`write_audio`:
    explicit *output_format* > path extension > WAV default.  For WAV output
    the subtype is chosen from *bit_depth*; for FLAC PCM_24 is used; for OGG
    VORBIS is used.

    Args:
        path: Destination path for the output file.
        sample_rate: Sample rate in Hz (e.g. 48000).
        channels: Number of audio channels (1 = mono, 2 = stereo).
        bit_depth: Bit depth of the source audio, used to choose WAV subtype.
        chunks: Iterable of numpy arrays.  Each chunk must have the same dtype
            and shape layout (1-D for mono, 2-D shaped (N, channels) for
            multi-channel).
        overwrite: When ``False``, raise :exc:`AudioFormatError` if *path*
            already exists.  Defaults to ``True``.
        output_format: Explicit format key — one of ``'wav'``, ``'flac'``,
            ``'ogg'``.  When ``None`` the format is inferred from *path*'s
            extension; unknown extensions fall back to WAV.

    Raises:
        AudioFormatError: If *output_format* names an unsupported format, if
            *path* exists and *overwrite* is ``False``, or if the audio data
            cannot be encoded or the path is not writable.
    """
    if path.exists() and not overwrite:
        raise AudioFormatError(f"output file already exists: {path}")

    # Determine format key: explicit arg > extension > WAV default.
    if output_format:
        fmt_key = output_format.lower()
    elif path.suffix.lower() in (".flac", ".ogg"):
        fmt_key = path.suffix.lower().lstrip(".")
    else:
        fmt_key = "wav"

    if fmt_key not in _FORMAT_MAP:
        raise AudioFormatError(f"Unsupported output format: {fmt_key}")

    sf_format, sf_subtype = _FORMAT_MAP[fmt_key]

    # For WAV, honour the source bit depth.
    if fmt_key == "wav":
        sf_subtype = _BIT_DEPTH_TO_SUBTYPE.get(bit_depth, "PCM_16")

    try:
        with sf.SoundFile(
            str(path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            format=sf_format,
            subtype=sf_subtype,
        ) as f:
            for chunk in chunks:
                f.write(chunk)
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise AudioFormatError(f"could not write audio to: {path}") from exc

    logger.debug(
        "Wrote (streaming) %s: %dHz %d-bit %d-ch (format=%s)",
        path.name,
        sample_rate,
        bit_depth,
        channels,
        sf_format,
    )


def resolve_output_path(
    input_path: Path,
    output: str | None,
    output_format: str | None = None,
) -> Path:
    """Determine the output audio file path.

    If an explicit output path is provided it is used as-is.  Otherwise the
    output file is placed in the same directory as the input file with
    ``_loop`` appended to the stem.  The extension is determined from
    *output_format* when given, or ``.wav`` by default.

    Args:
        input_path: Path to the source audio file.
        output: Explicit output path string, or ``None`` to use the default.
        output_format: Format key (``'wav'``, ``'flac'``, ``'ogg'``) used to
            choose the extension when *output* is ``None``.  Ignored when
            *output* is provided.

    Returns:
        Resolved :class:`pathlib.Path` for the output file.
    """
    if output is not None:
        return Path(output)
    ext = _FORMAT_TO_EXT.get(output_format or "", ".wav") if output_format else ".wav"
    return input_path.parent / f"{input_path.stem}_loop{ext}"
