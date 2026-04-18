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

"""Core data models for audioloop."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class ContentType(enum.Enum):
    """Classification of audio content for loop strategy selection."""

    RHYTHMIC = "rhythmic"
    AMBIENT = "ambient"
    MIXED = "mixed"


@dataclass
class AudioData:
    """Immutable representation of loaded audio samples.

    Attributes:
        samples: Raw PCM samples with shape (channels, num_samples) or (num_samples,).
        sample_rate: Samples per second, e.g. 44100.
        channels: Number of audio channels (1 = mono, 2 = stereo).
        bit_depth: Original bit depth of the source file, e.g. 16 or 24.
        source_format: Container format of the source file, e.g. "WAV", "FLAC", "OGG".
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    bit_depth: int
    source_format: str = "WAV"

    @property
    def duration(self) -> float:
        """Return total duration of the audio in seconds.

        Returns:
            Duration in seconds as a float.
        """
        num_samples = self.samples.shape[0] if self.samples.ndim > 1 else len(self.samples)
        return num_samples / self.sample_rate

    @property
    def is_mono(self) -> bool:
        """Return True if the audio has a single channel.

        Returns:
            True when channels == 1, False otherwise.
        """
        return self.channels == 1

    @property
    def mono(self) -> np.ndarray:
        """Return a 1-D mono mixdown of the audio samples.

        For single-channel audio this is just the samples array itself.
        For multi-channel audio the channels are averaged together.

        Returns:
            1-D float64 array of shape (num_samples,).
        """
        if self.samples.ndim == 1:
            return self.samples
        # samples shape is (num_samples, channels) — average across channel axis.
        return self.samples.mean(axis=1)


@dataclass
class LoopRegion:
    """Describes a detected or manually specified loop region within audio.

    Attributes:
        start_sample: Sample index where the loop begins (inclusive).
        end_sample: Sample index where the loop ends (exclusive).
        confidence: Detection confidence in [0.0, 1.0]; 1.0 means certain.
        content_type: Classified content type driving the loop strategy.
        crossfade_samples: Number of samples used for the crossfade transition.
    """

    start_sample: int
    end_sample: int
    confidence: float
    content_type: ContentType
    crossfade_samples: int
