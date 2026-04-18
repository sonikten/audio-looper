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

"""Domain exception hierarchy for audioloop."""


class AudioLoopError(Exception):
    """Base exception for all audioloop errors."""


class AudioFormatError(AudioLoopError):
    """Raised when an audio file has an unsupported or invalid format."""


class LoopDetectionError(AudioLoopError):
    """Raised when loop detection fails to find a suitable loop region."""


class CrossfadeError(AudioLoopError):
    """Raised when crossfade generation encounters an invalid configuration."""


class DurationParseError(AudioLoopError):
    """Raised when a duration string cannot be parsed into a numeric value."""


class BatchError(AudioLoopError):
    """Raised when batch processing encounters a fatal error."""
