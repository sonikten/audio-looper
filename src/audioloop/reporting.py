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

"""Console output and reporting utilities."""

from __future__ import annotations

import enum
import logging

import click

from audioloop.models import AudioData

logger = logging.getLogger(__name__)


class VerbosityLevel(enum.Enum):
    """Controls the amount of output written to the console.

    Values:
        QUIET: Suppress all non-error output.
        NORMAL: Standard informational messages.
        VERBOSE: Additional detail about processing steps.
        DEBUG: Full diagnostic output including internal state.
    """

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


# Ordered list used for >= comparisons.
_LEVEL_ORDER = [
    VerbosityLevel.QUIET,
    VerbosityLevel.NORMAL,
    VerbosityLevel.VERBOSE,
    VerbosityLevel.DEBUG,
]


def _level_index(level: VerbosityLevel) -> int:
    return _LEVEL_ORDER.index(level)


class Reporter:
    """Manages all console output for audioloop, gated by a verbosity level.

    Attributes:
        verbosity: The active VerbosityLevel controlling what is printed.
    """

    def __init__(self, verbosity: VerbosityLevel) -> None:
        """Initialise the Reporter with the given verbosity level.

        Args:
            verbosity: Controls which output methods actually produce output.
        """
        self.verbosity = verbosity

    def info(self, msg: str) -> None:
        """Print *msg* to stdout when verbosity is NORMAL or higher.

        In QUIET mode this call is a no-op.

        Args:
            msg: Message to display.
        """
        if _level_index(self.verbosity) >= _level_index(VerbosityLevel.NORMAL):
            click.echo(msg)

    def verbose(self, msg: str) -> None:
        """Print *msg* to stderr when verbosity is VERBOSE or higher.

        Args:
            msg: Message to display.
        """
        if _level_index(self.verbosity) >= _level_index(VerbosityLevel.VERBOSE):
            click.echo(msg, err=True)

    def debug(self, msg: str) -> None:
        """Print *msg* to stderr when verbosity is DEBUG or higher.

        Args:
            msg: Message to display.
        """
        if _level_index(self.verbosity) >= _level_index(VerbosityLevel.DEBUG):
            click.echo(msg, err=True)

    def error(self, msg: str) -> None:
        """Print *msg* to stderr regardless of verbosity level.

        Args:
            msg: Error message to display.
        """
        click.echo(msg, err=True)

    def format_metadata(self, filename: str, audio: AudioData) -> str:
        """Format audio metadata as a human-readable string.

        Produces two lines::

            Analyzing: <filename>
              Format: <bit_depth>-bit <sample_rate_khz>kHz <channel_label>, <duration>

        Duration is formatted as ``Xm Y.Zs`` when >= 60 seconds, or ``Y.Zs`` otherwise.

        Args:
            filename: The base file name to display (e.g. ``"track.wav"``).
            audio: Loaded AudioData whose metadata will be reported.

        Returns:
            Formatted metadata string (no trailing newline).
        """
        return format_metadata(filename, audio)

    def print_metadata(self, filename: str, audio: AudioData) -> None:
        """Print formatted audio metadata via :meth:`info`.

        Args:
            filename: The base file name to display.
            audio: Loaded AudioData whose metadata will be reported.
        """
        self.info(self.format_metadata(filename, audio))


def format_metadata(filename: str, audio: AudioData) -> str:
    """Format audio metadata as a human-readable string for stdout.

    Produces two lines:

        Analyzing: <filename>
          Format: <bit_depth>-bit <sample_rate_khz>kHz <channel_label>, <duration>

    Duration is formatted as ``Xm Y.Zs`` when >= 60 seconds, or ``Y.Zs`` otherwise.

    Args:
        filename: The base file name to display (e.g. ``"track.wav"``).
        audio: Loaded AudioData whose metadata will be reported.

    Returns:
        Formatted metadata string (no trailing newline).
    """
    channel_label = "mono" if audio.is_mono else "stereo"
    sample_rate_khz = audio.sample_rate // 1000

    total_seconds = audio.duration
    minutes = int(total_seconds) // 60
    seconds = total_seconds - minutes * 60
    duration_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

    format_line = (
        f"  Format: {audio.bit_depth}-bit {sample_rate_khz}kHz {channel_label}, {duration_str}"
    )
    return f"Analyzing: {filename}\n{format_line}"
