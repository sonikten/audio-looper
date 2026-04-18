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

"""Duration string parsing utilities."""

from __future__ import annotations

import logging
import re

from audioloop.exceptions import DurationParseError

logger = logging.getLogger(__name__)

#: Pattern matching bar-based loop length hints: "4bars", "1bar" (case-insensitive).
_BAR_PATTERN = re.compile(r"^(\d+)\s*bars?$", re.IGNORECASE)


def parse_duration(value: str) -> float:
    """Parse a human-readable duration string into seconds.

    Supported formats:
        - Bare number: ``"3.5"`` -> 3.5 seconds
        - Seconds suffix: ``"3.5s"`` -> 3.5 seconds
        - Milliseconds suffix: ``"500ms"`` -> 0.5 seconds
        - Minutes suffix: ``"1.5m"`` or ``"1.5min"`` -> 90.0 seconds
        - Hours suffix: ``"1h"`` -> 3600.0 seconds
        - Minutes:seconds: ``"1:30"`` -> 90.0 seconds

    Args:
        value: The duration string to parse.

    Returns:
        Duration in seconds as a float.

    Raises:
        DurationParseError: If the string cannot be interpreted as a duration.
    """
    value = value.strip()

    # Try mm:ss format (e.g. "1:30", "0:30.5")
    mm_ss = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", value)
    if mm_ss:
        minutes = int(mm_ss.group(1))
        seconds = float(mm_ss.group(2))
        return minutes * 60 + seconds

    # Try suffixed formats: "30s", "1.5m", "500ms", "2min", "1h"
    suffixed = re.match(r"^(\d+(?:\.\d+)?)\s*(ms|s|min|m|h)?$", value)
    if suffixed:
        num = float(suffixed.group(1))
        unit = suffixed.group(2) or "s"  # default to seconds
        if unit == "ms":
            return num / 1000
        elif unit == "s":
            return num
        elif unit in ("m", "min"):
            return num * 60
        elif unit == "h":
            return num * 3600

    # Try bare number (no suffix matched above — fallback for safety)
    try:
        return float(value)
    except ValueError:
        raise DurationParseError(f"Cannot parse duration: '{value}'") from None


def parse_loop_length(value: str, bpm: float | None = None, beats_per_bar: int = 4) -> float:
    """Parse a loop length hint, which can be time-based or bar-based.

    Bar-based values require a known BPM to convert to seconds.  The number
    of beats per bar (i.e. the time-signature numerator) is configurable via
    *beats_per_bar*; it defaults to 4 for standard 4/4 time.  Time-based
    values delegate to :func:`parse_duration`.

    Supported bar formats:
        - ``"4bars"`` → ``4 * beats_per_bar / bpm * 60`` seconds
        - ``"1bar"``  → ``1 * beats_per_bar / bpm * 60`` seconds
          (case-insensitive)

    Supported time formats: see :func:`parse_duration`.

    Args:
        value: The loop length string to parse (e.g. ``"8s"``, ``"4bars"``).
        bpm: Detected tempo in BPM, required for bar-based values.
        beats_per_bar: Number of beats per bar (time-signature numerator).
            Defaults to 4 for 4/4 time.

    Returns:
        Loop length in seconds as a float.

    Raises:
        DurationParseError: If *value* is a bar-based spec and *bpm* is
            ``None``, or if *value* cannot be parsed at all.
    """
    bar_match = _BAR_PATTERN.match(value.strip())
    if bar_match:
        bars = int(bar_match.group(1))
        if bpm is None:
            raise DurationParseError(
                "Bar-based loop length requires a detectable tempo. "
                "Try a time-based value instead (e.g., '8s')."
            )
        return bars * beats_per_bar / bpm * 60

    return parse_duration(value)
