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

"""Processing configuration dataclass for audioloop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Aggregated configuration for a single audioloop processing run.

    All options map 1-to-1 with CLI flags.  Defaults here mirror CLI defaults.

    Attributes:
        input_path: Absolute path to the input WAV file.
        output_path: Destination path for the processed output.
        loop_length: Human-readable loop length string (e.g. "4s", "2.5m"), or None.
        start: Human-readable start offset string, or None for auto-detect.
        end: Human-readable end offset string, or None for auto-detect.
        count: Number of loop repetitions to render.
        duration: Target total output duration string, or None.
        crossfade_ms: Crossfade duration in milliseconds, or None for auto.
        batch: Whether this run processes a directory of files.
        analyze_only: If True, analyse and report without writing output.
        verbosity: Verbosity level; 0 = normal, negative = quiet, positive = verbose.
    """

    input_path: Path = field(default_factory=Path)
    output_path: Path | None = None
    loop_length: str | None = None
    start: str | None = None
    end: str | None = None
    count: int = 4
    duration: str | None = None
    crossfade_ms: int | None = None
    batch: bool = False
    analyze_only: bool = False
    verbosity: int = 0
