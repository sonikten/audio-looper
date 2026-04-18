# audioloop

Auto-detect and create seamless audio loops from audio files (WAV, FLAC, OGG).

## Features

- Automatic content classification (rhythmic, ambient, or mixed) to select the best loop strategy
- Spectral similarity loop-point detection using MFCC analysis, beat-aligned for rhythmic content
- Zero-crossing alignment to eliminate click artefacts at loop boundaries
- Equal-power crossfade with per-content-type defaults and manual override
- Loop assembly with repetition count or target total duration
- Manual loop region override via `--start` and `--end`
- Loop length hint via `--loop-length` (seconds or bars)
- Analyze-only mode to inspect candidates and get suggested commands without writing output
- Batch processing of entire directories

## Prerequisites

- Python 3.12 or later
- libsndfile (required by the `soundfile` package)
  - Debian/Ubuntu: `sudo apt-get install libsndfile1`
  - macOS (Homebrew): `brew install libsndfile`
  - Windows: bundled with the `soundfile` wheel; no separate install needed

## Installation

```bash
cd project/
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

**Basic — auto-detect loop, output four repetitions:**
```bash
audioloop input.wav
```

**Custom output path:**
```bash
audioloop input.wav -o output.wav
```

**Repetition count:**
```bash
audioloop input.wav --count 8
```

**Target total output duration:**
```bash
audioloop input.wav --duration 2m
```

**Loop length hint (guides loop-point search):**
```bash
audioloop input.wav --loop-length 8s
```

**Bar-based loop length hint (requires detectable tempo, assumes 4/4):**
```bash
audioloop input.wav --loop-length 4bars
```

**Bar-based loop length hint with non-4/4 time signature:**
```bash
audioloop input.wav --loop-length 4bars --time-signature 3/4
```

**Output as FLAC:**
```bash
audioloop input.wav -o output.flac
```

**Output as OGG with explicit format flag:**
```bash
audioloop input.wav --format ogg -o output.ogg
```

**Manual loop region override:**
```bash
audioloop input.wav --start 30s --end 1:30
```

**Custom crossfade duration:**
```bash
audioloop input.wav --crossfade 200
```

**Analyze only — print metadata and candidates, write nothing:**
```bash
audioloop input.wav --analyze-only
```

**Batch — process all WAV files in a directory:**
```bash
audioloop ./tracks/ --batch
```

**Batch with output directory:**
```bash
audioloop ./tracks/ --batch -o ./looped/
```

**Overwrite an existing output file:**
```bash
audioloop input.wav --overwrite
```

**Limit accepted input file size to 512 MB:**
```bash
audioloop input.wav --max-file-size 512
```

**Verbose output:**
```bash
audioloop input.wav -v
```

**Quiet — suppress all non-error output:**
```bash
audioloop input.wav -q
```

## CLI Reference

| Option | Short | Default | Description |
|---|---|---|---|
| `INPUT` | | required | Path to a WAV, FLAC, or OGG file, or a directory when `--batch` is used |
| `--output PATH` | `-o` | `<input>_loop.wav` | Output file path. Extension determines format unless `--format` is given |
| `--format FMT` | | auto | Output format: `wav`, `flac`, or `ogg`. Overrides extension-based detection |
| `--loop-length HINT` | | none | Target loop length hint, e.g. `8s`, `2m`, `4bars`. Bar values require a detectable tempo |
| `--time-signature N/D` | | `4/4` | Time signature for bar-based loop length calculations, e.g. `3/4`, `6/8` |
| `--start OFFSET` | | none | Manual loop start offset, e.g. `1.5s`, `0:01` |
| `--end OFFSET` | | none | Manual loop end offset, e.g. `10s`, `0:10` |
| `--count N` | `-n` | `4` | Number of loop repetitions to render (maximum 10000). Mutually exclusive with `--duration` |
| `--duration TIME` | `-d` | none | Target total output duration, e.g. `30s`, `2m`. Mutually exclusive with `--count` |
| `--crossfade MS` | `-x` | content-type default | Crossfade duration in milliseconds (rhythmic: 20, ambient: 500, mixed: 250) |
| `--overwrite` | | off | Allow overwriting existing output files (default: refuse). In batch mode, existing outputs are skipped by default |
| `--max-file-size MB` | | `2048` | Maximum input file size in megabytes. Files exceeding this limit are rejected before loading |
| `--batch` | `-b` | off | Process all audio files (WAV, FLAC, OGG) found in the INPUT directory |
| `--analyze-only` | | off | Print metadata and loop candidates; do not write any output file |
| `--verbose` | `-v` | off | Increase verbosity. Repeat (`-vv`) for debug-level detail. Mutually exclusive with `--quiet` |
| `--quiet` | `-q` | off | Suppress all non-error output. Mutually exclusive with `--verbose` |
| `--version` | | | Print version and exit |
| `--help` | `-h` | | Show help and exit |

Time offsets accept the formats `Ns` (seconds), `N.Ns` (fractional seconds), `M:SS`, or `M:SS.S`.

## How It Works

1. **Content classification** — the input is classified as rhythmic, ambient, or mixed using onset-strength variance (primary), percussive energy ratio from harmonic-percussive source separation (secondary), and mean spectral flatness (tertiary).

2. **Loop point detection** — a spectral similarity search using Mel-frequency cepstral coefficients (MFCCs) finds the pair of positions where the audio transitions most smoothly. The search samples positions across the full track body (skipping intro/outro fade regions) and scores candidates using both MFCC similarity and RMS envelope consistency to avoid loops that pulse when repeated. For rhythmic and mixed content the search is constrained to beat-aligned positions.

3. **Zero-crossing alignment** — the detected start and end samples are nudged to the nearest zero-crossing on every channel to eliminate click artefacts at the boundary.

4. **Equal-power crossfade** — a raised-cosine equal-power envelope is applied as a circular crossfade at the loop head, blending the loop's end into its beginning. A tapered RMS correction reduces energy bumps caused by constructive interference between correlated signals. The crossfade duration defaults per content type (rhythmic: 20 ms, ambient: 500 ms, mixed: 250 ms) and can be overridden with `--crossfade`.

5. **Loop assembly** — the loop region is tiled for the requested number of repetitions (or until the target duration is met) and written at the source bit depth and sample rate. The output file is designed to loop seamlessly both internally and when placed end-to-end with copies of itself.

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success — output file written (or analysis printed) |
| `1` | Processing error — file could not be read, loop detection failed, or write error |
| `2` | Usage error — invalid option value or conflicting arguments |
| `3` | Batch partial failure — at least one file in a batch run failed; others may have succeeded |

## Development

Run the test suite:
```bash
pytest
```

Run the test suite with coverage:
```bash
pytest --cov=audioloop --cov-report=term-missing
```

Lint:
```bash
ruff check src/ tests/
```

Format:
```bash
ruff format src/ tests/
```

Type-check:
```bash
mypy src/
```

## Limitations

- Supported formats are WAV, FLAC, and OGG Vorbis. MP3 and other formats are not supported.
- Bar-based loop length hints (`--loop-length Nbars`) default to 4/4 time. Use `--time-signature N/D` for other time signatures. The denominator must be 2, 4, 8, or 16.
- Batch mode does not support `--start`, `--end`, `--duration`, or `--loop-length`; it applies auto-detection with the specified `--count` (default 4) to each file independently.
- No real-time preview; the output file must be opened in an external audio player.

## License

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
