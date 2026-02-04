# Listening Protocols — reusable audio→visual instruments (Python)

This toolchain packages three reusable **listening protocols** (renderers) that explore
**predictability vs novelty** in looping music:

1) **Compression Loom** — weaving as compression / repair.
2) **Recurrence Constellation** — a map of returning states.
3) **Mirror Residue** — prediction vs actual with a visible residual.

Each renderer:
- takes any audio file (`.wav`, `.mp3`, etc.)
- renders an MP4
- **muxes the original audio** back into the MP4 via ffmpeg

## Install
From repo root:
```bash
python3 -m pip install -r toolchains/listening-protocols/requirements.txt

# ffmpeg must be installed and available on PATH
ffmpeg -version
```

## Quick start
```bash
python3 toolchains/listening-protocols/compression_loom.py \
  --audio /path/to/song.wav \
  --out compression_loom.mp4 \
  --size 720 --fps 30

python3 toolchains/listening-protocols/recurrence_constellation.py \
  --audio /path/to/song.wav \
  --out recurrence_constellation.mp4 \
  --size 720 --fps 30

python3 toolchains/listening-protocols/mirror_residue.py \
  --audio /path/to/song.wav \
  --out mirror_residue.mp4 \
  --size 720 --fps 30
```

## Excerpts (optional)
All scripts support rendering an excerpt:
```bash
python3 toolchains/listening-protocols/compression_loom.py \
  --audio /path/to/song.wav \
  --out loom_excerpt.mp4 \
  --start 30 --duration 20
```

## Render all (wrapper)
```bash
python3 toolchains/listening-protocols/render_all.py \
  --audio /path/to/song.wav \
  --outdir out_protocols \
  --size 720 --fps 30

# Excerpt:
python3 toolchains/listening-protocols/render_all.py \
  --audio /path/to/song.wav \
  --outdir out_protocols \
  --start 30 --duration 20
```

## Listening protocol notes
### Shared features (what the protocols “listen” for)
We compute a small set of time-aligned features at the render FPS:
- **RMS** (loudness proxy)
- **Onset strength / flux**
- **Spectral centroid** (brightness)
- **Spectral bandwidth** (spread)
- **Entropy proxy** (noisiness / disorder)
- **Novelty** (feature-vector delta + flux)
- **Recurrence** (cosine similarity to rolling history embedding)

These are lightweight and robust, and they generalize well across songs.

## Modifying
All scripts are intentionally small and heavily commented. You should feel free to:
- swap palettes
- change which features drive which parameters
- add narrative events (e.g., chapter boundaries from novelty peaks)

## Outputs
The scripts write a temporary `*_silent.mp4` (video-only) and then mux audio into
`--out`. The silent file is deleted by default.

## Demo artifacts (included in this repo)
If you want a fast, zero-setup “what does this look like?” preview, see the bundled
15-second demos rendered from a synthetic loop:

- `toolchains/listening-protocols/demo/out/compression_loom.mp4`
- `toolchains/listening-protocols/demo/out/recurrence_constellation.mp4`
- `toolchains/listening-protocols/demo/out/mirror_residue.mp4`

Audio source:
- `toolchains/listening-protocols/demo/demo_loop.wav`

## Common problems
- **ffmpeg not found**
  - Symptom: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
  - Fix (macOS): `brew install ffmpeg`
  - Fix (Ubuntu/Debian): `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - Verify: `ffmpeg -version`

- **Audio won’t load / format issues**
  - Try converting to WAV:
    - `ffmpeg -y -i input.mp3 -ac 1 -ar 48000 output.wav`
  - Then render using `--audio output.wav`.

- **Slow renders**
  - Reduce resolution: `--size 540` (or 360 for quick iteration).
  - Reduce FPS: `--fps 24`.
  - Render an excerpt: `--start 30 --duration 20`.

- **No audio in output**
  - These scripts mux audio via ffmpeg at the end. If you see a silent MP4, check ffmpeg.
  - Verify with: `ffprobe -hide_banner -i output.mp4` (should show one video + one audio stream).

- **librosa / soundfile install errors**
  - Ensure you installed requirements: `pip install -r toolchains/listening-protocols/requirements.txt`
  - On some systems you may need system libs for `soundfile`.

- **Relative import errors**
  - These scripts are designed to run as plain scripts:
    - `python3 toolchains/listening-protocols/mirror_residue.py ...`
  - If you try to run them as a module inside a different package layout, keep the folder intact.
