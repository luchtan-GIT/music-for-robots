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
