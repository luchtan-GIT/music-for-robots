# Corrupted Variable — audio → matrix visualizer (Python)

This toolchain generates an MP4 by mapping **audio features** into a **coupled-cell matrix**. Each cell is a small dynamical system driven by a frequency band (mel bin) and coupled to its neighbors; patterns are not pre-drawn.

It was designed for the *Decompression Études* “Corrupted Variable”: not predictable, not chaotic — *pattern discovery*.

## Listening protocol (what the renderer “listens” for)
- **Frequency / intensity**: mel-spectrogram power drives per-cell excitation.
- **Regularity**: short-window onset autocorrelation controls how strongly cells entrain to neighbors.
- **Alterations**: mel-flux controls turbulence, gridline emphasis, and (optionally) resolution changes.
- **Onsets**: can trigger one-shot narrative events (e.g., neon burst).

## Run
From repo root:
```bash
python3 -m pip install numpy opencv-python librosa scipy tqdm

python3 toolchains/corrupted-variable/render_matrix.py \
  --audio /path/to/corrupted_variable.wav \
  --out corrupted_variable_matrix.mp4 \
  --size 720 --fps 30 \
  --grid 144 \
  --style cells \
  --rescale --rescale-steps 9 --rescale-min 24 --rescale-strength 1.15 \
  --narrative
```

Notes:
- `--grid` controls base resolution (cells-per-side). 96–144 is a good range at 720p.
- `--rescale` changes the *displayed* matrix resolution over time (true matrix resolution shift, not camera zoom).
- `--narrative` adds an arc (shape morphs + palette changes + neon burst).

## Output
- MP4 with original audio muxed via ffmpeg.
