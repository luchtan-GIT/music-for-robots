"""Compression Loom — listening protocol renderer.

Concept
-------
We treat a song as a process of **compression**: the listener learns to weave the signal
into a stable internal fabric. Novelty appears as snags; recurrence appears as a tighter,
more regular weave.

This renderer is intentionally simple:
- Warp threads (vertical) + weft threads (horizontal)
- Colors drift continuously with audio (centroid/novelty) and a slow narrative arc
- Recurrence increases persistence; novelty adds breaks

Usage
-----
```bash
python3 toolchains/listening-protocols/compression_loom.py --audio song.wav --out loom.mp4
```
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from tqdm import tqdm

try:
    # When run as a module (preferred): python -m toolchains.listening-protocols.compression_loom
    from ._audio_features import extract_tracks
    from ._mux import mux_audio
except ImportError:  # pragma: no cover
    # When run as a script: python toolchains/listening-protocols/compression_loom.py
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from _audio_features import extract_tracks
    from _mux import mux_audio


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """HSV (0..1) -> RGB (0..255)."""
    hsv = np.array([h * 179.0, s * 255.0, v * 255.0], np.uint8).reshape(1, 1, 3)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
    return rgb


def _to_bgr(rgb255: np.ndarray) -> tuple[int, int, int]:
    rgb255 = np.asarray(rgb255)
    return (int(rgb255[2]), int(rgb255[1]), int(rgb255[0]))


def render(
    audio: str,
    out: str,
    size: int = 720,
    fps: int = 30,
    seed: int = 22,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    tr = extract_tracks(audio, fps=fps, start=start, duration=duration)
    T = len(tr["rms"])

    rng = np.random.default_rng(seed)
    W = H = int(size)

    # We draw into a persistent ink layer and composite over a dark background.
    bg = np.array([8, 8, 10], np.uint8)
    ink = np.zeros((H, W, 3), np.uint8)

    n_threads = 90
    xs = np.linspace(0.05, 0.95, n_threads)
    phases = rng.uniform(0, 2 * math.pi, n_threads)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_silent = out.replace(".mp4", "_silent.mp4")
    vw = cv2.VideoWriter(out_silent, fourcc, fps, (W, H))

    # Global hue state gives continuity across frames.
    hue = 0.12

    for t in tqdm(range(T), desc="compression_loom"):
        rms = float(tr["rms"][t])
        nov = float(tr["novelty"][t])
        rec = float(tr["rec"][t])
        cen = float(tr["centroid"][t])

        # Narrative position across the whole piece (0..1).
        u = t / max(1, T - 1)

        # Arc: density highest in the mid-section.
        density = np.clip(0.55 + 0.75 * (0.5 - abs(u - 0.55)), 0.35, 0.95)
        settle = np.clip((u - 0.70) / 0.30, 0.0, 1.0)

        # Ink persistence: recurrence “remembers”, late settle “repairs”.
        ink = (ink.astype(np.float32) * (0.960 + 0.03 * rec + 0.01 * settle)).astype(np.uint8)

        img = np.zeros((H, W, 3), np.uint8)
        img[:] = bg

        # Audio-reactive palette drift:
        # - centroid pushes hue toward warmer colors
        # - novelty increases speed of drift
        hue = (0.990 * hue + 0.010 * (0.10 + 0.85 * cen) + 0.008 * nov) % 1.0

        sat = float(np.clip(0.60 + 0.35 * nov + 0.20 * (1.0 - rec), 0.35, 1.0))
        val = float(np.clip(0.50 + 0.45 * (0.6 + 0.4 * rec) + 0.15 * rms, 0.40, 1.0))

        base_col = _hsv_to_rgb(hue, sat, val)
        # Weft palette: rotate hue slightly for contrast.
        base_col2 = _hsv_to_rgb((hue + 0.22) % 1.0, sat * 0.9, min(1.0, val * 1.05))

        # Motion/tension parameters.
        warp = (0.3 + 1.6 * nov) * (1.0 - 0.35 * settle)
        tight = 0.4 + 2.1 * rec + 0.6 * density
        snag = max(0.0, nov - 0.66)

        # Fewer threads early, more mid, fewer late.
        stride = int(np.clip(7 - 4 * density, 2, 7))

        for k, x0 in enumerate(xs[::stride]):
            kk = k * stride
            x = int(x0 * W)

            # Thread “breathing”
            y = int((0.2 + 0.6 * (0.5 + 0.5 * math.sin(phases[kk] + 0.06 * t * tight + 3.0 * x0))) * H)
            y += int((rng.normal(0, 1.0) * warp * (0.6 + 1.0 * (1.0 - rec))))

            # Per-thread color variation: small shifts across x.
            jitter = kk / n_threads
            col = base_col * (0.65 + 0.55 * rms) + 40.0 * (jitter - 0.5)
            col = np.clip(col, 0, 255).astype(np.uint8)
            thickness = 1 + int(2 * rec + 1 * rms)

            # Warp (vertical)
            cv2.line(ink, (x, 0), (x, H), _to_bgr(col), thickness)

            # Weft (horizontal)
            if kk % max(1, int(5 - 3 * density)) == 0:
                y2 = int((y + (8 + 28 * nov) * math.sin(0.02 * t + x0 * 7 + 2.0 * nov)) % H)
                colw = np.clip(base_col2 * (0.55 + 0.55 * rms) + 35.0 * (0.5 - jitter), 0, 255).astype(np.uint8)
                cv2.line(ink, (0, y2), (W, y2), _to_bgr(colw), 1 + int(1 + 2 * density))

            # Snag events appear as white rings (local break in the weave).
            if snag > 0 and kk % 8 == 0:
                sx = int((x + rng.normal(0, 1.0) * 22) % W)
                sy = int((y + rng.normal(0, 1.0) * 22) % H)
                cv2.circle(ink, (sx, sy), int(6 + 38 * snag), (255, 255, 255), 1, cv2.LINE_AA)

        img = cv2.addWeighted(img, 1.0, ink, 0.96, 0)

        # Vignette to make the loom feel spatial.
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        vx = (xx - W * 0.5) / (W * 0.5)
        vy = (yy - H * 0.5) / (H * 0.5)
        vig = np.clip(1.0 - 0.18 * (vx * vx + vy * vy), 0.78, 1.0)
        img = np.clip(img.astype(np.float32) * vig[..., None], 0, 255).astype(np.uint8)

        vw.write(img)

    vw.release()

    mux_audio(out_silent, audio, out, start=start, duration=duration)


def main():
    ap = argparse.ArgumentParser(description="Compression Loom — listening protocol renderer")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--start", type=float, default=None, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="duration in seconds")
    args = ap.parse_args()

    render(
        args.audio,
        args.out,
        size=args.size,
        fps=args.fps,
        seed=args.seed,
        start=args.start,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
