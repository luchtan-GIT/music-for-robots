"""Recurrence Constellation — listening protocol renderer.

Concept
-------
Each frame becomes a point in a 2D state-space. We draw a constellation of recent points
and connect them when the current state resembles past states.

- **Recurrence** brightens and tightens connections.
- **Novelty / turbulence** increases jitter and speeds up palette drift.
- **Centroid** nudges the hue (cool↔warm).

This is intentionally not a "scientific" embedding; it's an instrument for perceiving
return, drift, and surprise.

Usage
-----
```bash
python3 toolchains/listening-protocols/recurrence_constellation.py --audio song.wav --out constellation.mp4
```
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ._audio_features import extract_tracks
    from ._mux import mux_audio
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from _audio_features import extract_tracks
    from _mux import mux_audio


def _rgb_from_hsv(h: float, s: float, v: float) -> np.ndarray:
    hsv = np.array([h * 179.0, s * 255.0, v * 255.0], np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def render(
    audio: str,
    out: str,
    size: int = 720,
    fps: int = 30,
    seed: int = 23,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    tr = extract_tracks(audio, fps=fps, start=start, duration=duration)
    T = len(tr["rms"])

    rng = np.random.default_rng(seed)
    W = H = int(size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_silent = out.replace(".mp4", "_silent.mp4")
    vw = cv2.VideoWriter(out_silent, fourcc, fps, (W, H))

    # Each point: (x, y, rec, nov, turb)
    pts: list[tuple[int, int, float, float, float]] = []

    hue = 0.58

    for t in tqdm(range(T), desc="recurrence_constellation"):
        rec = float(tr["rec"][t])
        nov = float(tr["novelty"][t])
        cen = float(tr["centroid"][t])
        bw = float(tr["bandwidth"][t])
        ent = float(tr["entropy"][t])
        flux = float(tr["flux"][t])

        # Turbulence proxy (same idea as in loop_probes):
        tb = float(np.clip(0.55 * ent + 0.45 * flux, 0.0, 1.0))

        # Hue drift: centroid pushes warmth; novelty accelerates drift.
        hue = (0.985 * hue + 0.015 * (0.10 + 0.85 * cen) + 0.010 * nov) % 1.0
        sat = float(np.clip(0.55 + 0.35 * tb + 0.25 * nov, 0.35, 1.0))
        val = float(np.clip(0.35 + 0.55 * rec + 0.25 * (1.0 - tb), 0.25, 1.0))
        rgb = _rgb_from_hsv(hue, sat, val)

        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (6, 8, 12)

        # A simple state-space mapping:
        # - x: brightness/centroid
        # - y: spread/bandwidth
        # Jitter rises with turbulence.
        x = int((0.12 + 0.76 * cen) * W + rng.normal(0, 1) * (5 + 10 * tb))
        y = int((0.12 + 0.76 * bw) * H + rng.normal(0, 1) * (5 + 10 * tb))

        pts.append((x, y, rec, nov, tb))
        if len(pts) > 260:
            pts.pop(0)

        # Connect to recent points.
        # Recurrence brightens edges; distance attenuates.
        for j in range(max(0, len(pts) - 90), len(pts) - 1, 3):
            x2, y2, r2, n2, tb2 = pts[j]
            d = math.hypot(x - x2, y - y2) / (0.9 * W)
            w = max(0.0, (0.7 * rec + 0.3 * r2) - 0.30) * (1.0 - d)
            if w <= 0:
                continue

            col = (
                int(np.clip(30 + 0.9 * rgb[0] * w, 0, 255)),
                int(np.clip(50 + 0.9 * rgb[1] * w, 0, 255)),
                int(np.clip(60 + 0.9 * rgb[2] * w, 0, 255)),
            )
            cv2.line(img, (x, y), (x2, y2), col, 1, cv2.LINE_AA)

        # Draw points with a mild glow.
        for (px, py, r0, n0, tb0) in pts[::2]:
            rad = int(1 + 4 * r0 + 4 * max(0.0, n0 - 0.60))
            glow = 0.35 + 0.65 * r0
            col = (
                int(np.clip(40 + rgb[0] * glow, 0, 255)),
                int(np.clip(40 + rgb[1] * glow, 0, 255)),
                int(np.clip(50 + rgb[2] * (0.6 + 0.4 * (1.0 - tb0)), 0, 255)),
            )
            cv2.circle(img, (px, py), rad, col, -1, cv2.LINE_AA)

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img = cv2.addWeighted(img, 0.85, blur, 0.35, 0)

        vw.write(img)

    vw.release()
    mux_audio(out_silent, audio, out, start=start, duration=duration)


def main():
    ap = argparse.ArgumentParser(description="Recurrence Constellation — listening protocol renderer")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=23)
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
