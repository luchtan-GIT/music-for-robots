"""Mirror Residue — listening protocol renderer.

Concept
-------
We maintain a tiny predictor (EMA) of a handful of scalar features and compare it
against the actual per-frame features.

This produces a simple but useful visualization:
- Left meter: **predict**
- Right meter: **actual**
- Bottom: a **residue flare** whose radius grows with error

This is explicitly an *analysis-friendly* visualization.

Usage
-----
```bash
python3 toolchains/listening-protocols/mirror_residue.py --audio song.wav --out residue.mp4
```
"""

from __future__ import annotations

import argparse

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
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.uint8)


def render(
    audio: str,
    out: str,
    size: int = 720,
    fps: int = 30,
    seed: int = 25,
    start: float | None = None,
    duration: float | None = None,
) -> None:
    tr = extract_tracks(audio, fps=fps, start=start, duration=duration)
    T = len(tr["rms"])

    W = H = int(size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_silent = out.replace(".mp4", "_silent.mp4")
    vw = cv2.VideoWriter(out_silent, fourcc, fps, (W, H))

    # Predictor state: EMA for multiple metrics.
    pred = {
        "centroid": 0.5,
        "bandwidth": 0.5,
        "flux": 0.5,
        "rec": 0.5,
        "nov": 0.5,
    }

    hue = 0.55

    metrics = [
        ("cen", "centroid"),
        ("bw", "bandwidth"),
        ("flux", "flux"),
        ("rec", "rec"),
        ("nov", "novelty"),
    ]

    for t in tqdm(range(T), desc="mirror_residue"):
        vals = {
            "centroid": float(tr["centroid"][t]),
            "bandwidth": float(tr["bandwidth"][t]),
            "flux": float(tr["flux"][t]),
            "rec": float(tr["rec"][t]),
            "nov": float(tr["novelty"][t]),
        }
        rec = vals["rec"]
        nov = vals["nov"]

        # EMA rate: recurrence makes the predictor more confident (learn faster).
        alpha = 0.035 + 0.11 * rec
        for k in pred:
            pred[k] = (1 - alpha) * pred[k] + alpha * vals[k]

        # Residue: weighted absolute error.
        err = (
            abs(vals["centroid"] - pred["centroid"]) * 0.26
            + abs(vals["bandwidth"] - pred["bandwidth"]) * 0.18
            + abs(vals["flux"] - pred["flux"]) * 0.22
            + abs(vals["rec"] - pred["rec"]) * 0.18
            + abs(vals["nov"] - pred["nov"]) * 0.16
        )
        err = float(np.clip(err * (1.1 + 1.7 * nov), 0, 1))

        # Accent palette drift.
        hue = (0.99 * hue + 0.01 * (0.15 + 0.8 * vals["centroid"]) + 0.008 * nov) % 1.0
        sat = float(np.clip(0.55 + 0.20 * nov, 0.35, 1.0))
        valc = float(np.clip(0.55 + 0.35 * rec, 0.35, 1.0))
        rgb = _rgb_from_hsv(hue, sat, valc)
        bar_col = (int(rgb[2]), int(rgb[1]), int(rgb[0]))  # RGB->BGR

        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (10, 10, 14)

        cx1 = int(W * 0.25)
        cx2 = int(W * 0.75)

        def draw_bars(cx: int, vec: dict[str, float], label: str) -> None:
            y0 = int(H * 0.22)
            base_h = int(H * 0.48)
            spacing = 46
            xstart = cx - (spacing * (len(metrics) - 1)) // 2

            for j, (abbr, key) in enumerate(metrics):
                x0 = xstart + j * spacing
                v = float(np.clip(vec["nov"] if key == "novelty" else vec[key], 0, 1))
                h = int(v * base_h)
                cv2.rectangle(img, (x0 - 14, y0 + base_h - h), (x0 + 14, y0 + base_h), bar_col, -1)
                cv2.putText(img, abbr, (x0 - 16, y0 + base_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)

            cv2.putText(img, label, (cx - 70, y0 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 1, cv2.LINE_AA)

        draw_bars(cx1, pred, "predict")
        draw_bars(cx2, vals, "actual")

        # Residue flare
        mid = int(W * 0.5)
        rad = int(18 + err * 285)
        col = (int(60 + 195 * err), int(55 + 170 * err), int(40 + 210 * err))
        cy = int(H * 0.78)
        cv2.circle(img, (mid, cy), rad, col, 3, cv2.LINE_AA)
        cv2.circle(img, (mid, cy), int(rad * 0.55), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"residue {err:.2f}", (mid - 80, cy + rad + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        vw.write(img)

    vw.release()
    mux_audio(out_silent, audio, out, start=start, duration=duration)


def main():
    ap = argparse.ArgumentParser(description="Mirror Residue — listening protocol renderer")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=25)
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
