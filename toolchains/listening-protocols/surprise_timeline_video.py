#!/usr/bin/env python3
"""Render a simple MP4 showing surprise (prediction error) over time.

This produces a self-contained video (curve + moving cursor) and muxes audio.
Designed as a quick artifact for sharing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from ._mux import mux_audio
except ImportError:  # pragma: no cover
    import os, sys

    sys.path.append(os.path.dirname(__file__))
    from _mux import mux_audio


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="probe output dir containing *_err_rmse_t.npy")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["mel", "emb"], default="mel")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    audio = str(Path(args.audio).expanduser().resolve())
    out = str(Path(args.out).expanduser().resolve())

    t = np.load(indir / "t_sec.npy").astype(np.float32)
    err = np.load(indir / ("mel_err_rmse_t.npy" if args.kind == "mel" else "emb_err_rmse_t.npy")).astype(np.float32)

    # robust y-range
    lo = 0.0
    hi = float(np.quantile(err[np.isfinite(err)], 0.995))
    hi = max(hi, 0.6)

    W = H = int(args.size)
    fps = int(args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_silent = out.replace(".mp4", "_silent.mp4")
    vw = cv2.VideoWriter(out_silent, fourcc, fps, (W, H))

    # precompute polyline in pixel coords
    pad_l, pad_r = 60, 30
    pad_t, pad_b = 50, 90
    x0, x1 = pad_l, W - pad_r
    y0, y1 = pad_t, H - pad_b

    # downsample for polyline: one point per frame is already fine
    xs = x0 + (t / max(t[-1], 1e-6)) * (x1 - x0)
    ys = y1 - np.clip((err - lo) / (hi - lo), 0.0, 1.0) * (y1 - y0)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)

    for i in range(len(t)):
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (12, 12, 18)

        # axes box
        cv2.rectangle(img, (x0, y0), (x1, y1), (70, 70, 80), 1)

        # curve
        cv2.polylines(img, [pts], False, (210, 210, 230), 2, cv2.LINE_AA)

        # cursor
        cx = int(pts[i, 0])
        cv2.line(img, (cx, y0), (cx, y1), (120, 200, 255), 1, cv2.LINE_AA)
        cy = int(pts[i, 1])
        cv2.circle(img, (cx, cy), 5, (120, 200, 255), -1, cv2.LINE_AA)

        e = float(err[i])
        n = _norm01(e, lo, hi)
        bar_w = int((x1 - x0) * n)
        cv2.rectangle(img, (x0, y1 + 28), (x0 + bar_w, y1 + 52), (60, 180, 255), -1)
        cv2.rectangle(img, (x0, y1 + 28), (x1, y1 + 52), (70, 70, 80), 1)

        cv2.putText(
            img,
            f"surprise ({args.kind})  {e:.3f}   t={t[i]:.2f}s",
            (x0, y1 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(img, "predictability probe", (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 1, cv2.LINE_AA)

        vw.write(img)

    vw.release()
    mux_audio(out_silent, audio, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
