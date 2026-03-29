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
    loop_sec: float | None = None,
    pred_driver: str = "none",  # none|mel|emb
) -> None:
    tr = extract_tracks(audio, fps=fps, start=start, duration=duration, return_embedding=True, return_mel=(pred_driver == "mel"))
    T = len(tr["rms"])

    # Optional: compute a real next-step predictor error and use it to drive the residue flare.
    pred_err01 = None
    if pred_driver in ("mel", "emb"):
        try:
            from _predictors import online_learning_curve

            X = tr["mel_db"] if pred_driver == "mel" else tr["emb"]
            # Conservative settings for stability.
            res = online_learning_curve(X.astype(np.float32), k=4, lr=(0.003 if pred_driver == "mel" else 0.006), l2=(5e-4 if pred_driver == "mel" else 3e-4), seed=7)
            e = res["err_rmse_t"].astype(np.float32)
            # Robust normalize to 0..1 for visual drive
            hi = float(np.quantile(e[np.isfinite(e)], 0.99)) if np.any(np.isfinite(e)) else 1.0
            hi = max(hi, 1e-3)
            pred_err01 = np.clip(e / hi, 0.0, 1.0).astype(np.float32)
        except Exception:
            pred_err01 = None

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
        "attn": 0.25,
    }

    hue = 0.55

    metrics = [
        ("cen", "centroid"),
        ("bw", "bandwidth"),
        ("flux", "flux"),
        ("conf", "rec"),
        ("attn", "attn"),
    ]

    for t in tqdm(range(T), desc="mirror_residue"):
        nov = float(tr["novelty"][t])
        nov_fast = float(tr.get("novelty_fast", tr["novelty"])[t])
        hab = float(tr.get("habituation", np.zeros(T, np.float32))[t])

        # Confidence: if we know loop period, compare against previous loop; otherwise use local recurrence.
        conf = float(tr["rec"][t])
        if loop_sec is not None and tr.get("emb", None) is not None:
            loop_frames = int(round(loop_sec * fps))
            if loop_frames > 0 and t - loop_frames >= 0:
                from _audio_features import cosine_sim

                conf = float(np.clip(cosine_sim(tr["emb"][t], tr["emb"][t - loop_frames]), 0.0, 1.0))

        # Attention proxy: fast novelty gated by (1 - habituation)
        attn = float(np.clip(nov_fast * (1.0 - hab) * 1.6, 0.0, 1.0))

        vals = {
            "centroid": float(tr["centroid"][t]),
            "bandwidth": float(tr["bandwidth"][t]),
            "flux": float(tr["flux"][t]),
            "rec": conf,
            "nov": nov,
            "attn": attn,
        }
        rec = vals["rec"]

        # EMA rate: recurrence makes the predictor more confident (learn faster).
        alpha = 0.035 + 0.11 * rec
        for k in pred:
            pred[k] = (1 - alpha) * pred[k] + alpha * vals[k]

        # Residue: weighted absolute error.
        err = (
            abs(vals["centroid"] - pred["centroid"]) * 0.24
            + abs(vals["bandwidth"] - pred["bandwidth"]) * 0.16
            + abs(vals["flux"] - pred["flux"]) * 0.18
            + abs(vals["rec"] - pred["rec"]) * 0.16
            + abs(vals["attn"] - pred["attn"]) * 0.18
            + abs(vals["nov"] - pred["nov"]) * 0.08
        )

        # Seam flare: if loop period is known, spike error near loop boundaries when join mismatch is high.
        seam = 0.0
        if loop_sec is not None and tr.get("emb", None) is not None:
            loop_frames = int(round(loop_sec * fps))
            if loop_frames > 8:
                w = int(0.30 * fps)  # ±0.30s window
                m = t % loop_frames
                if m < w or m > (loop_frames - w):
                    # measure mismatch between this moment and previous loop
                    if t - loop_frames >= 0:
                        from _audio_features import cosine_sim

                        sim = float(np.clip(cosine_sim(tr["emb"][t], tr["emb"][t - loop_frames]), 0.0, 1.0))
                        seam = float(np.clip(1.0 - sim, 0.0, 1.0))

        err = float(np.clip(err * (1.05 + 1.2 * vals["nov"] + 1.2 * vals["attn"] + 1.6 * seam), 0, 1))

        # If enabled, let true predictor surprise modulate the flare.
        if pred_err01 is not None:
            pe = float(pred_err01[t])
            # Blend: keep protocol residue (err) but let surprise take the wheel.
            err = float(np.clip(0.35 * err + 0.85 * pe, 0.0, 1.0))

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
        cv2.putText(
            img,
            (f"residue {err:.2f} | conf {rec:.2f} | attn {vals['attn']:.2f}" + (f" | pred {float(pred_err01[t]):.2f}" if pred_err01 is not None else "")),
            (mid - 220, cy + rad + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

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
    ap.add_argument("--loop-sec", type=float, default=None, help="Loop period in seconds (for loop-aware seam/recurrence probes)")
    ap.add_argument(
        "--pred-driver",
        type=str,
        default="none",
        choices=["none", "mel", "emb"],
        help="Drive residue flare by online predictor error (mel or embedding)",
    )
    args = ap.parse_args()

    render(
        args.audio,
        args.out,
        size=args.size,
        fps=args.fps,
        seed=args.seed,
        start=args.start,
        duration=args.duration,
        loop_sec=args.loop_sec,
        pred_driver=args.pred_driver,
    )


if __name__ == "__main__":
    main()
