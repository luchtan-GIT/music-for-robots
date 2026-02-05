#!/usr/bin/env python3
"""Plot predictability probe outputs.

Generates:
1) Surprise timeline plots (1-pass): err(t) curves + highlighted peaks.
2) Learning curve plots (multi-loop): per-loop mean error.

Expected inputs: directories produced by predictability_probe.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_meta(d: Path) -> dict:
    p = d / "predictability_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _safe(arr):
    arr = np.asarray(arr, np.float32)
    arr = arr[np.isfinite(arr)]
    return arr


def timeline_png(indir: Path, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.load(indir / "t_sec.npy").astype(np.float32)
    emb = np.load(indir / "emb_err_rmse_t.npy").astype(np.float32)
    mel = np.load(indir / "mel_err_rmse_t.npy").astype(np.float32)

    # robust scaling just for y-lims
    y_max = float(np.quantile(_safe(np.concatenate([emb, mel])), 0.995))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, emb, lw=1.2, label="embedding err")
    ax.plot(t, mel, lw=1.2, label="mel err")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMSE (std-space)")
    ax.set_ylim(0, max(0.5, y_max))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def learning_curve_png(indir_L5: Path, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = _load_meta(indir_L5)
    loops = int(meta["loops"])
    fps = int(meta["fps"])
    seg_dur = float(meta["seg_dur"])
    frames_per_loop = int(round(seg_dur * fps))

    emb = np.load(indir_L5 / "emb_err_rmse_t.npy").astype(np.float32)
    mel = np.load(indir_L5 / "mel_err_rmse_t.npy").astype(np.float32)

    def per_loop(err):
        vals = []
        for i in range(loops):
            a = i * frames_per_loop
            b = min(len(err), (i + 1) * frames_per_loop)
            vals.append(float(np.mean(err[a:b])))
        return np.array(vals, np.float32)

    emb_m = per_loop(emb)
    mel_m = per_loop(mel)
    xs = np.arange(1, loops + 1)

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, emb_m, marker="o", label="embedding")
    ax.plot(xs, mel_m, marker="o", label="mel")
    ax.set_title(title)
    ax.set_xlabel("loop (full listens)")
    ax.set_ylabel("per-loop mean RMSE (std-space)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="probe output dir (e.g. .../L1_full)")
    ap.add_argument("--indir_L5", default=None, help="L5 output dir for learning curve")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(indir)
    name = Path(meta["audio"]).name
    title = f"{name} — predictability surprise timeline" + (f" ({args.label})" if args.label else "")
    timeline_png(indir, outdir / f"{name}_timeline.png", title)

    if args.indir_L5:
        indir_L5 = Path(args.indir_L5).expanduser().resolve()
        meta5 = _load_meta(indir_L5)
        name5 = Path(meta5["audio"]).name
        title2 = f"{name5} — within-session learning curve (L{meta5['loops']})" + (f" ({args.label})" if args.label else "")
        learning_curve_png(indir_L5, outdir / f"{name5}_learning_curve.png", title2)

    print(f"Wrote plots to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
