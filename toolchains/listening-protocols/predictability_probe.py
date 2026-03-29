#!/usr/bin/env python3
"""Predictability probe (within-session learning) for audio.

Runs two online predictors:
1) mel_db next-frame prediction ("surface")
2) embedding next-frame prediction ("state")

Outputs a markdown report with:
- overall error stats
- per-loop learning curve (if loops>1)
- top surprise moments (single pass)

This is designed to answer: "how many listens until a bounded engine predicts what's next?"
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from _audio_features import extract_tracks
from _predictors import online_learning_curve


@dataclass
class ProbeConfig:
    fps: int = 30
    sr: int = 48_000
    k: int = 4
    # conservative learning rates for stability
    lr_emb: float = 0.006
    l2_emb: float = 3e-4
    lr_mel: float = 0.003
    l2_mel: float = 5e-4


def build_looped_wav(
    *,
    audio_path: Path,
    out_path: Path,
    loops: int,
    start: float | None,
    duration: float | None,
    sr: int,
) -> tuple[float, float]:
    """Write wav containing selected segment repeated N times.

    Returns (segment_duration_sec, total_duration_sec).
    """

    y, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if file_sr != sr:
        tmp = out_path.with_suffix(".tmp_resample.wav")
        cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(sr), str(tmp)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        y, file_sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        tmp.unlink(missing_ok=True)

    n0 = int((start or 0.0) * sr)
    if duration is None:
        seg = y[n0:]
    else:
        n1 = n0 + int(duration * sr)
        seg = y[n0:n1]

    seg_dur = len(seg) / sr
    if seg_dur <= 0.05:
        raise ValueError("Selected segment is too short (or empty).")

    tiled = np.tile(seg, int(loops)).astype("float32")
    sf.write(str(out_path), tiled, sr)
    return seg_dur, len(tiled) / sr


def top_surprise_times(err_t: np.ndarray, fps: int, n: int = 8) -> list[float]:
    """Return top-N surprise timestamps (seconds), ignoring first second."""
    err_t = np.asarray(err_t, np.float32)
    start = int(1.0 * fps)
    if err_t.size <= start + 5:
        return []
    idx = np.argsort(err_t[start:])[-n:]
    idx = (idx + start).astype(int)
    # unique-ish: keep separated by >=0.5s
    keep = []
    for i in sorted(idx, key=lambda j: float(err_t[j]), reverse=True):
        t = float(i) / float(fps)
        if all(abs(t - t2) >= 0.5 for t2 in keep):
            keep.append(t)
        if len(keep) >= n:
            break
    return keep


def per_loop_means(x: np.ndarray, frames_per_loop: int, loops: int) -> list[float]:
    vals = []
    for i in range(loops):
        a = i * frames_per_loop
        b = min(len(x), (i + 1) * frames_per_loop)
        if b - a < 5:
            break
        vals.append(float(np.mean(x[a:b])))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="Predictability probe (online learning)")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sr", type=int, default=48_000)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cfg = ProbeConfig(fps=int(args.fps), sr=int(args.sr), k=int(args.k))

    audio_path = Path(args.audio).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    loops = int(args.loops)
    if loops < 1 or loops > 20:
        raise ValueError("--loops must be in [1, 20] for this probe")

    start = float(args.start) if args.start is not None else 0.0
    duration = args.duration

    # Build looped audio (or single pass copy)
    probe_wav = outdir / f"probe_L{loops}_start{start:.2f}_dur{('full' if duration is None else f'{duration:.2f}')}.wav"
    seg_dur, total_dur = build_looped_wav(
        audio_path=audio_path,
        out_path=probe_wav,
        loops=loops,
        start=start,
        duration=duration,
        sr=cfg.sr,
    )

    tracks = extract_tracks(
        str(probe_wav),
        fps=cfg.fps,
        sr=cfg.sr,
        start=None,
        duration=None,
        return_embedding=True,
        return_mel=True,
    )

    emb = tracks["emb"].astype(np.float32)
    mel = tracks["mel_db"].astype(np.float32)

    emb_err = online_learning_curve(emb, k=cfg.k, lr=cfg.lr_emb, l2=cfg.l2_emb, seed=1)["err_rmse_t"]
    mel_err = online_learning_curve(mel, k=cfg.k, lr=cfg.lr_mel, l2=cfg.l2_mel, seed=2)["err_rmse_t"]

    frames_per_loop = int(round(seg_dur * cfg.fps))

    md = []
    md.append("# Predictability Probe — within-session learning\n")
    md.append(f"Audio: `{audio_path.name}`\n")
    md.append(f"Segment: start={start:.2f}s, duration={'FULL' if duration is None else f'{duration:.2f}s'}\n")
    md.append(f"Loops: {loops} (total ~{total_dur/60:.2f} min)\n")

    def summary_block(name: str, err: np.ndarray) -> None:
        md.append(f"## {name}\n")
        md.append(f"- mean err: **{float(np.mean(err)):.3f}**")
        md.append(f"- p90 err: **{float(np.quantile(err, 0.90)):.3f}**")
        md.append(f"- p99 err: **{float(np.quantile(err, 0.99)):.3f}**")

        if loops > 1 and frames_per_loop > 0:
            means = per_loop_means(err, frames_per_loop, loops)
            md.append(f"- per-loop mean err: {[round(v, 3) for v in means]}")
            if len(means) >= 2:
                md.append(f"  - improvement L1→L{len(means)}: {means[0]:.3f} → {means[-1]:.3f}")

        if loops == 1:
            peaks = top_surprise_times(err, cfg.fps, n=10)
            if peaks:
                md.append(f"- top surprise times (s): {[round(t, 2) for t in peaks]}")
        md.append("")

    summary_block("Embedding next-step (state)", emb_err)
    summary_block("Mel-dB next-step (surface)", mel_err)

    report = outdir / "predictability_report.md"
    report.write_text("\n".join(md), encoding="utf-8")

    # Save time series for plotting/animation.
    t_sec = (np.arange(len(emb_err), dtype=np.float32) / float(cfg.fps)).astype(np.float32)
    np.save(str(outdir / "t_sec.npy"), t_sec)
    np.save(str(outdir / "emb_err_rmse_t.npy"), emb_err.astype(np.float32))
    np.save(str(outdir / "mel_err_rmse_t.npy"), mel_err.astype(np.float32))

    meta = {
        "audio": str(audio_path),
        "probe_wav": str(probe_wav),
        "report": str(report),
        "fps": cfg.fps,
        "sr": cfg.sr,
        "k": cfg.k,
        "loops": loops,
        "start": start,
        "duration": duration,
        "seg_dur": seg_dur,
        "total_dur": total_dur,
    }
    (outdir / "predictability_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
