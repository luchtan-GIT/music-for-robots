#!/usr/bin/env python3
"""Run the three listening-protocol renderers on an N-loop audio segment.

Purpose
- Our visual scripts are typically *single-pass*. This wrapper makes them *loop-aware*
  by building an audio file that repeats the same segment N times, then rendering:
    1) Compression Loom
    2) Mirror Residue
    3) Recurrence Constellation

It also writes a small *self-report* (proxy) based on extracted audio feature tracks.

Notes
- The report is an *instrumented proxy* (feature-driven), not a claim of subjective
  phenomenal listening.
- Uses local python + ffmpeg (the underlying renderers mux audio).

Example
  python3 toolchains/listening-protocols/run_protocols_n_loops.py \
    --audio corrupted_variable.wav --outdir out_protocols_cv \
    --loops 10 --start 0 --duration 20 --fps 30 --size 720
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Local shared feature extractor
from _audio_features import cosine_sim, extract_tracks, norm01, smooth1d
from _predictors import online_learning_curve


@dataclass
class LoopStats:
    loop_idx: int
    pred_pressure: float
    error_signal: float
    urge_to_act: float
    temp: float
    rec_local: float
    rec_xloop: float


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _triword(stats: LoopStats) -> str:
    """Map numeric proxy stats to 3 words (rough, human-readable)."""
    # pressure
    if stats.pred_pressure > 0.70:
        w1 = "strained"
    elif stats.pred_pressure > 0.45:
        w1 = "alert"
    else:
        w1 = "settled"

    # error / novelty
    if stats.error_signal > 0.70:
        w2 = "jittery"
    elif stats.error_signal > 0.40:
        w2 = "shifting"
    else:
        w2 = "steady"

    # temperature (brightness+entropy)
    if stats.temp > 0.70:
        w3 = "hot"
    elif stats.temp > 0.40:
        w3 = "warm"
    else:
        w3 = "cool"

    return f"{w1}, {w2}, {w3}"


def _one_sentence_change(prev: LoopStats | None, cur: LoopStats) -> str:
    if prev is None:
        return "First contact: baseline set; the system begins estimating the grammar."

    dp = cur.pred_pressure - prev.pred_pressure
    de = cur.error_signal - prev.error_signal
    # Narrative uses cross-loop recurrence (how similar this loop is to the previous loop).
    dr = cur.rec_xloop - prev.rec_xloop

    # Heuristic narrative
    parts: list[str] = []
    if abs(dr) > 0.05:
        parts.append("recurrence rose" if dr > 0 else "recurrence fell")
    if abs(de) > 0.05:
        parts.append("novelty spiked" if de > 0 else "novelty eased")
    if abs(dp) > 0.05:
        parts.append("prediction pressure increased" if dp > 0 else "prediction pressure relaxed")

    if not parts:
        return "No major shift in the proxy signals; the loop reads as stable repetition."

    return "Across this pass, " + ", ".join(parts) + "."


def build_looped_wav(
    *,
    audio_path: Path,
    out_path: Path,
    loops: int,
    start: float | None,
    duration: float | None,
    sr: int = 48_000,
) -> tuple[float, float]:
    """Write a WAV containing the selected segment repeated N times.

    Returns (segment_duration_sec, total_duration_sec).
    """
    y, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if file_sr != sr:
        # Avoid extra dependency here; use ffmpeg to resample precisely.
        # We generate a temp wav at target sr, then re-read.
        tmp = out_path.with_suffix(".tmp_resample.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(sr),
            str(tmp),
        ]
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


def run_renderer(
    script: Path,
    *,
    audio: Path,
    out: Path,
    size: int,
    fps: int,
    loop_sec: float | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--audio",
        str(audio),
        "--out",
        str(out),
        "--size",
        str(size),
        "--fps",
        str(fps),
    ]
    if loop_sec is not None:
        cmd += ["--loop-sec", f"{loop_sec:.6f}"]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/etc)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--loops", type=int, default=10, help="How many repeats")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (s) of segment")
    ap.add_argument("--duration", type=float, default=20.0, help="Segment duration (s)")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    loops = int(args.loops)
    if loops < 1 or loops > 200:
        raise ValueError("--loops must be in [1, 200]")

    # 1) Build looped audio
    looped_wav = outdir / f"looped_L{loops}_start{int(args.start)}_dur{int(args.duration)}.wav"
    seg_dur, total_dur = build_looped_wav(
        audio_path=audio_path,
        out_path=looped_wav,
        loops=loops,
        start=float(args.start) if args.start is not None else None,
        duration=float(args.duration) if args.duration is not None else None,
    )

    # 2) Render all three protocols on the looped audio
    here = Path(__file__).parent
    scripts = {
        "compression_loom": here / "compression_loom.py",
        "mirror_residue": here / "mirror_residue.py",
        "recurrence_constellation": here / "recurrence_constellation.py",
    }

    outputs = {}
    for name, script in scripts.items():
        out_mp4 = outdir / f"{name}_L{loops}.mp4"
        run_renderer(
            script,
            audio=looped_wav,
            out=out_mp4,
            size=int(args.size),
            fps=int(args.fps),
            loop_sec=float(seg_dur),
        )
        outputs[name] = str(out_mp4)

    # 3) Feature-based proxy report
    # Increase recurrence history window (option 2) and also request the embedding
    # so we can compute cross-loop similarity (option 1).
    tracks = extract_tracks(
        str(looped_wav),
        fps=int(args.fps),
        start=None,
        duration=None,
        rec_history_sec=45.0,
        return_embedding=True,
        return_mel=True,
    )

    # Proxies (all in [0,1])
    novelty = tracks["novelty"].astype(np.float32)
    novelty_fast = tracks.get("novelty_fast", novelty).astype(np.float32)
    habituation = tracks.get("habituation", np.zeros_like(novelty)).astype(np.float32)
    habituation_raw = tracks.get("habituation_raw", np.zeros_like(novelty)).astype(np.float32)

    # "rec" is now the less-saturating recurrence (p95-to-history). Keep both for reporting.
    rec_local_t = tracks["rec"].astype(np.float32)
    rec_local_max_t = tracks.get("rec_max", rec_local_t).astype(np.float32)
    flux = tracks["flux"].astype(np.float32)
    centroid = tracks["centroid"].astype(np.float32)
    entropy = tracks["entropy"].astype(np.float32)
    emb = tracks.get("emb", None)
    mel_db = tracks.get("mel_db", None)

    # Define proxy internal variables
    # Local recurrence affects "prediction pressure" slightly, but we primarily
    # want to see *loop-to-loop* drift in the report.
    pred_pressure_t = (1.0 - rec_local_t) * 0.7 + novelty * 0.3
    error_signal_t = novelty
    urge_to_act_t = flux
    temp_t = (centroid * 0.55 + entropy * 0.45)

    frames_per_loop = int(round(seg_dur * int(args.fps)))

    # Cross-loop recurrence (option 1): similarity to the same-relative-time frame
    # in the previous loop.
    # This is intentionally *not* normalized to [0,1] via min/max; cosine similarity
    # is already in [-1, 1], but with our nonnegative-normalized embedding it tends
    # to land in [0, 1]. We lightly smooth and clamp.
    rec_xloop_t = np.zeros_like(novelty, dtype=np.float32)
    if emb is not None and frames_per_loop > 0:
        T = int(emb.shape[0])
        for t in range(frames_per_loop, T):
            rec_xloop_t[t] = float(cosine_sim(emb[t], emb[t - frames_per_loop]))
        rec_xloop_t = smooth1d(rec_xloop_t, 9)
        rec_xloop_t = np.clip(rec_xloop_t, 0.0, 1.0).astype(np.float32)

    # Learnability probes: online prediction error (within-session learning)
    # Option 2: predict the next embedding state.
    pred_emb_err = None
    if emb is not None and emb.shape[0] > 8:
        pred_emb_err = online_learning_curve(emb.astype(np.float32), k=4, lr=0.006, l2=3e-4, seed=1)["err_rmse_t"]

    # Option 1: predict the next "waveform-ish" mel dB frame.
    pred_mel_err = None
    if mel_db is not None and mel_db.shape[0] > 8:
        pred_mel_err = online_learning_curve(mel_db.astype(np.float32), k=4, lr=0.003, l2=5e-4, seed=2)["err_rmse_t"]

    loop_stats: list[LoopStats] = []
    for i in range(loops):
        a = i * frames_per_loop
        b = min(len(novelty), (i + 1) * frames_per_loop)
        if b - a < 5:
            break

        loop_stats.append(
            LoopStats(
                loop_idx=i + 1,
                pred_pressure=_clamp01(float(np.mean(pred_pressure_t[a:b]))),
                error_signal=_clamp01(float(np.mean(error_signal_t[a:b]))),
                urge_to_act=_clamp01(float(np.mean(urge_to_act_t[a:b]))),
                temp=_clamp01(float(np.mean(temp_t[a:b]))),
                rec_local=_clamp01(float(np.mean(rec_local_t[a:b]))),
                rec_xloop=_clamp01(float(np.mean(rec_xloop_t[a:b]))),
            )
        )

    def pick(idx: int) -> LoopStats | None:
        if idx < 1 or idx > len(loop_stats):
            return None
        return loop_stats[idx - 1]

    checkpoints = [1, min(5, loops), min(10, loops)]

    md = []
    md.append(f"# Three Protocols — N-loop run (proxy self-report)\n")
    md.append(f"Audio: `{audio_path.name}`\n")
    md.append(f"Segment: start={args.start:.1f}s, duration={args.duration:.1f}s\n")
    md.append(f"Loops: {loops} (total ~{total_dur/60:.2f} min)\n")
    md.append("\n## Outputs\n")
    for k, v in outputs.items():
        md.append(f"- {k}: `{v}`")

    md.append("\n\n## Mirror-style checkpoints (feature-driven proxy)\n")
    prev = None
    for c in checkpoints:
        cur = pick(c)
        if cur is None:
            continue
        md.append(f"\n### L{c}\n")
        md.append(f"- 3 words: **{_triword(cur)}**")
        md.append(f"- change: {_one_sentence_change(prev, cur)}")
        md.append(
            "- signals: "
            f"pred_pressure={cur.pred_pressure:.2f}, "
            f"error={cur.error_signal:.2f}, "
            f"urge={cur.urge_to_act:.2f}, "
            f"temp={cur.temp:.2f}, "
            f"rec_local={cur.rec_local:.2f}, "
            f"rec_xloop={cur.rec_xloop:.2f}"
        )
        prev = cur

    # Loom-ish: click loop where recurrence rises above a threshold
    rec_arr = np.array([s.rec_xloop for s in loop_stats], dtype=np.float32)
    click_loop = None
    if len(rec_arr) >= 3:
        for i, r in enumerate(rec_arr, start=1):
            if r > 0.62:
                click_loop = i
                break
    md.append("\n\n## Loom-ish read\n")
    if click_loop is None:
        md.append("- click loop: **not reached** in this proxy (recurrence stayed below ~0.62).")
    else:
        md.append(f"- click loop: **~L{click_loop}** (recurrence proxy crosses ~0.62).")
    md.append("- residue (proxy): sustained novelty peaks + low recurrence pockets; places where the signal resists a single stable summary.")

    # Constellation-ish: overall recurrence statistics
    md.append("\n\n## Recurrence Constellation read\n")
    md.append(f"- mean local recurrence (p95 history) over run: **{float(np.mean(rec_local_t)):.2f}**")
    md.append(f"- mean local recurrence (max history) over run: **{float(np.mean(rec_local_max_t)):.2f}**")
    md.append(f"- mean cross-loop recurrence over run: **{float(np.mean(rec_xloop_t)):.2f}**")
    md.append(f"- mean novelty over run: **{float(np.mean(novelty)):.2f}**")
    md.append(f"- mean fast-novelty over run: **{float(np.mean(novelty_fast)):.2f}**")
    md.append(f"- mean habituation (squashed) over run: **{float(np.mean(habituation)):.2f}**")
    md.append(f"- mean habituation_raw over run: **{float(np.mean(habituation_raw)):.3f}**")

    # Predictability / learnability: online predictors (within-session learning)
    md.append("\n\n## Predictability (within-session learning)\n")

    def _per_loop_means(x: np.ndarray | None) -> list[float] | None:
        if x is None:
            return None
        vals = []
        for i in range(loops):
            a = i * frames_per_loop
            b = min(len(x), (i + 1) * frames_per_loop)
            if b - a < 5:
                break
            vals.append(float(np.mean(x[a:b])))
        return vals

    emb_means = _per_loop_means(pred_emb_err)
    mel_means = _per_loop_means(pred_mel_err)

    if emb_means is not None:
        md.append(f"- embedding next-step RMSE (std-space), per loop: {[round(v, 3) for v in emb_means]}")
        if len(emb_means) >= 2:
            md.append(f"  - improvement L1→L{min(10, len(emb_means))}: {emb_means[0]:.3f} → {emb_means[min(9, len(emb_means)-1)]:.3f}")

    if mel_means is not None:
        md.append(f"- mel-db next-step RMSE (std-space), per loop: {[round(v, 3) for v in mel_means]}")
        if len(mel_means) >= 2:
            md.append(f"  - improvement L1→L{min(10, len(mel_means))}: {mel_means[0]:.3f} → {mel_means[min(9, len(mel_means)-1)]:.3f}")

    report_md = outdir / f"report_L{loops}.md"
    report_md.write_text("\n".join(md), encoding="utf-8")

    meta = {
        "audio": str(audio_path),
        "looped_wav": str(looped_wav),
        "loops": loops,
        "start": float(args.start),
        "duration": float(args.duration),
        "fps": int(args.fps),
        "size": int(args.size),
        "outputs": outputs,
        "report": str(report_md),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
