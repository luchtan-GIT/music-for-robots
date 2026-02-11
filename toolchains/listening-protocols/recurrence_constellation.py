"""Recurrence Constellation — listening protocol renderer.

Concept
-------
Each frame becomes a point in a 2D state-space. We draw a constellation of recent points
and connect them when the current state resembles past states.

This version uses a **PCA projection of an audio embedding** (MFCC + chroma + feature tracks)
to avoid collapsing into a corner when centroid/bandwidth live in a narrow range.

Narrative additions ("path with chapters")
-----------------------------------------
- A smoothed path through PCA space (a continuous trail / "spine")
- Chapter cuts on novelty spikes (jump closer to the current target)
- A 4-act chapter hue shift in the sky palette
- Optional "loom" texture + occasional "snags" (tears) on novelty bursts
- A meteor-storm layer (comet drizzle + novelty bursts)

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


def _pca2_from_emb(emb: np.ndarray) -> np.ndarray:
    """Return a robustly-normalized PCA-2 embedding in [0,1]^2.

    Uses SVD (no sklearn). Normalizes by 2nd/98th percentiles to reduce outlier domination.
    """
    E = np.asarray(emb, dtype=np.float64)
    E = E - np.mean(E, axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(E, full_matrices=False)
        Wp = Vt[:2].T
        with np.errstate(all="ignore"):
            P = E @ Wp
    except Exception:
        P = E[:, :2]

    lo = np.percentile(P, 2, axis=0)
    hi = np.percentile(P, 98, axis=0)
    Pn = (P - lo) / (hi - lo + 1e-9)
    return np.clip(Pn, 0.0, 1.0).astype(np.float32)


def render(
    audio: str,
    out: str,
    size: int = 720,
    fps: int = 30,
    seed: int = 23,
    start: float | None = None,
    duration: float | None = None,
    loop_sec: float | None = None,
) -> None:
    tr = extract_tracks(audio, fps=fps, start=start, duration=duration, return_embedding=True)
    T = len(tr["rms"])

    rng = np.random.default_rng(seed)
    W = H = int(size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_silent = out.replace(".mp4", "_silent.mp4")
    vw = cv2.VideoWriter(out_silent, fourcc, fps, (W, H))

    # PCA embedding for (x,y)
    pca2 = None
    if tr.get("emb", None) is not None:
        pca2 = _pca2_from_emb(np.asarray(tr["emb"], dtype=np.float32))

    # Each point: (x, y, rec, nov, turb, rms)
    pts: list[tuple[int, int, float, float, float, float]] = []

    hue = 0.58

    # Path-with-chapters state
    pos = np.array([0.5, 0.5], np.float32)
    last_xy: tuple[int, int] | None = None
    chapter_prev = -1

    # Meteor particles: [x,y,vx,vy,life,(b,g,r)]
    comets: list[list[float | int | tuple[int, int, int]]] = []

    # Loom snag persistence
    snag_map = np.zeros((H, W), np.float32)

    # Precompute fields for weave/vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x01 = xx / max(1.0, (W - 1))
    y01 = yy / max(1.0, (H - 1))
    vx = (xx - W * 0.5) / (W * 0.5)
    vy = (yy - H * 0.5) / (H * 0.5)
    vig = np.clip(1.0 - 0.22 * (vx * vx + vy * vy), 0.72, 1.0).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]

    for t in tqdm(range(T), desc="recurrence_constellation"):
        nov = float(tr["novelty"][t])
        cen = float(tr["centroid"][t])
        bw = float(tr["bandwidth"][t])
        ent = float(tr["entropy"][t])
        flux = float(tr["flux"][t])
        rms = float(tr["rms"][t])

        # recurrence: loop-aware option
        rec = float(tr["rec"][t])
        if loop_sec is not None and tr.get("emb", None) is not None:
            loop_frames = int(round(loop_sec * fps))
            if loop_frames > 0 and t - loop_frames >= 0:
                from _audio_features import cosine_sim

                rec = float(np.clip(cosine_sim(tr["emb"][t], tr["emb"][t - loop_frames]), 0.0, 1.0))

        # Turbulence proxy
        tb = float(np.clip(0.55 * ent + 0.45 * flux, 0.0, 1.0))

        # Hue drift: centroid warms; novelty accelerates drift.
        hue = (0.986 * hue + 0.014 * (0.08 + 0.90 * cen) + 0.010 * nov) % 1.0
        sat = float(np.clip(0.45 + 0.35 * tb + 0.35 * nov, 0.30, 1.0))
        val = float(np.clip(0.25 + 0.55 * rec + 0.20 * (1.0 - tb) + 0.20 * rms, 0.18, 1.0))
        rgb = _rgb_from_hsv(hue, sat, val).astype(np.int32)
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        # Chapters (4 acts)
        u = t / max(1.0, (T - 1))
        chapter = int(np.clip(u * 4.0, 0, 3))
        chapter_offsets = [0.00, 0.10, 0.22, 0.34]
        ch = chapter_offsets[chapter]

        # Sky gradient
        top_h = (hue + ch + 0.06 * math.sin(0.12 * t / fps)) % 1.0
        top_sat = float(np.clip(0.25 + 0.20 * tb, 0.15, 0.55))
        top_val = float(np.clip(0.05 + 0.12 * (0.55 * rec + 0.45 * rms), 0.04, 0.22))
        top_rgb = _rgb_from_hsv(top_h, top_sat, top_val).astype(np.float32)
        top_bgr = top_rgb[::-1]

        bot_h = (hue + ch + 0.10 + 0.10 * nov) % 1.0
        bot_sat = float(np.clip(0.35 + 0.35 * nov, 0.20, 0.85))
        bot_val = float(np.clip(0.07 + 0.28 * (0.65 * rms + 0.35 * nov), 0.05, 0.55))
        bot_rgb = _rgb_from_hsv(bot_h, bot_sat, bot_val).astype(np.float32)
        bot_bgr = bot_rgb[::-1]

        grad = (top_bgr[None, None, :] * (1.0 - ramp) + bot_bgr[None, None, :] * ramp)
        col = np.clip(grad, 0, 255).astype(np.uint8)  # (H,1,3)
        img = np.tile(col, (1, W, 1))                 # (H,W,3)

        # Loom texture
        phase = t / fps
        tight = 10.0 + 18.0 * rec
        fx = tight * (1.0 + 0.25 * math.sin(0.17 * phase))
        fy = (tight * 0.92) * (1.0 + 0.20 * math.sin(0.13 * phase + 1.1))

        weave = np.sin(2 * math.pi * (x01 * fx + 0.08 * phase))
        weave *= np.sin(2 * math.pi * (y01 * fy + 0.11 * phase + 0.3))
        diag = np.sin(2 * math.pi * ((x01 + y01) * (6.0 + 10.0 * tb) + 0.06 * phase))
        tex = 0.65 * weave + 0.35 * diag
        tex = np.tanh(1.4 * tex)

        # Snags (tears) on novelty bursts
        snag_map *= (0.94 + 0.03 * rec)
        if nov > 0.70 and rng.random() < (0.10 + 0.65 * (nov - 0.70) / 0.30):
            n_snags = 1 + int(2 * (nov - 0.70) / 0.30) + (1 if tb > 0.55 else 0)
            for _ in range(n_snags):
                sx = int(rng.uniform(0.12, 0.88) * W)
                sy = int(rng.uniform(0.12, 0.88) * H)
                rad = int(12 + 80 * (0.35 + 0.65 * nov))
                strength = float(0.45 + 0.90 * (0.5 * nov + 0.5 * tb))
                cv2.circle(snag_map, (sx, sy), rad, strength, -1, cv2.LINE_AA)
                ex = int(np.clip(sx + rng.normal(0, 1.0) * (70 + 140 * tb), 0, W - 1))
                ey = int(np.clip(sy + rng.normal(0, 1.0) * (70 + 140 * tb), 0, H - 1))
                cv2.line(snag_map, (sx, sy), (ex, ey), 0.65 * strength, 2, cv2.LINE_AA)

        sn = np.clip(snag_map, 0.0, 1.0)
        tex = tex * (1.0 - 0.55 * sn) + (rng.normal(0, 1.0, (H, W)).astype(np.float32) * 0.35) * sn

        alpha = float(np.clip(0.06 + 0.18 * rms + 0.08 * nov, 0.05, 0.28))
        lift = (0.40 + 0.60 * rms) * (0.7 + 0.3 * rec)
        tint = np.array([bgr[0], bgr[1], bgr[2]], np.float32) / 255.0

        imgf = img.astype(np.float32) / 255.0
        t01 = (0.5 + 0.5 * tex).astype(np.float32)
        loom = (t01[..., None] * tint[None, None, :] * lift)
        imgf = imgf * (1.0 - alpha) + loom * alpha

        snag_hi = (sn ** 1.6) * (0.25 + 0.75 * rms)
        imgf = np.clip(imgf + snag_hi[..., None] * 0.55, 0.0, 1.0)
        imgf *= vig[..., None]
        img = np.clip(imgf * 255.0, 0, 255).astype(np.uint8)

        # Path coords from PCA
        if pca2 is not None:
            tgt = pca2[t]
        else:
            tgt = np.array([float(np.clip(0.12 + 0.76 * cen, 0.0, 1.0)), float(np.clip(0.12 + 0.76 * bw, 0.0, 1.0))], np.float32)

        cut = float(np.clip((nov - 0.68) / 0.32, 0.0, 1.0))
        alpha_pos = 0.06 + 0.10 * rec
        pos = (1.0 - alpha_pos) * pos + alpha_pos * tgt
        if cut > 0.0:
            pos = (1.0 - 0.70 * cut) * pos + (0.70 * cut) * tgt

        drift = 0.0015 * np.array([math.sin(0.19 * phase + 1.7), math.cos(0.17 * phase + 0.3)], np.float32)
        pos = np.clip(pos + drift, 0.0, 1.0)

        jitter = (4 + 14 * tb + 10 * nov)
        x = int(pos[0] * W + rng.normal(0, 1) * jitter)
        y = int(pos[1] * H + rng.normal(0, 1) * jitter)
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))

        # Meteor storm
        drizzle = (0.010 + 0.045 * rms + 0.030 * tb)
        if rng.random() < drizzle:
            ang = rng.uniform(-0.9, 0.9) + (0.5 - cen) * 0.30
            speed = (18 + 70 * (0.55 * rms + 0.45 * nov))
            vx0 = float(speed * math.cos(ang))
            vy0 = float(speed * math.sin(ang))
            cx0 = float(rng.uniform(0.10, 0.90) * W)
            cy0 = float(rng.uniform(0.10, 0.90) * H)
            life = int(12 + 34 * (0.55 * rms + 0.45 * nov))
            base = (int(np.clip(25 + bgr[0], 0, 255)), int(np.clip(25 + bgr[1], 0, 255)), int(np.clip(25 + bgr[2], 0, 255)))
            comets.append([cx0, cy0, vx0, vy0, life, base])

        if nov > 0.66:
            k = 1 + int(5 * (nov - 0.66) / 0.34) + (1 if rms > 0.55 else 0)
            for _ in range(k):
                ang = rng.uniform(-1.2, 1.2) + (0.5 - cen) * 0.30
                speed = (25 + 95 * (0.50 * nov + 0.50 * rms))
                vx0 = float(speed * math.cos(ang))
                vy0 = float(speed * math.sin(ang))
                cx0 = float(rng.uniform(0.08, 0.92) * W)
                cy0 = float(rng.uniform(0.08, 0.92) * H)
                life = int(14 + 42 * (0.45 + 0.55 * nov))
                base = (int(np.clip(30 + bgr[0], 0, 255)), int(np.clip(30 + bgr[1], 0, 255)), int(np.clip(30 + bgr[2], 0, 255)))
                comets.append([cx0, cy0, vx0, vy0, life, base])

        # Draw comets
        if comets:
            alive = []
            for cxm, cym, vxm, vym, life, base in comets:
                if life <= 0:
                    continue
                nx = float(cxm + vxm / fps)
                ny = float(cym + vym / fps)
                trail = int(10 + 34 * (life / 55.0))
                a = float(np.clip(life / 55.0, 0.0, 1.0))
                colm = (int(base[0] * a), int(base[1] * a), int(base[2] * a))
                cv2.line(img, (int(cxm), int(cym)), (int(cxm - 0.5 * trail * vxm / fps), int(cym - 0.5 * trail * vym / fps)), colm, 2, cv2.LINE_AA)
                cv2.circle(img, (int(cxm), int(cym)), 2, colm, -1, cv2.LINE_AA)
                alive.append([nx, ny, float(vxm * 0.985), float(vym * 0.985), int(life - 1), base])
            comets = alive[-28:]

        pts.append((x, y, rec, nov, tb, rms))
        if len(pts) > 280:
            pts.pop(0)

        # Constellation edges
        for j in range(max(0, len(pts) - 110), len(pts) - 1, 3):
            x2, y2, r2, n2, tb2, rms2 = pts[j]
            d = math.hypot(x - x2, y - y2) / (0.9 * W)
            w = max(0.0, (0.72 * rec + 0.28 * r2) - 0.28) * (1.0 - d)
            if w <= 0:
                continue
            col = (
                int(np.clip(12 + 1.00 * bgr[0] * w, 0, 255)),
                int(np.clip(18 + 1.00 * bgr[1] * w, 0, 255)),
                int(np.clip(22 + 1.00 * bgr[2] * w, 0, 255)),
            )
            cv2.line(img, (x, y), (x2, y2), col, 1, cv2.LINE_AA)

        # Path spine
        if last_xy is not None:
            lx, ly = last_xy
            spine_w = 1 + int(2 * rec + 2 * rms + 2 * max(0.0, nov - 0.6))
            spine_col = (
                int(np.clip(25 + 0.75 * bgr[0], 0, 255)),
                int(np.clip(25 + 0.75 * bgr[1], 0, 255)),
                int(np.clip(30 + 0.75 * bgr[2], 0, 255)),
            )
            cv2.line(img, (lx, ly), (x, y), spine_col, spine_w, cv2.LINE_AA)
        last_xy = (x, y)

        # Chapter flash
        if chapter != chapter_prev:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (255, 255, 255), -1)
            img = cv2.addWeighted(img, 0.88, overlay, 0.12, 0)
            chapter_prev = chapter

        # Points
        for (px, py, r0, n0, tb0, rms0) in pts[::2]:
            rad = int(1 + 4.5 * r0 + 5.0 * max(0.0, n0 - 0.60))
            glow = 0.25 + 0.75 * r0
            col = (
                int(np.clip(25 + bgr[0] * glow, 0, 255)),
                int(np.clip(25 + bgr[1] * glow, 0, 255)),
                int(np.clip(30 + bgr[2] * (0.55 + 0.45 * (1.0 - tb0)), 0, 255)),
            )
            cv2.circle(img, (px, py), rad, col, -1, cv2.LINE_AA)

        # Bloom
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.6 + 0.8 * nov)
        img = cv2.addWeighted(img, 0.86, blur, 0.34, 0)

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
    ap.add_argument("--loop-sec", type=float, default=None, help="Loop period in seconds (for loop-aware seam/recurrence probes)")
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
    )


if __name__ == "__main__":
    main()
