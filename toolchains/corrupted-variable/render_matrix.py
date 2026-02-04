import argparse
import math
import os

import cv2
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [5, 95])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--max-seconds', type=float, default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--grid', type=int, default=64, help='matrix cells per side (try 96–144 for higher resolution)')
    ap.add_argument('--style', choices=['cells', 'analog', 'hybrid'], default='hybrid')
    ap.add_argument('--zoom', action='store_true', help='(legacy) viewport zoom; prefer --rescale')
    ap.add_argument('--zoom-strength', type=float, default=0.55, help='zoom amount (0..1ish)')
    ap.add_argument('--rescale', action='store_true', help='change apparent matrix resolution by aggregating the cell grid (true resolution shift)')
    ap.add_argument('--rescale-steps', type=int, default=6, help='number of discrete resolution steps (>=2)')
    ap.add_argument('--rescale-strength', type=float, default=1.0, help='how strongly audio drives resolution changes')
    ap.add_argument('--rescale-min', type=int, default=16, help='minimum displayed grid size when rescaling (coarsest)')
    ap.add_argument('--narrative', action='store_true', help='apply a slow narrative transform (palette drift + squares→dots morph)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load audio
    y, sr = librosa.load(args.audio, sr=args.sr, mono=True)
    if args.max_seconds is not None:
        y = y[: int(args.max_seconds * sr)]

    T = len(y) / sr
    n_frames = int(math.ceil(T * args.fps))

    # --- Listen without a human: derive raw control fields ---
    # Use mel spectrogram as a dense frequency/intensity field.
    hop = max(256, int(sr / args.fps))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=64, fmin=30, fmax=sr / 2)
    S = librosa.power_to_db(S, ref=np.max)
    S = norm01(S)  # 0..1

    # Align mel frames to video frames
    mlen = S.shape[1]
    vid_idx = np.clip(np.linspace(0, mlen - 1, n_frames).astype(int), 0, mlen - 1)
    mel = S[:, vid_idx]  # [mels, frames]

    # Global intensity + regularity estimates
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = norm01(rms)[vid_idx]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_env = norm01(onset_env)[vid_idx]

    # Regularity proxy: how strong is the best repeating lag in onset autocorr in a sliding window.
    # (Not tempo; a scalar 0..1 per frame.)
    win = int(args.fps * 3.0)
    reg = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a0 = max(0, i - win)
        a1 = min(n_frames, i + 1)
        w = onset_env[a0:a1]
        if len(w) < 16:
            reg[i] = 0.0
            continue
        ac = np.correlate(w - w.mean(), w - w.mean(), mode='full')
        ac = ac[len(ac) // 2 :]
        ac[:4] = 0
        best = float(np.max(ac))
        reg[i] = best / (float(ac[0]) + 1e-9) if ac[0] > 0 else 0.0
    reg = norm01(reg)

    # Alterations proxy: flux over mel bands
    flux = np.sqrt(((np.diff(mel, axis=1, prepend=mel[:, :1])) ** 2).mean(axis=0))
    flux = norm01(flux)

    # Beat phase (for scanlines / analog drift). If tracking fails, fallback to a simple phase.
    beat_phase = np.zeros(n_frames, dtype=np.float32)
    try:
        onset_full = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        _tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_full, sr=sr, hop_length=hop)
        beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(onset_full))]
        # convert beat frames (mel frame domain) to video frame domain (approx)
        beat_vid = np.clip((beat_frames / max(1, mlen - 1) * (n_frames - 1)).astype(int), 0, n_frames - 1)
        if len(beat_vid) >= 2:
            for a, b in zip(beat_vid[:-1], beat_vid[1:]):
                if b <= a:
                    continue
                beat_phase[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False, dtype=np.float32)
            beat_phase[: beat_vid[0]] = np.linspace(0.0, 1.0, beat_vid[0], endpoint=False, dtype=np.float32) if beat_vid[0] > 0 else 0.0
            beat_phase[beat_vid[-1] :] = np.linspace(0.0, 1.0, n_frames - beat_vid[-1], endpoint=False, dtype=np.float32) if beat_vid[-1] < n_frames else 0.0
        else:
            beat_phase = (np.linspace(0.0, 1.0, n_frames, dtype=np.float32) * 6.0) % 1.0
    except Exception:
        beat_phase = (np.linspace(0.0, 1.0, n_frames, dtype=np.float32) * 6.0) % 1.0

    # --- Visual matrix dynamics ---
    # Each cell has (x): state value, (p): phase, (v): velocity.
    g = int(args.grid)
    x = rng.random((g, g), dtype=np.float32)
    p = rng.random((g, g), dtype=np.float32)
    v = rng.normal(0, 0.05, (g, g)).astype(np.float32)

    # Assign each cell a preferred frequency band (mel index) and a sensitivity.
    # Use a 2D mapping (rows=low->high, cols=jitter) so we don't start with a pattern.
    yy, xx = np.mgrid[0:g, 0:g]
    band = (yy / max(1, g - 1) * (mel.shape[0] - 1)).astype(int)
    band = np.clip(band + (rng.integers(-2, 3, size=(g, g))), 0, mel.shape[0] - 1)

    sens = (0.35 + 0.75 * rng.random((g, g))).astype(np.float32)
    bias = rng.normal(0, 0.08, (g, g)).astype(np.float32)

    # Render setup
    tmp_video = args.out + '.tmp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(tmp_video, fourcc, args.fps, (args.size, args.size))

    # Convenience: upsample cell grid to pixels
    cell_px = max(1, args.size // g)
    W = cell_px * g
    H = cell_px * g

    # Border/padding if size isn't divisible
    pad_x = args.size - W
    pad_y = args.size - H

    # Precompute shape masks tiled per cell:
    # - circlemask: fills most of the cell
    # - dotmask: smaller, tighter dot
    def make_circle_mask(radius_frac: float, edge_pow: float = 1.6):
        t = np.ones((cell_px, cell_px), dtype=np.float32)
        if cell_px >= 2:
            t[:] = 0.0
            rr = max(1.0, (cell_px - 1) * float(radius_frac))
            cx = (cell_px - 1) / 2.0
            cy = (cell_px - 1) / 2.0
            yy2, xx2 = np.mgrid[0:cell_px, 0:cell_px].astype(np.float32)
            dist = np.sqrt((xx2 - cx) ** 2 + (yy2 - cy) ** 2)
            t = np.clip(1.0 - dist / rr, 0.0, 1.0) ** float(edge_pow)
        m = np.tile(t, (g, g))
        m = m[:H, :W]
        return np.repeat(m[:, :, None], 3, axis=2)

    circlemask3 = make_circle_mask(radius_frac=0.48, edge_pow=1.3)
    dotmask3 = make_circle_mask(radius_frac=0.33, edge_pow=1.9)

    neon_until = -1  # frame index for neon burst

    for i in tqdm(range(n_frames), desc='render matrix'):
        # Audio drives local forces
        m = mel[:, i]
        drive = m[band]  # [g,g]

        inten = float(rms[i])
        r = float(reg[i])
        alt = float(flux[i])
        on = float(onset_env[i])

        tsec = i / args.fps
        pp = tsec / max(1e-6, T)

        # Trigger a brief NEON burst once, based on audio (late-ish and a strong alteration/onset moment).
        if args.narrative and neon_until < 0 and pp > 0.62:
            if (alt > 0.72 and on > 0.55) or (alt > 0.82):
                neon_until = i + int(5.0 * args.fps)

        # Coupling strength: higher regularity -> stronger synchronization; higher alteration -> more turbulence.
        k_sync = 0.02 + 0.18 * r
        k_noise = 0.01 + 0.10 * alt
        k_drive = 0.05 + 0.35 * inten

        # Local neighborhood interaction: discrete Laplacian
        lap = (
            -4 * x
            + np.roll(x, 1, 0)
            + np.roll(x, -1, 0)
            + np.roll(x, 1, 1)
            + np.roll(x, -1, 1)
        )

        # Phase dynamics: cells try to entrain to neighbors when regularity is high.
        p_mean = (
            p
            + np.roll(p, 1, 0)
            + np.roll(p, -1, 0)
            + np.roll(p, 1, 1)
            + np.roll(p, -1, 1)
        ) / 5.0

        # Update: a hybrid of coupled oscillators + driven reaction-diffusion-ish state.
        # No predesigned pattern; pattern must emerge from audio+coupling.
        noise = rng.normal(0, 1.0, (g, g)).astype(np.float32)

        # Drive shifts the phase velocity; onset spikes cause abrupt phase resets (bit-flip feeling).
        p = (p + (0.012 + 0.08 * drive) + k_sync * (p_mean - p) + k_noise * noise * 0.02) % 1.0
        if on > 0.75:
            p = (p + 0.35 + 0.15 * noise) % 1.0

        # State x integrates a non-linear function of phase + drive, then diffuses.
        pulse = np.exp(-6.0 * p).astype(np.float32)
        target = (0.15 + 0.85 * (pulse * drive))
        v = 0.985 * v + k_drive * (target - x) + 0.12 * lap + bias
        v += (k_noise * 0.03) * noise
        x = np.clip(x + v, 0, 1)

        # Build visual channels
        # Hue: frequency band + phase (full-spectrum)
        hue = (0.02 + 0.96 * (band / max(1, mel.shape[0] - 1)) + 0.45 * p) % 1.0
        if args.narrative:
            tsec = i / args.fps
            pp = tsec / max(1e-6, T)
            # Narrative palette plan:
            # - Start palette (offset 0)
            # - Mid palette shift (offset +0.33)
            # - After blackout: return to start palette (offset 0) while staying in dots.
            def ss(a, b, x):
                x = (x - a) / max(1e-9, (b - a))
                x = max(0.0, min(1.0, x))
                return x * x * (3 - 2 * x)

            pal_shift = ss(0.42, 0.52, pp)  # transition into shifted palette
            # If neon burst happened, return palette immediately after it ends; else fallback return near end.
            after_neon = 0.0
            if neon_until >= 0:
                t_end = neon_until / args.fps
                after_neon = ss(t_end / max(1e-6, T), (t_end + 2.0) / max(1e-6, T), pp)
            # Force return by the final act so we end in the original palette.
            pal_return = max(after_neon, ss(0.82, 0.90, pp))
            hue_off = (0.33 * pal_shift) * (1.0 - pal_return)
            hue = (hue + hue_off) % 1.0
        # Saturation: keep it vivid; modulate with alterations + local gradient
        grad = np.sqrt((np.roll(x, 1, 0) - x) ** 2 + (np.roll(x, 1, 1) - x) ** 2)
        sat = np.clip(0.55 + 0.35 * alt + 0.65 * grad, 0, 1)
        # Value: energy with a higher floor so colors don't die
        val = np.clip(0.18 + 0.82 * (0.20 + 0.80 * x), 0, 1)

        # Neon burst (instead of going dark): punch saturation/value and rotate hue.
        if args.narrative and neon_until >= 0 and i <= neon_until:
            hue = (hue + 0.55 + 0.10 * np.sin(2 * np.pi * p)) % 1.0
            sat = np.clip(0.80 + 0.20 * sat, 0, 1)
            val = np.clip(0.55 + 0.45 * val, 0, 1)

        hsv = np.zeros((g, g, 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8((hue * 179) % 179)
        hsv[..., 1] = np.uint8(np.clip(sat * 255, 0, 255))
        hsv[..., 2] = np.uint8(np.clip(val * 255, 0, 255))

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Pixelate to a matrix aesthetic (base grid)
        img = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_NEAREST)

        # True resolution shift: aggregate the matrix itself (not viewport zoom).
        # We keep dynamics at base grid g, but display an effective grid g_eff by block pooling.
        if args.rescale:
            steps = max(2, int(args.rescale_steps))
            g_min = max(4, min(int(args.rescale_min), g))

            # drive: regularity/intensity -> finer; alterations -> coarser
            drive_res = clamp01(0.55 * r + 0.25 * inten + 0.20 * (1.0 - alt))
            drive_res = clamp01((drive_res - 0.30) * 1.6)
            drive_res = clamp01(drive_res * float(args.rescale_strength))

            # Map to discrete effective grid sizes between g_min..g
            level = int(round(drive_res * (steps - 1)))
            level = max(0, min(steps - 1, level))
            # linear in grid-size space (not powers of 2)
            g_eff = int(round(g_min + (g - g_min) * (level / max(1, steps - 1))))
            g_eff = max(4, min(g_eff, g))

            # Downsample WITHOUT washing out colors:
            # work in HSV float and do max-pooling for S,V; circular mean for H.
            hsv_f = np.zeros((g, g, 3), dtype=np.float32)
            hsv_f[..., 0] = hue
            hsv_f[..., 1] = sat
            hsv_f[..., 2] = val

            # Resize down for block aggregation
            # Hue: convert to unit circle then average
            ang = 2 * np.pi * hsv_f[..., 0]
            hx = np.cos(ang)
            hy = np.sin(ang)
            w = hsv_f[..., 2]  # weight by value

            hx_s = cv2.resize((hx * w).astype(np.float32), (g_eff, g_eff), interpolation=cv2.INTER_AREA)
            hy_s = cv2.resize((hy * w).astype(np.float32), (g_eff, g_eff), interpolation=cv2.INTER_AREA)
            w_s = cv2.resize(w.astype(np.float32), (g_eff, g_eff), interpolation=cv2.INTER_AREA) + 1e-9
            h_s = (np.arctan2(hy_s, hx_s) / (2 * np.pi)) % 1.0

            # Saturation/Value: max-ish via area then gamma boost
            s_s = cv2.resize(hsv_f[..., 1].astype(np.float32), (g_eff, g_eff), interpolation=cv2.INTER_AREA)
            v_s = cv2.resize(hsv_f[..., 2].astype(np.float32), (g_eff, g_eff), interpolation=cv2.INTER_AREA)
            s_s = np.clip(s_s ** 0.75, 0, 1)
            v_s = np.clip(v_s ** 0.85, 0, 1)

            hsv_u8_small = np.zeros((g_eff, g_eff, 3), dtype=np.uint8)
            hsv_u8_small[..., 0] = np.uint8((h_s * 179) % 179)
            hsv_u8_small[..., 1] = np.uint8(np.clip(s_s * 255, 0, 255))
            hsv_u8_small[..., 2] = np.uint8(np.clip(v_s * 255, 0, 255))
            small_bgr = cv2.cvtColor(hsv_u8_small, cv2.COLOR_HSV2BGR)

            img = cv2.resize(small_bgr, (W, H), interpolation=cv2.INTER_NEAREST)

        # Style pass
        if args.style == 'cells':
            # Keep it hard-edged. Optionally show grid always (faint).
            if alt > 0.15:
                grid_col = (12, 12, 12)
                step = cell_px
                for yy2 in range(0, H, step):
                    cv2.line(img, (0, yy2), (W - 1, yy2), grid_col, 1)
                for xx2 in range(0, W, step):
                    cv2.line(img, (xx2, 0), (xx2, H - 1), grid_col, 1)

        elif args.style == 'analog':
            # CRT-ish: blur/bleed + scanlines, driven by regularity/intensity.
            ph = float(beat_phase[i])
            sigma = 0.8 + 1.6 * r + 0.6 * inten
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            # scanlines
            sl = img.astype(np.float32)
            rows = np.arange(H, dtype=np.float32)
            scan = 0.88 + 0.12 * np.sin(2 * np.pi * (rows / 3.0 + 0.8 * ph))
            sl *= scan[:, None, None]
            img = np.clip(sl, 0, 255).astype(np.uint8)
            # slight horizontal smear when alterations spike
            if alt > 0.45:
                img = cv2.blur(img, (1 + int(10 * alt), 1))

        else:
            # hybrid: crisp cells, but coherence blur when regularity is high,
            # and grid lines appear when alterations spike.
            if r > 0.55:
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.35 + 1.4 * (r - 0.55))

            if alt > 0.55:
                grid_col = (18, 18, 18)
                step = cell_px
                for yy2 in range(0, H, step):
                    cv2.line(img, (0, yy2), (W - 1, yy2), grid_col, 1)
                for xx2 in range(0, W, step):
                    cv2.line(img, (xx2, 0), (xx2, H - 1), grid_col, 1)

        # Optional zoom: resize and crop around center (apparent resolution changes)
        frame = img
        if args.zoom:
            ph = float(beat_phase[i])
            # zoom factor: slow breathing + audio terms
            # regularity encourages slow zoom-in (inspection); alterations push zoom-out (context)
            z = 1.0 + args.zoom_strength * (0.30 * math.sin(2 * math.pi * (0.09 * (i / args.fps) + ph)) + 0.55 * r - 0.35 * alt + 0.20 * inten)
            z = max(0.75, min(1.65, z))
            Zw = int(W * z)
            Zh = int(H * z)
            # scale up/down
            resized = cv2.resize(frame, (Zw, Zh), interpolation=cv2.INTER_NEAREST if args.style == 'cells' else cv2.INTER_LINEAR)
            # center crop back to W,H
            cx = Zw // 2
            cy = Zh // 2
            x0 = max(0, cx - W // 2)
            y0 = max(0, cy - H // 2)
            x1 = min(Zw, x0 + W)
            y1 = min(Zh, y0 + H)
            cropped = resized[y0:y1, x0:x1]
            # if crop undershoots due to rounding, pad
            if cropped.shape[0] != H or cropped.shape[1] != W:
                canvas2 = np.zeros((H, W, 3), dtype=np.uint8)
                oy2 = (H - cropped.shape[0]) // 2
                ox2 = (W - cropped.shape[1]) // 2
                canvas2[oy2 : oy2 + cropped.shape[0], ox2 : ox2 + cropped.shape[1]] = cropped
                cropped = canvas2
            frame = cropped

        # Narrative morph: squares → dots (without changing dynamics)
        if args.narrative:
            tsec = i / args.fps
            pp = tsec / max(1e-6, T)
            # Narrative per your spec:
            # 1) grids
            # 2) squares → circles
            # 3) palette change
            # 4) circles → dots
            # 5) blackout ~5s (audio-triggered)
            # 6) return to starting palette, but now dots
            def ss(a, b, x):
                x = (x - a) / max(1e-9, (b - a))
                x = max(0.0, min(1.0, x))
                return x * x * (3 - 2 * x)

            circles = ss(0.14, 0.34, pp)
            dots = ss(0.54, 0.72, pp)

            if frame.shape[0] == H and frame.shape[1] == W:
                f32 = frame.astype(np.float32)
                f_circle = f32 * circlemask3
                mix1 = (1.0 - circles) * f32 + circles * f_circle
                f_dot = f32 * dotmask3
                mix2 = (1.0 - dots) * mix1 + dots * f_dot
                frame = np.clip(mix2, 0, 255).astype(np.uint8)

            # Neon section glow (instead of blackout)
            if neon_until >= 0 and i <= neon_until and frame.shape[0] == H and frame.shape[1] == W:
                blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2 + 1.2 * alt)
                frame = cv2.addWeighted(frame, 1.0, blur, 0.65, 0)
            elif neon_until >= 0 and i > neon_until and frame.shape[0] == H and frame.shape[1] == W:
                # Small post-neon "re-acclimation" lift (keeps the ending from dimming out)
                lift = 1.0 + 0.18 * min(1.0, (i - neon_until) / (2.0 * args.fps))
                frame = np.clip(frame.astype(np.float32) * lift, 0, 255).astype(np.uint8)

        # Pad to requested size
        if pad_x or pad_y:
            canvas = np.zeros((args.size, args.size, 3), dtype=np.uint8)
            ox = pad_x // 2
            oy = pad_y // 2
            canvas[oy : oy + H, ox : ox + W] = frame
            frame = canvas

        vw.write(frame)

    vw.release()

    cmd = (
        f'ffmpeg -y -loglevel error '
        f'-i "{tmp_video}" -i "{args.audio}" '
        f'-c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '
        f'"{args.out}"'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise SystemExit('ffmpeg mux failed')

    try:
        os.remove(tmp_video)
    except OSError:
        pass


if __name__ == '__main__':
    main()
