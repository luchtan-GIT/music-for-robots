#!/usr/bin/env python3
"""Real-time audio-reactive "uncertainty eye" visualizer.

Usage:
  python3 realtime_eye.py /path/to/audio.wav

Controls:
  SPACE  pause/resume
  ESC    quit

What it does:
- Plays the audio file.
- Computes lightweight real-time features (RMS, spectral flux, spectral centroid,
  spectral entropy).
- Drives a Julia set "camera" that *travels* through the fractal (zoom + pan +
  slight c-drift), plus an iris ring.

Design choice:
- The most "uncertain" feeling (for this track) tends to come from *novelty*
  (flux) combined with *complexity/noise-likeness* (entropy). So:
  - flux -> fast impulses (jitter + snap zoom)
  - entropy -> slow drift (camera wander + deeper exploration)
"""

import sys
import time
import math
import threading

import numpy as np
import soundfile as sf
import sounddevice as sd
import pygame

from numba import njit

# ------------------------------ DSP helpers ------------------------------

def hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(1, n - 1))).astype(np.float32)

class FeatureState:
    def __init__(self):
        self.lock = threading.Lock()
        self.paused = False

        # normalized-ish features (EMA relative)
        self.rms = 0.0
        self.flux = 0.0
        self.centroid = 0.0
        self.entropy = 0.0
        self.unc = 0.0

        # transport time
        self.t = 0.0

    def update(self, rms, flux, centroid, entropy, unc, t):
        with self.lock:
            self.rms = float(rms)
            self.flux = float(flux)
            self.centroid = float(centroid)
            self.entropy = float(entropy)
            self.unc = float(unc)
            self.t = float(t)

    def get(self):
        with self.lock:
            return self.rms, self.flux, self.centroid, self.entropy, self.unc, self.t

# ------------------------------ Julia renderer ------------------------------

@njit(cache=True)
def julia_escape(width, height, zoom, cx, cy, cre, cim, max_iter=110):
    """Escape-time Julia set (simple and fast). Returns iter/max_iter in [0..1]."""
    out = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            # map pixel to complex plane
            zr = (x / (width - 1) - 0.5) * zoom + cx
            zi = (y / (height - 1) - 0.5) * zoom + cy
            it = max_iter
            for i in range(max_iter):
                zr2 = zr * zr - zi * zi + cre
                zi2 = 2.0 * zr * zi + cim
                zr, zi = zr2, zi2
                if zr * zr + zi * zi > 4.0:
                    it = i
                    break
            out[y, x] = it / max_iter
    return out

# ------------------------------ Main ------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    path = sys.argv[1]

    # Always float32 so sounddevice is happy
    audio, sr = sf.read(path, always_2d=True, dtype='float32')
    mono = audio.mean(axis=1).astype(np.float32)

    block = 1024
    win = 2048
    w = hann(win)

    eps = 1e-9

    # running feature scales (EMA) for normalization
    smooth = 0.12
    rms_s = 0.0
    flux_s = 0.0
    cent_s = 0.0
    ent_s = 0.0

    state = FeatureState()

    duration = len(mono) / sr

    # audio stream state
    play_pos = 0
    prev_mag = None

    def audio_cb(outdata, frames, time_info, status):
        nonlocal play_pos, prev_mag, rms_s, flux_s, cent_s, ent_s

        if state.paused:
            outdata[:] = 0
            return

        end = min(play_pos + frames, len(audio))
        chunk = audio[play_pos:end]
        if len(chunk) < frames:
            pad = np.zeros((frames - len(chunk), audio.shape[1]), dtype=audio.dtype)
            chunk = np.vstack([chunk, pad])
        outdata[:] = chunk

        # analysis window ending at current output frame
        a0 = max(0, play_pos + frames - win)
        frame = mono[a0:play_pos + frames]
        if len(frame) < win:
            frame = np.pad(frame, (win - len(frame), 0))
        frame = frame.astype(np.float32) * w

        # RMS
        rms = float(np.sqrt(np.mean(frame * frame) + eps))

        # Spectrum
        spec = np.fft.rfft(frame)
        mag = (np.abs(spec).astype(np.float32) + eps)

        # Flux (half-wave rectified difference)
        if prev_mag is None:
            flux = 0.0
        else:
            diff = mag - prev_mag
            diff[diff < 0] = 0
            flux = float(np.sum(diff) / len(diff))
        prev_mag = mag

        # Centroid
        freqs = (np.arange(len(mag)) * (sr / win)).astype(np.float32)
        centroid = float((freqs * mag).sum() / (mag.sum() + eps))

        # Spectral entropy (normalized 0..1)
        p = mag / (mag.sum() + eps)
        ent = float(-(p * np.log2(p + eps)).sum() / math.log2(len(p)))

        # running normalization
        rms_s = (1 - smooth) * rms_s + smooth * rms
        flux_s = (1 - smooth) * flux_s + smooth * flux
        cent_s = (1 - smooth) * cent_s + smooth * centroid
        ent_s = (1 - smooth) * ent_s + smooth * ent

        rms_n = rms / (rms_s + eps)
        flux_n = flux / (flux_s + eps)
        cent_n = centroid / (cent_s + eps)
        ent_n = ent / (ent_s + eps)

        # uncertainty scalar: flux (surprise) + entropy (complexity) + brightness movement
        # Keep it vivid: allow flux to dominate spikes.
        u = (
            0.55 * np.tanh(0.95 * (flux_n - 1.0)) +
            0.30 * np.tanh(0.70 * (ent_n - 1.0)) +
            0.15 * np.tanh(0.60 * (cent_n - 1.0))
        )
        unc = float((u + 1.0) / 2.0)  # 0..1

        t = play_pos / sr
        state.update(rms_n, flux_n, cent_n, ent_n, unc, t)

        play_pos += frames
        if play_pos >= len(audio):
            raise sd.CallbackStop()

    # ---------- Pygame ----------
    pygame.init()
    W, H = 980, 980
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Invalid Pointer — Uncertainty Eye (real-time)')
    clock = pygame.time.Clock()

    # Render fractal at lower res then upscale
    fw, fh = 420, 420
    fract_surf = pygame.Surface((fw, fh))

    # Base Julia constant (c) near "eye" regions; we will drift it slightly.
    cre0, cim0 = -0.76, 0.12

    # Camera state (smoothed) — this is where the interesting motion lives
    zoom = 3.0
    cx = 0.0
    cy = 0.0
    cre = cre0
    cim = cim0

    # "wander" integrator variables (gives motion memory)
    vx = 0.0
    vy = 0.0

    # start audio
    stream = sd.OutputStream(
        samplerate=sr,
        channels=audio.shape[1],
        dtype='float32',
        blocksize=block,
        callback=audio_cb,
    )

    last_fractal = 0.0

    try:
        with stream:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            state.paused = not state.paused

                rms_n, flux_n, cent_n, ent_n, unc, t = state.get()

                # time phase (for rotation / axis selection)
                phase = (t / max(duration, 1e-6)) * (2 * math.pi)

                # ---------- map audio -> camera motion ----------
                # split into a fast component (flux) and slow component (entropy)
                # flux spikes should cause sharp zoom + jitter; entropy should cause drift.
                flux_drive = max(0.0, math.tanh(0.85 * (flux_n - 1.0)))          # 0..~1
                ent_drive  = max(0.0, math.tanh(0.70 * (ent_n  - 1.0)))          # 0..~1

                # Zoom: breathe + occasionally dive. (lower zoom number == deeper zoom because we scale plane.)
                target_zoom = 3.2 * (0.88 - 0.35 * ent_drive) / (1.0 + 0.75 * flux_drive)
                target_zoom = max(0.85, min(4.2, target_zoom))
                zoom = 0.90 * zoom + 0.10 * target_zoom

                # Pan: a "walker" in the complex plane. Entropy pushes exploration; flux adds tremor.
                # Choose axes that rotate with time, so it doesn't feel stuck on one line.
                ax = math.cos(phase * 0.73) + 0.4 * math.cos(phase * 2.1)
                ay = math.sin(phase * 0.61) + 0.4 * math.sin(phase * 1.7)
                norm = math.sqrt(ax*ax + ay*ay) + 1e-6
                ax /= norm
                ay /= norm

                # acceleration from entropy (slow) + flux (fast)
                acc = 0.0045 * (0.6 + 1.9 * ent_drive) + 0.0065 * flux_drive
                vx = 0.92 * vx + acc * ax
                vy = 0.92 * vy + acc * ay

                # add tiny jitter proportional to flux
                jx = 0.0012 * flux_drive * math.cos(phase * 6.0)
                jy = 0.0012 * flux_drive * math.sin(phase * 5.0)

                cx = 0.985 * cx + vx + jx
                cy = 0.985 * cy + vy + jy

                # Keep camera bounded so it doesn't fly off into boring space
                bound = 0.85
                cx = max(-bound, min(bound, cx))
                cy = max(-bound, min(bound, cy))

                # c drift: subtle but makes the "eyes" morph. Tie it to centroid (brightness) and unc.
                c_amt = 0.006 + 0.020 * unc
                cre_t = cre0 + c_amt * math.cos(phase * 0.9 + 1.3 * math.tanh(0.5 * (cent_n - 1.0)))
                cim_t = cim0 + c_amt * math.sin(phase * 1.1 + 0.9 * math.tanh(0.5 * (cent_n - 1.0)))
                cre = 0.92 * cre + 0.08 * cre_t
                cim = 0.92 * cim + 0.08 * cim_t

                # ---------- render ----------
                screen.fill((6, 8, 18))

                now = time.time()
                if now - last_fractal > 0.05:
                    esc = julia_escape(fw, fh, zoom, cx, cy, cre, cim, max_iter=110)
                    v = (1.0 - esc) ** 0.52

                    # palette modulation: rotate purple/orange balance with centroid + unc
                    pal = 0.65 + 0.35 * math.tanh(0.6 * (cent_n - 1.0))
                    hot = 0.35 + 0.65 * unc

                    r = np.clip(20 + (210 * (v ** (0.85 - 0.25*hot))) + 25*np.sin(v*7.0), 0, 255).astype(np.uint8)
                    g = np.clip(8  + (130 * (v ** (1.10))) * (0.75 + 0.35*pal), 0, 255).astype(np.uint8)
                    b = np.clip(45 + (255 * (v ** (0.65))) * (0.70 + 0.60*(1-pal)), 0, 255).astype(np.uint8)

                    rgb = np.dstack([r, g, b])
                    pygame.surfarray.blit_array(fract_surf, np.transpose(rgb, (1, 0, 2)))
                    last_fractal = now

                # composite fractal
                cxp, cyp = W // 2, H // 2
                scale = 1.75
                fs = pygame.transform.smoothscale(fract_surf, (int(fw * scale), int(fh * scale)))
                rect = fs.get_rect(center=(cxp, cyp))
                screen.blit(fs, rect)

                # iris ring
                outer = int(min(W, H) * 0.435)
                inner = int(min(W, H) * 0.285)
                thick = int(8 + 34 * (0.30*ent_drive + 0.70*unc))

                ring_r = int(110 + 145 * min(1.0, unc))
                ring_g = int(40 + 65  * min(1.0, flux_drive))
                ring_b = int(150 + 90 * min(1.0, ent_drive + 0.25))

                pygame.draw.circle(screen, (ring_r, ring_g, ring_b), (cxp, cyp), outer, thick)
                pygame.draw.circle(screen, (10, 10, 18), (cxp, cyp), inner, thick)

                # pupil marker: make it swirl faster when flux is high
                a = -math.pi/2 + phase * (1.0 + 0.35*flux_drive)
                px = cxp + int(((inner + outer) // 2) * math.cos(a))
                py = cyp + int(((inner + outer) // 2) * math.sin(a))
                pygame.draw.circle(screen, (255, 220, 175), (px, py), 6)

                # text overlay
                font = pygame.font.SysFont('Menlo', 15)
                txt = (
                    f"t={t:6.2f}s  rms={rms_n:4.2f}  flux={flux_n:4.2f}  "
                    f"cent={cent_n:4.2f}  ent={ent_n:4.2f}  unc={unc:4.2f}"
                )
                surf = font.render(txt, True, (235, 235, 240))
                screen.blit(surf, (14, 14))

                pygame.display.flip()
                clock.tick(60)

    except sd.CallbackStop:
        pass
    finally:
        pygame.quit()

if __name__ == '__main__':
    main()
