"""Shared audio feature extraction for listening-protocol renderers.

Design goals:
- **Time-align** features to the target render FPS.
- Keep it **lightweight**: only what we need for robust control signals.
- Produce signals normalized to [0, 1] so renderers can be parameterized cleanly.

This module is intentionally self-contained so people can copy it into their own projects.
"""

from __future__ import annotations

import numpy as np
import librosa


def norm01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize an array to [0,1]. If constant, return zeros."""
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def smooth1d(x: np.ndarray, win: int = 9) -> np.ndarray:
    """Hann-window smoothing for 1D signals."""
    win = int(max(3, win))
    if win % 2 == 0:
        win += 1
    k = np.hanning(win).astype(np.float32)
    k /= np.sum(k)
    return np.convolve(np.asarray(x, dtype=np.float32), k, mode="same").astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine similarity of two 1D vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_tracks(
    audio_path: str,
    fps: int = 30,
    sr: int = 48_000,
    start: float | None = None,
    duration: float | None = None,
    *,
    rec_history_sec: float = 45.0,
    return_embedding: bool = False,
    return_mel: bool = False,
) -> dict[str, np.ndarray]:
    """Compute per-frame control tracks aligned to `fps`.

    Returns a dict of 1D numpy arrays (length = n_frames):
      - rms, flux, centroid, bandwidth, entropy
      - novelty (smoothed)
      - rec (recurrence score to rolling history)

    Parameters
    - rec_history_sec: history window (seconds) for recurrence. Larger values
      emphasize "has this state occurred anywhere recently?" over local echo.
    - return_embedding: when True, include the per-frame embedding matrix
      (key: "emb", shape: [T, D]) for downstream analysis (e.g., cross-loop).
    - return_mel: when True, include the per-frame mel dB matrix
      (key: "mel_db", shape: [T, n_mels]) for waveform-ish next-frame prediction.

    Notes:
    - The hop length is `sr/fps` so one feature frame ≈ one video frame.
    - Recurrence uses an embedding of MFCC+chroma+basic scalars and compares
      to a rolling history window.
    """

    # `start` and `duration` are in seconds.
    # This trims the analysis window while keeping feature alignment simple.
    y, sr = librosa.load(
        audio_path,
        sr=sr,
        mono=True,
        offset=float(start) if start is not None else 0.0,
        duration=float(duration) if duration is not None else None,
    )

    hop = int(sr / fps)
    n_fft = 2048

    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))

    # Mel power in dB for MFCCs (stable across sources)
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel + 1e-9)

    rms = librosa.feature.rms(S=S)[0]
    flux = librosa.onset.onset_strength(S=S, sr=sr)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

    chroma = librosa.feature.chroma_stft(S=S, sr=sr).T
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13).T

    # Entropy proxy: distributional spread of spectral energy per frame.
    p = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    entropy = (-np.sum(p * np.log(p + 1e-9), axis=0)).astype(np.float32)

    rms = norm01(rms)
    flux = norm01(flux)
    centroid = norm01(centroid)
    bandwidth = norm01(bandwidth)
    entropy = norm01(entropy)

    # Embedding for recurrence: we want a stable summary of "audio state".
    emb = np.concatenate(
        [
            norm01(mfcc),
            norm01(chroma),
            flux[:, None],
            centroid[:, None],
            bandwidth[:, None],
            entropy[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    # Novelty: delta in embedding + flux.
    d = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
    novelty = np.concatenate([[0.0], d]).astype(np.float32)
    novelty = norm01(novelty + 0.6 * flux)
    novelty = smooth1d(novelty, 11)

    # Recurrence: similarity to rolling history.
    # IMPORTANT: using a raw max tends to saturate in stable textures.
    # We compute both:
    #   - rec_max: max similarity to history (legacy behavior)
    #   - rec_p95: 95th-percentile similarity to history (less "always-1.0")
    T = emb.shape[0]
    rec_max = np.zeros(T, dtype=np.float32)
    rec_p95 = np.zeros(T, dtype=np.float32)
    history = int(max(1.0, float(rec_history_sec)) * fps)  # seconds → frames

    for t in range(T):
        t0 = max(0, t - history)
        if t - t0 < 3:
            continue
        step = max(1, (t - t0) // 48)  # subsample for speed
        sims = np.array([cosine_sim(emb[t], emb[k]) for k in range(t0, t, step)], dtype=np.float32)
        if sims.size == 0:
            continue
        rec_max[t] = float(np.max(sims))
        rec_p95[t] = float(np.quantile(sims, 0.95))

    # Smooth, then normalize to [0,1] (within this excerpt) for renderer convenience.
    rec_max = norm01(smooth1d(rec_max, 9))
    rec_p95 = norm01(smooth1d(rec_p95, 9))

    # By default, expose the less-saturating recurrence as "rec".
    rec = rec_p95

    # Second-order-ish tracks
    # - novelty_fast: highpassed novelty (captures flicker / micro-events)
    # - habituation: slow EMA of novelty (a crude "getting used to it" state)
    novelty_slow = smooth1d(novelty, int(max(9, (1.2 * fps) // 2 * 2 + 1)))  # ~1.2s window
    novelty_fast = norm01(np.clip(novelty - novelty_slow, 0.0, None))

    # Habituation: slow EMA of novelty. Keep it *unnormalized* so longer runs / more loops
    # can actually accumulate a narrative arc (normalizing per-excerpt destroys that).
    hab_raw = np.zeros_like(novelty, dtype=np.float32)
    a = float(np.clip(1.0 / (fps * 6.0), 0.001, 0.2))  # ~6s time constant
    for t in range(1, T):
        hab_raw[t] = (1 - a) * hab_raw[t - 1] + a * novelty[t]
    hab_raw = smooth1d(hab_raw, 9)

    # Fixed squashing into [0,1] for downstream use (no per-excerpt min/max).
    # This maps "typical" EMA novelty levels (~0.1–0.4) into a useful dynamic range.
    k = 0.25
    hab = (1.0 - np.exp(-np.clip(hab_raw, 0.0, 10.0) / k)).astype(np.float32)

    out = {
        "rms": rms,
        "flux": flux,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "novelty": novelty,
        "novelty_fast": novelty_fast,
        "habituation": hab,
        "habituation_raw": hab_raw,
        # recurrence variants
        "rec": rec,
        "rec_p95": rec_p95,
        "rec_max": rec_max,
        "frames": np.arange(len(rms), dtype=np.int32),
        "sr": np.array([sr], dtype=np.int32),
    }

    if return_embedding:
        out["emb"] = emb

    if return_mel:
        # Use time-major mel dB for online next-frame prediction.
        out["mel_db"] = mel_db.T.astype(np.float32)

    return out
