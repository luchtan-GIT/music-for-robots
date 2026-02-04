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


def extract_tracks(audio_path: str, fps: int = 30, sr: int = 48_000) -> dict[str, np.ndarray]:
    """Compute per-frame control tracks aligned to `fps`.

    Returns a dict of 1D numpy arrays (length = n_frames):
      - rms, flux, centroid, bandwidth, entropy
      - novelty (smoothed)
      - rec (recurrence score to rolling history)

    Notes:
    - The hop length is `sr/fps` so one feature frame ≈ one video frame.
    - Recurrence uses an embedding of MFCC+chroma+basic scalars and compares
      to a rolling history window.
    """

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

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

    # Recurrence: max cosine similarity to rolling history.
    T = emb.shape[0]
    rec = np.zeros(T, dtype=np.float32)
    history = int(8 * fps)  # ~8 seconds of history
    for t in range(T):
        t0 = max(0, t - history)
        if t - t0 < 3:
            continue
        step = max(1, (t - t0) // 32)  # subsample for speed
        sims = [cosine_sim(emb[t], emb[k]) for k in range(t0, t, step)]
        rec[t] = max(sims) if sims else 0.0
    rec = norm01(smooth1d(rec, 9))

    return {
        "rms": rms,
        "flux": flux,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "entropy": entropy,
        "novelty": novelty,
        "rec": rec,
        "frames": np.arange(len(rms), dtype=np.int32),
        "sr": np.array([sr], dtype=np.int32),
    }
