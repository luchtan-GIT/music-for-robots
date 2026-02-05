"""Online predictors for learnability / unpredictability probes.

We want to operationalize: "how many listens until a bounded model predicts what's next?"

These predictors are intentionally small and fast:
- online standardization (running mean/var)
- online linear next-step prediction with SGD + L2

They are NOT intended as state-of-the-art audio models; they're instruments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class OnlineStandardizer:
    """Per-dimension running standardizer using Welford updates."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = int(dim)
        self.eps = float(eps)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.M2 = np.zeros(self.dim, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape}")
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim, dtype=np.float32)
        var = self.M2 / max(1, (self.n - 1))
        return np.sqrt(np.maximum(var, self.eps)).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std()


@dataclass
class OnlineLinearNextStep:
    """Online multi-output linear predictor for y[t] from a window of past y.

    Model: y_hat = x @ W
      - x: concat([y[t-1], ..., y[t-k], 1])  shape: (k*D + 1,)
      - W: shape: (k*D + 1, D)

    Updates via SGD on squared error with L2 weight decay.
    """

    dim: int
    k: int = 4
    lr: float = 0.02
    l2: float = 1e-4
    seed: int = 0

    def __post_init__(self):
        self.dim = int(self.dim)
        self.k = int(max(1, self.k))
        rng = np.random.default_rng(self.seed)
        self.W = (0.01 * rng.standard_normal((self.k * self.dim + 1, self.dim))).astype(np.float32)

    def predict(self, past: list[np.ndarray]) -> np.ndarray:
        if len(past) != self.k:
            raise ValueError(f"need {self.k} past frames")
        x = np.concatenate([np.asarray(p, np.float32) for p in past] + [np.array([1.0], np.float32)], axis=0)
        return (x @ self.W).astype(np.float32)

    def update(self, past: list[np.ndarray], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One online update step with basic numerical stabilization."""
        y = np.asarray(target, np.float32)
        x = np.concatenate([np.asarray(p, np.float32) for p in past] + [np.array([1.0], np.float32)], axis=0)

        # Clip inputs to avoid runaway gradients on early steps.
        x = np.clip(x, -6.0, 6.0).astype(np.float32)

        # Predict in float64 for stability, then cast.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            yhat64 = x.astype(np.float64) @ self.W.astype(np.float64)
        if not np.all(np.isfinite(yhat64)):
            # If we blew up, don't update; return a big clipped error.
            yhat = np.zeros_like(y, dtype=np.float32)
            err = np.clip(y - yhat, -6.0, 6.0).astype(np.float32)
            return yhat, err

        yhat = yhat64.astype(np.float32)
        err = (y - yhat).astype(np.float32)

        # Clip error to prevent blow-ups.
        err = np.clip(err, -6.0, 6.0).astype(np.float32)

        # SGD update with gradient clipping.
        grad = (x[:, None] * err[None, :]).astype(np.float32)
        gnorm = float(np.linalg.norm(grad))
        max_g = 80.0
        if gnorm > max_g and gnorm > 0:
            grad *= (max_g / gnorm)

        if np.all(np.isfinite(grad)) and np.all(np.isfinite(self.W)):
            self.W += self.lr * grad
            self.W *= (1.0 - self.lr * self.l2)

        return yhat, err


def online_learning_curve(
    Y: np.ndarray,
    *,
    k: int = 4,
    lr: float = 0.02,
    l2: float = 1e-4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run an online next-step predictor over Y (T x D).

    Returns:
      - err_rmse_t: (T,) RMSE in standardized space (0..~)
      - yhat: (T, D) predictions (zeros for first k frames)

    Standardization is online (running stats), to prevent scale from dominating.
    """

    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError("Y must be (T, D)")
    T, D = Y.shape

    stdzr = OnlineStandardizer(D)
    model = OnlineLinearNextStep(dim=D, k=k, lr=lr, l2=l2, seed=seed)

    yhat = np.zeros((T, D), dtype=np.float32)
    err_rmse_t = np.zeros(T, dtype=np.float32)

    # Prime standardizer with first few frames (no learning yet)
    for t in range(min(T, k + 1)):
        stdzr.update(Y[t])

    for t in range(k, T - 1):
        # update standardizer with current target too (online)
        stdzr.update(Y[t])
        stdzr.update(Y[t + 1])

        past = [stdzr.transform(Y[t - i]) for i in range(1, k + 1)]
        target = stdzr.transform(Y[t + 1])

        pred, err = model.update(past, target)
        yhat[t + 1] = pred
        err_rmse_t[t + 1] = float(np.sqrt(np.mean(err * err)))

    return {"err_rmse_t": err_rmse_t, "yhat": yhat}
