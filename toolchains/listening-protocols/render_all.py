"""Render all listening protocols for a given audio file.

This is a convenience wrapper for quickly generating a set of artifacts from one song.

Example
-------
```bash
python3 toolchains/listening-protocols/render_all.py \
  --audio song.wav \
  --outdir out_protocols \
  --size 720 --fps 30

# Render only a 20s excerpt starting at 30s:
python3 toolchains/listening-protocols/render_all.py \
  --audio song.wav \
  --outdir out_protocols \
  --start 30 --duration 20
```
"""

from __future__ import annotations

import argparse
import os

# Support both module-run and script-run.
try:
    from .compression_loom import render as render_loom
    from .recurrence_constellation import render as render_const
    from .mirror_residue import render as render_res
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(os.path.dirname(__file__))
    from compression_loom import render as render_loom
    from recurrence_constellation import render as render_const
    from mirror_residue import render as render_res


def main():
    ap = argparse.ArgumentParser(description="Render all listening protocols")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=22, help="base seed; protocols use seed+offset")
    ap.add_argument("--start", type=float, default=None, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="duration in seconds")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    render_loom(
        args.audio,
        os.path.join(args.outdir, "compression_loom.mp4"),
        size=args.size,
        fps=args.fps,
        seed=args.seed + 0,
        start=args.start,
        duration=args.duration,
    )

    render_const(
        args.audio,
        os.path.join(args.outdir, "recurrence_constellation.mp4"),
        size=args.size,
        fps=args.fps,
        seed=args.seed + 1,
        start=args.start,
        duration=args.duration,
    )

    render_res(
        args.audio,
        os.path.join(args.outdir, "mirror_residue.mp4"),
        size=args.size,
        fps=args.fps,
        seed=args.seed + 2,
        start=args.start,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
