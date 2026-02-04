"""Audio mux helper.

We render video frames to a silent MP4 and then mux the original audio into the final MP4.
This keeps the render code simple and avoids codec issues.

Requires: ffmpeg on PATH.
"""

from __future__ import annotations

import os
import subprocess


def mux_audio(video_silent: str, audio_path: str, out_path: str, delete_silent: bool = True) -> None:
    """Mux `audio_path` into `video_silent` and write `out_path`.

    - Copies the video stream (no re-encode).
    - Encodes audio to AAC for wide compatibility.
    - Uses `-shortest` so output ends when the shorter stream ends.
    """

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_silent,
        "-i",
        audio_path,
        "-shortest",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_path,
    ]
    subprocess.check_call(cmd)

    if delete_silent:
        try:
            os.remove(video_silent)
        except FileNotFoundError:
            pass
