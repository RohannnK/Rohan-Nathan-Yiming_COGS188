"""Optional rendering of episodes to GIF (rware renderer). Needs pyglet 1.x and a display; never import in tests.

The saboteur's cell is outlined in red, since rware draws every robot the same way.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from ..eval.rollout import run_episode

CELL = 31  # rware Viewer: 30 px cells plus 1 px grid lines


def outline(frame: np.ndarray, x: int, y: int, cols: int, colour=(220, 30, 30)) -> np.ndarray:
    """Draw a box around grid cell (x, y). Frames may be HiDPI-scaled (2x on Retina Macs)."""
    s = frame.shape[1] // (1 + cols * CELL)
    frame = frame.copy()
    top, left, size = s * (CELL * y + 1), s * (CELL * x + 1), s * (CELL - 1)
    for d in range(2 * s):
        frame[top + d, left:left + size] = colour
        frame[top + size - 1 - d, left:left + size] = colour
        frame[top:top + size, left + d] = colour
        frame[top:top + size, left + size - 1 - d] = colour
    return frame


def record_gif(spec: dict, seed: int, path: Path | str, every: int = 2, max_frames: int = 150, fps: int = 8) -> Path:
    """Run one episode, keep every `every`-th frame (at most `max_frames`) and save an infinitely looping GIF."""
    frames, step = [], [0]

    def on_step(env):
        if step[0] % every == 0 and len(frames) < max_frames:
            w = env.unwrapped
            frame = w.render(mode="rgb_array")
            # pyglet 1.5 on macOS raises when a window is closed outside its event loop; leave it to process exit.
            w.renderer.close = lambda: None
            if env.saboteur_id is not None:
                a = w.agents[env.saboteur_id]
                frame = outline(frame, a.x, a.y, w.grid_size[1])
            hidpi = frame.shape[1] // (1 + w.grid_size[1] * CELL)
            frames.append(frame[::hidpi, ::hidpi])
        step[0] += 1

    run_episode(spec, seed, on_step)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, duration=1000 / fps, loop=0)
    return path
