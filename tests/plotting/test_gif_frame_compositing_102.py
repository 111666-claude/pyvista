"""Regression tests for GIF frame compositing around transparent screenshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import pytest

import pyvista as pv

if TYPE_CHECKING:
    from pathlib import Path


def _red_pixel_count(image: np.ndarray, x_slice: slice) -> int:
    """Count saturated red ball pixels in the selected columns."""
    region = image[:, x_slice, :3].astype(np.int16)
    red = (region[..., 0] > 150) & (region[..., 1] < 110) & (region[..., 2] < 110)
    return int(red.sum())


def _two_frame_scene(path: Path, *, screenshot_background, frame_background=None):
    """Write a left-then-right red ball GIF and return its frames and a live image."""
    pl = pv.Plotter(window_size=(200, 200))
    pl.set_background('white')
    actor = pl.add_mesh(pv.Sphere(radius=0.4), color='red', lighting=False)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 1.5
    actor.position = (-1.1, 0.0, 0.0)

    frame_kwargs = {} if frame_background is None else {'transparent_background': frame_background}
    pl.open_gif(path, fps=1)
    pl.write_frame(**frame_kwargs)
    # An auxiliary transparent screenshot between the two frames must not change
    # how the rest of the GIF is composited.
    pl.screenshot(transparent_background=screenshot_background)

    actor.position = (1.1, 0.0, 0.0)
    pl.write_frame(**frame_kwargs)
    live = pl.screenshot()
    pl.close()

    return imageio.mimread(path), live


@pytest.mark.parametrize('screenshot_background', [True, 'straight', 'premultiplied'])
def test_gif_frame_has_no_ghost_after_transparent_screenshot(
    tmp_path: Path, screenshot_background
):
    """A moved actor leaves no residue and frame two matches the live scene."""
    frames, live = _two_frame_scene(
        tmp_path / 'ghost.gif', screenshot_background=screenshot_background
    )

    assert len(frames) == 2
    first, second = frames
    assert first.shape[:2] == (200, 200)
    assert second.shape[:2] == (200, 200)

    width = second.shape[1]
    left = slice(0, width // 3)
    right = slice(2 * width // 3, width)

    assert _red_pixel_count(first, left) > 500
    assert _red_pixel_count(first, right) == 0

    # The previous left ball must not survive in the second frame.
    assert _red_pixel_count(second, left) == 0
    assert _red_pixel_count(second, right) > 500

    # Region-wise the decoded second frame matches the live screenshot.
    second_rgb = second[..., :3].astype(np.int16)
    live_rgb = live[..., :3].astype(np.int16)
    left_difference = np.abs(second_rgb[:, left] - live_rgb[:, left]).mean()
    assert left_difference < 5.0
    full_difference = np.abs(second_rgb - live_rgb).mean()
    assert full_difference < 10.0


@pytest.mark.parametrize('transparent_background', [True, 'straight', 'premultiplied'])
def test_transparent_gif_animation_has_no_ghost(tmp_path: Path, transparent_background):
    """Transparent movies keep their alpha and do not composite over prior frames."""
    frames, _ = _two_frame_scene(
        tmp_path / 'transparent.gif',
        screenshot_background=False,
        frame_background=transparent_background,
    )

    second = frames[1]
    assert second.shape[-1] == 4
    width = second.shape[1]
    left = slice(0, width // 3)
    right = slice(2 * width // 3, width)

    assert _red_pixel_count(second, left) == 0
    assert _red_pixel_count(second, right) > 500
    # The vacated left region is fully transparent rather than a retained ball.
    assert second[:, left, 3].max() == 0


def test_explicit_transparent_background_is_single_frame(mocker):
    """An explicit per-frame value must not change later default frames."""
    pl = pv.Plotter(window_size=(80, 80))
    pl.add_mesh(pv.Sphere())
    writer = mocker.Mock()
    mocker.patch.object(pv.Plotter, '_get_mwriter_not_none', return_value=writer)
    pl._frame_transparent_background = False

    def written_channels(**kwargs):
        pl.write_frame(**kwargs)
        return writer.append_data.call_args.args[0].shape[-1]

    assert written_channels() == 3
    assert written_channels(transparent_background=True) == 4
    # The override applies to that frame only; the movie default stays opaque.
    assert written_channels() == 3
    pl.close()
