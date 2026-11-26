"""Typing cases for :meth:`pyvista.plotting.widgets.WidgetComponent.add_volume_clip_box`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

if TYPE_CHECKING:
    from pyvista.plotting.widgets import WidgetComponent


def a_component() -> WidgetComponent:
    """Return the widget component of a plotter."""
    return pv.Plotter().widgets


def a_volume() -> pv.ImageData:
    """Return a grid with scalars to render as a volume."""
    grid = pv.ImageData(dimensions=(10, 10, 10))
    grid.point_data['scalars'] = np.linspace(0, 255, grid.n_points).astype(np.uint8)
    return grid


assert_types(a_component().add_volume_clip_box(a_volume()), _vtk.vtkBoxWidget)
assert_types(a_component().add_volume_clip_box(a_volume(), invert=True), _vtk.vtkBoxWidget)
