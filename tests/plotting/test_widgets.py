from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Literal
from unittest.mock import ANY
import warnings

import numpy as np
import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista import examples
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.plotting import widgets
from pyvista.plotting.affine_widget import DARK_YELLOW
from pyvista.plotting.affine_widget import get_angle
from pyvista.plotting.affine_widget import ray_plane_intersection
from tests.conftest import flaky_test

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# skip all tests if unable to render
pytestmark = pytest.mark.skip_plotting


def r_mat_to_euler_angles(r):
    """Extract Euler angles from a 3x3 rotation matrix using the ZYX sequence.

    Returns the angles in radians.
    """
    # Check for gimbal lock: singular cases
    if abs(r[2, 0]) == 1:
        # Gimbal lock exists
        yaw = 0  # Set yaw to 0 and compute the others
        if r[2, 0] == -1:  # Directly looking up
            pitch = np.pi / 2
            roll = yaw + np.arctan2(r[0, 1], r[0, 2])
        else:  # Directly looking down
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-r[0, 1], -r[0, 2])
    else:
        # Not at the singularities
        pitch = -np.arcsin(r[2, 0])
        roll = np.arctan2(r[2, 1] / np.cos(pitch), r[2, 2] / np.cos(pitch))
        yaw = np.arctan2(r[1, 0] / np.cos(pitch), r[0, 0] / np.cos(pitch))

    if np.isclose(roll, np.pi):
        roll = 0
    if np.isclose(pitch, np.pi):
        pitch = 0
    if np.isclose(yaw, np.pi):
        yaw = 0

    return roll, pitch, yaw


def test_add_plane_widget_raises():
    pl = pv.Plotter()
    with pytest.raises(RuntimeError, match='assign_to_axis not understood'):
        pl.add_plane_widget(lambda *x: True, assign_to_axis='foo')  # noqa: ARG005


def test_add_slider_widget_raises():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RuntimeError, match=r'Cannot add a widget to a closed plotter.'):
        pl.add_slider_widget(lambda *x: True, rng=[0, 1])  # noqa: ARG005


def test_add_mesh_threshold_raises():
    pl = pv.Plotter()
    with pytest.raises(
        TypeError, match=r'MultiBlock datasets are not supported for threshold widget.'
    ):
        pl.add_mesh_threshold(mesh=pv.MultiBlock())

    pl = pv.Plotter()
    with pytest.raises(ValueError, match=r'No arrays present to threshold.'):
        pl.add_mesh_threshold(mesh=pv.PolyData())


def test_add_mesh_isovalue_raises():
    pl = pv.Plotter()
    with pytest.raises(TypeError, match=r'MultiBlock datasets are not supported for this widget.'):
        pl.add_mesh_isovalue(mesh=pv.MultiBlock())

    pl = pv.Plotter()
    with pytest.raises(
        ValueError, match=r'Input dataset for the contour filter must have data arrays.'
    ):
        pl.add_mesh_isovalue(mesh=pv.PolyData())

    pl = pv.Plotter()
    sp = pv.Sphere()
    sp.cell_data['foo'] = np.ones(sp.n_cells)
    match = re.escape('Contour filter only works on Point data. Array (foo) is in the Cell data.')
    with pytest.raises(TypeError, match=match):
        pl.add_mesh_isovalue(mesh=sp, scalars='foo')


def test_add_mesh_isovalue_without_active_scalars_raises():
    mesh = pv.Sphere()
    mesh.point_data['foo'] = np.ones(mesh.n_points)
    mesh.set_active_scalars(None)
    assert mesh.n_arrays > 0
    assert mesh.active_scalars_info.name is None

    pl = pv.Plotter()
    match = re.escape('No active scalars to contour. Set `scalars` explicitly.')
    with pytest.raises(ValueError, match=match):
        pl.add_mesh_isovalue(mesh=mesh)
    pl.close()


def test_add_mesh_isovalue_pointset_raises():
    pl = pv.Plotter()
    with pytest.raises(
        TypeError, match=r'PointSets are 0-dimensional and thus cannot produce contours.'
    ):
        pl.add_mesh_isovalue(mesh=pv.PointSet())


def test_add_measurement_widget_raises():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RuntimeError, match=r'Cannot add a widget to a closed plotter.'):
        pl.add_measurement_widget()


def test_widget_box(uniform):
    pl = pv.Plotter()
    func = lambda box: box  # Does nothing
    pl.add_mesh(uniform)
    pl.add_box_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    func = lambda box, widget: box  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_box_widget(callback=func, pass_widget=True)
    pl.close()

    # clip box with and without crinkle
    pl = pv.Plotter()
    pl.add_mesh_clip_box(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_box(uniform, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    # merge_points=True is the default and is tested above
    pl.add_mesh_clip_box(uniform, merge_points=False)
    pl.close()


def test_widget_plane(uniform):
    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, pass_widget=True, implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, implicit=False)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin, widget: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, pass_widget=True, implicit=False)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, assign_to_axis='z', implicit=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=func, normal_rotation=False, implicit=False)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(uniform, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(uniform)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_orthogonal(uniform)
    pl.close()


def test_widget_line(uniform):
    pl = pv.Plotter()
    func = lambda line: line  # Does nothing
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    func = lambda line, widget: line  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func, pass_widget=True)
    pl.close()

    pl = pv.Plotter()
    func = lambda a, b: (a, b)  # Does nothing
    pl.add_mesh(uniform)
    pl.add_line_widget(callback=func, use_vertices=True)
    pl.close()


def test_widget_text_slider(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    with pytest.raises(TypeError, match='must be a list'):
        pl.add_text_slider_widget(callback=func, data='foo')
    with pytest.raises(ValueError, match='list of values is empty'):
        pl.add_text_slider_widget(callback=func, data=[])
    for style in pv.global_theme.slider_styles:
        pl.add_text_slider_widget(callback=func, data=['foo', 'bar'], style=style)
    pl.close()


def test_widget_slider(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    pl.add_slider_widget(callback=func, rng=[0, 10], style='classic')
    pl.close()

    pl = pv.Plotter()
    for interaction_event in ['start', 'end', 'always']:
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event=interaction_event)
    with pytest.raises(TypeError, match='type for ``style``'):
        pl.add_slider_widget(callback=func, rng=[0, 10], style=0)
    with pytest.raises(AttributeError):
        pl.add_slider_widget(callback=func, rng=[0, 10], style='foo')
    with pytest.raises(TypeError, match='Expected type for `interaction_event`'):
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event=0)
    with pytest.raises(ValueError, match='Expected value for `interaction_event`'):
        pl.add_slider_widget(callback=func, rng=[0, 10], interaction_event='foo')
    pl.close()

    pl = pv.Plotter()
    func = lambda value, widget: value  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_slider_widget(callback=func, rng=[0, 10], style='modern', pass_widget=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(uniform, invert=True)
    pl.add_mesh(uniform.outline())
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(uniform, invert=False)
    pl.add_mesh(uniform.outline())
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_isovalue(uniform)
    pl.close()

    func = lambda value: value  # Does nothing
    pl = pv.Plotter()
    title_height = np.random.default_rng().random()
    s = pl.add_slider_widget(
        callback=func, rng=[0, 10], style='classic', title_height=title_height
    )
    assert s.GetRepresentation().GetTitleHeight() == title_height
    pl.close()

    pl = pv.Plotter()
    title_opacity = np.random.default_rng().random()
    s = pl.add_slider_widget(
        callback=func,
        rng=[0, 10],
        style='classic',
        title_opacity=title_opacity,
    )
    assert s.GetRepresentation().GetTitleProperty().GetOpacity() == title_opacity
    pl.close()

    pl = pv.Plotter()
    title_color = 'red'
    s = pl.add_slider_widget(callback=func, rng=[0, 10], style='classic', title_color=title_color)
    assert s.GetRepresentation().GetTitleProperty().GetColor() == pv.Color(title_color)
    pl.close()

    pl = pv.Plotter()
    fmt = '%0.9f'
    s = pl.add_slider_widget(callback=func, rng=[0, 10], style='classic', fmt=fmt)
    assert s.GetRepresentation().GetLabelFormat() == fmt
    pl.close()

    # custom width
    pl = pv.Plotter()
    slider = pl.add_slider_widget(
        callback=func,
        rng=[0, 10],
        fmt=fmt,
        tube_width=0.1,
        slider_width=0.2,
    )
    assert slider.GetRepresentation().GetSliderWidth() == 0.2
    assert slider.GetRepresentation().GetTubeWidth() == 0.1
    pl.close()


def test_widget_spline(uniform):
    pl = pv.Plotter()
    func = lambda spline: spline  # Does nothing
    pl.add_mesh(uniform)
    pl.add_spline_widget(callback=func)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pts = np.array([[1, 5, 4], [2, 4, 9], [3, 6, 2]])
    with pytest.raises(ValueError, match='`initial_points` must be length `n_handles`'):
        pl.add_spline_widget(callback=func, n_handles=4, initial_points=pts)
    pl.add_spline_widget(callback=func, n_handles=3, initial_points=pts)
    pl.close()

    pl = pv.Plotter()
    func = lambda spline, widget: spline  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    pl.add_spline_widget(callback=func, pass_widget=True, color=None, show_ribbon=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(uniform)
    pl.close()


def test_measurement_widget(random_hills):
    class DistanceCallback:
        def __init__(self):
            self.called = False
            self.args = None
            self.count = 0

        def __call__(self, *args, **kwargs):
            self.called = True
            self.args = args
            self.kwargs = kwargs
            self.count += 1

    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_mesh(random_hills)
    distance_callback = DistanceCallback()
    pl.add_measurement_widget(callback=distance_callback)
    pl.view_xy()
    pl.show(auto_close=False)
    _width, _height = pl.window_size

    pl.iren._mouse_left_button_click(300, 300)
    pl.iren._mouse_left_button_click(700, 700)

    assert distance_callback.called
    assert pytest.approx(distance_callback.args[2], 1.0) == 17.4

    pl.close()


def test_widget_sphere():
    pl = pv.Plotter()
    func = lambda center: center  # Does nothing
    pl.add_sphere_widget(callback=func, center=(0, 0, 0))
    pl.close()

    # pass multiple centers
    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    pl = pv.Plotter()
    func = lambda center, index: center  # Does nothing  # noqa: ARG005
    pl.add_sphere_widget(callback=func, center=nodes)
    pl.close()


def test_sphere_widget_returns_only_the_new_widgets():
    pl = pv.Plotter()
    nodes = np.array([[-1, -1, -1], [1, 1, 1]])
    first = pl.add_sphere_widget(None, center=nodes)
    second = pl.add_sphere_widget(None, center=nodes)

    assert len(first) == 2
    assert len(second) == 2
    assert not set(map(id, first)) & set(map(id, second))
    pl.close()


def test_volume_clip_plane_adds_one_widget(uniform):
    pl = pv.Plotter()
    pl.add_volume_clip_plane(uniform)

    assert len(pl.widgets.plane_widgets) == 1
    pl.close()


def test_sphere_widget_style_is_matched_exactly():
    pl = pv.Plotter()
    surface = pl.add_sphere_widget(None, style='surface')
    wireframe = pl.add_sphere_widget(None, style='wireframe')

    assert surface.GetSphereProperty().GetRepresentationAsString() == 'Surface'
    assert wireframe.GetSphereProperty().GetRepresentationAsString() == 'Wireframe'

    for style in ['wire', '']:
        match = re.escape(f'style {style!r} is not valid.')
        with pytest.raises(ValueError, match=match):
            pl.add_sphere_widget(None, style=style)
    pl.close()


def test_sphere_widget_color_sequence_is_applied_per_widget():
    nodes = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    pl = pv.Plotter()
    sphere_widgets = pl.add_sphere_widget(None, center=nodes, color=['red', 'blue'])

    colors = [widget.GetSphereProperty().GetColor() for widget in sphere_widgets]
    assert colors == [pv.Color('red').float_rgb, pv.Color('blue').float_rgb]
    pl.close()


def test_sphere_widget_single_color_is_shared_by_every_widget():
    nodes = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    yellow = pv.Color('yellow').float_rgb
    pl = pv.Plotter()
    sphere_widgets = pl.add_sphere_widget(None, center=nodes, color='yellow')
    single = pl.add_sphere_widget(None, color='yellow')

    colors = [widget.GetSphereProperty().GetColor() for widget in sphere_widgets]
    assert colors == [yellow, yellow]
    assert single.GetSphereProperty().GetColor() == yellow
    pl.close()


@pytest.mark.parametrize('color', [[255, 0, 0], [1.0, 0.0, 0.0], 'red'])
def test_sphere_widget_single_color_is_not_split_across_widgets(color):
    nodes = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    red = pv.Color('red').float_rgb
    pl = pv.Plotter()

    sphere_widgets = pl.add_sphere_widget(None, center=nodes, color=color)

    assert [widget.GetSphereProperty().GetColor() for widget in sphere_widgets] == [red] * 3
    pl.close()


def test_sphere_widget_color_sequence_length_must_match():
    nodes = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    pl = pv.Plotter()

    with pytest.raises(ValueError, match='Invalid color'):
        pl.add_sphere_widget(None, center=nodes, color=['red', 'green', 'blue'])
    pl.close()


def test_widget_checkbox_button(uniform):
    pl = pv.Plotter()
    func = lambda value: value  # Does nothing
    pl.add_mesh(uniform)
    pl.add_checkbox_button_widget(callback=func)
    pl.close()


def test_widget_checkbox_button_click_passes_the_new_state(uniform):
    states = []
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    size = 50
    position = (10.0, 10.0)
    button = pl.add_checkbox_button_widget(
        callback=states.append, value=False, size=size, position=position
    )
    pl.show(auto_close=False)

    center = (int(position[0] + size / 2), int(position[1] + size / 2))
    pl.iren._mouse_left_button_click(*center)
    assert button.GetRepresentation().GetState() == 1
    pl.iren._mouse_left_button_click(*center)
    assert button.GetRepresentation().GetState() == 0

    assert states == [True, False]
    pl.close()


def test_widget_checkbox_button_click_without_callback(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    size = 50
    position = (10.0, 10.0)
    button = pl.add_checkbox_button_widget(None, value=False, size=size, position=position)
    pl.show(auto_close=False)

    center = (int(position[0] + size / 2), int(position[1] + size / 2))
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter('always')
        pl.iren._mouse_left_button_click(*center)

    assert not [w for w in log if 'Encountered issue in callback' in str(w.message)]
    assert button.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_closed(uniform):
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_checkbox_button_widget(callback=lambda value: value)


def test_close_with_widgets_does_not_render(uniform):
    """Widget teardown during ``close`` must not trigger renders."""
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.add_plane_widget(callback=lambda normal, origin: None)  # noqa: ARG005
    pl.show(auto_close=False)
    render_starts = []
    pl.render_window.AddObserver('StartEvent', lambda *_: render_starts.append(1))
    pl.close()
    assert not render_starts


def test_widget_radio_button(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    b = pl.add_radio_button_widget(callback=func, radio_button_group='group')
    assert pl.widgets.radio_button_widget_dict['group'][0] == b
    pl.close()
    assert 'group' not in pl.widgets.radio_button_widget_dict


def test_widget_radio_button_click(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    size = 50
    position = (10.0, 10.0)
    b = pl.add_radio_button_widget(
        callback=func,
        radio_button_group='group',
        value=False,
        size=size,
        position=position,
    )
    pl.show(auto_close=False)
    # Test switching logic
    b_center = (int(position[0] + size / 2), int(position[1] + size / 2))
    assert b.GetRepresentation().GetState() == 0
    pl.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    pl.iren._mouse_left_button_click(*b_center)
    assert b.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_with_title(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    pl.add_radio_button_widget(callback=func, radio_button_group='group', title='my_button')
    assert len(pl.widgets.radio_button_title_dict['group']) == 1
    pl.close()
    assert 'group' not in pl.widgets.radio_button_title_dict


def test_widget_radio_button_multiple_on(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    b1 = pl.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    b2 = pl.add_radio_button_widget(callback=func, radio_button_group='group', value=True)
    assert len(pl.widgets.radio_button_widget_dict['group']) == 2
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_multiple_switch(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    size = 50
    b1_position = (10.0, 10.0)
    b2_position = (10.0, 70.0)
    b1 = pl.add_radio_button_widget(
        callback=func,
        radio_button_group='group',
        value=True,
        size=size,
        position=b1_position,
    )
    b2 = pl.add_radio_button_widget(
        callback=func, radio_button_group='group', size=size, position=b2_position
    )
    pl.show(auto_close=False)
    # Click b2 and switch active radio button
    b2_center = (int(b2_position[0] + size / 2), int(b2_position[1] + size / 2))
    pl.iren._mouse_left_button_click(*b2_center)
    assert b1.GetRepresentation().GetState() == 0
    assert b2.GetRepresentation().GetState() == 1
    pl.close()


def test_widget_radio_button_plotter_closed(uniform):
    pl = pv.Plotter()
    func = lambda: None  # Does nothing
    pl.add_mesh(uniform)
    pl.close()
    with pytest.raises(RuntimeError, match='closed plotter'):
        pl.add_radio_button_widget(callback=func, radio_button_group='group')


def test_add_camera_orientation_widget():
    pl = pv.Plotter()
    pl.add_camera_orientation_widget()
    assert pl.widgets.camera_widgets
    pl.close()
    assert not pl.widgets.camera_widgets


def _assert_camera_widget_viewport(widget, anchor):
    """Check the widget's pixel size and padding within its parent viewport."""
    renderer = widget.GetParentRenderer()
    window_size = np.array(renderer.GetRenderWindow().GetActualSize())
    parent = np.array(renderer.GetViewport()) * np.tile(window_size, 2)
    representation = widget.GetRepresentation()
    size = max(representation.GetSize())
    pad_x, pad_y = representation.GetPadding()
    x = parent[0] + pad_x if anchor.endswith('Left') else parent[2] - pad_x - size
    y = parent[1] + pad_y if anchor.startswith('Lower') else parent[3] - pad_y - size
    actual = np.array(widget.GetCurrentRenderer().GetViewport()) * np.tile(window_size, 2)
    assert np.allclose(actual, (x, y, x + size, y + size))


@pytest.mark.parametrize('shape', [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize('anchor', ['LowerLeft', 'LowerRight', 'UpperLeft', 'UpperRight'])
def test_camera_orientation_widget_subplots(shape, anchor):
    """Keep each widget in its own subplot during native renders and resizes."""
    pl = pv.Plotter(shape=shape, window_size=(800, 600))
    camera_widgets = []
    for index in np.ndindex(shape):
        pl.subplot(*index)
        pl.add_mesh(pv.Cone())
        widget = pl.add_camera_orientation_widget()
        representation = widget.GetRepresentation()
        getattr(representation, f'AnchorTo{anchor}')()
        representation.SetSize(90, 60)
        representation.SetPadding(7, 11)
        camera_widgets.append(widget)
    pl.show(auto_close=False, interactive=False)
    for size in [(800, 600), (600, 800), (1000, 800)]:
        pl.window_size = size
        for _ in range(2):
            pl.render()
            for widget in camera_widgets:
                _assert_camera_widget_viewport(widget, anchor)
    pl.close()


def test_camera_orientation_widget_viewport_changes():
    """Honor changes to the parent viewport and the representation anchor."""
    pl = pv.Plotter(window_size=(800, 600))
    widget = pl.add_camera_orientation_widget()
    pl.show(auto_close=False, interactive=False)
    pl.renderer.viewport = (0.1, 0.2, 0.6, 0.8)
    pl.render()
    _assert_camera_widget_viewport(widget, 'UpperRight')
    widget.GetRepresentation().AnchorToLowerLeft()
    pl.render()
    _assert_camera_widget_viewport(widget, 'LowerLeft')
    pl.close()


@pytest.mark.parametrize('window_size', [(0, 0), (0, 600), (800, 0)])
def test_camera_orientation_widget_zero_window_size(window_size, capsys):
    """Keep the viewport until the render window has a nonzero size."""
    pl = pv.Plotter(window_size=(800, 600))
    widget = pl.add_camera_orientation_widget()
    renderer = widget.GetCurrentRenderer()
    viewport = renderer.GetViewport()
    pl.render_window.SetSize(*window_size)
    renderer.InvokeEvent(_vtk.vtkCommand.StartEvent)
    assert not capsys.readouterr().err
    assert renderer.GetViewport() == viewport
    pl.render_window.SetSize(800, 600)
    renderer.InvokeEvent(_vtk.vtkCommand.StartEvent)
    _assert_camera_widget_viewport(widget, 'UpperRight')
    pl.close()


def test_camera_orientation_widget_clear():
    """Stop updating a retained widget after clearing camera widgets."""
    pl = pv.Plotter(shape=(1, 2))
    widget = pl.add_camera_orientation_widget()
    renderer = widget.GetCurrentRenderer()
    pl.clear_camera_widgets()
    renderer.SetViewport(0, 0, 1, 1)
    renderer.InvokeEvent(_vtk.vtkCommand.StartEvent)
    assert renderer.GetViewport() == (0, 0, 1, 1)
    assert not widget.GetEnabled()
    pl.close()


def test_camera_orientation_widget_detached():
    """Leave the viewport alone after detaching the parent renderer."""
    pl = pv.Plotter()
    widget = pl.add_camera_orientation_widget()
    renderer = widget.GetCurrentRenderer()
    widget.SetParentRenderer(None)
    renderer.SetViewport(0, 0, 1, 1)
    renderer.InvokeEvent(_vtk.vtkCommand.StartEvent)
    assert renderer.GetViewport() == (0, 0, 1, 1)
    pl.close()


def test_camera_orientation_widget_closed_plotter():
    pl = pv.Plotter()
    pl.close()
    with pytest.raises(RuntimeError, match=r'Cannot add a widget to a closed plotter.'):
        pl.add_camera_orientation_widget()


@pytest.mark.usefixtures('verify_image_cache')
def test_camera_orientation_widget_subplots_image():
    """Render independently placed camera widgets over distinct subplot scenes."""
    pl = pv.Plotter(shape=(1, 2), window_size=(400, 400))
    pl.add_mesh(pv.Cone(direction=(1, 2, 1)), color='tomato', show_edges=True)
    left_widget = pl.add_camera_orientation_widget()
    pl.camera_position = 'iso'
    pl.reset_camera()
    pl.subplot(0, 1)
    pl.add_mesh(pv.Arrow(direction=(0.5, 1, 2)), color='royalblue', show_edges=True)
    right_widget = pl.add_camera_orientation_widget()
    pl.camera_position = 'xy'
    pl.reset_camera()
    pl.render_window.Render()
    _assert_camera_widget_viewport(left_widget, 'UpperRight')
    _assert_camera_widget_viewport(right_widget, 'UpperRight')
    pl.show()


def test_plot_algorithm_widgets():
    algo = _vtk.vtkRTAnalyticSource()

    pl = pv.Plotter()
    pl.add_mesh_clip_box(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(algo, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_isovalue(algo)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(algo)
    pl.close()


def test_add_volume_clip_plane(uniform):
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='The `volume` parameter type must'):
        pl.add_volume_clip_plane(pv.Sphere())

    widget = pl.add_volume_clip_plane(uniform)
    assert isinstance(widget, _vtk.vtkImplicitPlaneWidget)
    assert pl.volume.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()

    pl = pv.Plotter()
    vol = pl.add_volume(uniform)
    assert vol.mapper.GetClippingPlanes() is None
    pl.add_volume_clip_plane(vol)
    assert vol.mapper.GetClippingPlanes().GetNumberOfItems() == 1
    pl.close()


def _gradient_volume():
    """Build a small ImageData whose scalars encode the point coordinates."""
    grid = pv.ImageData(dimensions=(8, 8, 8), spacing=(1, 1, 1))
    grid['scalars'] = (grid.x + 10 * grid.y + 100 * grid.z).astype(np.float32)
    return grid


def _box_planes(pl, bounds, *, rotation_z=None):
    """Build the plane collection a box widget produces for the given bounds."""
    box_widget = _vtk.vtkBoxWidget()
    box_widget.SetInteractor(pl.iren.interactor)
    box_widget.SetCurrentRenderer(pl.renderer)
    box_widget.SetPlaceFactor(1.0)
    box_widget.PlaceWidget(bounds)
    box_widget.On()
    if rotation_z is not None:
        transform = _vtk.vtkTransform()
        transform.PostMultiply()
        center = [(bounds[2 * i] + bounds[2 * i + 1]) / 2 for i in range(3)]
        transform.Translate(*center)
        transform.RotateZ(rotation_z)
        transform.Translate(*(-coord for coord in center))
        box_widget.SetTransform(transform)
    planes = _vtk.vtkPlanes()
    box_widget.GetPlanes(planes)
    return planes


def test_box_is_collapsed():
    from pyvista.plotting.widgets import _box_is_collapsed

    pl = pv.Plotter()
    # A normal box and a valid rotated box are not collapsed
    assert not _box_is_collapsed(_box_planes(pl, (1, 4, 1, 4, 1, 4)))
    assert not _box_is_collapsed(_box_planes(pl, (1, 6, 1, 6, 1, 6), rotation_z=60))
    # A flat but valid grid is not collapsed
    assert not _box_is_collapsed(_box_planes(pl, (0, 7, 0, 7, 0, 1)))
    # A fully collapsed box or a single collapsed face is collapsed
    assert _box_is_collapsed(_box_planes(pl, (4, 4, 4, 4, 4, 4)))
    assert _box_is_collapsed(_box_planes(pl, (3, 3, 1, 5, 1, 5)))
    pl.close()


def _clipped_volume(pl):
    """Return the visible dataset rendered by the clipped-volume actor.

    Returns ``None`` when the clipping region is empty and the clipped actor is
    hidden.
    """
    for actor in pl.renderer.actors.values():
        if isinstance(actor, _vtk.vtkVolume) and isinstance(
            actor.mapper, _vtk.vtkOpenGLProjectedTetrahedraMapper
        ):
            if not actor.GetVisibility():
                return None
            dataset = pv.wrap(actor.mapper.GetInput())
            return dataset if dataset.n_cells else None
    return None


def _move_box(widget, bounds):
    """Reposition a box widget and fire its end-of-interaction callback."""
    widget.PlaceWidget(bounds)
    widget.InvokeEvent(_vtk.vtkCommand.EndInteractionEvent)


def test_add_volume_clip_box(uniform):
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(uniform, factor=1.0)
    assert isinstance(widget, _vtk.vtkBoxWidget)
    assert len(pl.widgets.box_widgets) == 1
    assert _clipped_volume(pl) is not None
    pl.close()

    # An existing Volume actor is accepted and keeps rendering through the same prop
    pl = pv.Plotter()
    volume_actor = pl.add_volume(uniform)
    widget = pl.add_volume_clip_box(volume_actor, factor=1.0)
    assert _clipped_volume(pl) is not None
    clipped_actor = next(
        actor
        for actor in pl.renderer.actors.values()
        if isinstance(actor.mapper, _vtk.vtkOpenGLProjectedTetrahedraMapper)
    )
    assert clipped_actor.prop is volume_actor.prop
    pl.close()

    # The `name` keyword carries through to the clipped actor
    pl = pv.Plotter()
    pl.add_volume_clip_box(uniform, factor=1.0, name='clipped-volume')
    named_actor = pl.renderer.actors.get('clipped-volume')
    assert isinstance(named_actor, _vtk.vtkVolume)
    assert isinstance(named_actor.mapper, _vtk.vtkOpenGLProjectedTetrahedraMapper)
    pl.close()


def test_add_volume_clip_box_scalar_bar():
    grid = _gradient_volume()

    # The scalar bar created with the volume survives the mapper replacement
    pl = pv.Plotter()
    pl.add_volume_clip_box(grid, factor=1.0)
    assert len(pl.scalar_bars._scalar_bar_mappers) == 1
    clipped_mapper = next(
        actor.mapper
        for actor in pl.renderer.actors.values()
        if isinstance(actor.mapper, _vtk.vtkOpenGLProjectedTetrahedraMapper)
    )
    registered = next(iter(pl.scalar_bars._scalar_bar_mappers.values()))
    assert registered == [clipped_mapper]
    assert any(isinstance(actor, _vtk.vtkScalarBarActor) for actor in pl.renderer.actors.values())
    pl.close()

    # Disabling the scalar bar keeps none after the swap
    pl = pv.Plotter()
    pl.add_volume_clip_box(grid, factor=1.0, show_scalar_bar=False)
    assert len(pl.scalar_bars._scalar_bar_mappers) == 0
    pl.close()


def test_add_volume_clip_box_shared_scalar_bar():
    # Two volumes sharing one scalar-bar title: clipping one must swap only its
    # mapper and leave the other volume's mapper registered on the same bar.
    grid_a = _gradient_volume()
    grid_b = _gradient_volume()
    pl = pv.Plotter()
    actor_a = pl.add_volume(grid_a, name='vol-a')
    actor_b = pl.add_volume(grid_b, name='vol-b')
    title = next(iter(pl.scalar_bars._scalar_bar_mappers))
    assert len(pl.scalar_bars._scalar_bar_mappers[title]) == 2

    pl.add_volume_clip_box(actor_a, factor=1.0)
    registered = pl.scalar_bars._scalar_bar_mappers[title]
    assert len(registered) == 2
    assert actor_b.mapper in registered
    assert actor_a.mapper not in registered
    pl.close()


def test_add_volume_clip_box_blending():
    grid = _gradient_volume()
    pl = pv.Plotter()
    pl.add_volume_clip_box(grid, factor=1.0, blending='maximum')
    clipped_actor = next(
        actor
        for actor in pl.renderer.actors.values()
        if isinstance(actor.mapper, _vtk.vtkOpenGLProjectedTetrahedraMapper)
    )
    assert clipped_actor.mapper.blend_mode == 'maximum'
    pl.close()

    # The projected-tetrahedra mapper only supports composite and maximum
    pl = pv.Plotter()
    with pytest.raises(ValueError, match='not supported by the projected-tetrahedra'):
        pl.add_volume_clip_box(grid, factor=1.0, blending='additive')
    pl.close()


def test_add_volume_clip_box_voxel_range():
    grid = _gradient_volume()
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0)

    # Initially the whole volume is visible
    full = _clipped_volume(pl)
    assert full.bounds == grid.bounds

    # A sub-box shows only the voxels within that box
    _move_box(widget, (2, 5, 2, 5, 2, 5))
    inside = _clipped_volume(pl)
    assert inside.bounds.x_min == 2.0
    assert inside.bounds.x_max == 5.0
    assert inside.bounds.y_min == 2.0
    assert inside.bounds.z_max == 5.0
    assert inside.n_cells < full.n_cells

    # The clipped volume renders off-screen and changes with the box
    image = pl.screenshot(return_img=True)
    assert image.ndim == 3
    _move_box(widget, (1, 3, 1, 3, 1, 3))
    assert not np.array_equal(image, pl.screenshot(return_img=True))

    # Inverting shows the voxels outside the same box
    pl.close()
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0, invert=True)
    _move_box(widget, (2, 5, 2, 5, 2, 5))
    outside = _clipped_volume(pl)
    assert outside.bounds.x_min < 2.0
    assert outside.n_cells != inside.n_cells
    pl.close()


def test_add_volume_clip_box_collapsed_and_out_of_bounds():
    grid = _gradient_volume()

    # A collapsed box clips everything when showing the interior
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0)
    _move_box(widget, (3, 3, 3, 3, 3, 3))
    assert _clipped_volume(pl) is None

    # A single collapsed face behaves the same
    _move_box(widget, (3, 3, 1, 5, 1, 5))
    assert _clipped_volume(pl) is None
    pl.close()

    # Inverted, a collapsed box shows the full volume
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0, invert=True)
    _move_box(widget, (3, 3, 3, 3, 3, 3))
    clipped = _clipped_volume(pl)
    assert clipped is not None
    assert clipped.bounds == grid.bounds
    pl.close()

    # A box fully outside the volume contains the whole volume
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0)
    _move_box(widget, (-50, 50, -50, 50, -50, 50))
    clipped = _clipped_volume(pl)
    assert clipped is not None
    assert clipped.bounds == grid.bounds
    pl.close()

    # Inverted, the same box leaves nothing visible
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0, invert=True)
    _move_box(widget, (-50, 50, -50, 50, -50, 50))
    assert _clipped_volume(pl) is None
    pl.close()


def test_add_volume_clip_box_preserves_input():
    grid = _gradient_volume()
    original_points = grid.points.copy()
    original_cells = grid.n_cells
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(grid, factor=1.0)
    _move_box(widget, (2, 5, 2, 5, 2, 5))
    _move_box(widget, (3, 3, 3, 3, 3, 3))
    assert grid.n_cells == original_cells
    assert np.array_equal(grid.points, original_points)
    pl.close()


def test_add_volume_clip_box_invalid_inputs():
    pl = pv.Plotter()
    with pytest.raises(TypeError, match='The `volume` parameter type must'):
        pl.add_volume_clip_box(pv.Sphere())

    rectilinear = pv.RectilinearGrid(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2])
    rectilinear['scalars'] = np.arange(rectilinear.n_points)
    with pytest.raises(TypeError, match='ImageData'):
        pl.add_volume_clip_box(rectilinear)

    empty_grid = pv.ImageData()
    empty_grid['scalars'] = np.zeros(empty_grid.n_points, dtype=np.float32)
    with pytest.raises(ValueError, match='at least one cell'):
        pl.add_volume_clip_box(empty_grid)

    # A flat, fewer-than-three-dimensional ImageData cannot be volume-clipped
    flat_grid = pv.ImageData(dimensions=(8, 8, 2))
    flat_grid['scalars'] = np.arange(flat_grid.n_points, dtype=np.uint8)
    with pytest.raises(ValueError, match='three-dimensional'):
        pl.add_volume_clip_box(flat_grid)

    volume_actor = pl.add_volume(pv.examples.load_uniform())
    with pytest.raises(TypeError):
        pl.add_volume_clip_box(volume_actor, cmap='bwr')
    pl.close()

    # A Volume actor whose mapper does not render an ImageData is rejected
    unstructured = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid()
    unstructured['scalars'] = np.arange(unstructured.n_points, dtype=np.uint8)
    pl = pv.Plotter()
    unstructured_actor = pl.add_volume(unstructured)
    with pytest.raises(TypeError, match='ImageData'):
        pl.add_volume_clip_box(unstructured_actor)
    pl.close()


def test_add_volume_clip_box_interaction_updates(uniform):
    pl = pv.Plotter()
    widget = pl.add_volume_clip_box(uniform, factor=1.0, interaction_event='always')
    first = _clipped_volume(pl)
    assert first is not None
    first_cells = first.n_cells

    widget.PlaceWidget((2, 6, 2, 6, 2, 6))
    widget.InvokeEvent(_vtk.vtkCommand.InteractionEvent)
    second = _clipped_volume(pl)

    assert second is not None
    assert second.n_cells != first_cells
    pl.close()


def test_plot_pointset_widgets(pointset):
    pointset = pointset.elevation()

    assert isinstance(pointset, pv.PointSet)

    pl = pv.Plotter()
    pl.add_mesh_clip_box(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_clip_plane(pointset, crinkle=True)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_threshold(pointset)
    pl.close()

    pl = pv.Plotter()
    pl.add_mesh_slice_spline(pointset)
    pl.close()


def test_ray_plane_intersection():
    # Test data
    start_point = np.array([0, 0, 0])
    direction = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 5])
    normal = np.array([0, 0, 1])

    # Expected result
    expected_result = np.array([0, 0, 5])
    result = ray_plane_intersection(
        start_point=start_point, direction=direction, plane_point=plane_point, normal=normal
    )
    np.testing.assert_array_almost_equal(result, expected_result)


def test_get_angle():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    # Expect 90 degrees between these two orthogonal vectors
    expected_angle = 90.0
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)

    # Test with parallel vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 3])
    expected_angle = 0.0  # angle between two identical vectors should be 0
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)

    # Test with opposite vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([-1, 0, 0])
    expected_angle = 180.0  # angle between two opposite vectors should be 180
    result_angle = get_angle(v1, v2)

    assert np.isclose(result_angle, expected_angle, atol=1e-8)


@pytest.fixture
def affine_widget():
    """Create an affine widget for behavioral interaction tests."""
    pl = pv.Plotter(window_size=(400, 400))
    actor = pl.add_mesh(pv.Cube())
    return pl, pl.add_affine_transform_widget(actor)


@pytest.mark.parametrize('style', ['zoom_style', 'terrain_style', 'trackball_style', None])
@pytest.mark.parametrize('finish', ['release', 'disable', 'remove'])
def test_affine_widget_restores_interactor_style(affine_widget, style, finish):
    """Ending a drag restores the style selected after the widget was created."""
    pl, widget = affine_widget
    pl.iren.style = style
    original_style = pl.iren.get_interactor_style()
    widget._selected_actor = widget._arrows[0]
    widget._press_callback(pl.iren.interactor, 'LeftButtonPressEvent')
    assert pl.iren.get_interactor_style() is not original_style
    transform = np.eye(4)
    transform[0, 3] = 2
    widget._main_actor.user_matrix = transform
    actions = {
        'release': lambda: widget._release_callback(pl.iren.interactor, 'LeftButtonReleaseEvent'),
        'disable': widget.disable,
        'remove': widget.remove,
    }

    actions[finish]()

    assert pl.iren.get_interactor_style() is original_style
    assert not widget._pressing_down
    assert np.array_equal(widget._cached_matrix, transform)


@pytest.mark.parametrize('hovered', [False, True])
def test_affine_widget_release_without_drag_preserves_style(affine_widget, hovered):
    """A release without a widget drag leaves the interactor style untouched."""
    pl, widget = affine_widget
    pl.enable_zoom_style()
    original_style = pl.iren.get_interactor_style()
    widget._selected_actor = widget._arrows[0] if hovered else None

    widget._release_callback(pl.iren.interactor, 'LeftButtonReleaseEvent')

    assert pl.iren.get_interactor_style() is original_style


def test_affine_widget_repeated_drags_preserve_current_style(affine_widget):
    """Each drag captures the current style, even with repeated press events."""
    pl, widget = affine_widget
    widget._selected_actor = widget._arrows[0]
    for style in ['zoom_style', 'terrain_style']:
        pl.iren.style = style
        original_style = pl.iren.get_interactor_style()

        widget._press_callback(pl.iren.interactor, 'LeftButtonPressEvent')
        widget._press_callback(pl.iren.interactor, 'LeftButtonPressEvent')
        widget._release_callback(pl.iren.interactor, 'LeftButtonReleaseEvent')

        assert pl.iren.get_interactor_style() is original_style


def test_affine_widget_restores_custom_style_observers(affine_widget):
    """Restoring a custom style retains its instance and registered observers."""
    pl, widget = affine_widget
    style = _vtk.vtkInteractorStyleJoystickCamera()
    calls = []
    style.AddObserver('UserEvent', lambda *_: calls.append('called'))
    pl.iren.style = style
    widget._selected_actor = widget._arrows[0]

    widget._press_callback(pl.iren.interactor, 'LeftButtonPressEvent')
    widget._release_callback(pl.iren.interactor, 'LeftButtonReleaseEvent')

    assert pl.iren.get_interactor_style() is style
    style.InvokeEvent('UserEvent')
    assert calls == ['called']


@flaky_test
@pytest.mark.parametrize('style', ['trackball_style', 'zoom_style', 'terrain_style'])
def test_affine_widget(sphere, style):
    interact_calls = []
    release_calls = []

    def interact_callback(transform):
        interact_calls.append(transform)

    def release_callback(transform):
        release_calls.append(transform)

    pl = pv.Plotter(window_size=(400, 400))
    actor = pl.add_mesh(sphere)

    with pytest.raises(TypeError, match='callable'):
        pl.add_affine_transform_widget(actor, interact_callback='foo')

    with pytest.raises(ValueError, match=r'(3, 3)'):
        pl.add_affine_transform_widget(actor, axes=np.eye(5))

    axes = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    with pytest.raises(ValueError, match='right hand'):
        pl.add_affine_transform_widget(actor, axes=axes)

    widget = pl.add_affine_transform_widget(
        actor,
        interact_callback=interact_callback,
        release_callback=release_callback,
    )
    pl.iren.style = style
    original_style = pl.iren.get_interactor_style()
    pl.show(auto_close=False)

    assert not widget._selected_actor

    # move in the center and ensure that an actor is selected
    width, height = pl.window_size
    pl.iren._mouse_move(width // 2, height // 2)
    assert widget._selected_actor in widget._arrows + widget._circles
    assert widget._selected_actor.prop.color == pv.Color(DARK_YELLOW)

    # ensure that the actor gets deselected
    pl.iren._mouse_move(0, 0)
    assert not widget._selected_actor

    def test_translation(
        press_pos: tuple[float, float],
        move_pos: tuple[float, float],
        idx: int,
        direction: Literal['neg', 'pos'],
    ):
        pl.iren._mouse_left_button_press(*press_pos)
        assert widget._selected_actor is widget._arrows[idx]
        assert widget._pressing_down

        pl.iren._mouse_move(*move_pos)
        trans_val = actor.user_matrix[idx, 3]
        assert (trans_val < 0) if direction == 'neg' else (trans_val > 0)

        pl.iren._mouse_left_button_release(*move_pos)
        trans_val = actor.user_matrix[idx, 3]
        assert (trans_val < 0) if direction == 'neg' else (trans_val > 0)
        assert not widget._pressing_down
        assert pl.iren.get_interactor_style() is original_style
        widget._reset()
        assert np.allclose(widget._cached_matrix, np.eye(4))

    # test X axis translation
    test_translation(
        press_pos=(width // 2 - 4, height // 2 - 4),
        move_pos=(width, height // 2),
        idx=0,
        direction='neg',
    )

    # test callback called
    assert len(interact_calls) == 2
    assert interact_calls[0].shape == (4, 4)
    assert len(release_calls) == 1
    assert release_calls[0].shape == (4, 4)

    # test Y axis translation
    test_translation(
        press_pos=(width // 2 + 2, height // 2 - 3),
        move_pos=(width, height // 2),
        idx=1,
        direction='pos',
    )

    # test Z axis translation
    test_translation(
        press_pos=(width // 2, height // 2 + 5),
        move_pos=(width // 2, 0),
        idx=2,
        direction='neg',
    )

    def test_rotation(
        press_pos: tuple[float, float],
        move_pos: tuple[float, float],
        idx: int,
        rotation: Literal['ccw', 'cw'],
    ):
        pl.iren._mouse_left_button_press(*press_pos)
        assert widget._selected_actor is widget._circles[idx]
        assert widget._pressing_down
        pl.iren._mouse_move(*move_pos)
        euler_angles = r_mat_to_euler_angles(actor.user_matrix)
        assert euler_angles[idx] > 0 if rotation == 'ccw' else euler_angles[idx] < 0
        assert np.count_nonzero(np.isclose(euler_angles, 0)) == len(euler_angles) - 1
        pl.iren._mouse_left_button_release()
        assert not widget._pressing_down
        assert pl.iren.get_interactor_style() is original_style
        widget._reset()
        assert np.allclose(widget._cached_matrix, np.eye(4))

    # test X axis rotation, counterclockwise
    test_rotation(
        press_pos=(width // 2 + 30, height // 2),
        move_pos=(width // 2 + 21, height // 2 + 17),
        idx=0,
        rotation='ccw',
    )

    # test Y axis rotation, counterclockwise
    test_rotation(
        press_pos=(width // 2 - 20, height // 2 + 20),
        move_pos=(width // 2 - 30, height // 2 + 4),
        idx=1,
        rotation='ccw',
    )

    # test Z axis rotation, clockwise
    test_rotation(
        press_pos=(width // 2, height // 2 - 29),
        move_pos=(width // 2 - 12, height // 2 - 23),
        idx=2,
        rotation='cw',
    )

    # test change axes
    axes = np.array(
        [
            [0.70710678, 0.70710678, 0.0],
            [-0.70710678, 0.70710678, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    widget.axes = axes
    assert np.allclose(widget.axes, axes)

    # test X axis translation with new axes
    test_translation(
        press_pos=(width // 2, height // 2 - 38),
        move_pos=(width // 2, height // 2 - 50),
        idx=0,
        direction='pos',
    )

    # test origin
    origin = np.random.default_rng().random(3)
    widget.origin = origin
    assert np.allclose(widget.origin, origin)

    # test disable
    assert pl.picking._picker_in_use
    widget.disable()
    assert not pl.picking._picker_in_use

    widget.remove()
    assert not widget._circles
    assert not widget._arrows


@pytest.mark.usefixtures('verify_image_cache')
def test_logo_widget():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_logo_widget()
    pl.show()

    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    pl.add_logo_widget(position=(0.01, 0.01), size=(0.8, 0.8))
    pl.show()

    pl = pv.Plotter()
    # has a 2 x 1 aspect ratio
    pl.add_logo_widget(examples.mapfile, position=(0.0, 0.0), size=(0.99, 0.495))
    pl.show()

    pl = pv.Plotter()
    pl.add_logo_widget(
        examples.download_vtk_logo().to_image(),
        position=(0.0, 0.0),
        size=(0.8, 0.8),
    )
    pl.show()

    pl = pv.Plotter()
    with pytest.raises(TypeError, match=r'must be a pyvista.ImageData or a file path'):
        pl.add_logo_widget(logo=0)


@pytest.mark.usefixtures('verify_image_cache')
def test_camera3d_widget():
    sphere = pv.Sphere()
    pl = pv.Plotter(window_size=[400, 200], shape=(1, 2))
    pl.add_mesh(sphere)
    pl.subplot(0, 1)
    pl.add_mesh(sphere)
    pl.add_camera3d_widget()
    pl.show(cpos=pl.camera_position)


@pytest.mark.parametrize('outline_opacity', [True, False, np.random.default_rng(0).random()])
def test_outline_opacity(uniform, outline_opacity):
    pl = pv.Plotter()
    func = lambda normal, origin: normal  # Does nothing  # noqa: ARG005
    pl.add_mesh(uniform)
    plane_widget = pl.add_plane_widget(
        callback=func, implicit=True, outline_opacity=outline_opacity
    )
    assert plane_widget.GetOutlineProperty().GetOpacity() == float(outline_opacity)
    pl.close()


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_box_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_box_widget(None)
    pl.clear_box_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_plane_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_plane_widget(None)
    pl.clear_plane_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_line_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_line_widget(None)
    pl.clear_line_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_slider_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_slider_widget(None, [0, 1])
    pl.clear_slider_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_spline_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_spline_widget(None)
    pl.clear_spline_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_measure_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_measurement_widget(None)
    pl.clear_measure_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_sphere_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_sphere_widget(None)
    pl.clear_sphere_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_camera_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera_orientation_widget()
    pl.clear_camera_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_button_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_checkbox_button_widget(None)
    pl.clear_button_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_logo_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_logo_widget(None)
    pl.clear_logo_widgets()
    pl.show(cpos='xy')


@pytest.mark.usefixtures('verify_image_cache')
def test_clear_camera3d_widget():
    mesh = pv.Cube()
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.add_camera3d_widget()
    pl.clear_camera3d_widgets()
    pl.show(cpos='xy')


class TestEventParser:
    """Class to regroup tests for widgets that use the  ``_parse_interaction_event()`` function"""

    @pytest.fixture
    def plotter(self):
        yield (pl := pv.Plotter())
        pl.close()

    @pytest.mark.parametrize(
        ('method', 'widget'),
        [
            ('add_box_widget', 'vtkBoxWidget'),
            ('add_plane_widget', 'vtkImplicitPlaneWidget'),
            ('add_line_widget', 'vtkLineWidget'),
            ('add_text_slider_widget', 'vtkSliderWidget'),
            ('add_slider_widget', 'vtkSliderWidget'),
            ('add_sphere_widget', 'vtkSphereWidget'),
            ('add_spline_widget', 'vtkSplineWidget'),
        ],
    )
    def test_add_widget(
        self,
        plotter: pv.Plotter,
        method: str,
        widget: str,
        mocker: MockerFixture,
    ):
        # Arrange
        mock = mocker.patch.object(widgets, '_parse_interaction_event')
        mock_vtk = mocker.patch.object(widgets, '_vtk')

        if widget == 'vtkSplineWidget':
            mocker.patch.object(widgets.pv, 'wrap').return_value = pv.PolyData()

        kwargs = dict(callback=lambda *b: b, interaction_event=(e := 'foo'))
        if widget == 'vtkLineWidget':
            mock_vtk.vtkLineWidget().GetPoint1.return_value = (0,) * 3
            mock_vtk.vtkLineWidget().GetPoint2.return_value = (0,) * 3

        elif widget == 'vtkSliderWidget':
            k = 'data' if method == 'add_text_slider_widget' else 'rng'
            kwargs[k] = [0, 1]

        # Act
        method = getattr(plotter, method)
        method(**kwargs)

        # Assert
        mock.assert_called_with(e)
        getattr(mock_vtk, widget)().AddObserver.assert_called_with(mock(e), ANY)


@pytest.mark.parametrize(
    'attr',
    [
        'camera_widgets',
        'box_widgets',
        'box_clipped_meshes',
        'plane_widgets',
        'plane_clipped_meshes',
        'plane_sliced_meshes',
        'line_widgets',
        'slider_widgets',
        'threshold_meshes',
        'isovalue_meshes',
        'spline_widgets',
        'spline_sliced_meshes',
        'sphere_widgets',
        'button_widgets',
        'radio_button_widget_dict',
        'radio_button_title_dict',
        'distance_widgets',
        'logo_widgets',
        'camera3d_widgets',
    ],
)
def test_deprecated_widget_state_forward(attr):
    """Each deprecated state collection forwards to ``Plotter.widgets`` with a warning."""
    pl = pv.Plotter()
    with pytest.warns(PyVistaDeprecationWarning, match=attr):
        value = getattr(pl, attr)
    assert value is getattr(pl.widgets, attr)
    pl.close()


def test_clear_radio_button_widgets_forward(uniform):
    """``Plotter.clear_radio_button_widgets`` forwards to the widgets component."""
    pl = pv.Plotter()
    pl.add_mesh(uniform)
    pl.add_radio_button_widget(callback=lambda: None, radio_button_group='group')
    assert pl.widgets.radio_button_widget_dict['group']
    pl.clear_radio_button_widgets()
    assert pl.widgets.radio_button_widget_dict == {}
    pl.close()
