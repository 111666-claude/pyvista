"""Regression tests for face normal arrows on meshes with triangle strips."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv


def _capture_face_arrows(monkeypatch, mesh, **kwargs):
    arrows = []

    def capture_add_arrows(_plotter, cent, direction, **_kwargs):
        arrows.append(
            (np.asarray(cent, dtype=float), np.asarray(direction, dtype=float)),
        )

    monkeypatch.setattr(pv.Plotter, 'add_arrows', capture_add_arrows)
    monkeypatch.setattr(pv.Plotter, 'show', lambda *_args, **_kwargs: None)
    mesh.plot_normals(faces=True, off_screen=True, **kwargs)
    assert len(arrows) == 1
    return arrows[0]


@pytest.mark.skip_plotting
def test_plot_normals_faces_extruded_rectangle(monkeypatch):
    mesh = pv.Rectangle().extrude((0.0, 0.0, 1.0), capping=True)
    points_before = mesh.points.copy()
    faces_before = mesh.faces.copy()
    strips_before = mesh.strips.copy()

    centers, normals = _capture_face_arrows(monkeypatch, mesh)

    # Two cap polygons and two triangles from each of the four wall strips.
    assert centers.shape == normals.shape == (10, 3)

    third = 1.0 / 3.0
    two_thirds = 2.0 / 3.0
    expected_centers = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
            [1.0, third, third],
            [1.0, two_thirds, two_thirds],
            [two_thirds, 1.0, third],
            [third, 1.0, two_thirds],
            [0.0, two_thirds, third],
            [0.0, third, two_thirds],
            [third, 0.0, third],
            [two_thirds, 0.0, two_thirds],
        ],
    )
    assert np.allclose(centers, expected_centers)

    # Hand-computed outward normals for the same rows. Extrusion winding makes
    # the consistent cell normals point inward, so the plotted normals are the
    # negatives of these.
    outward = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )
    assert np.allclose(normals, -outward)

    # Every center/normal pair is strictly one-to-one and axis aligned.
    assert np.allclose(np.abs(np.sum(normals * outward, axis=1)), 1.0)

    # The source mesh topology and data are untouched.
    assert np.array_equal(mesh.points, points_before)
    assert np.array_equal(mesh.faces, faces_before)
    assert np.array_equal(mesh.strips, strips_before)


@pytest.mark.skip_plotting
@pytest.mark.parametrize(
    ('faces', 'strips', 'n_arrows'),
    [
        ([4, 0, 1, 2, 3], [], 1),  # polygon only, explicit empty strips array
        ([], [4, 0, 1, 2, 3], 2),  # one strip decomposed into two triangles
        ([], [2, 0, 1], 0),  # degenerate strip decomposes into no triangles
        ([4, 0, 1, 2, 3], [4, 4, 5, 6, 7], 3),  # polygon plus strip
    ],
)
def test_plot_normals_faces_mixed_polygons_and_strips(
    monkeypatch,
    faces,
    strips,
    n_arrows,
):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
    )
    faces_arr = np.asarray(faces, dtype=np.int64) if faces else None
    strips_arr = np.asarray(strips, dtype=np.int64)
    mesh = pv.PolyData(points, faces=faces_arr, strips=strips_arr)
    centers, normals = _capture_face_arrows(monkeypatch, mesh)
    assert centers.shape == normals.shape == (n_arrows, 3)


@pytest.mark.skip_plotting
def test_plot_normals_flip_does_not_mutate_source_normals(monkeypatch):
    mesh = pv.Plane().triangulate()
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    normals_before = mesh.cell_data['Normals'].copy()

    _capture_face_arrows(monkeypatch, mesh, flip=True)

    assert np.allclose(mesh.cell_data['Normals'], normals_before)
