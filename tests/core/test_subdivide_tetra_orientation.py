"""Tests for positively oriented, conforming tetrahedron subdivision."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import pyvista as pv

# A tetrahedron whose determinant is 6, hence whose volume is exactly 1.
VOLUME_ONE_POINTS = np.array([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.5)])


def tetra_grid(points=VOLUME_ONE_POINTS, order=(0, 1, 2, 3)):
    """Build a single TETRA UnstructuredGrid with the given vertex order."""
    cells = np.concatenate(([4], np.asarray(order)))
    return pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)


def signed_volumes(grid):
    """Signed determinant volume of every tetrahedral cell."""
    conn = grid.cell_connectivity.reshape(-1, 4)
    points = grid.points
    p0, p1, p2, p3 = (points[conn[:, i]] for i in range(4))
    return np.linalg.det(np.stack((p1 - p0, p2 - p0, p3 - p0), axis=1)) / 6.0


def nondegenerate_volumes(grid, tolerance=1e-9):
    """Return child volumes whose magnitude is above the degeneracy tolerance."""
    volumes = signed_volumes(grid)
    return volumes[~np.isclose(volumes, 0.0, atol=tolerance)]


def face_multiplicities(grid):
    """Count how many tetrahedra reference each triangular face (sorted ids)."""
    conn = grid.cell_connectivity.reshape(-1, 4)
    counts = {}
    faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for cell in conn:
        for face in faces:
            key = tuple(sorted(int(cell[i]) for i in face))
            counts[key] = counts.get(key, 0) + 1
    return np.array(list(counts.values()))


def test_subdivide_tetra_twelve_children():
    """Each tetrahedron yields twelve child tetrahedra."""
    subdivided = tetra_grid().subdivide_tetra()
    assert subdivided.n_cells == 12
    assert np.all(subdivided.celltypes == pv.CellType.TETRA)


def test_subdivide_tetra_all_children_have_positive_volume():
    """Child determinant volumes are positive and sum to the input volume."""
    mesh = tetra_grid()
    subdivided = mesh.subdivide_tetra()
    volumes = signed_volumes(subdivided)
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), 1.0)
    assert np.isclose(subdivided.volume, mesh.volume)
    assert np.isclose(mesh.volume, 1.0)


@pytest.mark.parametrize('order', list(itertools.permutations(range(4))))
def test_subdivide_tetra_orientation_is_order_independent(order):
    """Every vertex ordering, including reversed order, gives positive children."""
    subdivided = tetra_grid(order=order).subdivide_tetra()
    volumes = nondegenerate_volumes(subdivided)
    assert volumes.size == 12
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), 1.0)


def test_subdivide_tetra_reversed_input():
    """An explicitly reversed tetrahedron is reoriented to positive volumes."""
    reversed_grid = tetra_grid(order=(0, 2, 1, 3))
    assert reversed_grid.volume < 0.0
    subdivided = reversed_grid.subdivide_tetra()
    volumes = signed_volumes(subdivided)
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), 1.0)
    assert np.isclose(subdivided.volume, 1.0)


@pytest.mark.parametrize('levels', [1, 2, 3])
def test_subdivide_tetra_repeated_subdivision(levels):
    """Repeated subdivision keeps orientation, count, and total volume."""
    subdivided = tetra_grid()
    for _ in range(levels):
        subdivided = subdivided.subdivide_tetra()
    assert subdivided.n_cells == 12**levels
    volumes = signed_volumes(subdivided)
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), 1.0)


@pytest.mark.parametrize(
    'points',
    [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
    ids=['coplanar', 'collinear'],
)
def test_subdivide_tetra_degenerate_input(points):
    """Degenerate tetrahedra subdivide without producing non-degenerate children."""
    subdivided = tetra_grid(points=points).subdivide_tetra()
    assert subdivided.n_cells == 12
    volumes = signed_volumes(subdivided)
    assert np.allclose(volumes, 0.0, atol=1e-9)
    assert np.isclose(subdivided.volume, 0.0, atol=1e-9)


def test_subdivide_tetra_repeated_vertex_is_degenerate():
    """A repeated connectivity id produces zero-volume children and no error."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mesh = pv.UnstructuredGrid([4, 0, 1, 1, 2], [pv.CellType.TETRA], points)
    subdivided = mesh.subdivide_tetra()
    assert subdivided.n_cells == 12
    assert np.allclose(signed_volumes(subdivided), 0.0, atol=1e-9)


def test_subdivide_tetra_valid_and_degenerate_together():
    """A degenerate cell does not spoil the volumes of valid neighbors."""
    points = np.vstack(
        [
            VOLUME_ONE_POINTS,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    cells = np.concatenate(([4, 0, 1, 2, 3], [4, 4, 5, 6, 7]))
    mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA] * 2, points)
    subdivided = mesh.subdivide_tetra()
    volumes = nondegenerate_volumes(subdivided)
    assert volumes.size == 12
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), 1.0)


def test_subdivide_tetra_point_data_tracks_geometry():
    """Interpolated point data equals the field evaluated at the new positions."""
    mesh = tetra_grid()
    mesh.point_data['coords'] = mesh.points.copy()
    mesh.point_data['x'] = mesh.points[:, 0].copy()
    subdivided = mesh.subdivide_tetra()
    assert np.allclose(subdivided.point_data['coords'], subdivided.points)
    assert np.allclose(subdivided.point_data['x'], subdivided.points[:, 0])


def test_subdivide_tetra_point_data_midpoint_values():
    """New edge midpoint data is the mean of its endpoint values."""
    mesh = tetra_grid(
        points=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    )
    mesh.point_data['v'] = np.array([0.0, 2.0, 4.0, 8.0])
    subdivided = mesh.subdivide_tetra()
    # Midpoint of edge 0-1 sits at (1, 0, 0) and must carry (0 + 2) / 2 = 1.
    target = np.array([1.0, 0.0, 0.0])
    midpoint_id = np.flatnonzero(np.all(np.isclose(subdivided.points, target), axis=1))
    assert midpoint_id.size == 1
    assert np.isclose(subdivided.point_data['v'][midpoint_id[0]], 1.0)


def test_subdivide_tetra_integer_point_data_is_interpolated():
    """Integer point labels take fractional midpoint values."""
    mesh = tetra_grid()
    mesh.point_data['id'] = np.arange(4, dtype=np.int64)
    subdivided = mesh.subdivide_tetra()
    values = subdivided.point_data['id']
    assert np.issubdtype(values.dtype, np.floating)
    # Six edge midpoints and one centroid take interpolated (fractional) values.
    assert np.any(values % 1.0 != 0.0)


def test_subdivide_tetra_cell_data_copied_to_children():
    """Each parent cell's data is repeated across its twelve children."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    cells = np.concatenate(([4, 0, 1, 2, 3], [4, 0, 2, 1, 4]))
    mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA] * 2, points)
    mesh.cell_data['cid'] = np.array([10, 20], dtype=np.int32)
    subdivided = mesh.subdivide_tetra()
    expected = np.repeat([10, 20], 12)
    assert np.array_equal(subdivided.cell_data['cid'], expected)
    assert subdivided.cell_data['cid'].dtype == np.int32


def test_subdivide_tetra_does_not_mutate_input():
    """Subdivision leaves the input geometry and data unchanged."""
    mesh = tetra_grid()
    mesh.point_data['p'] = np.arange(4.0)
    mesh.cell_data['c'] = np.array([3])
    points_before = mesh.points.copy()
    conn_before = mesh.cell_connectivity.copy()
    point_data_before = mesh.point_data['p'].copy()
    cell_data_before = mesh.cell_data['c'].copy()

    mesh.subdivide_tetra()

    assert np.array_equal(mesh.points, points_before)
    assert np.array_equal(mesh.cell_connectivity, conn_before)
    assert np.array_equal(mesh.point_data['p'], point_data_before)
    assert np.array_equal(mesh.cell_data['c'], cell_data_before)


def test_subdivide_tetra_shared_face_has_no_cracks():
    """Two tetrahedra sharing a face refine it into matching, welded triangles."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    cells = np.concatenate(([4, 0, 1, 2, 3], [4, 0, 2, 1, 4]))
    mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA] * 2, points)
    subdivided = mesh.subdivide_tetra()

    # Five corners, nine unique edges, and two centroids are all welded.
    assert subdivided.n_points == 16
    assert len(np.unique(subdivided.points, axis=0)) == subdivided.n_points

    multiplicities = face_multiplicities(subdivided)
    assert set(np.unique(multiplicities)).issubset({1, 2})
    # Interior refined faces are shared by a pair of child tetrahedra.
    assert np.any(multiplicities == 2)
    # The shared parent face splits into four triangles, each shared twice.
    assert np.count_nonzero(multiplicities == 2) == 36


def test_subdivide_tetra_tetbeam(tetbeam):
    """A conformal tetrahedral mesh stays conformal with positive volumes."""
    subdivided = tetbeam.subdivide_tetra()
    assert subdivided.n_cells == tetbeam.n_cells * 12
    volumes = signed_volumes(subdivided)
    assert np.all(volumes > 0.0)
    assert np.isclose(volumes.sum(), tetbeam.volume)
    multiplicities = face_multiplicities(subdivided)
    assert set(np.unique(multiplicities)).issubset({1, 2})
    assert subdivided.validate_mesh().is_valid
