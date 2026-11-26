from __future__ import annotations

import pytest

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.mapper import ProjectedTetrahedraVolumeMapper


@pytest.fixture
def volume_mapper():
    vol = pv.ImageData(dimensions=(10, 10, 10))
    vol['scalars'] = 255 - vol.z * 25
    pl = pv.Plotter()
    actor = pl.add_volume(vol)
    return actor.mapper


def test_volume_mapper_dataset(volume_mapper):
    assert isinstance(volume_mapper.dataset, pv.ImageData)


def test_volume_mapper_blend_mode(volume_mapper):
    assert isinstance(volume_mapper.blend_mode, str)

    volume_mapper.blend_mode = _vtk.vtkVolumeMapper.COMPOSITE_BLEND
    assert volume_mapper.blend_mode == 'composite'

    for mode in ['average', 'minimum', 'maximum', 'composite', 'additive']:
        volume_mapper.blend_mode = mode
        assert volume_mapper.blend_mode == mode

    with pytest.raises(ValueError, match='Please choose either "additive"'):
        volume_mapper.blend_mode = 'not a mode'

    with pytest.raises(TypeError, match='blend_mode must be an instance of'):
        volume_mapper.blend_mode = 0.5


def test_projected_tetrahedra_mapper_blend_mode():
    mapper = ProjectedTetrahedraVolumeMapper()

    # Only composite and maximum are supported by the projected-tetrahedra mapper
    for mode in ['composite', 'maximum']:
        mapper.blend_mode = mode
        assert mapper.blend_mode == mode

    mapper.blend_mode = _vtk.vtkVolumeMapper.COMPOSITE_BLEND
    assert mapper.blend_mode == 'composite'
    mapper.blend_mode = _vtk.vtkVolumeMapper.MAXIMUM_INTENSITY_BLEND
    assert mapper.blend_mode == 'maximum'

    for mode in ['additive', 'average', 'minimum']:
        with pytest.raises(ValueError, match='not supported by the projected-tetrahedra'):
            mapper.blend_mode = mode

    # Integer blend modes outside composite/maximum are rejected as well
    for mode in [
        _vtk.vtkVolumeMapper.ADDITIVE_BLEND,
        _vtk.vtkVolumeMapper.MINIMUM_INTENSITY_BLEND,
        _vtk.vtkVolumeMapper.AVERAGE_INTENSITY_BLEND,
        99,
    ]:
        with pytest.raises(ValueError, match='not supported by the projected-tetrahedra'):
            mapper.blend_mode = mode

    with pytest.raises(TypeError, match='blend_mode must be an instance of'):
        mapper.blend_mode = 0.5
