"""
.. _create_tri_surface_example:

Create Triangulated Surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a surface from a set of points through a Delaunay triangulation.

This example uses :func:`pyvista.PolyDataFilters.delaunay_2d`.

"""

import numpy as np

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv

# Seed random numbers for reproducibility
rng = np.random.default_rng(seed=0)

# %%
# Simple Triangulations
# +++++++++++++++++++++
#
# First, create some points for the surface.

# Define a simple Gaussian surface
n = 20
x = np.linspace(-200, 200, num=n) + rng.uniform(-5, 5, size=n)
y = np.linspace(-200, 200, num=n) + rng.uniform(-5, 5, size=n)
xx, yy = np.meshgrid(x, y)
A, b = 100, 100
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# Get the points as a 2D NumPy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
points[0:5, :]

# %%
# Now use those points to create a point cloud PyVista data object. This will
# be encompassed in a :class:`pyvista.PolyData` object.

# simply pass the numpy points to the PolyData constructor
cloud = pv.PolyData(points)
cloud.plot(point_size=15)

# %%
# Now that we have a PyVista data structure of the points, we can perform a
# triangulation to turn those boring discrete points into a connected surface.

surf = cloud.delaunay_2d()
surf.plot(show_edges=True)


# %%
# Masked Triangulations
# +++++++++++++++++++++
#

x = np.arange(10, dtype=float)
xx, yy, zz = np.meshgrid(x, x, [0])
points = np.column_stack((xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')))
# Perturb the points
points[:, 0] += rng.random(len(points)) * 0.3
points[:, 1] += rng.random(len(points)) * 0.3
# Create the point cloud mesh to triangulate from the coordinates
cloud = pv.PolyData(points)
cloud

# %%
# Run the triangulation on these points
surf = cloud.delaunay_2d()
surf.plot(cpos='xy', show_edges=True)


# %%
# Note that some of the outer edges are unconstrained and the triangulation
# added unwanted triangles. We can mitigate that with the ``alpha`` parameter.
surf = cloud.delaunay_2d(alpha=1.0)
surf.plot(cpos='xy', show_edges=True)


# %%
# We could also constrain the triangulation to keep an outer boundary and
# carve out a hole through the ``edge_source`` parameter. Polygon winding is
# normalized automatically: the outermost polygon keeps its interior and any
# enclosed polygon becomes a hole regardless of the winding used to define it.

# Define a polygonal hole from a subset of the point ids
hole_ids = [22, 23, 24, 25, 35, 45, 44, 43, 42, 32]

# Build an outer boundary around the point cloud extent
margin = 0.5
xmin, xmax = points[:, 0].min() - margin, points[:, 0].max() + margin
ymin, ymax = points[:, 1].min() - margin, points[:, 1].max() + margin
outer_ids = [
    cloud.find_closest_point((xmin, ymin, 0.0)),
    cloud.find_closest_point((xmax, ymin, 0.0)),
    cloud.find_closest_point((xmax, ymax, 0.0)),
    cloud.find_closest_point((xmin, ymax, 0.0)),
]

# Create a polydata with the same points; its faces define the boundary and hole
polygon = pv.PolyData()
polygon.points = points
polygon.faces = np.concatenate(
    [
        np.insert(outer_ids, 0, len(outer_ids)),
        np.insert(hole_ids, 0, len(hole_ids)),
    ]
)

surf = cloud.delaunay_2d(edge_source=polygon)

pl = pv.Plotter()
pl.add_mesh(surf, show_edges=True)
pl.add_mesh(polygon, color='red', opacity=0.5)
pl.show(cpos='xy')
# %%
# .. tags:: load
