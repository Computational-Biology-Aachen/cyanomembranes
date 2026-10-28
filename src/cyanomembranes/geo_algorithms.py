from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import polygonize, unary_union

if TYPE_CHECKING:
    from shapely import Point
    from shapely.geometry.base import BaseGeometry


def _add_edge(
    edges: set[tuple[int, int]],
    edge_points: list[np.ndarray],
    coords: np.ndarray,
    i: int,
    j: int,
) -> None:
    """Add the edge between the i-th and j-th points to `edges` and
    `edge_points`, if it is not in there already (in either direction)"""
    if (i, j) in edges or (j, i) in edges:
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])  # type: ignore


def alpha_shape(
    points: list[Point], alpha: float
) -> tuple[BaseGeometry, list[np.ndarray]]:
    """computes the alpha shape (concave hull) of a set of points. This
    implementation of the alpha-shape algorithm was taken from
    https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889 and
    slightly modified. All credits for the implementation without modifications
    belong to the original author

    Parameters
    ----------
    points : list[Point]
        list of at least four shapely Points (x, y)
    alpha : float
        alpha value to influence the gooeyness of the border. Must be positive.
        A Delaunay triangle is kept if its circumradius is smaller than
        1 / alpha. Smaller numbers don't fall inward as much as larger numbers.
        Too large, and you lose everything!

    Returns
    -------
    tuple[BaseGeometry, list[np.ndarray]]
        The concave shape (a Polygon, a MultiPolygon for disconnected parts, or an
        empty geometry if no triangle is kept) and the edges of all kept
        triangles, each as an array of shape (2, 2) with the coordinates of its
        two end points. The edges include inner edges, not only the outline.

    Raises
    ------
    ValueError
        If `alpha` is not positive or fewer than four points are given.

    Warns
    -----
    UserWarning
        If no triangle is kept (alpha too large) and the shape is empty.

    Notes
    -----
    Enclosed holes are filled; only concavities open to the outside are kept.
    """
    if alpha <= 0.0:
        msg = "alpha must be positive"
        raise ValueError(msg)
    if len(points) < 4:  # noqa: PLR2004
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        # return geometry.MultiPoint(list(points)).convex_hull
        msg = "Alpha shape needs at least four points"
        raise ValueError(msg)

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges: set[tuple[int, int]] = set()
    edge_points: list[np.ndarray] = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Area of triangle # modified from original implementation
        area = (
            abs((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
            / 2
        )
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            _add_edge(edges, edge_points, coords, ia, ib)
            _add_edge(edges, edge_points, coords, ib, ic)
            _add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    final_form = unary_union(triangles)
    if final_form.is_empty:
        warnings.warn(
            "No triangle passed the radius filter, alpha was too large",
            UserWarning,
            stacklevel=2,
        )
    return final_form, edge_points
