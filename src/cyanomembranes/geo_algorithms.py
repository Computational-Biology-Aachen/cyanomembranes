from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import Delaunay
from shapely import geometry
from shapely.ops import polygonize, unary_union

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


def _add_edge(
    edges: set[tuple[float, float]],
    edge_points: list[np.ndarray],
    coords: np.ndarray,
    i: float,
    j: float,
) -> None:
    """Add a line between the i-th and j-th points, if not in
    the list already"""
    if (i, j) in edges or (j, i) in edges:
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])  # type: ignore


def alpha_shape(
    points: list[Any], alpha: float
) -> tuple[BaseGeometry, list[np.ndarray]]:  # FIXME: double check typing
    """computes the alpha shape (concave hull) of a set of points. This
    implementation of the alpha-shape algorithm was taken from
    https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889.
    All credits for the implementation belong to the original author

    Parameters
    ----------
    points : list[Any]
        list of points
    alpha : float
        alpha value to influence the gooeyness of the border. Smaller
                    numbers don't fall inward as much as larger numbers.
                    Too large, and you lose everything!

    Returns
    -------
    tuple[list[Polygon], list[Tuple[float, float]]]
        a tuple of the new concave polygon and a list of the edge points
    """
    if len(points) < 4:  # noqa: PLR2004
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        # return geometry.MultiPoint(list(points)).convex_hull
        msg = "Alpha shape needs at least four points"
        raise ValueError(msg)

    coords = np.array([point.coords[0] for point in points])

    tri = Delaunay(coords)
    edges: set[tuple[float, float]] = set()  # FIXME: Check Typing
    edge_points: list[np.ndarray] = []  # FIXME: Check Typing
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

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            _add_edge(edges, edge_points, coords, ia, ib)
            _add_edge(edges, edge_points, coords, ib, ic)
            _add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points
