"""Tests for `cyanomembranes.geo_algorithms.alpha_shape`.

Every strict xfail names one confirmed bug (see INFO/OPEN_ITEMS.md, section
"geo_algorithms.py"). Inputs are small and seeded.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from cyanomembranes.geo_algorithms import alpha_shape

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _points(xy):
    return [Point(*p) for p in np.asarray(xy)]


def _grid(n=8):
    return np.array([[i, j] for i in range(n) for j in range(n)], dtype=float)


def _random(n=400, seed=0, size=10.0):
    return np.random.default_rng(seed).uniform(0, size, (n, 2))


def _accepted_triangle_union(xy, alpha):
    """Reference: union of the Delaunay triangles with circumradius < 1 / alpha."""
    triangles = []
    for simplex in Delaunay(xy).simplices:
        a, b, c = xy[simplex]
        side_a, side_b, side_c = (
            np.linalg.norm(a - b),
            np.linalg.norm(b - c),
            np.linalg.norm(c - a),
        )
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        if side_a * side_b * side_c / (4 * area) < 1 / alpha:
            triangles.append(Polygon([a, b, c]))
    return unary_union(triangles)


def _annulus(n=1500, r_in=3.0, r_out=5.0, seed=0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(r_in**2, r_out**2, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(phi), r * np.sin(phi)]


# --------------------------------------------------------------------------
# shape of the result
# --------------------------------------------------------------------------


def test_square_of_four_points():
    shape, _ = alpha_shape(_points([[0, 0], [1, 0], [1, 1], [0, 1]]), alpha=1.0)
    assert isinstance(shape, Polygon)
    assert shape.area == pytest.approx(1.0)


def test_regular_grid_gives_full_square():
    shape, _ = alpha_shape(_points(_grid(8)), alpha=1.0)
    assert shape.is_valid
    assert shape.area == pytest.approx(49.0)
    assert shape.bounds == pytest.approx((0, 0, 7, 7))


def test_tiny_alpha_gives_convex_hull():
    xy = _random(300)
    shape, _ = alpha_shape(_points(xy), alpha=1e-6)
    assert shape.area == pytest.approx(MultiPoint(xy).convex_hull.area)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 1.0])
def test_equals_union_of_accepted_triangles(alpha):
    xy = _random(400, seed=3)
    shape, _ = alpha_shape(_points(xy), alpha=alpha)
    reference = _accepted_triangle_union(xy, alpha)
    assert shape.symmetric_difference(reference).area == pytest.approx(0, abs=1e-9)


def test_larger_alpha_gives_smaller_or_equal_area():
    xy = _random(400, seed=4)
    areas = [alpha_shape(_points(xy), alpha=a)[0].area for a in (0.3, 0.6, 1.0, 1.5)]
    assert all(x >= y - 1e-9 for x, y in zip(areas, areas[1:], strict=False))


def test_points_stay_inside_or_on_the_shape_for_small_alpha():
    xy = _random(200, seed=5)
    shape, _ = alpha_shape(_points(xy), alpha=0.2)
    assert all(shape.buffer(1e-9).contains(Point(p)) for p in xy)


def test_input_order_does_not_matter():
    xy = _random(300, seed=6)
    a, _ = alpha_shape(_points(xy), alpha=0.5)
    b, _ = alpha_shape(_points(xy[::-1]), alpha=0.5)
    assert a.symmetric_difference(b).area == pytest.approx(0, abs=1e-9)


def test_duplicated_points_do_not_change_the_shape():
    xy = _random(50, seed=0)
    a, _ = alpha_shape(_points(xy), alpha=1.0)
    b, _ = alpha_shape(_points(np.r_[xy, xy]), alpha=1.0)
    assert a.area == pytest.approx(b.area)


def test_translation_and_scaling_covariance():
    xy = _random(300, seed=7)
    base, _ = alpha_shape(_points(xy), alpha=0.5)
    moved, _ = alpha_shape(_points(xy + [100.0, -50.0]), alpha=0.5)
    scaled, _ = alpha_shape(_points(xy * 2.0), alpha=0.25)  # radius scales with xy
    assert moved.area == pytest.approx(base.area)
    assert scaled.area == pytest.approx(4 * base.area)


def test_two_far_clusters_give_multipolygon():
    rng = np.random.default_rng(0)
    xy = np.r_[rng.uniform(0, 3, (100, 2)), rng.uniform(20, 23, (100, 2))]
    shape, _ = alpha_shape(_points(xy), alpha=0.5)
    assert shape.geom_type == "MultiPolygon"
    assert len(shape.geoms) == 2


def test_enclosed_holes_are_filled():
    # Documented behaviour (user decision 2026-10-23): holes bounded by accepted
    # triangles are filled, only concavities open to the outside are kept.
    shape, _ = alpha_shape(_points(_annulus()), alpha=1.0)
    assert isinstance(shape, Polygon)
    assert len(shape.interiors) == 0
    assert shape.area > 0.9 * np.pi * 5.0**2


def test_open_concavity_is_kept():
    # U shape: the notch is open to the outside and must stay cut out
    xy = np.array(
        [
            [x, y]
            for x in np.arange(0, 9)
            for y in np.arange(0, 9)
            if not (3 < x < 5 and y > 3)
        ],
        dtype=float,
    )
    shape, _ = alpha_shape(_points(xy), alpha=1.0)
    assert not shape.contains(Point(4, 6))
    assert shape.contains(Point(1, 6))


# --------------------------------------------------------------------------
# edge points
# --------------------------------------------------------------------------


def test_edge_points_are_segments_between_input_points():
    xy = _random(100, seed=1)
    _, edges = alpha_shape(_points(xy), alpha=0.5)
    assert len(edges) > 0
    for e in edges:
        assert e.shape == (2, 2)
        assert all(np.any(np.all(np.isclose(xy, p), axis=1)) for p in e)


def test_edge_points_have_no_duplicate_segments():
    _, edges = alpha_shape(_points(_random(100, seed=2)), alpha=0.5)
    keys = {tuple(sorted(map(tuple, e.round(12).tolist()))) for e in edges}
    assert len(keys) == len(edges)


def test_edge_points_cover_the_outline_vertices():
    xy = _random(100, seed=2)
    shape, edges = alpha_shape(_points(xy), alpha=0.5)
    ends = {tuple(np.round(p, 9)) for e in edges for p in e}
    assert all(tuple(np.round(v, 9)) in ends for v in shape.exterior.coords)


@pytest.mark.xfail(
    reason="Low: edge_points holds every edge of every accepted triangle, not only"
    " the outline (geo_algorithms.py:15-27, 91-92)",
    strict=True,
)
def test_edge_points_are_only_the_outline():
    # 5x5 grid: the outline has 16 unit segments, the triangulation many more
    _, edges = alpha_shape(_points(_grid(5)), alpha=1.0)
    assert len(edges) == 16


# --------------------------------------------------------------------------
# edge cases and current behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_fewer_than_four_points_raise(n):
    with pytest.raises(ValueError, match="at least four points"):
        alpha_shape(_points(_random(n)), alpha=1.0)


def test_alpha_rejecting_all_triangles_gives_empty_result():
    # current behaviour: silent empty GeometryCollection, see the xfail below
    shape, edges = alpha_shape(_points(_random(100)), alpha=100.0)
    assert shape.is_empty
    assert edges == []


def test_alpha_zero_raises_zero_division_error():
    # current behaviour, see the xfail below
    with pytest.raises(ZeroDivisionError):
        alpha_shape(_points(_random(50)), alpha=0.0)


def test_collinear_points_raise():
    # current behaviour: QhullError (a ValueError subclass in scipy is not
    # guaranteed), so only check that something is raised
    with pytest.raises(Exception, match="(?i)qhull|flat|simplex"):
        alpha_shape(_points(np.c_[np.arange(6.0), np.zeros(6)]), alpha=1.0)


def test_nan_coordinates_raise_value_error():
    xy = np.r_[_random(20), [[np.nan, 0.0]]]
    with pytest.raises(ValueError):  # noqa: PT011
        alpha_shape(_points(xy), alpha=1.0)


@pytest.mark.xfail(
    reason="Low: an empty result (no accepted triangle) is returned silently,"
    " make_polygon would write 'GEOMETRYCOLLECTION EMPTY' (geo_algorithms.py:91-92)",
    strict=True,
)
def test_empty_result_raises_or_warns():
    with pytest.warns(UserWarning):
        alpha_shape(_points(_random(100)), alpha=100.0)


@pytest.mark.xfail(
    reason="Low: alpha <= 0 is not validated (alpha=0 ZeroDivisionError, alpha < 0"
    " silently empty) (geo_algorithms.py:76)",
    strict=True,
)
@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_non_positive_alpha_raises_value_error(alpha):
    with pytest.raises(ValueError, match="alpha"):
        alpha_shape(_points(_random(50)), alpha=alpha)


@pytest.mark.xfail(
    reason="Low: Heron's formula rounds the area of thin triangles to 0"
    " (ZeroDivisionError) or makes it negative (ValueError) (geo_algorithms.py:70-79)",
    strict=True,
)
@pytest.mark.parametrize("deviation", [1e-10, 1e-12])
def test_nearly_collinear_points_do_not_crash(deviation):
    x = np.linspace(0, 10, 20)
    y = deviation * np.sin(np.arange(20))
    xy = np.c_[x, y]
    xy[0] = [5.0, 3.0]
    shape, _ = alpha_shape(_points(xy), alpha=0.3)
    assert shape.area >= 0
