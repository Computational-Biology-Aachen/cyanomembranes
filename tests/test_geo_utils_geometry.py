"""Tests for the geometry and placement code of `cyanomembranes.geo_utils`.

Every strict xfail names one confirmed bug (see INFO/OPEN_ITEMS.md, section
"geo_utils.py"). Inputs are small and seeded; no test needs more than a few
seconds.
"""

from __future__ import annotations

import random
import subprocess
import sys
import warnings

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.geometry.polygon import orient

import cyanomembranes.geo_utils as gu

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _star(size, nv=8, seed=0, center=(0.0, 0.0)):
    """Simple star-shaped polygon around `center` (always valid)."""
    rng = np.random.default_rng(seed)
    step = 2 * np.pi / nv
    angles = np.arange(nv) * step + rng.uniform(-0.2, 0.2, nv) * step
    radii = rng.uniform(0.5, 1.0, nv) * size
    coords = np.c_[
        radii * np.cos(angles) + center[0], radii * np.sin(angles) + center[1]
    ]
    return Polygon(coords)


def _convex(seed, n=7, size=10.0):
    rng = np.random.default_rng(seed)
    hull = MultiPoint(rng.uniform(-size, size, (n, 2))).convex_hull
    return orient(hull, 1.0)


def _overlaps(polygons, tile=None):
    """Intersection areas of all overlapping pairs (periodic images if `tile`)."""
    shifts = [(0, 0)]
    if tile is not None:
        shifts = [(i * tile, j * tile) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    found = []
    for i, a in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            for dx, dy in shifts:
                b = translate(polygons[j], dx, dy)
                if a.intersects(b):
                    area = a.intersection(b).area
                    if area > 0:
                        found.append(area)
    return found


def _protrusion_depth(polygon, other):
    """Largest distance by which `polygon` sticks out of `other`."""
    outside = polygon.difference(other)
    if outside.is_empty:
        return 0.0
    geoms = getattr(outside, "geoms", [outside])
    return max(Point(c).distance(other) for g in geoms for c in g.exterior.coords)


def _signed_area_of_edges(edges):
    pts = np.vstack([np.zeros(2), np.cumsum(edges, axis=0)])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


@pytest.fixture
def py_random():
    state = random.getstate()
    yield random
    random.setstate(state)


# --------------------------------------------------------------------------
# lattice directions
# --------------------------------------------------------------------------


def test_lattice_directions_square():
    assert gu._get_lattice_directions("square") == [(1, 0), (-1, 0), (0, 1), (0, -1)]


@pytest.mark.parametrize("lattice", ["triangular", "hexagonal"])
def test_lattice_directions_unit_vectors_in_opposite_pairs(lattice):
    dirs = np.array(gu._get_lattice_directions(lattice))
    assert dirs.shape == (6, 2)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    for d in dirs:  # every direction has its opposite
        assert np.isclose(np.linalg.norm(dirs + d, axis=1), 0.0).any()


def test_lattice_directions_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported lattice type"):
        gu._get_lattice_directions("cubic")  # type: ignore[arg-type]


@pytest.mark.xfail(
    reason='Low: "triangular" and "hexagonal" return the same six directions'
    " (geo_utils.py:70-83)",
    strict=True,
)
def test_triangular_and_hexagonal_directions_differ():
    tri = np.array(gu._get_lattice_directions("triangular"))
    hexa = np.array(gu._get_lattice_directions("hexagonal"))
    assert not np.allclose(tri, hexa)


# --------------------------------------------------------------------------
# ghosts, translate, reflect
# --------------------------------------------------------------------------


def test_create_ghost_eight_neighbours():
    p = box(10, 10, 20, 20)
    ghosts = gu._create_ghost(p, (0, 100))
    assert len(ghosts) == 8
    offsets = {(g.bounds[0] - 10, g.bounds[1] - 10) for g in ghosts}
    assert offsets == {(dx, dy) for dx in (-100, 0, 100) for dy in (-100, 0, 100)} - {
        (0, 0)
    }
    assert all(g.area == p.area for g in ghosts)


@pytest.mark.xfail(
    reason="Low: _create_ghost uses dimensions[1] as tile size, not max - min"
    " (geo_utils.py:119)",
    strict=True,
)
def test_create_ghost_tile_size_is_max_minus_min():
    ghost = gu._create_ghost(box(20, 20, 30, 30), (10, 50))[1]  # (0, 1) offset
    assert ghost.bounds[1] == pytest.approx(20 + 40)


def test_translate_polygon_moves_centroid_to_target():
    p = _star(20, seed=3, center=(100, 200))
    moved = gu._translate_polygon(p, 5.0, -7.0)
    assert (moved.centroid.x, moved.centroid.y) == pytest.approx((5.0, -7.0))
    assert moved.area == pytest.approx(p.area)


def test_polygon_translate_returns_polygon_and_new_centroid():
    p = _star(20, seed=3)
    moved, cx, cy = gu.polygon_translate(p, xoff=3.0, yoff=4.0)
    assert float(cx) == pytest.approx(p.centroid.x + 3.0)
    assert float(cy) == pytest.approx(p.centroid.y + 4.0)
    assert moved.area == pytest.approx(p.area)


def test_point_reflect_polygon_and_array():
    p = _star(20, seed=3, center=(30, 10))
    r = gu.point_reflect(p)
    assert isinstance(r, Polygon)
    assert (r.centroid.x, r.centroid.y) == pytest.approx((-p.centroid.x, -p.centroid.y))
    assert r.area == pytest.approx(p.area)
    assert gu.point_reflect(r).equals(p)  # involution
    arr = np.array([[1.0, 2.0], [-3.0, 4.0]])
    assert np.array_equal(gu.point_reflect(arr), -arr)


def test_point_reflect_rejects_other_types():
    with pytest.raises(TypeError):
        gu.point_reflect("polygon")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# edges and angles
# --------------------------------------------------------------------------


def test_get_edges_closed_loop_and_count():
    p = orient(_star(20, seed=1), 1.0)
    edges = gu.get_edges(p)
    assert len(edges) == len(p.exterior.coords) - 1
    assert np.allclose(edges.sum(axis=0), 0.0)


def test_get_edges_ccw_false_gives_clockwise_edges():
    ccw_poly = orient(_star(20, seed=1), 1.0)
    cw_poly = orient(_star(20, seed=1), -1.0)
    assert _signed_area_of_edges(gu.get_edges(ccw_poly, ccw=False)) < 0
    assert _signed_area_of_edges(gu.get_edges(cw_poly, ccw=False)) < 0


@pytest.mark.xfail(
    reason="Low: get_edges(ccw=True) does not orient, a clockwise polygon returns"
    " clockwise edges (geo_utils.py:529)",
    strict=True,
)
def test_get_edges_ccw_true_gives_counterclockwise_edges():
    cw_poly = orient(_star(20, seed=1), -1.0)
    assert _signed_area_of_edges(gu.get_edges(cw_poly, ccw=True)) > 0


def test_sort_edges_by_angle_orders_counterclockwise_from_x_axis():
    edges = np.array([[0, -1.0], [-1, 0], [1, 1], [0, 1], [1, 0], [1, -1e-17]])
    idx, angles = gu._sort_edges_by_angle(edges)
    assert (angles >= 0).all() and (angles <= 2 * np.pi).all()  # -1e-17 rounds to 2 pi
    assert (np.diff(angles[idx]) >= 0).all()
    assert idx[0] == 4  # exactly along +x comes first
    assert idx[-1] == 5  # a hair below +x wraps to just under 2 pi


# --------------------------------------------------------------------------
# projection and DOP
# --------------------------------------------------------------------------


def test_projection_directions_k8_are_four_unit_directions():
    directions, normals = gu._get_projection_directions_normals(8)
    assert directions.shape == (4, 2)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(directions * normals, axis=1), 0.0, atol=1e-12)
    angles = np.sort(np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 180)
    assert angles == pytest.approx([0, 45, 90, 135], abs=1e-9)


@pytest.mark.parametrize("k", [16, 32])
def test_projection_directions_are_distinct_lines(k):
    directions, _ = gu._get_projection_directions_normals(k)
    lines = np.round(
        np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 180, 0
    )
    lines = lines % 180  # 180 and 0 are the same line
    assert len(directions) == k // 2
    assert len(set(lines)) == len(directions)


@pytest.mark.parametrize("n", [0, 2])
def test_projection_directions_reject_n_below_four(n):
    with pytest.raises(ValueError, match="at least 4"):
        gu._get_projection_directions_normals(n)


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_projection_directions_reject_odd_n(n):
    with pytest.raises(ValueError, match="even"):
        gu._get_projection_directions_normals(n)


def test_project_polygon_matches_bounds():
    p = box(2, 3, 12, 8)
    proj = gu._project_polygon(p, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert proj == pytest.approx(np.array([[2, 12], [3, 8]]))
    diag = np.array([[np.sqrt(0.5), np.sqrt(0.5)]])
    assert gu._project_polygon(p, diag)[0] == pytest.approx(
        [5 * np.sqrt(0.5), 20 * np.sqrt(0.5)]
    )


@pytest.mark.parametrize("seed", range(6))
def test_get_dop_is_convex_ccw_and_close_to_polygon(seed):
    p = gu._translate_polygon(_star(40, seed=seed), 0, 0)
    dop = gu.get_dop(p, k=8, extension=100_000)
    assert dop.is_valid and dop.exterior.is_ccw
    assert dop.convex_hull.area == pytest.approx(dop.area, rel=1e-9)  # convex
    assert dop.area >= p.convex_hull.area * (1 - 1e-9)  # encloses the hull


@pytest.mark.parametrize("seed", range(6))
def test_get_dop_protrusion_is_bounded(seed):
    # The directions are unit vectors, so the polygon must not protrude at all.
    size = 100
    p = gu._translate_polygon(_star(size, seed=seed), 0, 0)
    dop = gu.get_dop(p, k=8, extension=100_000)
    assert _protrusion_depth(p, dop) < 1e-9 * size


@pytest.mark.parametrize("k", [8, 16, 32])
@pytest.mark.parametrize("seed", range(6))
def test_get_dop_contains_polygon(seed, k):
    p = gu._translate_polygon(_star(100, seed=seed), 0, 0)
    dop = gu.get_dop(p, k=k, extension=100_000)
    assert p.difference(dop).area <= 1e-9


def test_get_dop_halfspaces_hf():
    p = gu._translate_polygon(_star(40, seed=2), 0, 0)
    lines = gu.get_dop(p, k=8, hf=True)
    assert len(lines) == 8  # min and max line for each of the 4 directions


def test_get_dop_too_short_halfspaces_raise_value_error():
    p = gu._translate_polygon(_star(60, seed=2), 0, 0)
    with pytest.raises(ValueError, match="Could not find a DOP"):
        gu.get_dop(p, k=8, extension=5)


def test_get_dop_default_extension_handles_large_polygon():
    p = gu._translate_polygon(_star(300, seed=0), 0, 0)
    assert gu.get_dop(p, k=8).is_valid


def test_get_dop_default_extension_handles_far_polygon():
    p = _star(40, seed=0, center=(1000, 1000))
    assert gu.get_dop(p, k=8).is_valid


@pytest.mark.parametrize("size", [0.5, 1.0, 5000.0])
def test_get_dop_default_extension_contains_polygon_at_any_size(size):
    p = gu._translate_polygon(_star(size, seed=1), 0, 0)
    dop = gu.get_dop(p, k=16)
    assert p.difference(dop.buffer(1e-9 * size)).is_empty


# --------------------------------------------------------------------------
# Minkowski sum
# --------------------------------------------------------------------------


def _hull_of_vertex_sums(a, b):
    sums = [
        (p[0] + q[0], p[1] + q[1]) for p in a.exterior.coords for q in b.exterior.coords
    ]
    return orient(MultiPoint(sums).convex_hull, 1.0)


@pytest.mark.parametrize("seed", range(8))
def test_mink_sum_equals_hull_of_vertex_sums(seed):
    a, b = _convex(seed), _convex(seed + 100)
    result = gu.mink_sum(a, b)
    reference = _hull_of_vertex_sums(a, b)
    assert result.is_valid
    assert result.symmetric_difference(reference).area <= 1e-9 * reference.area


def test_mink_sum_of_squares():
    result = gu.mink_sum(box(0, 0, 2, 2), box(-1, -1, 1, 1))
    assert result.equals(box(-1, -1, 3, 3))


def test_mink_sum_clockwise_input():
    a, b = _convex(1), _convex(101)
    a_cw, b_cw = orient(a, -1.0), orient(b, -1.0)
    result = gu.mink_sum(a_cw, b_cw)
    reference = _hull_of_vertex_sums(a, b)
    assert result.is_valid
    assert result.symmetric_difference(reference).area <= 1e-9 * reference.area


# --------------------------------------------------------------------------
# elongate, longest direction
# --------------------------------------------------------------------------


def test_elongate_polygon_area_scales_with_factor():
    p = _star(100, seed=1, center=(500, 300))
    assert gu.elongate_polygon(p, factor=1.2).area == pytest.approx(1.2 * p.area)


def test_elongate_polygon_box():
    e = gu.elongate_polygon(box(0, 0, 10, 2), factor=1.5)
    assert e.bounds == pytest.approx((-2.5, 0.0, 12.5, 2.0))


@pytest.mark.xfail(
    reason="Low: elongate_polygon moves the centroid (rotation and scale use"
    " different centres) (geo_utils.py:452-456)",
    strict=True,
)
def test_elongate_polygon_keeps_centroid():
    p = _star(100, seed=1, center=(500, 300))
    e = gu.elongate_polygon(p, factor=1.2)
    assert (e.centroid.x, e.centroid.y) == pytest.approx(
        (p.centroid.x, p.centroid.y), abs=1.0
    )


def test_elongate_polygon_non_polygon_rectangle_raises():
    line_like = Polygon([(0, 0), (1, 1), (2, 2)])  # degenerate: no area
    with pytest.raises((TypeError, ValueError, IndexError)):
        gu.elongate_polygon(line_like)


@pytest.mark.parametrize("angle", [0, 30, 100, 200])
def test_get_longest_direction(angle):
    p = rotate(box(-10, -2, 10, 2), angle, origin=(0, 0))
    direction = np.asarray(gu.get_longest_direction(p), dtype=float)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    expected = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    assert abs(direction @ expected) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# sampling in free space
# --------------------------------------------------------------------------


def test_sample_point_inside_free_space_and_constraint():
    free = box(0, 0, 10, 10).difference(box(1, 1, 9, 9))
    rng = np.random.default_rng(0)
    for _ in range(100):
        x, y = gu._sample_point_in_free_space(free, rng=rng)
        assert free.contains(Point(x, y))
    constraint = box(0, 0, 10, 5)
    for _ in range(50):
        x, y = gu._sample_point_in_free_space(free, constraint=constraint, rng=rng)
        assert free.contains(Point(x, y)) and y < 5


def test_sample_point_reproducible_with_rng():
    free = box(0, 0, 10, 10)
    a = gu._sample_point_in_free_space(free, rng=np.random.default_rng(3))
    b = gu._sample_point_in_free_space(free, rng=np.random.default_rng(3))
    assert a == b


def test_sample_point_raises_runtime_error_after_max_tries():
    sliver = Polygon([(0, 0), (100, 100), (100, 100.001)])
    with pytest.raises(RuntimeError, match="Failed to find a free point"):
        gu._sample_point_in_free_space(
            sliver, max_tries=5, rng=np.random.default_rng(0)
        )


@pytest.mark.parametrize("case", ["empty", "disjoint"])
def test_sample_point_empty_free_space_raises_runtime_error(case):
    if case == "empty":
        free, constraint = box(0, 0, 10, 10).difference(box(-1, -1, 11, 11)), None
    else:
        free, constraint = box(0, 0, 1, 1), box(5, 5, 6, 6)
    with pytest.raises(RuntimeError):
        gu._sample_point_in_free_space(
            free, constraint=constraint, max_tries=5, rng=np.random.default_rng(0)
        )


# --------------------------------------------------------------------------
# skeleton
# --------------------------------------------------------------------------


def test_make_skeleton_consistent_shapes_and_masks():
    polygons = [box(20, 20, 30, 30), box(60, 60, 75, 70)]
    skeleton, dist, crop = gu.make_skeleton(polygons, 1.0, 100, [100, 200])
    assert skeleton.shape == dist.shape == crop.shape == (100, 100)
    free = ~crop.astype(bool)
    assert not (skeleton & ~free).any()  # the skeleton lies in free space
    assert (dist[~free] == 0).all() and (dist[free] > 0).all()
    assert skeleton.any()
    assert crop.sum() > 0  # the (enlarged) obstacles are rasterized


# --------------------------------------------------------------------------
# position_polygons_nfp
# --------------------------------------------------------------------------


def _place(polygons, dims, seed=0, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gu.position_polygons_nfp(
            polygons, dimensions=dims, rng=np.random.default_rng(seed), **kwargs
        )


def test_nfp_places_without_overlap_including_periodic_images():
    dim = 400
    polygons = [_star(30, seed=s) for s in range(30)]
    world, _, not_placed, ghosts = _place(polygons, [0, dim], seed=1)
    assert len(world) == 30 and not not_placed
    assert _overlaps(world, tile=dim) == []
    assert all(0 <= p.centroid.x <= dim and 0 <= p.centroid.y <= dim for p in world)
    assert all(len(g) == 8 for g in ghosts)
    assert sum(p.area for p in world) == pytest.approx(
        sum(_star(30, seed=s).area for s in range(30))
    )  # shapes are only translated and rotated


def test_nfp_first_polygon_reproducible_with_seed():
    polygons = [_star(30, seed=s) for s in range(3)]
    a = _place(polygons, [0, 400], seed=5)[0][0]
    b = _place(polygons, [0, 400], seed=5)[0][0]
    c = _place(polygons, [0, 400], seed=6)[0][0]
    assert a.equals(b) and not a.equals(c)


def test_nfp_reproducible_with_seed():
    polygons = [_star(30, seed=s) for s in range(8)]
    a = _place(polygons, [0, 400], seed=5)[0]
    b = _place(polygons, [0, 400], seed=5)[0]
    assert all(p.equals(q) for p, q in zip(a, b, strict=True))


def test_nfp_different_seeds_differ():
    polygons = [_star(30, seed=s) for s in range(8)]
    a = _place(polygons, [0, 400], seed=5)[0]
    c = _place(polygons, [0, 400], seed=6)[0]
    assert any(not p.equals(q) for p, q in zip(a, c, strict=True))


def test_nfp_first_polygon_random_for_zero_max_convention():
    world = _place([_star(30, seed=1), _star(30, seed=2)], [0, 400], seed=3)[0]
    assert 0 < world[0].centroid.x < 400 and 0 < world[0].centroid.y < 400


def test_nfp_first_polygon_in_corner_for_n_n_convention():
    # The tutorial passes dimensions=[N, N]: rng.uniform(N, N) puts the first
    # polygon at the corner (N, N). Pinned as current behaviour.
    world = _place([_star(30, seed=1), _star(30, seed=2)], [400, 400], seed=3)[0]
    assert (world[0].centroid.x, world[0].centroid.y) == pytest.approx((400, 400))
    assert _overlaps(world, tile=400) == []


def test_nfp_positions_cover_all_placed_polygons():
    world, positions, _, _ = _place([_star(30, seed=s) for s in range(3)], [0, 400])
    assert len(positions) == len(world)


def test_nfp_positions_are_the_centroids_of_the_placed_polygons():
    world, positions, _, _ = _place([_star(30, seed=s) for s in range(4)], [0, 400])
    for w, (x, y) in zip(world, positions, strict=True):
        assert w.centroid.x == pytest.approx(x)
        assert w.centroid.y == pytest.approx(y)


def test_nfp_existing_world_is_kept_and_respected():
    dim = 400
    existing = translate(_star(30, seed=9), 200, 200)
    start_world = [existing]
    start_ghosts = [gu._create_ghost(existing, [0, dim])]
    world, _, _, ghosts = _place(
        [_star(30, seed=s) for s in range(10)],
        [0, dim],
        world=start_world,
        ghosts=start_ghosts,
    )
    assert world[0] is existing and len(world) == 11 and len(ghosts) == 11
    assert _overlaps(world, tile=dim) == []


def test_nfp_does_not_mutate_input_world():
    existing = translate(_star(30, seed=9), 200, 200)
    start_world = [existing]
    start_ghosts = [gu._create_ghost(existing, [0, 400])]
    _place([_star(30, seed=1)], [0, 400], world=start_world, ghosts=start_ghosts)
    assert len(start_world) == 1 and len(start_ghosts) == 1


def test_nfp_world_without_ghosts_builds_ghosts():
    existing = translate(_star(30, seed=9), 200, 200)
    world, _, not_placed, ghosts = _place(
        [_star(30, seed=1)], [0, 400], world=[existing]
    )
    assert len(world) == 2 and not not_placed
    assert len(ghosts) == len(world)
    assert all(len(g) == 8 for g in ghosts)
    assert _overlaps(world, tile=400) == []


def test_nfp_world_ghosts_length_mismatch_raises_value_error():
    existing = translate(_star(30, seed=9), 200, 200)
    with pytest.raises(ValueError, match="differ"):
        _place(
            [_star(30, seed=1)],
            [0, 400],
            world=[existing],
            ghosts=[gu._create_ghost(existing, [0, 400])] * 2,
        )


def _full_world(dim):
    full = box(0, 0, dim, dim)
    return [full], [gu._create_ghost(full, [0, dim])]


def test_nfp_unplaceable_polygon_warns_and_is_returned():
    world, ghosts = _full_world(200)
    with pytest.warns(UserWarning, match="Could not place polygon"):
        placed, _, not_placed, _ = gu.position_polygons_nfp(
            [_star(20, seed=1), _star(20, seed=2)],
            dimensions=[0, 200],
            world=world,
            ghosts=ghosts,
            rng=np.random.default_rng(0),
        )
    assert len(placed) == 1 and len(not_placed) == 2


def test_nfp_break_immediately_returns_remaining_polygons():
    world, ghosts = _full_world(200)
    with pytest.warns(UserWarning):
        _, _, remaining, _ = gu.position_polygons_nfp(
            [_star(20, seed=s) for s in range(3)],
            dimensions=[0, 200],
            world=world,
            ghosts=ghosts,
            break_immediately=True,
            rng=np.random.default_rng(0),
        )
    assert len(remaining) == 3  # the first one failed, all are returned


def _tried_orientations(monkeypatch, max_rot):
    angles: list[float] = []
    original = gu.rotate

    def spy(geom, angle, *args, **kwargs):
        angles.append(angle)
        return original(geom, angle, *args, **kwargs)

    monkeypatch.setattr(gu, "rotate", spy)
    world, ghosts = _full_world(200)
    with pytest.warns(UserWarning):
        gu.position_polygons_nfp(
            [_star(20, seed=1)],
            dimensions=[0, 200],
            max_rot=max_rot,
            world=world,
            ghosts=ghosts,
            rng=np.random.default_rng(0),
        )
    return len({round(float(a) % 360, 6) for a in angles}), len(angles)


@pytest.mark.parametrize("max_rot", [3, 5, 6, 8, 16])
def test_nfp_tries_max_rot_distinct_orientations(monkeypatch, max_rot):
    distinct, tries = _tried_orientations(monkeypatch, max_rot)
    assert tries == max_rot
    assert distinct == max_rot


@pytest.mark.parametrize("seed", range(5))
def test_nfp_rotated_polygon_is_placed_without_overlap(seed):
    # Elongated shapes: the polygon placed must be the rotated one whose NFP was
    # used, otherwise rotations other than the first overlap their neighbours.
    polygons = [box(0, 0, 60, 8) for _ in range(30)]
    world, _, _, _ = _place(polygons, [0, 200], seed=seed, max_rot=4)
    assert _overlaps(world, tile=200) == []


def test_nfp_not_placed_polygon_keeps_original_orientation():
    original = box(0, 0, 60, 8)
    world, ghosts = _full_world(200)
    with pytest.warns(UserWarning, match="Could not place polygon"):
        _, _, not_placed, _ = gu.position_polygons_nfp(
            [original],
            dimensions=[0, 200],
            max_rot=4,
            world=world,
            ghosts=ghosts,
            rng=np.random.default_rng(0),
        )
    expected = gu._translate_polygon(original, 0, 0)
    assert len(not_placed) == 1
    assert not_placed[0].equals_exact(expected, tolerance=1e-9)


def test_nfp_fully_blocked_constraint_does_not_crash():
    existing = translate(_star(60, seed=9), 300, 300)
    _, _, not_placed, _ = _place(
        [_star(30, seed=3)],
        [0, 600],
        world=[existing],
        ghosts=[gu._create_ghost(existing, [0, 600])],
        constraint=box(295, 295, 305, 305),
    )
    assert len(not_placed) == 1


# --------------------------------------------------------------------------
# compress
# --------------------------------------------------------------------------


def _centroid_distance(p, target=(0.0, 0.0)):
    return float(np.hypot(p.centroid.x - target[0], p.centroid.y - target[1]))


def test_compress_moves_towards_target_by_step_size():
    polygons = [
        translate(box(-5, -5, 5, 5), 300, 300),
        translate(box(-5, -5, 5, 5), 500, 100),
    ]
    out, ghosts = gu.compress(
        polygons, max_iter=1, step_size=10, direction=(0, 0), dimension=(0, 600)
    )
    for before, after in zip(polygons, out, strict=True):
        assert _centroid_distance(after) == pytest.approx(
            _centroid_distance(before) - 10
        )
        assert after.area == pytest.approx(before.area)
    assert ghosts == [[], []]  # no input ghosts: nothing to update


def test_compress_does_not_mutate_input():
    polygons = [translate(box(-5, -5, 5, 5), 300, 300)]
    snapshot = [p.wkt for p in polygons]
    gu.compress(polygons, max_iter=2, direction=(0, 0), dimension=(0, 600))
    assert [p.wkt for p in polygons] == snapshot


def test_compress_updates_ghosts_of_moved_polygons():
    p = translate(box(-5, -5, 5, 5), 300, 300)
    out, ghosts = gu.compress(
        [p],
        max_iter=1,
        direction=(0, 0),
        dimension=(0, 600),
        input_ghosts=[gu._create_ghost(p, (0, 600))],
    )
    expected = gu._create_ghost(out[0], (0, 600))
    assert all(a.equals(b) for a, b in zip(ghosts[0], expected, strict=True))


def test_compress_blocked_by_neighbour():
    a = box(-5, -5, 5, 5)  # centroid at the target: never moves
    b = translate(box(-5, -5, 5, 5), 15, 0)  # a 10-unit step would hit `a`
    out, _ = gu.compress(
        [a, b], max_iter=3, step_size=10, direction=(0, 0), dimension=(0, 600)
    )
    assert out[0].equals(a) and out[1].equals(b)


def test_compress_polygon_outside_boundary_does_not_move():
    p = translate(box(-5, -5, 5, 5), 800, 800)
    out, _ = gu.compress([p], max_iter=2, direction=(0, 0), dimension=(0, 600))
    assert out[0].equals(p)


def test_compress_degenerate_boundary_moves_nothing():
    # What placement_routine produces with dimensions=[0, N]; see the xfail below.
    p = translate(box(-5, -5, 5, 5), 300, 300)
    out, _ = gu.compress([p], max_iter=3, direction=(0, 0), dimension=(0, 0))
    assert out[0].equals(p)


def test_compress_random_direction_reproducible():
    polygons = [
        translate(box(-5, -5, 5, 5), 300, 300),
        translate(box(-5, -5, 5, 5), 100, 400),
    ]
    kwargs = {"max_iter": 3, "random_direction": True, "dimension": (0, 600)}
    a, _ = gu.compress(polygons, rng=np.random.default_rng(1), **kwargs)
    b, _ = gu.compress(polygons, rng=np.random.default_rng(1), **kwargs)
    c, _ = gu.compress(polygons, rng=np.random.default_rng(2), **kwargs)
    assert all(p.equals(q) for p, q in zip(a, b, strict=True))
    assert not all(p.equals(q) for p, q in zip(a, c, strict=True))


def test_compress_rotation_keeps_area_and_avoids_overlap():
    polygons = [translate(_star(15, seed=s), 100 + 80 * s, 300) for s in range(4)]
    out, _ = gu.compress(
        polygons,
        max_iter=5,
        random_direction=True,
        rotation_angle=360,
        dimension=(0, 600),
        rng=np.random.default_rng(0),
    )
    assert [p.area for p in out] == pytest.approx([p.area for p in polygons])
    assert _overlaps(out) == []


# --------------------------------------------------------------------------
# placement_routine
# --------------------------------------------------------------------------


def _routine(dims, monkeypatch, polygons):
    calls = []
    original = gu.compress

    def spy(world, **kwargs):
        out, ghosts = original(world, **kwargs)
        moved = sum(1 for a, b in zip(world, out, strict=True) if not a.equals(b))
        calls.append((list(kwargs["dimension"]), moved))
        return out, ghosts

    monkeypatch.setattr(gu, "compress", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gu.placement_routine(
            polygons,
            dims,
            max_iter_placement=3,
            max_rot=4,
            max_iter_shuffle=3,
            step_size=10,
        )
    return result, calls


def _routine_polygons():
    # Two small polygons that fit, one that never fits: forces several rounds.
    return [_star(15, seed=1), _star(15, seed=2), box(-300, -300, 300, 300)]


def test_placement_routine_returns_four_lists_and_places_all():
    polygons = [_star(30, seed=s) for s in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world, positions, not_placed, ghosts = gu.placement_routine(
            polygons, [400, 400], max_rot=8
        )
    assert isinstance(world, list) and isinstance(ghosts, list)
    assert len(world) == 20 and not_placed == []
    assert all(len(g) == 8 for g in ghosts)
    assert _overlaps(world, tile=400) == []


def test_placement_routine_compress_runs_with_n_n_convention(monkeypatch):
    (world, _, not_placed, _), calls = _routine(
        [200, 200], monkeypatch, _routine_polygons()
    )
    assert len(world) == 2 and len(not_placed) == 1
    assert calls and all(dim == [0, 200] for dim, _ in calls)
    assert sum(moved for _, moved in calls) > 0  # the shuffle does move polygons
    assert _overlaps(world, tile=200) == []


def test_placement_routine_zero_max_convention_is_not_supported(monkeypatch):
    # dimensions is documented as [N, N]; [0, N] makes compress get [0, 0] and
    # move nothing (geo_utils.py:921)
    _, calls = _routine([0, 200], monkeypatch, _routine_polygons())
    assert calls and all(dim == [0, 0] for dim, _ in calls)
    assert sum(moved for _, moved in calls) == 0


def _seeded_routine(seed):
    # Several rounds (one polygon never fits), so that compress runs as well.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gu.placement_routine(
            _routine_polygons(),
            [200, 200],
            max_iter_placement=3,
            max_rot=4,
            max_iter_shuffle=3,
            step_size=10,
            rng=np.random.default_rng(seed),
        )[0]


def test_placement_routine_reproducible_with_seed():
    a, b = _seeded_routine(1), _seeded_routine(1)
    assert len(a) == len(b)
    assert all(p.equals(q) for p, q in zip(a, b, strict=True))


def test_placement_routine_different_seeds_differ():
    a, c = _seeded_routine(1), _seeded_routine(2)
    assert len(a) != len(c) or any(not p.equals(q) for p, q in zip(a, c, strict=True))


def test_placement_routine_positions_cover_placed_polygons(monkeypatch):
    (world, positions, _, _), _ = _routine([200, 200], monkeypatch, _routine_polygons())
    assert len(positions) == len(world)


def test_placement_routine_positions_match_final_world(monkeypatch):
    # compress moves polygons after they were placed: the positions must follow
    (world, positions, _, _), calls = _routine(
        [200, 200], monkeypatch, _routine_polygons()
    )
    assert sum(moved for _, moved in calls) > 0
    for w, (x, y) in zip(world, positions, strict=True):
        assert w.centroid.x == pytest.approx(x)
        assert w.centroid.y == pytest.approx(y)


def test_placement_routine_positions_include_start_world():
    start = translate(_star(20, seed=9), 100, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world, positions, _, _ = gu.placement_routine(
            [_star(20, seed=1)],
            [200, 200],
            max_rot=4,
            start_world=[start],
            rng=np.random.default_rng(0),
        )
    assert len(world) == len(positions) == 2
    assert positions[0] == pytest.approx(tuple(start.centroid.coords[0]))


# --------------------------------------------------------------------------
# make_random_crystal_2d
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lattice", ["square", "triangular", "hexagonal"])
def test_crystal_has_n_polygons_without_overlap(py_random, lattice):
    py_random.seed(1)
    crystal = gu.make_random_crystal_2d(_star(30, seed=1), 10, lattice_type=lattice)
    assert len(crystal) == 10
    assert _overlaps(crystal) == []
    assert all(
        a.intersects(b) is False
        for i, a in enumerate(crystal)
        for b in crystal[i + 1 :]
    )


def test_crystal_seed_is_first_and_centred_at_origin(py_random):
    py_random.seed(1)
    seed = _star(30, seed=1, center=(400, 500))
    crystal = gu.make_random_crystal_2d(seed, 4)
    assert (crystal[0].centroid.x, crystal[0].centroid.y) == pytest.approx(
        (0, 0), abs=1e-9
    )
    assert crystal[0].area == pytest.approx(seed.area)
    assert len(gu.make_random_crystal_2d(seed, 4, exclude_first=True)) == 3


def test_crystal_reproducible_with_python_random_seed(py_random):
    py_random.seed(7)
    a = gu.make_random_crystal_2d(_star(30, seed=1), 8, chaos=[0.9, 1.1])
    py_random.seed(7)
    b = gu.make_random_crystal_2d(_star(30, seed=1), 8, chaos=[0.9, 1.1])
    assert all(p.equals(q) for p, q in zip(a, b, strict=True))


def test_crystal_regular_squares_sit_on_a_lattice(py_random):
    py_random.seed(2)
    crystal = gu.make_random_crystal_2d(box(-5, -5, 5, 5), 8, regular=True)
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in crystal])
    spacing = np.abs(centroids[np.abs(centroids) > 1e-9]).min()
    assert spacing > 10  # squares of side 10 never touch
    assert np.allclose(centroids / spacing, np.round(centroids / spacing), atol=1e-6)


def test_crystal_chaos_keeps_polygons_apart(py_random):
    py_random.seed(3)
    crystal = gu.make_random_crystal_2d(_star(20, seed=1), 8, chaos=[0.95, 1.2, 20])
    assert len(crystal) == 8 and _overlaps(crystal) == []


def test_crystal_constraint_is_respected(py_random):
    py_random.seed(4)
    constraint = box(-150, -150, 150, 150)
    crystal = gu.make_random_crystal_2d(_star(15, seed=1), 6, constraint=constraint)
    assert all(constraint.intersects(p) for p in crystal[1:])


def test_crystal_large_seed():
    gu.make_random_crystal_2d(_star(400, seed=1), 3)


_HANG = """
import random
from shapely.geometry import box
import cyanomembranes.geo_utils as g
random.seed(0)
g.make_random_crystal_2d(
    box(-5, -5, 5, 5), 50, constraint=box(-40, -40, 40, 40), max_attempts=200
)
"""


def test_crystal_full_constraint_raises_instead_of_hanging():
    # The constraint cannot hold 50 polygons: the loop must give up with a
    # RuntimeError after max_attempts consecutive failures (subprocess so that
    # a regression cannot freeze the test run).
    result = subprocess.run(
        [sys.executable, "-c", _HANG],
        timeout=30,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr


def test_crystal_max_attempts_does_not_limit_feasible_crystals():
    random.seed(0)
    placed = gu.make_random_crystal_2d(box(-5, -5, 5, 5), 12, max_attempts=50)
    assert len(placed) == 12
