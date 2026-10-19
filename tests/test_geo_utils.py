"""Tests for the geometry helpers used by the lattice diffusion examples."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from rasterio.errors import ShapeSkipWarning
from shapely import to_wkt, wkt
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, Point, Polygon, box

from cyanomembranes import geo_utils


def test_get_lattice_directions_square():
    directions = np.array(geo_utils._get_lattice_directions("square"), dtype=float)
    assert directions.shape == (4, 2)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.allclose(directions.sum(axis=0), 0.0)
    assert len({tuple(d) for d in directions}) == 4


def test_get_lattice_directions_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported lattice type"):
        geo_utils._get_lattice_directions("bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("resolution", "extent", "expected"),
    [
        (1, (0, 0, 10, 6), (6, 10)),
        (2, (0, 0, 10, 6), (3, 5)),
        (3, (0, 0, 10, 6), (2, 4)),
    ],
)
def test_make_raster_shape_and_dtype(resolution, extent, expected):
    raster = geo_utils.make_raster(box(1.5, 1.5, 2.5, 2.5), resolution, *extent)
    assert raster.shape == expected  # (height, width) = ceil(extent / resolution)
    assert raster.dtype == np.uint8


def test_make_raster_polygon_cells_and_orientation():
    # Polygon edges lie inside pixels (not on pixel borders), so the 3x3 block
    # of touched cells is unambiguous. Row 0 is the top (largest y).
    raster = geo_utils.make_raster(box(0.5, 0.5, 2.5, 2.5), 1, 0, 0, 10, 10)
    assert raster.shape == (10, 10)
    assert raster.sum() == 9
    assert raster[7:10, 0:3].all()
    assert raster[0:7].sum() == 0


def test_make_raster_negative_origin():
    # Ghost case: domain (-5, 5), polygon in the lower-left quadrant.
    raster = geo_utils.make_raster(box(-2.5, -2.5, -0.5, -0.5), 1, -5, -5, 5, 5)
    assert raster.shape == (10, 10)
    assert raster.sum() == 9
    assert raster[5:8, 2:5].all()


def test_make_raster_empty_geometry_all_zero():
    # No obstacles (as in tutorial section 1) must give a completely free raster.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ShapeSkipWarning)
        raster = geo_utils.make_raster(GeometryCollection(), 1, 0, 0, 5, 5)
    assert raster.shape == (5, 5)
    assert raster.sum() == 0


@pytest.mark.xfail(
    reason="Low: rasterio ShapeSkipWarning for empty geometry (geo_utils.make_raster)",
    strict=True,
)
def test_make_raster_empty_geometry_emits_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        geo_utils.make_raster(GeometryCollection(), 1, 0, 0, 5, 5)
    assert not [w for w in caught if issubclass(w.category, ShapeSkipWarning)]


# --------------------------------------------------------------------------
# readwkt (obstacle files of the ensemble experiments)
# --------------------------------------------------------------------------


def _write_wkt(path, geometries):
    path.write_text("".join(to_wkt(g) + "\n" for g in geometries))
    return path


def test_readwkt_roundtrip(tmp_path):
    shapes = [
        Point(10, 20).buffer(5),
        box(0, 0, 3, 4),
        Point(2500.5, 1900.25).buffer(77),
    ]
    polygons = geo_utils.readwkt(_write_wkt(tmp_path / "shapes.wkt", shapes))
    assert len(polygons) == 3
    assert all(isinstance(p, Polygon) for p in polygons)
    for original, loaded in zip(shapes, polygons, strict=True):
        assert loaded.area == pytest.approx(original.area, rel=1e-8)
        assert loaded.bounds == pytest.approx(original.bounds, abs=1e-5)


def test_readwkt_skips_blank_lines_and_accepts_str_path(tmp_path):
    path = tmp_path / "shapes.wkt"
    path.write_text(f"\n{to_wkt(box(0, 0, 1, 1))}\n   \n\n{to_wkt(box(2, 2, 4, 4))}\n")
    polygons = geo_utils.readwkt(str(path))
    assert [p.area for p in polygons] == [1.0, 4.0]


def test_readwkt_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "empty.wkt"
    path.write_text("")
    assert geo_utils.readwkt(path) == []


def test_readwkt_invalid_line_raises(tmp_path):
    path = tmp_path / "bad.wkt"
    path.write_text("POLYGON((0 0, 1 1\n")
    with pytest.raises(GEOSException):
        geo_utils.readwkt(path)


def test_wkt_transfer_is_lossless():
    # The ensemble sends geometries to the workers as `.wkt` strings.
    polygon = Point(2500.123456789, 1900.987654321).buffer(77.7777)
    restored = wkt.loads(polygon.wkt)
    assert restored.equals_exact(polygon, tolerance=0)
    assert restored.area == polygon.area


@pytest.mark.xfail(
    reason=(
        "Low: polygons with edges exactly on cell borders are rasterized"
        " asymmetrically in x and y (geo_utils.make_raster)"
    ),
    strict=True,
)
def test_make_raster_symmetric_shape_gives_symmetric_raster():
    square = box(10, 10, 14, 14)
    ring = square.buffer(2).difference(square)  # symmetric under x <-> y
    raster = geo_utils.make_raster(ring, 1, 0, 0, 40, 40)[::-1]  # row = y
    assert np.array_equal(raster, raster.T)
