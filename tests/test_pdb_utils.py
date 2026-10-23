"""Tests for `cyanomembranes.pdb_utils`.

All inputs are written to `tmp_path` from a tiny synthetic structure (no real PDB
data). Every strict xfail names one confirmed bug (see INFO/OPEN_ITEMS.md, section
"pdb_utils.py").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

import cyanomembranes.geo_utils as gu
import cyanomembranes.pdb_utils as pu

pytestmark = pytest.mark.filterwarnings(
    "ignore::Bio.PDB.PDBExceptions.PDBConstructionWarning"
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# name, residue, residue number, x, y, z, element
PROTEIN_ATOMS = [
    ("N", "ALA", 1, 0.0, 0.0, 0.0, "N"),
    ("CA", "ALA", 1, 5.0, 0.0, 2.0, "C"),
    ("C", "ALA", 1, 5.0, 5.0, 4.0, "C"),
    ("O", "ALA", 1, 0.0, 5.0, 6.0, "O"),
    ("CB", "ALA", 1, 2.0, 2.0, 3.0, "C"),
]
DUMMY_ATOMS = [
    ("P", "DUM", 2, 20.0, 20.0, 0.0, "P"),
    ("O", "DUM", 2, 25.0, 20.0, 10.0, "O"),
]
M_N, M_C, M_O = 14.0067, 12.0107, 15.9994  # Biopython atomic masses


def _line(serial, atom, charge=""):
    name, res, resi, x, y, z, element = atom
    return (
        f"ATOM  {serial:5d} {name:<4s} {res:>3s} A{resi:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}{charge:>2s}\n"
    )


def _write_pdb(path, atoms, charges=None):
    charges = charges or [""] * len(atoms)
    path.write_text(
        "".join(
            _line(i + 1, a, c)
            for i, (a, c) in enumerate(zip(atoms, charges, strict=True))
        )
    )
    return path


@pytest.fixture
def pdb_file(tmp_path):
    return _write_pdb(tmp_path / "a.pdb", PROTEIN_ATOMS + DUMMY_ATOMS)


@pytest.fixture
def clean_pdb_file(tmp_path):
    return _write_pdb(tmp_path / "clean.pdb", PROTEIN_ATOMS)


def _coords(structure):
    return np.array([a.get_coord() for a in structure.get_atoms()])


# --------------------------------------------------------------------------
# read_pdb / remove_membrane
# --------------------------------------------------------------------------


def test_read_pdb_atom_count_and_coordinates(pdb_file):
    s = pu.read_pdb(pdb_file)
    expected = [a[3:6] for a in PROTEIN_ATOMS + DUMMY_ATOMS]
    assert len(list(s.get_atoms())) == 7
    assert np.allclose(_coords(s), expected)


def test_read_pdb_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pu.read_pdb(tmp_path / "missing.pdb")


def test_remove_membrane_drops_dummies_keeps_protein(pdb_file, tmp_path):
    out = tmp_path / "a_wo.pdb"
    pu.remove_membrane(pdb_file, out)
    s = pu.read_pdb(out)
    assert {r.resname for r in s.get_residues()} == {"ALA"}
    assert np.allclose(_coords(s), [a[3:6] for a in PROTEIN_ATOMS])


def test_remove_membrane_without_dummies_keeps_everything(clean_pdb_file, tmp_path):
    out = tmp_path / "out.pdb"
    pu.remove_membrane(clean_pdb_file, out)
    assert len(list(pu.read_pdb(out).get_atoms())) == 5


# --------------------------------------------------------------------------
# mass, centre of mass, charge
# --------------------------------------------------------------------------


def test_get_total_mass_in_kda(clean_pdb_file):
    s = pu.read_pdb(clean_pdb_file)
    assert pu.get_total_mass(s) == pytest.approx((M_N + 3 * M_C + M_O) / 1000)


def test_get_center_of_mass_is_mass_weighted(clean_pdb_file):
    s = pu.read_pdb(clean_pdb_file)
    masses = np.array([M_N, M_C, M_C, M_O, M_C])
    expected = (masses[:, None] * _coords(s)).sum(axis=0) / masses.sum()
    assert np.allclose(pu.get_center_of_mass(s), expected)


def test_get_center_of_mass_equal_masses_is_centroid(tmp_path):
    atoms = [("C1", "ALA", 1, 0, 0, 0, "C"), ("C2", "ALA", 1, 2, 4, 6, "C")]
    s = pu.read_pdb(_write_pdb(tmp_path / "c.pdb", atoms))
    assert np.allclose(pu.get_center_of_mass(s), [1, 2, 3])


@pytest.fixture
def unknown_element_file(tmp_path):
    atoms = [*PROTEIN_ATOMS, ("X", "ALA", 1, 2.0, 2.0, 3.0, "X")]
    return _write_pdb(tmp_path / "x.pdb", atoms)


def test_get_center_of_mass_ignores_atoms_without_mass(
    unknown_element_file, clean_pdb_file
):
    # the atom with unknown element has NaN mass and must not shift the result
    ref = pu.get_center_of_mass(pu.read_pdb(clean_pdb_file))
    got = pu.get_center_of_mass(pu.read_pdb(unknown_element_file))
    assert np.allclose(got, ref)


@pytest.mark.xfail(
    reason="Low: get_total_mass returns NaN when one atom has an unknown element,"
    " get_center_of_mass skips such atoms (pdb_utils.py:91-94)",
    strict=True,
)
def test_get_total_mass_ignores_atoms_without_mass(unknown_element_file):
    s = pu.read_pdb(unknown_element_file)
    assert pu.get_total_mass(s) == pytest.approx((M_N + 3 * M_C + M_O) / 1000)


def test_get_total_charge_is_zero_without_charge_information(clean_pdb_file):
    assert pu.get_total_charge(pu.read_pdb(clean_pdb_file)) == 0


@pytest.mark.xfail(
    reason="Medium: get_total_charge is always 0 for PDB files, Biopython leaves"
    " atom.get_charge() None (pdb_utils.py:112-130)",
    strict=True,
)
def test_get_total_charge_reads_charge_column(tmp_path):
    f = _write_pdb(tmp_path / "q.pdb", PROTEIN_ATOMS, ["1+", "", "", "", ""])
    assert pu.get_total_charge(pu.read_pdb(f)) == 1


# --------------------------------------------------------------------------
# make_polygon
# --------------------------------------------------------------------------


def test_make_polygon_projects_onto_xy_plane(clean_pdb_file):
    coords, _, poly = pu.make_polygon(clean_pdb_file, None)
    assert coords.shape == (5, 3)
    assert np.allclose(coords[:, :2], _coords(pu.read_pdb(clean_pdb_file))[:, :2])
    assert np.allclose(coords[:, 2], 0)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area == pytest.approx(25.0)
    assert poly.bounds == pytest.approx((0, 0, 5, 5))


def test_make_polygon_returns_edge_points_on_the_outline(clean_pdb_file):
    _, edges, poly = pu.make_polygon(clean_pdb_file, None)
    assert len(edges) >= 3
    # each edge is a segment given by its two end points
    outline = poly.buffer(1e-9)
    assert all(outline.intersects(Point(p[0], p[1])) for e in edges for p in e)


def test_make_polygon_atom_z_range_filters_atoms(clean_pdb_file):
    # z of the atoms: 0, 2, 4, 6, 3 -> the oxygen at z=6 is dropped
    coords, _, poly = pu.make_polygon(clean_pdb_file, None, atom_z_range=(0, 4.5))
    assert len(coords) == 4
    assert not np.any((coords[:, 0] == 0) & (coords[:, 1] == 5))  # oxygen dropped
    assert poly.is_valid
    assert 0 < poly.area < 25


def test_make_polygon_atom_z_range_without_atoms_raises(clean_pdb_file):
    with pytest.raises(ValueError, match="at least four points"):
        pu.make_polygon(clean_pdb_file, None, atom_z_range=(100, 200))


def test_make_polygon_writes_wkt_readable_by_readwkt(clean_pdb_file, tmp_path):
    out = tmp_path / "poly.wkt"
    _, _, poly = pu.make_polygon(clean_pdb_file, out)
    (back,) = gu.readwkt(out)
    assert back.equals(poly)


def test_make_polygon_without_out_file_writes_nothing(clean_pdb_file, tmp_path):
    before = set(tmp_path.iterdir())
    pu.make_polygon(clean_pdb_file, None)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.xfail(
    reason="Low: viewing_axis other than z fails (QhullError), points are cut to"
    " (x, y) instead of in-plane coordinates (pdb_utils.py:184-189). Remove if the"
    " parameter is removed.",
    strict=True,
)
@pytest.mark.parametrize("axis", [(1.0, 0, 0), (0, 1.0, 0), (1.0, 1.0, 0)])
def test_make_polygon_other_viewing_axes(clean_pdb_file, axis):
    _, _, poly = pu.make_polygon(clean_pdb_file, None, viewing_axis=np.array(axis))
    assert poly.area > 0


@pytest.mark.xfail(
    reason="Low: viewing_axis is normalised in place and mutates the caller's"
    " array (pdb_utils.py:169)",
    strict=True,
)
def test_make_polygon_does_not_mutate_viewing_axis(clean_pdb_file):
    axis = np.array([0.0, 0.0, 2.0])
    pu.make_polygon(clean_pdb_file, None, viewing_axis=axis)
    assert np.allclose(axis, [0, 0, 2])


@pytest.mark.xfail(
    reason="Low: in-place normalisation fails for an integer viewing_axis"
    " (UFuncTypeError, pdb_utils.py:169)",
    strict=True,
)
def test_make_polygon_accepts_integer_viewing_axis(clean_pdb_file):
    _, _, poly = pu.make_polygon(clean_pdb_file, None, viewing_axis=np.array([0, 0, 1]))
    assert poly.area == pytest.approx(25.0)


# --------------------------------------------------------------------------
# process_proteins
# --------------------------------------------------------------------------


@pytest.fixture
def in_dir(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for name in ["a.pdb", "b.pdb", "c.trpdb"]:
        _write_pdb(d / name, PROTEIN_ATOMS + DUMMY_ATOMS)
    return d


def test_process_proteins_creates_files_and_properties(in_dir, tmp_path):
    out = tmp_path / "out" / "nested"
    res = pu.process_proteins(in_dir, out, ["a.pdb"])
    assert {p.name for p in out.iterdir()} == {"a_wo.pdb", "a.wkt", "info.csv"}
    assert list(res) == ["a"]
    entry = res["a"]
    assert entry["total_mass"] == pytest.approx((M_N + 3 * M_C + M_O) / 1000)
    assert entry["total_charge"] == 0
    assert len(list(entry["structure"].get_atoms())) == 5  # dummies removed
    assert entry["center_of_mass"].shape == (3,)
    assert entry["polygon"][0].area == pytest.approx(25.0)


def test_process_proteins_multiple_files_and_suffixes(in_dir, tmp_path):
    out = tmp_path / "out"
    res = pu.process_proteins(in_dir, out, ["a.pdb", "b.pdb"])
    assert list(res) == ["a", "b"]


def test_process_proteins_second_call_uses_cache(in_dir, tmp_path):
    out = tmp_path / "out"
    pu.process_proteins(in_dir, out, ["a.pdb"])
    stamps = {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.suffix != ".csv"}
    res = pu.process_proteins(in_dir, out, ["a.pdb"])
    assert stamps == {
        p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.suffix != ".csv"
    }
    assert res["a"]["total_mass"] == pytest.approx((M_N + 3 * M_C + M_O) / 1000)


def test_process_proteins_missing_input_raises(in_dir, tmp_path):
    with pytest.raises(FileNotFoundError):
        pu.process_proteins(in_dir, tmp_path / "out", ["missing.pdb"])


@pytest.mark.xfail(
    reason="Low: cache check uses `and`, so when only the .wkt is missing nothing is"
    " regenerated and readwkt raises FileNotFoundError (pdb_utils.py:232)",
    strict=True,
)
def test_process_proteins_regenerates_missing_wkt(in_dir, tmp_path):
    out = tmp_path / "out"
    pu.process_proteins(in_dir, out, ["a.pdb"])
    (out / "a.wkt").unlink()
    res = pu.process_proteins(in_dir, out, ["a.pdb"])
    assert (out / "a.wkt").exists()
    assert res["a"]["polygon"][0].area == pytest.approx(25.0)


@pytest.mark.xfail(
    reason="Low: for .trpdb input the cache file names keep the .trpdb name, so the"
    " cache never hits and files are rewritten (pdb_utils.py:229-232)",
    strict=True,
)
def test_process_proteins_trpdb_uses_cache(in_dir, tmp_path):
    out = tmp_path / "out"
    pu.process_proteins(in_dir, out, ["c.trpdb"])
    assert {p.name for p in out.iterdir()} >= {"c_wo.pdb", "c.wkt"}
    stamp = (out / "c_wo.pdb").stat().st_mtime_ns
    pu.process_proteins(in_dir, out, ["c.trpdb"])
    assert (out / "c_wo.pdb").stat().st_mtime_ns == stamp


@pytest.mark.xfail(
    reason="Low: info.csv holds only the proteins of the last call (earlier runs are"
    " overwritten) and objects as repr strings (pdb_utils.py:262)",
    strict=True,
)
def test_process_proteins_info_csv_accumulates_over_calls(in_dir, tmp_path):
    out = tmp_path / "out"
    pu.process_proteins(in_dir, out, ["a.pdb"])
    pu.process_proteins(in_dir, out, ["b.pdb"])
    info = pd.read_csv(out / "info.csv", index_col=0)
    assert set(info.columns) == {"a", "b"}
