"""
Utilities for working with PDB (Protein Data Bank) files.

This module provides functions to read, process, and analyze PDB files, including:
- Reading PDB files into Bio.PDB structures
- Removing membrane dummy atoms
- Calculating total mass and center of mass
- Generating 2D concave polygons from protein structures
- Batch processing of multiple protein files
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from shapely import Point, Polygon

from cyanomembranes import geo_algorithms, geo_utils

if TYPE_CHECKING:
    from pathlib import Path

    from Bio.PDB.Structure import Structure


# FIXME: find out what type PDBParser returns
def read_pdb(file: Path) -> Any:
    """
    Read a PDB file and return its structure object.

    Parameters
    ----------
    file : Path
        Path to the PDB file to read.

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        Parsed PDB structure object.
    """
    parser = PDBParser()
    return parser.get_structure("protein", str(file))


class ProteinSelect(Select):
    def accept_residue(  # type: ignore
        self, residue: Any
    ) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
        # Remove membrane dummies
        return residue.resname not in ["DUM"]


def remove_membrane(file: Path, out: Path) -> None:
    """
    Remove dummy atoms (e.g., membrane markers) from a PDB file and save the cleaned
    structure.

    Parameters
    ----------
    file : Path
        Path to the input PDB file.
    out : Path
        Path to save the cleaned PDB file.
    """
    structure = read_pdb(file)
    io = PDBIO()
    io.set_structure(structure)

    # this function somehow needs a class for selection
    io.save(str(out), select=ProteinSelect())


def get_total_mass(structure: Structure) -> float:
    """
    Calculate the total mass of a protein structure in kilodaltons (kDa).

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Protein structure object.

    Returns
    -------
    float
        Total mass in kDa.
    """
    masses = [atom.mass for atom in structure.get_atoms()]
    # atom without masses return NaN making normal sum to NaN
    total_mass = np.nansum(masses)
    return total_mass / 1_000


def get_center_of_mass(structure: Structure):
    atoms = list(structure.get_atoms())
    coords = np.array([atom.get_vector().get_array() for atom in atoms])
    masses = np.array([atom.mass for atom in atoms])

    # Filter nan
    nan_idx = np.isnan(masses)
    # atoms_without_mass = np.array(atoms)[nan_idx]
    coords = coords[~nan_idx]
    masses = masses[~nan_idx]
    total_mass = masses.sum()

    return (masses[:, np.newaxis] * coords).sum(axis=0) / total_mass


def make_polygon(
    in_file: Path,
    out_file: Path | None,
    atom_z_range: tuple | None = None,
) -> tuple[np.ndarray, list[np.ndarray], Polygon]:
    """
    Create a 2D concave polygon (outline) of a protein structure projected onto the
    x, y plane.

    Drops the z coordinate of the atoms, computes a concave hull, and optionally
    saves the result as a WKT polygon file.

    Parameters
    ----------
    in_file : Path
        Path to the input PDB file.
    out_file : Path | None
        Path to save the WKT polygon file. If None, nothing is written.
    atom_z_range : tuple or None, optional
        If provided, only atoms within this z-range are projected (default: None).

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], Polygon]
        Projected x, y coordinates with shape (n, 2), edge points, and the concave
        polygon.
    """
    # Parse PDB file
    structure = read_pdb(in_file)

    projected_coords = []
    for atom in structure.get_atoms():
        coords = atom.get_coord()

        if atom_z_range and not (atom_z_range[0] <= coords[-1] <= atom_z_range[1]):
            continue

        projected_coords.append(coords[:2])

    projected_coords_array = np.array(projected_coords)

    Point_lst = [Point(p[0], p[1]) for p in projected_coords_array]

    concave_hull, edge_points = geo_algorithms.alpha_shape(
        Point_lst, alpha=0.3
    )  # use concave hull by alphashape
    if not isinstance(concave_hull, Polygon):
        msg = f"concave hull is not a Polygon but {type(concave_hull)}. Check shape"
        raise ValueError(msg)

    if out_file:
        # Save to WKT (Well-Known Text) format
        with out_file.open("w") as f:
            f.write(concave_hull.wkt)

    return projected_coords_array, edge_points, concave_hull


def process_proteins(in_path: Path, out_path: Path, pdb_files: list[str]) -> dict:
    """
    Batch process a list of protein PDB files: clean, polygonize, and extract
    properties.

    For each PDB file, removes membrane dummy atoms, generates a 2D polygon outline,
    reads the cleaned structure, and computes total mass and center of mass.

    Parameters
    ----------
    in_path : Path
        Directory containing the input PDB files.
    out_path : Path
        Directory to save processed files (cleaned PDBs and polygons).
    pdb_files : list[str]
        List of PDB file names to process.

    Returns
    -------
    dict
        Dictionary mapping protein names to their structure, total mass,
        center of mass, and polygon.
    """
    out_path.mkdir(exist_ok=True, parents=True)
    proteins = {}
    for pdbf in pdb_files:
        stem = pdbf.removesuffix(".trpdb").removesuffix(".pdb")
        wo_path = out_path / f"{stem}_wo.pdb"
        wkt_path = out_path / f"{stem}.wkt"
        if not (wo_path.exists() and wkt_path.exists()):
            remove_membrane(
                in_path / pdbf,
                wo_path,
            )
            make_polygon(
                wo_path,
                wkt_path,
            )
        structure = read_pdb(wo_path)
        polygon = geo_utils.readwkt(wkt_path)
        total_mass = get_total_mass(structure)
        center_of_mass = get_center_of_mass(structure)

        proteins[stem] = {
            "structure": structure,
            "total_mass": total_mass,
            "center_of_mass": center_of_mass,
            "polygon": polygon,
        }

    (pd.DataFrame(proteins)).to_csv(out_path / "info.csv")

    return proteins
