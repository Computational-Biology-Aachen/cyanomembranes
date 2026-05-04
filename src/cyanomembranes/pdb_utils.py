"""
Utilities for working with PDB (Protein Data Bank) files.

This module provides functions to read, process, and analyze PDB files, including:
- Reading PDB files into Bio.PDB structures
- Removing membrane dummy atoms
- Calculating total mass and charge
- Generating 2D concave polygons from protein structures
- Batch processing of multiple protein files
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
    total_mass = 0
    for atom in structure.get_atoms():
        total_mass += atom.mass
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


def get_total_charge(structure: Structure) -> float:
    """
    Calculate the total charge of a protein structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure.Structure
        Protein structure object.

    Returns
    -------
    float
        Total charge of the protein.
    """
    total_charge = 0
    for atom in structure.get_atoms():
        if (charge := atom.get_charge()) is not None:
            total_charge += charge
    return total_charge


def make_polygon(
    in_file: Path,
    out_file: Path | None,
    viewing_axis: np.ndarray | None = None,
    atom_z_range: tuple | None = None,
) -> tuple[np.ndarray, list[np.ndarray], Polygon]:
    """
    Create a 2D concave polygon (outline) of a protein structure as viewed along a
    given axis.

    Projects atomic coordinates onto a plane orthogonal to the viewing axis, computes a
    concave hull, and saves the result as a WKT polygon file.

    Parameters
    ----------
    in_file : Path
        Path to the input PDB file.
    out_file : Path
        Path to save the WKT polygon file.
    viewing_axis : np.ndarray, optional
        3D vector specifying the viewing direction (default: [0, 0, 1]).
    atom_z_range : tuple or None, optional
        If provided, only atoms within this z-range are projected (default: None).

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray], Polygon]
        Projected coordinates, edge points, and the concave polygon.
    """
    if viewing_axis is None:
        viewing_axis = np.array([0.0, 0.0, 1.0])

    # Parse PDB file
    structure = read_pdb(in_file)

    # Define viewing axis
    viewing_axis /= np.linalg.norm(viewing_axis)  # Normalize to unit vector

    # Project coordinates onto 2D plane orthogonal to viewing axis
    if atom_z_range:
        support_vector = np.array([0, 0, (atom_z_range[0] + atom_z_range[1]) / 2])
    else:
        support_vector = np.array([0, 0, 0])

    projected_coords = []
    for atom in structure.get_atoms():
        coords = atom.get_coord()

        if atom_z_range and not (atom_z_range[0] <= coords[-1] <= atom_z_range[1]):
            continue

        projection = np.dot(coords - support_vector, viewing_axis) * viewing_axis
        projected_coords.append(coords - projection)

    projected_coords = np.array(projected_coords)

    Point_lst = [Point(p[0], p[1]) for p in projected_coords]

    concave_hull, edge_points = geo_algorithms.alpha_shape(
        Point_lst, alpha=0.3
    )  # use concave hull by alphashape
    concave_hull = cast(Polygon, concave_hull)

    if out_file:
        # Save to WKT (Well-Known Text) format
        with out_file.open("w") as f:
            f.write(concave_hull.wkt)

    return projected_coords, edge_points, concave_hull


def process_proteins(in_path: Path, out_path: Path, pdb_files: list[str]) -> dict:
    """
    Batch process a list of protein PDB files: clean, polygonize, and extract
    properties.

    For each PDB file, removes membrane dummy atoms, generates a 2D polygon outline,
    reads the cleaned structure, and computes total mass and charge.

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
        Dictionary mapping protein names to their structure, mass, charge, and polygon.
    """
    out_path.mkdir(exist_ok=True, parents=True)
    proteins = {}
    for pdbf in pdb_files:
        ls_dir = [f.name for f in out_path.iterdir()]
        wo_file = pdbf.replace(".pdb", "_wo.pdb")
        wkt_file = pdbf.replace(".pdb", ".wkt")
        if (wo_file not in ls_dir) and (wkt_file not in ls_dir):
            remove_membrane(
                in_path / pdbf,
                out_path / pdbf.replace(".pdb", "_wo.pdb").replace(".trpdb", "_wo.pdb"),
            )
            make_polygon(
                out_path / pdbf.replace(".pdb", "_wo.pdb").replace(".trpdb", "_wo.pdb"),
                out_path / pdbf.replace(".pdb", ".wkt").replace(".trpdb", ".wkt"),
            )
        structure = read_pdb(
            out_path / pdbf.replace(".pdb", "_wo.pdb").replace(".trpdb", "_wo.pdb")
        )
        polygon = geo_utils.readwkt(
            out_path / pdbf.replace(".pdb", ".wkt").replace(".trpdb", ".wkt")
        )
        total_mass = get_total_mass(structure)
        total_charge = get_total_charge(structure)
        center_of_mass = get_center_of_mass(structure)

        for s in [".pdb", ".trpdb"]:
            pdbf = pdbf.removesuffix(s) if pdbf.endswith(s) else pdbf  # noqa: PLW2901

        proteins[pdbf] = {
            "structure": structure,
            "total_mass": total_mass,
            "total_charge": total_charge,
            "center_of_mass": center_of_mass,
            "polygon": polygon,
        }

        (pd.DataFrame(proteins)).to_csv(out_path / "info.csv")

    return proteins
