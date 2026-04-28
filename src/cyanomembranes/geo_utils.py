"""
Geometric utilities for polygon operations, placement, and spatial analysis.

This module provides functions for:
- Lattice direction generation
- Polygon translation, projection, and reflection
- Sampling points in free space
- Polygon placement and packing (including DOP and NFP methods)
- Rasterization and medial axis computation
- Batch placement and compression routines
- Reading and writing polygons in WKT format
"""

from __future__ import annotations

import logging
import math
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely import LineString, MultiPolygon, Polygon, wkt
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, box, mapping
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from skimage.morphology import medial_axis
from tqdm import tqdm

from cyanomembranes.utils import XY, float2d_

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shapely.geometry.base import BaseGeometry


def _get_lattice_directions(
    lattice_type: Literal["square", "triangular", "hexagonal"],
) -> list[tuple[float, float]]:
    """
    Return lattice direction unit vectors for a given lattice type.

    Parameters
    ----------
    lattice_type : Literal["square", "triangular", "hexagonal"]
        The type of lattice for which to generate directions.

    Returns
    -------
    list[tuple[float, float]]
        List of direction unit vectors for the lattice.

    Raises
    ------
    ValueError
        If the lattice type is not supported.
    """
    if lattice_type == "square":
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if lattice_type == "triangular":
        angle_deg = [0, 60, 120, 180, 240, 300]
        return [
            (math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angle_deg
        ]
    if lattice_type == "hexagonal":
        return [
            (1, 0),
            (0.5, math.sqrt(3) / 2),
            (-0.5, math.sqrt(3) / 2),
            (-1, 0),
            (-0.5, -math.sqrt(3) / 2),
            (0.5, -math.sqrt(3) / 2),
        ]
    msg = f"Unsupported lattice type: {lattice_type}"
    raise ValueError(msg)


def _create_ghost(p: Polygon, dimensions: tuple[float, float]) -> list[Polygon]:
    """
    Create ghost polygons to simulate periodic boundary conditions.

    For a given polygon, generate translated copies (ghosts) in all eight neighboring
    positions around the original workspace, as if the workspace were tiled.

    Parameters
    ----------
    p : Polygon
        The original polygon to create ghosts for.
    dimensions : tuple[float, float]
        The (min, max) dimensions of the workspace.

    Returns
    -------
    list[Polygon]
        List of ghost polygons in the eight neighboring positions (upper left, upper
        center,upper right, left, right, lower left, lower center, lower right).
    """
    offsets = [
        (-1, 1),
        (0, 1),
        (1, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    return [
        translate(p, dimensions[1] * off[0], dimensions[1] * off[1]) for off in offsets
    ]


def _translate_polygon(p: Polygon, x: float, y: float) -> Polygon:
    """
    Translate a polygon so that its centroid moves to a specified (x, y) position.

    Parameters
    ----------
    p : Polygon
        The polygon to translate.
    x : float
        Target x-coordinate for the centroid.
    y : float
        Target y-coordinate for the centroid.

    Returns
    -------
    Polygon
        The translated polygon with its centroid at (x, y).
    """
    centroid = np.array(p.centroid.coords)[0]
    dx = x - centroid[0]
    dy = y - centroid[1]
    return translate(p, dx, dy)


def _sample_point_in_free_space(
    free_space: BaseGeometry,
    *,
    max_tries: int = 1000,
    constraint: None | Polygon = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Sample a random point within a free space polygon using rejection sampling.

    Parameters
    ----------
    free_space : BaseGeometry
        The geometry representing the available free space.
    max_tries : int, optional
        Maximum number of attempts to find a valid point (default: 1000).
    constraint : Polygon or None, optional
        Additional constraint geometry (default: None).
    rng : np.random.Generator or None, optional
        Random number generator (default: None, uses default_rng).

    Returns
    -------
    tuple[float, float]
        Coordinates of a sampled point.

    Raises
    ------
    RuntimeError
        If a valid point cannot be found after max_tries attempts.
    """
    rng = np.random.default_rng() if rng is None else rng
    if constraint:
        free_space = free_space.intersection(constraint)
    minx, miny, maxx, maxy = free_space.bounds
    prepared = prep(free_space)
    for _ in range(max_tries):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        point = Point(x, y)
        if prepared.contains(point):
            return x, y
    msg = f"Failed to find a free point after {max_tries} tries"
    raise RuntimeError(msg)


def _get_projection_directions_normals(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create evenly spaced unit direction vectors and their normals in 2D.

    Parameters
    ----------
    n : int
        Number of unit vectors to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (directions, normals), each of shape (n, 2).
    """
    # Select evenly space direction
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Because each opposite directions on the unit circle are
    # connected to one direction vector we need to remove doubles
    angles = np.where(
        angles >= np.pi, angles - np.pi, angles
    )  # remove negative directions angels > pi
    angles = np.unique(angles)

    # Create direction vectors using trigonometry
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1).round(3)

    # Create normals to the direction vectors
    normals = np.array([-directions[:, 1], directions[:, 0]]).T
    return directions, normals


def _project_polygon(polygon: Polygon, directions: np.ndarray) -> np.ndarray:
    """
    Project a polygon onto multiple axes (directions) in 2D.

    For each direction, computes the minimum and maximum projection of the polygon's
    exterior coordinates, effectively giving the extent of the polygon along each axis.

    Parameters
    ----------
    polygon : Polygon
        The polygon to project.
    directions : np.ndarray
        Array of unit vectors (shape (N, 2)) representing projection axes.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2), where each row contains (min, max) projection values for
        one axis.
    """
    coords = np.array(polygon.exterior.coords)
    projections = coords @ directions.T
    min_proj = projections.min(axis=0)
    max_proj = projections.max(axis=0)
    return np.stack((min_proj, max_proj), axis=1)  # shape (N, 2)


def _create_halfspace(
    minmax_lst: np.ndarray,
    *,
    directions: np.ndarray,
    normals: np.ndarray,
    extension: int = 100,
) -> list[LineString]:
    """Create LineStrings that define the halfspaces

    Parameters
    ----------
    minmax_lst : List[Tuple[float, float]]
        Min und Max Projection of a Polygon along an axis
    directions : np.ndarray
        Projection axis
    normals : np.ndarray
        Normal of a Projection axis
    extension : int, optional
        Extension factor for LineStrings, by default 100

    Returns
    -------
    List[LineString]
        List of LineStrings defining halfspaces
    """
    minmax = np.array(minmax_lst)  # shape (N, 2)
    dirs = np.repeat(directions, 2, axis=0)  # shape (2N, 2)
    norms = np.repeat(normals, 2, axis=0)  # shape (2N, 2)
    scalars = minmax.flatten()  # shape (2N,)

    starts = dirs * scalars[:, None]  # shape (2N, 2)
    ends = starts + norms  # shape (2N, 2)

    deltas = ends - starts
    lengths = np.linalg.norm(deltas, axis=1)

    # Extension in both directions
    factors = extension / lengths
    new_starts = starts - deltas * factors[:, None]
    new_ends = ends + deltas * factors[:, None]

    # Build LineStrings
    return [LineString([s, e]) for s, e in zip(new_starts, new_ends, strict=False)]


def _select_dop(polygon: Polygon, halfspace: list[LineString]) -> Polygon:
    """Identifies all polygons defined by intersections of Linestrings and selects DOP

    Parameters
    ----------
    polygon : Polygon
        Shapely polygon
    halfspace : List[LineString]
        Linestrings defining halfspaces

    Returns
    -------
    Polygon
        The DOP
    """
    merged = unary_union(halfspace)

    # Split lines at all intersection points
    split_lines = linemerge(merged)  # type: ignore - shapely
    split_lines = list(split_lines.geoms)  # type: ignore - shapely

    # Polygonize: Extract polygons from the network of lines
    polygons = list(polygonize(split_lines))

    # Select DOP
    scaled_body = scale(polygon, 0.5, 0.5)
    for i in polygons:
        # Need to scale down otherwise shapely will not see within relationship
        if scaled_body.within(i):
            return i
    msg = "Could not find a DOP"
    raise ValueError(msg)


def _sort_edges_by_angle(edge_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorts an edge_list according to their angle w.r.t the x-axis
    Parameters
    ----------
    edge_lst : np.ndarray
        Edge list
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Sorted indices and angles
    """
    # Get angles of edges with x-axis
    angles = np.arctan2(edge_lst[:, 1], edge_lst[:, 0])

    # Translate from (-pi/2, pi/2) to (0, 2pi)
    angles = np.mod(angles, 2 * np.pi)

    # Get sorted indices of edges w.r.t angeles
    sorted_indices = np.argsort(angles)

    return sorted_indices, angles


def readwkt(filepath: Path) -> list[Polygon]:
    """
    Read polygons from a WKT (Well-Known Text) file.

    Parameters
    ----------
    filepath : Path
        Path to the WKT file.

    Returns
    -------
    list[Polygon]
        List of polygons read from the file.
    """
    polygons = []
    with Path.open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                geom = wkt.loads(stripped_line)
                polygons.append(geom)
    return polygons


def make_raster(
    multipolygons: BaseGeometry,
    resolution: float,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> np.ndarray:
    """Rasterize multipolygon with possibly negative coordinates.

    Parameters
    ----------
    multipolygons : MultiPolygon
        Union of polygons (can include periodic copies)
    resolution : float
        Pixel size in world units
    minx, miny, maxx, maxy : float
        Full domain extent (can include negative values)

    Returns
    -------
    np.ndarray
        Raster image (uint8), y-axis top-down
    """
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # Start at lower-left corner (minx, miny) and flip y-axis for raster
    transform = Affine.translation(minx, maxy) * Affine.scale(resolution, -resolution)

    if isinstance(multipolygons, MultiPolygon):
        shapes = [(mapping(p), 1) for p in multipolygons.geoms]
    else:
        shapes = [(mapping(multipolygons), 1)]  # When only one polygon is present
    return rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def elongate_polygon(polygon: Polygon, factor: float = 1.001) -> Polygon:
    """Elongates a polygon in its the longest direction

    Parameters
    ----------
    polygon : Polygon
        Shapely polygon that should be elongated
    factor : float, optional
        Elongation factor

    Returns
    -------
    Polygon
        Elongated Shapely polygon
    """

    # Vector (longest side of MRR) and Angle
    min_rect = polygon.minimum_rotated_rectangle
    if not isinstance(min_rect, Polygon):
        msg = f"Expected polygon got {type(polygon).__name__}"
        raise TypeError(msg)
    coords = np.array(min_rect.exterior.coords)[:-1]  # We don't need the last point
    side_lengths = [np.linalg.norm(coords[i] - coords[(i + 1) % 4]) for i in range(4)]
    longest_side_index = np.argmax(side_lengths)  # Index of longest side
    longest_side = coords[(longest_side_index + 1) % 4] - coords[longest_side_index]
    angle = np.arctan2(
        longest_side[1], longest_side[0]
    )  # Angle between longest side and x-axis

    # Rotate to make longest side x-axis aligned
    rotated_polygon = rotate(polygon, -angle, use_radians=True)
    rotated_polygon_scaled = scale(rotated_polygon, xfact=factor, yfact=1)

    # Rotate back
    return rotate(rotated_polygon_scaled, angle, use_radians=True)


def point_reflect(polygon: Polygon | np.ndarray) -> Polygon | np.ndarray:
    """Creates a point reflection of polygon through (0,0)

    Parameters
    ----------
    polygon : Polygon | np.ndarray
        Shapely polygon or np.ndarray of coords
        specifying a polygon.

    Returns
    -------
    Polygon | np.ndarray
        Shapely polygon or np.ndarray of coords
        specifying a polygon. Output type depends
        on input type.

    Raises
    ------
    TypeError
        If input is neither a shapely polygon or an
        np.ndarray of coords a Type error is raised.
    """
    if isinstance(polygon, Polygon):
        return Polygon([-c for c in np.array(polygon.exterior.coords)])
    if isinstance(polygon, np.ndarray):
        return -polygon
    msg = "Expected np.ndarray or shapely polygon"
    raise TypeError(msg)


def polygon_translate(
    polygon: Polygon, *, xoff: float = 0.0, yoff: float = 0.0
) -> tuple[Polygon, np.ndarray, np.ndarray]:
    """Translates polygon and returns coordinates of new centroid.

    Parameters
    ----------
    polygon : Polygon
        Shapely polygon
    xoff : float, optional
        x-offset, by default 0.
    yoff : float, optional
        y-offset, by default 0.

    Returns
    -------
    Polygon
        translated polygon
    """
    polygon_copy = translate(polygon, xoff, yoff)
    x, y = polygon_copy.centroid.xy
    return polygon_copy, np.array(x[0]), np.array(y[0])


def get_edges(polygon: Polygon, *, ccw: bool = True) -> np.ndarray:
    """Generates a list of edge vector

    Parameters
    ----------
    polygon : Polygon
        Shapely polygon
    ccw : bool, optional
        travel direction, by default True (counterclockwise)

    Returns
    -------
    np.ndarray
        2D array of edges
    """
    # Reorient polygon coords if direction if clockwise
    if ccw is False:
        polygon = orient(polygon, sign=-1.0)
    coords = np.array(polygon.exterior.coords)
    return coords[1:] - coords[:-1]


def mink_sum(polygon_a: Polygon, polygon_b: Polygon) -> Polygon:
    """Gets the Minkowski sum of two convex polygons.
    Imagine polygon_B sliding around polygon_A, whereby the
    center of polygon_B moves along the edges

    Parameters
    ----------
    polygon_A : Polygon
        Shapely polygon
    polygon_B : Polygon
        Shapely polygon
    Returns
    -------
    Polygon
        Shapely polygon (Minkowski sum)
    """
    A = np.array(polygon_a.exterior.coords)[:-1]
    B = np.array(polygon_b.exterior.coords)[:-1]

    idx_A = np.lexsort((A[:, 0], A[:, 1]))[0]
    idx_B = np.lexsort((B[:, 0], B[:, 1]))[0]

    A = np.roll(A, -idx_A, axis=0)
    B = np.roll(B, -idx_B, axis=0)

    edges_A = np.roll(A, -1, axis=0) - A
    angles_A = np.arctan2(edges_A[:, 1], edges_A[:, 0]) % (2 * np.pi)
    edges_B = np.roll(B, -1, axis=0) - B
    angles_B = np.arctan2(edges_B[:, 1], edges_B[:, 0]) % (2 * np.pi)

    i = j = 0
    len_A = len(A)
    len_B = len(B)
    output = []

    while i < len_A or j < len_B:
        output.append(A[i % len_A] + B[j % len_B])
        angle_A = angles_A[i % len_A] if i < len_A else np.inf
        angle_B = angles_B[j % len_B] if j < len_B else np.inf

        if angle_A < angle_B:
            i += 1
        elif angle_A > angle_B:
            j += 1
        else:
            i += 1
            j += 1
    return Polygon(output)


def get_dop(
    polygon: Polygon,
    *,
    k: int = 8,
    extension: int = 100,
    hf: bool = False,
) -> Polygon | list[LineString]:
    """Creates a k-Discrete Orientation Polytope (DOP)

    Parameters
    ----------
    poly : Polygon
        Original Polygon
    k : int, optional
        Number of edges in the DOP, by default 8
    extension: int, optional
        Extension factor for LineString

    Returns
    -------
    Polygon
        The DOP
    """
    # 1. Get projection directions and normals
    directions, normals = _get_projection_directions_normals(n=k)
    # 2. Project polygon along directions
    minmax_lst = _project_polygon(polygon, directions)
    # 3. Create LineStrings that define the halfspaces
    halfspace = _create_halfspace(
        minmax_lst, directions=directions, normals=normals, extension=extension
    )
    if hf:
        return halfspace
    # 4. Select the dop from the polygon halfspaces
    dop = _select_dop(polygon, halfspace)

    # 5. reorient
    return orient(dop, sign=1.0)


def make_skeleton(
    polygons: Iterable[Polygon],
    resolution: float,
    max_dim: float,
    crop: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dops = [get_dop(i, k=32, extension=100_000) for i in polygons]
    raster = make_raster(
        unary_union(dops),  # type: ignore
        resolution,
        -max_dim,
        -max_dim,
        2 * max_dim,
        2 * max_dim,
    )

    crop = raster[slice(*crop), slice(*crop)]
    free = ~crop.astype(bool)

    skeleton, dist = medial_axis(free, return_distance=True)

    return skeleton, dist, crop


def position_polygons_nfp(
    polygons: list[Polygon],
    *,
    dimensions: list[float],
    max_rot: int = 8,
    extension: int = 100_000,
    world: list[Polygon] | None = None,
    constraint: Polygon | None = None,
    break_immediately: bool = False,
    ghosts: list[list[Polygon]] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[Polygon], list[Any], list, list[list[Polygon]]]:
    """Places polygons on a sheet (workspace) without overlaps using
    discrete-oriented polygons and no-fit polygons

    Parameters
    ----------
    polygons : List[Polygon]
        Shapely polygons that should be placed
    dimensions : List[float]
        Dimension of sheet
    max_rot : int, optional
        Max. number of rotation a shapely shape is allowed to do
        before assumed not placable, by default 8

    Returns
    -------
    Tuple[List[Polygon],List[Tuple[float]]]
        Placed polygons and positions (cnetroid) of new polygons
    """
    rng = np.random.default_rng() if rng is None else rng
    polygons = [
        _translate_polygon(p, 0, 0) for p in polygons
    ]  # ref point for NFP is centroid at (0,0)
    min_c, max_c = dimensions[0], dimensions[1]
    workspace = box(0, 0, max_c, max_c)
    not_placed = []
    if not isinstance(world, list):
        world = []
    if not isinstance(ghosts, list):
        ghosts = []
    positions = []

    for pos, p in enumerate(tqdm(polygons, desc="Placing polygons")):
        placed = False

        for i in range(max_rot):
            angle = i * (360 / max_rot)
            p = rotate(p, angle, origin="centroid")  # noqa: PLW2901

            if not world:
                x = rng.uniform(min_c, max_c)
                y = rng.uniform(min_c, max_c)
                new_p = _translate_polygon(p, x, y)
                world.append(new_p)
                ghosts.append(_create_ghost(new_p, dimensions))  # type: ignore
                placed = True
                break

            # Check edge overlap. We need ghosts for these
            ghost_idx = [
                idx for idx, w in enumerate(world) if workspace.contains(w) is False
            ]
            current_ghosts = np.array(ghosts)[ghost_idx].ravel()

            # Make extended world
            extended_world = world + list(current_ghosts)

            # Discrete oriented polygons
            dops_world = [get_dop(w, extension=extension) for w in extended_world]

            dop_p = get_dop(p, extension=extension)

            # Actual NFP calculation
            reflected_dop_p = point_reflect(dop_p)  # type: ignore
            if not isinstance(reflected_dop_p, Polygon):
                msg = "DOP is not a polygon"
                raise TypeError(msg)
            nfps = [mink_sum(dw, reflected_dop_p) for dw in dops_world]  # type: ignore
            nfp_union = unary_union(nfps)

            # Place in free space
            free_space = workspace.difference(nfp_union)

            if free_space.area == 0:
                continue

            try:
                random_point = _sample_point_in_free_space(
                    free_space, constraint=constraint
                )
                new_p = _translate_polygon(p, random_point[0], random_point[1])
                ghosts_new_p = _create_ghost(new_p, dimensions)  # type: ignore

                overlap = any(g.intersects(w) for g in ghosts_new_p for w in world)

                if overlap:
                    continue

                # Safe to place
                world.append(new_p)
                ghosts.append(ghosts_new_p)
                positions.append(random_point)
                placed = True
                break
            except RuntimeError:
                continue

        if placed is False:
            warnings.warn(
                "Could not place polygon. Proceed with the next one",
                stacklevel=2,
            )
            if break_immediately:
                return world, positions, polygons[pos:], ghosts
            not_placed.append(p)
    return world, positions, not_placed, ghosts


def compress(
    polygons: list[Polygon],
    max_iter: int = 10,
    input_ghosts: list[list[Polygon]] | None = None,
    *,
    step_size: int = 10,
    rotation_angle: int | None = None,
    direction: tuple[int, int] | np.ndarray = (0, 0),
    random_direction: bool = False,
    dimension: tuple[float, float] = (0, 2000),
    rng: np.random.Generator | None = None,
) -> tuple[list[Polygon], list[list[Polygon]]]:
    rng = np.random.default_rng() if rng is None else rng
    polygons = deepcopy(polygons)
    ghosts = deepcopy(input_ghosts) if input_ghosts else [[] for _ in polygons]
    boundary = box(dimension[0], dimension[0], dimension[1], dimension[1])
    direction = np.array(direction)

    for _ in range(max_iter):
        for i, p in enumerate(polygons):
            tree_world = STRtree(polygons)
            flat_ghosts = [g for gs in ghosts for g in gs]
            tree_ghosts = STRtree(flat_ghosts) if flat_ghosts else None
            if random_direction:
                angle = rng.uniform(0, 2 * np.pi)
                vector = float2d_(np.cos(angle), np.sin(angle))
            else:
                centroid = np.array(p.centroid.coords[0])
                vector = direction - centroid
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue
                vector /= norm

            dx, dy = vector * step_size
            moved_p = translate(p, dx, dy)

            if rotation_angle:
                angle = rng.uniform(0, rotation_angle)
                moved_p = rotate(moved_p, angle)

            # Skip if out of bounds
            if not boundary.intersects(moved_p):
                continue

            # Check collision with other real polygons
            candidates = tree_world.query(moved_p)
            if any(
                moved_p.intersects(polygons[c]) for c in candidates if polygons[c] != p
            ):
                continue

            # Check collision with ghosts
            if tree_ghosts:
                ghost_candidates = tree_ghosts.query(moved_p)
                if any(moved_p.intersects(flat_ghosts[g]) for g in ghost_candidates):
                    continue

            # Accept move
            polygons[i] = moved_p
            if input_ghosts:
                ghosts[i] = _create_ghost(moved_p, dimension)

    return polygons, ghosts


def placement_routine(
    polygons: list[Polygon],
    dimensions: list[float],
    *,
    max_iter_placement: int = 50,
    max_rot: int = 16,
    extension: int = 100_000,
    max_iter_shuffle: int = 10,
    step_size: int = 10,
    rotation_angle: int | None = None,
    direction: np.ndarray | None = None,
    random_direction: bool = True,
    constraint: Polygon | None = None,
    start_world: list[Polygon] | None = None,
    start_ghosts: list[list[Polygon]] | None = None,
) -> list[Polygon]:
    """Placement routine that positions polygons on surfaces.
    It runs position_polygons_nfp and compress iteratively until
    either all polygons have been placed or the maximal number of
    iteration has been reached

    Parameters
    ----------
    polygons : List[Polygon]
        List of polygons that should be placed
    dimensions : List[float]
        Dimension of the surface
    max_iter_placement : int, optional
        Max. number of placement iterations, by default 50
    max_rot : int, optional
        Max. number of rotations for a shape, by default 16
    extension : int, optional
        Extension of halfspaces. Necessary of DOP creation, by
        default 100_000
    max_iter_shuffle : int, optional
        Max. number of random movements/ see compress, by default 10
    step_size : int, optional
        Step size in the direction of movement in compress , by
        default 10
    direction : np.ndarray, optional
        Direction of movement in compress
    random_direction : bool, optional
        If True direction of movement is random (see compress), by
        default True
    constraint : Optional[Polygon], optional
        Polygon indicates an area where on a surface where new
        polygons should be placed, by default None
    start_world : Optional[List[Polygon]], optional
        Already placed polygons with which the procedure should star,
        by default None

    Returns
    -------
    List[Polygon]
        List of placed polygons
    """
    direction = np.array([0, 0]) if direction is None else direction

    not_placed = []
    placed_positions = []
    for i in range(max_iter_placement):
        logger.info("PLACEMENT ROUND %s", i + 1)
        if i == 0:
            world, positions, not_placed, ghosts = position_polygons_nfp(
                polygons,
                dimensions=dimensions,
                max_rot=max_rot,
                extension=extension,
                break_immediately=True,
                constraint=constraint,
                world=start_world,
                ghosts=start_ghosts,
            )
            placed_positions.extend(positions)
        else:
            len_np = len(not_placed)
            logger.info("Number of remaining shapes %s", len_np)
            if len_np == 0:
                break
            world, ghosts = compress(
                world,  # type: ignore
                max_iter=max_iter_shuffle,
                step_size=step_size,
                rotation_angle=rotation_angle,
                direction=direction,
                random_direction=random_direction,
                input_ghosts=ghosts,  # type: ignore
                dimension=[0, dimensions[0]],  # type: ignore
            )
            logger.info("WORLD %s", len(world))  # type: ignore
            world, positions, not_placed, ghosts = position_polygons_nfp(
                not_placed,
                dimensions=dimensions,
                max_rot=max_rot,
                extension=extension,
                world=world,
                break_immediately=True,
                constraint=constraint,
                ghosts=ghosts,
            )
            placed_positions.extend(positions)

    return world, positions, not_placed, ghosts  # type: ignore


def make_random_crystal_2d(
    seed: Polygon,
    n: int,
    *,
    lattice_type: Literal["square", "triangular", "hexagonal"] = "square",
    max_variation: float = 1.5,
    regular: bool = False,
    constraint: None | Polygon = None,
    exclude_first: bool = False,
) -> list[Polygon]:
    seed = _translate_polygon(seed, 0, 0)
    seed_dop = get_dop(seed, k=16)
    seed_dop_reverse = point_reflect(seed_dop)  # type: ignore

    nfp = mink_sum(seed_dop, seed_dop_reverse)  # type: ignore

    dir_dist = []
    for i in _get_lattice_directions(lattice_type):
        direction = np.array(i)
        centroid = np.array(seed.centroid.coords[0])
        line = LineString([centroid, centroid + direction * 10_000])
        intersections = line.intersection(nfp.exterior)
        dist = line.project(intersections)  # type: ignore
        dir_dist.append([direction, dist])

    max_dist = max(x[1] for x in dir_dist)

    placed = [seed]

    with tqdm(total=n - 1, desc="Placed Polygons") as pbar:
        while n > 1:
            frontier_idx = random.choice(range(len(placed)))
            fpoly = placed[frontier_idx]

            if regular is True:
                dist = max_dist
                direction, _ = random.choice(dir_dist)
            else:
                direction, dist = random.choice(dir_dist)

            dx, dy = direction * dist * max_variation
            new_polygon = translate(fpoly, dx, dy)

            tree = STRtree(placed)
            candidates = tree.query(new_polygon)

            overlaps = [c for c in candidates if placed[c].intersects(new_polygon)]

            if overlaps:
                continue  # skip this placement

            if constraint:
                if constraint.intersects(new_polygon):
                    placed.append(new_polygon)
                    n -= 1
                    pbar.update(1)
                else:
                    continue

            else:
                placed.append(new_polygon)
                n -= 1
                pbar.update(1)

    if exclude_first:
        placed.pop(0)

    return placed


def get_longest_direction(polygon: Polygon) -> XY:
    min_rec = polygon.minimum_rotated_rectangle
    coords = np.array(min_rec.exterior.coords)  # type: ignore
    edges = coords[1:] - coords[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    return edges[np.argmax(lengths)] / lengths[np.argmax(lengths)]
