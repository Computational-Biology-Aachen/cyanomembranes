import numpy as np
from numba import njit
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from shapely.ops import unary_union

import cyanomembranes as cm

from .config import Condition


def _mv_str(mv) -> str:
    """File-name part of the membrane variation: always with a decimal point.

    The value is read back from a CSV as a float, so 2 and 2.0 must give the same
    name ("2.0", as the floats written by the scenarios always did).
    """
    return str(float(mv))


def _n_originals(points, ghosts):
    """Number of original points; with `ghosts` the array holds nine periodic copies."""
    n = len(points)
    if not ghosts:
        return n
    if n % 9 != 0:
        msg = f"with ghosts the length of points must be a multiple of 9, got {n}"
        raise ValueError(msg)
    return n // 9


def _cg_str(cg) -> str:
    """File-name part of the crystal degree: 0 -> "0", 0.4 -> "0.4", 1.0 -> "1".

    An integer 0 in a scenario and the 0.0 read back from a CSV give the same name.
    """
    return f"{cg:g}"


def _timeseries_filename(condition: Condition, with_mv: bool = False) -> str:
    if condition.cg is not None:
        assert condition.crystal_prot is not None
        parts = [
            condition.pkey,
            str(condition.nprot),
            condition.crystal_prot,
            _mv_str(condition.mv),
            _cg_str(condition.cg),
        ]
    else:
        parts = [
            condition.pkey,
            str(condition.nprot),
        ]

        if with_mv:
            parts.append(_mv_str(condition.mv))

    return "_".join(parts) + ".csv"


def rdf(polygons, n_bins, box_size, r_max=None, ghosts=True):
    """Radial distribution function of a periodic point set.

    With `ghosts`, `polygons` holds nine periodic copies of the box and the first
    ninth must be the originals (`nndf` describes the parameters).
    """
    N = _n_originals(polygons, ghosts)

    rho = N / box_size**2

    if r_max is None:
        r_max = box_size / 2  # critical for correctness

    if r_max > box_size / 2:
        msg = "rmax should not be bigger than box_size / 2"
        raise ValueError(msg)

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]
    shell_area = 2 * np.pi * bin_centers * dr

    tree = cKDTree(polygons)

    counts_per_particle = []

    for i in range(N):
        neighbors = tree.query_ball_point(polygons[i], r_max)
        neighbors = [j for j in neighbors if j != i]
        delta = polygons[i] - polygons[neighbors]
        dist = np.linalg.norm(delta, axis=1)

        counts, _ = np.histogram(dist, bin_edges)
        counts_per_particle.append(counts)

    mean_count = np.mean(counts_per_particle, axis=0)
    std_count = np.std(counts_per_particle, axis=0, ddof=1)
    sem_count = std_count / np.sqrt(N)

    gr = mean_count / (rho * shell_area)
    gr_std = std_count / (rho * shell_area)
    gr_sem = sem_count / (rho * shell_area)
    return gr, gr_std, gr_sem, bin_centers


def nndf(polygons, n_bins, box_size, r_max=None, ghosts=True):
    """Nearest-neighbour distance distribution of a periodic point set.

    Parameters
    ----------
    polygons : np.ndarray
        Centroids, shape (n, 2). With `ghosts` the array holds nine periodic
        copies of the box and the first ninth must be the originals.
    n_bins : int
        Number of distance bins between 0 and `r_max`.
    box_size : float
        Edge length of the periodic box.
    r_max : float | None, optional
        Largest distance, at most `box_size / 2` (default `box_size / 2`).
    ghosts : bool, optional
        If True, only the first ninth of `polygons` are used as reference points.

    Returns
    -------
    tuple
        `G_empirical` (cumulative fraction of particles with a nearest neighbour
        within r), `G_std`, `g_nn_empirical` (density per bin), `g_nn_std`, and
        the bin centres.

    Notes
    -----
    `G_std` is the binomial standard error `sqrt(G (1 - G) / N)`. It assumes
    independent nearest-neighbour distances and therefore slightly underestimates
    the scatter (about 0.7 to 1.0 of it for an ideal gas), because two particles
    that are each other's nearest neighbour are counted twice. `g_nn_std` is the
    binomial error of a single bin.
    """

    N = _n_originals(polygons, ghosts)

    if r_max is None:
        r_max = box_size / 2  # critical for correctness

    if r_max > box_size / 2:
        msg = "rmax should not be bigger than box_size / 2"
        raise ValueError(msg)

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]

    tree = cKDTree(polygons)

    distances, _ = tree.query(polygons[:N], k=2)

    nn_distances = distances[:, 1]

    counts, _ = np.histogram(nn_distances, bins=bin_edges)

    p = counts / N

    g_nn_empirical = p / dr
    g_nn_std = np.sqrt(N * p * (1 - p)) / (N * dr)

    G_empirical = np.cumsum(counts) / N
    G_std = np.sqrt(G_empirical * (1 - G_empirical) / N)

    return G_empirical, G_std, g_nn_empirical, g_nn_std, bin_centers


def min_neighbor_distance(
    polygons, start_idx, target_idx, box_size, r_max=None, ghosts=True
):
    """Distance from each start point to its nearest target point.

    With `ghosts`, `polygons` holds nine periodic copies of the box (originals
    first) and `start_idx` the indices of the start points copy by copy; only the
    first ninth of `start_idx` is used. Points without a target within `r_max` give
    NaN.
    """
    if ghosts:
        start_idx = start_idx[: _n_originals(start_idx, ghosts)]

    tree = cKDTree(polygons)

    if r_max is None:
        r_max = box_size / 2

    targets = set(np.asarray(target_idx).tolist())

    min_distance = []
    for i in start_idx:
        neighbors = tree.query_ball_point(polygons[i], r_max)
        neighbors = [j for j in neighbors if j != i and j in targets]

        if len(neighbors) == 0:
            min_distance.append(np.nan)
            continue

        delta = polygons[i] - polygons[neighbors]
        dist = np.linalg.norm(delta, axis=1)
        min_distance.append(dist.min())

    return np.array(min_distance)


@njit
def bfs(grid, visited, start_r, start_c):
    R, C = grid.shape
    queue_r = np.empty(R * C, dtype=np.int32)
    queue_c = np.empty(R * C, dtype=np.int32)

    head = 0
    tail = 0

    queue_r[tail] = start_r
    queue_c[tail] = start_c
    tail += 1

    visited[start_r, start_c] = 1

    size = 1

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            if nr < 0:
                nr += R
            elif nr >= R:
                nr -= R
            nc = c + dc
            if nc < 0:
                nc += C
            elif nc >= C:
                nc -= C

            if visited[nr, nc] == 0 and grid[nr, nc] != 1:
                visited[nr, nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1
                size += 1
    return size


def flood_fill(grid):
    sizes = []
    visited = np.zeros_like(grid, dtype=np.uint8)
    R, C = grid.shape
    for r in range(R):
        for c in range(C):
            if visited[r, c] == 0 and grid[r, c] != 1:
                size = bfs(grid, visited, r, c)
                sizes.append(size)
    return sizes


def cluster_sizes(
    polygons,
    resolution=1,
    max_dim=5000,
    particle_radius=5,
):
    raster = cm.geo_utils.make_raster(
        unary_union(polygons),  # type: ignore
        resolution,
        -max_dim,
        -max_dim,
        2 * max_dim,
        2 * max_dim,
    )
    n = int(max_dim / resolution)
    smaller_raster = raster[n : 2 * n, n : 2 * n]
    particle = cm.brownian_lattice._make_lattice_particle(particle_radius)
    free = smaller_raster == 0
    pad = particle.shape[0] // 2
    free = np.pad(free, pad, mode="wrap")
    accessible = binary_erosion(free, structure=particle)
    accessible = accessible[pad : pad + n, pad : pad + n]
    sizes = flood_fill(~accessible)

    return (
        sizes,
        smaller_raster,
        ~accessible,
    )
