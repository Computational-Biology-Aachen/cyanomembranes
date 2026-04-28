"""methods to simulate brownian motion"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba
import numba as nb
import numpy as np
import pandas as pd
import shapely.vectorized as sv
from joblib import Parallel, delayed, dump, load
from shapely import Point, Polygon, unary_union
from shapely.affinity import translate
from shapely.geometry import box
from shapely.prepared import prep
from shapely.strtree import STRtree
from sklearn.linear_model import LinearRegression

from cyanomembranes.geo_utils import _get_lattice_directions, make_raster, readwkt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from shapely.geometry.base import BaseGeometry


def _make_lattice_particle(radius_cells: int) -> np.ndarray:
    x, y = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    mask = x**2 + y**2 <= radius_cells**2
    return mask.astype(bool)


@nb.njit
def check_local_free(raster, positions, kernel):
    ky, kx = kernel.shape
    ry, rx = ky // 2, kx // 2
    mask_free = np.empty(positions.shape[0], dtype=nb.boolean)

    for i in range(positions.shape[0]):
        y, x = positions[i]
        y0, y1 = y - ry, y + ry + 1
        x0, x1 = x - rx, x + rx + 1

        if y0 < 0 or x0 < 0 or y1 > raster.shape[0] or x1 > raster.shape[1]:
            mask_free[i] = False
            continue

        overlap = False
        for j in range(ky):
            for k in range(kx):
                if kernel[j, k] and raster[y0 + j, x0 + k] != 0:
                    overlap = True
                    break
            if overlap:
                break
        mask_free[i] = not overlap

    return mask_free


def make_aim_mask(
    raster: np.ndarray, aim_area: Polygon, a: float, has_ghost: bool = True
) -> np.ndarray:
    if has_ghost:
        _, max_dim = raster.shape
        max_dim //= 3
        aim_area = translate(aim_area, xoff=max_dim, yoff=max_dim)

    mask = np.zeros(raster.shape, dtype=bool)

    minx, miny, maxx, maxy = map(
        int,
        (
            aim_area.bounds[0] / a,
            aim_area.bounds[1] / a,
            aim_area.bounds[2] / a,
            aim_area.bounds[3] / a,
        ),
    )

    ys, xs = np.meshgrid(
        np.arange(miny, maxy) * a, np.arange(minx, maxx) * a, indexing="ij"
    )
    mask[miny:maxy, minx:maxx] = sv.contains(aim_area, xs, ys)

    return np.flipud(mask)


def _sample_point_in_free_space(
    free_space: BaseGeometry,
    *,
    max_tries: int = 1000,
    constraint: None | Polygon = None,
    rng: np.random.Generator | None = None,
    particle_radius: None | float = None,
) -> tuple[float, float]:
    """
    Sample a random point within a free space geometry, optionally constrained by
    another polygon.

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
    particle_radius: float or None, optional
        Radius of the random point.

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
        if particle_radius:
            point = point.buffer(particle_radius)
        if prepared.contains(point):
            return x, y
    msg = f"Failed to find a free point after {max_tries} tries"
    raise RuntimeError(msg)


def _check_point_in_shapes(
    x0: float,
    y0: float,
    tree: STRtree,
    shapes: list[Polygon],
    particle_radius: None | float = None,
) -> bool:
    """
    Check if a buffered point is contained in any of the provided shapes.

    Parameters
    ----------
    x0, y0 : float
        Coordinates of the point.
    tree : STRtree or None
        Spatial index for efficient querying (optional).
    shapes : list[Polygon]
        List of polygons to check against.
    particle_radius: float pr None
        Radius of buffered point.

    Returns
    -------
    bool
        True if the point is inside any shape, False otherwise.
    """
    point = Point(x0, y0)
    if particle_radius:
        point = point.buffer(particle_radius)
    hits = tree.query(point, predicate="intersects")
    return any(not shapes[i].touches(point) for i in hits)


def prepare_geometry(shapes: list[Polygon] | None) -> list[Polygon]:
    """
    Convert a list of polygons to a list of polygons.

    Parameters
    ----------
    shapes : list[Polygon], or None
        Input geometry.

    Returns
    -------
    list[Polygon]
        List of polygons.
    """
    if shapes is None:
        polygons = []
    else:
        polygons = shapes
    return polygons


@dataclass
class DiffusionTrajectories:
    """
    Stores and analyzes a single diffusion trajectory.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing trajectory data (e.g., X, Y, Time, MSD).
    """

    data: pd.DataFrame

    def mean_over_time(self, column: str) -> float:
        """
        Compute the mean of a column over all time points.

        Parameters
        ----------
        column : str
            Name of the column to average.

        Returns
        -------
        float
            The mean value of the specified column.
        """
        return self.data[column].mean()


@dataclass
class ExperimentRun:
    """
    Stores the results of multiple diffusion trajectories (replicates).

    Attributes
    ----------
    trajectories : list[DiffusionTrajectories]
        List of trajectory objects for each replicate.
    """

    trajectories: list[DiffusionTrajectories]
    shapes: list[Polygon]
    raster: np.ndarray = None
    has_ghost: bool = None  # only relevant for lattice

    def mean_trajectories(self, group_by: str = "Time") -> pd.DataFrame:
        """
        Compute the mean trajectory over all replicates, grouped by a column.

        Parameters
        ----------
        group_by : str, optional
            Column to group by (default is "Time").

        Returns
        -------
        pd.DataFrame
            DataFrame with mean values for each group.
        """
        df_all = pd.concat(
            [t.data.assign(traj_id=i) for i, t in enumerate(self.trajectories)],
            ignore_index=True,
        )
        return df_all.groupby(group_by, as_index=False).mean(numeric_only=True)

    def get_mean_diff_coefficients(
        self,
        *,
        intercept: bool = False,
    ) -> tuple[LinearRegression, float]:
        """
        Fit a linear regression to the mean squared displacement (MSD) vs. time
        and estimate the diffusion coefficient from the slope.

        Parameters
        ----------
        intercept : bool, optional
            Whether to fit the intercept (default: False).

        Returns
        -------
        tuple[LinearRegression, float]
            The fitted LinearRegression object and the estimated diffusion coefficient.
        """
        mean_traj_df = self.mean_trajectories()
        X = mean_traj_df["Time"].to_numpy().reshape(-1, 1)
        Y = mean_traj_df["MSD"].to_numpy()
        lr = LinearRegression(fit_intercept=intercept)
        lr.fit(X, Y)
        return lr, lr.coef_[0] / 4

    def get_mean_diff_coefficients_pw(
        self,
        *,
        intercept: bool = True,
    ) -> tuple[LinearRegression, float, float]:
        """
        Fit a power-law (log-log) regression to MSD vs. time and estimate the
        scaling exponent and prefactor.

        Parameters
        ----------
        intercept : bool, optional
            Whether to fit the intercept (default: True).

        Returns
        -------
        tuple[LinearRegression, float, float]
            The fitted LinearRegression object, the prefactor (exp(intercept)),
            and the scaling exponent (slope).
        """
        mean_traj_df = self.mean_trajectories()
        X = np.log(mean_traj_df["Time"].to_numpy().reshape(-1, 1)[1:])
        Y = np.log(mean_traj_df["MSD"].to_numpy()[1:])
        pw = LinearRegression(fit_intercept=intercept)
        pw.fit(X, Y)
        return pw, math.exp(pw.intercept_), pw.coef_[0]

    def get_mean_diff_coefficients_dist(
        self,
    ) -> pd.Series:
        """
        Compute the instantaneous diffusion coefficient as MSD divided by time.

        Returns
        -------
        pd.Series
            Series of instantaneous diffusion coefficients at each time point.
        """
        mean_traj_df = self.mean_trajectories()
        return mean_traj_df["MSD"] / mean_traj_df["Time"]

    def plot_run(self, *, ax: None | Axes = None, aim_area: bool = False) -> Axes:
        if self.raster is not None:
            h, w = self.raster.shape[:2]

            if self.has_ghost:
                cw = w // 3
                ch = h // 3
                central_raster = self.raster[ch : 2 * ch, cw : 2 * cw]
                ax.imshow(central_raster, origin="lower", cmap="gray_r")

                if aim_area:  # TODO make this more efficient
                    for traj in self.trajectories:
                        traj_aim_df = traj.data[traj.data["Aim"] == 1]
                        traj_no_aim_df = traj.data[traj.data["Aim"] == 0]
                        ax.scatter(
                            traj_aim_df["X"] - cw,
                            traj_aim_df["Y"] - ch,
                            s=0.1,
                            c="lightblue",
                        )
                        ax.scatter(
                            traj_no_aim_df["X"] - cw,
                            traj_no_aim_df["Y"] - ch,
                            s=0.1,
                            c="green",
                        )
                else:
                    for traj in self.trajectories:
                        ax.plot(traj.data["X"] - cw, traj.data["Y"] - ch, lw=0.8)

            else:
                if aim_area:  # TODO make this more efficient
                    for traj in self.trajectories:
                        traj_aim_df = traj.data[traj.data["Aim"] == 1]
                        traj_no_aim_df = traj.data[traj.data["Aim"] == 0]
                        ax.scatter(
                            traj_aim_df["X"], traj_aim_df["Y"], s=0.1, c="lightblue"
                        )
                        ax.scatter(
                            traj_no_aim_df["X"], traj_no_aim_df["Y"], s=0.1, c="green"
                        )
                else:
                    for traj in self.trajectories:
                        ax.plot(traj.data["X"], traj.data["Y"], lw=0.8)

            return ax
        else:
            if self.shapes:
                for p in self.shapes:
                    ax.plot(*p.exterior.xy, lw=1, c="blue")
            if aim_area:  # TODO make this more efficient
                for traj in self.trajectories:
                    traj_aim_df = traj.data[traj.data["Aim"] == 1]
                    traj_no_aim_df = traj.data[traj.data["Aim"] == 0]
                    ax.scatter(traj_aim_df["X"], traj_aim_df["Y"], s=1, c="lightblue")
                    ax.scatter(traj_no_aim_df["X"], traj_no_aim_df["Y"], s=1, c="green")
            else:
                for traj in self.trajectories:
                    ax.plot(traj.data["X"], traj.data["Y"], lw=0.8)
            return ax


@dataclass
class EnsembleExperimentRun:
    """
    Stores the results of an ensemble of experiment runs.

    Attributes
    ----------
    runs : list[ExperimentRun]
        List of experiment runs (one per geometry or file).
    name : str | None
        Optional name for the ensemble.
    """

    runs: list[ExperimentRun]
    coverages: list[float]
    name: str | None = None

    def mean_over_runs(self, group_by: str = "Time") -> pd.DataFrame:
        """
        Compute the mean over all experiment runs, grouped by a column.

        Parameters
        ----------
        group_by : str, optional
            Column to group by (default is "Time").

        Returns
        -------
        pd.DataFrame
            DataFrame with mean values for each group across all runs.
        """
        run_all = pd.concat(
            [r.mean_trajectories().assign(run_id=i) for i, r in enumerate(self.runs)]
        )
        return run_all.groupby(group_by, as_index=False).mean(numeric_only=True)

    def get_mean_diff_coefficients(
        self,
        *,
        intercept: bool = False,
    ) -> tuple[LinearRegression, float]:
        """
        Fit a linear regression to the mean squared displacement (MSD) vs. time
        across all runs and estimate the diffusion coefficient from the slope.

        Parameters
        ----------
        intercept : bool, optional
            Whether to fit the intercept (default: False).

        Returns
        -------
        tuple[LinearRegression, float]
            The fitted LinearRegression object and the estimated diffusion coefficient.
        """
        mean_traj_df = self.mean_over_runs()
        X = mean_traj_df["Time"].to_numpy().reshape(-1, 1)
        Y = mean_traj_df["MSD"].to_numpy()
        lr = LinearRegression(fit_intercept=intercept)
        lr.fit(X, Y)
        return lr, lr.coef_[0] / 4

    def get_mean_diff_coefficients_pw(
        self,
        *,
        intercept: bool = True,
    ) -> tuple[LinearRegression, float, float]:
        """
        Fit a power-law (log-log) regression to MSD vs. time across all runs and
        estimate the scaling exponent and prefactor.

        Parameters
        ----------
        intercept : bool, optional
            Whether to fit the intercept (default: True).

        Returns
        -------
        tuple[LinearRegression, float, float]
            The fitted LinearRegression object, the prefactor (exp(intercept)),
            and the scaling exponent (slope).
        """
        mean_traj_df = self.mean_over_runs()
        X = np.log(mean_traj_df["Time"].to_numpy().reshape(-1, 1)[1:])
        Y = np.log(mean_traj_df["MSD"].to_numpy()[1:])
        pw = LinearRegression(fit_intercept=intercept)
        pw.fit(X, Y)
        return pw, math.exp(pw.intercept_), pw.coef_[0]

    def get_mean_diff_coefficients_dist(
        self,
    ) -> pd.DataFrame:
        """
        Compute the instantaneous diffusion coefficient as MSD divided by time,
        across all runs.

        Returns
        -------
        pd.Series
            Series of instantaneous diffusion coefficients at each time point.
        """
        mean_traj_df = self.mean_over_runs()
        return pd.concat(
            [
                mean_traj_df["Time"],
                mean_traj_df["MSD"],
                mean_traj_df["MSD"] / mean_traj_df["Time"],
            ],
            axis=1,
        )

    def get_mean_coverage(self) -> float:
        return float(np.mean(self.coverages))


def random_walk_2d(
    shapes: list[Polygon] | None,
    *,
    nsteps: int = 1000,
    dt: float = 0.001,
    start: tuple[float, float] = (0.0, 0.0),
    diff_coefficient: float = 1.0,
    particle_radius: None | float = None,
    aim_area: None | Polygon = None,
    rng: np.random.Generator | None = None,
) -> DiffusionTrajectories:
    """
    Simulate a 2D random walk (Brownian motion) with geometric constraints.

    Parameters
    ----------
    shapes : list[Polygon] | None
        Obstacles or geometry for the simulation.
    nsteps : int, optional
        Number of steps (default: 1000).
    dt : float, optional
        Time step size (default: 0.001).
    start : tuple[float, float], optional
        Starting position (default: (0.0, 0.0)).
    diff_coefficient : float, optional
        Diffusion coefficient (default: 1.0).
    rng : np.random.Generator or None, optional
        Random number generator (default: None, uses default_rng).

    Returns
    -------
    DiffusionTrajectories
        Object containing the simulated trajectory data.

    Raises
    ------
    ValueError
        If the starting point is inside an obstacle.
    """
    # Setup
    rng = np.random.default_rng() if rng is None else rng
    shapes = prepare_geometry(shapes)
    tree = STRtree(shapes)
    step_size = np.sqrt(2 * diff_coefficient * dt)

    # Geometry
    if _check_point_in_shapes(start[0], start[1], tree, shapes, particle_radius):
        msg = "Particle starts in shape. Please choose different starting conditions"
        raise ValueError(msg)

    # Preallocation
    x = np.empty(nsteps)
    y = np.empty(nsteps)
    t = np.empty(nsteps)
    ai = np.empty(nsteps)  # indicates if particle is in aim area

    x[0], y[0] = start
    t[0] = 0

    if aim_area:
        if particle_radius:
            test_point = Point(*start).buffer(particle_radius)
        else:
            test_point = Point(*start)
        if aim_area.intersects(test_point):
            ai[0] = 1
        else:
            ai[0] = 0

    for i in range(1, nsteps):  # +1?
        overlap = True  # Only advance when no overlap. Physically realistic?
        while overlap:
            dx = rng.standard_normal() * step_size
            dy = rng.standard_normal() * step_size
            new_x = x[i - 1] + dx
            new_y = y[i - 1] + dy
            overlap = _check_point_in_shapes(
                new_x, new_y, tree, shapes, particle_radius
            )

        x[i] = new_x  # type: ignore
        y[i] = new_y  # type: ignore
        t[i] = t[i - 1] + dt

        if aim_area:
            if particle_radius:
                test_point = Point(new_x, new_y).buffer(particle_radius)
            else:
                test_point = Point(new_x, new_y)
            if aim_area.intersects(test_point):
                ai[i] = 1
            else:
                ai[i] = 0

    dx = x - x[0]
    dy = y - y[0]
    msd = dx**2 + dy**2

    data = {"X": x, "Y": y, "Time": t, "MSD": msd, "Aim": ai}  # type: ignore

    return DiffusionTrajectories(data=pd.DataFrame(data))


def random_walk_2d_lattice(
    raster: np.ndarray,
    nsteps: int = 10_000,
    nreps: int = 50,
    diff_coefficient: float = 1.0,
    a: float = 1.0,
    start: tuple[int, int] | None = None,
    aim_mask: np.ndarray | None = None,
    particle_radius: None | int = None,
    rng: np.random.Generator | None = None,
):
    rng = np.random.default_rng() if rng is None else rng
    dt = a**2 / (4 * diff_coefficient)

    directions = _get_lattice_directions("square")

    pos_hist = np.empty((nsteps + 1, nreps, 2), dtype=int)
    ai_hist = np.empty((nsteps + 1, nreps), dtype=int)

    pos = start
    pos_hist[0] = pos

    if aim_mask is not None:
        ai_hist[0] = aim_mask[pos[:, 0], pos[:, 1]]

    kernel = None
    if particle_radius:
        kernel = _make_lattice_particle(particle_radius)

    for t in range(1, nsteps + 1):
        move_idx = rng.integers(0, 4, size=nreps)
        step = np.array(directions)[move_idx]

        new_pos = pos + step
        if kernel is not None:
            mask_free = check_local_free(raster, new_pos, kernel)
        else:
            mask_free = raster[new_pos[:, 0], new_pos[:, 1]] == 0

        pos = np.where(mask_free[:, None], new_pos, pos)

        pos_hist[t] = pos

        if aim_mask is not None:
            ai_hist[t] = aim_mask[pos[:, 0], pos[:, 1]]

    disp = (pos_hist - pos_hist[0]) * a
    msd = np.sum(disp**2, axis=2)

    replicate_ids = np.tile(np.arange(nreps), nsteps + 1)

    df = pd.DataFrame(
        {
            "Y": pos_hist[:, :, 0].ravel(),
            "X": pos_hist[:, :, 1].ravel(),
            "Y_real": disp[:, :, 0].ravel(),
            "X_real": disp[:, :, 1].ravel(),
            "Time": np.repeat(np.arange(nsteps + 1) * dt, nreps),
            "MSD": msd.ravel(),
            "Aim": ai_hist.ravel(),
            "Replicate": replicate_ids,
        }
    )

    return df


@dataclass(slots=True)  # TODO prevent users from using the class itself
class ExperimentConfig:
    """
    Configuration for a random walk or diffusion experiment.

    Attributes
    ----------
    nsteps : int
        Number of steps per trajectory.
    dt : float
        Time step size.
    shapes : list[Polygon] | None
        Obstacles or geometry for the simulation.
    replicates : int
        Number of independent trajectories to simulate.
    diff_coefficient : float
        Diffusion coefficient.
    start : tuple[float, float]
        Starting position.
    dimensions : tuple[float, float]
        Simulation box dimensions.
    random_start : bool
        Whether to randomize the starting position.
    """

    nsteps: int = 1000
    dt: float = 0.001
    shapes: list[Polygon] | None = None
    replicates: int = 100
    diff_coefficient: float = 1.0
    start: tuple[float, float] = (0.0, 0.0)
    dimensions: tuple[float, float] = (0, 2000)
    coverage_dimension: tuple[float, float] | None = (
        None  # just if you want to calculate coverage in other areas
    )
    random_start: bool = False
    aim_area: None | Polygon = None
    particle_radius: None | float = None
    lattice_type: None | str = None
    has_ghost: None | bool = True  # only relevant for lattice
    workers: None | int = 4


@dataclass
class Experiment:
    """
    Run random walker experiments in parallel.

    Attributes
    ----------
    config : ExperimentConfig
        Configuration for the experiment.
    free_space : Polygon
        Computed free space for random starting positions.
    """

    config: ExperimentConfig

    def __post_init__(self) -> None:
        """
        Compute the free space for the experiment based on the configuration.
        """
        min_dim, max_dim = self.config.dimensions
        union_shapes = unary_union(self.config.shapes)
        workspace = box(min_dim, min_dim, max_dim, max_dim)
        self.free_space = workspace.difference(union_shapes)
        if self.config.coverage_dimension:
            min_dim, max_dim = self.config.coverage_dimension
            workspace = box(min_dim, min_dim, max_dim, max_dim)
        self.coverage = workspace.intersection(union_shapes).area / (max_dim * max_dim)

        if self.config.lattice_type:
            if self.config.has_ghost:
                raster = make_raster(
                    union_shapes, 1.0, -max_dim, -max_dim, 2 * max_dim, 2 * max_dim
                )
            else:
                raster = make_raster(union_shapes, 1.0, 0, 0, max_dim, max_dim)
            self.raster = np.flipud(raster)

            if self.config.aim_area is not None:
                self.aim_mask = make_aim_mask(
                    raster,
                    self.config.aim_area,
                    a=1.0,
                    has_ghost=self.config.has_ghost,
                )

    def worker(self, _: int) -> DiffusionTrajectories:
        """
        Worker function for parallel execution of a single trajectory.

        Parameters
        ----------
        _ : int
            Index of the replicate (unused).

        Returns
        -------
        DiffusionTrajectories
            The simulated trajectory.
        """

        if self.config.lattice_type:
            if self.config.random_start:
                _, max_dim = self.config.dimensions
                max_dim = int(max_dim)

                if self.config.has_ghost:
                    sub_raster = self.raster[
                        max_dim : 2 * max_dim, max_dim : 2 * max_dim
                    ]
                else:
                    sub_raster = self.raster

                free_cells = np.argwhere(sub_raster == 0)
                rng = np.random.default_rng()
                start_local = np.empty((self.config.replicates, 2), dtype=int)

                kernel = None
                if self.config.particle_radius:
                    kernel = _make_lattice_particle(self.config.particle_radius)

                # Initially sample all candidates
                idx = rng.integers(len(free_cells), size=self.config.replicates)
                start_local[:] = free_cells[idx]

                if kernel is not None:
                    # Check which ones are valid
                    mask_free = check_local_free(sub_raster, start_local, kernel)

                    # Resample invalid ones until all are valid
                    while not mask_free.all():
                        n_invalid = (~mask_free).sum()
                        idx = rng.integers(len(free_cells), size=n_invalid)
                        start_local[~mask_free] = free_cells[idx]
                        mask_free = check_local_free(sub_raster, start_local, kernel)

                start = start_local + max_dim

            else:
                start = np.full((self.config.replicates, 2), self.config.start)

            runs = random_walk_2d_lattice(
                raster=self.raster,
                nsteps=self.config.nsteps,
                nreps=self.config.replicates,
                diff_coefficient=self.config.diff_coefficient,
                a=1.0,
                start=start,
                aim_mask=self.aim_mask,
                particle_radius=self.config.particle_radius,
            )

            return [
                DiffusionTrajectories(g.reset_index(drop=True))
                for _, g in runs.groupby("Replicate")
            ]

        else:
            if self.config.random_start:
                start = _sample_point_in_free_space(
                    self.free_space, particle_radius=self.config.particle_radius
                )
            else:
                start = self.config.start

            return random_walk_2d(
                shapes=self.config.shapes,
                nsteps=self.config.nsteps,
                dt=self.config.dt,
                start=start,
                diff_coefficient=self.config.diff_coefficient,
                aim_area=self.config.aim_area,
                particle_radius=self.config.particle_radius,
            )

    def run(self) -> ExperimentRun:
        """
        Run all replicates of the experiment in parallel.

        Returns
        -------
        ExperimentRun
            Object containing all simulated trajectories.
        """

        if self.config.lattice_type:
            runs = self.worker(self.config.replicates)
            return ExperimentRun(
                runs, self.config.shapes, self.raster, self.config.has_ghost
            )
        else:
            with Pool(processes=self.config.workers) as pool:
                runs = pool.map(self.worker, range(self.config.replicates))
            return ExperimentRun(runs, self.config.shapes)


@dataclass
class EnsembleExperiment:
    """
    Run an ensemble of experiments, each with a different geometry or file.

    Attributes
    ----------
    file_paths : list[str]
        List of file paths to geometry files.
    config : ExperimentConfig
        Configuration for the experiments.
    """

    file_paths: list[str]
    config: ExperimentConfig

    def run(self) -> EnsembleExperimentRun:
        """
        Run random walk experiments for each file in the ensemble.

        For each file path, reads the geometry, updates the experiment configuration,
        runs the experiment, and collects the results into an ensemble.

        Returns
        -------
        EnsembleExperimentRun
            An object containing the results of all experiments in the ensemble.
        """
        ensemble_lst = []
        coverage_lst = []
        for f in self.file_paths:
            polygons = readwkt(f)
            self.config.shapes = polygons
            exp = Experiment(self.config)
            ensemble_lst.append(exp.run())
            coverage_lst.append(exp.coverage)
        return EnsembleExperimentRun(ensemble_lst, coverage_lst)
