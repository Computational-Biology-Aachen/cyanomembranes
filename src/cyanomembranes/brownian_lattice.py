"""methods to simulate brownian motion"""

from __future__ import annotations

import dataclasses
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.vectorized as sv
import shapely.wkt
from scipy.ndimage import binary_erosion
from shapely import Polygon
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from tqdm.notebook import trange

from cyanomembranes.geo_utils import _get_lattice_directions, make_raster, readwkt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from shapely.geometry.base import BaseGeometry


def get_buffered_shapes(
    comparison_size: float, shapes: list, buffer: float, constraint_area: Polygon
) -> list[Polygon]:
    selection_lst = []
    b = constraint_area
    for s in shapes:
        if np.isclose(s.area, comparison_size) and s.intersects(b):
            difference = s.buffer(buffer).difference(s)
            selection_lst.append(difference)
    return selection_lst


def _make_lattice_particle(radius_cells: int) -> np.ndarray:
    x, y = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
    mask = x**2 + y**2 <= radius_cells**2
    return mask.astype(bool)


def check_local_free2(
    free_raster: np.ndarray | None, positions: np.ndarray
) -> np.ndarray:
    """Check whether lattice positions are free according to a raster.

    Parameters
    ----------
    free_raster : np.ndarray
        2D raster where cells with value 1 indicate free/allowed centers.
    positions : np.ndarray
        Array of integer indices with shape (n, 2) where each row is (row, col).

    Returns
    -------
    np.ndarray
        Boolean array of length n: True for positions that are free.
    """
    if free_raster is None:
        msg = "Free raster is not allowed to be None"
        raise ValueError(msg)
    return free_raster[positions[:, 0], positions[:, 1]] == 1


def make_aim_mask(
    raster: np.ndarray,
    aim_area: Polygon,
    lattice_resolution: float,
) -> np.ndarray:
    """Create a boolean mask for the given aim area aligned to the raster.

    The returned mask has the same shape as `raster`. Cells inside the
    `aim_area` polygon are True. `lattice_resolution` is used to convert
    between polygon coordinates and raster indices.

    Parameters
    ----------
    raster : np.ndarray
        Reference raster whose shape is used for the output mask.
    aim_area : Polygon
        Shapely polygon describing the region of interest (aim area).
    lattice_resolution : float
        Size of one raster cell in the same coordinate units as `aim_area`.

    Returns
    -------
    np.ndarray
        Boolean mask of the same shape as `raster` with True inside `aim_area`.
    """
    mask = np.zeros(raster.shape, dtype=bool)

    minx, miny, maxx, maxy = map(
        int,
        (
            aim_area.bounds[0] / lattice_resolution,
            aim_area.bounds[1] / lattice_resolution,
            aim_area.bounds[2] / lattice_resolution,
            aim_area.bounds[3] / lattice_resolution,
        ),
    )

    ys, xs = np.meshgrid(
        np.arange(miny, maxy) * lattice_resolution,
        np.arange(minx, maxx) * lattice_resolution,
        indexing="ij",
    )
    mask[miny:maxy, minx:maxx] = sv.contains(aim_area, xs, ys)

    return mask


@dataclass
class DiffusionLatticeTrajectories:
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
class ExperimentLatticeRun:
    """
    Stores the results of multiple diffusion trajectories (replicates).

    Attributes
    ----------
    trajectories : list[DiffusionTrajectories]
        List of trajectory objects for each replicate.
    """

    trajectories: list[DiffusionLatticeTrajectories]
    shapes: Sequence[BaseGeometry] | None
    raster: np.ndarray
    has_ghost: bool
    fpt: pd.DataFrame
    hits: pd.DataFrame | None = None
    waiting_times: np.ndarray | None = None

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
        df_all = [t.data.copy() for t in self.trajectories]
        groups = df_all[0][group_by].to_numpy()
        n_traj = len(df_all)
        metrics = ["MSD", "Active"]
        if "Aim" in df_all[0].columns:
            metrics.append("Aim")

        data_arr = {
            m: np.array([df[m].to_numpy() for df in df_all], dtype=np.float32)
            for m in metrics
        }

        df_stats = pd.DataFrame({group_by: groups})

        for m in metrics:
            arr = data_arr[m]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1)
            sem = std / np.sqrt(n_traj)
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem

            df_stats[f"{m}"] = mean
            df_stats[f"{m}_std"] = std
            df_stats[f"{m}_sem"] = sem
            df_stats[f"{m}_ci_low"] = ci_low
            df_stats[f"{m}_ci_high"] = ci_high

        return df_stats

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
        x_var = mean_traj_df["Time"].to_numpy().reshape(-1, 1)
        y_var = mean_traj_df["MSD"].to_numpy()
        lr = LinearRegression(fit_intercept=intercept)
        lr.fit(x_var, y_var)
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
        x_var = np.log(mean_traj_df["Time"].to_numpy().reshape(-1, 1)[1:])
        y_var = np.log(mean_traj_df["MSD"].to_numpy()[1:])
        pw = LinearRegression(fit_intercept=intercept)
        pw.fit(x_var, y_var)
        return pw, math.exp(pw.intercept_), pw.coef_[0]

    def get_mean_diff_coefficients_dist(
        self,
    ) -> pd.DataFrame:
        """
        Compute the instantaneous diffusion coefficient as MSD divided by time.

        Returns
        -------
        pd.Series
            Series of instantaneous diffusion coefficients at each time point.
        """
        mean_traj_df = self.mean_trajectories()
        return pd.DataFrame(
            {
                "Time": mean_traj_df["Time"],
                "D_dist": mean_traj_df["MSD"] / mean_traj_df["Time"],
                "D_dist_std": mean_traj_df["MSD_std"] / mean_traj_df["Time"],
                "D_dist_sem": mean_traj_df["MSD_sem"] / mean_traj_df["Time"],
                "D_dist_ci_low": mean_traj_df["MSD_ci_low"] / mean_traj_df["Time"],
                "D_dist_ci_high": mean_traj_df["MSD_ci_high"] / mean_traj_df["Time"],
            }
        )

    def get_mean_fpt(
        self,
    ) -> pd.Series:
        """Compute summary statistics for first-passage (inactive) times.

        Calculates mean, standard deviation, standard error of the mean,
        95% confidence interval, median and interquartile range of the
        non-NaN values in `FirstInactiveTime` from this run's FPT table.

        Returns
        -------
        pd.Series
            Series with keys: 'Mean', 'Std', 'Sem', 'Ci_low', 'Ci_high',
            'Median' and 'IQR'.
        """
        n_fpt = np.sqrt(len(self.fpt))
        fi_val = self.fpt.dropna()["FirstInactiveTime"].to_numpy()

        mean = np.mean(fi_val)
        median = np.median(fi_val)
        q25 = np.percentile(fi_val, 25)
        q75 = np.percentile(fi_val, 75)
        std = np.std(fi_val, ddof=1)
        sem = std / np.sqrt(n_fpt)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
        return pd.Series(
            {
                "Mean": mean,
                "Std": std,
                "Sem": sem,
                "Ci_low": ci_low,
                "Ci_high": ci_high,
                "Median": median,
                "IQR": q75 - q25,
            }
        )

    def plot_run(
        self,
        *,
        ax: None | Axes = None,
        aim_area: bool = False,
        scatter: bool = False,
        steps: int = None,
    ) -> Axes:
        """Plot trajectories from this experiment run onto the raster.

        Parameters
        ----------
        ax : None | Axes, optional
            Matplotlib axes to draw on. If None a new Axes is created.
        aim_area : bool, optional
            If True, mark points where trajectories are inside the aim area
            using a different color; otherwise draw full trajectories.

        Returns
        -------
        Axes
            The Matplotlib Axes with the plotted trajectories.
        """
        if ax is None:
            _, ax = plt.subplots()
        h, w = self.raster.shape[:2]
        if self.has_ghost:
            cw = w // 3
            ch = h // 3
            central_raster = self.raster[ch : 2 * ch, cw : 2 * cw]
            ax.imshow(central_raster, origin="lower", cmap="gray_r")
            if aim_area:
                for traj in self.trajectories:
                    traj_aim_df = traj.data[traj.data["Aim"] == 1]
                    traj_no_aim_df = traj.data[traj.data["Aim"] == 0]
                    ax.scatter(
                        traj_aim_df["X"][slice(steps)] - cw,
                        traj_aim_df["Y"][slice(steps)] - ch,
                        s=0.001,
                        marker=".",
                        c="lightblue",
                    )
                    ax.scatter(
                        traj_no_aim_df["X"][slice(steps)] - cw,
                        traj_no_aim_df["Y"][slice(steps)] - ch,
                        s=0.001,
                        marker=".",
                        c="green",
                    )
            else:
                for traj in self.trajectories:
                    if scatter:
                        ax.scatter(
                            traj.data["X"][slice(steps)] - cw,
                            traj.data["Y"][slice(steps)] - ch,
                            marker=".",
                            s=0.1,
                        )
                    else:
                        ax.plot(
                            traj.data["X"][slice(steps)] - cw,
                            traj.data["Y"][slice(steps)] - ch,
                            lw=0.3,
                        )
        elif aim_area:
            for traj in self.trajectories:
                traj_aim_df = traj.data[traj.data["Aim"] == 1]
                traj_no_aim_df = traj.data[traj.data["Aim"] == 0]
                ax.scatter(
                    traj_aim_df["X"][slice(steps)],
                    traj_aim_df["Y"][slice(steps)],
                    marker=".",
                    s=0.1,
                    c="lightblue",
                )
                ax.scatter(
                    traj_no_aim_df["X"][slice(steps)],
                    traj_no_aim_df["Y"][slice(steps)],
                    marker=".",
                    s=0.1,
                    c="green",
                )
        else:
            for traj in self.trajectories:
                if scatter:
                    ax.scatter(
                        traj.data["X"][slice(steps)],
                        traj.data["Y"][slice(steps)],
                        marker=".",
                        s=0.01,
                    )
                else:
                    ax.plot(
                        traj.data["X"][slice(steps)],
                        traj.data["Y"][slice(steps)],
                        lw=0.3,
                    )
        return ax


@dataclass
class EnsembleExperimentLatticeRun:
    """
    Stores the results of an ensemble of experiment runs.

    Attributes
    ----------
    runs : list[ExperimentRun]
        List of experiment runs (one per geometry or file).
    name : str | None
        Optional name for the ensemble.
    """

    runs: list[ExperimentLatticeRun]
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
        run_all = [r.mean_trajectories().copy() for r in self.runs]
        value_cols = [c for c in run_all[0] if c not in [group_by, "run_id"]]
        value_cols = [
            c for c in value_cols if not any(sub in c for sub in ["ci", "std", "sem"])
        ]
        groups = run_all[0][group_by].to_numpy()
        n_run = len(run_all)

        data_arr = {
            m: np.array([df[m].to_numpy() for df in run_all], dtype=np.float32)
            for m in value_cols
        }

        df_stats = pd.DataFrame({group_by: groups})

        for m in value_cols:
            arr = data_arr[m]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1)
            sem = std / np.sqrt(n_run)
            ci_low = mean - 1.96 * sem
            ci_high = mean + 1.96 * sem

            df_stats[f"{m}"] = mean
            df_stats[f"{m}_std"] = std
            df_stats[f"{m}_sem"] = sem
            df_stats[f"{m}_ci_low"] = ci_low
            df_stats[f"{m}_ci_high"] = ci_high

        return df_stats

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
        x_var = mean_traj_df["Time"].to_numpy().reshape(-1, 1)
        y_var = mean_traj_df["MSD"].to_numpy()
        lr = LinearRegression(fit_intercept=intercept)
        lr.fit(x_var, y_var)
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
        x_var = np.log(mean_traj_df["Time"].to_numpy().reshape(-1, 1)[1:])
        y_var = np.log(mean_traj_df["MSD"].to_numpy()[1:])
        pw = LinearRegression(fit_intercept=intercept)
        pw.fit(x_var, y_var)
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
        mean_run_df = self.mean_over_runs()
        return pd.DataFrame(
            {
                "Time": mean_run_df["Time"],
                "MSD": mean_run_df["MSD"],
                "D_dist": mean_run_df["MSD"] / mean_run_df["Time"],
                "D_dist_std": mean_run_df["MSD_std"] / mean_run_df["Time"],
                "D_dist_sem": mean_run_df["MSD_sem"] / mean_run_df["Time"],
                "D_dist_ci_low": mean_run_df["MSD_ci_low"] / mean_run_df["Time"],
                "D_dist_ci_high": mean_run_df["MSD_ci_high"] / mean_run_df["Time"],
            }
        )

    def get_mean_fpt(self) -> pd.Series:
        """Compute summary statistics of first-passage times across runs.

        Aggregates each run's mean first-passage time and computes the
        ensemble mean, standard deviation, SEM, 95% confidence interval,
        median and IQR.

        Returns
        -------
        pd.Series
            Series with keys: 'Mean', 'Std', 'Sem', 'Ci_low', 'Ci_high',
            'Median' and 'IQR'.
        """
        fi_val = [i.get_mean_fpt()["Mean"] for i in self.runs]

        n_fpt = np.sqrt(len(fi_val))

        mean = np.mean(fi_val)
        median = np.median(fi_val)
        q25 = np.percentile(fi_val, 25)
        q75 = np.percentile(fi_val, 75)
        std = np.std(fi_val, ddof=1)
        sem = std / np.sqrt(n_fpt)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem
        return pd.Series(
            {
                "Mean": mean,
                "Std": std,
                "Sem": sem,
                "Ci_low": ci_low,
                "Ci_high": ci_high,
                "Median": median,
                "IQR": q75 - q25,
            }
        )

    def get_mean_hits(self, group_by: str = "Time") -> pd.DataFrame:
        """Compute mean hit counts across all runs, grouped by time.

        Parameters
        ----------
        group_by : str, optional
            Column name to group by (typically 'Time').

        Returns
        -------
        pd.DataFrame
            DataFrame with mean, std, sem and 95% CI for hit counts per group.

        Raises
        ------
        ValueError
            If some runs do not include hit information.
        """
        hits_lst = [i.hits for i in self.runs if i.hits is not None]

        if len(hits_lst) != len(self.runs):
            msg = "Some runs do not include hits"
            raise ValueError(msg)

        all_hits_arr = np.array(
            [i.drop(group_by, axis=1).sum(axis=1).to_numpy() for i in hits_lst]
        )

        groups = hits_lst[0][group_by].to_numpy()
        n_rep = all_hits_arr.shape[0]

        df_stats_hits = {group_by: groups}

        mean = all_hits_arr.mean(axis=0)
        std = all_hits_arr.std(axis=0, ddof=1)
        sem = std / np.sqrt(n_rep)
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

        df_stats_hits["Hits"] = mean
        df_stats_hits["Hits_std"] = std
        df_stats_hits["Hits_sem"] = sem
        df_stats_hits["Hits_ci_low"] = ci_low
        df_stats_hits["Hits_ci_high"] = ci_high

        return pd.DataFrame(df_stats_hits)

    def get_mean_coverage(self) -> float:
        """Return the mean coverage across ensemble members.

        Returns
        -------
        float
            Mean fractional coverage (area covered by shapes) across runs.
        """
        return float(np.mean(self.coverages))


def spawn(
    raster: np.ndarray,
    k: int,
    dimensions: tuple[int, int],
    particle_radius: int | None,
    start: tuple[int, int],
    start_area: np.ndarray | None = None,
    free_centers: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    *,
    random_start: bool = True,
    has_ghost: bool = True,
) -> np.ndarray:
    """Select or generate starting positions for `k` particles on the raster.

    If `random_start` is False the provided `start` coordinate is returned
    repeated `k` times. Otherwise this function samples `k` distinct free
    lattice centers. If `has_ghost` is True the central (unpadded) region is
    used for sampling and an offset is added before returning positions.

    Parameters
    ----------
    raster : np.ndarray
        2D raster where non-zero values indicate occupied/obstacle cells.
    k : int
        Number of start positions to return.
    dimensions : tuple[int, int]
        Simulation box dimensions (min_dim, max_dim) used to compute offsets
        when `has_ghost` is True.
    particle_radius : int | None
        If provided, centers are validated against `free_centers` instead of
        simple raster lookup.
    start : tuple[int, int]
        Fallback or non-random start coordinate.
    start_area : np.ndarray | None, optional
        Boolean mask limiting allowed start locations.
    free_centers : np.ndarray | None, optional
        Boolean raster marking allowed particle centers (used when
        `particle_radius` is provided).
    rng : np.random.Generator | None, optional
        Random number generator for reproducible sampling.
    random_start : bool, optional
        If False, return `start` repeated `k` times; otherwise sample.
    has_ghost : bool, optional
        If True, sampling is performed on a central sub-region and an offset
        of `max_dim` is applied to the returned positions.

    Returns
    -------
    np.ndarray
        Array of shape (k, 2) with integer lattice coordinates for start
        positions.

    Raises
    ------
    RuntimeError
        If no valid start locations are available after filtering.
    RuntimeError
        If less than `k` candidate locations exist.
    """
    if not random_start:
        return np.full((k, 2), start, dtype=int)

    rng = np.random.default_rng() if rng is None else rng

    _, max_dim = dimensions
    max_dim = int(max_dim)

    if has_ghost:
        s = slice(max_dim, 2 * max_dim)
        sub_raster = raster[s, s]
        if start_area is not None:
            start_area = start_area[s, s]
        if free_centers is not None:
            free_centers = free_centers[s, s]
    else:
        sub_raster = raster

    _, w = sub_raster.shape

    valid_mask = sub_raster == 0

    if start_area is not None:
        valid_mask &= start_area != 0

    if particle_radius:
        # kernel = _make_lattice_particle(particle_radius)

        flat_idx = np.flatnonzero(valid_mask.ravel())
        if len(flat_idx) == 0:
            msg = "No valid start lcoations after start area filtering"
            raise RuntimeError(msg)

        candidates = np.column_stack((flat_idx // w, flat_idx % w))

        mask = check_local_free2(free_centers, candidates)
        candidates = candidates[mask]
    else:
        flat_idx = np.flatnonzero(valid_mask.ravel())
        candidates = np.column_stack((flat_idx // w, flat_idx % w))

    # if len(candidates) < k:
    #     msg = f"Not enough spawn points: {len(candidates)} available, {k} requested."
    #     raise RuntimeError(msg)

    idx = rng.choice(len(candidates), size=k, replace=True)
    start_local = candidates[idx]

    return start_local + max_dim


def random_walk_2d_lattice(
    raster: np.ndarray,
    nsteps: int = 10_000,
    nreps: int = 50,
    diff_coefficient: float = 1.0,
    a: int = 1,
    start: np.ndarray | None = None,
    aim_mask: np.ndarray | None = None,
    raster_chosen_obstacles: np.ndarray | None = None,
    free_raster_chosen_obstacles: np.ndarray | None = None,
    particle_radius: None | int = None,
    rng: np.random.Generator | None = None,
    save_every: int = 1,
    dimensions: tuple[int, int] = (0, 2000),
    start_mask: np.ndarray | None = None,
    free_centers: np.ndarray | None = None,
    *,
    store_history: bool = False,
    has_ghost: bool = False,
    random_start: bool = False,
    steady_state_chosen_obstacle: bool = False,
) -> dict:
    """Simulate multiple 2D random-walk trajectories on a lattice raster.

    The function performs `nreps` independent random-walks for `nsteps`
    discrete steps on the provided `raster`. Movement is restricted by the
    raster and optional obstacle masks; displacement statistics (MSD),
    activity flags, first-passage times and (optionally) hit statistics are
    collected and returned.

    Parameters
    ----------
    raster : np.ndarray
        2D occupancy raster where non-zero values indicate blocked cells.
    nsteps : int, optional
        Number of timesteps to simulate per replicate.
    nreps : int, optional
        Number of independent replicates.
    diff_coefficient : float, optional
        Diffusion coefficient used to compute the timestep `dt`.
    a : int, optional
        Lattice spacing (used to compute `dt`).
    start : np.ndarray | None, optional
        Initial integer positions with shape (nreps, 2).
    aim_mask : np.ndarray | None, optional
        Boolean mask marking an "aim" area; used to record when a walker
        is inside that area.
    raster_chosen_obstacles : np.ndarray | None, optional
        Mask selecting a subset of obstacles used for hit detection.
    particle_radius : None | int, optional
        If provided, use a kernel to validate moves against `free_centers`.
    rng : np.random.Generator | None, optional
        RNG for stochastic choices.
    save_every : int, optional
        Interval at which trajectory snapshots are saved.
    dimensions : tuple[int, int], optional
        Simulation box extents used for respawning when required.
    start_mask : np.ndarray | None, optional
        Boolean mask restricting valid respawn locations.
    free_centers : np.ndarray | None, optional
        Precomputed erosion map of allowed particle centers for finite-radius
        particles.
    store_history : bool, optional
        If True, store and return full trajectory coordinates.
    has_ghost : bool, optional
        Whether the raster contains a padded "ghost" border.
    random_start : bool, optional
        If True, respawned particles are placed at random valid starts.
    steady_state_chosen_obstacle : bool, optional
        If True, collect hit counts and waiting times for chosen obstacles.

    Returns
    -------
    dict
        Dictionary containing:
        - 'traj': DataFrame with trajectory data (Time, Replicate, MSD, Active,
          and optionally X, Y and Aim when `store_history` is True).
        - 'fpt': DataFrame with columns ['Replicate', 'FirstInactiveTime'].
        - 'hits': DataFrame or None with per-time hit counts and a 'Time'
          column (present when `steady_state_chosen_obstacle` is True).
        - 'waiting_times': ndarray or None of recorded waiting times for hits.
    """
    if start is None:
        msg = "No start was provided"
        raise ValueError(msg)
    # Preparation: RNG, lattice, time
    rng = np.random.default_rng() if rng is None else rng
    dt = a**2 / (4 * diff_coefficient)
    directions = np.array(_get_lattice_directions("square"), dtype=np.int32)

    # Initial position
    pos = start.astype(np.int32)
    start_pos = pos.copy()

    # Saving setup
    saved_times = np.arange(0, nsteps + 1, save_every, dtype=np.int32)
    n_saved = saved_times.shape[0]

    if store_history:
        pos_hist = np.zeros((n_saved, nreps, 2), dtype=pos.dtype)
        pos_hist[0] = pos
    else:
        pos_hist = None

    if aim_mask is not None:
        ai_hist = np.zeros((n_saved, nreps), dtype=np.int8)
        ai_hist[0] = aim_mask[pos[:, 0], pos[:, 1]]
    else:
        ai_hist = None

    msd = np.zeros((n_saved, nreps), dtype=np.uint32)
    active = np.zeros((n_saved, nreps), dtype=np.uint8)

    # Particle / obstacle helpers
    kernel = _make_lattice_particle(particle_radius) if particle_radius else None
    use_kernel = kernel is not None
    use_chosen = raster_chosen_obstacles is not None

    # State variables
    active_t = np.ones(nreps, dtype=bool)
    inactive_time = np.full(nreps, np.nan, dtype=np.float32)

    hits: np.ndarray | None = None
    hits_buffer: np.ndarray | None = None
    waiting_time: np.ndarray | None = None
    waiting_time_all: list[np.ndarray] = []

    if steady_state_chosen_obstacle:
        hits = np.zeros((n_saved, nreps), dtype=np.uint16)
        hits_buffer = np.zeros(nreps, dtype=np.uint16)
        waiting_time = np.zeros(nreps, dtype=np.float32)

    # Save t = 0
    save_idx = 0
    msd[0] = np.sum((pos - start_pos) ** 2, axis=1)
    active[0] = active_t
    save_idx = 1

    # Main loop
    for t in trange(1, nsteps + 1):
        move_idx = rng.integers(0, 4, size=nreps)
        step = directions[move_idx]
        new_pos = pos + step

        # Free-space check

        if use_kernel:
            mask_free = check_local_free2(free_centers, new_pos)
        else:
            mask_free = raster[new_pos[:, 0], new_pos[:, 1]] == 0

        if use_chosen:
            if use_kernel:
                mask_free_obs = check_local_free2(free_raster_chosen_obstacles, new_pos)
                mask_hit_chosen = ~mask_free_obs
            else:
                mask_hit_chosen = (
                    raster_chosen_obstacles[new_pos[:, 0], new_pos[:, 1]] == 1
                )
            newly_inactive = mask_hit_chosen & np.isnan(inactive_time)
            inactive_time[newly_inactive] = t * dt

            if hits_buffer is not None and waiting_time is not None:
                hits_buffer += mask_hit_chosen.astype(hits_buffer.dtype)
                waiting_time += dt

                if mask_hit_chosen.any():
                    waiting_time_all.extend(waiting_time[mask_hit_chosen])
                    waiting_time[mask_hit_chosen] = 0

                    respawned_pos = spawn(
                        raster,
                        mask_hit_chosen.sum(),
                        dimensions,
                        particle_radius,
                        start[0],
                        start_mask,
                        free_centers,
                        rng,
                        random_start=random_start,
                        has_ghost=has_ghost,
                    )
                    pos[mask_hit_chosen] = respawned_pos
                    active_t[mask_hit_chosen] = True

            else:
                active_t[mask_hit_chosen] = False

        # Update position
        mask_move = mask_free & active_t
        pos[mask_move] = new_pos[mask_move]

        # MSD
        disp = pos - start_pos
        msd_t = np.sum(disp**2, axis=1)

        # Save
        if save_idx < n_saved and t == saved_times[save_idx]:
            if pos_hist is not None:
                pos_hist[save_idx] = pos

            if ai_hist is not None and aim_mask is not None:  # due to type testing
                ai_hist[save_idx] = aim_mask[pos[:, 0], pos[:, 1]]

            msd[save_idx] = msd_t
            active[save_idx] = active_t

            if use_chosen and hits is not None and hits_buffer is not None:
                hits[save_idx] = hits_buffer
                hits_buffer[:] = 0

            save_idx += 1

    # Output

    time_vec = saved_times.astype(np.float32) * dt

    if pos_hist is not None:
        df_traj = pd.DataFrame(
            {
                "Time": np.repeat(time_vec, nreps),
                "Replicate": np.tile(np.arange(nreps), n_saved),
                "Y": pos_hist[:, :, 0].ravel(),
                "X": pos_hist[:, :, 1].ravel(),
                "MSD": msd.ravel(),
                "Active": active.ravel(),
            }
        )
        if ai_hist is not None:
            df_traj["Aim"] = ai_hist.ravel()

    else:
        df_traj = pd.DataFrame(
            {
                "Time": np.repeat(time_vec, nreps),
                "Replicate": np.tile(np.arange(nreps), n_saved),
                "MSD": msd.ravel(),
                "Active": active.ravel(),
            }
        )

        if ai_hist is not None:
            df_traj["Aim"] = ai_hist.ravel()

    df_ftp = pd.DataFrame(
        {"Replicate": np.arange(nreps), "FirstInactiveTime": inactive_time}
    )

    df_hits = None
    if steady_state_chosen_obstacle and use_chosen:
        df_hits = pd.DataFrame(hits)
        df_hits["Time"] = time_vec

    return {
        "traj": df_traj,
        "fpt": df_ftp,
        "hits": df_hits,
        "waiting_times": (
            np.asarray(waiting_time_all) if steady_state_chosen_obstacle else None
        ),
    }


@dataclass(slots=True)
class ExperimentLatticeConfig:
    """
    Configuration for a random walk or diffusion experiment.

    Attributes
    ----------
    nsteps : int
        Number of steps per trajectory.
    dt : float
        Time step size.
    shapes : list[Polygon] | World | None
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
    shapes: Sequence[BaseGeometry] | None = None
    lattice_type: str = "square"
    lattice_resolution: int = 1
    replicates: int = 100
    diff_coefficient: float = 1.0
    start: tuple[int, int] = (0, 0)
    dimensions: tuple[int, int] = (0, 2000)
    coverage_dimension: tuple[float, float] | None = None
    random_start: bool = False
    aim_area: None | Polygon = None
    start_area: None | Polygon = None
    start_area_by_size: None | tuple = None
    chosen_obstacles_by_area: float | None = None
    particle_radius: None | int = None
    has_ghost: bool = True
    store_history: bool = False
    save_every: int = 1
    shift_origin: bool = False
    steady_state_chosen_obstacle: bool = False
    workers: None | int = 4
    rng: None | np.random.Generator = None


@dataclass
class ExperimentLattice:
    """
    Run random walker experiments in parallel.

    Attributes
    ----------
    config : ExperimentConfig
        Configuration for the experiment.
    free_space : Polygon
        Computed free space for random starting positions.
    """

    config: ExperimentLatticeConfig

    def __post_init__(self) -> None:
        """
        Compute the free space for the experiment based on the configuration.
        """
        self._config = deepcopy(self.config)
        raster_chosen_obstacles: None | np.ndarray = None
        free_raster_chosen_obstacles: None | np.ndarray = None
        start_mask: None | np.ndarray = None

        if self._config.start_area and self._config.start_area_by_size:
            msg = "You cannot specify start_area and start_area_by_size at once"
            raise ValueError(msg)

        if self._config.start_area_by_size:
            comparison_size, buffer, constraint_area = self._config.start_area_by_size
            shapes = get_buffered_shapes(
                comparison_size=comparison_size,
                shapes=self._config.shapes,
                buffer=buffer,
                constraint_area=constraint_area,
            )
            self._config.start_area = unary_union(shapes)

        if self._config.shift_origin:
            max_dim = self._config.dimensions[1]
            if self._config.shapes is not None:
                self._config.shapes = [
                    translate(p, xoff=max_dim / 2, yoff=max_dim / 2)
                    for p in self._config.shapes
                ]
            if self._config.aim_area is not None:
                self._config.aim_area = translate(
                    self._config.aim_area, xoff=max_dim / 2, yoff=max_dim / 2
                )
            sx, sy = self._config.start
            self._config.start = (int(sx + max_dim / 2), int(sy + max_dim / 2))
        min_dim, max_dim = self._config.dimensions
        min_dim_cov = min_dim
        max_dim_cov = max_dim
        union_shapes = unary_union(self._config.shapes)
        workspace = box(min_dim, min_dim, max_dim, max_dim)
        if self._config.coverage_dimension:
            min_dim_cov, max_dim_cov = self._config.coverage_dimension
            workspace = box(min_dim_cov, min_dim_cov, max_dim_cov, max_dim_cov)
        self.coverage = workspace.intersection(union_shapes).area / (
            max_dim_cov * max_dim_cov
        )

        if self.config.has_ghost:
            raster = make_raster(
                union_shapes,
                self._config.lattice_resolution,
                -max_dim,
                -max_dim,
                2 * max_dim,
                2 * max_dim,
            )

            if self._config.chosen_obstacles_by_area:
                shapes = self._config.shapes
                if not shapes:
                    msg = "Cannot choose obstacle by area from empty shapes"
                    raise ValueError(msg)
                chosen_particles_by_area = [
                    i
                    for i in shapes
                    if math.isclose(i.area, self._config.chosen_obstacles_by_area)
                ]
                raster_chosen_obstacles = make_raster(
                    unary_union(chosen_particles_by_area),
                    self._config.lattice_resolution,
                    -max_dim,
                    -max_dim,
                    2 * max_dim,
                    2 * max_dim,
                )

            if self._config.start_area:
                start_mask = make_raster(
                    self._config.start_area,
                    self._config.lattice_resolution,
                    -max_dim,
                    -max_dim,
                    2 * max_dim,
                    2 * max_dim,
                )

            self.aim_area = self._config.aim_area

        else:
            raster = make_raster(
                union_shapes, self.config.lattice_resolution, 0, 0, max_dim, max_dim
            )
            raster = np.pad(
                array=raster,
                pad_width=int(max_dim / self.config.lattice_resolution),
                mode="constant",
                constant_values=0,
            )  # type: ignore

            if self._config.chosen_obstacles_by_area:
                shapes = self._config.shapes
                if not shapes:
                    msg = "Cannot choose obstacle by area from empty shapes"
                    raise ValueError(msg)
                chosen_particles_by_area = [
                    i
                    for i in shapes
                    if math.isclose(i.area, self._config.chosen_obstacles_by_area)
                ]

                raster_chosen_obstacles = make_raster(
                    unary_union(chosen_particles_by_area),
                    self._config.lattice_resolution,
                    0,
                    0,
                    max_dim,
                    max_dim,
                )

                raster_chosen_obstacles = np.pad(
                    array=raster_chosen_obstacles,
                    pad_width=int(max_dim / self.config.lattice_resolution),
                    mode="constant",
                    constant_values=0,
                )  # type: ignore

            if self._config.start_area:
                start_mask = make_raster(
                    self._config.start_area,
                    self._config.lattice_resolution,
                    0,
                    0,
                    max_dim,
                    max_dim,
                )

                start_mask = np.pad(
                    array=start_mask,
                    pad_width=int(max_dim / self.config.lattice_resolution),
                    mode="constant",
                    constant_values=0,
                )  # type: ignore

            if self._config.shapes is not None:
                self._config.shapes = [
                    translate(p, xoff=max_dim, yoff=max_dim)
                    for p in self._config.shapes
                ]

            if self._config.aim_area is not None:
                self.aim_area = translate(
                    self._config.aim_area, xoff=max_dim, yoff=max_dim
                )
            else:
                self.aim_area = None

            if self._config.start_area is not None:
                self.start_area = translate(
                    self._config.start_area, xoff=max_dim, yoff=max_dim
                )
            else:
                self.start_area = None

            self._config.has_ghost = True

            self._config.start = tuple(
                np.array(self._config.start) + max_dim / self._config.lattice_resolution
            )

        self._config.dimensions = tuple(
            np.array(self._config.dimensions) / self.config.lattice_resolution
        )

        self.raster = np.flipud(raster)
        self.raster_chosen_obstacles = (
            np.flipud(raster_chosen_obstacles)
            if raster_chosen_obstacles is not None
            else None
        )

        self.start_mask = np.flipud(start_mask) if start_mask is not None else None

        if self.aim_area is not None:
            self.aim_mask = make_aim_mask(
                self.raster,
                self.aim_area,
                lattice_resolution=self._config.lattice_resolution,
            )
        else:
            self.aim_mask = None

        self.free_raster_chosen_obstacles = free_raster_chosen_obstacles
        if self._config.particle_radius:
            kernel = _make_lattice_particle(self._config.particle_radius)
            self.kernel = kernel
            free = ~self.raster.astype(bool)
            self.free_centers = binary_erosion(free, kernel, border_value=0)

            if self.raster_chosen_obstacles is not None:
                free_raster_chosen_obstacles = ~self.raster_chosen_obstacles.astype(
                    bool
                )
                self.free_raster_chosen_obstacles = binary_erosion(
                    free_raster_chosen_obstacles, kernel, border_value=0
                )
        else:
            self.free_centers = None

    def worker(self, nreps: int) -> tuple:
        """Run a single experiment batch and return its results.

        This method prepares start positions and executes `random_walk_2d_lattice`
        for `nreps` replicates. It returns a tuple containing the list of
        `DiffusionLatticeTrajectories` (one per replicate), the FPT DataFrame,
        the hits DataFrame (or None) and an array of waiting times (or None).

        Parameters
        ----------
        nreps : int
            Number of replicates to simulate in this worker call.

        Returns
        -------
        tuple
            (list[DiffusionLatticeTrajectories], pd.DataFrame, pd.DataFrame|None,
             np.ndarray|None)
        """

        start = spawn(
            self.raster,
            self._config.replicates,
            self._config.dimensions,
            self._config.particle_radius,
            self._config.start,
            self.start_mask,
            self.free_centers,
            self._config.rng,
            random_start=self._config.random_start,
            has_ghost=self._config.has_ghost,
        )

        res = random_walk_2d_lattice(
            raster=self.raster,
            nsteps=self._config.nsteps,
            nreps=nreps,
            diff_coefficient=self._config.diff_coefficient,
            a=self._config.lattice_resolution,
            start=start,
            aim_mask=self.aim_mask,
            raster_chosen_obstacles=self.raster_chosen_obstacles,
            free_raster_chosen_obstacles=self.free_raster_chosen_obstacles,
            particle_radius=self._config.particle_radius,
            save_every=self._config.save_every,
            dimensions=self._config.dimensions,
            start_mask=self.start_mask,
            free_centers=self.free_centers,
            rng=self._config.rng,
            store_history=self._config.store_history,
            has_ghost=self._config.has_ghost,
            random_start=self._config.random_start,
            steady_state_chosen_obstacle=self._config.steady_state_chosen_obstacle,
        )

        runs = res["traj"]
        fpt = res["fpt"]
        hits = res["hits"]
        waiting_times = res["waiting_times"]

        return (
            [
                DiffusionLatticeTrajectories(g.reset_index(drop=True))
                for _, g in runs.groupby("Replicate")
            ],
            fpt,
            hits,
            waiting_times,
        )

    def run(self) -> ExperimentLatticeRun:
        """
        Run all replicates of the experiment in parallel.

        Returns
        -------
        ExperimentRun
            Object containing all simulated trajectories.
        """

        runs, fpt, hits, waiting_times = self.worker(self.config.replicates)
        return ExperimentLatticeRun(
            runs,
            self._config.shapes,
            self.raster,
            self._config.has_ghost,
            fpt,
            hits,
            waiting_times,
        )


def _run_single_lattice_experiment(
    shapes_wkt: str, config_dict: dict
) -> tuple[ExperimentLatticeRun, float]:
    polygons = [shapely.wkt.loads(w) for w in shapes_wkt]
    cfg = ExperimentLatticeConfig(**config_dict)
    cfg.shapes = polygons
    exp = ExperimentLattice(cfg)
    run = exp.run()
    return run, exp.coverage


@dataclass
class EnsembleExperimentLattice:
    """
    Run an ensemble of experiments, each with a different geometry or file.

    Attributes
    ----------
    file_paths : list[str]
        List of file paths to geometry files.
    config : ExperimentConfig
        Configuration for the experiments.
    """

    file_paths: list[Path]
    config: ExperimentLatticeConfig

    def run(self) -> EnsembleExperimentLatticeRun:
        """
        Run random walk experiments for each file in the ensemble.

        For each file path, reads the geometry, updates the experiment configuration,
        runs the experiment, and collects the results into an ensemble.

        Returns
        -------
        EnsembleExperimentRun
            An object containing the results of all experiments in the ensemble.
        """
        config_dict = dataclasses.asdict(self.config)

        all_wkt_lists = []
        for f in self.file_paths:
            polygons = readwkt(f)
            wkt_list = [p.wkt for p in polygons]
            all_wkt_lists.append(wkt_list)

        ensemble_lst = []
        coverage_lst = []

        with ProcessPoolExecutor(max_workers=self.config.workers) as pool:
            futures = [
                pool.submit(_run_single_lattice_experiment, wkt_list, config_dict)
                for wkt_list in all_wkt_lists
            ]

            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Running experiments"
            ):
                run, cov = fut.result()
                ensemble_lst.append(run)
                coverage_lst.append(cov)

        return EnsembleExperimentLatticeRun(ensemble_lst, coverage_lst)
