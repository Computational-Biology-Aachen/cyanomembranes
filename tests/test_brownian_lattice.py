"""Tests for lattice diffusion: tutorial sections 1-6 and core diffusion.

All settings are deliberately tiny (small boxes, few steps, fixed seeds) so the
tests are fast and cannot exhaust memory. The tutorial itself uses a 2000x2000
box, which is far too large for tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
import warnings
from concurrent.futures import Future

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import shapely
from scipy.ndimage import binary_erosion, distance_transform_edt
from shapely import to_wkt
from shapely.geometry import Point, box

from cyanomembranes import brownian_lattice as bl

pytestmark = [
    pytest.mark.filterwarnings("ignore::rasterio.errors.ShapeSkipWarning"),
    pytest.mark.filterwarnings(
        "ignore:The 'shapely.vectorized.contains':DeprecationWarning"
    ),
]

DIFF = 3.0
NSTEPS = 100
NREPS = 300


def _walk(
    *,
    nreps=20,
    nsteps=30,
    diff=2.0,
    a=1,
    seed=0,
    size=101,
    start=None,
    **kwargs,
):
    """Run `random_walk_2d_lattice` on an empty raster, starting in the middle."""
    raster = np.zeros((size, size), dtype=np.uint8)
    if start is None:
        start = np.full((nreps, 2), size // 2)
    return bl.random_walk_2d_lattice(
        raster,
        nsteps=nsteps,
        nreps=nreps,
        diff_coefficient=diff,
        a=a,
        start=start,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def _config(**overrides):
    params = {
        "dimensions": (0, 20),
        "replicates": 10,
        "nsteps": 20,
        "diff_coefficient": DIFF,
        "has_ghost": False,
        "store_history": True,
        "rng": np.random.default_rng(0),
    }
    params.update(overrides)
    return bl.ExperimentLatticeConfig(**params)


def _stack(traj: pd.DataFrame, column: str, nreps: int) -> np.ndarray:
    """Reshape a flat trajectory column to (n_saved, nreps)."""
    return traj[column].to_numpy().reshape(-1, nreps)


@pytest.fixture(scope="module")
def lattice_run():
    cfg = _config(
        dimensions=(0, 200),
        replicates=NREPS,
        nsteps=NSTEPS,
        rng=np.random.default_rng(1),
    )
    return bl.ExperimentLattice(cfg).run()


# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------


def test_wrap_positions_periodic():
    pos = np.array([[10, 10], [20, 19], [9, 25]])
    wrapped = bl._wrap_positions(pos, offset=10, size=10)
    assert np.array_equal(wrapped, [[10, 10], [10, 19], [19, 15]])
    assert wrapped.min() >= 10
    assert wrapped.max() < 20


def test_spawn_fixed_start():
    raster = np.zeros((20, 20), dtype=np.uint8)
    out = bl.spawn(raster, 5, (0, 10), None, (3, 4), random_start=False)
    assert out.shape == (5, 2)
    assert np.issubdtype(out.dtype, np.integer)
    assert (out == [3, 4]).all()


def test_spawn_random_start_free_and_in_bounds():
    max_dim = 10
    raster = np.zeros((3 * max_dim, 3 * max_dim), dtype=np.uint8)
    raster[12:15, 12:15] = 1  # obstacle inside the central region
    out = bl.spawn(
        raster,
        200,
        (0, max_dim),
        None,
        (0, 0),
        rng=np.random.default_rng(0),
        random_start=True,
        has_ghost=True,
    )
    assert out.shape == (200, 2)
    assert (out >= max_dim).all()
    assert (out < 2 * max_dim).all()
    assert (raster[out[:, 0], out[:, 1]] == 0).all()


def test_spawn_random_start_without_ghost_stays_in_raster():
    # Docstring: the offset is only applied when `has_ghost` is True.
    raster = np.zeros((20, 20), dtype=np.uint8)
    out = bl.spawn(
        raster,
        50,
        (0, 10),
        None,
        (0, 0),
        rng=np.random.default_rng(0),
        random_start=True,
        has_ghost=False,
    )
    assert (out >= 0).all()
    assert (out < 20).all()


def test_spawn_random_start_samples_with_replacement():
    # k may exceed the number of free cells: replicates are independent walkers,
    # so equal start positions are allowed.
    raster = np.ones((30, 30), dtype=np.uint8)
    raster[14, 14] = 0
    raster[15, 15] = 0  # two free cells in the central region
    out = bl.spawn(
        raster,
        50,
        (0, 10),
        None,
        (0, 0),
        rng=np.random.default_rng(0),
        random_start=True,
        has_ghost=True,
    )
    assert out.shape == (50, 2)
    # the central region is raster[10:20, 10:20]; the offset max_dim = 10 is
    # added to the local cells (4, 4) and (5, 5)
    assert {tuple(p) for p in out} == {(14, 14), (15, 15)}


# --------------------------------------------------------------------------
# random_walk_2d_lattice
# --------------------------------------------------------------------------


def test_random_walk_requires_start():
    raster = np.zeros((11, 11), dtype=np.uint8)
    with pytest.raises(ValueError, match="No start"):
        bl.random_walk_2d_lattice(raster, nsteps=3, nreps=2, start=None)


def test_random_walk_steps_are_unit_moves():
    nreps, nsteps = 20, 30
    traj = _walk(nreps=nreps, nsteps=nsteps, store_history=True)["traj"]
    y = _stack(traj, "Y", nreps).astype(int)
    x = _stack(traj, "X", nreps).astype(int)
    assert y.shape == (nsteps + 1, nreps)
    step_size = np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0))
    assert (step_size == 1).all()  # every step is exactly one of the 4 moves


def test_random_walk_msd_matches_history():
    nreps = 20
    traj = _walk(nreps=nreps, store_history=True)["traj"]
    y = _stack(traj, "Y", nreps).astype(int)
    x = _stack(traj, "X", nreps).astype(int)
    expected = (y - y[0]) ** 2 + (x - x[0]) ** 2
    assert np.array_equal(_stack(traj, "MSD", nreps), expected)
    assert (_stack(traj, "MSD", nreps)[0] == 0).all()


@pytest.mark.parametrize("store_history", [True, False])
def test_random_walk_mad_is_float_and_not_truncated(store_history):
    traj = _walk(store_history=store_history)["traj"]
    assert traj["MAD"].dtype.kind == "f"
    assert (traj["MAD"] % 1 != 0).any()  # distances such as sqrt(2) are kept


def test_random_walk_mad_independent_of_store_history():
    with_hist = _walk(store_history=True)["traj"]
    without = _walk(store_history=False)["traj"]
    assert np.array_equal(with_hist["MAD"], without["MAD"])


@pytest.mark.parametrize("store_history", [True, False])
def test_random_walk_mean_mad_follows_rayleigh_ratio(store_history):
    # 2D free diffusion: <|r|> / sqrt(<r^2>) = sqrt(pi) / 2 (about 0.886).
    traj = _walk(
        nreps=2000, nsteps=200, size=1001, store_history=store_history, seed=1
    )["traj"]
    last = traj[traj["Time"] == traj["Time"].max()]
    ratio = last["MAD"].mean() / np.sqrt(last["MSD"].mean())
    assert ratio == pytest.approx(np.sqrt(np.pi) / 2, abs=0.02)


def test_random_walk_time_axis():
    nreps, nsteps, diff, a = 5, 40, 2.0, 2
    traj = _walk(nreps=nreps, nsteps=nsteps, diff=diff, a=a)["traj"]
    dt = a**2 / (4 * diff)
    times = traj["Time"].to_numpy().reshape(-1, nreps)
    assert len(traj) == (nsteps + 1) * nreps
    assert (times[0] == 0).all()
    assert np.allclose(times[:, 0], np.arange(nsteps + 1) * dt, rtol=1e-5)
    assert (traj["Replicate"].to_numpy().reshape(-1, nreps) == np.arange(nreps)).all()


def test_random_walk_save_every():
    nreps, nsteps, every = 10, 30, 5
    full = _walk(nreps=nreps, nsteps=nsteps, store_history=True)["traj"]
    sub = _walk(nreps=nreps, nsteps=nsteps, save_every=every, store_history=True)[
        "traj"
    ]
    assert len(sub) == (nsteps // every + 1) * nreps
    for col in ("Y", "X", "MSD"):
        assert np.array_equal(
            _stack(sub, col, nreps), _stack(full, col, nreps)[::every]
        )
    assert np.allclose(_stack(sub, "Time", nreps), _stack(full, "Time", nreps)[::every])


def test_random_walk_reproducible_with_seed():
    first = _walk(seed=3, store_history=True)["traj"]
    again = _walk(seed=3, store_history=True)["traj"]
    other = _walk(seed=4, store_history=True)["traj"]
    pd.testing.assert_frame_equal(first, again)
    assert not first["X"].equals(other["X"])


def test_random_walk_mad_matches_sqrt_msd_with_history():
    traj = _walk(store_history=True)["traj"]
    assert np.allclose(traj["MAD"], np.sqrt(traj["MSD"]), atol=1e-6)


def test_random_walk_mad_matches_sqrt_msd_without_history():
    traj = _walk(store_history=False)["traj"]
    assert np.allclose(traj["MAD"], np.sqrt(traj["MSD"]), atol=1e-6)


def test_random_walk_msd_grows_linearly():
    nreps, nsteps = 2000, 50
    traj = _walk(nreps=nreps, nsteps=nsteps, size=201)["traj"]
    mean_msd = _stack(traj, "MSD", nreps).mean(axis=1)
    steps = np.arange(nsteps + 1)
    slope = (steps * mean_msd).sum() / (steps**2).sum()  # fit through the origin
    assert slope == pytest.approx(1.0, rel=0.05)  # MSD(n) = n on the square lattice


def test_random_walk_positions_stay_in_raster_bounds():
    size, nreps = 5, 50
    start = np.full((nreps, 2), size // 2)
    traj = _walk(nreps=nreps, nsteps=30, size=size, start=start, store_history=True)[
        "traj"
    ]
    assert (traj[["Y", "X"]].to_numpy() >= 0).all()
    assert (traj[["Y", "X"]].to_numpy() < size).all()


def test_random_walk_negative_positions_not_allowed():
    nreps = 50
    traj = _walk(
        nreps=nreps,
        nsteps=1,
        size=5,
        start=np.zeros((nreps, 2), dtype=int),
        store_history=True,
    )["traj"]
    assert (traj[["Y", "X"]].to_numpy().astype(int) >= 0).all()


@pytest.mark.parametrize("corner", [(0, 0), (0, 4), (4, 0), (4, 4)])
def test_random_walk_corner_start_stays_inside_and_is_blocked(corner):
    nreps, nsteps, size = 30, 200, 5
    start = np.tile(corner, (nreps, 1))
    traj = _walk(
        nreps=nreps, nsteps=nsteps, size=size, start=start, store_history=True
    )["traj"]
    y = _stack(traj, "Y", nreps).astype(int)
    x = _stack(traj, "X", nreps).astype(int)
    assert ((y >= 0) & (y < size) & (x >= 0) & (x < size)).all()
    step_size = np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0))
    assert set(np.unique(step_size)) == {0, 1}  # blocked steps leave the walker put


def test_random_walk_edge_wall_with_chosen_obstacle():
    nreps, size = 30, 7
    chosen = np.zeros((size, size), dtype=np.uint8)
    chosen[3, 3] = 1
    res = _walk(
        nreps=nreps,
        nsteps=300,
        size=size,
        start=np.zeros((nreps, 2), dtype=int),
        store_history=True,
        raster_chosen_obstacles=chosen,
    )
    pos = res["traj"][["Y", "X"]].to_numpy().astype(int)
    assert ((pos >= 0) & (pos < size)).all()
    assert res["fpt"]["FirstInactiveTime"].notna().any()  # the obstacle is found


def test_random_walk_wall_msd_below_free_msd():
    nreps, nsteps = 400, 40
    free = _walk(nreps=nreps, nsteps=nsteps, size=101, seed=2)["traj"]
    boxed = _walk(
        nreps=nreps,
        nsteps=nsteps,
        size=5,
        start=np.full((nreps, 2), 2),
        seed=2,
    )["traj"]
    last = free["Time"].max()
    assert (
        boxed[boxed["Time"] == last]["MSD"].mean()
        < 0.5 * free[free["Time"] == last]["MSD"].mean()
    )


# --------------------------------------------------------------------------
# Config and ExperimentLattice setup
# --------------------------------------------------------------------------


def test_config_defaults():
    cfg = bl.ExperimentLatticeConfig()
    assert cfg.nsteps == 1000
    assert cfg.replicates == 100
    assert cfg.diff_coefficient == 1.0
    assert cfg.start == (0, 0)
    assert cfg.dimensions == (0, 2000)
    assert cfg.lattice_resolution == 1
    assert cfg.shapes is None
    assert cfg.has_ghost is True
    assert cfg.store_history is False
    assert cfg.save_every == 1
    assert cfg.random_start is False
    assert cfg.wrap is False
    assert cfg.rng is None


@pytest.mark.parametrize("has_ghost", [True, False])
@pytest.mark.parametrize("resolution", [1, 2])
def test_experiment_raster_shape_ghost_vs_no_ghost(has_ghost, resolution):
    exp = bl.ExperimentLattice(
        _config(has_ghost=has_ghost, lattice_resolution=resolution)
    )
    side = 3 * 20 // resolution  # raster spans -max_dim .. 2 * max_dim
    assert exp.raster.shape == (side, side)
    assert exp.raster.sum() == 0  # no shapes: completely free


def test_experiment_start_shifted_without_ghost():
    cfg = _config(has_ghost=False, lattice_resolution=2, start=(0, 0))
    exp = bl.ExperimentLattice(cfg)
    assert tuple(exp._config.start) == (10, 10)  # start + max_dim / resolution
    assert cfg.start == (0, 0)  # the user's config is not mutated
    assert cfg.has_ghost is False


def test_experiment_start_shifted_with_ghost():
    cfg = _config(has_ghost=True, start=(3, 4))
    exp = bl.ExperimentLattice(cfg)
    assert tuple(exp._config.start) == (23, 24)  # start + max_dim / resolution
    assert cfg.start == (3, 4)  # the user's config is not mutated


@pytest.mark.parametrize("resolution", [1, 2])
def test_experiment_start_and_aim_same_for_ghost_and_non_ghost(resolution):
    kwargs = {
        "dimensions": (0, 60),
        "lattice_resolution": resolution,
        "start": (10, 20),
        "aim_area": Point(10, 20).buffer(6),
    }
    ghost = bl.ExperimentLattice(_config(has_ghost=True, **kwargs))
    plain = bl.ExperimentLattice(_config(has_ghost=False, **kwargs))
    assert tuple(ghost._config.start) == tuple(plain._config.start)
    assert ghost.aim_area.equals(plain.aim_area)
    assert np.array_equal(ghost.aim_mask, plain.aim_mask)


def test_experiment_dimensions_scaled_by_resolution():
    cfg = _config(lattice_resolution=2)
    exp = bl.ExperimentLattice(cfg)
    assert tuple(exp._config.dimensions) == (0, 10)
    assert cfg.dimensions == (0, 20)


def test_experiment_coverage_zero_without_shapes():
    assert bl.ExperimentLattice(_config()).coverage == 0.0


def test_run_structure():
    cfg = _config(replicates=7, nsteps=15)
    run = bl.ExperimentLattice(cfg).run()
    assert len(run.trajectories) == 7
    expected = ["Time", "Replicate", "Y", "X", "MSD", "MAD", "Active"]
    for rep, trajectory in enumerate(run.trajectories):
        assert list(trajectory.data.columns) == expected
        assert len(trajectory.data) == 16  # nsteps + 1
        assert (trajectory.data["Replicate"] == rep).all()
    assert len(run.fpt) == 7
    assert run.fpt["FirstInactiveTime"].isna().all()  # no chosen obstacles


def test_run_without_history_has_no_positions():
    run = bl.ExperimentLattice(_config(store_history=False)).run()
    assert "X" not in run.trajectories[0].data.columns
    assert "Y" not in run.trajectories[0].data.columns


def test_run_reproducible_with_config_rng():
    first = bl.ExperimentLattice(_config(rng=np.random.default_rng(7))).run()
    again = bl.ExperimentLattice(_config(rng=np.random.default_rng(7))).run()
    pd.testing.assert_frame_equal(first.mean_trajectories(), again.mean_trajectories())


def test_worker_nreps_differs_from_replicates():
    exp = bl.ExperimentLattice(_config(replicates=6))
    trajectories, _, _, _ = exp.worker(4)
    assert len(trajectories) == 4


@pytest.mark.parametrize("random_start", [False, True])
def test_worker_runs_exactly_nreps_walkers(random_start):
    exp = bl.ExperimentLattice(_config(replicates=6, random_start=random_start))
    trajectories, fpt, _, _ = exp.worker(3)
    assert len(trajectories) == 3
    assert len(fpt) == 3


def test_run_uses_all_replicates():
    run = bl.ExperimentLattice(_config(replicates=6, nsteps=2)).run()
    assert len(run.trajectories) == 6


# --------------------------------------------------------------------------
# Run statistics
# --------------------------------------------------------------------------


def test_mean_trajectories_columns_and_values(lattice_run):
    df = lattice_run.mean_trajectories()
    metrics = ["MSD", "Active", "MAD"]
    expected = ["Time"]
    for m in metrics:
        expected += [m, f"{m}_std", f"{m}_sem", f"{m}_ci_low", f"{m}_ci_high"]
    assert set(df.columns) == set(expected)
    assert len(df) == NSTEPS + 1
    assert df["Time"].iloc[0] == 0
    assert df["MSD"].iloc[0] == 0
    manual = np.mean([t.data["MSD"].to_numpy() for t in lattice_run.trajectories], 0)
    assert np.allclose(df["MSD"], manual, rtol=1e-5)
    for m in metrics:
        assert (df[f"{m}_ci_low"] <= df[m] + 1e-6).all()
        assert (df[m] <= df[f"{m}_ci_high"] + 1e-6).all()


def test_mean_trajectories_std_sem_ci_formulas(lattice_run):
    df = lattice_run.mean_trajectories()
    n = len(lattice_run.trajectories)
    for m in ("MSD", "MAD", "Active"):
        # Independent implementation with pandas (rows: replicates, cols: time).
        wide = pd.DataFrame(
            [t.data[m].to_numpy(dtype=float) for t in lattice_run.trajectories]
        )
        mean = wide.mean(axis=0).to_numpy()
        std = wide.std(axis=0, ddof=1).to_numpy()
        sem = wide.sem(axis=0, ddof=1).to_numpy()
        assert np.allclose(df[m], mean, rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_std"], std, rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_sem"], sem, rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_sem"], std / np.sqrt(n), rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_ci_low"], mean - 1.96 * sem, rtol=1e-4, atol=1e-4)
        assert np.allclose(df[f"{m}_ci_high"], mean + 1.96 * sem, rtol=1e-4, atol=1e-4)


def test_mean_trajectories_ci_coverage(lattice_run):
    # On the square lattice the true mean MSD after n steps is exactly n.
    df = lattice_run.mean_trajectories().iloc[1:]
    steps = np.arange(1, NSTEPS + 1)
    inside = (df["MSD_ci_low"].to_numpy() <= steps) & (
        steps <= df["MSD_ci_high"].to_numpy()
    )
    assert inside.mean() >= 0.85  # nominal 95 %, loose bound


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mean_trajectories_single_replicate():
    run = bl.ExperimentLattice(_config(replicates=1)).run()
    df = run.mean_trajectories()
    assert len(df) == 21
    assert df["MSD_std"].isna().all()  # ddof=1 with a single replicate


def test_diff_coefficient_recovers_D(lattice_run):
    _, d_est = lattice_run.get_mean_diff_coefficients()
    assert d_est == pytest.approx(DIFF, rel=0.15)


def test_diff_coefficient_resolution_scaling():
    cfg = _config(
        dimensions=(0, 200),
        replicates=NREPS,
        nsteps=NSTEPS,
        lattice_resolution=2,
        rng=np.random.default_rng(2),
    )
    run = bl.ExperimentLattice(cfg).run()
    _, d_est = run.get_mean_diff_coefficients()
    # The tutorial multiplies by resolution**2 to get back to physical units.
    assert d_est * 2**2 == pytest.approx(DIFF, rel=0.15)


def test_diff_coefficient_pw_exponent_one(lattice_run):
    _, _, alpha = lattice_run.get_mean_diff_coefficients_pw()
    assert alpha == pytest.approx(1.0, abs=0.1)


def test_diff_coefficient_pw_prefactor_is_4D(lattice_run):
    # Pins the current behaviour: the prefactor is 4 D / a^2, not D. The
    # tutorial prints it as "diffusion coefficient (power law)".
    _, prefactor, _ = lattice_run.get_mean_diff_coefficients_pw()
    assert prefactor == pytest.approx(4 * DIFF, rel=0.15)


def test_diff_coefficient_dist_constant(lattice_run):
    dist = lattice_run.get_mean_diff_coefficients_dist()
    assert np.isnan(dist["D_dist"].iloc[0])  # 0 / 0 at t = 0
    assert dist["D_dist"].iloc[1:].median() == pytest.approx(4 * DIFF, rel=0.10)


# ==========================================================================
# Section 2: aim area
# ==========================================================================


def _inside_oracle(shape, polygon, resolution=1):
    """Pointwise reference mask: lattice point (row, col) is at (col, row) * res."""
    mask = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        for col in range(shape[1]):
            mask[row, col] = polygon.contains(Point(col * resolution, row * resolution))
    return mask


def _aim_config(**overrides):
    params = {
        "dimensions": (0, 60),
        "replicates": 10,
        "nsteps": 20,
        "start": (10, 10),
        "aim_area": Point(10, 10).buffer(3),
    }
    params.update(overrides)
    return _config(**params)


def test_make_aim_mask_shape_dtype_and_symmetry():
    raster = np.zeros((50, 60), dtype=np.uint8)
    mask = bl.make_aim_mask(raster, Point(25, 20).buffer(5), 1)
    assert mask.shape == raster.shape
    assert mask.dtype == bool
    assert mask[20, 25]  # row = y, col = x
    assert not mask[0, 0]
    assert mask.sum() == pytest.approx(np.pi * 5**2, rel=0.15)  # boundary excluded
    assert mask[20, 25 - 4 : 25 + 5].all()
    assert not mask[20, 25 + 5]  # radius 5: boundary point is outside


def test_make_aim_mask_boundary_points_are_outside():
    # `contains` semantics: lattice points on the polygon boundary are not inside.
    mask = bl.make_aim_mask(np.zeros((30, 30), dtype=np.uint8), box(10, 20, 15, 30), 1)
    assert mask.sum() == 4 * 9  # interior points x in 11..14, y in 21..29
    assert mask[21:30, 11:15].all()


def test_make_aim_mask_resolution_scaling():
    polygon = box(10.0, 12.0, 30.0, 26.0)  # in world units, resolution 2
    mask = bl.make_aim_mask(np.zeros((20, 20), dtype=np.uint8), polygon, 2)
    assert np.array_equal(mask, _inside_oracle(mask.shape, polygon, resolution=2))


def test_make_aim_mask_matches_pointwise_containment():
    polygon = Point(15.3, 14.7).buffer(6.2)  # fractional bounds on purpose
    mask = bl.make_aim_mask(np.zeros((30, 30), dtype=np.uint8), polygon, 1)
    assert np.array_equal(mask, _inside_oracle(mask.shape, polygon))


def test_make_aim_mask_polygon_partly_outside_raster():
    raster = np.zeros((30, 30), dtype=np.uint8)
    polygon = Point(28, 15).buffer(5)
    mask = bl.make_aim_mask(raster, polygon, 1)
    assert mask.shape == raster.shape
    assert np.array_equal(mask, _inside_oracle(raster.shape, polygon))


@pytest.mark.parametrize("shape", [(20, 60), (60, 20)])
def test_make_aim_mask_non_square_raster(shape):
    polygon = box(5.5, 4.5, 16.5, 12.5)
    mask = bl.make_aim_mask(np.zeros(shape, dtype=np.uint8), polygon, 1)
    assert np.array_equal(mask, _inside_oracle(shape, polygon))
    assert mask.sum() == 11 * 8


@pytest.mark.parametrize("resolution", [1, 2])
def test_make_aim_mask_negative_lower_bound(resolution):
    polygon = box(-3.4, -2.6, 6.3, 5.7)  # partly left of / below the raster
    shape = (12, 15)
    mask = bl.make_aim_mask(np.zeros(shape, dtype=np.uint8), polygon, resolution)
    assert np.array_equal(mask, _inside_oracle(shape, polygon, resolution=resolution))
    assert mask.any()
    assert not mask[:, -1].any()  # nothing wrapped around to the far edge


@pytest.mark.parametrize(
    "polygon", [box(100, 100, 110, 110), box(-30, -30, -10, -10), box(-5, 40, 5, 50)]
)
def test_make_aim_mask_polygon_outside_raster_is_empty(polygon):
    mask = bl.make_aim_mask(np.zeros((30, 30), dtype=np.uint8), polygon, 1)
    assert mask.shape == (30, 30)
    assert not mask.any()


def test_make_aim_mask_emits_no_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bl.make_aim_mask(np.zeros((30, 30), dtype=np.uint8), Point(15, 15).buffer(4), 1)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_experiment_without_aim_has_no_aim_mask_or_column():
    exp = bl.ExperimentLattice(_config())
    assert exp.aim_mask is None
    run = exp.run()
    assert "Aim" not in run.trajectories[0].data.columns
    assert "Aim" not in run.mean_trajectories().columns


def test_experiment_aim_mask_aligned_with_raster_without_ghost():
    exp = bl.ExperimentLattice(_aim_config())
    assert exp.aim_mask.shape == exp.raster.shape
    assert exp.aim_mask.dtype == bool
    # World (10, 10) -> raster index (10, 10) + max_dim (60) in the padded raster.
    assert np.argwhere(exp.aim_mask).mean(axis=0) == pytest.approx([70, 70], abs=0.6)


def test_experiment_aim_mask_resolution_two():
    exp = bl.ExperimentLattice(_aim_config(lattice_resolution=2))
    # World (10, 10) at resolution 2 -> index 5 + max_dim / 2 (30).
    assert np.argwhere(exp.aim_mask).mean(axis=0) == pytest.approx([35, 35], abs=0.6)
    assert exp.aim_mask.sum() < _aim_config().aim_area.area  # fewer, coarser cells


@pytest.mark.parametrize("resolution", [1, 2])
def test_experiment_aim_mask_aligned_with_raster_with_ghost(resolution):
    exp = bl.ExperimentLattice(
        _aim_config(has_ghost=True, lattice_resolution=resolution)
    )
    # World (10, 10) -> index 10 / resolution + max_dim / resolution.
    expected = (10 + 60) / resolution
    assert np.argwhere(exp.aim_mask).mean(axis=0) == pytest.approx(
        [expected, expected], abs=0.6
    )


def test_run_aim_at_fixed_start_with_ghost():
    inside = bl.ExperimentLattice(_aim_config(has_ghost=True)).run()
    outside = bl.ExperimentLattice(_aim_config(has_ghost=True, start=(40, 40))).run()
    assert inside.mean_trajectories()["Aim"].iloc[0] == 1.0
    assert outside.mean_trajectories()["Aim"].iloc[0] == 0.0


def test_run_aim_at_start_inside_and_outside():
    inside = bl.ExperimentLattice(_aim_config()).run().mean_trajectories()
    outside = (
        bl.ExperimentLattice(_aim_config(start=(40, 40))).run().mean_trajectories()
    )
    assert inside["Aim"].iloc[0] == 1.0
    assert outside["Aim"].iloc[0] == 0.0


@pytest.mark.parametrize("store_history", [True, False])
def test_run_aim_column_present_with_and_without_history(store_history):
    run = bl.ExperimentLattice(_aim_config(store_history=store_history)).run()
    aim = run.trajectories[0].data["Aim"]
    assert len(aim) == 21
    assert set(aim.unique()) <= {0, 1}
    assert aim.iloc[0] == 1


def test_random_walk_aim_matches_positions():
    nreps, size = 20, 101
    mask = np.zeros((size, size), dtype=bool)
    mask[:, 50:] = True  # right half is the aim area
    res = _walk(nreps=nreps, nsteps=30, aim_mask=mask, store_history=True)
    traj = res["traj"]
    y = _stack(traj, "Y", nreps).astype(int)
    x = _stack(traj, "X", nreps).astype(int)
    assert np.array_equal(_stack(traj, "Aim", nreps), mask[y, x].astype(int))


def test_mean_trajectories_aim_columns_and_fraction():
    run = bl.ExperimentLattice(_aim_config(replicates=50)).run()
    df = run.mean_trajectories()
    for suffix in ("", "_std", "_sem", "_ci_low", "_ci_high"):
        assert f"Aim{suffix}" in df.columns
    assert df["Aim"].between(0, 1).all()  # a fraction of walkers, not a percentage
    manual = np.mean([t.data["Aim"].to_numpy() for t in run.trajectories], axis=0)
    assert np.allclose(df["Aim"], manual, atol=1e-6)


def test_mean_trajectories_aim_std_sem_ci_formulas():
    run = bl.ExperimentLattice(_aim_config(replicates=50)).run()
    df = run.mean_trajectories()
    wide = pd.DataFrame([t.data["Aim"].to_numpy(dtype=float) for t in run.trajectories])
    mean, sem = wide.mean(axis=0).to_numpy(), wide.sem(axis=0, ddof=1).to_numpy()
    assert np.allclose(df["Aim_std"], wide.std(axis=0, ddof=1), atol=1e-5)
    assert np.allclose(df["Aim_sem"], sem, atol=1e-5)
    assert np.allclose(df["Aim_ci_low"], mean - 1.96 * sem, atol=1e-5)
    assert np.allclose(df["Aim_ci_high"], mean + 1.96 * sem, atol=1e-5)


def test_aim_fraction_matches_diffusion_theory():
    # Start in the middle of a disc (radius 3). After n steps the walker density
    # is Gaussian with sigma^2 = n / 2 per axis; the disc holds 25 lattice points
    # (13 of one parity, 12 of the other, each reachable on alternate steps).
    cfg = _aim_config(replicates=2000, nsteps=100, rng=np.random.default_rng(5))
    df = bl.ExperimentLattice(cfg).run().mean_trajectories()
    assert df["Aim"].iloc[0] == 1.0
    late = df["Aim"].iloc[90:101].mean()
    assert 0.06 < late < 0.11  # theory: about 0.083
    assert df["Aim"].iloc[3] < 1.0  # first walkers leave after 3 steps


@pytest.mark.parametrize("has_ghost", [False, True])
def test_aim_covering_domain_contains_all_random_starts(has_ghost):
    cfg = _aim_config(
        has_ghost=has_ghost,
        dimensions=(0, 40),
        random_start=True,
        replicates=100,
        aim_area=box(-0.5, -0.5, 40.0, 40.0),
    )
    df = bl.ExperimentLattice(cfg).run().mean_trajectories()
    assert df["Aim"].iloc[0] == 1.0


def test_plot_run_aim_area_splits_points_by_aim():
    mpl.use("Agg")
    run = bl.ExperimentLattice(_aim_config(replicates=3, nsteps=40)).run()
    ax = run.plot_run(aim_area=True)
    assert len(ax.collections) == 2 * len(run.trajectories)
    data = run.trajectories[0].data
    in_aim, out_aim = ax.collections[0], ax.collections[1]
    assert len(in_aim.get_offsets()) == (data["Aim"] == 1).sum()
    assert len(out_aim.get_offsets()) == (data["Aim"] == 0).sum()
    # Positions are plotted in world units: the max_dim (60) padding is removed.
    expected = data.loc[data["Aim"] == 1, ["X", "Y"]].to_numpy() - 60
    assert np.array_equal(np.ma.getdata(in_aim.get_offsets()), expected)
    mpl.pyplot.close("all")


# ==========================================================================
# Section 3: chosen obstacles (hits, first hitting time, steady state)
# ==========================================================================

CELL = (5, 5)
DT_UNIT = 0.25  # dt = a^2 / (4 D) with a = D = 1


def _enclosed():
    """11x11 raster with CELL enclosed by four chosen obstacle cells."""
    raster = np.zeros((11, 11), dtype=np.uint8)
    for cell in [(4, 5), (6, 5), (5, 4), (5, 6)]:
        raster[cell] = 1
    return raster, raster.copy()


def _chosen_walk(raster, chosen, *, nreps, nsteps, seed=0, **kwargs):
    return bl.random_walk_2d_lattice(
        raster,
        nsteps=nsteps,
        nreps=nreps,
        diff_coefficient=1.0,
        a=1,
        start=np.full((nreps, 2), CELL),
        raster_chosen_obstacles=chosen,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def _ring():
    return Point(20, 20).buffer(8).difference(Point(20, 20).buffer(6))


def _ring_config(**overrides):
    ring = _ring()
    params = {
        "dimensions": (0, 40),
        "shapes": [ring],
        "chosen_obstacles_by_area": ring.area,
        "start": (20, 20),
        "replicates": 100,
        "nsteps": 300,
        "aim_area": None,
        "rng": np.random.default_rng(3),
    }
    params.update(overrides)
    return _config(**params)


@pytest.fixture(scope="module")
def ring_terminating():
    return bl.ExperimentLattice(_ring_config()).run()


def _ring_arrays(run):
    y = np.stack([t.data["Y"].to_numpy() for t in run.trajectories], axis=1)
    x = np.stack([t.data["X"].to_numpy() for t in run.trajectories], axis=1)
    active = np.stack([t.data["Active"].to_numpy() for t in run.trajectories], axis=1)
    msd = np.stack([t.data["MSD"].to_numpy() for t in run.trajectories], axis=1)
    return y, x, active, msd


def test_enclosed_walker_terminating():
    raster, chosen = _enclosed()
    nreps, nsteps = 5, 6
    res = _chosen_walk(raster, chosen, nreps=nreps, nsteps=nsteps, store_history=True)
    assert np.allclose(res["fpt"]["FirstInactiveTime"], DT_UNIT)
    active = _stack(res["traj"], "Active", nreps)
    assert (active[0] == 1).all()
    assert (active[1:] == 0).all()
    assert (res["traj"][["Y", "X"]].to_numpy() == CELL[0]).all()  # never moves
    assert res["hits"] is None
    assert res["waiting_times"] is None


def test_enclosed_walker_steady_state():
    raster, chosen = _enclosed()
    nreps, nsteps = 4, 6
    res = _chosen_walk(
        raster, chosen, nreps=nreps, nsteps=nsteps, steady_state_chosen_obstacle=True
    )
    hits = res["hits"].drop(columns="Time").to_numpy()
    assert hits.shape == (nsteps + 1, nreps)
    assert (hits[0] == 0).all()
    assert (hits[1:] == 1).all()  # every step is a hit
    assert np.allclose(res["hits"]["Time"], np.arange(nsteps + 1) * DT_UNIT)
    assert len(res["waiting_times"]) == nreps * nsteps
    assert np.allclose(res["waiting_times"], DT_UNIT)
    assert np.allclose(res["fpt"]["FirstInactiveTime"], DT_UNIT)
    assert (res["traj"]["Active"] == 1).all()  # respawned walkers stay active


def test_hits_per_save_interval():
    raster, chosen = _enclosed()
    res = _chosen_walk(
        raster,
        chosen,
        nreps=2,
        nsteps=9,
        save_every=3,
        steady_state_chosen_obstacle=True,
    )
    hits = res["hits"].drop(columns="Time").to_numpy()
    assert np.array_equal(hits, [[0, 0], [3, 3], [3, 3], [3, 3]])


def test_hits_total_equals_waiting_times():
    raster, chosen = _enclosed()
    res = _chosen_walk(
        raster,
        chosen,
        nreps=2,
        nsteps=9,
        save_every=3,
        steady_state_chosen_obstacle=True,
    )
    assert res["hits"].drop(columns="Time").to_numpy().sum() == len(
        res["waiting_times"]
    )


def test_trailing_hits_are_dropped_with_warning():
    raster, chosen = _enclosed()
    with pytest.warns(UserWarning, match="hits of last interval"):
        res = _chosen_walk(
            raster,
            chosen,
            nreps=2,
            nsteps=10,
            save_every=3,
            steady_state_chosen_obstacle=True,
        )
    in_table = res["hits"].drop(columns="Time").to_numpy().sum()
    assert in_table < len(res["waiting_times"])  # the last step (10) is missing
    assert len(res["waiting_times"]) == 2 * 10  # enclosed: every step is a hit


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nsteps": 9, "save_every": 3, "steady_state_chosen_obstacle": True},
        {"nsteps": 10, "save_every": 3, "steady_state_chosen_obstacle": False},
    ],
)
def test_no_trailing_hits_warning_when_nothing_is_dropped(kwargs):
    raster, chosen = _enclosed()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _chosen_walk(raster, chosen, nreps=2, **kwargs)


def test_only_chosen_obstacles_count_as_hits():
    # Three blocked non-chosen sides, one chosen side: p(hit per step) = 1/4,
    # so the first hitting time is geometric with mean 4 steps.
    raster = np.zeros((11, 11), dtype=np.uint8)
    for cell in [(4, 5), (6, 5), (5, 4), (5, 6)]:
        raster[cell] = 1
    chosen = np.zeros_like(raster)
    chosen[5, 6] = 1
    res = _chosen_walk(raster, chosen, nreps=4000, nsteps=200)
    steps = res["fpt"]["FirstInactiveTime"].to_numpy() / DT_UNIT
    assert not np.isnan(steps).any()
    assert steps.min() >= 1
    assert steps.mean() == pytest.approx(4.0, abs=0.3)


def test_nonchosen_obstacle_blocks_without_hit():
    raster, _ = _enclosed()
    chosen = np.zeros_like(raster)  # nothing is chosen
    res = _chosen_walk(raster, chosen, nreps=5, nsteps=20, store_history=True)
    assert res["fpt"]["FirstInactiveTime"].isna().all()
    assert (res["traj"]["Active"] == 1).all()
    assert (res["traj"][["Y", "X"]].to_numpy() == CELL[0]).all()  # blocked in


def test_first_waiting_time_equals_fpt():
    raster = np.zeros((11, 11), dtype=np.uint8)
    for cell in [(4, 5), (6, 5), (5, 4), (5, 6)]:
        raster[cell] = 1
    chosen = np.zeros_like(raster)
    chosen[5, 6] = 1
    nsteps = 200
    res = _chosen_walk(
        raster, chosen, nreps=1, nsteps=nsteps, steady_state_chosen_obstacle=True
    )
    waiting = res["waiting_times"]
    assert waiting[0] == pytest.approx(res["fpt"]["FirstInactiveTime"].iloc[0])
    assert (waiting >= DT_UNIT * (1 - 1e-6)).all()
    assert waiting.sum() <= nsteps * DT_UNIT * (1 + 1e-4)  # rest is the open wait
    assert res["hits"].drop(columns="Time").to_numpy().sum() == len(waiting)


def test_steady_state_respawn_returns_to_start():
    # 1x3 corridor in a block of chosen obstacles; start at its left end. Leaving
    # the corridor is a hit and respawns at the start, so the only non-unit jump
    # (right end -> left end) must land exactly on the start cell.
    raster = np.ones((7, 7), dtype=np.uint8)
    raster[3, 2:5] = 0
    nreps = 20
    res = bl.random_walk_2d_lattice(
        raster,
        nsteps=60,
        nreps=nreps,
        diff_coefficient=1.0,
        start=np.full((nreps, 2), (3, 2)),
        raster_chosen_obstacles=raster.copy(),
        rng=np.random.default_rng(0),
        store_history=True,
        steady_state_chosen_obstacle=True,
    )
    traj = res["traj"]
    y = _stack(traj, "Y", nreps).astype(int)
    x = _stack(traj, "X", nreps).astype(int)
    assert (raster[y, x] == 0).all()  # never inside the block
    jump = np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0))
    assert jump.max() <= 2
    assert (jump == 2).any()  # respawns from the right end do happen
    landing_y, landing_x = y[1:][jump == 2], x[1:][jump == 2]
    assert (landing_y == 3).all()
    assert (landing_x == 2).all()


# --------------------------------------------------------------------------
# ExperimentLattice with chosen obstacles
# --------------------------------------------------------------------------


def test_chosen_raster_selected_by_area():
    small, large = box(5, 5, 9, 9), box(20, 20, 26, 26)
    exp = bl.ExperimentLattice(
        _config(
            dimensions=(0, 40),
            shapes=[small, large],
            chosen_obstacles_by_area=small.area,
        )
    )
    chosen, raster = exp.raster_chosen_obstacles, exp.raster
    assert chosen.shape == raster.shape
    assert 0 < chosen.sum() < raster.sum()
    assert (chosen <= raster).all()  # same orientation, chosen is a subset
    # World (7, 7) -> index (7, 7) + max_dim (40) in the padded raster.
    assert np.argwhere(chosen).mean(axis=0) == pytest.approx([47, 47], abs=1.5)


def test_chosen_equal_area_selects_all():
    first, second = box(5, 5, 9, 9), box(20, 20, 24, 24)
    exp = bl.ExperimentLattice(
        _config(
            dimensions=(0, 40),
            shapes=[first, second],
            chosen_obstacles_by_area=first.area,
        )
    )
    # Both boxes (world centres 7 and 22, plus the 40 padding) are obstacles ...
    assert exp.raster[47, 47] == 1
    assert exp.raster[62, 62] == 1
    # ... and both are chosen, because the selection is by area only.
    assert np.array_equal(exp.raster_chosen_obstacles, exp.raster)


def test_chosen_requires_shapes():
    with pytest.raises(ValueError, match="empty shapes"):
        bl.ExperimentLattice(_config(chosen_obstacles_by_area=10.0))


@pytest.mark.parametrize("has_ghost", [False, True])
def test_chosen_area_without_match_raises(has_ghost):
    with pytest.raises(ValueError, match="chosen"):
        bl.ExperimentLattice(
            _config(
                shapes=[box(5, 5, 8, 8)],
                chosen_obstacles_by_area=123.0,
                has_ghost=has_ghost,
            )
        )


@pytest.mark.parametrize("has_ghost", [False, True])
def test_chosen_area_with_match_builds_chosen_raster(has_ghost):
    exp = bl.ExperimentLattice(
        _config(
            shapes=[box(5, 5, 8, 8), box(12, 12, 14, 14)],  # areas 9 and 4
            chosen_obstacles_by_area=9.0,
            has_ghost=has_ghost,
        )
    )
    assert exp.raster_chosen_obstacles is not None
    assert exp.raster_chosen_obstacles.sum() > 0


def test_steady_state_without_chosen_obstacles_is_consistent():
    run = bl.ExperimentLattice(_config(steady_state_chosen_obstacle=True)).run()
    assert run.hits is None
    assert run.waiting_times is None


def test_steady_state_with_chosen_obstacles_but_no_hit_gives_empty_results():
    # Only "no chosen obstacles" gives None: a run without any hit has a table and
    # an empty array of waiting times.
    chosen = box(30, 30, 34, 34)
    cfg = _config(
        dimensions=(0, 40),
        shapes=[chosen],
        chosen_obstacles_by_area=chosen.area,
        steady_state_chosen_obstacle=True,
        start=(5, 5),
        replicates=5,
        nsteps=2,
    )
    run = bl.ExperimentLattice(cfg).run()
    assert isinstance(run.hits, pd.DataFrame)
    assert (run.hits.drop(columns="Time") == 0).all().all()
    assert isinstance(run.waiting_times, np.ndarray)
    assert run.waiting_times.size == 0


def test_run_structure_steady_vs_terminating():
    nreps, nsteps, every = 6, 40, 4
    steady = bl.ExperimentLattice(
        _ring_config(
            replicates=nreps,
            nsteps=nsteps,
            save_every=every,
            steady_state_chosen_obstacle=True,
        )
    ).run()
    assert list(steady.hits.columns) == [*range(nreps), "Time"]
    assert len(steady.hits) == nsteps // every + 1
    assert isinstance(steady.waiting_times, np.ndarray)
    terminating = bl.ExperimentLattice(
        _ring_config(replicates=nreps, nsteps=nsteps, save_every=every)
    ).run()
    assert terminating.hits is None
    assert terminating.waiting_times is None


def test_walkers_never_enter_obstacles():
    shapes = [box(10, 10, 14, 14), box(25, 5, 28, 12), box(12, 24, 22, 27)]
    cfg = _config(
        dimensions=(0, 60),
        shapes=shapes,
        chosen_obstacles_by_area=shapes[0].area,
        start=(30, 30),
        replicates=50,
        nsteps=200,
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    y, x, _, _ = _ring_arrays(run)
    assert (exp.raster[y, x] == 0).all()


def test_ring_scenario_terminating(ring_terminating):
    steps = ring_terminating.fpt["FirstInactiveTime"].to_numpy() * 12  # dt = 1 / 12
    assert np.isnan(steps).mean() <= 0.01  # almost every walker hits the ring
    assert 20 < np.nanmean(steps) < 60  # exit from a disc: E[T] is about R^2


def test_terminating_walker_frozen_after_hit(ring_terminating):
    y, x, active, msd = _ring_arrays(ring_terminating)
    hit_step = np.rint(ring_terminating.fpt["FirstInactiveTime"].to_numpy() * 12)
    time = np.arange(y.shape[0])
    for walker, step in enumerate(hit_step):
        if np.isnan(step):
            assert (active[:, walker] == 1).all()
            continue
        after = time >= step
        assert (active[~after, walker] == 1).all()
        assert (active[after, walker] == 0).all()
        assert (y[after, walker] == y[int(step), walker]).all()
        assert (x[after, walker] == x[int(step), walker]).all()
        assert (msd[after, walker] == msd[int(step), walker]).all()


def test_active_matches_fpt(ring_terminating):
    hit_step = np.rint(ring_terminating.fpt["FirstInactiveTime"].to_numpy() * 12)
    time = np.arange(301)[:, None]
    survival = (np.isnan(hit_step)[None, :] | (hit_step[None, :] > time)).mean(axis=1)
    mean_active = ring_terminating.mean_trajectories()["Active"].to_numpy()
    assert np.allclose(mean_active, survival, atol=1e-6)


def test_ring_scenario_steady_state():
    run = bl.ExperimentLattice(
        _ring_config(steady_state_chosen_obstacle=True, save_every=10)
    ).run()
    hits = run.hits.drop(columns="Time").to_numpy()
    assert hits.sum() > 100  # many respawn cycles within 300 steps
    assert run.fpt["FirstInactiveTime"].notna().all()  # first hit still recorded
    assert (run.mean_trajectories()["Active"] == 1).all()
    assert hits.sum() == len(run.waiting_times)  # nsteps is a multiple of save_every


# --------------------------------------------------------------------------
# get_mean_fpt
# --------------------------------------------------------------------------


def test_get_mean_fpt_matches_numpy(ring_terminating):
    fpt = ring_terminating.fpt["FirstInactiveTime"].dropna().to_numpy()
    summary = ring_terminating.get_mean_fpt()
    assert list(summary.index) == [
        "Mean",
        "Std",
        "Sem",
        "Ci_low",
        "Ci_high",
        "Median",
        "IQR",
    ]
    assert summary["Mean"] == pytest.approx(fpt.mean())
    assert summary["Std"] == pytest.approx(fpt.std(ddof=1))
    assert summary["Median"] == pytest.approx(np.median(fpt))
    q25, q75 = np.percentile(fpt, [25, 75])
    assert summary["IQR"] == pytest.approx(q75 - q25)
    assert summary["Ci_low"] < summary["Mean"] < summary["Ci_high"]


def test_get_mean_fpt_sem_uses_number_of_hits(ring_terminating):
    run = ring_terminating
    original = run.fpt
    try:
        run.fpt = pd.DataFrame(
            {
                "Replicate": range(5),
                "FirstInactiveTime": [1.0, 2.0, 3.0, np.nan, np.nan],
            }
        )
        expected = np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3)
        assert run.get_mean_fpt()["Sem"] == pytest.approx(expected)
    finally:
        run.fpt = original


def test_get_mean_fpt_no_hits_returns_nan(ring_terminating):
    run = ring_terminating
    original = run.fpt
    try:
        run.fpt = pd.DataFrame(
            {"Replicate": range(3), "FirstInactiveTime": [np.nan] * 3}
        )
        assert run.get_mean_fpt().isna().all()
    finally:
        run.fpt = original


# ==========================================================================
# Section 4: many obstacles and finite particle radius
# ==========================================================================


def _obstacle_field(seed, count, extent, rmin, rmax):
    rng = np.random.default_rng(seed)
    return [
        Point(rng.uniform(0, extent), rng.uniform(0, extent)).buffer(
            rng.uniform(rmin, rmax)
        )
        for _ in range(count)
    ]


def _first_positions(run):
    y, x, _, _ = _ring_arrays(run)
    return y, x


@pytest.mark.parametrize("radius", [1, 2, 5])
def test_make_lattice_particle_shape_and_symmetry(radius):
    kernel = bl._make_lattice_particle(radius)
    side = 2 * radius + 1
    assert kernel.shape == (side, side)
    assert kernel.dtype == bool
    assert kernel[radius, radius]
    assert np.array_equal(kernel, kernel[::-1, :])
    assert np.array_equal(kernel, kernel[:, ::-1])
    assert np.array_equal(kernel, kernel.T)
    assert not kernel[0, 0]  # corners are outside the disc
    assert {1: 5, 2: 13}.get(radius, kernel.sum()) == kernel.sum()


def test_make_lattice_particle_matches_distance():
    radius = 5
    kernel = bl._make_lattice_particle(radius)
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            assert kernel[i + radius, j + radius] == (i * i + j * j <= radius**2)


def test_check_local_free2_lookup():
    free = np.array([[1, 0], [0, 1]], dtype=bool)
    positions = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert np.array_equal(
        bl.check_local_free2(free, positions), [True, False, True, False]
    )
    as_uint8 = free.astype(np.uint8)
    assert np.array_equal(
        bl.check_local_free2(as_uint8, positions), [True, False, True, False]
    )


def test_check_local_free2_none_raises():
    with pytest.raises(ValueError, match="None"):
        bl.check_local_free2(None, np.array([[0, 0]]))


def test_free_centers_match_bruteforce_erosion():
    exp = bl.ExperimentLattice(_config(shapes=[box(8, 8, 12, 12)], particle_radius=2))
    raster = exp.raster.astype(bool)
    height, width = raster.shape
    offsets = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if i * i + j * j <= 4]
    expected = np.zeros_like(raster)
    for row in range(height):
        for col in range(width):
            expected[row, col] = all(
                0 <= row + i < height
                and 0 <= col + j < width
                and not raster[row + i, col + j]
                for i, j in offsets
            )
    assert np.array_equal(exp.free_centers, expected)


def test_free_centers_only_with_radius():
    assert bl.ExperimentLattice(_config()).free_centers is None
    assert bl.ExperimentLattice(_config(particle_radius=0)).free_centers is None
    exp = bl.ExperimentLattice(_config(shapes=[box(8, 8, 12, 12)], particle_radius=2))
    free = exp.raster == 0
    assert exp.free_centers.shape == exp.raster.shape
    assert exp.free_centers.dtype == bool
    assert (exp.free_centers <= free).all()  # centres are a subset of free cells
    assert exp.free_centers.sum() < free.sum()  # and strictly fewer near obstacles


def test_particle_radius_is_in_lattice_cells():
    # At resolution 2 the radius is still counted in cells (2 cells = 4 world units).
    exp = bl.ExperimentLattice(
        _config(shapes=[box(10, 10, 14, 14)], particle_radius=2, lattice_resolution=2)
    )
    assert exp.kernel.shape == (5, 5)
    obstacle = np.argwhere(exp.raster)
    first_row = obstacle[:, 0].min()
    col = int(obstacle[:, 1].mean())
    assert not exp.free_centers[first_row - 2, col]  # 2 cells away: disc touches
    assert exp.free_centers[first_row - 3, col]  # 3 cells away: free


def test_spawn_random_start_respects_free_centers():
    raster = np.zeros((30, 30), dtype=np.uint8)
    raster[12:18, 12:18] = 1  # obstacle inside the central region (10..19)
    kernel = bl._make_lattice_particle(2)
    free_centers = binary_erosion(~raster.astype(bool), kernel, border_value=0)
    out = bl.spawn(
        raster,
        300,
        (0, 10),
        2,
        (0, 0),
        None,
        free_centers,
        np.random.default_rng(0),
        random_start=True,
        has_ghost=True,
    )
    assert (out >= 10).all()
    assert (out < 20).all()
    assert free_centers[out[:, 0], out[:, 1]].all()
    assert (distance_transform_edt(raster == 0)[out[:, 0], out[:, 1]] > 2).all()


def test_spawn_random_start_uniform_without_obstacles():
    raster = np.zeros((30, 30), dtype=np.uint8)
    out = bl.spawn(
        raster,
        4000,
        (0, 10),
        None,
        (0, 0),
        rng=np.random.default_rng(0),
        random_start=True,
        has_ghost=True,
    )
    assert out.mean(axis=0) == pytest.approx([14.5, 14.5], abs=0.25)
    counts = np.bincount(out[:, 0] - 10, minlength=10)
    assert counts.min() > 300
    assert counts.max() < 500  # 400 expected per row


def test_spawn_all_blocked_raises_runtime_error():
    raster = np.ones((30, 30), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        bl.spawn(
            raster,
            3,
            (0, 10),
            None,
            (0, 0),
            rng=np.random.default_rng(0),
            random_start=True,
            has_ghost=True,
        )


def test_spawn_erosion_leaves_no_center_raises():
    raster = np.zeros((30, 30), dtype=np.uint8)  # free cells exist ...
    no_centers = np.zeros((30, 30), dtype=bool)  # ... but no particle fits
    with pytest.raises(RuntimeError):
        bl.spawn(
            raster,
            3,
            (0, 10),
            2,
            (0, 0),
            None,
            no_centers,
            np.random.default_rng(0),
            random_start=True,
            has_ghost=True,
        )


def test_run_random_starts_valid_and_distinct():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(8, 8, 12, 12), box(25, 20, 30, 30)],
        particle_radius=2,
        random_start=True,
        replicates=100,
        nsteps=5,
    )
    exp = bl.ExperimentLattice(cfg)
    y, x = _first_positions(exp.run())
    assert (y[0] >= 40).all()
    assert (y[0] < 80).all()
    assert (x[0] >= 40).all()
    assert (x[0] < 80).all()
    assert exp.free_centers[y[0], x[0]].all()
    assert len(set(zip(y[0], x[0], strict=True))) > 20  # random, not one fixed start


@pytest.mark.parametrize("radius", [None, 3])
def test_walkers_keep_distance_to_obstacles(radius):
    cfg = _config(
        dimensions=(0, 60),
        shapes=_obstacle_field(4, 12, 60, 3, 6),
        particle_radius=radius,
        random_start=True,
        replicates=50,
        nsteps=100,
    )
    exp = bl.ExperimentLattice(cfg)
    y, x = _first_positions(exp.run())
    distance = distance_transform_edt(exp.raster == 0)[y, x]
    # Disc of radius r inside free space <=> nearest obstacle cell farther than r.
    assert (distance > (radius or 0)).all()


def test_obstacles_reduce_msd():
    def final_msd(shapes, radius):
        cfg = _config(
            dimensions=(0, 60),
            shapes=shapes,
            particle_radius=radius,
            random_start=True,
            replicates=300,
            nsteps=200,
            rng=np.random.default_rng(1),
        )
        df = bl.ExperimentLattice(cfg).run().mean_trajectories()
        return df["MSD"].iloc[-20:].mean()

    obstacles = _obstacle_field(10, 15, 60, 3, 5)
    free, point, disc = (
        final_msd(None, None),
        final_msd(obstacles, None),
        final_msd(obstacles, 3),
    )
    assert point < 0.95 * free  # obstacles slow diffusion down
    assert disc < 0.95 * point  # a finite particle is slowed down further


def test_run_shapes_in_raster_frame():
    original = box(8, 8, 12, 12)
    cfg = _config(shapes=[original])
    run = bl.ExperimentLattice(cfg).run()
    assert run.shapes[0].bounds == (28.0, 28.0, 32.0, 32.0)  # shifted by max_dim
    assert cfg.shapes[0].bounds == (8.0, 8.0, 12.0, 12.0)  # user's config untouched


@pytest.mark.parametrize(
    ("shapes", "covered"),
    [
        ([box(5, 5, 9, 9)], 16),
        ([box(-5, -5, 5, 5)], 25),  # clipped at the domain edge
        ([box(0, 0, 10, 10), box(5, 5, 15, 15)], 175),  # overlap counted once
    ],
)
def test_experiment_coverage_value(shapes, covered):
    # the start must not lie inside a shape
    exp = bl.ExperimentLattice(_config(shapes=shapes, start=(18, 18)))
    assert exp.coverage == pytest.approx(covered / 400)


def test_coverage_dimension_full_cover():
    exp = bl.ExperimentLattice(
        _config(shapes=[box(5, 5, 15, 15)], coverage_dimension=(5, 15))
    )
    assert exp.coverage == pytest.approx(1.0)


def test_coverage_dimension_half_cover():
    # the window (5, 15) has the area 100, the shape covers its left half
    exp = bl.ExperimentLattice(
        _config(shapes=[box(5, 5, 10, 15)], coverage_dimension=(5, 15))
    )
    assert exp.coverage == pytest.approx(0.5)


def test_plot_run_shows_obstacles_in_world_coordinates():
    mpl.use("Agg")
    run = bl.ExperimentLattice(
        _config(shapes=[box(8, 8, 12, 12)], replicates=2, nsteps=10)
    ).run()
    ax = run.plot_run()
    image = np.ma.getdata(ax.images[0].get_array())
    assert image.shape == (20, 20)  # central tile only
    assert np.array_equal(image, run.raster[20:40, 20:40])
    assert image[10, 10] == 1  # obstacle at world (10, 10)
    assert image[2, 2] == 0
    data = run.trajectories[0].data
    assert np.allclose(ax.lines[0].get_xdata(), data["X"] - 20)
    assert np.allclose(ax.lines[0].get_ydata(), data["Y"] - 20)
    mpl.pyplot.close("all")


def test_scaled_tutorial_scenario():
    cfg = _config(
        dimensions=(0, 100),
        shapes=_obstacle_field(7, 30, 100, 2, 8),
        particle_radius=2,
        random_start=True,
        replicates=200,
        nsteps=100,
        rng=np.random.default_rng(2),
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    y, x = _first_positions(run)
    assert (distance_transform_edt(exp.raster == 0)[y, x] > 2).all()
    _, d_est = run.get_mean_diff_coefficients()
    assert 0.3 * DIFF < d_est < DIFF  # slowed by obstacles, but not blocked


# ==========================================================================
# Section 5: obstacles together with an aim area
# ==========================================================================


def _obstacle_aim_config(resolution=1, **overrides):
    """Same geometry in lattice cells for every resolution (60 x 60 cells)."""
    res = resolution
    params = {
        "dimensions": (0, 60 * res),
        "shapes": _obstacle_field(7, 12, 60 * res, 3 * res, 6 * res),
        "aim_area": Point(30 * res, 30 * res).buffer(15 * res),
        "lattice_resolution": res,
        "particle_radius": 2,
        "random_start": True,
        "replicates": 1500,
        "nsteps": 100,
        "start": (0, 0),
        "rng": np.random.default_rng(1),
    }
    params.update(overrides)
    return _config(**params)


def _expected_aim_fraction(exp):
    """Share of valid start centres (central tile) that lie inside the aim mask."""
    max_dim = int(exp._config.dimensions[1])
    tile = (slice(max_dim, 2 * max_dim), slice(max_dim, 2 * max_dim))
    valid = exp.raster == 0
    if exp.free_centers is not None:
        valid &= exp.free_centers
    return (valid & exp.aim_mask)[tile].sum() / valid[tile].sum()


@pytest.mark.parametrize("resolution", [1, 2])
def test_random_start_aim_fraction_matches_free_centers(resolution):
    exp = bl.ExperimentLattice(_obstacle_aim_config(resolution))
    assert (exp.aim_mask & (exp.raster == 1)).any()  # the aim overlaps obstacles
    expected = _expected_aim_fraction(exp)
    aim0 = exp.run().mean_trajectories()["Aim"].iloc[0]
    sigma = np.sqrt(expected * (1 - expected) / 1500)
    assert abs(aim0 - expected) < 4 * sigma


@pytest.mark.parametrize("radius", [None, 2])
@pytest.mark.parametrize("resolution", [1, 2])
def test_aim_fraction_stationary_with_uniform_start(resolution, radius):
    # Moves between free centres are symmetric, so a uniform start stays uniform
    # and the fraction of walkers inside the aim area must not drift.
    cfg = _obstacle_aim_config(resolution, particle_radius=radius)
    aim = bl.ExperimentLattice(cfg).run().mean_trajectories()["Aim"].to_numpy()
    assert np.abs(aim - aim[0]).max() < 0.03


def test_plot_run_aim_points_match_polygon_in_world_coordinates():
    mpl.use("Agg")
    cfg = _obstacle_aim_config(replicates=50, nsteps=60)
    run = bl.ExperimentLattice(cfg).run()
    ax = run.plot_run(aim_area=True)
    assert len(ax.collections) == 100
    for inside, outside in zip(ax.collections[0::2], ax.collections[1::2], strict=True):
        pts_in = np.ma.getdata(inside.get_offsets())
        pts_out = np.ma.getdata(outside.get_offsets())
        assert shapely.contains_xy(cfg.aim_area, pts_in[:, 0], pts_in[:, 1]).all()
        assert not shapely.contains_xy(cfg.aim_area, pts_out[:, 0], pts_out[:, 1]).any()
    mpl.pyplot.close("all")


def test_aim_covering_all_space_is_always_one():
    cfg = _obstacle_aim_config(
        aim_area=box(-59.5, -59.5, 119.5, 119.5), replicates=100, nsteps=60
    )
    aim = bl.ExperimentLattice(cfg).run().mean_trajectories()["Aim"]
    assert (aim == 1.0).all()


def test_aim_inside_obstacle_is_never_visited():
    cfg = _obstacle_aim_config(
        shapes=[box(20, 20, 40, 40)],
        aim_area=Point(30, 30).buffer(5),
        replicates=200,
        nsteps=60,
    )
    exp = bl.ExperimentLattice(cfg)
    assert exp.aim_mask.sum() > 0  # the mask exists ...
    assert (exp.raster[exp.aim_mask] == 1).all()  # ... but only on obstacle cells
    assert (exp.run().mean_trajectories()["Aim"] == 0.0).all()


def test_aim_column_matches_mask_at_positions_with_obstacles():
    exp = bl.ExperimentLattice(_obstacle_aim_config(replicates=100, nsteps=60))
    run = exp.run()
    y, x = _first_positions(run)
    aim = np.stack([t.data["Aim"].to_numpy() for t in run.trajectories], axis=1)
    assert np.array_equal(aim, exp.aim_mask[y, x].astype(int))


def test_scaled_tutorial_scenario_obstacles_and_aim():
    cfg = _obstacle_aim_config(
        dimensions=(0, 100),
        shapes=_obstacle_field(7, 30, 100, 2, 8),
        aim_area=Point(50, 50).buffer(20),
        replicates=200,
        rng=np.random.default_rng(2),
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    df = run.mean_trajectories()
    y, x = _first_positions(run)
    assert (distance_transform_edt(exp.raster == 0)[y, x] > 2).all()
    assert df["Aim"].between(0, 1).all()
    assert (df["Aim_ci_low"] <= df["Aim"] + 1e-6).all()
    assert (df["Aim"] <= df["Aim_ci_high"] + 1e-6).all()
    expected = _expected_aim_fraction(exp)
    assert abs(df["Aim"].iloc[0] - expected) < 4 * np.sqrt(
        expected * (1 - expected) / 200
    )
    _, d_est = run.get_mean_diff_coefficients()
    assert 0.3 * DIFF < d_est < DIFF


# ==========================================================================
# Section 6: ensembles of experiments
# ==========================================================================


class _InlineExecutor:
    """Runs submitted jobs immediately in this process (no worker processes)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - mirror a real executor
            future.set_exception(exc)
        return future


def _in_submission_order(futures):
    return iter(list(futures))


def _in_reverse_order(futures):
    return iter(list(futures)[::-1])


@pytest.fixture
def inline_executor(monkeypatch):
    monkeypatch.setattr(bl, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(bl, "as_completed", _in_submission_order)


def _write_files(directory, shape_lists):
    paths = []
    for i, shapes in enumerate(shape_lists):
        path = directory / f"shapes_{i}.wkt"
        path.write_text("".join(to_wkt(p) + "\n" for p in shapes))
        paths.append(path)
    return paths


def _ensemble_config(**overrides):
    params = {
        "dimensions": (0, 40),
        "start": (20, 20),
        "replicates": 100,
        "nsteps": 60,
        "rng": None,
        "workers": 2,
    }
    params.update(overrides)
    return _config(**params)


def _run_inline(files, cfg):
    """Run an ensemble without worker processes, in submission order."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bl, "ProcessPoolExecutor", _InlineExecutor)
        patch.setattr(bl, "as_completed", _in_submission_order)
        return bl.EnsembleExperimentLattice(files, cfg).run()


@pytest.fixture(scope="module")
def free_ensemble(tmp_path_factory):
    # Four members with one tiny, far away obstacle each: almost free diffusion.
    directory = tmp_path_factory.mktemp("free")
    files = _write_files(directory, [[box(2 + i, 2, 4 + i, 4)] for i in range(4)])
    return _run_inline(files, _ensemble_config(store_history=False))


@pytest.fixture(scope="module")
def hits_ensemble(tmp_path_factory):
    # Four members, each with one chosen 4 x 4 box close to the start.
    directory = tmp_path_factory.mktemp("hits")
    files = _write_files(directory, [[box(26 + i, 18, 30 + i, 22)] for i in range(4)])
    cfg = _ensemble_config(
        nsteps=200,
        save_every=10,
        chosen_obstacles_by_area=16.0,
        steady_state_chosen_obstacle=True,
    )
    return _run_inline(files, cfg)


# --------------------------------------------------------------------------
# EnsembleExperimentLattice.run
# --------------------------------------------------------------------------


def test_ensemble_runs_each_file_with_own_geometry(inline_executor, tmp_path):
    boxes = [box(5, 5, 7, 7), box(5, 5, 8, 8), box(5, 5, 9, 9)]  # areas 4, 9, 16
    files = _write_files(tmp_path, [[b] for b in boxes])
    result = bl.EnsembleExperimentLattice(files, _ensemble_config(nsteps=5)).run()
    assert result.coverages == pytest.approx([4 / 1600, 9 / 1600, 16 / 1600])
    assert [run.shapes[0].area for run in result.runs] == pytest.approx([4, 9, 16])
    assert len(result.runs[0].trajectories) == 100


def test_ensemble_results_follow_file_order(monkeypatch, tmp_path):
    monkeypatch.setattr(bl, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(bl, "as_completed", _in_reverse_order)  # last finishes first
    boxes = [box(5, 5, 7, 7), box(5, 5, 8, 8), box(5, 5, 9, 9)]
    files = _write_files(tmp_path, [[b] for b in boxes])
    result = bl.EnsembleExperimentLattice(files, _ensemble_config(nsteps=5)).run()
    assert result.coverages == pytest.approx([4 / 1600, 9 / 1600, 16 / 1600])
    assert [run.shapes[0].area for run in result.runs] == pytest.approx([4, 9, 16])


def test_ensemble_members_are_independent_with_seed(inline_executor, tmp_path):
    files = _write_files(tmp_path, [[box(5, 5, 9, 9)], [box(5, 5, 9, 9)]])
    cfg = _ensemble_config(rng=np.random.default_rng(5))
    result = bl.EnsembleExperimentLattice(files, cfg).run()
    first, second = (r.mean_trajectories()["MSD"].to_numpy() for r in result.runs)
    assert not np.array_equal(first, second)


def _member_msd(result):
    return np.array([r.mean_trajectories()["MSD"].to_numpy() for r in result.runs])


def test_ensemble_seeded_is_reproducible(inline_executor, tmp_path):
    files = _write_files(tmp_path, [[box(5, 5, 9, 9)], [box(6, 6, 9, 9)]])
    cfg = _ensemble_config(rng=np.random.default_rng(5))
    first = _member_msd(bl.EnsembleExperimentLattice(files, cfg).run())
    same_config = _member_msd(bl.EnsembleExperimentLattice(files, cfg).run())
    new_config = _member_msd(
        bl.EnsembleExperimentLattice(
            files, _ensemble_config(rng=np.random.default_rng(5))
        ).run()
    )
    assert np.array_equal(first, same_config)  # the user's rng is not advanced
    assert np.array_equal(first, new_config)
    other_seed = _member_msd(
        bl.EnsembleExperimentLattice(
            files, _ensemble_config(rng=np.random.default_rng(6))
        ).run()
    )
    assert not np.array_equal(first, other_seed)


def test_ensemble_seeded_independent_of_completion_order(
    inline_executor, monkeypatch, tmp_path
):
    files = _write_files(
        tmp_path, [[box(5, 5, 9, 9)], [box(6, 6, 9, 9)], [box(7, 7, 9, 9)]]
    )
    cfg = _ensemble_config(rng=np.random.default_rng(5))
    in_order = _member_msd(bl.EnsembleExperimentLattice(files, cfg).run())
    monkeypatch.setattr(bl, "as_completed", _in_reverse_order)
    reverse = _member_msd(bl.EnsembleExperimentLattice(files, cfg).run())
    assert np.array_equal(in_order, reverse)


def test_ensemble_unseeded_members_use_fresh_streams(inline_executor, tmp_path):
    files = _write_files(tmp_path, [[box(5, 5, 9, 9)], [box(5, 5, 9, 9)]])
    result = bl.EnsembleExperimentLattice(files, _ensemble_config(rng=None)).run()
    first, second = _member_msd(result)
    assert not np.array_equal(first, second)


def test_ensemble_does_not_mutate_config(inline_executor, tmp_path):
    files = _write_files(tmp_path, [[box(5, 5, 9, 9)], [box(6, 6, 9, 9)]])
    cfg = _ensemble_config(rng=np.random.default_rng(5), nsteps=5)
    state = cfg.rng.bit_generator.state
    bl.EnsembleExperimentLattice(files, cfg).run()
    assert cfg.shapes is None
    assert cfg.rng.bit_generator.state == state


def test_ensemble_accepts_str_and_path(inline_executor, tmp_path):
    files = _write_files(tmp_path, [[box(5, 5, 9, 9)], [box(6, 6, 9, 9)]])
    result = bl.EnsembleExperimentLattice(
        [str(files[0]), files[1]], _ensemble_config(nsteps=5)
    ).run()
    assert len(result.runs) == 2


def test_scaled_tutorial_scenario_ensemble(tmp_path):
    # The only test that starts real worker processes ("spawn").
    shape_lists = [_obstacle_field(20 + i, 20, 100, 2, 8) for i in range(6)]
    files = _write_files(tmp_path, shape_lists)
    cfg = _ensemble_config(
        dimensions=(0, 100),
        start=(0, 0),
        particle_radius=2,
        random_start=True,
        replicates=100,
        nsteps=100,
        workers=2,
    )
    result = bl.EnsembleExperimentLattice(files, cfg).run()
    expected = [
        bl.ExperimentLattice(dataclasses.replace(cfg, shapes=shapes)).coverage
        for shapes in shape_lists
    ]
    assert len(result.runs) == 6
    # Order-insensitive: the completion order of the workers is not defined.
    assert sorted(result.coverages) == pytest.approx(sorted(expected))
    assert result.get_mean_coverage() == pytest.approx(np.mean(expected))
    assert all(len(run.trajectories) == 100 for run in result.runs)
    assert len(result.mean_over_runs()) == 101
    _, d_est = result.get_mean_diff_coefficients()
    assert 0.3 * DIFF < d_est < DIFF


# --------------------------------------------------------------------------
# EnsembleExperimentLatticeRun statistics
# --------------------------------------------------------------------------


def test_mean_over_runs_columns_and_mean(free_ensemble):
    df = free_ensemble.mean_over_runs()
    expected = ["Time"]
    for m in ("MSD", "Active", "MAD"):
        expected += [m, f"{m}_std", f"{m}_sem", f"{m}_ci_low", f"{m}_ci_high"]
    assert set(df.columns) == set(expected)
    assert len(df) == 61
    manual = np.mean(
        [r.mean_trajectories()["MSD"].to_numpy() for r in free_ensemble.runs], axis=0
    )
    assert np.allclose(df["MSD"], manual, rtol=1e-5)


def test_mean_over_runs_std_sem_ci_formulas(free_ensemble):
    df = free_ensemble.mean_over_runs()
    n_runs = len(free_ensemble.runs)
    for m in ("MSD", "MAD", "Active"):
        wide = pd.DataFrame(
            [r.mean_trajectories()[m].to_numpy(dtype=float) for r in free_ensemble.runs]
        )
        mean = wide.mean(axis=0).to_numpy()
        std = wide.std(axis=0, ddof=1).to_numpy()
        sem = std / np.sqrt(n_runs)
        assert np.allclose(df[f"{m}_std"], std, rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_sem"], sem, rtol=1e-4, atol=1e-5)
        assert np.allclose(df[f"{m}_ci_low"], mean - 1.96 * sem, rtol=1e-4, atol=1e-4)
        assert np.allclose(df[f"{m}_ci_high"], mean + 1.96 * sem, rtol=1e-4, atol=1e-4)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mean_over_runs_single_run_gives_nan_std(free_ensemble):
    single = bl.EnsembleExperimentLatticeRun(
        free_ensemble.runs[:1], free_ensemble.coverages[:1]
    )
    df = single.mean_over_runs()
    assert df["MSD_std"].isna().all()  # ddof=1 with one run


def test_ensemble_diff_coefficient_recovers_D(free_ensemble):
    _, d_est = free_ensemble.get_mean_diff_coefficients()
    assert 0.75 * DIFF < d_est < 1.2 * DIFF  # 400 unseeded walkers: about 5 % noise
    _, _, alpha = free_ensemble.get_mean_diff_coefficients_pw()
    assert alpha == pytest.approx(1.0, abs=0.1)


def test_ensemble_diff_coefficient_dist(free_ensemble):
    dist = free_ensemble.get_mean_diff_coefficients_dist()
    mean_df = free_ensemble.mean_over_runs()
    assert list(dist.columns) == [
        "Time",
        "MSD",
        "D_dist",
        "D_dist_std",
        "D_dist_sem",
        "D_dist_ci_low",
        "D_dist_ci_high",
    ]
    assert np.isnan(dist["D_dist"].iloc[0])  # 0 / 0 at t = 0
    assert np.allclose(
        dist["D_dist"].iloc[1:], mean_df["MSD"].iloc[1:] / mean_df["Time"].iloc[1:]
    )


def test_ensemble_get_mean_coverage(free_ensemble):
    expected = [4 / 1600] * 4  # 2 x 2 boxes in a 40 x 40 domain
    assert free_ensemble.coverages == pytest.approx(expected)
    assert free_ensemble.get_mean_coverage() == pytest.approx(4 / 1600)


def test_ensemble_get_mean_fpt_matches_numpy(hits_ensemble):
    means = np.array([r.get_mean_fpt()["Mean"] for r in hits_ensemble.runs])
    summary = hits_ensemble.get_mean_fpt()
    std = means.std(ddof=1)
    sem = std / np.sqrt(len(means))  # one value per run
    assert summary["Mean"] == pytest.approx(means.mean())
    assert summary["Std"] == pytest.approx(std)
    assert summary["Sem"] == pytest.approx(sem)
    assert summary["Ci_low"] == pytest.approx(means.mean() - 1.96 * sem)
    assert summary["Ci_high"] == pytest.approx(means.mean() + 1.96 * sem)
    assert summary["Median"] == pytest.approx(np.median(means))
    q25, q75 = np.percentile(means, [25, 75])
    assert summary["IQR"] == pytest.approx(q75 - q25)


def test_ensemble_get_mean_fpt_member_without_hits_gives_nan(hits_ensemble):
    no_hits = dataclasses.replace(
        hits_ensemble.runs[0],
        fpt=pd.DataFrame({"Replicate": range(3), "FirstInactiveTime": [np.nan] * 3}),
    )
    runs = [no_hits, *hits_ensemble.runs[1:]]
    ensemble = bl.EnsembleExperimentLatticeRun(runs, hits_ensemble.coverages)
    summary = ensemble.get_mean_fpt()
    assert summary.isna().all()  # the NaN of one member propagates


def test_ensemble_get_mean_hits_stats(hits_ensemble):
    df = hits_ensemble.get_mean_hits()
    assert list(df.columns) == [
        "Time",
        "Hits",
        "Hits_std",
        "Hits_sem",
        "Hits_ci_low",
        "Hits_ci_high",
    ]
    assert len(df) == 21  # nsteps / save_every + 1
    # Hits per save interval, summed over the replicates of each run.
    per_run = np.array(
        [r.hits.drop(columns="Time").sum(axis=1).to_numpy() for r in hits_ensemble.runs]
    )
    sem = per_run.std(axis=0, ddof=1) / np.sqrt(len(per_run))
    assert df["Hits"].sum() > 0
    assert np.allclose(df["Hits"], per_run.mean(axis=0))
    assert np.allclose(df["Hits_std"], per_run.std(axis=0, ddof=1))
    assert np.allclose(df["Hits_sem"], sem)
    assert np.allclose(df["Hits_ci_low"], per_run.mean(axis=0) - 1.96 * sem)
    assert np.allclose(df["Hits_ci_high"], per_run.mean(axis=0) + 1.96 * sem)


def test_ensemble_get_mean_hits_requires_all_runs(hits_ensemble):
    runs = list(hits_ensemble.runs)
    runs[0] = dataclasses.replace(runs[0], hits=None)
    broken = bl.EnsembleExperimentLatticeRun(runs, hits_ensemble.coverages)
    with pytest.raises(ValueError, match="hits"):
        broken.get_mean_hits()


# ==========================================================================
# Core diffusion A: physics validation
# ==========================================================================


def _free_walk(nreps, nsteps, seed=0):
    """Positions (time, replicate) of unobstructed walkers, far from the edges."""
    traj = _walk(
        nreps=nreps,
        nsteps=nsteps,
        size=2 * nsteps + 21,
        seed=seed,
        store_history=True,
    )["traj"]
    return (
        _stack(traj, "Y", nreps).astype(int),
        _stack(traj, "X", nreps).astype(int),
    )


def test_displacement_is_isotropic():
    nreps, nsteps = 4000, 40
    y, x = _free_walk(nreps, nsteps)
    dy, dx = y[-1] - y[0], x[-1] - x[0]
    for displacement in (dy, dx):
        assert abs(displacement.mean()) < 0.4  # no drift (standard error 0.07)
        assert displacement.var() == pytest.approx(nsteps / 2, rel=0.1)
    assert abs(np.corrcoef(dx, dy)[0, 1]) < 0.05


def test_step_direction_frequencies():
    y, x = _free_walk(4000, 40)
    moves = np.stack([np.diff(y, axis=0), np.diff(x, axis=0)], axis=-1).reshape(-1, 2)
    assert (np.abs(moves).sum(axis=1) == 1).all()
    for step in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        fraction = np.mean((moves == step).all(axis=1))
        assert fraction == pytest.approx(0.25, abs=0.006)


def test_replicates_are_independent():
    nreps, nsteps = 4000, 40
    y, x = _free_walk(nreps, nsteps)
    assert abs(np.corrcoef((x[-1] - x[0])[:-1], (x[-1] - x[0])[1:])[0, 1]) < 0.05
    code = (np.diff(y, axis=0) + 1) * 3 + (np.diff(x, axis=0) + 1)
    same_move = np.mean(code[:, :-1] == code[:, 1:])  # neighbouring replicates
    assert same_move == pytest.approx(0.25, abs=0.006)


@pytest.mark.parametrize("diff", [0.5, 3.0, 10.0])
@pytest.mark.parametrize("a", [1, 2])
def test_msd_over_time_slope_is_4D_over_a2(diff, a):
    nsteps = 30
    cfg = _config(
        dimensions=(0, 20 * a),
        lattice_resolution=a,
        diff_coefficient=diff,
        replicates=1000,
        nsteps=nsteps,
        start=(10 * a, 10 * a),
        rng=np.random.default_rng(int(10 * diff) + a),
    )
    df = bl.ExperimentLattice(cfg).run().mean_trajectories()
    assert df["Time"].iloc[-1] == pytest.approx(nsteps * a**2 / (4 * diff), rel=1e-5)
    time, msd = df["Time"].to_numpy(), df["MSD"].to_numpy()
    slope = (time * msd).sum() / (time**2).sum()
    assert slope == pytest.approx(4 * diff / a**2, rel=0.1)


def test_msd_independent_of_resolution():
    # Same world geometry (4 % coverage, edges on even coordinates), resolution 1/2.
    shapes = [
        box(10, 10, 16, 16),
        box(30, 20, 36, 26),
        box(14, 40, 20, 46),
        box(44, 44, 50, 50),
    ]
    estimates = {}
    for resolution in (1, 2):
        cfg = _config(
            dimensions=(0, 60),
            shapes=shapes,
            lattice_resolution=resolution,
            random_start=True,
            replicates=800,
            nsteps=100,
            rng=np.random.default_rng(resolution),
        )
        _, d_est = bl.ExperimentLattice(cfg).run().get_mean_diff_coefficients()
        estimates[resolution] = d_est * resolution**2  # physical units
    assert estimates[1] == pytest.approx(estimates[2], rel=0.15)
    assert 0.75 * DIFF < estimates[1] < 1.05 * DIFF


@pytest.mark.parametrize("diff", [1.0, 2.0])
@pytest.mark.parametrize("x_start", [1, 2, 3])
def test_corridor_hitting_time_exact(x_start, diff):
    # 1D corridor x = 1..5 with absorbing chosen cells at x = 0 and x = 6. Vertical
    # moves are blocked (and still cost a step), so E[T] = 2 x (L - x) steps.
    raster = np.ones((7, 7), dtype=np.uint8)
    raster[3, 1:6] = 0
    chosen = np.zeros_like(raster)
    chosen[3, 0] = chosen[3, 6] = 1
    nreps = 4000
    res = bl.random_walk_2d_lattice(
        raster,
        nsteps=400,
        nreps=nreps,
        diff_coefficient=diff,
        a=1,
        start=np.full((nreps, 2), (3, x_start)),
        raster_chosen_obstacles=chosen,
        rng=np.random.default_rng(x_start),
    )
    fpt = res["fpt"]["FirstInactiveTime"].to_numpy()
    assert not np.isnan(fpt).any()
    steps = fpt * 4 * diff  # FPT = steps * dt with dt = 1 / (4 D)
    assert steps.mean() == pytest.approx(2 * x_start * (6 - x_start), abs=0.8)


def test_diffusion_decreases_with_coverage():
    levels = []
    for count in (0, 18, 36, 60):
        cfg = _config(
            dimensions=(0, 60),
            shapes=_obstacle_field(3, count, 60, 3, 5) if count else None,
            random_start=True,
            replicates=1500,
            nsteps=150,
            rng=np.random.default_rng(count),
        )
        exp = bl.ExperimentLattice(cfg)
        msd = exp.run().mean_trajectories()["MSD"].iloc[-20:].mean()
        levels.append((exp.coverage, msd))
    coverage, msd = zip(*levels, strict=True)
    assert all(a < b for a, b in zip(coverage, coverage[1:], strict=False))
    assert all(b < 0.97 * a for a, b in zip(msd, msd[1:], strict=False))


# ==========================================================================
# Core diffusion C: edge cases and input validation
# ==========================================================================


def test_nsteps_zero_gives_single_row():
    run = bl.ExperimentLattice(_config(nsteps=0)).run()
    assert all(len(t.data) == 1 for t in run.trajectories)
    df = run.mean_trajectories()
    assert len(df) == 1
    assert df["MSD"].iloc[0] == 0


def test_save_every_larger_than_nsteps_saves_only_start():
    run = bl.ExperimentLattice(_config(nsteps=5, save_every=50)).run()
    assert all(len(t.data) == 1 for t in run.trajectories)


def test_save_every_not_dividing_nsteps():
    run = bl.ExperimentLattice(
        _config(nsteps=5, save_every=2, diff_coefficient=1)
    ).run()
    times = run.trajectories[0].data["Time"].to_numpy()
    assert np.allclose(times, [0, 0.5, 1.0])  # steps 0, 2, 4 with dt = 1 / 4


def test_negative_world_start_is_valid():
    exp = bl.ExperimentLattice(_config(start=(-5, -5)))
    first = exp.run().trajectories[0].data.iloc[0]
    assert (first["Y"], first["X"]) == (15, 15)  # -5 + max_dim (20): padding region


def test_single_run_ignores_workers():
    run = bl.ExperimentLattice(_config(workers=0, nsteps=3)).run()
    assert len(run.trajectories) == 10


def test_start_area_and_start_area_by_size_conflict_raises():
    with pytest.raises(ValueError, match="at once"):
        bl.ExperimentLattice(
            _config(
                start_area=box(1, 1, 3, 3),
                start_area_by_size=(1.0, 1.0, box(0, 0, 5, 5)),
                shapes=[box(5, 5, 9, 9)],
            )
        )


@pytest.mark.parametrize(
    ("overrides", "names"),
    [
        pytest.param({"nsteps": -3}, "nsteps", id="nsteps<0"),
        pytest.param({"replicates": 0}, "replicates", id="replicates=0"),
        pytest.param({"save_every": 0}, "save_every", id="save_every=0"),
        pytest.param({"save_every": -1}, "save_every", id="save_every<0"),
        pytest.param(
            {"diff_coefficient": 0},
            "diff_coefficient",
            id="D=0",
        ),
        pytest.param(
            {"diff_coefficient": -1},
            "diff_coefficient",
            id="D<0",
        ),
        pytest.param(
            {"lattice_resolution": 0},
            "lattice_resolution",
            id="resolution=0",
        ),
        pytest.param({"dimensions": (0, 0)}, "dimensions", id="dims=0"),
        pytest.param(
            {"dimensions": (10, 50)},
            "dimensions",
            id="dims-min!=0",
        ),
        pytest.param(
            {"particle_radius": -2},
            "particle_radius",
            id="radius<0",
        ),
        pytest.param({"start": (500, 500)}, "start", id="start-outside"),
        pytest.param(
            {"aim_area": box(500, 500, 510, 510)},
            "aim_area",
            id="aim-outside",
        ),
        pytest.param(
            {"random_start": True, "start_area": box(500, 500, 510, 510)},
            "start_area",
            id="start_area-outside",
        ),
        pytest.param(
            {"shapes": [box(8, 8, 14, 14)], "start": (10, 10)},
            "start",
            id="start-in-obstacle",
        ),
    ],
)
def test_invalid_config_raises_value_error(overrides, names):
    # A clear ValueError that names the offending parameter is expected.
    with pytest.raises(ValueError, match=names):
        bl.ExperimentLattice(_config(**overrides)).run()


@pytest.mark.parametrize("has_ghost", [False, True])
def test_start_outside_raster_raises_in_both_modes(has_ghost):
    with pytest.raises(ValueError, match="start"):
        bl.ExperimentLattice(_config(start=(500, 500), has_ghost=has_ghost))


def test_start_on_obstacle_border_is_accepted():
    # contains() is False on the boundary: only a start strictly inside is rejected
    cfg = _config(shapes=[box(8, 8, 14, 14)], start=(8, 8), nsteps=2)
    assert len(bl.ExperimentLattice(cfg).run().trajectories) == 10


def test_start_inside_obstacle_is_accepted_with_random_start():
    # a random start is drawn from the free cells, so the fixed start is unused
    cfg = _config(
        shapes=[box(8, 8, 14, 14)], start=(10, 10), random_start=True, nsteps=2
    )
    assert len(bl.ExperimentLattice(cfg).run().trajectories) == 10


def test_start_in_padding_region_is_accepted():
    cfg = _config(start=(-5, -5), nsteps=2)
    assert len(bl.ExperimentLattice(cfg).run().trajectories) == 10


def test_particle_radius_larger_than_domain_is_rejected():
    # Radius 40 cells in a domain of 20 cells; the probe radius 500 (out of memory)
    # is not run here.
    with pytest.raises(ValueError, match="particle_radius"):
        bl.ExperimentLattice(_config(particle_radius=40))


def test_particle_radius_that_fits_is_accepted():
    cfg = _config(particle_radius=3, nsteps=2)  # 7 cells in a domain of 20
    assert len(bl.ExperimentLattice(cfg).run().trajectories) == 10


@pytest.mark.parametrize(("radius", "fits"), [(4, True), (5, False)])
def test_particle_radius_is_checked_in_lattice_cells(radius, fits):
    # dimensions 20 at resolution 2 are 10 cells: a radius of 5 needs 11 cells
    cfg = _config(lattice_resolution=2, particle_radius=radius, nsteps=2)
    if fits:
        bl.ExperimentLattice(cfg)
    else:
        with pytest.raises(ValueError, match="particle_radius"):
            bl.ExperimentLattice(cfg)


def test_dimensions_not_multiple_of_resolution_raises():
    # 20 / 3 would be 6.67 cells: rejected instead of leaving the last domain
    # row and column out of the random starts
    with pytest.raises(ValueError, match="dimensions"):
        bl.ExperimentLattice(
            _config(lattice_resolution=3, random_start=True, replicates=5, nsteps=1)
        )


def test_dimensions_multiple_of_resolution_is_accepted():
    cfg = _config(lattice_resolution=4, random_start=True, replicates=5, nsteps=1)
    assert len(bl.ExperimentLattice(cfg).run().trajectories) == 5


# ==========================================================================
# Core diffusion B: mechanics beyond the tutorial
# ==========================================================================

# --- periodic boundaries (wrap=True) -------------------------------------


def _wrap_walk(*, nreps=50, nsteps=100, box_size=10, raster=None, start=None, **kw):
    """Walk in the central tile [box_size, 2 * box_size) of a 3x3 raster."""
    if raster is None:
        raster = np.zeros((3 * box_size, 3 * box_size), dtype=np.uint8)
    if start is None:
        start = np.full((nreps, 2), box_size + box_size // 2)
    res = bl.random_walk_2d_lattice(
        raster,
        nsteps=nsteps,
        nreps=nreps,
        diff_coefficient=1.0,
        a=1,
        start=start,
        rng=np.random.default_rng(0),
        dimensions=(0, box_size),
        has_ghost=True,
        wrap=True,
        store_history=True,
        **kw,
    )
    traj = res["traj"]
    return (
        res,
        _stack(traj, "Y", nreps).astype(int),
        _stack(traj, "X", nreps).astype(int),
    )


def _unwrap(positions, box_size):
    """Displacement from the first row, from minimal-image steps (unit moves)."""
    steps = ((np.diff(positions, axis=0) + box_size // 2) % box_size) - box_size // 2
    return np.vstack([np.zeros((1, positions.shape[1]), dtype=int), steps.cumsum(0)])


def test_wrap_keeps_positions_in_box():
    _, y, x = _wrap_walk()
    for pos in (y, x):
        assert pos.min() == 10  # all borders are reached ...
        assert pos.max() == 19
    assert ((y >= 10) & (y < 20) & (x >= 10) & (x < 20)).all()  # ... never left


def test_wrap_msd_equals_unwrapped_displacement():
    res, y, x = _wrap_walk()
    expected = _unwrap(y, 10) ** 2 + _unwrap(x, 10) ** 2
    assert np.array_equal(_stack(res["traj"], "MSD", 50), expected)
    assert expected.max() > 10**2  # walkers really crossed the box several times


def test_wrap_free_diffusion_statistics():
    nreps, nsteps = 2000, 100
    res, y, x = _wrap_walk(nreps=nreps, nsteps=nsteps)
    for pos in (y, x):
        assert _unwrap(pos, 10)[-1].var() == pytest.approx(nsteps / 2, rel=0.1)
    msd = _stack(res["traj"], "MSD", nreps)
    assert msd[-1].mean() == pytest.approx(nsteps, rel=0.1)


def test_wrap_blocked_boundary_cell_prevents_crossing():
    raster = np.zeros((30, 30), dtype=np.uint8)
    raster[10:20, 10] = 1  # wall on the first tile column
    nreps = 30
    res, y, x = _wrap_walk(
        nreps=nreps, nsteps=200, raster=raster, start=np.full((nreps, 2), (15, 18))
    )
    assert x.min() >= 11  # the wall cannot be entered, from either side
    assert x.max() <= 19
    expected = _unwrap(y, 10) ** 2 + (x - 18) ** 2  # x never wraps
    assert np.array_equal(_stack(res["traj"], "MSD", nreps), expected)


def test_wrap_experiment_stays_in_central_tile():
    cfg = _config(dimensions=(0, 10), start=(5, 5), replicates=50, nsteps=30)
    y, x = _first_positions(
        bl.ExperimentLattice(dataclasses.replace(cfg, wrap=True)).run()
    )
    assert ((y >= 10) & (y < 20) & (x >= 10) & (x < 20)).all()
    y, x = _first_positions(bl.ExperimentLattice(cfg).run())  # wrap=False
    assert ((y < 10) | (y >= 20) | (x < 10) | (x >= 20)).any()  # leaves the tile


def test_wrap_with_respawn_resets_displacement():
    # Chosen block at the centre of a 20 box; start far away at (22, 22). After a hit
    # the walker respawns there and its displacement (wrap shift) starts from zero.
    box_size, nreps, nsteps = 20, 50, 200
    raster = np.zeros((60, 60), dtype=np.uint8)
    raster[28:32, 28:32] = 1
    res, y, x = _wrap_walk(
        nreps=nreps,
        nsteps=nsteps,
        box_size=box_size,
        raster=raster,
        start=np.full((nreps, 2), (22, 22)),
        raster_chosen_obstacles=raster.copy(),
        steady_state_chosen_obstacle=True,
    )
    disp_y = np.zeros(nreps, dtype=int)
    disp_x = np.zeros(nreps, dtype=int)
    expected = np.zeros((nsteps + 1, nreps), dtype=int)
    respawns = 0
    for t in range(1, nsteps + 1):
        dy = ((y[t] - y[t - 1] + 10) % 20) - 10
        dx = ((x[t] - x[t - 1] + 10) % 20) - 10
        jump = np.abs(dy) + np.abs(dx) > 1  # respawn: not a unit move
        respawns += jump.sum()
        disp_y = np.where(jump, y[t] - 22, disp_y + dy)
        disp_x = np.where(jump, x[t] - 22, disp_x + dx)
        expected[t] = disp_y**2 + disp_x**2
    assert respawns > 0
    assert np.array_equal(_stack(res["traj"], "MSD", nreps), expected)


def test_wrap_ghost_mode_random_start_stays_in_tile():
    cfg = _config(
        dimensions=(0, 20),
        has_ghost=True,
        wrap=True,
        shapes=[box(8, 8, 12, 12)],
        random_start=True,
        replicates=50,
        nsteps=100,
    )
    y, x = _first_positions(bl.ExperimentLattice(cfg).run())
    assert ((y >= 20) & (y < 40) & (x >= 20) & (x < 40)).all()


# --- shift_origin --------------------------------------------------------


def test_shift_origin_moves_shapes_aim_and_start_together():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(8, 8, 12, 12)],  # next to the start, which must stay free
        aim_area=Point(0, 0).buffer(3),
        start=(0, 0),
        shift_origin=True,
        replicates=5,
        nsteps=5,
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    # origin -> domain centre (20) -> raster frame (+ 40)
    assert run.shapes[0].bounds == (68.0, 68.0, 72.0, 72.0)
    assert np.argwhere(exp.aim_mask).mean(axis=0) == pytest.approx([60, 60], abs=0.6)
    assert run.trajectories[0].data[["Y", "X"]].iloc[0].tolist() == [60, 60]
    assert run.mean_trajectories()["Aim"].iloc[0] == 1.0


def test_shift_origin_resolution_two_start_matches_aim_and_shapes():
    cfg = _config(
        dimensions=(0, 40),
        lattice_resolution=2,
        shapes=[box(8, 8, 12, 12)],  # next to the start, which must stay free
        aim_area=Point(0, 0).buffer(3),
        start=(0, 0),
        shift_origin=True,
        replicates=5,
        nsteps=5,
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    # origin -> domain centre (20 world = 10 cells) -> raster frame (+ 20 cells)
    assert np.argwhere(exp.aim_mask).mean(axis=0) == pytest.approx([30, 30], abs=0.6)
    assert run.trajectories[0].data[["Y", "X"]].iloc[0].tolist() == [30, 30]
    assert run.mean_trajectories()["Aim"].iloc[0] == 1.0


def test_shift_origin_does_not_mutate_config():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(8, 8, 12, 12)],
        aim_area=Point(0, 0).buffer(3),
        start=(0, 0),
        shift_origin=True,
    )
    bl.ExperimentLattice(cfg)
    assert cfg.shapes[0].bounds == (8.0, 8.0, 12.0, 12.0)
    assert cfg.aim_area.bounds == Point(0, 0).buffer(3).bounds
    assert cfg.start == (0, 0)


def test_shift_origin_with_start_area():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(-5, -5, 5, 5)],
        start_area=box(-8, -8, 8, 8),  # around the (shifted) shape
        shift_origin=True,
        random_start=True,
        replicates=300,
        nsteps=1,
    )
    y, x = _first_positions(bl.ExperimentLattice(cfg).run())
    assert ((y[0] >= 51) & (y[0] <= 69) & (x[0] >= 51) & (x[0] <= 69)).all()


def test_shift_origin_with_start_area_by_size():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(-5, -5, 5, 5), box(10, 10, 15, 15)],
        start_area_by_size=(100.0, 3.0, box(-20, -20, 20, 20)),
        shift_origin=True,
        random_start=True,
        replicates=300,
        nsteps=1,
    )
    y, x = _first_positions(bl.ExperimentLattice(cfg).run())
    assert ((y[0] >= 51) & (y[0] <= 69) & (x[0] >= 51) & (x[0] <= 69)).all()


# --- start convention ----------------------------------------------------


def test_start_tuple_is_x_y():
    cfg = _config(dimensions=(0, 40), start=(10, 20), replicates=3, nsteps=1)
    first = bl.ExperimentLattice(cfg).run().trajectories[0].data.iloc[0]
    assert (first["X"], first["Y"]) == (10 + 40, 20 + 40)  # x = 10, y = 20


@pytest.mark.parametrize("has_ghost", [True, False])
def test_asymmetric_start_is_inside_aim_at_same_x_y(has_ghost):
    cfg = _aim_config(
        has_ghost=has_ghost,
        start=(10, 20),
        aim_area=Point(10, 20).buffer(3),
        replicates=3,
        nsteps=1,
    )
    run = bl.ExperimentLattice(cfg).run()
    assert run.mean_trajectories()["Aim"].iloc[0] == 1.0
    first = run.trajectories[0].data.iloc[0]
    assert (first["X"], first["Y"]) == (10 + 60, 20 + 60)


def test_start_is_in_lattice_cells_not_world_units():
    # Resolution 2: start (10, 20) cells is world (20, 40); the aim is in world units.
    cfg = _aim_config(
        lattice_resolution=2,
        start=(10, 20),
        aim_area=Point(20, 40).buffer(3),
        replicates=3,
        nsteps=1,
    )
    run = bl.ExperimentLattice(cfg).run()
    assert run.mean_trajectories()["Aim"].iloc[0] == 1.0
    first = run.trajectories[0].data.iloc[0]
    assert (first["X"], first["Y"]) == (10 + 30, 20 + 30)  # + max_dim / resolution


def test_symmetric_start_is_unaffected_by_axis_convention():
    cfg = _config(dimensions=(0, 40), start=(15, 15), replicates=3, nsteps=1)
    first = bl.ExperimentLattice(cfg).run().trajectories[0].data.iloc[0]
    assert (first["X"], first["Y"]) == (55, 55)


# --- start_area and start_area_by_size -----------------------------------


def test_start_area_restricts_random_starts():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(14, 14, 16, 16)],  # an obstacle inside the start area
        start_area=box(10, 10, 20, 20),
        random_start=True,
        replicates=500,
        nsteps=1,
    )
    exp = bl.ExperimentLattice(cfg)
    y, x = _first_positions(exp.run())
    world_y, world_x = y[0] - 40, x[0] - 40
    assert ((world_y >= 9) & (world_y <= 20) & (world_x >= 9) & (world_x <= 20)).all()
    assert (exp.raster[y[0], x[0]] == 0).all()  # not on the obstacle
    assert world_y.min() <= 11
    assert world_y.max() >= 18  # spread over the area


def test_start_area_used_by_steady_state_respawn():
    chosen = box(30, 30, 34, 34)
    cfg = _config(
        dimensions=(0, 50),
        shapes=[chosen],
        chosen_obstacles_by_area=chosen.area,
        steady_state_chosen_obstacle=True,
        start_area=box(5, 5, 10, 10),
        random_start=True,
        replicates=100,
        nsteps=600,
    )
    y, x, _, _ = _ring_arrays(bl.ExperimentLattice(cfg).run())
    jump = np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0)) > 1
    assert jump.sum() > 0  # respawns happened
    landing_y, landing_x = y[1:][jump] - 50, x[1:][jump] - 50
    assert ((landing_y >= 4) & (landing_y <= 10)).all()
    assert ((landing_x >= 4) & (landing_x <= 10)).all()


def test_start_area_ignored_without_random_start():
    cfg = _config(
        dimensions=(0, 40),
        start=(20, 20),
        start_area=box(2, 2, 6, 6),
        replicates=3,
        nsteps=1,
    )
    first = bl.ExperimentLattice(cfg).run().trajectories[0].data.iloc[0]
    assert (first["Y"], first["X"]) == (60, 60)  # the fixed start wins, silently


def test_start_area_by_size_ring_around_matching_shape():
    near, far = box(10, 10, 14, 14), box(25, 25, 31, 31)  # areas 16 and 36
    cfg = _config(
        dimensions=(0, 40),
        shapes=[near, far],
        start_area_by_size=(16.0, 2.0, box(0, 0, 40, 40)),
        random_start=True,
        replicates=500,
        nsteps=1,
    )
    exp = bl.ExperimentLattice(cfg)
    y, x = _first_positions(exp.run())
    assert (exp.raster[y[0], x[0]] == 0).all()  # outside both shapes
    points = [Point(px - 40, py - 40) for py, px in zip(y[0], x[0], strict=True)]
    assert max(near.distance(p) for p in points) <= 3.0  # ring width 2 (+1 cell)
    assert min(far.distance(p) for p in points) > 5.0


def test_start_area_by_size_without_match_raises():
    cfg = _config(
        shapes=[box(5, 5, 9, 9)],
        start_area_by_size=(99.0, 2.0, box(0, 0, 20, 20)),
        random_start=True,
    )
    with pytest.raises(ValueError, match="start_area_by_size"):
        bl.ExperimentLattice(cfg).run()


def test_start_area_by_size_requires_shapes():
    cfg = _config(start_area_by_size=(16.0, 2.0, box(0, 0, 20, 20)))
    with pytest.raises(ValueError, match="requires shapes"):
        bl.ExperimentLattice(cfg)


# --- ghost mode ----------------------------------------------------------


def test_ghost_fixed_start_in_world_frame():
    cfg = _config(dimensions=(0, 40), has_ghost=True, start=(20, 20), replicates=3)
    first = bl.ExperimentLattice(cfg).run().trajectories[0].data.iloc[0]
    assert (first["Y"], first["X"]) == (60, 60)  # centre of the central tile


def test_ghost_random_start_valid_with_obstacles():
    cfg = _config(
        dimensions=(0, 40),
        has_ghost=True,
        shapes=[box(10, 10, 14, 14), box(25, 20, 31, 28)],
        particle_radius=2,
        random_start=True,
        replicates=200,
        nsteps=1,
    )
    exp = bl.ExperimentLattice(cfg)
    y, x = _first_positions(exp.run())
    assert ((y[0] >= 40) & (y[0] < 80) & (x[0] >= 40) & (x[0] < 80)).all()
    assert exp.free_centers[y[0], x[0]].all()


def test_ghost_and_non_ghost_rasters_match_for_shapes_inside_domain():
    shapes = [box(10, 10, 14, 14), box(25, 20, 31, 28)]
    ghost = bl.ExperimentLattice(
        _config(dimensions=(0, 40), shapes=shapes, has_ghost=True)
    )
    plain = bl.ExperimentLattice(_config(dimensions=(0, 40), shapes=shapes))
    assert np.array_equal(ghost.raster, plain.raster)


# --- chosen obstacle with a particle radius ------------------------------


def _chosen_radius_experiment(**overrides):
    chosen, other = box(20, 20, 26, 26), box(8, 8, 12, 12)
    params = {
        "dimensions": (0, 50),
        "shapes": [chosen, other],
        "chosen_obstacles_by_area": chosen.area,
        "particle_radius": 2,
        "random_start": True,
        "replicates": 300,
        "nsteps": 300,
    }
    params.update(overrides)
    return bl.ExperimentLattice(_config(**params))


def test_free_raster_chosen_obstacles_bruteforce():
    exp = _chosen_radius_experiment()
    chosen = exp.raster_chosen_obstacles.astype(bool)
    height, width = chosen.shape
    offsets = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if i * i + j * j <= 4]
    expected = np.zeros_like(chosen)
    for row in range(height):
        for col in range(width):
            expected[row, col] = all(
                0 <= row + i < height
                and 0 <= col + j < width
                and not chosen[row + i, col + j]
                for i, j in offsets
            )
    assert np.array_equal(exp.free_raster_chosen_obstacles, expected)


def test_chosen_radius_hit_distance_invariant():
    exp = _chosen_radius_experiment()
    run = exp.run()
    y, x, _, _ = _ring_arrays(run)
    chosen = exp.raster_chosen_obstacles == 1
    other = (exp.raster == 1) & ~chosen
    dist_chosen = distance_transform_edt(~chosen)
    dist_other = distance_transform_edt(~other)
    assert (dist_chosen[y, x] > 2).all()  # the disc never overlaps the chosen one
    assert (dist_other[y, x] > 2).all()  # nor any other obstacle
    fpt = run.fpt["FirstInactiveTime"].to_numpy()
    assert 0.1 < np.mean(~np.isnan(fpt)) < 1  # some, but not all, walkers hit
    for walker in np.flatnonzero(~np.isnan(fpt)):
        py, px = y[-1, walker], x[-1, walker]  # frozen where the hit happened
        neighbours = [(py + 1, px), (py - 1, px), (py, px + 1), (py, px - 1)]
        assert min(dist_chosen[n] for n in neighbours) <= 2


def test_chosen_radius_nonchosen_obstacle_never_counts_as_hit():
    chosen = box(40, 40, 46, 46)  # far away from the start area
    other = box(8, 8, 12, 12)  # walkers wander around this one
    cfg = _config(
        dimensions=(0, 50),
        shapes=[chosen, other],
        chosen_obstacles_by_area=chosen.area,
        particle_radius=2,
        start_area=box(5, 5, 15, 15),
        random_start=True,
        replicates=200,
        nsteps=100,
    )
    exp = bl.ExperimentLattice(cfg)
    run = exp.run()
    y, x, active, _ = _ring_arrays(run)
    near_other = distance_transform_edt(
        ~((exp.raster == 1) & (exp.raster_chosen_obstacles == 0))
    )
    assert (
        near_other[y, x].min(axis=0) <= 3
    ).mean() > 0.3  # many touch its surroundings
    assert (
        run.fpt["FirstInactiveTime"].isna().all()
    )  # but a non-chosen obstacle never hits
    assert (active == 1).all()


def test_hits_equal_respawn_jumps():
    chosen = box(30, 30, 34, 34)
    cfg = _config(
        dimensions=(0, 50),
        shapes=[chosen],
        chosen_obstacles_by_area=chosen.area,
        steady_state_chosen_obstacle=True,
        start=(22, 22),
        replicates=60,
        nsteps=400,
    )
    run = bl.ExperimentLattice(cfg).run()
    # Per-walker hits from the hits table, jumps from the stored positions.
    hits = run.hits.drop(columns="Time").to_numpy().sum(axis=0)
    y, x, _, _ = _ring_arrays(run)
    jumps = (np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0)) > 1).sum(axis=0)
    assert hits.sum() > 0
    assert np.array_equal(hits, jumps)


def test_hits_per_walker_independent_of_save_every():
    chosen = box(30, 30, 34, 34)
    totals = {}
    for save_every in (1, 5):
        cfg = _config(
            dimensions=(0, 50),
            shapes=[chosen],
            chosen_obstacles_by_area=chosen.area,
            steady_state_chosen_obstacle=True,
            start=(22, 22),
            replicates=60,
            nsteps=400,
            save_every=save_every,
            rng=np.random.default_rng(11),
        )
        run = bl.ExperimentLattice(cfg).run()
        totals[save_every] = run.hits.drop(columns="Time").to_numpy().sum(axis=0)
    assert totals[1].sum() > 0
    assert np.array_equal(totals[1], totals[5])  # same seed, same walks


def test_steady_state_random_respawn_positions_valid():
    exp = _chosen_radius_experiment(steady_state_chosen_obstacle=True, nsteps=400)
    y, x, _, _ = _ring_arrays(exp.run())
    jump = np.abs(np.diff(y, axis=0)) + np.abs(np.diff(x, axis=0)) > 1
    assert jump.sum() > 20
    landing_y, landing_x = y[1:][jump], x[1:][jump]
    assert ((landing_y >= 50) & (landing_y < 100)).all()  # central tile
    assert ((landing_x >= 50) & (landing_x < 100)).all()
    assert exp.free_centers[landing_y, landing_x].all()
    assert len(set(zip(landing_y, landing_x, strict=True))) > 10  # random places


# ==========================================================================
# Core diffusion D: statistics robustness
# ==========================================================================


def _enclosed_run(nsteps=10, nreps=5, diff=1.0):
    """A run of walkers that cannot move (enclosed by obstacles): MSD is zero."""
    raster, _ = _enclosed()
    res = bl.random_walk_2d_lattice(
        raster,
        nsteps=nsteps,
        nreps=nreps,
        diff_coefficient=diff,
        start=np.full((nreps, 2), CELL),
        rng=np.random.default_rng(0),
    )
    trajectories = [
        bl.DiffusionLatticeTrajectories(group.reset_index(drop=True))
        for _, group in res["traj"].groupby("Replicate")
    ]
    return bl.ExperimentLatticeRun(
        trajectories, None, raster, True, res["fpt"], None, None
    )


def test_diff_coefficient_intercept_options():
    cfg = _config(
        dimensions=(0, 60),
        diff_coefficient=DIFF,
        replicates=800,
        nsteps=60,
        rng=np.random.default_rng(4),
    )
    run = bl.ExperimentLattice(cfg).run()
    model, d_est = run.get_mean_diff_coefficients(intercept=True)
    final_msd = run.mean_trajectories()["MSD"].iloc[-1]
    assert d_est == pytest.approx(DIFF, rel=0.1)
    assert abs(model.intercept_) < 0.05 * final_msd  # free diffusion: through zero
    _, prefactor, exponent = run.get_mean_diff_coefficients_pw(intercept=False)
    assert prefactor == 1.0  # exp(0): the fit is forced through the origin
    assert np.isfinite(exponent)


def test_zero_msd_gives_zero_diff_coefficient():
    run = _enclosed_run()
    assert run.get_mean_diff_coefficients()[1] == pytest.approx(0.0, abs=1e-12)
    dist = run.get_mean_diff_coefficients_dist()
    assert np.isnan(dist["D_dist"].iloc[0])
    assert (dist["D_dist"].iloc[1:] == 0).all()


def test_pw_fit_with_zero_msd_raises_clear_error():
    with pytest.raises(ValueError, match="MSD"):
        _enclosed_run().get_mean_diff_coefficients_pw()


def test_pw_fit_with_single_time_point_raises_clear_error():
    run = bl.ExperimentLattice(_config(nsteps=0)).run()
    with pytest.raises(ValueError, match="time points"):
        run.get_mean_diff_coefficients_pw()


def test_pw_fit_needs_two_points_after_start():
    run = bl.ExperimentLattice(_config(nsteps=1)).run()
    with pytest.raises(ValueError, match="time points"):
        run.get_mean_diff_coefficients_pw()


def test_ensemble_pw_fit_raises_without_time_points(tmp_path):
    files = _write_files(tmp_path, [[box(2, 2, 4, 4)], [box(3, 2, 5, 4)]])
    ensemble = _run_inline(files, _ensemble_config(nsteps=0, replicates=5))
    with pytest.raises(ValueError, match="time points"):
        ensemble.get_mean_diff_coefficients_pw()


def test_diff_coefficient_with_only_start_point():
    run = bl.ExperimentLattice(_config(nsteps=0)).run()
    assert run.get_mean_diff_coefficients()[1] == pytest.approx(0.0, abs=1e-12)
    assert len(run.get_mean_diff_coefficients_dist()) == 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_get_mean_fpt_single_hit(ring_terminating):
    fpt = pd.DataFrame(
        {
            "Replicate": range(5),
            "FirstInactiveTime": [2.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    summary = dataclasses.replace(ring_terminating, fpt=fpt).get_mean_fpt()
    assert summary["Mean"] == 2.0
    assert summary["Median"] == 2.0
    assert summary["IQR"] == 0.0
    assert summary[["Std", "Sem", "Ci_low", "Ci_high"]].isna().all()  # ddof=1, one hit


def test_get_mean_fpt_is_conditional_on_hitting():
    # Corridor x = 1..5, start at x = 3: the true mean is 18 steps = 4.5 time units
    # (D = 1), but only walkers that hit within 15 steps enter the statistic.
    raster = np.ones((7, 7), dtype=np.uint8)
    raster[3, 1:6] = 0
    chosen = np.zeros_like(raster)
    chosen[3, 0] = chosen[3, 6] = 1
    nreps = 4000
    res = bl.random_walk_2d_lattice(
        raster,
        nsteps=15,
        nreps=nreps,
        diff_coefficient=1.0,
        start=np.full((nreps, 2), (3, 3)),
        raster_chosen_obstacles=chosen,
        rng=np.random.default_rng(1),
    )
    trajectories = [
        bl.DiffusionLatticeTrajectories(group.reset_index(drop=True))
        for _, group in res["traj"].groupby("Replicate")
    ]
    run = bl.ExperimentLatticeRun(
        trajectories, None, raster, True, res["fpt"], None, None
    )
    hit = res["fpt"]["FirstInactiveTime"].notna().mean()
    assert 0.2 < hit < 0.9  # many walkers are censored ...
    summary = run.get_mean_fpt()
    assert summary["Mean"] < 0.75 * 4.5  # ... so the mean is biased low
    assert list(summary.index) == [
        "Mean",
        "Std",
        "Sem",
        "Ci_low",
        "Ci_high",
        "Median",
        "IQR",
    ]  # and no censoring information is reported


def test_mean_trajectories_matches_float64_reference():
    rng = np.random.default_rng(0)
    nreps, ntimes = 200, 50
    msd = rng.integers(10**6, 10**8, size=(nreps, ntimes))
    mad = rng.integers(10**3, 10**4, size=(nreps, ntimes))
    active = rng.integers(0, 2, size=(nreps, ntimes))
    trajectories = [
        bl.DiffusionLatticeTrajectories(
            pd.DataFrame(
                {
                    "Time": np.arange(ntimes, dtype=np.float32),
                    "Replicate": i,
                    "MSD": msd[i].astype(np.uint32),
                    "MAD": mad[i].astype(np.uint32),
                    "Active": active[i].astype(np.uint8),
                }
            )
        )
        for i in range(nreps)
    ]
    run = bl.ExperimentLatticeRun(
        trajectories, None, np.zeros((3, 3)), True, pd.DataFrame(), None, None
    )
    df = run.mean_trajectories()
    for name, values in (("MSD", msd), ("MAD", mad), ("Active", active)):
        reference = pd.DataFrame(values.astype(float))
        assert np.allclose(df[name], reference.mean(axis=0), rtol=1e-4)
        assert np.allclose(df[f"{name}_std"], reference.std(axis=0, ddof=1), rtol=1e-4)
        assert np.allclose(df[f"{name}_sem"], reference.sem(axis=0, ddof=1), rtol=1e-4)


def _plot_run(**overrides):
    mpl.use("Agg")
    params = {"replicates": 3, "nsteps": 40}
    params.update(overrides)
    return bl.ExperimentLattice(_aim_config(**params)).run()


def test_plot_run_scatter_and_lines():
    run = _plot_run()
    lines = run.plot_run()
    assert len(lines.lines) == 3
    assert all(len(line.get_xdata()) == 41 for line in lines.lines)  # nsteps + 1
    assert len(run.plot_run(steps=10).lines[0].get_xdata()) == 10
    scatter = run.plot_run(scatter=True)
    assert len(scatter.collections) == 3
    assert len(scatter.lines) == 0
    mpl.pyplot.close("all")


def test_plot_run_steps_limits_time_steps_in_aim_mode():
    run = _plot_run()
    ax = run.plot_run(aim_area=True, steps=10)
    # two colours per trajectory (3 replicates): together 10 points each
    drawn = sum(len(c.get_offsets()) for c in ax.collections)
    assert drawn == 3 * 10
    mpl.pyplot.close("all")


@pytest.mark.parametrize("has_ghost", [False, True])
@pytest.mark.parametrize("aim", [False, True])
def test_plot_run_steps_limits_points_in_both_ghost_modes(has_ghost, aim):
    run = _plot_run(has_ghost=has_ghost)
    ax = run.plot_run(aim_area=aim, steps=10, scatter=True)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 3 * 10
    mpl.pyplot.close("all")


def test_plot_run_without_history_raises_clear_error():
    run = bl.ExperimentLattice(_config(store_history=False, replicates=3)).run()
    with pytest.raises(ValueError, match="store_history"):
        run.plot_run()


# ==========================================================================
# Core diffusion E: reproducibility
# ==========================================================================


def _pipeline_config(**overrides):
    chosen = box(30, 10, 34, 14)
    params = {
        "dimensions": (0, 40),
        "shapes": [chosen, box(8, 8, 14, 14)],
        "chosen_obstacles_by_area": chosen.area,
        "steady_state_chosen_obstacle": True,
        "particle_radius": 2,
        "random_start": True,
        "aim_area": Point(20, 20).buffer(8),
        "replicates": 40,
        "nsteps": 100,
        "rng": np.random.default_rng(21),
    }
    params.update(overrides)
    return _config(**params)


def test_same_config_gives_identical_results():
    cfg = _pipeline_config()
    first = bl.ExperimentLattice(cfg).run()
    second = bl.ExperimentLattice(cfg).run()  # the config's own rng is not advanced
    for a, b in zip(first.trajectories, second.trajectories, strict=True):
        pd.testing.assert_frame_equal(a.data, b.data)
    pd.testing.assert_frame_equal(first.hits, second.hits)
    pd.testing.assert_frame_equal(first.fpt, second.fpt)
    assert np.array_equal(first.waiting_times, second.waiting_times)


def test_history_flag_does_not_change_dynamics():
    with_history = bl.ExperimentLattice(_pipeline_config(store_history=True)).run()
    without = bl.ExperimentLattice(_pipeline_config(store_history=False)).run()
    for column in ("MSD", "Active", "Aim"):
        a = with_history.mean_trajectories()[column]
        b = without.mean_trajectories()[column]
        assert np.array_equal(a, b)
    assert np.array_equal(with_history.hits.to_numpy(), without.hits.to_numpy())


def test_save_every_does_not_change_dynamics():
    full = bl.ExperimentLattice(_pipeline_config(save_every=1)).run()
    sparse = bl.ExperimentLattice(_pipeline_config(save_every=5)).run()
    for column in ("MSD", "Aim"):
        a = full.mean_trajectories()[column].to_numpy()[::5]
        assert np.array_equal(a, sparse.mean_trajectories()[column].to_numpy())
    pd.testing.assert_frame_equal(full.fpt, sparse.fpt)
    assert np.array_equal(
        full.hits.drop(columns="Time").to_numpy().sum(axis=0),
        sparse.hits.drop(columns="Time").to_numpy().sum(axis=0),
    )


def test_unseeded_runs_differ():
    first = bl.ExperimentLattice(_pipeline_config(rng=None)).run()
    second = bl.ExperimentLattice(_pipeline_config(rng=None)).run()
    assert not np.array_equal(
        first.mean_trajectories()["MSD"], second.mean_trajectories()["MSD"]
    )


def test_run_twice_on_same_experiment_continues_stream():
    cfg = _pipeline_config()
    experiment = bl.ExperimentLattice(cfg)
    first = experiment.run().mean_trajectories()["MSD"].to_numpy()
    second = experiment.run().mean_trajectories()["MSD"].to_numpy()
    fresh = bl.ExperimentLattice(cfg).run().mean_trajectories()["MSD"].to_numpy()
    assert not np.array_equal(first, second)  # the copied rng has advanced
    assert np.array_equal(first, fresh)  # a new experiment reproduces the first run


def test_run_and_config_pickle_roundtrip():
    cfg = _pipeline_config()
    run = bl.ExperimentLattice(cfg).run()
    restored = pickle.loads(pickle.dumps(run))  # noqa: S301 - our own object
    pd.testing.assert_frame_equal(run.mean_trajectories(), restored.mean_trajectories())
    assert np.array_equal(run.raster, restored.raster)
    assert [s.bounds for s in run.shapes] == [s.bounds for s in restored.shapes]
    cfg_copy = pickle.loads(pickle.dumps(cfg))  # noqa: S301 - our own object
    again = bl.ExperimentLattice(cfg_copy).run()
    pd.testing.assert_frame_equal(run.mean_trajectories(), again.mean_trajectories())


# Golden values (numpy 2.4.6): a safety net for refactoring. numpy does not promise
# identical random streams across versions, so the tests skip on other versions.
_GOLDEN_NUMPY = "2.4"
_golden = pytest.mark.skipif(
    ".".join(np.__version__.split(".")[:2]) != _GOLDEN_NUMPY,
    reason=f"golden values were recorded with numpy {_GOLDEN_NUMPY}",
)


@_golden
def test_golden_free_walk_seed0():
    nreps = 20
    traj = _walk(nreps=nreps, nsteps=30, seed=0, store_history=True)["traj"]
    y, x = _stack(traj, "Y", nreps), _stack(traj, "X", nreps)
    assert int(traj["MSD"].sum()) == 12082
    assert (int(y[-1].sum()), int(x[-1].sum())) == (994, 986)
    digest = hashlib.sha256(np.stack([y, x]).astype(np.int32).tobytes()).hexdigest()
    assert digest[:16] == "21a07847b9897f92"


@_golden
def test_golden_experiment_with_obstacles_radius_and_aim():
    cfg = _config(
        dimensions=(0, 40),
        shapes=[box(10, 10, 16, 16)],
        particle_radius=2,
        random_start=True,
        aim_area=Point(20, 20).buffer(8),
        replicates=50,
        nsteps=60,
        rng=np.random.default_rng(0),
    )
    df = bl.ExperimentLattice(cfg).run().mean_trajectories()
    assert df["MSD"].sum() == pytest.approx(1948.76, abs=1e-2)
    assert df["Aim"].sum() == pytest.approx(6.62, abs=1e-3)
    assert df["MSD"].iloc[-1] == pytest.approx(60.02, abs=1e-2)
