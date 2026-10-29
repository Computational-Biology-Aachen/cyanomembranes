"""Tests for the `membrane_analysis` package (repo root, not part of `src/`).

Inputs are small and synthetic. The tests cover the three Medium bugs that were
fixed on 2026-10-29 (see INFO/OPEN_ITEMS.md, section "membrane_analysis"):
the r^2 of the protein-count regression, the units of MSD/MAD for lattice
resolutions above one, and the periodic cluster analysis.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from shapely import to_wkt  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

# membrane_analysis lives in the repository root, which is not on the path when
# pytest is started from elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cyanomembranes as cm  # noqa: E402
from membrane_analysis import utils as mu  # noqa: E402
from membrane_analysis.config import (  # noqa: E402
    AnalysisType,
    Condition,
    ScenarioConfig,
)
from membrane_analysis.pipeline import build_diffusion_metrics  # noqa: E402
from membrane_analysis.plotting import plot_diffusion  # noqa: E402

bl = cm.brownian_lattice

# --------------------------------------------------------------------------
# flood_fill (periodic connected regions)
# --------------------------------------------------------------------------


def test_flood_fill_free_grid_is_one_region():
    assert mu.flood_fill(np.zeros((4, 5), dtype=np.uint8)) == [20]


def test_flood_fill_single_wall_does_not_split_a_periodic_grid():
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[:, 3] = 1  # the free columns 4, 5, 0, 1, 2 are connected around the edge
    assert mu.flood_fill(grid) == [30]


def test_flood_fill_two_walls_split_a_periodic_grid():
    grid = np.zeros((6, 6), dtype=np.uint8)
    grid[:, 1] = 1
    grid[:, 4] = 1
    # columns 2-3 and columns 5-0 (connected across the edge)
    assert sorted(mu.flood_fill(grid)) == [12, 12]


def test_flood_fill_accepts_boolean_grids():
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, :] = True
    assert mu.flood_fill(grid) == [20]  # rows 3, 4, 0, 1 wrap around


# --------------------------------------------------------------------------
# cluster_sizes
# --------------------------------------------------------------------------


def _cluster(polygons, *, resolution=1, max_dim=100, radius=3):
    return mu.cluster_sizes(
        polygons, resolution=resolution, max_dim=max_dim, particle_radius=radius
    )


def test_cluster_sizes_tile_follows_max_dim_and_resolution():
    # the crop was hard-coded to 5000 and returned an empty raster otherwise
    for max_dim, resolution in [(100, 1), (100, 2), (60, 1)]:
        sizes, tile, not_accessible = _cluster(
            [Point(max_dim / 2, max_dim / 2).buffer(8)],
            resolution=resolution,
            max_dim=max_dim,
        )
        n = max_dim // resolution
        assert tile.shape == (n, n)
        assert not_accessible.shape == (n, n)
        assert sum(sizes) == (~not_accessible).sum() > 0


def test_cluster_sizes_one_obstacle_is_one_cluster():
    sizes, _, _ = _cluster([Point(50, 50).buffer(10)])
    assert len(sizes) == 1


def test_cluster_sizes_edge_strip_is_accessible():
    # border_value=0 made every cell within the radius of the tile edge
    # inaccessible although the space continues on the other side
    _, _, not_accessible = _cluster([Point(50, 50).buffer(10)])
    accessible = ~not_accessible
    assert accessible.shape == (100, 100)  # an empty tile would pass vacuously
    assert accessible[:3, :].all()
    assert accessible[-3:, :].all()
    assert accessible[:, :3].all()
    assert accessible[:, -3:].all()


def test_cluster_sizes_matches_periodic_dilation_reference():
    # reference by duality: the particle centre is blocked where the particle,
    # placed there, overlaps an obstacle (periodic dilation of the obstacles)
    _, tile, not_accessible = _cluster([Point(50, 50).buffer(10)])
    particle = cm.brownian_lattice._make_lattice_particle(3)
    pad = particle.shape[0] // 2
    occupied = np.pad(tile == 1, pad, mode="wrap")
    blocked = binary_dilation(occupied, structure=particle)[pad:-pad, pad:-pad]
    assert np.array_equal(not_accessible, blocked)


def test_cluster_sizes_obstacle_across_the_edge_is_periodic():
    # an obstacle centred on the tile corner is the same as one in the middle
    # of a shifted tile: the accessible area has to match
    centre, _, na_centre = _cluster([Point(50, 50).buffer(10)])
    polygons = [Point(x, y).buffer(10) for x in (0, 100) for y in (0, 100)]
    corner, _, na_corner = _cluster(polygons)
    assert len(corner) == 1
    assert (~na_corner).sum() == (~na_centre).sum()
    assert len(centre) == 1


def test_cluster_sizes_wall_over_full_height_is_one_cluster():
    # the free space left and right of the wall is connected across the edge
    sizes, _, _ = _cluster([box(40, -5, 60, 105)])
    assert len(sizes) == 1


def test_cluster_sizes_two_walls_give_two_clusters():
    sizes, _, _ = _cluster([box(20, -5, 30, 105), box(60, -5, 70, 105)])
    assert len(sizes) == 2


def test_cluster_sizes_radius_zero_counts_all_free_cells():
    sizes, tile, _ = _cluster([Point(50, 50).buffer(10)], radius=0)
    assert sizes == [int((tile == 0).sum())]


# --------------------------------------------------------------------------
# plot_diffusion: r^2 of the two regressions
# --------------------------------------------------------------------------

PKEY = "3WU2-PSII-ThermosynVul"


def _diffusion_scenario(nprots):
    return ScenarioConfig(
        name="diff",
        analysis_type=AnalysisType.DIFFUSION,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={PKEY: nprots},
    )


def _write_diffusion_outputs(root, nprots, coverage, last_d):
    out = root / "diff"
    (out / "timeseries").mkdir(parents=True)
    rows = []
    for n, cov, d in zip(nprots, coverage, last_d, strict=True):
        rows.append(
            {
                "pkey": PKEY,
                "nprot": n,
                "total_prot": n,
                "mean_coverage": cov,
                "last_D_dist": d,
                "last_D_dist_ci_low": d - 0.02,
                "last_D_dist_ci_high": d + 0.02,
            }
        )
        pd.DataFrame(
            {
                "Sqrt_MSD": [1.0, 2.0, 3.0],
                "Norm_Diff_Dist": [1.0, 0.9, d],
                "Norm_Diff_Dist_ci_low": [1.0, 0.88, d - 0.02],
                "Norm_Diff_Dist_ci_high": [1.0, 0.92, d + 0.02],
                "MAD": [0.8, 1.7, 2.6],
            }
        ).to_csv(out / "timeseries" / f"{PKEY}_{n}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "scalars.csv", index=False)
    return out


def _r2(x, y):
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def test_plot_diffusion_r2_belongs_to_each_regression(tmp_path):
    nprots = [10, 20, 30, 40]
    coverage = [0.04, 0.09, 0.30, 0.32]  # not proportional to the protein count
    last_d = [0.95, 0.90, 0.62, 0.60]
    out = _write_diffusion_outputs(tmp_path, nprots, coverage, last_d)
    expected_cov = _r2(np.array(coverage) * 100, last_d)
    expected_prot = _r2(nprots, last_d)
    assert abs(expected_cov - expected_prot) > 0.02  # the test can tell them apart

    plot_diffusion(_diffusion_scenario(nprots), tmp_path)

    params = pd.read_csv(out / "picture" / "reg_param_df.csv", index_col=0)
    r2 = params.set_index("type")["r2"]
    assert r2["coverage"] == pytest.approx(expected_cov)
    assert r2["proteins"] == pytest.approx(expected_prot)


def test_plot_diffusion_writes_the_figures(tmp_path):
    nprots = [10, 20, 30]
    out = _write_diffusion_outputs(
        tmp_path, nprots, [0.05, 0.10, 0.15], [0.95, 0.90, 0.85]
    )
    plot_diffusion(_diffusion_scenario(nprots), tmp_path)
    pictures = {p.name for p in (out / "picture").iterdir()}
    assert {
        f"membrane_diffusion_{PKEY}.png",
        f"mad_sqrtmsd_ratio_{PKEY}.png",
        "membrane_diffusion_last_D_dist.png",
        "reg_param_df.csv",
    } <= pictures


def test_plot_diffusion_rejects_other_analysis_types(tmp_path):
    scenario = _diffusion_scenario([10])
    scenario.analysis_type = AnalysisType.FPT
    with pytest.raises(ValueError, match="diffusion"):
        plot_diffusion(scenario, tmp_path)


# --------------------------------------------------------------------------
# build_diffusion_metrics: units of MSD / MAD for lattice resolutions
# --------------------------------------------------------------------------

D = 100.0


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


def _free_ensemble(tmp_path, monkeypatch, resolution):
    monkeypatch.setattr(bl, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(bl, "as_completed", lambda futures: iter(list(futures)))
    files = []
    for i, shape in enumerate([box(2, 2, 4, 4), box(30, 30, 32, 32)]):
        path = tmp_path / f"shapes_{i}.wkt"
        path.write_text(to_wkt(shape) + "\n")
        files.append(path)
    cfg = bl.ExperimentLatticeConfig()
    cfg.dimensions = (0, 40)
    cfg.lattice_resolution = resolution
    cfg.replicates = 300
    cfg.nsteps = 40
    cfg.save_every = 10
    cfg.diff_coefficient = D
    cfg.random_start = True
    cfg.has_ghost = True
    cfg.workers = 1
    cfg.rng = np.random.default_rng(0)
    run = bl.EnsembleExperimentLattice(files, cfg).run()
    condition = Condition(
        key=("p", 2),
        file_lst=files,
        pkey="p",
        nprot=2,
        mv=1.2,
        cg=None,
        crystal_prot=None,
        test="",
    )
    return run, condition, cfg


@pytest.mark.parametrize("resolution", [1, 2])
def test_diffusion_timeseries_are_in_physical_units(tmp_path, monkeypatch, resolution):
    run, condition, cfg = _free_ensemble(tmp_path, monkeypatch, resolution)
    _, ts = build_diffusion_metrics(run, condition, cfg)
    raw = run.mean_over_runs()

    # the simulation counts in lattice cells: MSD in cells^2, MAD in cells
    assert np.allclose(ts["MSD"], resolution**2 * raw["MSD"])
    assert np.allclose(ts["MSD_ci_low"], resolution**2 * raw["MSD_ci_low"])
    assert np.allclose(ts["MSD_ci_high"], resolution**2 * raw["MSD_ci_high"])
    assert np.allclose(ts["MAD"], resolution * raw["MAD"])
    assert np.allclose(ts["MAD_ci_low"], resolution * raw["MAD_ci_low"])
    assert np.allclose(ts["MAD_ci_high"], resolution * raw["MAD_ci_high"])
    assert np.allclose(ts["Sqrt_MSD"], np.sqrt(ts["MSD"]))


@pytest.mark.parametrize("resolution", [1, 2])
def test_free_diffusion_matches_theory_in_physical_units(
    tmp_path, monkeypatch, resolution
):
    run, condition, cfg = _free_ensemble(tmp_path, monkeypatch, resolution)
    _, ts = build_diffusion_metrics(run, condition, cfg)
    late = ts.iloc[1:]  # skip t = 0
    # <r^2> = 4 D t in two dimensions, independent of the lattice resolution
    assert (late["MSD"] / (4 * D * late["Time"])).to_numpy() == pytest.approx(
        1.0, abs=0.15
    )
    # the mean absolute displacement of a 2D random walk is sqrt(pi) / 2 * sqrt(MSD)
    # (Rayleigh distribution); the plot shows exactly this ratio
    ratio = (late["MAD"] / late["Sqrt_MSD"]).to_numpy()
    assert ratio == pytest.approx(np.sqrt(np.pi) / 2, abs=0.1)
    # the normalised coefficient D(r) / D_0 is one for free diffusion
    assert late["Norm_Diff_Dist"].to_numpy() == pytest.approx(1.0, abs=0.15)
