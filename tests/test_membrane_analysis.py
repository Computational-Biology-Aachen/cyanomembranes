"""Tests for the `membrane_analysis` package (repo root, not part of `src/`).

Inputs are small and synthetic. The tests cover the three Medium bugs that were
fixed on 2026-10-29 (see INFO/OPEN_ITEMS.md, section "membrane_analysis"):
the r^2 of the protein-count regression, the units of MSD/MAD for lattice
resolutions above one, and the periodic cluster analysis.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from shapely import to_wkt  # noqa: E402
from shapely.affinity import translate  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

# membrane_analysis lives in the repository root, which is not on the path when
# pytest is started from elsewhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cyanomembranes as cm  # noqa: E402
import membrane_analysis.plotting as mplot  # noqa: E402
from membrane_analysis import utils as mu  # noqa: E402
from membrane_analysis.config import (  # noqa: E402
    COLORBLIND_PALETTE,
    AnalysisType,
    Condition,
    PlotRunsConfig,
    ScenarioConfig,
    make_exp_config,
)
from membrane_analysis.pipeline import (  # noqa: E402
    ScalarMetrics,
    _iter_conditions,
    _run_one_nprot,
    build_aim_metrics,
    build_diffusion_metrics,
    build_fpt_metrics,
    build_rate_metrics,
    load_and_run,
    run_scenario,
)
from membrane_analysis.plotting import (  # noqa: E402
    _timeseries_filename_from_row,
    plot_aim,
    plot_diffusion,
    plot_fpt,
    plot_rate,
)

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


# --------------------------------------------------------------------------
# nndf: nearest-neighbour distribution and its error band
# --------------------------------------------------------------------------

BOX = 1000.0


def _ideal_gas_with_ghosts(n, seed):
    points = np.random.default_rng(seed).uniform(0, BOX, (n, 2))
    shifts = [(dx * BOX, dy * BOX) for dx in (0, -1, 1) for dy in (0, -1, 1)]
    return np.vstack([points + s for s in shifts])  # the originals come first


def test_nndf_g_std_is_the_binomial_standard_error():
    n = 300
    G, G_std, _, _, _ = mu.nndf(_ideal_gas_with_ghosts(n, 0), 50, BOX, r_max=150)
    assert G_std == pytest.approx(np.sqrt(G * (1 - G) / n))
    assert G[-1] == 1.0
    assert G_std[-1] == 0.0  # every particle has a neighbour within r_max
    assert (G_std >= 0).all()


def test_nndf_g_std_follows_the_scatter_of_g():
    n = 300
    results = [
        mu.nndf(_ideal_gas_with_ghosts(n, seed), 40, BOX, r_max=80)
        for seed in range(200)
    ]
    G = np.array([r[0] for r in results])
    band = np.array([r[1] for r in results])
    scatter = G.std(axis=0, ddof=1)
    for b in [5, 15, 25, 35]:
        # independent-neighbour approximation: slightly below the true scatter
        assert 0.6 < band[:, b].mean() / scatter[b] < 1.1

    # the former band, sqrt(cumsum(N p (1 - p))) / N, grew far too wide for large G
    p = np.diff(np.concatenate([np.zeros((len(G), 1)), G], axis=1), axis=1)
    old_band = np.sqrt(np.cumsum(n * p * (1 - p), axis=1)) / n
    for b in [20, 25, 30]:
        assert old_band[:, b].mean() / scatter[b] > 1.5
        assert band[:, b].mean() / scatter[b] < 1.1


def test_nndf_matches_the_poisson_distribution():
    n = 600
    G, _, _, _, centres = mu.nndf(_ideal_gas_with_ghosts(n, 1), 40, BOX, r_max=60)
    dr = centres[1] - centres[0]
    expected = 1 - np.exp(-(n / BOX**2) * np.pi * (centres + dr / 2) ** 2)
    assert pytest.approx(expected, abs=0.08) == G


# --------------------------------------------------------------------------
# _plot_rate_constants
# --------------------------------------------------------------------------


def _rate_scenario(cdegree=None):
    return ScenarioConfig(
        name="rate",
        analysis_type=AnalysisType.RATE,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={},
        cdegree=cdegree,
    )


def _rate_scalars(groups):
    """Two rows (coverages) per group; `groups` is a list of dicts of group columns."""
    rows = []
    for g, group in enumerate(groups):
        for j, coverage in enumerate([0.1, 0.2]):
            rows.append(
                {
                    **group,
                    "mean_coverage": coverage,
                    "k_mean": 1.0 + g + j,
                    "k_std": 0.1,
                }
            )
    return pd.DataFrame(rows)


def _run_rate_constants(monkeypatch, tmp_path, scenario, scalars):
    """Run the plot and return the figure and the number of `plt.close` calls."""
    closed = []
    original_close = mplot.plt.close
    monkeypatch.setattr(mplot.plt, "close", lambda fig=None: closed.append(fig))
    try:
        mplot._plot_rate_constants(scenario, scalars, tmp_path)
        fig = closed[0]
        legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    finally:
        for fig in closed:
            original_close(fig)
    return legend, len(closed)


def test_rate_constants_legend_uses_readable_protein_names(tmp_path, monkeypatch):
    # pandas returns 1-tuples as group keys for a list of one column
    scalars = _rate_scalars(
        [
            {"pkey": "3WU2-PSII-ThermosynVul"},
            {"pkey": "4H13-cytb6f"},
            {"pkey": "avg_membrane"},
        ]
    )
    legend, _ = _run_rate_constants(monkeypatch, tmp_path, _rate_scenario(), scalars)
    assert legend == ["PSII", "Cytb6f", "Avg"]


def test_rate_constants_figure_is_saved_and_closed_once(tmp_path, monkeypatch):
    scalars = _rate_scalars([{"pkey": "avg_membrane"}, {"pkey": "4H13-cytb6f"}])
    _, closes = _run_rate_constants(monkeypatch, tmp_path, _rate_scenario(), scalars)
    assert closes == 1  # it was closed after every group before
    assert (tmp_path / "rate_constant.png").exists()


def test_rate_constants_more_groups_than_colours(tmp_path, monkeypatch):
    scalars = _rate_scalars([{"pkey": f"protein{i:02d}"} for i in range(12)])
    legend, _ = _run_rate_constants(monkeypatch, tmp_path, _rate_scenario(), scalars)
    assert len(legend) == 12  # the colours repeat instead of raising StopIteration


def test_rate_constants_crystal_labels_round_the_percentage(tmp_path, monkeypatch):
    scenario = _rate_scenario(cdegree={"1JB0-PSI-syn-cocc": [0.29, 0.4]})
    scalars = _rate_scalars(
        [
            {"crystal_prot": "1JB0-PSI-syn-cocc", "cg": 0.29},
            {"crystal_prot": "1JB0-PSI-syn-cocc", "cg": 0.4},
        ]
    )
    legend, _ = _run_rate_constants(monkeypatch, tmp_path, scenario, scalars)
    assert legend == ["PSI, 29", "PSI, 40"]  # int(0.29 * 100) is 28


# --------------------------------------------------------------------------
# time-series file names: writer (utils) and reader (plotting)
# --------------------------------------------------------------------------


def _condition(nprot=100, mv=1.2, cg=0.4, crystal_prot="1JB0-PSI-syn-cocc"):
    return Condition(
        key=("3WU2-PSII-ThermosynVul", nprot),
        file_lst=[],
        pkey="3WU2-PSII-ThermosynVul",
        nprot=nprot,
        mv=mv,
        cg=cg,
        crystal_prot=crystal_prot if cg is not None else None,
        test="",
    )


def test_timeseries_names_are_unchanged_for_the_scenario_values():
    # the output files of the existing analyses keep their names
    pkey = "3WU2-PSII-ThermosynVul"
    assert (
        mu._timeseries_filename(_condition(100, 1.2, 0.4))
        == f"{pkey}_100_1JB0-PSI-syn-cocc_1.2_0.4.csv"
    )
    assert (
        mu._timeseries_filename(_condition(100, 1.01, 0))
        == f"{pkey}_100_1JB0-PSI-syn-cocc_1.01_0.csv"
    )
    assert mu._timeseries_filename(_condition(10, 1.2, None)) == f"{pkey}_10.csv"
    # the AIM analysis adds the variation, floats keep their decimal point
    assert (
        mu._timeseries_filename(_condition(10, 2.0, None), with_mv=True)
        == f"{pkey}_10_2.0.csv"
    )


@pytest.mark.parametrize("cg", [0, 0.2, 0.4, 0.8, 1, 2.5, 0.29])
@pytest.mark.parametrize("mv", [1.01, 1.2, 2, 2.0])
def test_timeseries_name_survives_the_csv_round_trip(tmp_path, cg, mv):
    # the plots rebuild the name from scalars.csv, where cg and mv are floats
    condition = _condition(nprot=100, mv=mv, cg=cg)
    base = {
        "pkey": condition.pkey,
        "nprot": condition.nprot,
        "crystal_prot": condition.crystal_prot,
    }
    # a second row with fractional values: as in a real sweep the columns are float
    scalars = pd.DataFrame(
        [
            {**base, "mv": condition.mv, "cg": condition.cg},
            {**base, "mv": 1.5, "cg": 0.5},
        ]
    )
    scalars.to_csv(tmp_path / "scalars.csv", index=False)
    row = pd.read_csv(tmp_path / "scalars.csv").iloc[0]
    assert _timeseries_filename_from_row(row) == mu._timeseries_filename(condition)


# --------------------------------------------------------------------------
# _plot_fpt_crystals: percentages in labels and titles
# --------------------------------------------------------------------------


def test_fpt_crystal_plot_rounds_the_percentages(tmp_path, monkeypatch):
    pkey, crystal = "3WU2-PSII-ThermosynVul", "1JB0-PSI-syn-cocc"
    scenario = ScenarioConfig(
        name="fpt",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={pkey: [10]},
        cdegree={crystal: [0.29, 0.57]},
        mv=[1.2],
    )
    (tmp_path / "timeseries").mkdir()
    (tmp_path / "picture").mkdir()
    rows = []
    for cg in (0.29, 0.57):
        row = {
            "pkey": pkey,
            "crystal_prot": crystal,
            "mv": 1.2,
            "nprot": 10,
            "cg": cg,
            "mean_coverage": 0.29,  # the title shows the coverage, too
            "first_inactive_time": 0.015,
            "first_inactive_time_ci_low": 0.014,
            "first_inactive_time_ci_high": 0.016,
        }
        rows.append(row)
        pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0],
                "Active": [1.0, 0.6, 0.3],
                "Active_ci_low": [1.0, 0.55, 0.25],
                "Active_ci_high": [1.0, 0.65, 0.35],
            }
        ).to_csv(
            tmp_path / "timeseries" / _timeseries_filename_from_row(pd.Series(row)),
            index=False,
        )

    closed = []  # the plot closes its figures; keep them to look at
    original_close = mplot.plt.close
    monkeypatch.setattr(mplot.plt, "close", lambda fig=None: closed.append(fig))
    try:
        mplot._plot_fpt_crystals(
            scenario, pd.DataFrame(rows), tmp_path, tmp_path / "picture"
        )
        ax = closed[0].axes[0]  # the Active-fraction plot is closed first
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["29", "57"]  # int(0.29 * 100) is 28, int(0.57 * 100) is 56
        assert ax.get_title() == "nprot=10 - coverage = 29%"
    finally:
        for fig in closed:
            original_close(fig)


# --------------------------------------------------------------------------
# _iter_conditions: which simulation conditions a scenario yields
# --------------------------------------------------------------------------


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")


def _conditions_scenario(**kwargs):
    return ScenarioConfig(
        name="s",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={"P": [10]},
        **kwargs,
    )


def test_fully_crystal_yields_only_the_crystal_conditions(tmp_path):
    # the regular membrane files exist as well: they must not be picked up
    _touch(
        tmp_path / "P_crystal",
        "crystal_P_hexagonal_regularTrue_1-2_a.wkt",
        "crystal_P_hexagonal_regularTrue_1-4_a.wkt",
    )
    _touch(tmp_path / "P", "x_hexagonal-1.wkt")
    scenario = _conditions_scenario(mv=[1.2, 1.4], fully_crystal=True)

    conditions = list(_iter_conditions(scenario, "P", "hexagonal", tmp_path, True))

    assert [c.key for c in conditions] == [
        ("P", "hexagonal", 1.2),
        ("P", "hexagonal", 1.4),
    ]
    assert [len(c.file_lst) for c in conditions] == [1, 1]
    assert all(c.cg is None and c.crystal_prot is None for c in conditions)


def test_fully_crystal_condition_records_its_glob_pattern(tmp_path):
    scenario = _conditions_scenario(mv=[1.2], fully_crystal=True)
    (condition,) = _iter_conditions(scenario, "P", "hexagonal", tmp_path, True)
    assert condition.test == "crystal_P_hexagonal_regularTrue_1-2*"


def test_without_cdegree_one_regular_condition(tmp_path):
    _touch(tmp_path / "P", "a_10-1.wkt", "b_10-2.wkt", "c_100-1.wkt")
    scenario = _conditions_scenario(mv=[1.2, 1.4])

    (condition,) = _iter_conditions(scenario, "P", 10, tmp_path)

    assert condition.key == ("P", 10)
    assert condition.mv == 1.2  # the first variation
    assert sorted(f.name for f in condition.file_lst) == ["a_10-1.wkt", "b_10-2.wkt"]


def test_with_cdegree_one_condition_per_variation_and_degree(tmp_path):
    _touch(
        tmp_path / "P_X_crystal",
        "m_10_X_1.2_0_0.wkt",
        "m_10_X_1.2_0.4_0.wkt",
        "m_10_X_1.4_0_0.wkt",
    )
    scenario = _conditions_scenario(
        mv=[1.2, 1.4], cdegree={"X": [0, 0.4]}, crystal_suffix="_crystal"
    )

    conditions = list(_iter_conditions(scenario, "P", 10, tmp_path))

    assert [c.key for c in conditions] == [
        ("P", 10, "X", 1.2, 0),
        ("P", 10, "X", 1.2, 0.4),
        ("P", 10, "X", 1.4, 0),
        ("P", 10, "X", 1.4, 0.4),
    ]
    assert [len(c.file_lst) for c in conditions] == [1, 1, 1, 0]


# --------------------------------------------------------------------------
# PlotRunsConfig.should_plot
# --------------------------------------------------------------------------


def test_should_plot_is_off_by_default():
    config = PlotRunsConfig()
    assert not any(config.should_plot(i, 10) for i in range(10))


def test_should_plot_enabled_without_limits_plots_all():
    config = PlotRunsConfig(enabled=True)
    assert all(config.should_plot(i, 10) for i in range(10))


def test_should_plot_max_runs_takes_the_first_runs():
    config = PlotRunsConfig(enabled=True, max_runs=3)
    assert [i for i in range(10) if config.should_plot(i, 10)] == [0, 1, 2]


def test_should_plot_indices_win_over_max_runs():
    config = PlotRunsConfig(enabled=True, indices=[2, 5], max_runs=1)
    assert [i for i in range(10) if config.should_plot(i, 10)] == [2, 5]


def test_should_plot_disabled_ignores_indices():
    config = PlotRunsConfig(enabled=False, indices=[1], max_runs=5)
    assert not config.should_plot(1, 10)


# --------------------------------------------------------------------------
# make_exp_config
# --------------------------------------------------------------------------

CYTB6F_AREA, PSII_AREA = 111.0, 222.0


def _exp_config(analysis_type, **kwargs):
    kwargs.setdefault("random_start_over_whole_membrane", True)
    scenario = ScenarioConfig(
        name="s",
        analysis_type=analysis_type,
        description=None,
        number_of_proteins={"P": [10]},
        **kwargs,
    )
    return make_exp_config(scenario, CYTB6F_AREA, PSII_AREA, n_processes=3)


def test_exp_config_shared_parameters():
    cfg = _exp_config(AnalysisType.FPT, wrap=True)
    assert cfg.replicates == 3000
    assert cfg.diff_coefficient == 3.5e9
    assert cfg.particle_radius == 5
    assert cfg.dimensions == (0, 5000)
    assert cfg.random_start is True
    assert cfg.has_ghost is True
    assert cfg.store_history is False
    assert cfg.workers == 3
    assert cfg.wrap is True
    assert cfg.lattice_resolution == 1


def test_exp_config_fpt():
    cfg = _exp_config(AnalysisType.FPT)
    assert (cfg.nsteps, cfg.save_every) == (1_000_000, 3_000)
    assert cfg.chosen_obstacles_by_area == CYTB6F_AREA
    assert cfg.steady_state_chosen_obstacle is False


def test_exp_config_rate_is_a_steady_state_run():
    cfg = _exp_config(AnalysisType.RATE)
    assert (cfg.nsteps, cfg.save_every) == (100_000, 500)
    assert cfg.chosen_obstacles_by_area == CYTB6F_AREA
    assert cfg.steady_state_chosen_obstacle is True


def test_exp_config_diffusion_has_no_chosen_obstacle():
    cfg = _exp_config(AnalysisType.DIFFUSION)
    assert (cfg.nsteps, cfg.save_every) == (7_000_000, 7_000)
    assert cfg.chosen_obstacles_by_area is None
    assert cfg.steady_state_chosen_obstacle is False


def test_exp_config_aim_passes_the_areas():
    start, aim = Point(10, 10).buffer(5), Point(0, 0).buffer(20)
    cfg = _exp_config(AnalysisType.AIM, start_area=start, aim_area=aim)
    assert (cfg.nsteps, cfg.save_every) == (1_000_000, 3_000)
    assert cfg.start_area is start
    assert cfg.aim_area is aim
    assert cfg.chosen_obstacles_by_area is None


def test_exp_config_membrane_size_sets_the_dimensions():
    cfg = _exp_config(AnalysisType.FPT, membrane_size=(0, 2500))
    assert cfg.dimensions == (0, 2500)


def test_exp_config_start_area_by_size_only_around_psii():
    around = _exp_config(AnalysisType.FPT, random_start_over_whole_membrane=False)
    comparison_size, buffer, constraint = around.start_area_by_size
    assert (comparison_size, buffer) == (PSII_AREA, 50)
    assert constraint.bounds == (0.0, 0.0, 5000.0, 5000.0)
    whole = _exp_config(AnalysisType.FPT, random_start_over_whole_membrane=True)
    assert whole.start_area_by_size is None


@pytest.mark.parametrize(
    ("analysis_type", "nsteps", "save_every"),
    [
        (AnalysisType.FPT, 10_000, 30),
        (AnalysisType.RATE, 1_000, 5),
        (AnalysisType.DIFFUSION, 70_000, 70),
        (AnalysisType.AIM, 10_000, 30),
    ],
)
def test_exp_config_lattice_resolution_scales_the_time_steps(
    analysis_type, nsteps, save_every
):
    cfg = _exp_config(analysis_type, lattice_resolution=10)
    # dt = a^2 / (4 D) grows with the squared lattice constant: fewer steps
    assert cfg.lattice_resolution == 10
    assert (cfg.nsteps, cfg.save_every) == (nsteps, save_every)
    assert cfg.particle_radius is None  # a point particle on the coarse lattice


def test_coarse_exp_config_is_accepted_by_the_experiment():
    # the config validation of ExperimentLattice must not reject a real scenario
    cfg = _exp_config(AnalysisType.DIFFUSION, lattice_resolution=10)
    cfg.shapes = [box(1000, 1000, 1100, 1100)]
    experiment = bl.ExperimentLattice(cfg)
    assert experiment.raster.shape == (1500, 1500)  # three tiles of 500 cells


# --------------------------------------------------------------------------
# rdf, nndf (known point sets), min_neigbor_distance
# --------------------------------------------------------------------------


def test_rdf_of_an_ideal_gas_is_one():
    n = 600
    g, g_std, g_sem, r = mu.rdf(_ideal_gas_with_ghosts(n, 3), 20, BOX, r_max=200)
    assert g.mean() == pytest.approx(1.0, abs=0.05)
    assert g_sem == pytest.approx(g_std / np.sqrt(n))
    assert r[0] == pytest.approx(5.0) and r[-1] == pytest.approx(195.0)


def test_rdf_counts_the_expected_number_of_neighbours():
    n = 600
    g, _, _, r = mu.rdf(_ideal_gas_with_ghosts(n, 4), 20, BOX, r_max=200)
    rho, dr = n / BOX**2, r[1] - r[0]
    neighbours = (g * rho * 2 * np.pi * r * dr).sum()
    assert neighbours == pytest.approx(rho * np.pi * 200**2, rel=0.05)


def test_rdf_peaks_at_the_lattice_spacing():
    spacing, count = 100.0, 10
    grid = np.array(
        [(i * spacing, j * spacing) for i in range(count) for j in range(count)]
    )
    shifts = [(dx * BOX, dy * BOX) for dx in (0, -1, 1) for dy in (0, -1, 1)]
    ghosts = np.vstack([grid + s for s in shifts])
    g, _, _, r = mu.rdf(ghosts, 50, BOX, r_max=250)  # bin width 5
    assert r[np.argmax(g)] == pytest.approx(spacing + 2.5)
    assert g[r < 95].max() == 0.0  # no neighbours closer than the spacing


@pytest.mark.parametrize("function", [mu.rdf, mu.nndf])
def test_r_max_above_half_the_box_is_rejected(function):
    with pytest.raises(ValueError, match="rmax"):
        function(_ideal_gas_with_ghosts(50, 0), 10, BOX, r_max=BOX / 2 + 1)


@pytest.mark.parametrize("function", [mu.rdf, mu.nndf])
def test_ghost_arrays_must_hold_nine_copies(function):
    points = _ideal_gas_with_ghosts(50, 0)[:-1]  # 449 points
    with pytest.raises(ValueError, match="multiple of 9, got 449"):
        function(points, 10, BOX, r_max=100)


def test_min_neighbour_distance_start_indices_must_hold_nine_copies():
    points = _ideal_gas_with_ghosts(50, 0)
    with pytest.raises(ValueError, match="multiple of 9, got 17"):
        mu.min_neigbor_distance(points, np.arange(17), np.arange(450), BOX, r_max=100)


@pytest.mark.parametrize("function", [mu.rdf, mu.nndf])
def test_without_ghosts_any_length_is_accepted(function):
    points = np.random.default_rng(0).uniform(0, BOX, size=(449, 2))
    function(points, 10, BOX, r_max=100, ghosts=False)


def test_min_neighbour_distance_uses_the_first_ninth_of_the_start_indices():
    points = _ideal_gas_with_ghosts(50, 0)
    distance = mu.min_neigbor_distance(
        points, np.arange(450), np.arange(450), BOX, r_max=100
    )
    assert distance.shape == (50,)


def test_nndf_of_a_square_lattice_jumps_at_the_spacing():
    spacing, count = 100.0, 10
    grid = np.array(
        [(i * spacing, j * spacing) for i in range(count) for j in range(count)]
    )
    G, _, g, _, r = mu.nndf(grid, 40, BOX, r_max=200, ghosts=False)  # bin width 5
    assert G[r < 100].max() == 0.0
    assert G[r > 100].min() == 1.0
    assert g.sum() * (r[1] - r[0]) == pytest.approx(1.0)


def test_min_neighbour_distance_between_two_species():
    # species A: indices 0-1, species B: indices 2-3
    points = np.array([[0.0, 0.0], [100.0, 0.0], [3.0, 4.0], [100.0, 12.0]])
    distance = mu.min_neigbor_distance(
        points, np.array([0, 1]), [2, 3], BOX, r_max=50, ghosts=False
    )
    assert distance == pytest.approx([5.0, 12.0])


def test_min_neighbour_distance_is_nan_without_a_neighbour_in_range():
    points = np.array([[0.0, 0.0], [400.0, 0.0]])
    distance = mu.min_neigbor_distance(
        points, np.array([0]), [1], BOX, r_max=100, ghosts=False
    )
    assert np.isnan(distance).all()


def test_min_neighbour_distance_ignores_the_point_itself():
    points = np.array([[0.0, 0.0], [30.0, 0.0]])
    distance = mu.min_neigbor_distance(
        points, np.array([0, 1]), [0, 1], BOX, r_max=100, ghosts=False
    )
    assert distance == pytest.approx([30.0, 30.0])  # not 0


def test_min_neighbour_distance_uses_only_the_original_cell_with_ghosts():
    ghosts = _ideal_gas_with_ghosts(40, 5)
    indices = np.arange(len(ghosts))
    distance = mu.min_neigbor_distance(ghosts, indices, indices, BOX, ghosts=True)
    assert len(distance) == 40
    assert not np.isnan(distance).any()  # periodic copies supply the neighbours


# --------------------------------------------------------------------------
# pipeline: metric builders, run_scenario and the FPT / rate plots
# --------------------------------------------------------------------------

TILE = 40.0
CHOSEN = box(10, 10, 14, 14)  # area 16: the chosen obstacle of the tiny experiments


@contextmanager
def _inline_ensemble():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bl, "ProcessPoolExecutor", _InlineExecutor)
        patch.setattr(bl, "as_completed", lambda futures: iter(list(futures)))
        yield


def _write_membrane(path, polygons):
    """A membrane file as the pipeline expects it: the nine periodic copies."""
    shifts = [(dx * TILE, dy * TILE) for dx in (0, -1, 1) for dy in (0, -1, 1)]
    copies = [translate(p, *s) for p in polygons for s in shifts]
    path.write_text("".join(to_wkt(p) + "\n" for p in copies))
    return path


def _tiny_config(analysis_type, **overrides):
    cfg = bl.ExperimentLatticeConfig()
    cfg.dimensions = (0, int(TILE))
    cfg.replicates = 100
    cfg.nsteps = 4_000
    cfg.save_every = 100
    cfg.diff_coefficient = 100.0
    cfg.random_start = True
    cfg.has_ghost = True
    cfg.workers = 1
    cfg.rng = np.random.default_rng(0)
    if analysis_type in (AnalysisType.FPT, AnalysisType.RATE):
        cfg.chosen_obstacles_by_area = CHOSEN.area
        cfg.steady_state_chosen_obstacle = analysis_type == AnalysisType.RATE
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _membrane_files(directory, nprot):
    """Two members per condition; the coverage grows with `nprot`."""
    directory.mkdir(parents=True, exist_ok=True)
    shapes = [CHOSEN] + ([box(25, 25, 31, 31)] * (nprot - 1))
    return [
        _write_membrane(directory / f"m{i}_{nprot}-0.wkt", shapes) for i in range(2)
    ]


def _pipeline_condition(files, nprot=1):
    return Condition(
        key=("P", nprot),
        file_lst=files,
        pkey="P",
        nprot=nprot,
        mv=1.2,
        cg=None,
        crystal_prot=None,
        test="",
    )


def _tiny_run(directory, analysis_type, **overrides):
    files = _membrane_files(directory, 1)
    cfg = _tiny_config(analysis_type, **overrides)
    with _inline_ensemble():
        run = load_and_run(_pipeline_condition(files), cfg)
    return run, _pipeline_condition(files), cfg


@pytest.fixture(scope="module")
def fpt_run(tmp_path_factory):
    return _tiny_run(tmp_path_factory.mktemp("fpt"), AnalysisType.FPT)


@pytest.fixture(scope="module")
def rate_run(tmp_path_factory):
    return _tiny_run(tmp_path_factory.mktemp("rate"), AnalysisType.RATE)


@pytest.fixture(scope="module")
def aim_run(tmp_path_factory):
    return _tiny_run(
        tmp_path_factory.mktemp("aim"),
        AnalysisType.AIM,
        start_area=Point(20, 20).buffer(3),
        aim_area=Point(20, 20).buffer(5),
    )


def test_load_and_run_without_files_returns_none():
    assert load_and_run(_pipeline_condition([]), _tiny_config(AnalysisType.FPT)) is None


def test_scalar_metrics_to_dict_joins_the_source_files():
    metrics = ScalarMetrics(
        pkey="P",
        nprot=3,
        total_prot=3.0,
        mv=1.2,
        crystal_prot="X",
        cg=0.4,
        mean_coverage=0.1,
        source_files=["a.wkt", "b.wkt"],
    )
    as_dict = metrics.to_dict()
    assert as_dict["source_files"] == "a.wkt;b.wkt"
    assert as_dict["pkey"] == "P" and as_dict["cg"] == 0.4
    assert metrics.source_files == ["a.wkt", "b.wkt"]  # the object keeps its list


def test_build_fpt_metrics_matches_the_run(fpt_run):
    run, condition, _ = fpt_run
    scalars, timeseries = build_fpt_metrics(run, condition)
    fpt = run.get_mean_fpt()
    assert scalars.first_inactive_time == fpt["Mean"]
    assert scalars.first_inactive_time_ci_low == fpt["Ci_low"]
    assert scalars.first_inactive_time_ci_high == fpt["Ci_high"]
    assert scalars.mean_coverage == pytest.approx(16 / TILE**2)
    assert scalars.total_prot == 1.0  # nine periodic copies are one protein
    assert (scalars.pkey, scalars.nprot, scalars.mv) == ("P", 1, 1.2)
    assert scalars.source_files == [str(f) for f in condition.file_lst]
    assert list(timeseries.columns) == [
        "Time",
        "Active",
        "Active_ci_low",
        "Active_ci_high",
    ]
    assert timeseries["Active"].iloc[0] == 1.0
    assert timeseries["Active"].iloc[-1] < 1.0  # walkers hit the obstacle
    assert 0 < scalars.first_inactive_time < 4_000 / (4 * 100)  # within the run time


def test_build_rate_metrics_computes_the_rate_per_particle(rate_run):
    run, condition, cfg = rate_run
    window = 5
    scalars, timeseries = build_rate_metrics(
        run, condition, replicates=cfg.replicates, steady_state_window=window
    )
    hits = run.get_mean_hits()
    interval = hits["Time"].iloc[1] - hits["Time"].iloc[0]
    rate = hits["Hits"] / interval / cfg.replicates  # per particle and time
    assert scalars.k_mean == pytest.approx(rate.iloc[-window:].mean())
    assert scalars.k_std == pytest.approx(rate.iloc[-window:].std())
    assert scalars.k_mean > 0
    assert scalars.first_inactive_time == run.get_mean_fpt()["Mean"]
    # the time series drops the t = 0 row
    assert list(timeseries.columns) == ["Time", "Hits", "Hits_ci_low", "Hits_ci_high"]
    assert len(timeseries) == len(hits) - 1
    assert timeseries["Time"].iloc[0] == hits["Time"].iloc[1]


def test_build_aim_metrics_starts_inside_the_aim_area(aim_run):
    run, condition, _ = aim_run
    scalars, timeseries = build_aim_metrics(run, condition)
    assert list(timeseries.columns) == ["Time", "Aim", "Aim_ci_low", "Aim_ci_high"]
    assert timeseries["Aim"].iloc[0] == 1.0  # the start area lies in the aim area
    # without wrapping the walkers leave the tile (the aim area exists only in the
    # central tile): the fraction inside falls towards zero
    assert timeseries["Aim"].iloc[-1] < 0.1
    assert scalars.mean_coverage == pytest.approx(16 / TILE**2)


def test_build_aim_metrics_with_wrapping_reaches_the_area_fraction(tmp_path):
    run, condition, _ = _tiny_run(
        tmp_path,
        AnalysisType.AIM,
        start_area=Point(20, 20).buffer(3),
        aim_area=Point(20, 20).buffer(5),
        wrap=True,
    )
    _, timeseries = build_aim_metrics(run, condition)
    late = timeseries["Aim"].iloc[-20:].mean()
    # periodic: the walkers fill the tile evenly, the aim disc is 4.9 % of it
    assert late == pytest.approx(np.pi * 5**2 / TILE**2, abs=0.03)


def _run_tiny_scenario(root, analysis_type, name):
    scenario = ScenarioConfig(
        name=name,
        analysis_type=analysis_type,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={"P": [1, 2]},
        steady_state_window=5,
    )
    for nprot in (1, 2):
        _membrane_files(root / "P", nprot)
    cfg = _tiny_config(analysis_type)
    with _inline_ensemble():
        scalars, rate_constants = run_scenario(scenario, cfg, root)
    return scenario, scalars, rate_constants, cfg


@pytest.fixture(scope="module")
def fpt_scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("fpt_scenario")
    return (root, *_run_tiny_scenario(root, AnalysisType.FPT, "fpt"))


@pytest.fixture(scope="module")
def rate_scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("rate_scenario")
    return (root, *_run_tiny_scenario(root, AnalysisType.RATE, "rate"))


def test_run_scenario_fpt_writes_scalars_and_timeseries(fpt_scenario):
    root, scenario, scalars, rate_constants, _ = fpt_scenario
    assert sorted(scalars["nprot"]) == [1, 2]
    coverage = scalars.set_index("nprot")["mean_coverage"]
    assert coverage[1] == pytest.approx(16 / TILE**2)
    assert coverage[2] == pytest.approx((16 + 36) / TILE**2)
    assert "k_mean" not in scalars.columns  # empty columns are dropped
    assert rate_constants == {}
    out = root / "fpt"
    assert pd.read_csv(out / "scalars.csv").shape[0] == 2
    for nprot in (1, 2):
        series = pd.read_csv(out / "timeseries" / f"P_{nprot}.csv")
        assert list(series.columns) == [
            "Time",
            "Active",
            "Active_ci_low",
            "Active_ci_high",
        ]


def test_run_scenario_rate_collects_the_rate_constants(rate_scenario):
    root, scenario, scalars, rate_constants, _ = rate_scenario
    assert set(rate_constants) == {("P", None, None)}
    entries = rate_constants[("P", None, None)]
    assert [e[0] for e in entries] == [1, 2]
    assert all(e[1] > 0 for e in entries)  # (nprot, k_mean, k_std)
    assert scalars["k_mean"].tolist() == [e[1] for e in entries]
    assert (root / "rate" / "timeseries" / "P_1.csv").exists()


def test_plot_fpt_writes_the_figures(fpt_scenario):
    root, scenario, *_ = fpt_scenario
    plot_fpt(scenario, root)
    pictures = {p.name for p in (root / "fpt" / "picture").iterdir()}
    assert {"membrane_fpt_P.png", "3d_membrane_fpt_P.png"} <= pictures


def test_plot_rate_writes_the_figures(rate_scenario):
    root, scenario, *_ = rate_scenario
    plot_rate(scenario, root)
    pictures = {p.name for p in (root / "rate" / "picture").iterdir()}
    assert {"membrane_rate_P.png", "rate_constant.png"} <= pictures


@pytest.mark.parametrize("function", [plot_fpt, plot_rate, plot_aim])
def test_plots_reject_the_wrong_analysis_type(function, tmp_path):
    scenario = _diffusion_scenario([10])
    with pytest.raises(ValueError, match="not correct plotting function"):
        function(scenario, tmp_path)


def test_fully_crystal_scenario_keeps_one_timeseries_per_mv(tmp_path):
    """Without `mv` in the name the files of different `mv` overwrite each other."""
    scenario = ScenarioConfig(
        name="crystal",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={"P": [1]},
        mv=[1.2, 1.4],
        fully_crystal=True,
    )
    (tmp_path / "P_crystal").mkdir()
    for mv in scenario.mv:
        for i in range(2):  # two members per condition
            name = f"crystal_P_1_regularTrue_{str(mv).replace('.', '-')}_{i}.wkt"
            _write_membrane(tmp_path / "P_crystal" / name, [CHOSEN])
    cfg = _tiny_config(AnalysisType.FPT)
    with _inline_ensemble():
        scalars, _ = run_scenario(scenario, cfg, tmp_path)
    series = sorted(p.name for p in (tmp_path / "crystal" / "timeseries").iterdir())
    assert series == ["P_1_1.2.csv", "P_1_1.4.csv"]
    assert sorted(scalars["mv"]) == [1.2, 1.4]


def _aim_scenario(mv, number_of_proteins):
    return ScenarioConfig(
        name="aim",
        analysis_type=AnalysisType.AIM,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins=number_of_proteins,
        mv=mv,
        fully_crystal=True,
    )


def _write_aim_timeseries(root, number_of_proteins, mv):
    directory = root / "aim" / "timeseries"
    directory.mkdir(parents=True)
    for pkey, nprots in number_of_proteins.items():
        for nprot in nprots:
            for value in mv:
                pd.DataFrame(
                    {
                        "Time": [0.0, 1.0, 2.0],
                        "Aim": [1.0, 0.6, 0.3],
                        "Aim_ci_low": [1.0, 0.5, 0.2],
                        "Aim_ci_high": [1.0, 0.7, 0.4],
                    }
                ).to_csv(
                    directory / f"{pkey}_{nprot}_{mu._mv_str(value)}.csv", index=False
                )


def test_plot_aim_writes_one_figure_per_protein_and_nprot(tmp_path):
    proteins = {"P": ["hexagonal", "square"], "Q": ["hexagonal"]}
    mv = [1.2, 2]  # 2 is read as `2.0`
    _write_aim_timeseries(tmp_path, proteins, mv)
    plot_aim(_aim_scenario(mv, proteins), tmp_path)
    pictures = {p.name for p in (tmp_path / "aim" / "picture").iterdir()}
    assert pictures == {
        "membrane_aim_P_hexagonal.png",
        "membrane_aim_P_square.png",
        "membrane_aim_Q_hexagonal.png",
    }


def test_plot_aim_closes_its_figures(tmp_path):
    proteins = {"P": ["hexagonal", "square"]}
    _write_aim_timeseries(tmp_path, proteins, [1.2])
    mplot.plt.close("all")
    plot_aim(_aim_scenario([1.2], proteins), tmp_path)
    assert mplot.plt.get_fignums() == []


def test_plot_aim_more_mv_than_colours(tmp_path):
    proteins = {"P": ["hexagonal"]}
    mv = [1.0 + 0.1 * i for i in range(len(COLORBLIND_PALETTE) + 2)]
    _write_aim_timeseries(tmp_path, proteins, mv)
    plot_aim(_aim_scenario(mv, proteins), tmp_path)
    assert (tmp_path / "aim" / "picture" / "membrane_aim_P_hexagonal.png").exists()


def test_figure_format_defaults_to_png(tmp_path):
    proteins = {"P": ["hexagonal"]}
    _write_aim_timeseries(tmp_path, proteins, [1.2])
    plot_aim(_aim_scenario([1.2], proteins), tmp_path)
    pictures = {p.name for p in (tmp_path / "aim" / "picture").iterdir()}
    assert pictures == {"membrane_aim_P_hexagonal.png"}


@pytest.mark.parametrize("figure_format", ["png", "pdf", "svg", "eps"])
def test_plot_aim_writes_the_requested_figure_format(tmp_path, figure_format):
    proteins = {"P": ["hexagonal"]}
    _write_aim_timeseries(tmp_path, proteins, [1.2])
    scenario = _aim_scenario([1.2], proteins)
    scenario.figure_format = figure_format
    plot_aim(scenario, tmp_path)
    pictures = {p.name for p in (tmp_path / "aim" / "picture").iterdir()}
    assert pictures == {f"membrane_aim_P_hexagonal.{figure_format}"}


def test_figure_format_keeps_dots_in_the_file_name(tmp_path):
    fig = mplot.plt.figure()
    scenario = _aim_scenario([1.2], {"P": ["hexagonal"]})
    scenario.figure_format = "svg"
    mplot._save_figure(fig, tmp_path / "name_mv1.2", scenario)
    mplot.plt.close(fig)
    assert [p.name for p in tmp_path.iterdir()] == ["name_mv1.2.svg"]


def test_unknown_figure_format_is_rejected():
    with pytest.raises(ValueError, match="figure_format must be one of"):
        ScenarioConfig(
            name="s",
            analysis_type=AnalysisType.AIM,
            description=None,
            random_start_over_whole_membrane=True,
            number_of_proteins={"P": [1]},
            figure_format="jpg",
        )


def _crystal_outputs(root, pkey, crystal):
    """Scalars and time series for the crystal FPT and rate plots, one panel."""
    (root / "timeseries").mkdir()
    (root / "picture").mkdir()
    rows = []
    for cg in (0.29, 0.57):
        row = {
            "pkey": pkey,
            "crystal_prot": crystal,
            "mv": 1.2,
            "nprot": 10,
            "cg": cg,
            "mean_coverage": 0.29,
            "first_inactive_time": 0.015,
            "first_inactive_time_ci_low": 0.014,
            "first_inactive_time_ci_high": 0.016,
        }
        rows.append(row)
        pd.DataFrame(
            {
                "Time": [0.0, 1.0, 2.0],
                "Active": [1.0, 0.6, 0.3],
                "Active_ci_low": [1.0, 0.55, 0.25],
                "Active_ci_high": [1.0, 0.65, 0.35],
                "Hits": [0.0, 2.0, 1.0],
                "Hits_ci_low": [0.0, 1.5, 0.5],
                "Hits_ci_high": [0.0, 2.5, 1.5],
            }
        ).to_csv(
            root / "timeseries" / _timeseries_filename_from_row(pd.Series(row)),
            index=False,
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", ["_plot_fpt_crystals", "_plot_rate_crystals"])
def test_crystal_plots_leave_no_figure_open(tmp_path, name):
    pkey, crystal = "3WU2-PSII-ThermosynVul", "1JB0-PSI-syn-cocc"
    scenario = ScenarioConfig(
        name="s",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={pkey: [10]},
        cdegree={crystal: [0.29, 0.57]},
        mv=[1.2],
    )
    scalars = _crystal_outputs(tmp_path, pkey, crystal)
    mplot.plt.close("all")
    getattr(mplot, name)(scenario, scalars, tmp_path, tmp_path / "picture")
    assert len(list((tmp_path / "picture").iterdir())) > 0
    assert mplot.plt.get_fignums() == []


def test_plot_diffusion_labels_an_unknown_protein_with_its_key(tmp_path, monkeypatch):
    nprots = [10, 20, 30]
    _write_diffusion_outputs(tmp_path, nprots, [0.05, 0.10, 0.15], [0.95, 0.90, 0.85])
    monkeypatch.setattr(mplot, "LABEL_DICT", {})  # PKEY is now unknown
    closed = []
    original_close = mplot.plt.close
    monkeypatch.setattr(mplot.plt, "close", lambda fig=None: closed.append(fig))
    try:
        plot_diffusion(_diffusion_scenario(nprots), tmp_path)
        summary = closed[-1]  # the coverage / protein summary is closed last
        for ax in summary.axes:
            assert len(ax.collections) > 0  # the points are drawn
        legend = summary.axes[1].get_legend().get_texts()
        assert [t.get_text() for t in legend] == [PKEY]
    finally:
        for fig in closed:
            original_close(fig)


def test_more_proteins_than_colours_are_still_plotted(tmp_path, monkeypatch):
    n_extra = len(COLORBLIND_PALETTE) + 1
    nprots = [10, 20]
    out = tmp_path / "diff"
    (out / "timeseries").mkdir(parents=True)
    rows = []
    pkeys = [f"P{i}" for i in range(n_extra)]
    for pkey in pkeys:
        for n, cov, d in zip(nprots, [0.05, 0.10], [0.95, 0.90], strict=True):
            rows.append(
                {
                    "pkey": pkey,
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
                }
            ).to_csv(out / "timeseries" / f"{pkey}_{n}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "scalars.csv", index=False)
    scenario = ScenarioConfig(
        name="diff",
        analysis_type=AnalysisType.DIFFUSION,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins=dict.fromkeys(pkeys, nprots),
    )
    plot_diffusion(scenario, tmp_path)
    assert (out / "picture" / "membrane_diffusion_last_D_dist.png").exists()


def test_rate_axis_label_names_what_is_plotted():
    fig, ax = mplot.plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    mplot._finalise_rate_ax(ax, _aim_scenario([1.2], {"P": ["hexagonal"]}))
    assert ax.get_ylabel() == "Hits per save interval"
    mplot.plt.close(fig)


def test_run_scenario_leaves_the_callers_config_unchanged_when_fully_crystal(tmp_path):
    scenario = ScenarioConfig(
        name="crystal",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={"P": [1]},
        mv=[1.2],
        fully_crystal=True,
    )
    (tmp_path / "P_crystal").mkdir()
    for i in range(2):
        _write_membrane(
            tmp_path / "P_crystal" / f"crystal_P_1_regularTrue_1-2_{i}.wkt", [CHOSEN]
        )
    cfg = _tiny_config(AnalysisType.FPT)
    assert cfg.has_ghost is True
    assert cfg.shift_origin is False
    with _inline_ensemble():
        scalars, _ = run_scenario(scenario, cfg, tmp_path)
    assert len(scalars) == 1
    assert cfg.has_ghost is True
    assert cfg.shift_origin is False


def test_a_condition_without_files_is_skipped_without_a_run(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("load_and_run must not be called without files")

    monkeypatch.setattr("membrane_analysis.pipeline.load_and_run", fail)
    scenario = _conditions_scenario()
    scalar_list, rate_constants = _run_one_nprot(
        scenario,
        _tiny_config(AnalysisType.FPT),
        tmp_path,
        tmp_path,
        "P",
        10,
        [],
        {},
    )
    assert scalar_list == []
    assert rate_constants == {}


def test_scalar_metrics_keep_a_string_nprot_and_no_crystal_fields():
    row = ScalarMetrics(
        pkey="P",
        nprot="hexagonal",
        total_prot=1,
        mv=1.2,
        crystal_prot=None,
        cg=None,
        mean_coverage=0.1,
    ).to_dict()
    assert row["nprot"] == "hexagonal"
    assert row["crystal_prot"] is None
    assert row["cg"] is None


def test_make_exp_config_seed_makes_the_random_stream_reproducible():
    scenario = ScenarioConfig(
        name="s",
        analysis_type=AnalysisType.FPT,
        description=None,
        random_start_over_whole_membrane=True,
        number_of_proteins={"P": [10]},
    )

    def draw(seed):
        cfg = make_exp_config(scenario, CYTB6F_AREA, PSII_AREA, 3, seed=seed)
        return list(cfg.rng.integers(0, 1_000_000, size=5))

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
    assert _exp_config(AnalysisType.FPT).rng is None


def test_start_area_box_follows_the_membrane_size():
    cfg = _exp_config(
        AnalysisType.FPT,
        random_start_over_whole_membrane=False,
        membrane_size=(0, 2500),
    )
    assert cfg.start_area_by_size[2].bounds == (0, 0, 2500, 2500)


def test_start_area_box_defaults_to_5000():
    cfg = _exp_config(AnalysisType.FPT, random_start_over_whole_membrane=False)
    assert cfg.start_area_by_size[2].bounds == (0, 0, 5000, 5000)


def _start_area_bounds(constraint):
    cfg = _tiny_config(AnalysisType.FPT)
    shifts = [(dx * TILE, dy * TILE) for dx in (0, -1, 1) for dy in (0, -1, 1)]
    cfg.shapes = [translate(CHOSEN, *s) for s in shifts]
    cfg.chosen_obstacles_by_area = None
    cfg.steady_state_chosen_obstacle = False
    cfg.random_start = True
    cfg.start_area_by_size = (CHOSEN.area, 5, constraint)
    return cm.brownian_lattice.ExperimentLattice(cfg)._config.start_area.bounds


def test_start_area_stays_in_the_original_tile_with_a_tile_sized_box():
    _, _, max_x, max_y = _start_area_bounds(box(0, 0, TILE, TILE))
    assert max_x == pytest.approx(19.0)  # the shape ends at 14, the buffer is 5
    assert max_y == pytest.approx(19.0)


def test_start_area_reaches_the_copies_with_an_oversized_box():
    _, _, max_x, _ = _start_area_bounds(box(0, 0, 5000, 5000))
    assert max_x > TILE
