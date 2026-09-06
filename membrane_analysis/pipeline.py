from __future__ import annotations

import gc
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cyanomembranes as cm

from .config import COLORBLIND_PALETTE, AnalysisType, Condition, ScenarioConfig
from .plotting import plot_individual_runs
from .utils import _timeseries_filename


def _iter_conditions(
    scenario: ScenarioConfig,
    pkey: str,
    nprot: int,
    out_root: Path,
    fully_crystal: bool = False,
) -> Iterator[Condition]:
    """Yields one Conditions per simulation condition to process."""

    if fully_crystal:
        for mv in scenario.mv:
            str_mv = str(mv).replace(".", "-")
            file_lst = list(
                (out_root / (pkey + "_crystal")).glob(
                    f"crystal_{pkey}_{nprot}_regularTrue_{str_mv}*"
                )
            )
            yield Condition(
                key=(pkey, nprot, mv),
                file_lst=file_lst,
                pkey=pkey,
                nprot=nprot,
                mv=mv,
                cg=None,
                crystal_prot=None,
                test="crystal_{pkey}_{nprot}_regularTrue_{mv}*",
            )
    if scenario.cdegree is None:
        file_lst = list((out_root / pkey).glob(f"*_{nprot}-*.wkt"))
        yield Condition(
            key=(pkey, nprot),
            file_lst=file_lst,
            pkey=pkey,
            nprot=nprot,
            mv=scenario.mv[0],
            cg=None,
            crystal_prot=None,
            test=f"*_{nprot}-*.wkt",
        )
    else:
        for crystal_prot, cg_lst in scenario.cdegree.items():
            subdir = out_root / f"{pkey}_{crystal_prot}{scenario.crystal_suffix}"
            for mv in scenario.mv:
                for cg in cg_lst:
                    file_lst = list(
                        subdir.glob(f"*_{nprot}_{crystal_prot}_{mv}_{cg}_*.wkt")
                    )
                    yield Condition(
                        key=(pkey, nprot, crystal_prot, mv, cg),
                        file_lst=file_lst,
                        pkey=pkey,
                        nprot=nprot,
                        mv=mv,
                        cg=cg,
                        crystal_prot=crystal_prot,
                        test=f"*_{nprot}_{crystal_prot}_{mv}_{cg}_*.wkt",
                    )


def load_and_run(
    condition: Condition, exp_config
) -> cm.brownian_lattice.EnsembleExperimentLatticeRun:
    """Loads files, runs enesmble, returns raw results"""
    if len(condition.file_lst) == 0:
        return None

    print(f"Processing {condition.key}: {len(condition.file_lst)} files")

    ens_exp = cm.brownian_lattice.EnsembleExperimentLattice(
        condition.file_lst, exp_config
    )
    run = ens_exp.run()
    del ens_exp
    gc.collect()
    return run


@dataclass
class ScalarMetrics:
    pkey: str
    nprot: float
    total_prot: float
    mv: float
    crystal_prot: str
    cg: float
    mean_coverage: float
    last_D_dist: float | None = None
    last_D_dist_ci_low: float | None = None
    last_D_dist_ci_high: float | None = None
    first_inactive_time: float | None = None
    first_inactive_time_ci_low: float | None = None
    first_inactive_time_ci_high: float | None = None
    k_mean: float | None = None
    k_std: float | None = None
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = deepcopy(self.__dict__)
        d["source_files"] = ";".join(self.source_files)
        return d


def build_aim_metrics(run, condition) -> tuple[ScalarMetrics, pd.DataFrame]:
    """Extracts AIM metrics from a completed run"""

    df = run.mean_over_runs()
    coverage = run.get_mean_coverage()

    _, nprot, *rest = condition.key

    scalars = ScalarMetrics(
        pkey=condition.pkey,
        nprot=nprot,
        total_prot=len(run.runs[0].shapes) / 9,
        mv=condition.mv,
        crystal_prot=condition.crystal_prot,
        cg=condition.cg,
        mean_coverage=coverage,
        source_files=[str(f) for f in condition.file_lst],
    )

    timeseries = df[["Time", "Aim", "Aim_ci_low", "Aim_ci_high"]].copy()

    return scalars, timeseries  # type: ignore


def build_fpt_metrics(run, condition) -> tuple[ScalarMetrics, pd.DataFrame]:
    """Extracts FPT metrics from a completed run"""

    df = run.mean_over_runs()
    coverage = run.get_mean_coverage()
    fpt = run.get_mean_fpt()

    _, nprot, *rest = condition.key

    scalars = ScalarMetrics(
        pkey=condition.pkey,
        nprot=nprot,
        total_prot=len(run.runs[0].shapes) / 9,
        mv=condition.mv,
        crystal_prot=condition.crystal_prot,
        cg=condition.cg,
        mean_coverage=coverage,
        first_inactive_time=fpt["Mean"],
        first_inactive_time_ci_low=fpt["Ci_low"],
        first_inactive_time_ci_high=fpt["Ci_high"],
        source_files=[str(f) for f in condition.file_lst],
    )

    timeseries = df[["Time", "Active", "Active_ci_low", "Active_ci_high"]].copy()

    return scalars, timeseries  # type: ignore


def build_rate_metrics(
    run, condition, replicates: int, steady_state_window: int
) -> tuple[ScalarMetrics, pd.DataFrame]:
    """Extracts rate constant metrics from completed run"""
    coverage = run.get_mean_coverage()
    df = run.get_mean_hits().iloc[1:]

    df_hits = run.get_mean_hits()

    hits_per_int = df_hits.drop("Time", axis=1)["Hits"]
    time = df_hits["Time"]
    dt = time.diff().iloc[1]
    flux = hits_per_int / dt
    rate_series = flux / replicates
    k_mean = rate_series.iloc[-steady_state_window:].mean()
    k_std = rate_series.iloc[-steady_state_window:].std()

    _, nprot, *rest = condition.key

    fpt = run.get_mean_fpt()

    scalars = ScalarMetrics(
        pkey=condition.pkey,
        nprot=nprot,
        total_prot=len(run.runs[0].shapes) / 9,
        mv=condition.mv,
        crystal_prot=condition.crystal_prot,
        cg=condition.cg,
        mean_coverage=coverage,
        k_mean=k_mean,
        k_std=k_std,
        source_files=[str(f) for f in condition.file_lst],
        first_inactive_time=fpt["Mean"],
    )

    timeseries = df[["Time", "Hits", "Hits_ci_low", "Hits_ci_high"]].copy()

    return scalars, timeseries


def build_diffusion_metrics(
    run, condition, exp_config
) -> tuple[ScalarMetrics, pd.DataFrame]:
    """Extracts FPT metrics from a completed run"""

    df = run.mean_over_runs()
    coverage = run.get_mean_coverage()

    run_dif = run.get_mean_diff_coefficients_dist()
    norm_diff_dist = (
        exp_config.lattice_resolution**2
        * run_dif["D_dist"]
        / (4 * exp_config.diff_coefficient)
    )
    norm_diff_dist_ci_low = (
        exp_config.lattice_resolution**2
        * run_dif["D_dist_ci_low"]
        / (4 * exp_config.diff_coefficient)
    )
    norm_diff_dist_ci_high = (
        exp_config.lattice_resolution**2
        * run_dif["D_dist_ci_high"]
        / (4 * exp_config.diff_coefficient)
    )
    norm_diff_dist.iloc[0] = 1.0
    sqrt_msd = np.sqrt(run_dif["MSD"])

    last_D_dist = norm_diff_dist.iloc[-1]
    last_D_dist_ci_low = norm_diff_dist_ci_low.iloc[-1]
    last_D_dist_ci_high = norm_diff_dist_ci_high.iloc[-1]

    _, nprot, *rest = condition.key

    scalars = ScalarMetrics(
        pkey=condition.pkey,
        nprot=nprot,
        total_prot=len(run.runs[0].shapes) / 9,
        mv=condition.mv,
        crystal_prot=condition.crystal_prot,
        cg=condition.cg,
        mean_coverage=coverage,
        last_D_dist=last_D_dist,
        last_D_dist_ci_low=last_D_dist_ci_low,
        last_D_dist_ci_high=last_D_dist_ci_high,
        source_files=[str(f) for f in condition.file_lst],
    )

    timeseries = df[
        ["Time", "MSD", "MSD_ci_low", "MSD_ci_high", "MAD", "MAD_ci_low", "MAD_ci_high"]
    ].copy()
    timeseries["Sqrt_MSD"] = sqrt_msd
    timeseries["Norm_Diff_Dist"] = norm_diff_dist
    timeseries["Norm_Diff_Dist_ci_low"] = norm_diff_dist_ci_low
    timeseries["Norm_Diff_Dist_ci_high"] = norm_diff_dist_ci_high

    return scalars, timeseries  # type: ignore


def _cleanup(run) -> None:
    """Clean up run"""
    del run
    gc.collect()


def _save_condition_outputs(
    scalar_lst: list[ScalarMetrics],
    scalars: ScalarMetrics,
    timeseries: pd.DataFrame,
    condition: Condition,
    new_out: Path,
    with_mv: bool = False,
) -> None:

    ts_dir = new_out / "timeseries"
    ts_dir.mkdir(exist_ok=True)

    timeseries.to_csv(ts_dir / _timeseries_filename(condition, with_mv), index=False)

    scalar_lst.append(scalars)


def _save_total(
    df_scalars: pd.DataFrame,
    rate_constants: dict,
    scenario: ScenarioConfig,
    new_out: Path,
) -> None:
    df_scalars.to_csv(new_out / "scalars.csv", index=False)


def _run_one_nprot(
    scenario: ScenarioConfig,
    exp_config,
    out_root: Path,
    new_out: Path,
    pkey: str,
    nprot: int,
    scalar_list: list[ScalarMetrics],
    rate_constants: dict,
) -> tuple[list[ScalarMetrics], dict]:

    iterator = _iter_conditions(
        scenario, pkey, nprot, out_root, fully_crystal=scenario.fully_crystal
    )

    if scenario.fully_crystal:
        exp_config.has_ghost = False
        exp_config.shift_origin = True

    for condition in iterator:
        if len(condition.file_lst) == 0:
            print(f"No files for {condition.key}, skipping")

        print(condition.file_lst)

        run = load_and_run(condition, exp_config)
        if run is None:
            continue

        if scenario.plot_runs.enabled:
            plot_individual_runs(
                run, condition, scenario, new_out, scenario.plot_runs.aim
            )

        if scenario.analysis_type == AnalysisType.FPT:
            scalars, time_series = build_fpt_metrics(run, condition)
            _save_condition_outputs(
                scalar_lst=scalar_list,
                scalars=scalars,
                timeseries=time_series,
                condition=condition,
                new_out=new_out,
            )

        elif scenario.analysis_type == AnalysisType.RATE:
            scalars, time_series = build_rate_metrics(
                run,
                condition,
                replicates=exp_config.replicates,
                steady_state_window=scenario.steady_state_window,
            )
            _save_condition_outputs(
                scalar_lst=scalar_list,
                scalars=scalars,
                timeseries=time_series,
                condition=condition,
                new_out=new_out,
            )

            rate_constants.setdefault(
                (pkey, condition.crystal_prot, condition.cg), []
            ).append((scalars.nprot, scalars.k_mean, scalars.k_std))

        elif scenario.analysis_type == AnalysisType.DIFFUSION:
            scalars, time_series = build_diffusion_metrics(run, condition, exp_config)
            _save_condition_outputs(
                scalar_lst=scalar_list,
                scalars=scalars,
                timeseries=time_series,
                condition=condition,
                new_out=new_out,
            )

        elif scenario.analysis_type == AnalysisType.AIM:
            scalars, time_series = build_aim_metrics(run, condition)
            _save_condition_outputs(
                scalar_lst=scalar_list,
                scalars=scalars,
                timeseries=time_series,
                condition=condition,
                new_out=new_out,
                with_mv=True,
            )

        _cleanup(run)

    return scalar_list, rate_constants


def run_scenario(
    scenario: ScenarioConfig, exp_config, out_root: Path
) -> tuple[pd.DataFrame, dict]:
    """Master pipeline for one scenario x analysis type combination"""
    new_out = out_root / scenario.name
    new_out.mkdir(parents=True, exist_ok=True)

    scalar_list = []
    rate_constants = {}

    for pkey, nprot_lst in scenario.number_of_proteins.items():
        for nprot in nprot_lst:
            scalar_list, rate_constants = _run_one_nprot(
                scenario=scenario,
                exp_config=exp_config,
                out_root=out_root,
                new_out=new_out,
                pkey=pkey,
                nprot=nprot,
                scalar_list=scalar_list,
                rate_constants=rate_constants,
            )

    df_scalars = pd.DataFrame([s.to_dict() for s in scalar_list])
    df_scalars = df_scalars.dropna(
        axis=1,
        how="all",
    )
    _save_total(df_scalars, rate_constants, scenario, new_out)

    return df_scalars, rate_constants
