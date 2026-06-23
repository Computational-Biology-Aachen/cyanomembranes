import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from membrane_analysis.config import (
    COLORBLIND_PALETTE,
    LABEL_DICT,
    MARKER_LST,
    AnalysisType,
)

from .config import Condition
from .utils import _timeseries_filename


def plot_individual_runs(run, condition, scenario, new_out, aim=False):
    runs_dir = new_out / "individual_runs"
    runs_dir.mkdir(exist_ok=True)

    for i, single_run in enumerate(run.runs):
        if not scenario.plot_runs.should_plot(i, len(run.runs)):
            continue

        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        single_run.plot_run(ax=ax, aim_area=aim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\mathrm{\AA}$")
        ax.set_ylabel(r"$\mathrm{\AA}$")
        fig.tight_layout()
        if scenario.fully_crystal:
            with_mv = True
        else:
            with_mv = False
        fig.savefig(
            runs_dir / f"{_timeseries_filename(condition, with_mv)}_run{i}.png", dpi=300
        )
        plt.close(fig)


def plot_diffusion(scenario, out_root):

    new_out = out_root / scenario.name
    pic_out = new_out / "picture"
    pic_out.mkdir(parents=True, exist_ok=True)
    if not new_out.exists():
        msg = f"no directiory {new_out}"
        raise ValueError(msg)

    if scenario.analysis_type != AnalysisType.DIFFUSION:
        msg = "not correct plotting function for analysis type diffusion"
        raise ValueError(msg)

    scalars = pd.read_csv(new_out / "scalars.csv")

    colors_palette = iter(COLORBLIND_PALETTE)
    markers = iter(MARKER_LST)
    fig1, axes1 = plt.subplots(1, 2, figsize=(7, 3))
    parameter_rows = []
    for pkey, nprot in scenario.number_of_proteins.items():
        current_prot = scalars[scalars.pkey == pkey]
        x = current_prot["mean_coverage"].to_numpy() * 100
        x2 = current_prot["total_prot"].to_numpy()
        y = current_prot["last_D_dist"].to_numpy()
        yerr = np.vstack(
            (
                y - current_prot["last_D_dist_ci_low"],
                current_prot["last_D_dist_ci_high"] - y,
            )
        )

        lm = LinearRegression().fit(x.reshape(-1, 1), y)
        x_fit = np.linspace(0, max(x), len(x))
        y_fit_curve = lm.predict(x_fit.reshape(-1, 1))
        r2 = lm.score(x.reshape(-1, 1), y)

        parameter_rows.append(
            {
                "pkey": pkey,
                "type": "coverage",
                "slope": lm.coef_[0],
                "intercept": lm.intercept_,
                "r2": r2,
            }
        )

        lm2 = LinearRegression().fit(x2.reshape(-1, 1), y)
        x_fit2 = np.linspace(0, max(x2), len(x2))
        y_fit_curve2 = lm2.predict(x_fit2.reshape(-1, 1))
        r22 = lm.score(x.reshape(-1, 1), y)

        parameter_rows.append(
            {
                "pkey": pkey,
                "type": "proteins",
                "slope": lm2.coef_[0],
                "intercept": lm2.intercept_,
                "r2": r22,
            }
        )

        try:

            current_color = next(colors_palette)
            current_marker = next(markers)
            axes1[1].scatter(
                x,
                y,
                s=25,
                edgecolors=current_color,
                facecolor="white",
                marker=current_marker,
                linewidth=0.6,
                zorder=3,
                label=f"{LABEL_DICT[pkey]}",
            )
            axes1[1].errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=current_color,
                capsize=5,
                elinewidth=0.5,
                capthick=0.5,
            )

            axes1[1].plot(
                x_fit,
                y_fit_curve,
                linestyle="--",
                linewidth=1,
                color=current_color,
                zorder=2,
            )

            axes1[0].scatter(
                x2,
                y,
                s=25,
                edgecolors=current_color,
                facecolor="white",
                marker=current_marker,
                linewidth=0.6,
                zorder=3,
                label=f"{LABEL_DICT[pkey]}",
            )
            axes1[0].errorbar(
                x2,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=current_color,
                capsize=5,
                elinewidth=0.5,
                capthick=0.5,
            )

            axes1[0].plot(
                x_fit2,
                y_fit_curve2,
                linestyle="--",
                linewidth=1,
                color=current_color,
                zorder=2,
            )
        except:
            pass

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        norm = colors.Normalize(vmin=min(nprot), vmax=max(nprot))
        cmap = mlp.colormaps["viridis"]

        for _idx, row in current_prot.iterrows():
            n = row.nprot
            current_color2 = cmap(norm(n))
            ts_df = pd.read_csv(new_out / "timeseries" / f"{pkey}_{int(n)}.csv")
            coverage = row["mean_coverage"] * 100
            ax2.plot(
                ts_df["Sqrt_MSD"].to_numpy(),
                ts_df["Norm_Diff_Dist"].to_numpy(),
                label=f"r = {coverage:.2f}%",
                linewidth=2,
                alpha=0.7,
                color=current_color2,
            )

            ax2.fill_between(
                ts_df["Sqrt_MSD"].to_numpy(),
                ts_df["Norm_Diff_Dist_ci_low"].to_numpy(),
                ts_df["Norm_Diff_Dist_ci_high"].to_numpy(),
                alpha=0.3,
                color=ax2.lines[-1].get_color(),
            )

        ax2.set_xlabel(r"$\langle r \rangle$ (Å)", fontsize=12)
        ax2.set_ylabel(r"$D(r) / D_0$", fontsize=12)
        ax2.tick_params(labelsize=10)
        ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)  # noqa: FBT003
        ax2.legend(loc="lower right")
        # ax2.set_xlim(0, 500)
        for spines in ax2.spines.values():
            spines.set_linewidth(0.5)
        fig2.tight_layout()
        fig2.savefig(pic_out / f"membrane_diffusion_{pkey}.png", dpi=300)
        plt.close(fig2)

    fig1.tight_layout()
    axes1[1].set_xlabel("% Coverage", fontsize=8)
    axes1[1].set_ylabel(r"$D_\infty$", fontsize=8)
    axes1[1].tick_params(axis="both", which="major", labelsize=7)
    axes1[1].legend(fontsize=7)

    axes1[0].set_xlabel("# Proteins", fontsize=8)
    axes1[0].set_ylabel(r"$D_\infty$", fontsize=8)
    axes1[0].tick_params(axis="both", which="major", labelsize=7)
    axes1[0].legend(fontsize=7)

    fig1.tight_layout()
    fig1.savefig(pic_out / "membrane_diffusion_last_D_dist.png", dpi=300)
    plt.close(fig1)

    pd.DataFrame(parameter_rows).to_csv(pic_out / "reg_param_df.csv")


def plot_3d_fpt(Active_df, scenario):

    coverages = Active_df.columns[1:].astype(float) * 100
    times = np.array(Active_df["Time"] * scenario.time_scale)
    Z_raw = Active_df.drop("Time", axis=1).to_numpy().T
    coverage_new = np.linspace(coverages.min(), coverages.max(), 300)
    Z = np.zeros((len(coverage_new), len(times)))

    for i in range(len(times)):
        f = interp1d(coverages, Z_raw[:, i], fill_value="extrapolate")
        Z[:, i] = f(coverage_new)

    T, C = np.meshgrid(times, coverage_new)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    _ = ax.plot_surface(
        T,
        C,
        Z,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.2,
        antialiased=True,
        alpha=0.95,
    )

    for cov, active in zip(coverages, Z_raw):
        ax.plot(times, np.full_like(times, cov), active, color="grey", linewidth=2)

    ax.set_xlabel(scenario.time_label, size=12, labelpad=16)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("% Mean Coverage", size=12)
    ax.set_zlabel("Reduced fraction", size=12)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_ylim(coverages.min(), coverages.max())
    ax.view_init(elev=20, azim=-45)
    fig.tight_layout()
    return fig, ax


def _finalise_fpt_ax(ax1, scenario):
    ax1.set_xlabel(scenario.time_label, fontsize=8)
    ax1.set_ylabel("Reduced fraction %", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)  # noqa: FBT003
    ax1.legend(title="% in array", loc="upper right")
    for spines in ax1.spines.values():
        spines.set_linewidth(0.5)


def _plot_fpt_no_crystals(scenario, scalars, new_out, pic_out):
    for pkey, nprot in scenario.number_of_proteins.items():
        current_prot = scalars[scalars.pkey == pkey]
        Active_dict = {}
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        norm = colors.Normalize(vmin=min(nprot), vmax=max(nprot))
        cmap = mlp.colormaps["viridis"]
        for _idx, row in current_prot.iterrows():
            n = row.nprot
            current_color2 = cmap(norm(n))
            ts_df = pd.read_csv(new_out / "timeseries" / f"{pkey}_{int(n)}.csv")
            coverage = row["mean_coverage"]
            ax1.plot(
                ts_df["Time"] * scenario.time_scale,
                ts_df["Active"],
                label=f"r = {coverage:.2f}",
                linewidth=2,
                alpha=0.7,
                color=current_color2,
            )
            ax1.fill_between(
                ts_df["Time"] * scenario.time_scale,
                ts_df["Active_ci_low"],
                ts_df["Active_ci_high"],
                alpha=0.3,
                color=ax1.lines[-1].get_color(),
            )
            Active_dict[coverage] = ts_df[["Time", "Active"]]

        _finalise_fpt_ax(ax1, scenario)
        fig1.tight_layout()
        fig1.savefig(pic_out / f"membrane_fpt_{pkey}.png", dpi=300)
        plt.close(fig1)

        Active_dict = dict(sorted(Active_dict.items()))
        Active_df = pd.DataFrame(
            {key: df.set_index("Time")["Active"] for key, df in Active_dict.items()}
        )
        Active_df = Active_df.reset_index()
        fig2, _ax2 = plot_3d_fpt(Active_df, scenario)
        fig2.savefig(pic_out / f"3d_membrane_fpt_{pkey}.png", dpi=300)
        plt.close(fig2)


def _timeseries_filename_from_row(row):
    parts = [row["pkey"], str(int(row["nprot"]))]
    if "crystal_prot" in row and not pd.isna(row["crystal_prot"]):
        parts.append(row["crystal_prot"])
    parts.append(str(row["mv"]))
    if "cg" in row and not pd.isna(row["cg"]):
        if row["cg"] == 0.0:
            parts.append(str(0))
        else:
            parts.append(str(row["cg"]))
    return "_".join(parts) + ".csv"


def _plot_fpt_crystals(scenario, scalars, new_out, pic_out):
    for pkey, nprot_lst in scenario.number_of_proteins.items():
        for crystal_prot in scenario.cdegree:
            for mv in scenario.mv:
                # --- filter to (pkey, crystal_prot, mv)
                mask = (
                    (scalars["pkey"] == pkey)
                    & (scalars["crystal_prot"] == crystal_prot)
                    & (scalars["mv"] == mv)
                )

                subset = scalars[mask]
                if subset.empty:
                    continue

                n_panels = len(nprot_lst)
                fig, axes = plt.subplots(
                    1, n_panels, figsize=(3 * n_panels, 3), sharey=False
                )

                if n_panels == 1:
                    axes = [axes]

                for ax, nprot in zip(axes, nprot_lst):
                    nprot_rows = subset[subset["nprot"] == nprot]
                    coverage = nprot_rows["mean_coverage"].round(3).unique()[0]
                    color_iter = iter(COLORBLIND_PALETTE)

                    for _, row in nprot_rows.sort_values("cg").iterrows():
                        color = next(color_iter)
                        ts_df = pd.read_csv(
                            new_out / "timeseries" / _timeseries_filename_from_row(row)
                        )

                        ax.plot(
                            ts_df["Time"] * scenario.time_scale,
                            ts_df["Active"],
                            label=f"{int(row['cg']*100)}",
                            linewidth=2,
                            alpha=0.7,
                            color=color,
                        )

                        ax.fill_between(
                            ts_df["Time"] * scenario.time_scale,
                            ts_df["Active_ci_low"],
                            ts_df["Active_ci_high"],
                            alpha=0.3,
                            color=color,
                        )

                    ax.set_title(
                        f"nprot={int(nprot)} - coverage = {int(coverage*100)}%",
                        fontsize=8,
                    )
                    _finalise_fpt_ax(ax, scenario)
                fig.suptitle(
                    f"{LABEL_DICT.get(pkey, pkey)}-"
                    f"{LABEL_DICT.get(crystal_prot, crystal_prot)}, "
                    rf"$\delta$={mv}",
                    fontsize=9,
                )
                fig.tight_layout()
                fig.savefig(
                    pic_out / f"membrane_fpt_{pkey}_{crystal_prot}_mv{mv}.png", dpi=400
                )


def plot_fpt(scenario, out_root):

    new_out = out_root / scenario.name
    pic_out = new_out / "picture"
    pic_out.mkdir(parents=True, exist_ok=True)

    if scenario.analysis_type != AnalysisType.FPT:
        msg = "not correct plotting function for analysis type fpt"
        raise ValueError(msg)

    scalars = pd.read_csv(new_out / "scalars.csv")

    if scenario.cdegree is None:
        _plot_fpt_no_crystals(scenario, scalars, new_out, pic_out)
    else:
        _plot_fpt_crystals(scenario, scalars, new_out, pic_out)


def _finalise_rate_ax(ax1, scenario):
    ax1.set_xlabel(scenario.time_label, fontsize=8)
    ax1.set_ylabel("Hits %", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)  # noqa: FBT003
    ax1.legend(loc="upper left")
    for spines in ax1.spines.values():
        spines.set_linewidth(0.5)


def _plot_rate_no_crystals(scenario, scalars, new_out, pic_out):
    for pkey, nprot in scenario.number_of_proteins.items():
        current_prot = scalars[scalars.pkey == pkey]
        fig, ax = plt.subplots(figsize=(6, 4))
        norm = colors.Normalize(vmin=min(nprot), vmax=max(nprot))
        cmap = mlp.colormaps["viridis"]
        for _idx, row in current_prot.iterrows():
            n = row.nprot
            current_color = cmap(norm(n))
            ts_df = pd.read_csv(new_out / "timeseries" / f"{pkey}_{int(n)}.csv")
            coverage = row["mean_coverage"]
            ax.plot(
                ts_df["Time"] * scenario.time_scale,
                ts_df["Hits"],
                label=f"r = {coverage:.2f}",
                linewidth=2,
                alpha=0.7,
                color=current_color,
            )
            ax.fill_between(
                ts_df["Time"] * scenario.time_scale,
                ts_df["Hits_ci_low"],
                ts_df["Hits_ci_high"],
                alpha=0.3,
                color=ax.lines[-1].get_color(),
            )

        _finalise_rate_ax(ax, scenario)
        fig.tight_layout()
        fig.savefig(pic_out / f"membrane_rate_{pkey}.png", dpi=300)
        plt.close(fig)


def _plot_rate_crystals(scenario, scalars, new_out, pic_out):
    for pkey, nprot_lst in scenario.number_of_proteins.items():
        for crystal_prot in scenario.cdegree:
            for mv in scenario.mv:
                # --- filter to (pkey, crystal_prot, mv)
                mask = (
                    (scalars["pkey"] == pkey)
                    & (scalars["crystal_prot"] == crystal_prot)
                    & (scalars["mv"] == mv)
                )

                subset = scalars[mask]
                if subset.empty:
                    continue

                n_panels = len(nprot_lst)
                fig, axes = plt.subplots(
                    1, n_panels, figsize=(3 * n_panels, 4), sharey=True
                )

                if n_panels == 1:
                    axes = [axes]

                for ax, nprot in zip(axes, nprot_lst):
                    nprot_rows = subset[subset["nprot"] == nprot]
                    coverage = nprot_rows["mean_coverage"].round(3).unique()[0]
                    color_iter = iter(COLORBLIND_PALETTE)

                    for _, row in nprot_rows.sort_values("cg").iterrows():
                        color = next(color_iter)
                        ts_df = pd.read_csv(
                            new_out / "timeseries" / _timeseries_filename_from_row(row)
                        )

                        ax.plot(
                            ts_df["Time"] * scenario.time_scale,
                            ts_df["Hits"],
                            label=f"c%={row['cg']}",
                            linewidth=2,
                            alpha=0.7,
                            color=color,
                        )

                        ax.fill_between(
                            ts_df["Time"] * scenario.time_scale,
                            ts_df["Hits_ci_low"],
                            ts_df["Hits_ci_high"],
                            alpha=0.3,
                            color=color,
                        )

                    ax.set_title(
                        f"nprot={int(nprot)} - coverage = {coverage}",
                        fontsize=8,
                    )
                    _finalise_rate_ax(ax, scenario)
                fig.suptitle(
                    f"{LABEL_DICT.get(pkey, pkey)}-"
                    f"{LABEL_DICT.get(crystal_prot, crystal_prot)}, "
                    rf"$\delta$={mv}",
                    fontsize=9,
                )
                fig.tight_layout()
                fig.savefig(pic_out / f"membrane_rate_{pkey}_{crystal_prot}_mv{mv}.png")


def _plot_rate_constants(scenario, scalars, pic_out):
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    color_iter = iter(COLORBLIND_PALETTE)

    group_cols = ["crystal_prot", "cg"] if scenario.cdegree is not None else ["pkey"]

    for group_key, group in scalars.groupby(group_cols):
        group = group.sort_values("mean_coverage")
        color = next(color_iter)
        if scenario.cdegree is not None:
            crystal_prot, cg = group_key
            label = f"{LABEL_DICT.get(crystal_prot, crystal_prot)}, {int(cg*100)}"
        else:
            label = LABEL_DICT.get(group_key, group_key)

        ax.errorbar(
            group["mean_coverage"] * 100,
            group["k_mean"],
            yerr=group["k_std"],
            color=color,
            label=label,
            capsize=5,
        )

        ax.set_xlabel("% coverage", fontsize=8)
        ax.set_ylabel("Rate constant k [1/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="% in array", loc="upper left", fontsize=7)
        ax.tick_params(direction="in")
        fig.tight_layout()
        fig.savefig(pic_out / "rate_constant.png", dpi=300)
        plt.close(fig)


def plot_rate(scenario, out_root):

    new_out = out_root / scenario.name
    pic_out = new_out / "picture"
    pic_out.mkdir(parents=True, exist_ok=True)

    if scenario.analysis_type != AnalysisType.RATE:
        msg = "not correct plotting function for analysis type rate"
        raise ValueError(msg)

    scalars = pd.read_csv(new_out / "scalars.csv")

    if scenario.cdegree is None:
        _plot_rate_no_crystals(scenario, scalars, new_out, pic_out)
        _plot_rate_constants(scenario, scalars, pic_out)
    else:
        _plot_rate_crystals(scenario, scalars, new_out, pic_out)
        _plot_rate_constants(scenario, scalars, pic_out)
