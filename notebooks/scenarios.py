from shapely import Point

from membrane_analysis.config import (AnalysisType, PlotRunsConfig,
                                      ScenarioConfig)

# ----------------
# DIFFUSION ANALYSES
# ----------------

no_md_diff_random_start = ScenarioConfig(
    name="no_MD_diff_random_start",
    analysis_type=AnalysisType.DIFFUSION,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={
        "3WU2-PSII-ThermosynVul": [100, 200, 300, 400, 500, 600, 700, 800, 900],
        "4H13-cytb6f": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1100, 1400],
        "1JB0-PSI-syn-cocc": [100, 200, 300, 400, 500],
        "avg_membrane": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

no_md_diff_random_start_coarse = ScenarioConfig(
    name="no_MD_diff_random_start_coarse",
    analysis_type=AnalysisType.DIFFUSION,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={
        "3WU2-PSII-ThermosynVul": [700, 800, 900],
        "4H13-cytb6f": [1100, 1400],
        "1JB0-PSI-syn-cocc": [500],
        "avg_membrane": [90, 100, 110],
    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    lattice_resolution=10,
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)


no_md_diff = ScenarioConfig(
    name="no_MD_diff",
    analysis_type=AnalysisType.DIFFUSION,
    description=None,
    random_start_over_whole_membrane=False,
    number_of_proteins={
        "3WU2-PSII-ThermosynVul": [100, 200, 300, 400, 500, 600, 700, 800, 900],
        "avg_membrane": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

# ---------------
# FPT ANALYSES
# ----------------

no_md_fpt_random_start = ScenarioConfig(
    name="no_MD_fpt_random_start",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={
        "4H13-cytb6f": [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        "avg_membrane": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

no_md_fpt = ScenarioConfig(
    name="no_MD_fpt",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={
       "avg_membrane": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
       "4H13-cytb6f": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
       "PSII-cytbf": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 110]

    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

no_md_fpt_coarse = ScenarioConfig(
    name="no_MD_fpt_coarse",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={
        "avg_membrane": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
    lattice_resolution=10,
)


big_md_fpt_random_start = ScenarioConfig(
    name="big_MD_fpt_random_start",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

big_md_fpt = ScenarioConfig(
    name="big_MD_fpt",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

big_md_fpt_coarse = ScenarioConfig(
    name="big_MD_fpt_coarse",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
    lattice_resolution=10,
)

big_md_fpt_mv_sweep = ScenarioConfig(
    name="big_MD_fpt_mv_sweep",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
    mv=[1.01, 1.3, 1.5]
)


small_md_fpt_random_start = ScenarioConfig(
    name="small_MD_fpt_random_start",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)


small_md_fpt = ScenarioConfig(
    name="small_MD_fpt",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

# PSII domains

big_md_fpt_random_start_PSII = ScenarioConfig(
    name="big_MD_fpt_random_start_PSII",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

big_md_fpt_PSII = ScenarioConfig(
    name="big_MD_fpt_PSII",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

small_md_fpt_random_start_PSII = ScenarioConfig(
    name="small_MD_fpt_random_start_PSII",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)


small_md_fpt_PSII = ScenarioConfig(
    name="small_MD_fpt_PSII",
    analysis_type=AnalysisType.FPT,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

# -----------------------
# RATE CONSTANT ANALYSES
# -----------------------

big_md_rate = ScenarioConfig(
    name="big_MD_rate",
    analysis_type=AnalysisType.RATE,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

big_md_rate_coarse = ScenarioConfig(
    name="big_MD_rate_coarse",
    analysis_type=AnalysisType.RATE,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
    lattice_resolution=10,
)

small_md_rate = ScenarioConfig(
    name="small_MD_rate",
    analysis_type=AnalysisType.RATE,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 80]},
    cdegree={"1JB0-PSI-syn-cocc": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)


big_md_rate_PSII = ScenarioConfig(
    name="big_MD_rate_PSII",
    analysis_type=AnalysisType.RATE,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.4, 0.8]},
    crystal_suffix="_crystal",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)

small_md_rate_PSII = ScenarioConfig(
    name="small_MD_rate_PSII",
    analysis_type=AnalysisType.RATE,
    description=None,
    random_start_over_whole_membrane=False,  # means start around PSII
    number_of_proteins={"avg_membrane": [40, 60, 70]},
    cdegree={"3WU2-PSII-ThermosynVul": [0, 0.2]},
    crystal_suffix="_crystal_small",
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1),
)


crystal_analysis = ScenarioConfig(
    name="crystal_analysis",
    analysis_type=AnalysisType.AIM,
    description=None,
    random_start_over_whole_membrane=True,
    number_of_proteins={
        "3WU2-PSII-ThermosynVul": ["hexagonal"],
        "1JB0-PSI-syn-cocc": ["hexagonal"],
    },
    mv=[1.2, 1.4, 1.8, 2.0],
    cdegree=None,
    time_scale=1000.0,
    time_label="Time / ms",
    plot_runs=PlotRunsConfig(enabled=True, max_runs=1, aim=True),
    fully_crystal=True,
    start_area=Point(2500, 2500).buffer(300),
    aim_area=Point(0, 0).buffer(500),
)


# --------------------
# ALL SCENARIOS
# --------------------

ALL_SCENARIOS = [
    # Diffusion
    # no_md_diff_random_start,
    # no_md_diff_random_start_coarse
    # no_md_diff,
    # NO MD FPT
    # no_md_fpt_random_start,
    no_md_fpt,
    # no_md_fpt_coarse,
    # FPT PSI
    # big_md_fpt_random_start,
    # big_md_fpt,
    # big_md_fpt_coarse
    # big_md_fpt_mv_sweep,
    # small_md_fpt_random_start,
    # small_md_fpt,
    # RATE PSI
    # big_md_rate,
    # big_md_rate_coarse,
    # small_md_rate,
    # FPT PSII
    # big_md_fpt_random_start_PSII,
    # big_md_fpt_PSII,
    # small_md_fpt_random_start_PSII,
    # small_md_fpt_PSII,
    # RATE PSII
    # big_md_rate_PSII,
    # small_md_rate_PSII,
    # CRYSTALS
    # crystal_analysis
]
