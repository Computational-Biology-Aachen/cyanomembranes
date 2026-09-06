from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from shapely.geometry import box

import cyanomembranes as cm

# --- Analysis types ---


class AnalysisType(Enum):
    FPT = auto()  # First-passage time: tracks Active % over time
    RATE = auto()  # Steady-state rate: tracks Hits and rate constant
    DIFFUSION = auto()
    AIM = auto()


# --- Color palette ---
# Wong (2011) color-blind-safe palette

COLORBLIND_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # gray
]

# --- Protein label map ---
LABEL_DICT = {
    "3WU2-PSII-ThermosynVul": "PSII",
    "4H13-cytb6f": "Cytb6f",
    "1JB0-PSI-syn-cocc": "PSI",
    "avg_membrane": "Avg",
}

# --- Marker list ---
MARKER_LST = ["o", "s", "^", "D"]


@dataclass
class Condition:
    """Everything needed to identify and load one simulation condition.
    Yielded by _iter_conditions for each (pkey, nprot, mv, cg) combination.
    """

    key: tuple
    file_lst: list[Path]
    pkey: str
    nprot: int
    mv: float
    cg: float | None
    crystal_prot: str | None
    test: str


@dataclass
class PlotRunsConfig:
    enabled: bool = False
    indices: Optional[list[int]] = None
    max_runs: Optional[int] = None
    aim: bool = False

    def should_plot(self, index: int, total: int) -> bool:
        if not self.enabled:
            return False
        if self.indices is not None:
            return index in self.indices
        if self.max_runs is not None:
            return index < self.max_runs
        return True


# --- Scenario configuration ---
@dataclass
class ScenarioConfig:
    # --- Identitiy ---
    name: str
    analysis_type: AnalysisType
    description: str | None

    # --- Spatial arrangement ---
    random_start_over_whole_membrane: bool

    # --- Protein sweep ---
    number_of_proteins: dict  # {pkey: [n1, n2, ...]}

    # --- Crystalline domain sweep ---
    cdegree: Optional[dict] = None  # {crystal_pro: [cg1, cg2, ...]}
    crystal_suffix: str = ""

    # --- Parameter sweeps ---
    mv: list[float] = field(default_factory=lambda: [1.2])
    sweep_mv: bool = False

    # --- Plot options ---
    time_scale: float = 1.0
    time_label: str = "Time (s)"

    # --- Steady State ---
    steady_state_window: int = 20

    # --- Start and Aim areas ---
    start_area: Optional[object] = None
    aim_area: Optional[object] = None

    # --- Lattice resolution ---
    lattice_resolution: Optional[int] = None

    # --- fully_crystal ----
    fully_crystal: bool = False

    # --- membranesize ---
    membrane_size: Optional[tuple[int, int]] = (
        None  # assume 5000 to be the standars (see make_exp_config)
    )

    plot_runs: PlotRunsConfig = field(default_factory=PlotRunsConfig)


def make_exp_config(
    scenario: ScenarioConfig,
    cytb6f_area: float,
    psii_area: float,
    n_processes: int,
) -> cm.brownian_lattice.ExperimentLatticeConfig:
    """Single source of truth for experiment configurations

    Shared parameters are set unconditionally.
    Analysis-type-specific parameters are set in the if/elif block
    Spatial arrangement (random_start) is handled last.
    """

    cfg = cm.brownian_lattice.ExperimentLatticeConfig()

    # --- Shared parameters ---
    cfg.replicates = 3000
    cfg.diff_coefficient = 3.5e9  # Å²/s == 3.5×10⁻⁷ cm²/s
    cfg.particle_radius = 5
    cfg.dimensions = (
        scenario.membrane_size if scenario.membrane_size is not None else (0, 5000)
    )
    cfg.random_start = True
    cfg.has_ghost = True
    cfg.store_history = False
    cfg.workers = n_processes

    # --- Analysis-type-specific parameters
    if scenario.analysis_type == AnalysisType.FPT:
        cfg.nsteps = 1_000_000
        cfg.save_every = 3_000
        cfg.chosen_obstacles_by_area = cytb6f_area

    elif scenario.analysis_type == AnalysisType.RATE:
        cfg.nsteps = 100_000
        cfg.save_every = 500
        cfg.chosen_obstacles_by_area = cytb6f_area
        cfg.steady_state_chosen_obstacle = True

    elif scenario.analysis_type == AnalysisType.DIFFUSION:
        cfg.nsteps = 7_000_000
        cfg.save_every = 7_000

    if scenario.analysis_type == AnalysisType.AIM:
        cfg.nsteps = 1_000_000
        cfg.save_every = 3_000
        cfg.start_area = scenario.start_area
        cfg.aim_area = scenario.aim_area

    # --- Spatial arrangement ---
    if not scenario.random_start_over_whole_membrane:
        cfg.start_area_by_size = (psii_area, 50, box(0, 0, 5000, 5000))

    if scenario.lattice_resolution:
        cfg.lattice_resolution = scenario.lattice_resolution
        cfg.nsteps = int(
            cfg.nsteps / cfg.lattice_resolution**2
        )  # correction for lattice resolution
        cfg.particle_radius = (
            None  # For this experiment the particle radius should be None
        )
        cfg.save_every = int(cfg.save_every / cfg.lattice_resolution**2)

    return cfg
