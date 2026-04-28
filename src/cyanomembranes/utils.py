"""Bundle some utility functions"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

type XY = npt.NDArray[np.float64]
type XYZ = npt.NDArray[np.float64]

def float2d(x: list[float]) -> np.ndarray:
    return np.array(x, dtype=float)


def float2d_(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=float)


def zero_xy() -> XY:
    return np.zeros(2, dtype=float)
