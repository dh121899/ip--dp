"""Contribution data containers for TPC-H query extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ContributionData:
    """Join-row contributions extracted for an SJA or SPJA scalar query."""

    query_id: str
    query_type: Literal["SJA", "SPJA"]
    join_values: NDArray[np.float64]
    private_ids_per_join_row: list[list[str]]
    projection_keys: list[Any] | None
    projected_values: dict[Any, float] | None
    true_answer: float
    extraction_time_sec: float


@dataclass(frozen=True)
class ContributionBounds:
    """Simple lower and upper contribution bounds placeholder."""

    lower: float
    upper: float

    def width(self) -> float:
        """Return the interval width."""

        return self.upper - self.lower
