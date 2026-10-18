"""Common result container shared by all solvers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Result:
    tour: List[int]
    length: float
    evals: int                                   # candidate solutions evaluated
    seconds: float
    history: List[Tuple[int, float]] = field(default_factory=list)  # (evals, best-so-far)
    extra: dict = field(default_factory=dict)
