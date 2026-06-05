"""Input loaders for the three data files that ``parameters.m`` read from Excel.

We use CSV instead of the original ``.xls`` (portable, numeric-only, and the original
files are gone). Column order for ``profile.csv`` follows the indices used in
``default_scenario.m``:

    col (1-based)  meaning              used as
    1              id                   (ignored)
    2              age                  profilep(p,2)
    3              salary               profilep(p,3)
    4              gender               (ignored)
    5              savings balance      profilep(p,5)   (column "savings"; legacy "oldDB" still read)
    6              spouse age           profilep(p,6)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class Participant:
    pid: int
    age: int
    salary: float
    gender: int
    savings: float
    spouse_age: int


def load_profile(path: str | None = None) -> list[Participant]:
    path = path or os.path.join(DATA_DIR, "profile.csv")
    df = pd.read_csv(path)
    savings_col = "savings" if "savings" in df.columns else "oldDB"  # legacy fallback
    return [
        Participant(
            pid=int(r["id"]),
            age=int(r["age"]),
            salary=float(r["salary"]),
            gender=int(r["gender"]),
            savings=float(r[savings_col]),
            spouse_age=int(r["spouseAge"]),
        )
        for _, r in df.iterrows()
    ]


def load_morttable(path: str | None = None) -> np.ndarray:
    """One-year death probabilities q_x indexed by integer age (q[a] at age a)."""
    path = path or os.path.join(DATA_DIR, "morttable.csv")
    return pd.read_csv(path)["qx"].to_numpy(dtype=float)


def load_q_bondprices(path: str | None = None) -> np.ndarray:
    """Annuity/bond price column. Load-only: not referenced by the scenario pipeline."""
    path = path or os.path.join(DATA_DIR, "q_bondprices.csv")
    return pd.read_csv(path)["price"].to_numpy(dtype=float)
