"""Generate the three sample input files (profile / mortality / q_bondprices).

Reproducible stand-ins for the lost Excel inputs. The mortality table is a
Gompertz-Makeham approximation of a standard US life table (the original cited a 2001
US census table, which is approximated here). Run:  python make_sample_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def make_profile() -> None:
    """Three sample participants (id, age, salary, gender, savings, spouseAge)."""
    rows = [
        {"id": 1, "age": 45, "salary": 60000, "gender": 1, "savings": 20000, "spouseAge": 43},
        {"id": 2, "age": 50, "salary": 85000, "gender": 1, "savings": 40000, "spouseAge": 49},
        {"id": 3, "age": 38, "salary": 50000, "gender": 2, "savings": 10000, "spouseAge": 40},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "profile.csv"), index=False)


def make_morttable(max_age: int = 120) -> None:
    """Gompertz-Makeham one-year death probabilities q_x for ages 0..max_age."""
    A = 0.00022          # age-independent (accident) component
    B = 0.0000288        # Gompertz level
    c = 1.098            # Gompertz growth
    ages = np.arange(max_age + 1)
    mu = A + B * c ** ages.astype(float)      # force of mortality
    qx = 1.0 - np.exp(-mu)
    qx = np.clip(qx, 0.0, 1.0)
    qx[-1] = 1.0                              # certain death at the terminal age
    pd.DataFrame({"age": ages, "qx": qx}).to_csv(
        os.path.join(DATA_DIR, "morttable.csv"), index=False)


def make_q_bondprices(M: int = 50, flat_rate: float = 0.03) -> None:
    """Minimal decreasing zero-price column (load-only placeholder)."""
    m = np.arange(M + 1)
    price = np.exp(-flat_rate * m)
    pd.DataFrame({"maturity": m, "price": price}).to_csv(
        os.path.join(DATA_DIR, "q_bondprices.csv"), index=False)


if __name__ == "__main__":
    make_profile()
    make_morttable()
    make_q_bondprices()
    print(f"Sample data written to {DATA_DIR}")
