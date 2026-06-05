"""Contribution schedule — reconstructs ``nomcontribution`` and ``adjcontrib``.

``nomcontribution`` is a standard two-tier (franchise) DC contribution and is well
constrained. ``adjcontrib`` is the **largest documented approximation** in the model:
the original meaning of ``meanreductionfactor`` (5.85) and ``multfactor`` (2.2) is
lost. We implement a transparent reduction/scaling applied to an ``fr``-grown
contribution stream; the absolute contribution level is therefore approximate and
the two factors are exposed for tuning. See the project README.
"""

from __future__ import annotations

import numpy as np


def nomcontribution(isalary: float, ss: float, cbrb: float, cbra: float,
                    vsw: float) -> float:
    """Nominal annual contribution from a two-tier (SSC franchise) DC formula.

    ``cbrb`` applies to salary up to the franchise ``ss``; ``cbra`` to the excess;
    ``vsw`` is a flat voluntary top-up. Salary/franchise growth is already applied by
    the caller before this is invoked (see ``default_scenario.m`` lines 175-176), so
    the growth rates are not used here.
    """
    below = cbrb * min(isalary, ss)
    above = cbra * max(isalary - ss, 0.0)
    return below + above + vsw


def adjcontrib(bp_year: np.ndarray, maxyrs: int, retireage: int, age: int,
               fr: float, ap: float, fap: float, nomcontrib: float,
               meanreductionfactor: float, multfactor: float, t: int) -> np.ndarray:
    """Adjusted contribution schedule (a vector), index 0 = current year, k = year t+k.

    ``adj[0]`` is the contribution actually made this year; ``adj[k]`` (k>=1) are the
    projected future contributions used to value human capital. We grow the nominal
    contribution at the fixed rate ``fr`` and scale by ``multfactor/meanreductionfactor``.

    The schedule is intentionally independent of the bond curve ``bp_year`` (a person's
    contribution is a function of pay, not of rates); ``bp_year`` is accepted for
    call-site fidelity. Rate sensitivity of human capital is computed downstream in
    :func:`smartnest.liability.hcduradjcontrib` by bumping the discount curve.

    Returns a vector of length ``maxyrs + 1`` (entries beyond ``yrstoretire`` are
    harmless — they are never indexed by the orchestrator).
    """
    k = np.arange(maxyrs + 1, dtype=float)
    scale = (multfactor / meanreductionfactor) if meanreductionfactor != 0 else multfactor
    return nomcontrib * (1.0 + fr) ** k * scale


def flat_contribution_schedule(savings_rate: float, salary0: float, salg: float,
                               maxyrs: int, retireage: int, age: int,
                               t: int) -> np.ndarray:
    """Flat-rate contribution schedule: ``adj[k]`` = savings_rate * gross salary in year t+k.

    Models "save a constant X% of income every year." ``salary0`` is the base salary;
    salary in projection year ``y`` is ``salary0 * (1+salg)**y``. Index 0 is the
    contribution deployed this year (year ``t``); index ``k`` is the projected
    contribution in year ``t+k`` (used to value human capital). Same return contract as
    :func:`adjcontrib` — a vector of length ``maxyrs + 1``.
    """
    k = np.arange(maxyrs + 1, dtype=float)
    return savings_rate * salary0 * (1.0 + salg) ** (t + k)
