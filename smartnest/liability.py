"""Liability & duration math — ``marriedlduration``, ``hcduradjcontrib``,
``required_duration``.

These three are well constrained by how their outputs are consumed in
``default_scenario.m``:

* ``mq`` is divided into a future value to produce income  ->  annuity factor.
* ``mliabdur`` is a Macaulay duration of the liability cash flows.
* ``required_duration`` is the classic immunization solve for the duration the
  Current Balance must carry so total-asset duration matches the liability.

Each function accepts either a single discount curve of shape ``(M+1,)`` (returns a
scalar) or a block of per-simulation curves of shape ``(M+1, numsim)`` (returns a
vector of length ``numsim``). This lets the orchestrator evaluate all simulations at
once while the unit tests exercise the scalar path.
"""

from __future__ import annotations

import numpy as np


def _as_2d(curve: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (curve_2d, was_1d). 1D -> (M+1, 1)."""
    if curve.ndim == 1:
        return curve[:, None], True
    return curve, False


def _survival_curve(mort: np.ndarray, start_age: int, n_max: int) -> np.ndarray:
    """Survival probabilities S[n] = P(alive at age start_age+n), for n = 0..n_max.

    S[0] = 1. ``mort[a]`` is the one-year death probability q_x at integer age ``a``.
    Ages beyond the table are treated as certain death (S -> 0).
    """
    S = np.ones(n_max + 1, dtype=float)
    p = 1.0
    for n in range(1, n_max + 1):
        a = start_age + (n - 1)
        q = mort[a] if 0 <= a < len(mort) else 1.0
        p *= max(0.0, 1.0 - q)
        S[n] = p
    return S


def marriedlduration(mort: np.ndarray, age: int, retireage: int, maxage: int,
                     bp_year: np.ndarray, convratio: float, jsratio: float,
                     spage: int):
    """Joint-and-survivor annuity factor ``mq`` and liability duration ``mliabdur``.

    The annuity pays $1/yr from retirement for as long as either life survives; the
    survivor receives ``jsratio`` of the benefit. Returns ``(mliabdur, mq)`` as scalars
    (1D curve) or arrays of length ``numsim`` (2D curve).

    * ``mq``      = ``sum_n df_n * w_n``                      (price per $1 income/yr)
    * ``mliabdur``= ``sum_n n * w_n * df_n / sum_n w_n * df_n``  (Macaulay duration)

    NOTE on ``convratio``: the call site passes ``convratio`` (=0.24) but its role
    *inside* this function is unknown. Using it as a multiplier on ``mq`` shrinks the
    annuity factor to ~4 (a ~25% lifetime payout), which makes income implausibly
    high. We therefore leave ``mq`` as the survival-weighted discounted annuity factor
    (lands ~14-18, in line with the model's ``ap=23.1``/``fap=22.2`` constants) and do
    not apply ``convratio`` here. It is accepted for signature fidelity.

    with ``w_n = S_x*S_y + jsratio*(S_x + S_y - 2 S_x S_y)`` and ``df_n = bp_year[n]``.
    ``n`` runs from years-to-retirement out to the end of the mortality table / curve.

    If ``spage`` is None there is no spouse: the annuity is single-life on the
    participant, ``w_n = S_x`` (``jsratio`` unused).
    """
    curve, was_1d = _as_2d(bp_year)
    M = curve.shape[0] - 1
    yrstoretire = max(0, retireage - age)
    single = spage is None
    if single:
        n_max = min(M, maxage - age)
    else:
        n_max = min(M, maxage - age, maxage - spage)
    n_max = max(n_max, yrstoretire)

    Sx = _survival_curve(mort, age, n_max)
    n = np.arange(n_max + 1, dtype=float)
    if single:
        w = Sx                                        # single-life annuity
    else:
        Sy = _survival_curve(mort, spage, n_max)
        w = Sx * Sy + jsratio * (Sx + Sy - 2.0 * Sx * Sy)
    w = w * (n >= yrstoretire)                        # pay only from retirement on

    df = curve[: n_max + 1, :]                        # (n_max+1, S)
    wdf = w[:, None] * df                              # (n_max+1, S)
    denom = wdf.sum(axis=0)                            # (S,)
    with np.errstate(divide="ignore", invalid="ignore"):
        mliabdur = np.where(denom > 0, (n[:, None] * wdf).sum(axis=0) / denom, 0.0)
    mq = denom    # survival-weighted discounted annuity factor (convratio not applied; see docstring)

    if was_1d:
        return float(mliabdur[0]), float(mq[0])
    return mliabdur, mq


def hcduradjcontrib(retireage: int, age: int, adj: np.ndarray, adj_up: np.ndarray,
                    adj_down: np.ndarray, bp_year: np.ndarray, shift: float = 1e-4):
    """Human-capital duration via a central finite difference on the discount curve.

    Human capital = PV of future contributions, ``PV = sum_i adj[i] * bp_year[i]`` for
    ``i = 1..yrstoretire``. Bumping the curve yields by ``+/-shift`` and central-
    differencing gives the modified duration (= the cash-flow-weighted Macaulay
    duration). Returns ``-1`` (sentinel) where ``PV <= 0``, matching the orchestrator.
    ``adj_up``/``adj_down`` are accepted for call-site fidelity; the schedule is
    curve-independent here, so the curve bump carries the sensitivity.

    Scalar for a 1D curve; vector of length ``numsim`` for a 2D curve.
    """
    curve, was_1d = _as_2d(bp_year)
    yrstoretire = retireage - age
    if yrstoretire <= 0:
        return 0.0 if was_1d else np.zeros(curve.shape[1])

    i = np.arange(1, yrstoretire + 1)
    c = adj[i][:, None]                                # (yr, 1)
    P = curve[i, :]                                    # (yr, S)
    PV = (c * P).sum(axis=0)                            # (S,)
    PV_up = (c * P * np.exp(-shift * i)[:, None]).sum(axis=0)
    PV_down = (c * P * np.exp(shift * i)[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dur = np.where(PV > 0, (PV_down - PV_up) / (2.0 * PV * shift), -1.0)

    if was_1d:
        return float(dur[0])
    return dur


def required_duration(cb, gbf, hc, pmaturity, hcdur, liabdur):
    """Duration the Current Balance must target so total-asset duration = liability.

    Immunization identity: ``cb*rd + gbf*pmaturity + hc*hcdur = TA*liabdur`` with
    ``TA = cb+gbf+hc``. Returns 0 where there is no CB; negatives clamp to 0. Works on
    scalars or numpy arrays (elementwise). The caller clamps to ``maxduration``.
    """
    cb = np.asarray(cb, dtype=float)
    TA = cb + np.asarray(gbf, dtype=float) + np.asarray(hc, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = (TA * liabdur - np.asarray(gbf) * pmaturity - np.asarray(hc) * hcdur) / cb
    rd = np.where(cb > 0, rd, 0.0)
    rd = np.maximum(0.0, rd)
    return float(rd) if rd.ndim == 0 else rd
