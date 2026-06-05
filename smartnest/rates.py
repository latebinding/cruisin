"""Interest-rate engine — reconstructs ``covsim``, ``getbondprices``, ``bpconverter``.

Array convention (matches ``default_scenario.m`` indexing ``bpaa(maturity+1, t, s)``):

    bp[m, t, s]  =  price of a zero-coupon bond of maturity ``m`` years,
                    observed in projection year ``t`` (0-based here),
                    on simulation path ``s``.

Row ``m = 0`` is always price 1.0 (maturity 0). Shape is ``(M+1, year, numsim)``.

Modeling choice: a single-factor **Vasicek** short-rate process drives the simulated
rates (param names in ``parameters.m`` match: ``a`` mean-reversion, ``b`` long-run
mean, ``sigma`` vol, ``r0`` initial; ``b`` cites Ang-Bekaert). Zero-coupon bonds are
priced by **discounting the expected future short-rate path** (expectations
hypothesis), NOT by the Vasicek closed form. With this near-zero mean reversion
(``a=0.0042``) the closed-form convexity term ``~sigma^2 * m^3`` explodes and yields
bond prices far above 1 (negative long yields); expectations-based pricing is robust,
always lands in ``(0, 1]`` and decreasing in maturity, while ``sigma`` still drives
the cross-simulation dispersion that matters for duration and income risk. The
original body is unknown; this is the documented, sane reconstruction.
"""

from __future__ import annotations

import numpy as np


def covsim(covar: float, year: int, numsim: int, seed: int | None = None) -> np.ndarray:
    """Generate the standard-normal innovations that drive the short-rate paths.

    Returns an array of shape ``(year, numsim)`` of i.i.d. N(0, 1) draws. The
    short-rate volatility is applied in :func:`getbondprices` via ``sigma``, so
    ``covar`` is treated as a legacy/unused scaling hook (kept in the signature for
    fidelity to ``parameters.m``). A ``seed`` makes runs reproducible.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((year, numsim))


def getbondprices(a: float, b: float, sigma: float, M: int, r0: float,
                  numsim: int, year: int, zsim: np.ndarray) -> np.ndarray:
    """Simulate Vasicek short rates and price the zero-coupon curve each year.

    Parameters mirror the MATLAB call site
    ``getbondprices(a, b, sigma, M, r0, numsim, year, zsim)``.

    At observation year ``t`` on path ``s`` with short rate ``R = r[t,s]``, the zero of
    maturity ``m`` is priced by discounting the **expected** future short rates under
    the Vasicek mean-reverting drift: ``E[r_{k} | R] = b + (R-b) e^{-a k}``, floored at
    0 to keep the curve in ``(0, 1]``::

        P(m) = exp( - sum_{k=0}^{m-1} max(0, b + (R-b) e^{-a k}) ),   P(0) = 1.

    Returns ``bp`` of shape ``(M+1, year, numsim)`` with ``bp[0] == 1`` everywhere,
    every price in ``(0, 1]`` and decreasing in maturity.
    """
    if zsim.shape != (year, numsim):
        raise ValueError(f"zsim must be (year, numsim)=({year},{numsim}), got {zsim.shape}")

    # Short-rate paths: r[t, s], one Euler step per projection year.
    r = np.empty((year, numsim), dtype=float)
    prev = np.full(numsim, r0, dtype=float)
    for t in range(year):
        prev = prev + a * (b - prev) + sigma * zsim[t]
        r[t] = prev

    k = np.arange(M, dtype=float)                 # horizons 0..M-1
    decay = np.exp(-a * k)                          # (M,)
    bp = np.empty((M + 1, year, numsim), dtype=float)
    bp[0] = 1.0
    for t in range(year):
        R = r[t]                                    # (numsim,)
        exp_rate = b + (R - b)[None, :] * decay[:, None]      # (M, numsim)
        np.maximum(exp_rate, 0.0, out=exp_rate)
        bp[1:, t, :] = np.exp(-np.cumsum(exp_rate, axis=0))   # P(m)=exp(-sum_{k<m})
    return bp


def bpconverter(bp: np.ndarray, shift: float, numsim: int | None = None,
                year: int | None = None, M: int | None = None, mode: int = 0) -> np.ndarray:
    """Apply a parallel continuously-compounded **yield bump** of ``shift`` to a curve.

    Used to build the +/-1bp ("haircut") curves for finite-difference durations.
    Since ``P = exp(-y*m)``, a yield shift maps ``P -> P * exp(-shift * m)`` — exact,
    and the maturity-0 row (m=0) is left at 1.0 automatically. Extra positional args
    mirror the MATLAB signature ``bpconverter(bpaa, shift, numsim, year, M, 0)``;
    ``mode`` is a legacy flag that is unused.
    """
    Mp1 = bp.shape[0]
    m = np.arange(Mp1, dtype=float)
    factor = np.exp(-shift * m)
    return bp * factor[:, None, None]
