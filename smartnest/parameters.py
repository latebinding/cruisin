"""Model parameters — direct port of the surviving ``parameters.m``.

Every scalar here mirrors a line in ``parameters.m`` (kept as a comment for
traceability). Equity params (``mu``/``sigmae``) are included for completeness but
are unused by the required-duration / minimum-income pipeline (there is no equity
simulation in ``default_scenario.m``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Parameters:
    # --- General -------------------------------------------------------------
    numsim: int = 1000          # number of simulation runs
    gbf_switch: int = 1         # 1 = keep GBF; 0 = kill GBF
    maxduration: int = 35       # max duration the Current Balance can be invested
    maxiteration: int = 50      # legacy XIRR iteration cap (unused here)
    realvsnominal: int = 1      # 1 = nominal annuity; 2 = real annuity

    # --- Equity simulation (unused in this pipeline) -------------------------
    mu: float = (0.7880 * 12) / 100        # mean equity return
    sigmae: float = (5.4101 * (12 ** 0.5)) / 100  # std of equity return

    # --- Nominal short-rate (Vasicek) ---------------------------------------
    a: float = 0.0042           # mean-reversion speed
    b: float = 0.0563           # long-run mean (Ang-Bekaert)
    sigma: float = 0.0241       # short-rate volatility
    r0: float = 0.03            # initial short rate
    year: int = 50              # horizon length
    M: int = 50                 # longest maturity required

    # --- Real short-rate -----------------------------------------------------
    a_real: float = 0.0042
    b_real: float = 0.017
    sigma_real: float = 0.0289
    r0_real: float = 0.017

    # --- Population / planner ------------------------------------------------
    retireage: int = 62         # retirement age
    salg: float = 0.02          # salary growth rate
    sr: float = 0.80            # amount invested in FI
    rri: float = 0.10           # replacement-rate improvement
    incrmin: float = 0.10       # excess of desired over minimum income

    # --- Annuity assumptions -------------------------------------------------
    jsratio: float = 0.70       # joint-and-survivor ratio
    convratio: float = 0.24     # conversion ratio
    ap: float = 23.1481481481481  # annuity price
    fap: float = 22.1887        # fixed annuity factor

    # --- Contribution assumptions -------------------------------------------
    fr: float = 0.04            # fixed growth rate of contributions
    cbra: float = 0.235         # contribution rate above the SSC franchise
    cbrb: float = 0.015         # contribution rate below the SSC franchise
    sscg: float = 0.02          # SSC (franchise) growth
    ss: float = 56400.0         # initial social-security franchise
    vsw: float = 0.0            # voluntary saving
    # Calibrated 2026-06 so participant id=1 hits a 70% replacement rate under
    # db_mode="balance" (scale = multfactor/meanreductionfactor = 1.299). Original
    # values were meanreductionfactor=5.85, multfactor=2.2 (scale 0.376).
    meanreductionfactor: float = 1.6938  # adjusted-contribution reduction factor
    multfactor: float = 2.2

    # --- Simulation shock covariance ----------------------------------------
    covar: float = 0.0012

    # --- Reproducibility (new; original used MATLAB default RNG) ------------
    seed: int = 12345

    # --- DB-balance interpretation (reconstruction choice) ------------------
    # "convert": faithful port of default_scenario.m line 130 — grows the old-DB
    #            figure at 6%, discounts by the bond price, and multiplies by
    #            ap/(125/12) (~2.2x). Treats the input as a legacy DB entitlement.
    # "balance": treats the old-DB figure as a plain market balance entering year 1
    #            (what "current savings" usually means). Recommended for 401k-style
    #            balances; avoids the ~2.2x inflation that floors income near ~100%.
    db_mode: str = "balance"

    # --- Contribution interpretation ----------------------------------------
    # "franchise": the model's two-tier (below/above SSC) formula scaled by
    #              multfactor/meanreductionfactor (the original behavior).
    # "flat_rate": each year's contribution = savings_rate * that year's gross salary.
    #              Interpretable as "save X% of income"; used by savings_plan.py.
    contribution_mode: str = "franchise"
    savings_rate: float = 0.0          # flat fraction of gross salary (flat_rate mode)

    def scaled(self, *, numsim: int | None = None, pop: int | None = None) -> "Parameters":
        """Return a copy with ``numsim`` overridden (for fast bring-up runs)."""
        from dataclasses import replace
        kw = {}
        if numsim is not None:
            kw["numsim"] = numsim
        return replace(self, **kw) if kw else self
