"""Orchestrator — Python port of ``default_scenario.m``.

Builds simulated Vasicek bond-price curves, sets up the rolling 5-bucket GBF ladder,
then loops participants x projection-years, evaluating all Monte-Carlo paths at once
(vectorised over the simulation axis). For each participant-year it produces the
average (over paths) Current-Balance **required duration**, liability duration, human-
capital duration, married annuity factor, **minimum income**, and the GBF/CB/HC
balances — the same ten matrices the MATLAB script wrote to ``required_duration.xls``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contributions import adjcontrib, flat_contribution_schedule, nomcontribution
from .dataio import Participant
from .liability import hcduradjcontrib, marriedlduration, required_duration
from .parameters import Parameters
from .rates import covsim, getbondprices

# The ten output matrices the original wrote to the required_duration workbook.
RESULT_KEYS = [
    "requireddur", "hcdur", "liabdur", "marriedq", "mininc",
    "gbf_value", "gbf_facevalue", "cb_value", "cb_facevalue", "hcpv",
]


@dataclass
class ScenarioResult:
    matrices: dict[str, np.ndarray]   # each (pop, maxyrs); column j = projection year j+1
    maxyrs: int
    participants: list[Participant]
    params: Parameters
    first_year_age: list[int]         # age at projection year 1, per participant


def _mean_nonzeros(x: np.ndarray) -> float:
    nz = x[x != 0]
    return float(nz.mean()) if nz.size else 0.0


def _build_buckets(maxyrs: int) -> np.ndarray:
    """Rolling bond-ladder maturities, row t = projection year t (1-based)."""
    buckets = np.zeros((maxyrs + 3, 5), dtype=int)
    buckets[1] = [5, 10, 15, 20, 25]
    for t in range(1, maxyrs + 2):
        buckets[t + 1] = buckets[t] - 1
        if buckets[t + 1, 0] == 0:
            buckets[t + 1, 0] = buckets[t + 1, 1]
            buckets[t + 1, 1] = buckets[t + 1, 2]
            buckets[t + 1, 2] = buckets[t + 1, 3]
            buckets[t + 1, 3] = buckets[t + 1, 4]
            buckets[t + 1, 4] = 30
    return buckets


def _assign_maturity(yrstoretire: int, brow: np.ndarray) -> int:
    """Pick the bucket maturity matching years-to-retirement (else 0 = cash)."""
    for b in (4, 3, 2, 1, 0):
        if yrstoretire >= brow[b]:
            return int(brow[b])
    return 0


def run_scenario(params: Parameters, participants: list[Participant],
                 mort: np.ndarray, numsim: int | None = None) -> ScenarioResult:
    p = params
    numsim = numsim if numsim is not None else p.numsim
    maxage = len(mort) - 1
    ages = [pt.age for pt in participants]
    maxyrs = p.retireage - min(ages)
    pop = len(participants)

    # --- Interest-rate curves (nominal Vasicek). bpaa[m, year, sim] -----------
    zsim = covsim(p.covar, p.year, numsim, seed=p.seed)
    bpaa = getbondprices(p.a, p.b, p.sigma, p.M, p.r0, numsim, p.year, zsim)
    buckets = _build_buckets(maxyrs)
    sidx = np.arange(numsim)

    out = {k: np.zeros((pop, maxyrs)) for k in RESULT_KEYS}
    first_year_age: list[int] = []

    for pi, part in enumerate(participants):
        age0, spage0 = part.age, part.spouse_age
        yrstoretire0 = p.retireage - age0
        if p.db_mode == "convert":
            # Faithful port of default_scenario.m L130 (legacy DB entitlement).
            cbbp = (part.savings * (1.06 ** yrstoretire0)
                    * bpaa[yrstoretire0, 0, :] * p.ap / (125.0 / 12.0))
        else:
            # Treat the figure as a plain current market balance entering year 1.
            cbbp = np.full(numsim, float(part.savings))

        # Per-simulation, per-year state (index by t = 1..maxyrs; col 0 unused).
        requireddur = np.zeros((numsim, maxyrs + 1))
        liabdur = np.zeros((numsim, maxyrs + 1))
        marriedq = np.zeros((numsim, maxyrs + 1))
        hcdur = np.zeros((numsim, maxyrs + 1))
        mininc = np.zeros((numsim, maxyrs + 1))
        gbf_pv = np.zeros((maxyrs + 1, 2, numsim))   # [t, 0=value/1=face, s]
        cb_pv = np.zeros((maxyrs + 1, 2, numsim))
        hc_pv = np.zeros((maxyrs + 1, numsim))

        age, spage, isal, ssv = age0, spage0, part.salary, p.ss
        first_year_age.append(age0 + 1)

        for t in range(1, maxyrs + 1):
            age += 1
            spage += 1
            yrstoretire = p.retireage - age
            if yrstoretire < 0:
                continue

            # Geometric growth from the base. NOTE: the MATLAB used
            # `isalary = isalary*(1+salg)^t`, which re-compounds the exponent every
            # iteration (salary ~ base*(1+g)^(t(t+1)/2), ~21x over 17 yrs) — a latent
            # bug that makes income explode. We use correct annual growth instead.
            isal = part.salary * (1 + p.salg) ** t
            ssv = p.ss * (1 + p.sscg) ** t
            if p.contribution_mode == "flat_rate":
                # Contribution = savings_rate * gross salary each year.
                adjm = flat_contribution_schedule(p.savings_rate, part.salary, p.salg,
                                                  maxyrs, p.retireage, age, t)
            else:
                nomc = nomcontribution(isal, ssv, p.cbrb, p.cbra, p.vsw)
                adjm = adjcontrib(bpaa[:, t - 1, 0], maxyrs, p.retireage, age,
                                  p.fr, p.ap, p.fap, nomc, p.meanreductionfactor,
                                  p.multfactor, t)

            pmat = _assign_maturity(yrstoretire, buckets[t])
            bondprice = bpaa[pmat, t - 1, :]                       # (numsim,)
            gbf_port = bondprice * p.gbf_switch * adjm[0]
            cb_port = (1 - bondprice * p.gbf_switch) * adjm[0]

            if t == 1:
                gbf_cbbp = cbbp * bondprice * p.gbf_switch
                cb_cbbp = cbbp * (1 - bondprice * p.gbf_switch)
            else:
                gbf_cbbp = 0.0
                cb_cbbp = 0.0

            # --- Current Balance present value -------------------------------
            if t >= 2:
                pbm = np.clip(np.round(requireddur[:, t - 1]).astype(int), 1, p.maxduration)
                cb_pbondprice = bpaa[pbm, t - 2, sidx]
                cb_pv[t - 1, 1, :] = cb_pv[t - 1, 0, :] / cb_pbondprice   # prev face value
                cb_cbondprice = bpaa[pbm - 1, t - 1, sidx]
                cb_pv[t, 0, :] = cb_pv[t - 1, 1, :] * cb_cbondprice + cb_port
            else:
                cb_pv[t, 0, :] = cb_port + cb_cbbp

            # --- Guaranteed Benefit Fund present value -----------------------
            if t >= 2:
                gbf_pv[t, 1, :] = gbf_pv[t - 1, 1, :] + adjm[0] * p.gbf_switch
            else:
                gbf_pv[t, 1, :] = (gbf_pv[t - 1, 1, :]
                                   + (gbf_cbbp / bondprice + adjm[0]) * p.gbf_switch)
            gbf_pv[t, 0, :] = gbf_pv[t, 1, :] * bondprice

            # --- Human capital present value (PV of future contributions) ----
            if yrstoretire > 0:
                ii = np.arange(1, yrstoretire + 1)
                hc_pv[t, :] = (adjm[ii][:, None] * bpaa[ii, t - 1, :]).sum(axis=0)

            # --- Liability duration & married annuity factor -----------------
            curve = bpaa[:, t - 1, :]                              # (M+1, numsim)
            mld, mqv = marriedlduration(mort, age, p.retireage, maxage, curve,
                                        p.convratio, p.jsratio, spage)
            liabdur[:, t] = mld
            marriedq[:, t] = mqv

            # --- Human-capital duration --------------------------------------
            if yrstoretire > 0:
                hcdd = hcduradjcontrib(p.retireage, age, adjm, adjm, adjm, curve)
                hcdur[:, t] = np.where(hcdd == -1, 0.0, hcdd)

            # --- Required CB duration (immunisation) -------------------------
            # Clamp to the investable band [0, maxduration]: this is the duration the
            # CB can actually be set to, and it is the value fed back into next year's
            # CB roll (the MATLAB code clamped at the point of use, L276).
            rd = required_duration(cb_pv[t, 0, :], gbf_pv[t, 0, :], hc_pv[t, :],
                                   pmat, hcdur[:, t], liabdur[:, t])
            requireddur[:, t] = np.minimum(rd, p.maxduration)

            # --- Minimum income ----------------------------------------------
            liab_pv = cb_pv[t, 0, :] + gbf_pv[t, 0, :] + hc_pv[t, :]
            liab_fv = liab_pv / bpaa[yrstoretire, t - 1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                mininc[:, t] = np.where(marriedq[:, t] > 0, liab_fv / marriedq[:, t], 0.0)

        # --- Average across simulations into per-person results --------------
        for t in range(1, maxyrs + 1):
            j = t - 1
            out["requireddur"][pi, j] = _mean_nonzeros(requireddur[:, t])
            out["hcdur"][pi, j] = _mean_nonzeros(hcdur[:, t])
            out["liabdur"][pi, j] = _mean_nonzeros(liabdur[:, t])
            out["marriedq"][pi, j] = _mean_nonzeros(marriedq[:, t])
            out["mininc"][pi, j] = _mean_nonzeros(mininc[:, t])
            out["gbf_value"][pi, j] = _mean_nonzeros(gbf_pv[t, 0, :])
            out["gbf_facevalue"][pi, j] = _mean_nonzeros(gbf_pv[t, 1, :])
            out["cb_value"][pi, j] = float(cb_pv[t, 0, :].mean())
            out["cb_facevalue"][pi, j] = float(cb_pv[t, 1, :].mean())
            out["hcpv"][pi, j] = _mean_nonzeros(hc_pv[t, :])

    return ScenarioResult(matrices=out, maxyrs=maxyrs, participants=participants,
                          params=params, first_year_age=first_year_age)
