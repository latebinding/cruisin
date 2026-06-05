"""Per-function and end-to-end sanity checks (the plan's verification strategy)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartnest.contributions import (adjcontrib, flat_contribution_schedule,
                                      nomcontribution)
from smartnest.dataio import load_morttable, load_profile
from smartnest.liability import hcduradjcontrib, marriedlduration, required_duration
from smartnest.parameters import Parameters
from smartnest.rates import bpconverter, covsim, getbondprices
from smartnest.scenario import run_scenario

P = Parameters()


@pytest.fixture(scope="module")
def bonds():
    z = covsim(P.covar, P.year, 200, seed=P.seed)
    return getbondprices(P.a, P.b, P.sigma, P.M, P.r0, 200, P.year, z), z


# --- rate engine ---------------------------------------------------------------

def test_bond_shape_and_range(bonds):
    bp, _ = bonds
    assert bp.shape == (P.M + 1, P.year, 200)
    assert np.allclose(bp[0], 1.0)                  # maturity-0 price == 1
    assert np.all(bp > 0) and np.all(bp <= 1.0 + 1e-9)


def test_bond_decreasing_in_maturity(bonds):
    bp, _ = bonds
    # average price should decline with maturity
    avg = bp.mean(axis=(1, 2))
    assert np.all(np.diff(avg) <= 1e-9)


def test_bpconverter_bumps(bonds):
    bp, _ = bonds
    up = bpconverter(bp, +1e-4)
    down = bpconverter(bp, -1e-4)
    # +yield bump lowers prices; row 0 (maturity 0) unchanged
    assert np.allclose(up[0], bp[0]) and np.allclose(down[0], bp[0])
    assert np.all(up[1:] < bp[1:]) and np.all(down[1:] > bp[1:])


def test_covsim_distribution():
    z = covsim(P.covar, P.year, 5000, seed=1)
    assert z.shape == (P.year, 5000)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05


# --- contributions -------------------------------------------------------------

def test_nomcontribution_two_tier():
    # below franchise: only the below-rate applies
    assert nomcontribution(40000, 56400, 0.015, 0.235, 0) == pytest.approx(0.015 * 40000)
    # above franchise: both tiers
    expect = 0.015 * 56400 + 0.235 * (60000 - 56400)
    assert nomcontribution(60000, 56400, 0.015, 0.235, 0) == pytest.approx(expect)


def test_adjcontrib_positive_and_grows():
    adj = adjcontrib(np.ones(P.M + 1), 20, P.retireage, 45, P.fr, P.ap, P.fap,
                     1000.0, P.meanreductionfactor, P.multfactor, 1)
    assert len(adj) == 21
    assert adj[0] > 0
    assert adj[-1] > adj[0]                          # grows at fr


# --- liability / durations -----------------------------------------------------

def test_marriedlduration_reasonable():
    mort = load_morttable()
    bp = np.exp(-0.03 * np.arange(P.M + 1))          # flat 3% curve
    mliabdur, mq = marriedlduration(mort, 60, 62, len(mort) - 1, bp,
                                    P.convratio, P.jsratio, 58)
    assert mq > 0
    # realistic joint-and-survivor lifetime annuity factor (~5-7% payout)
    assert 10.0 < mq < 25.0
    assert 0 < mliabdur < 30


def test_marriedlduration_single_life():
    # No spouse (spage=None): valid annuity factor, and cheaper than joint-and-survivor
    mort = load_morttable()
    bp = np.exp(-0.03 * np.arange(P.M + 1))
    _, mq_joint = marriedlduration(mort, 60, 62, len(mort) - 1, bp,
                                   P.convratio, P.jsratio, 58)
    dur_single, mq_single = marriedlduration(mort, 60, 62, len(mort) - 1, bp,
                                             P.convratio, P.jsratio, None)
    assert mq_single > 0 and 0 < dur_single < 30
    assert mq_single < mq_joint        # single-life annuity costs less than joint


def test_scenario_single_life_runs():
    from smartnest.dataio import Participant
    mort = load_morttable()
    p = Participant(1, 50, 90000, 1, 200000, None)
    res = run_scenario(P, [p], mort, numsim=100)
    assert np.all(np.isfinite(res.matrices["mininc"]))
    assert res.matrices["mininc"][0][-1] > 0


def test_hcduration_sentinel_and_sign():
    bp = np.exp(-0.03 * np.arange(P.M + 1))
    adj = np.full(25, 1000.0)
    d = hcduradjcontrib(62, 45, adj, adj, adj, bp)
    assert d > 0                                     # positive duration when PV>0
    # zero contributions -> PV<=0 -> sentinel
    assert hcduradjcontrib(62, 45, np.zeros(25), np.zeros(25), np.zeros(25), bp) == -1.0


def test_required_duration_immunisation_identity():
    # CB is the only asset -> required duration equals liability duration
    assert required_duration(100.0, 0.0, 0.0, 5, 0, 12.3) == pytest.approx(12.3)
    # no CB -> 0
    assert required_duration(0.0, 50.0, 50.0, 5, 8, 12.0) == 0.0


def test_liability_vector_matches_scalar():
    mort = load_morttable()
    block = np.exp(-0.03 * np.arange(P.M + 1))[:, None] * np.ones((1, 4))
    mld_v, mq_v = marriedlduration(mort, 60, 62, len(mort) - 1, block,
                                   P.convratio, P.jsratio, 58)
    mld_s, mq_s = marriedlduration(mort, 60, 62, len(mort) - 1, block[:, 0],
                                   P.convratio, P.jsratio, 58)
    assert np.allclose(mld_v, mld_s) and np.allclose(mq_v, mq_s)


# --- end-to-end ----------------------------------------------------------------

def test_scenario_end_to_end():
    parts = load_profile()
    mort = load_morttable()
    res = run_scenario(P, parts, mort, numsim=100)
    rd = res.matrices["requireddur"]
    mi = res.matrices["mininc"]
    for m in res.matrices.values():
        assert np.all(np.isfinite(m))                # no NaN/inf anywhere
    # required duration within the allowed band
    assert np.all((rd >= 0) & (rd <= P.maxduration + 1e-6))
    # positive minimum income in valid (nonzero) years
    assert np.all(mi[rd != 0] > 0)


def test_scenario_deterministic():
    parts = load_profile()
    mort = load_morttable()
    a = run_scenario(P, parts, mort, numsim=100).matrices["requireddur"]
    b = run_scenario(P, parts, mort, numsim=100).matrices["requireddur"]
    assert np.array_equal(a, b)


# --- required savings rate -----------------------------------------------------

def test_flat_contribution_schedule():
    adj = flat_contribution_schedule(0.10, 100000, 0.02, 20, 62, 45, t=1)
    assert len(adj) == 21
    assert adj[0] == pytest.approx(0.10 * 100000 * 1.02 ** 1)   # year t=1 salary
    assert adj[5] == pytest.approx(0.10 * 100000 * 1.02 ** 6)   # year t+5
    assert adj[-1] > adj[0]


def test_income_monotonic_in_savings_rate():
    from savings_plan import income_at
    parts = load_profile()
    mort = load_morttable()
    incs = [income_at(r, P, parts, mort, 0, 100) for r in (0.0, 0.05, 0.15)]
    assert incs[0] < incs[1] < incs[2]


def test_required_savings_rate_roundtrip():
    from dataclasses import replace
    from savings_plan import income_at, required_savings_rate
    parts = load_profile()
    mort = load_morttable()
    rate, inc, final_sal = required_savings_rate(0.70, P, parts, mort, 0, numsim=300)
    assert rate is not None and rate > 0
    # re-running at the solved rate reproduces ~70% replacement
    back = income_at(rate, P, parts, mort, 0, 300)
    assert abs(back / final_sal - 0.70) < 0.01


def test_required_savings_rate_floor():
    # An enormous starting balance reaches the target with 0% saving.
    from dataclasses import replace
    from savings_plan import required_savings_rate
    mort = load_morttable()
    parts = load_profile()
    parts[0].savings = 50_000_000
    rate, inc, final_sal = required_savings_rate(0.70, P, parts, mort, 0, numsim=200)
    assert rate is None
