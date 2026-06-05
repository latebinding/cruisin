"""Solve for the flat savings rate needed to hit a target replacement rate.

Answers: *"to achieve a 70% plan, what % of my income must I save each year?"*

Runs the model in ``contribution_mode="flat_rate"`` (contribution = rate x gross
salary) and bisects on the rate — retirement income is monotonic increasing in it —
until income at retirement equals ``target x final salary``. Reuses the monotonic-
solve pattern from ``calibrate.py``.

Usage:  python savings_plan.py [--target 0.70] [--id 1] [--numsim 1000]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from smartnest.dataio import load_morttable, load_profile
from smartnest.parameters import Parameters
from smartnest.scenario import run_scenario


def income_at(rate: float, base: Parameters, parts, mort, pidx, numsim) -> float:
    """Income at retirement for a flat savings rate (fraction of gross salary)."""
    p = replace(base, contribution_mode="flat_rate", savings_rate=rate)
    res = run_scenario(p, parts, mort, numsim=numsim)
    return res.matrices["mininc"][pidx, res.maxyrs - 1]


def required_savings_rate(target_replacement: float, base: Parameters, parts, mort,
                          pidx: int, numsim: int = 1000, tol: float = 5e-4):
    """Return (rate, income, final_salary). ``rate`` is None if 0% already suffices."""
    maxyrs = base.retireage - parts[pidx].age
    final_sal = parts[pidx].salary * (1 + base.salg) ** maxyrs
    target_income = target_replacement * final_sal

    floor = income_at(0.0, base, parts, mort, pidx, numsim)
    if floor >= target_income:
        return None, floor, final_sal

    lo, hi = 0.0, 0.25
    while income_at(hi, base, parts, mort, pidx, numsim) < target_income:
        hi *= 2
        if hi > 4.0:                       # 400% of salary — clearly unreachable
            break
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        inc = income_at(mid, base, parts, mort, pidx, numsim)
        if abs(inc - target_income) <= tol * target_income:
            return mid, inc, final_sal
        if inc < target_income:
            lo = mid
        else:
            hi = mid
    return mid, inc, final_sal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.70, help="replacement rate (0-1)")
    ap.add_argument("--id", type=int, default=1, help="participant id")
    ap.add_argument("--numsim", type=int, default=1000)
    args = ap.parse_args()

    base = Parameters()
    parts = load_profile()
    mort = load_morttable()
    pidx = next(i for i, pt in enumerate(parts) if pt.pid == args.id)
    sal = parts[pidx].salary

    rate, inc, final_sal = required_savings_rate(
        args.target, base, parts, mort, pidx, args.numsim)

    print(f"Participant id={args.id}: salary {sal:,.0f}, current savings "
          f"{parts[pidx].savings:,.0f}, retire at {base.retireage}")
    print(f"Target: {args.target:.0%} of final salary ({final_sal:,.0f}) "
          f"= {args.target*final_sal:,.0f}/yr")
    if rate is None:
        print(f"  Your current savings alone already reach "
              f"{inc/final_sal:.0%} replacement — 0% additional saving needed.")
    else:
        print(f"\n  To achieve a {args.target:.0%} replacement plan, save "
              f"{rate:.1%} of your income each year")
        print(f"  (≈ {rate*sal:,.0f} in year 1; delivers {inc/final_sal:.1%} "
              f"replacement = {inc:,.0f}/yr at retirement).")


if __name__ == "__main__":
    main()
