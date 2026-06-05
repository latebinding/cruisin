"""Calibrate the contribution scale to a target replacement rate.

The adjusted-contribution scale is ``multfactor / meanreductionfactor`` (the lost
factor whose original meaning is unknown). Income at retirement is monotonic in this
scale, so we solve for the scale that makes near-retirement minimum income equal a
target fraction of final pre-retirement salary, then report the (multfactor,
meanreductionfactor) to drop into parameters.py.

Usage:  python calibrate.py [--target 0.70] [--id 1] [--numsim 1000]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from smartnest.dataio import load_morttable, load_profile
from smartnest.parameters import Parameters
from smartnest.scenario import run_scenario


def income_and_salary(scale: float, base: Parameters, parts, mort, pidx, numsim):
    p = replace(base, multfactor=scale, meanreductionfactor=1.0)
    res = run_scenario(p, parts, mort, numsim=numsim)
    maxyrs = res.maxyrs
    inc = res.matrices["mininc"][pidx, maxyrs - 1]      # income at retirement
    final_sal = parts[pidx].salary * (1 + base.salg) ** maxyrs
    return inc, final_sal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.70, help="replacement rate")
    ap.add_argument("--id", type=int, default=1, help="participant id to calibrate")
    ap.add_argument("--numsim", type=int, default=1000)
    args = ap.parse_args()

    base = Parameters()
    parts = load_profile()
    mort = load_morttable()
    pidx = next(i for i, pt in enumerate(parts) if pt.pid == args.id)

    _, final_sal = income_and_salary(0.0, base, parts, mort, pidx, args.numsim)
    target_income = args.target * final_sal

    # Income(scale) = floor + slope*scale (very nearly linear). Check the floor.
    floor_inc, _ = income_and_salary(0.0, base, parts, mort, pidx, args.numsim)
    if floor_inc > target_income:
        print(f"Floor income at scale=0 is {floor_inc:,.0f} "
              f"({100*floor_inc/final_sal:.1f}%), already above the "
              f"{100*args.target:.0f}% target of {target_income:,.0f}.")
        print("Target unreachable by contributions alone "
              f"(db_mode='{base.db_mode}'). Lower the starting balance or db_mode.")
        return

    # Secant solve on a monotonic, near-linear function.
    lo, hi = 0.0, 2.0
    f_lo = floor_inc - target_income
    inc_hi, _ = income_and_salary(hi, base, parts, mort, pidx, args.numsim)
    f_hi = inc_hi - target_income
    while inc_hi < target_income:                  # widen if needed
        hi *= 2
        inc_hi, _ = income_and_salary(hi, base, parts, mort, pidx, args.numsim)
        f_hi = inc_hi - target_income

    scale = hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        inc, _ = income_and_salary(mid, base, parts, mort, pidx, args.numsim)
        f = inc - target_income
        if abs(f) < 0.0005 * target_income:
            scale = mid
            break
        if (f > 0) == (f_hi > 0):
            hi, f_hi = mid, f
        else:
            lo, f_lo = mid, f
        scale = mid

    inc, _ = income_and_salary(scale, base, parts, mort, pidx, args.numsim)
    # Report as multfactor with meanreductionfactor pinned at the original 2.2-based pair.
    # scale = multfactor/meanreductionfactor; keep multfactor=2.2 -> mrf = 2.2/scale.
    mrf = 2.2 / scale if scale > 0 else float("inf")
    print(f"Participant id={args.id}: final salary {final_sal:,.0f}, "
          f"target {100*args.target:.0f}% = {target_income:,.0f}/yr")
    print(f"Calibrated contribution scale = {scale:.4f}  ->  income {inc:,.0f}/yr "
          f"({100*inc/final_sal:.1f}%)")
    print(f"Set in parameters.py:  multfactor = 2.2 ,  meanreductionfactor = {mrf:.4f}")


if __name__ == "__main__":
    main()
