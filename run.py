"""Entry point: load inputs, run the scenario, write outputs, print a summary.

Usage:
    python run.py                 # full run (numsim from parameters.py)
    python run.py --numsim 200    # faster bring-up run
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from smartnest.dataio import load_morttable, load_profile, load_q_bondprices
from smartnest.parameters import Parameters
from smartnest.scenario import run_scenario

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numsim", type=int, default=None,
                    help="override number of simulation paths (default: parameters.py)")
    ap.add_argument("--savings-plan", type=float, default=None, metavar="TARGET",
                    help="also report the flat savings rate needed for this replacement "
                         "rate (e.g. 0.70)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    params = Parameters()
    participants = load_profile()
    mort = load_morttable()
    _ = load_q_bondprices()        # load-only, per the original (unused downstream)

    res = run_scenario(params, participants, mort, numsim=args.numsim)
    years = np.arange(1, res.maxyrs + 1)

    # Write the ten result matrices, one CSV each (rows = participants).
    for key, mat in res.matrices.items():
        df = pd.DataFrame(mat, columns=[f"year_{y}" for y in years])
        df.insert(0, "participant_id", [p.pid for p in participants])
        df.to_csv(os.path.join(OUT_DIR, f"required_duration_{key}.csv"), index=False)

    # Console summary: per participant, the current (year-1) figures and the
    # near-retirement projection (last year where the person is still pre-retirement).
    print(f"\nSmartNest results  (numsim={args.numsim or params.numsim}, "
          f"retireage={params.retireage}, maxyrs={res.maxyrs})")
    print("=" * 78)
    rd = res.matrices["requireddur"]
    mi = res.matrices["mininc"]
    ld = res.matrices["liabdur"]
    for pi, part in enumerate(participants):
        valid = np.where(rd[pi] != 0)[0]
        if valid.size == 0:
            continue
        j0, jL = valid[0], valid[-1]
        print(f"\nParticipant {part.pid}: age {part.age}, salary {part.salary:,.0f}, "
              f"spouse {part.spouse_age}, savings {part.savings:,.0f}")
        print(f"  current  (yr {j0+1}, age {res.first_year_age[pi]+j0}): "
              f"required CB duration = {rd[pi, j0]:6.2f} yrs | "
              f"liability duration = {ld[pi, j0]:6.2f} | "
              f"min income = {mi[pi, j0]:,.0f}/yr")
        print(f"  near-ret (yr {jL+1}, age {res.first_year_age[pi]+jL}): "
              f"required CB duration = {rd[pi, jL]:6.2f} yrs | "
              f"liability duration = {ld[pi, jL]:6.2f} | "
              f"min income = {mi[pi, jL]:,.0f}/yr")
    print(f"\nFull matrices written to {OUT_DIR}/required_duration_*.csv")

    if args.savings_plan is not None:
        from savings_plan import required_savings_rate
        print("\nRequired savings rate")
        print("-" * 78)
        for pi, part in enumerate(participants):
            rate, inc, final_sal = required_savings_rate(
                args.savings_plan, params, participants, mort, pi,
                args.numsim or params.numsim)
            if rate is None:
                print(f"  Participant {part.pid}: current savings alone reach "
                      f"{inc/final_sal:.0%} — 0% additional saving needed.")
            else:
                print(f"  Participant {part.pid}: save {rate:.1%} of income/yr "
                      f"(≈ {rate*part.salary:,.0f} in yr 1) for a "
                      f"{args.savings_plan:.0%} plan.")


if __name__ == "__main__":
    main()
