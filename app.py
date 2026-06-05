"""SmartNest — simple Streamlit web app.

Page 1: enter a profile, see CB duration, liability duration and minimum income.
Gated: enter an email to unlock the "savings rate, retire at 62 vs 67" table.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import re
from dataclasses import replace

import pandas as pd
import streamlit as st

import lead_store
from savings_plan import required_savings_rate
from smartnest.dataio import Participant, load_morttable
from smartnest.parameters import Parameters
from smartnest.scenario import run_scenario

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RETIRE_AGES = (62, 67)


@st.cache_resource
def get_mort():
    return load_morttable()


@st.cache_data
def compute_scenario(profile: tuple) -> pd.DataFrame:
    """Run the model for one profile; return a per-year results table."""
    pid, age, salary, gender, savings, spouse_age = profile
    part = Participant(pid=pid, age=age, salary=salary, gender=gender,
                       savings=savings, spouse_age=spouse_age)
    params = Parameters()
    res = run_scenario(params, [part], get_mort(), numsim=1000)
    n = res.maxyrs
    return pd.DataFrame({
        "Year": range(1, n + 1),
        "Age": [res.first_year_age[0] + j for j in range(n)],
        "CB duration (yrs)": res.matrices["requireddur"][0].round(2),
        "Liability duration (yrs)": res.matrices["liabdur"][0].round(2),
        "Min income / yr": res.matrices["mininc"][0].round(0),
    })


@st.cache_data
def savings_table(profile: tuple, target: float) -> pd.DataFrame:
    """Required flat savings rate for each retirement age, for a target replacement."""
    pid, age, salary, gender, savings, spouse_age = profile
    part = Participant(pid=pid, age=age, salary=salary, gender=gender,
                       savings=savings, spouse_age=spouse_age)
    base = Parameters()
    rows = []
    for ra in RETIRE_AGES:
        b = replace(base, retireage=ra)
        rate, inc, final_sal = required_savings_rate(target, b, [part], get_mort(), 0,
                                                     numsim=400)
        rows.append({
            "Retirement age": ra,
            "Years to save": ra - age,
            "Required savings rate": "0% (already on track)" if rate is None
                else f"{rate:.1%}",
            "≈ Year-1 saving": "—" if rate is None else f"${rate * salary:,.0f}",
            "Target income / yr": f"${target * final_sal:,.0f}",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Cruisin retirement planner", page_icon="🪺")
st.title("🪺 Cruisin retirement planner")
st.caption("Enter your details to see the bond duration your savings should target "
           "and your projected retirement income.")

with st.form("profile_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Your age", min_value=18, max_value=61, value=47, step=1)
        salary = st.number_input("Annual salary ($)", min_value=1, value=250000,
                                 step=1000)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        savings = st.number_input("Current savings ($)", min_value=0, value=1000000,
                                  step=1000)
        has_spouse = st.checkbox("Include a spouse/partner", value=True)
        spouse_age = st.number_input("Spouse age", min_value=18, max_value=100,
                                     value=44, step=1, disabled=not has_spouse)
    submitted = st.form_submit_button("Calculate")

if submitted:
    st.session_state.profile = (
        1, int(age), float(salary), 1 if gender == "Male" else 2,
        float(savings), int(spouse_age) if has_spouse else None,
    )

profile = st.session_state.get("profile")
if profile:
    df = compute_scenario(profile)
    retire_row = df.iloc[-1]   # last pre-retirement year

    st.subheader("Your results at retirement (age 62)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Required CB duration", f"{retire_row['CB duration (yrs)']:.1f} yrs")
    m2.metric("Liability duration", f"{retire_row['Liability duration (yrs)']:.1f} yrs")
    m3.metric("Minimum income / yr *", f"${retire_row['Min income / yr']:,.0f}")

    with st.expander("Year-by-year detail"):
        df_show = df.copy()
        df_show["Min income / yr *"] = df_show.pop("Min income / yr").map(
            lambda x: f"${x:,.0f}")
        st.dataframe(df_show, hide_index=True, width="stretch")
        st.caption(
            "\\* The minimum income per year figure is simply the future, "
            "inflation-boosted price tag required when you retire to buy the exact same "
            "standard of living that a normal, comfortable salary buys you today.")
        st.line_chart(df.set_index("Age")["Min income / yr"])

    # ---------------------------------------------------------------- email gate
    st.divider()
    st.subheader("How much should you save?")
    if not st.session_state.get("unlocked"):
        st.write("Enter your email to unlock your **required savings rate** "
                 "(retire at 62 vs 67).")
        with st.form("email_gate"):
            email = st.text_input("Email address")
            ok = st.form_submit_button("Unlock savings rate")
        if ok:
            if EMAIL_RE.match(email.strip()):
                lead_store.save_lead(email.strip(), profile)
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("Please enter a valid email address.")
    else:
        target_pct = st.slider("Target replacement rate", min_value=50, max_value=90,
                                value=70, step=5, format="%d%%")
        target = target_pct / 100.0
        with st.spinner("Calculating required savings rate…"):
            table = savings_table(profile, target)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(f"Flat % of gross salary, saved every year, to reach "
                   f"{target:.0%} of your final salary as lifetime income.")
