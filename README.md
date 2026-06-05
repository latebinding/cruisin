# SmartNest (Python) — required-duration & minimum-income model

A working Python reimplementation of the SmartNest retirement model's
`default_scenario.m` pipeline. Given a participant's profile (age, salary, current
DB balance, spouse age) it computes, for every projection year up to retirement:

- the **required bond duration** the Current Balance must target (immunisation), and
- the **minimum retirement income** the strategy supports,

plus liability duration, human-capital duration, the married annuity factor, and the
GBF / CB / HC balances.

## Why this is a reimplementation (read this)

The original MATLAB project survived only as the orchestrator (`default_scenario.m`)
and config (`parameters.m`). **All eight computational helper functions it called
were lost** (the repo was a stripped SVN shell with no recoverable history), as were
the three Excel input files. This package reconstructs them.

The *interfaces* are pinned tightly by the surviving call sites (array shapes,
indexing, how each result is consumed). The *internal formulas* are standard,
documented financial-math choices — so the outputs **approximate the original model's
intent, not its exact numbers**. The main modeling choices / corrections:

1. **Bond pricing** (`rates.getbondprices`): a Vasicek short-rate process drives the
   simulated rates, but bonds are priced by **discounting the expected future
   short-rate path** (expectations hypothesis), not the Vasicek closed form. With the
   near-zero mean reversion in `parameters.m` (`a=0.0042`) the closed form's
   convexity term explodes and produces bond prices far above 1; expectations-based
   pricing stays in `(0,1]` and decreasing in maturity.
2. **Adjusted contribution** (`contributions.adjcontrib`): the original meaning of
   `meanreductionfactor` (5.85) and `multfactor` (2.2) is unknown. We scale an
   `fr`-grown contribution by `multfactor/meanreductionfactor`. The absolute
   contribution level is the least certain part of the model; both factors are exposed
   in `parameters.py` for tuning.
3. **Salary growth** (`scenario.run_scenario`): the MATLAB line
   `isalary = isalary*(1+salg)^t` re-compounds the exponent each year (~21× over 17
   years — a latent bug). We use correct geometric growth `salary*(1+salg)^t`.
4. **Human-capital duration** (`liability.hcduradjcontrib`): computed by bumping the
   discount curve ±1bp (central difference), equivalent to the cash-flow Macaulay
   duration.
5. **Reported required duration** is clamped to `[0, maxduration]` — the investable
   band, and the value actually fed back into the next year's CB roll.

## Layout

```
smartnest/      core package (parameters, rates, contributions, liability, dataio, scenario)
data/           input CSVs (profile, morttable, q_bondprices) — see make_sample_data.py
outputs/        required_duration_*.csv written by run.py
tests/          pytest sanity + end-to-end checks
make_sample_data.py   regenerate the sample inputs
run.py          entry point
```

## Run it

```bash
pip install -r requirements.txt
python make_sample_data.py        # writes data/*.csv (only needed once)
python run.py                     # full run (numsim=1000); prints a summary
python run.py --numsim 200        # faster bring-up run
python run.py --savings-plan 0.70 # also report the savings rate for a 70% plan
python -m pytest tests/ -q        # verification suite
```

## Web app (Streamlit)

A simple browser UI over the model:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a profile (`id, age, salary, gender, savings, spouseAge`) to see your required
CB duration, liability duration, and minimum income per year. To unlock the **required
savings rate (retire at 62 vs 67)** table, enter an email — captured emails and the
submitted profile are appended to `leads.csv`. A slider lets you vary the target
replacement rate; the table recomputes live.

### Deploy to Streamlit Community Cloud

1. **Push to GitHub.** Put this `python/` directory in a repo. The deploy needs:
   `app.py`, `smartnest/`, `savings_plan.py`, `requirements.txt`,
   `.streamlit/config.toml`, and the input data `data/morttable.csv` +
   `data/q_bondprices.csv`. (`leads.csv`, `outputs/`, and `__pycache__/` are
   gitignored.)
2. **Privacy:** `data/profile.csv` holds personal figures and is **not** used by the web
   app (the app builds the profile from the form). If the repo is public, add
   `data/profile.csv` to `.gitignore` so your salary/savings aren't published.
3. **Create the app** at <https://share.streamlit.io> → *New app* → pick the repo and
   branch → set **Main file path** to `python/app.py` (or just `app.py` if `python/` is
   the repo root) → under *Advanced*, select **Python 3.11**. Click *Deploy*.
   Dependencies install from `requirements.txt` automatically.
### Durable lead capture (Google Sheets)

Community Cloud's filesystem is **ephemeral**, so `leads.csv` is wiped on reboot. When a
Google service account is configured, the email gate appends leads to a Google Sheet
instead (`lead_store.py`); without it, it falls back to the local CSV automatically.

One-time setup:
1. In Google Cloud Console: create a project, **enable the Google Sheets API**, then
   create a **service account** and download its **JSON key**.
2. Create a Google Sheet and **share it** (Editor) with the service account's
   `client_email` (e.g. `smartnest-leads@<project>.iam.gserviceaccount.com`).
3. Put credentials in secrets — copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml` locally, or paste the same content into the Cloud app's
   **Settings → Secrets**. Fill in the service-account fields and the sheet `url`.
   (`secrets.toml` is gitignored — never commit it.)

The app writes a header row on first use; each unlocked email appends
`timestamp, email, age, salary, gender, savings, spouse_age`. `gspread` + `google-auth`
are already in `requirements.txt`.

## Required savings rate

Answers *"to achieve an X% replacement plan, what % of my income must I save?"* It
runs the model in flat-rate contribution mode (contribution = rate x gross salary,
constant every year) and solves for the rate whose retirement income equals
`target x final-year salary`:

```bash
python savings_plan.py --target 0.70 --id 1     # -> "save 24.1% of your income each year"
```

If your current savings alone already reach the target, it reports `0% needed`. The
underlying knobs are `contribution_mode` (`"franchise"` | `"flat_rate"`) and
`savings_rate` in `smartnest/parameters.py`; `savings_plan.py` drives them for you.
(`calibrate.py` is the related forward tool — it tunes the franchise-mode scale to a
target instead of reporting a savings rate.)

## Use your own numbers

Edit `data/profile.csv` (one row per person):

| column     | meaning                          |
|------------|----------------------------------|
| id         | any integer label                |
| age        | current age (must be < retireage)|
| salary     | current annual salary            |
| gender     | 1 = male, 2 = female (unused)    |
| oldDB      | current/legacy DB plan balance   |
| spouseAge  | spouse's current age             |

Other assumptions (retirement age, contribution rates, the SSC franchise, the rate
model, etc.) live in `smartnest/parameters.py`. Re-run `python run.py`; results land
in `outputs/`.

## Out of scope

The optimizer (`durationmatching` / `Optimization.m`), GARCH calibration
(`calibration.m`), and the EPP_2.0 probability code were not part of this build.
