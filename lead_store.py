"""Durable lead capture: write to a Google Sheet, fall back to local CSV.

On Streamlit Community Cloud the filesystem is ephemeral, so leads written to a local
CSV are lost on reboot. When a Google service account + sheet URL are configured in
``st.secrets`` this module appends each lead to the sheet instead. With no secrets (or
on any error) it falls back to ``leads.csv`` so the app still works locally.

Secrets format (``.streamlit/secrets.toml`` — see ``secrets.toml.example``):

    [gcp_service_account]
    type = "service_account"
    ... (the full service-account JSON, as TOML keys) ...

    [gsheets]
    url = "https://docs.google.com/spreadsheets/d/<id>/edit"
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

LEADS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leads.csv")
HEADER = ["timestamp", "email", "age", "salary", "gender", "savings", "spouse_age"]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _row(email: str, profile: tuple) -> list:
    _pid, age, salary, gender, savings, spouse_age = profile
    return [datetime.now().isoformat(timespec="seconds"), email, age, salary,
            gender, savings, spouse_age]


def _save_csv(email: str, profile: tuple) -> None:
    new = not os.path.exists(LEADS_FILE)
    with open(LEADS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow(_row(email, profile))


def _gsheets_secrets():
    """Return the st.secrets mapping if Google Sheets is configured, else None."""
    try:
        import streamlit as st
        if "gcp_service_account" in st.secrets and "gsheets" in st.secrets:
            return st.secrets
    except Exception:
        return None
    return None


def _worksheet(secrets):
    """Authorize gspread and return the first worksheet (cached per session)."""
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(secrets["gcp_service_account"]), scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(secrets["gsheets"]["url"])
    return sh.sheet1


def _save_gsheet(email: str, profile: tuple, secrets) -> None:
    ws = _worksheet(secrets)
    if not ws.get_all_values():          # empty sheet -> write header first
        ws.append_row(HEADER, value_input_option="USER_ENTERED")
    ws.append_row(_row(email, profile), value_input_option="USER_ENTERED")


def save_lead(email: str, profile: tuple) -> str:
    """Persist a lead. Returns the destination: 'gsheet', 'local', or 'local (...)'.

    Tries Google Sheets when configured; on any failure falls back to local CSV so a
    lead is never dropped.
    """
    secrets = _gsheets_secrets()
    if secrets is not None:
        try:
            _save_gsheet(email, profile, secrets)
            return "gsheet"
        except Exception as e:                # noqa: BLE001 - never lose a lead
            _save_csv(email, profile)
            return f"local (Google Sheets error: {e})"
    _save_csv(email, profile)
    return "local"
