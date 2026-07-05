"""Feature engineering for no-show prediction.

Pure pandas so it's unit-testable. The important detail is `prior_no_shows`:
it must count only appointments that happened *before* the current one, or the
model would peek at the future (label leakage).
"""
from __future__ import annotations

import pandas as pd

AGE_BINS = [0, 12, 25, 40, 60, 120]
AGE_LABELS = ["child", "youth", "adult", "middle", "senior"]


def add_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lead_time_days"] = (
        df["appointment_date"] - df["scheduled_date"]
    ).dt.days
    return df


def add_prior_no_shows(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative count of a patient's PRIOR no-shows, ordered by appointment date
    and shifted by one so the current row's own outcome is excluded."""
    df = df.sort_values(["patient_id", "appointment_date", "appointment_id"]).copy()
    grp = df.groupby("patient_id")["no_show"]
    df["prior_no_shows"] = grp.cumsum() - df["no_show"]  # exclude current row
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["appointment_date"].dt.day_name()
    df["age_band"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    df["same_day"] = (df["lead_time_days"] == 0).astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df[(df["lead_time_days"] >= 0) & (df["age"].between(0, 115))]
    dropped = before - len(df)
    if dropped:
        print(f"[features] dropped {dropped} invalid rows "
              f"(negative lead time or impossible age)")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline. Returns a frame with engineered columns + target."""
    df = add_lead_time(df)
    df = add_prior_no_shows(df)
    df = add_calendar(df)
    df = clean(df)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from ingest.load import load_appointments

    feats = build_features(load_appointments())
    print(feats[["appointment_id", "lead_time_days", "prior_no_shows",
                 "day_of_week", "no_show"]].head())
