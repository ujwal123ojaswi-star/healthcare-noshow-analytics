"""Feature-engineering tests (pure pandas, no model/DB needed)."""
import pandas as pd

from features import add_lead_time, add_prior_no_shows, build_features, clean


def _sample():
    return pd.DataFrame({
        "appointment_id": [1, 2, 3, 4],
        "patient_id": [10, 10, 10, 20],
        "scheduled_date": pd.to_datetime(
            ["2024-01-01", "2024-01-10", "2024-02-01", "2024-01-05"]),
        "appointment_date": pd.to_datetime(
            ["2024-01-05", "2024-01-12", "2024-02-10", "2024-01-06"]),
        "age": [30, 30, 30, 45],
        "gender": ["F", "F", "F", "M"],
        "sms_received": [1, 0, 1, 0],
        "hypertension": [0, 0, 0, 1],
        "diabetes": [0, 0, 0, 0],
        "clinic": ["A", "A", "A", "B"],
        "no_show": [1, 0, 1, 0],
    })


def test_lead_time():
    df = add_lead_time(_sample())
    assert df.sort_values("appointment_id")["lead_time_days"].tolist() == [4, 2, 9, 1]


def test_prior_no_shows_excludes_current_and_future():
    df = add_prior_no_shows(_sample()).sort_values("appointment_id")
    # patient 10: appt1 (first, 0 prior), appt2 (1 prior no-show), appt3 (1 prior)
    got = df.set_index("appointment_id")["prior_no_shows"].to_dict()
    assert got[1] == 0
    assert got[2] == 1   # appt1 was a no-show
    assert got[3] == 1   # appt2 showed up, so still 1
    assert got[4] == 0   # different patient


def test_clean_drops_invalid():
    bad = _sample()
    bad.loc[0, "appointment_date"] = pd.Timestamp("2023-12-01")  # before scheduled
    out = clean(add_lead_time(bad))
    assert (out["lead_time_days"] >= 0).all()


def test_build_features_columns():
    feats = build_features(_sample())
    for col in ["lead_time_days", "prior_no_shows", "day_of_week", "age_band"]:
        assert col in feats.columns
