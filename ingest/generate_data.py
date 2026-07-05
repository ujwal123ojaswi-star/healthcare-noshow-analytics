"""Generate a realistic synthetic appointment dataset.

Lets the project run end to end with zero downloads. No-show probability is
driven by lead time, prior no-shows, reminders, and age — the same signals seen
in real no-show data — so the model has genuine structure to learn. The output
schema matches what a DocAppointment export would provide, so real data can be
swapped in without code changes.

Clearly synthetic: do not present the numbers as real. Swap in the public
"Medical Appointment No Shows" dataset for the resume-grade version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ingest.config import N_APPOINTMENTS, N_PATIENTS, RANDOM_SEED, REAL_DATA_CSV

CLINICS = ["Downtown", "Riverside", "Hillcrest", "Eastgate"]
GENDERS = ["F", "M"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_appointments: int = N_APPOINTMENTS, n_patients: int = N_PATIENTS,
             seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # per-patient baseline propensity to miss (latent trait)
    patient_bias = rng.normal(0, 0.8, size=n_patients)
    patient_age = rng.integers(1, 95, size=n_patients)
    patient_gender = rng.choice(GENDERS, size=n_patients, p=[0.65, 0.35])
    patient_htn = rng.binomial(1, 0.20, size=n_patients)
    patient_dm = rng.binomial(1, 0.08, size=n_patients)

    pid = rng.integers(0, n_patients, size=n_appointments)
    base_day = np.datetime64("2024-01-01")
    scheduled_offset = rng.integers(0, 300, size=n_appointments)
    lead_time = rng.gamma(shape=1.4, scale=8.0, size=n_appointments).astype(int)
    lead_time = np.clip(lead_time, 0, 120)

    scheduled_date = base_day + scheduled_offset.astype("timedelta64[D]")
    appointment_date = scheduled_date + lead_time.astype("timedelta64[D]")
    sms = rng.binomial(1, np.where(lead_time >= 3, 0.55, 0.15))

    age = patient_age[pid]
    gender = patient_gender[pid]
    htn = patient_htn[pid]
    dm = patient_dm[pid]

    # linear predictor for no-show (tuned for ~20% positive rate)
    logit = (
        -1.15
        + 0.030 * lead_time            # longer wait -> more no-shows
        + patient_bias[pid]            # individual tendency
        - 0.60 * sms                   # reminders reduce no-shows
        - 0.012 * age                  # older patients show up more
        + rng.normal(0, 0.4, size=n_appointments)
    )
    prob = _sigmoid(logit)
    no_show = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "appointment_id": np.arange(1, n_appointments + 1),
            "patient_id": pid,
            "scheduled_date": scheduled_date,
            "appointment_date": appointment_date,
            "age": age,
            "gender": gender,
            "sms_received": sms,
            "hypertension": htn,
            "diabetes": dm,
            "clinic": rng.choice(CLINICS, size=n_appointments),
            "no_show": no_show,
        }
    )
    return df.sort_values(["appointment_date", "appointment_id"]).reset_index(drop=True)


def write_csv(path=REAL_DATA_CSV) -> str:
    df = generate()
    df.to_csv(path, index=False)
    print(f"[generate] wrote {len(df):,} synthetic appointments -> {path} "
          f"(no-show rate {df['no_show'].mean():.1%})")
    return str(path)


if __name__ == "__main__":
    write_csv()
