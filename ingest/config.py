"""Central configuration for the healthcare no-show analytics project."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Where the appointment data lives. If REAL_DATA_CSV exists it is used; otherwise
# synthetic data is generated. Point this at the public "Medical Appointment No
# Shows" CSV, or a DocAppointment export, for the resume-grade version.
REAL_DATA_CSV = DATA_DIR / "appointments.csv"
WAREHOUSE_PATH = DATA_DIR / "warehouse.duckdb"
MODEL_PATH = DATA_DIR / "model.joblib"
SCORED_PATH = DATA_DIR / "scored_appointments.parquet"

# Synthetic data size (used only when no real CSV is present).
N_PATIENTS = 1800
N_APPOINTMENTS = 7000
RANDOM_SEED = 42

# Modeling
TEST_SIZE = 0.25
POSITIVE_LABEL = 1  # 1 = no-show

# Feature columns fed to the model
NUMERIC_FEATURES = ["age", "lead_time_days", "prior_no_shows"]
CATEGORICAL_FEATURES = ["gender", "day_of_week", "sms_received",
                        "hypertension", "diabetes", "clinic"]
TARGET = "no_show"

# Column mapping for a DocAppointment / real export -> our internal names.
# Edit the right-hand side to match your export's column names if you use one.
COLUMN_MAP = {
    "appointment_id": "appointment_id",
    "patient_id": "patient_id",
    "scheduled_date": "scheduled_date",
    "appointment_date": "appointment_date",
    "age": "age",
    "gender": "gender",
    "sms_received": "sms_received",
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "clinic": "clinic",
    "no_show": "no_show",
}

MODEL_SEED = int(os.getenv("MODEL_SEED", RANDOM_SEED))
