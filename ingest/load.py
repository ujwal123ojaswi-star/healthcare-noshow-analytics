"""Load appointment data into DuckDB and hand back a clean DataFrame.

If a real CSV is present it's used (and mapped via COLUMN_MAP); otherwise
synthetic data is generated. DuckDB is used as the store so the analytics
queries in the dashboard can run SQL if desired.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from ingest.config import COLUMN_MAP, REAL_DATA_CSV, WAREHOUSE_PATH
from ingest.generate_data import generate


def load_appointments() -> pd.DataFrame:
    if REAL_DATA_CSV.exists():
        print(f"[load] using real data: {REAL_DATA_CSV}")
        raw = pd.read_csv(REAL_DATA_CSV)
        # map external column names to our internal names where provided
        rename = {v: k for k, v in COLUMN_MAP.items() if v in raw.columns}
        df = raw.rename(columns=rename)
    else:
        print("[load] no real CSV found — generating synthetic data")
        df = generate()

    for col in ("scheduled_date", "appointment_date"):
        df[col] = pd.to_datetime(df[col])

    con = duckdb.connect(str(WAREHOUSE_PATH))
    try:
        con.execute("CREATE OR REPLACE TABLE appointments AS SELECT * FROM df")
        n = con.execute("SELECT count(*) FROM appointments").fetchone()[0]
        print(f"[load] appointments: {n:,} rows")
    finally:
        con.close()
    return df


if __name__ == "__main__":
    load_appointments()
