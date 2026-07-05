"""Operations analytics over the appointment data — pure SQL in DuckDB.

This is the descriptive / BI side of the project (distinct from the predictive
model): the operational questions a clinic analyst reports on. Each function
runs a SQL aggregation against the scored appointments and returns a DataFrame.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from ingest.config import SCORED_PATH

_LEAD_BAND = (
    "case when lead_time_days < 3 then '0-2d' "
    "when lead_time_days < 8 then '3-7d' "
    "when lead_time_days < 15 then '8-14d' "
    "when lead_time_days < 30 then '15-29d' "
    "else '30d+' end"
)


def _query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(sql, [str(SCORED_PATH)]).df()
    finally:
        con.close()


def monthly_trend() -> pd.DataFrame:
    return _query(
        "select date_trunc('month', appointment_date) as month, "
        "count(*) as appointments, avg(no_show) as no_show_rate "
        "from read_parquet(?) group by 1 order by 1"
    )


def by_clinic() -> pd.DataFrame:
    return _query(
        "select clinic, count(*) as appointments, avg(no_show) as no_show_rate "
        "from read_parquet(?) group by 1 order by no_show_rate desc"
    )


def reminder_effectiveness() -> pd.DataFrame:
    return _query(
        f"select {_LEAD_BAND} as lead_band, "
        "case when sms_received = 1 then 'reminder' else 'no reminder' end as reminder, "
        "avg(no_show) as no_show_rate, count(*) as appointments "
        "from read_parquet(?) group by 1, 2 order by 1, 2"
    )


def weekday_leadtime_matrix() -> pd.DataFrame:
    return _query(
        f"select day_of_week, {_LEAD_BAND} as lead_band, avg(no_show) as no_show_rate "
        "from read_parquet(?) group by 1, 2"
    )


def prior_history_segments() -> pd.DataFrame:
    return _query(
        "select case when prior_no_shows = 0 then '0' "
        "when prior_no_shows = 1 then '1' when prior_no_shows = 2 then '2' "
        "else '3+' end as prior_no_shows_bucket, "
        "count(*) as appointments, avg(no_show) as no_show_rate "
        "from read_parquet(?) group by 1 order by 1"
    )


if __name__ == "__main__":
    print(monthly_trend().head())
    print(by_clinic())
    print(reminder_effectiveness())
