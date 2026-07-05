# Healthcare No-Show Analytics

**Live dashboard:** _<add your Streamlit Community Cloud URL here once deployed>_

> Which appointments will patients miss, and where should a clinic focus reminders?
> This project analyzes appointment data, predicts no-shows, explains *why* each one
> is high-risk, and quantifies the payoff of targeting the riskiest appointments.
> **Headline finding:** targeting the riskiest ~10% of appointments captures roughly
> [40]% of all no-shows (fill in your run's number).

Built as the analytics + prediction layer on top of the DocAppointment scheduling
system — same appointment domain, turned into operational intelligence.

## Architecture

```
appointment data ─► feature engineering ─► model (LR + RF) ─► explanations ─► dashboard
 (real CSV or        (lead time, prior      (imbalance-aware   (per-appt         (drivers,
  synthetic)          no-shows, reminders)    metrics)           risk factors)     scorer, gains)
```

## Stack

Python 3.11+ (3.14 supported) · pandas · scikit-learn · DuckDB · Streamlit · Plotly

Explanations use the logistic model's own coefficients (contribution = coefficient ×
feature value), so there's **no SHAP/numba dependency** — it installs cleanly on Python 3.14.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -e .

python build.py                   # load/generate -> features -> train -> evaluate
streamlit run app/dashboard.py
```

`build.py` runs the whole pipeline in one command and prints the model metrics
(ROC-AUC, recall) plus the top-decile capture rate.

Run the tests: `pip install -e ".[dev]"` then `pytest`.

## Data

Runs immediately on **bundled synthetic data** (generated with realistic no-show
drivers — no download needed). For the resume-grade version, drop one of these into
`data/appointments.csv` and re-run `python build.py`:

- The public **"Medical Appointment No Shows"** dataset (~110k real appointments).
- A **DocAppointment** export of your own appointments (map columns in `ingest/config.py`).

## What it does

- **Operations analytics** (`analytics.py`): SQL aggregations in DuckDB for the
  descriptive/BI side — monthly no-show trend and volume, rate by clinic, reminder
  effectiveness by lead-time band, a weekday × lead-time heatmap, and patient-history
  segments. This is the analyst layer, separate from the model.
- **Feature engineering** (`features.py`): lead time, leakage-safe prior-no-show counts,
  age bands, day of week, reminders, chronic-condition flags.
- **Modeling** (`model.py`): logistic regression (interpretable) + random forest,
  reporting ROC-AUC / PR-AUC / recall — because accuracy is misleading when only ~20%
  of appointments are no-shows.
- **Explanations**: every risk score comes with its top contributing factors.
- **Action layer**: a gains curve and a ranked list of the highest-risk appointments to
  target with reminders.

## Notes / limitations (interview talking points)

- The train/test split is stratified random; a stricter setup would split by time or by
  patient to fully rule out leakage. Prior-no-show counts are already computed
  leakage-safe (only past appointments).
- Class imbalance is handled with `class_weight="balanced"`; recall is prioritized over
  raw accuracy because missing a no-show is the costly error.
