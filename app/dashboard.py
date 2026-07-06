"""Healthcare No-Show Analytics dashboard.

Sections: KPIs, what drives no-shows, model performance, an interactive risk
scorer with per-appointment explanations, and a "where to act" targeting view.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.config import MODEL_PATH, SCORED_PATH  # noqa: E402
from model import explain_appointment  # noqa: E402
import analytics  # noqa: E402

from app.theme import THEME_CSS, apply_plotly_theme, plotly_chart

st.set_page_config(page_title="Healthcare No-Show Analytics", page_icon="🩺", layout="wide")
apply_plotly_theme(pio)
st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_scored():
    if not Path(SCORED_PATH).exists():
        return None
    return pd.read_parquet(SCORED_PATH)


def rate_bar(df, col, title, order=None):
    g = df.groupby(col, observed=True)["no_show"].mean().reset_index()
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    fig = px.bar(g, x=col, y="no_show", title=title)
    fig.update_yaxes(tickformat=".0%", title="No-show rate")
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.markdown(
        '<div class="hero"><h1>🩺 Healthcare No-Show Analytics</h1>'
        '<p>Which appointments will be missed, and where should the clinic focus '
        'reminders? Analytics + prediction on appointment data.</p></div>',
        unsafe_allow_html=True,
    )

    bundle = load_bundle()
    scored = load_scored()
    if bundle is None or scored is None:
        st.warning("No model found. Run `python build.py` from the project folder, then reload.")
        return

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # ---- KPIs ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Appointments", f"{len(scored):,}")
    k2.metric("No-show rate", f"{scored['no_show'].mean():.1%}")
    k3.metric("Avg lead time", f"{scored['lead_time_days'].mean():.0f} days")
    k4.metric("Reminder coverage", f"{scored['sms_received'].mean():.0%}")
    st.divider()

    tabs = st.tabs(["Operations analytics", "What drives no-shows",
                    "Model performance", "Risk scorer", "Where to act"])

    # ---- operations analytics (SQL / BI) ----
    with tabs[0]:
        st.caption("Descriptive operations analytics — SQL aggregations over the appointment data.")
        trend = analytics.monthly_trend()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend["month"], y=trend["appointments"],
                             name="Appointments", marker_color="#CBD5E1", yaxis="y2"))
        fig.add_trace(go.Scatter(x=trend["month"], y=trend["no_show_rate"],
                                 name="No-show rate", mode="lines+markers",
                                 line=dict(color="#0F766E")))
        fig.update_layout(title="Monthly no-show rate and volume", height=340,
                          yaxis=dict(title="No-show rate", tickformat=".0%"),
                          yaxis2=dict(title="Appointments", overlaying="y",
                                      side="right", showgrid=False),
                          legend=dict(orientation="h", y=-0.2))
        plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            clinic = analytics.by_clinic()
            fig = px.bar(clinic, x="clinic", y="no_show_rate",
                         title="No-show rate by clinic")
            fig.update_yaxes(tickformat=".0%", title="No-show rate")
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            plotly_chart(fig, use_container_width=True)

            seg = analytics.prior_history_segments()
            fig = px.bar(seg, x="prior_no_shows_bucket", y="no_show_rate",
                         title="No-show rate by prior no-show history")
            fig.update_yaxes(tickformat=".0%", title="No-show rate")
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            plotly_chart(fig, use_container_width=True)
        with c2:
            rem = analytics.reminder_effectiveness()
            order = ["0-2d", "3-7d", "8-14d", "15-29d", "30d+"]
            rem["lead_band"] = pd.Categorical(rem["lead_band"], categories=order, ordered=True)
            fig = px.bar(rem.sort_values("lead_band"), x="lead_band", y="no_show_rate",
                         color="reminder", barmode="group",
                         title="Reminder effectiveness by lead time",
                         color_discrete_map={"reminder": "#0F766E", "no reminder": "#DC2626"})
            fig.update_yaxes(tickformat=".0%", title="No-show rate")
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            plotly_chart(fig, use_container_width=True)

            mat = analytics.weekday_leadtime_matrix()
            pivot = mat.pivot(index="day_of_week", columns="lead_band", values="no_show_rate")
            pivot = pivot.reindex(index=dow_order, columns=order)
            fig = px.imshow(pivot, color_continuous_scale="Teal", aspect="auto",
                            title="No-show rate: weekday × lead time", text_auto=".0%")
            fig.update_layout(height=320, margin=dict(t=40, b=10))
            plotly_chart(fig, use_container_width=True)

    # ---- drivers ----
    with tabs[1]:
        scored = scored.copy()
        scored["lead_bucket"] = pd.cut(
            scored["lead_time_days"], [0, 3, 8, 15, 30, 400], right=False,
            labels=["0-2d", "3-7d", "8-14d", "15-29d", "30d+"],
        )
        c1, c2 = st.columns(2)
        with c1:
            rate_bar(scored, "lead_bucket", "No-show rate by lead time")
            rate_bar(scored, "sms_received", "No-show rate by reminder sent")
        with c2:
            rate_bar(scored, "day_of_week", "No-show rate by day of week", order=dow_order)
            if "age_band" in scored:
                rate_bar(scored, "age_band", "No-show rate by age band",
                         order=["child", "youth", "adult", "middle", "senior"])

    # ---- model performance ----
    with tabs[2]:
        m = bundle["metrics"]
        st.caption("Accuracy alone is misleading on an imbalanced target — "
                   "these prioritize ROC-AUC, PR-AUC, and recall on the no-show class.")
        table = pd.DataFrame({
            "Logistic regression": m["logistic_regression"],
            "Random forest": m["random_forest"],
        }).drop(index="confusion").T
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

        y = np.array(bundle["y_test"])
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for name, key in (("Logistic", "prob_lr"), ("Random forest", "prob_rf")):
                fpr, tpr, _ = roc_curve(y, np.array(bundle[key]))
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"),
                                     showlegend=False))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", height=340)
            plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            for name, key in (("Logistic", "prob_lr"), ("Random forest", "prob_rf")):
                prec, rec, _ = precision_recall_curve(y, np.array(bundle[key]))
                fig.add_trace(go.Scatter(x=rec, y=prec, name=name, mode="lines"))
            fig.update_layout(title="Precision-recall curve", xaxis_title="Recall",
                              yaxis_title="Precision", height=340)
            plotly_chart(fig, use_container_width=True)

    # ---- risk scorer ----
    with tabs[3]:
        st.caption("Enter an appointment to get its no-show probability and the factors driving it.")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 110, 35)
        lead = c2.number_input("Lead time (days)", 0, 200, 14)
        prior = c3.number_input("Prior no-shows", 0, 30, 0)
        c4, c5, c6 = st.columns(3)
        gender = c4.selectbox("Gender", ["F", "M"])
        dow = c5.selectbox("Day of week", dow_order)
        sms = c6.selectbox("Reminder sent", [0, 1])
        c7, c8, c9 = st.columns(3)
        htn = c7.selectbox("Hypertension", [0, 1])
        dm = c8.selectbox("Diabetes", [0, 1])
        clinic = c9.selectbox("Clinic", sorted(scored["clinic"].unique()))

        if st.button("Score appointment", type="primary"):
            row = pd.DataFrame([{
                "age": age, "lead_time_days": lead, "prior_no_shows": prior,
                "gender": gender, "day_of_week": dow, "sms_received": sms,
                "hypertension": htn, "diabetes": dm, "clinic": clinic,
            }])
            prob = bundle["lr_pipeline"].predict_proba(row[bundle["features"]])[0, 1]
            st.metric("Predicted no-show probability", f"{prob:.0%}")
            contribs = explain_appointment(bundle, row)
            exp = pd.DataFrame(contribs, columns=["factor", "contribution"])
            exp["effect"] = np.where(exp["contribution"] >= 0, "increases risk", "reduces risk")
            fig = px.bar(exp, x="contribution", y="factor", orientation="h", color="effect",
                         color_discrete_map={"increases risk": "#DC2626", "reduces risk": "#059669"},
                         title="Top factors for this prediction")
            fig.update_layout(height=320, yaxis=dict(autorange="reversed"))
            plotly_chart(fig, use_container_width=True)

    # ---- where to act ----
    with tabs[4]:
        g = bundle["gains_lr"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["frac_pop"], y=g["cum_captured"], mode="lines",
                                 name="Model targeting"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="gray"),
                                 name="Random"))
        fig.update_layout(title="Gains curve — no-shows captured by targeting the riskiest first",
                          xaxis_title="Share of appointments contacted",
                          yaxis_title="Share of no-shows caught", height=360)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        plotly_chart(fig, use_container_width=True)
        st.success(f"Targeting the riskiest 10% of appointments captures "
                   f"{g['top_decile_capture']:.0%} of all no-shows.")
        st.markdown("**Highest-risk appointments to prioritize**")
        cols = ["appointment_id", "clinic", "lead_time_days", "prior_no_shows",
                "sms_received", "no_show_prob"]
        st.dataframe(
            scored.sort_values("no_show_prob", ascending=False)[cols].head(20)
            .style.format({"no_show_prob": "{:.0%}"}),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
