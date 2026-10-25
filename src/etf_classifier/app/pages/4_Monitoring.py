"""Monitoring: acceptance rates and bias alerts, Azure costs, current model."""

from datetime import date, timedelta

import streamlit as st

from etf_classifier.app import components, services

PREFIX = "monitoring"

st.set_page_config(page_title="Monitoring · ETF Classifier", layout="wide")
settings = components.app_settings()
bundle, _, model_error = components.sidebar(settings)

st.title("Monitoring")
period = st.selectbox("Period", ["Last 30 days", "Last 7 days", "Since the start"], key=f"{PREFIX}_period")
since = {"Last 30 days": date.today() - timedelta(days=30), "Last 7 days": date.today() - timedelta(days=7),
         "Since the start": None}[period]

try:
    feedback, costs = services.monitoring(settings, since)
except services.AppError as exc:
    st.error(str(exc))
    st.stop()

validations, model_tab, costs_tab = st.tabs(["Validations", "Current model", "Azure costs"])

with validations:
    stats = feedback.stats
    if stats["n_validations"] == 0:
        st.info("No validation recorded over this period.")
    else:
        rate = stats["acceptance_rate"]
        cols = st.columns(3)
        cols[0].metric("Validations", stats["n_validations"])
        cols[1].metric("Accepted without change", f"{rate:.0%}" if rate is not None else "–",
                       help="Share of the proposals validated exactly as proposed")
        cols[2].metric("Cancelled", stats["n_cancelled"])
        for alert in stats["bias_alerts"]:
            who = "the whole team" if alert == "global" else alert
            st.warning(f"Possible automatic acceptance by {who}: the acceptance rate is above "
                       f"{settings['feedback']['bias_acceptance_threshold']:.0%}. Proposals may not be reviewed.")
        st.markdown(services.without_title(feedback.markdown))

with model_tab:
    if model_error:
        st.info(model_error)
    else:
        metrics = bundle.metrics
        cols = st.columns(3)
        cols[0].metric("Version", bundle.version)
        cols[1].metric("Macro F1 (test set)", f"{metrics.get('f1_macro', 0):.3f}")
        cols[2].metric("Trivial baseline", f"{metrics.get('baseline', {}).get('f1_macro', 0):.3f}")
        card = bundle.path / "MODEL_CARD.md"
        if card.exists():
            with st.expander("Model card"):
                st.markdown(services.without_title(card.read_text(encoding="utf-8")))

with costs_tab:
    st.markdown(services.without_title(costs.markdown))
