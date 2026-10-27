"""Single ETF: enter an ETF, review the proposal and the closest ETFs, validate."""

import streamlit as st

from etf_classifier.app import components, services

PREFIX = "single"

st.set_page_config(page_title="Single ETF · ETF Classifier", layout="wide")
settings = components.app_settings()
bundle, validator, error = components.sidebar(settings)

st.title("Categorise one ETF")
if error:
    st.error(error)
    st.stop()

ticker_column, name_column = st.columns([1, 3])
ticker = ticker_column.text_input("Ticker", key=f"{PREFIX}_ticker", help="Identifies the ETF (required)")
name = name_column.text_input("Name", key=f"{PREFIX}_name")
description = st.text_area("Description", key=f"{PREFIX}_description", height=90,
                           help="Optional when a composition is given")
holdings = components.composition_editor(PREFIX, st.session_state.get(f"{PREFIX}_holdings_init"))
signature = (ticker, name, description, holdings.to_json())

if st.button("Propose a category", key=f"{PREFIX}_propose", type="primary"):
    components.reset(PREFIX)
    st.session_state[f"{PREFIX}_request"] = signature

request = st.session_state.get(f"{PREFIX}_request")
if request is None:
    st.stop()
if request != signature:
    st.info("The ETF has changed since the last proposal: propose again.")
    st.stop()

proposal = st.session_state.get(f"{PREFIX}_proposal")
if proposal is None:
    checked = services.etf_from_inputs(ticker, name, description, holdings, settings)
    for message in checked.errors:
        st.error(message)
    if checked.errors:
        st.stop()
    st.session_state[f"{PREFIX}_warnings"] = checked.warnings
    plan = services.estimate(checked.data, bundle, settings)
    confirmed = st.session_state.get(f"{PREFIX}_cost_confirmed", False)
    if plan is not None and plan.n_to_compute and not confirmed:
        components.cost_notice(plan)
        if not st.button("Confirm and continue", key=f"{PREFIX}_confirm_cost"):
            st.stop()
        st.session_state[f"{PREFIX}_cost_confirmed"] = confirmed = True
    with st.spinner("Computing the proposal…"):
        proposal = services.propose(checked.data, bundle, settings, confirmed=confirmed)[0]
    st.session_state[f"{PREFIX}_proposal"] = proposal

for message in st.session_state.get(f"{PREFIX}_warnings", []):
    st.warning(components.md_escape(message))
left, right = st.columns([3, 2], gap="large")
with left:
    components.result_panel(proposal, settings)
    components.justification_section(PREFIX, proposal, settings)
with right:
    components.validation_form(PREFIX, proposal, settings, validator)
