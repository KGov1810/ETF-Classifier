"""Widgets shared by the pages: settings, sidebar, composition editor, result panel, validation form.

Only display and session state here; every decision is taken by `services`.
"""

import os

import pandas as pd
import streamlit as st

from etf_classifier.app import services
from etf_classifier.config import load_settings
from etf_classifier.model.registry import current_version

NO_VALIDATOR = "—"
CONFIG_ENV = "ETF_CLASSIFIER_CONFIG"  # optional path to another settings file (used by the tests)


def app_settings() -> dict:
    return load_settings(os.environ.get(CONFIG_ENV))


@st.cache_resource(show_spinner="Loading the model…")
def _cached_model(config_path: str | None, version: str):
    """One model per process and per version: a newly promoted model is picked up automatically."""
    return services.load_current_model(load_settings(config_path))


def sidebar(settings: dict):
    """Validator choice and model status. Returns (bundle or None, validator or "", error message or None)."""
    st.sidebar.header("ETF Classifier")
    names = [NO_VALIDATOR] + services.validators(settings)
    # Streamlit forgets widget values when switching pages: the choice is kept in a plain key as well
    if "validator" not in st.session_state and st.session_state.get("_validator") in names:
        st.session_state["validator"] = st.session_state["_validator"]
    choice = st.sidebar.selectbox("Validator", names, key="validator")
    st.session_state["_validator"] = choice
    validator = "" if choice == NO_VALIDATOR else choice

    bundle, error = None, None
    version = current_version(settings)
    if version is None:
        error = "No current model: train one with `python -m etf_classifier.model.train`."
    else:
        try:
            bundle = _cached_model(os.environ.get(CONFIG_ENV), version)
        except services.AppError as exc:
            error = str(exc)
    if bundle is not None:
        st.sidebar.caption(f"Model {bundle.version}")
    if st.sidebar.button("Reload the model", help="After the daily retraining"):
        _cached_model.clear()
        st.rerun()
    return bundle, validator, error


def composition_editor(prefix: str, initial: pd.DataFrame | None = None) -> pd.DataFrame:
    """Editable Ticker / Weight table, with the running total of the weights."""
    if initial is None:  # a few empty rows: a cell can be selected and a block pasted from Excel
        initial = pd.DataFrame({services.TICKER_COLUMN: pd.Series([None] * 5, dtype="object"),
                                services.WEIGHT_COLUMN: pd.Series([None] * 5, dtype="float")})
    st.markdown("Composition")
    table = st.data_editor(
        initial, key=f"{prefix}_holdings", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            services.TICKER_COLUMN: st.column_config.TextColumn("Ticker", help="As in the reference CSV, e.g. AAPL"),
            services.WEIGHT_COLUMN: st.column_config.NumberColumn("Weight (%)", format="%.2f"),
        },
    )
    total = pd.to_numeric(table[services.WEIGHT_COLUMN], errors="coerce").sum()
    filled = table[services.TICKER_COLUMN].fillna("").astype(str).str.strip().ne("").sum()
    st.caption(f"{filled} holding(s), total weight {total:.2f} %. "
               "A partial composition (top holdings only) is fine. You can paste two columns from Excel.")
    return table


def cost_notice(plan) -> None:
    cost = f"${plan.estimate.cost_usd:.4f}" if plan.estimate else "an unknown amount (deployment not in the price list)"
    st.info(f"This model uses Azure embeddings: {plan.n_to_compute} new text(s) to embed, "
            f"estimated cost {cost}. Already-seen texts are free.")


def result_panel(proposal, settings: dict) -> None:
    """Proposed categories, scores, closest ETFs and near-duplicate alert."""
    st.subheader("Proposed categories")
    st.markdown(", ".join(f"**{c}**" for c in proposal.categories))
    best = proposal.scores[0][1] if proposal.scores else 0.0
    if proposal.uncertain:
        threshold = settings["model"]["uncertain_threshold"]
        st.warning(f"Uncertain: the best score ({best:.2f}) is below {threshold:.2f}. "
                   "Check the categories carefully, a new category may be needed.")
    if proposal.near_duplicate is not None:
        dup = proposal.near_duplicate
        st.info(f"Near-duplicate of {dup.name} ({dup.overlap_pct:.0f} % of the weight in common), "
                f"categorised {', '.join(dup.categories)}.")

    scores = pd.DataFrame(proposal.scores, columns=["Category", "Score"])
    st.dataframe(scores, hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0,
                                                                        format="%.2f")})

    st.subheader("Closest ETFs")
    if not proposal.similarity_available:
        st.info("This model has no similarity index: retrain it to see the closest ETFs.")
        return
    if proposal.text_only:
        st.caption("This ETF has no composition: neighbours are compared on the text only.")
    lines = []
    for neighbor in proposal.neighbors:
        shared = ", ".join(ticker for ticker, _ in neighbor.shared_holdings)
        detail = f"{neighbor.overlap_pct:.0f} % of the weight in common" + (f" ({shared})" if shared else "")
        lines.append(f"- **{neighbor.name}**, {', '.join(neighbor.categories)}. "
                     f"Score {neighbor.score:.2f}, {detail}")
    st.markdown("\n".join(lines) if lines else "No similar ETF found.")


def _do_validate(prefix: str, proposal, settings: dict, validator: str) -> None:
    """Callback of the Validate button: runs before the page is redrawn."""
    state = st.session_state
    new_categories = {}
    if state.get(f"{prefix}_create"):
        name = str(state.get(f"{prefix}_new_name", "")).strip()
        if name:
            new_categories[name] = str(state.get(f"{prefix}_new_description", "")).strip()
    try:
        record = services.validate(proposal, state.get(f"{prefix}_categories", []), validator, settings,
                                   comment=state.get(f"{prefix}_comment", ""), new_categories=new_categories)
        state[f"{prefix}_validated"] = record
        state.pop(f"{prefix}_validation_error", None)
    except services.AppError as exc:
        state[f"{prefix}_validation_error"] = str(exc)


def validation_form(prefix: str, proposal, settings: dict, validator: str) -> dict | None:
    """Category choice (pre-ticked with the proposal) and Validate button. Returns the record once validated."""
    st.subheader("Validate")
    record = st.session_state.get(f"{prefix}_validated")
    options = sorted(set(services.known_categories(settings)) | set(proposal.categories))
    st.multiselect("Categories", options, default=proposal.categories, key=f"{prefix}_categories",
                   disabled=record is not None)
    create = st.checkbox("Create a new category", key=f"{prefix}_create", disabled=record is not None)
    if create:
        st.text_input("New category", key=f"{prefix}_new_name")
        st.text_input("Description of the new category", key=f"{prefix}_new_description")
        st.caption("A new category is only proposed by the model after the next retraining, "
                   "once it has enough examples.")
    st.text_input("Comment (optional)", key=f"{prefix}_comment", disabled=record is not None)
    if not validator:
        st.warning("Choose a validator in the sidebar to validate.")
    st.button("Validate", key=f"{prefix}_validate", type="primary", disabled=not validator or record is not None,
              on_click=_do_validate, args=(prefix, proposal, settings, validator))
    if error := st.session_state.get(f"{prefix}_validation_error"):
        st.error(error)
    if record is not None:
        st.success(f"Validated as {record['Decision']} (id {record['ValidationId']}).")
    return record


def reset(prefix: str, *keys: str) -> None:
    """Forgets the proposal and validation state of a screen."""
    for key in (*keys, "proposal", "validated", "validation_error", "cost_confirmed", "request", "warnings"):
        st.session_state.pop(f"{prefix}_{key}", None)
