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
_MARKDOWN_SPECIALS = set("\\`*_{}[]()<>#+-!|~")


def md_escape(text) -> str:
    """Untrusted text (LLM answers, ETF names and descriptions) shown literally: no link, image or formatting.

    st.markdown, st.info, st.warning and st.caption all render Markdown; a description or an answer could
    otherwise make the browser load an external image (OWASP LLM05: treat model output as untrusted data).
    """
    return "".join("\\" + char if char in _MARKDOWN_SPECIALS else char for char in str(text))
CONFIG_ENV = "ETF_CLASSIFIER_CONFIG"  # optional path to another settings file (used by the tests)


def app_settings() -> dict:
    return load_settings(os.environ.get(CONFIG_ENV))


@st.cache_resource(show_spinner="Loading the model…")
def _cached_model(config_path: str | None, version: str):
    """One model per process and per version: a newly promoted model is picked up automatically.

    `version` is only part of the cache key (the current model is read from the registry).
    """
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
        if missing := services.reference_mismatch(settings):
            reference = settings["feedback"]["reference_csv"]
            st.sidebar.warning(
                f"The current model knows {len(missing)} category(ies) missing from {reference} "
                f"(e.g. {md_escape(missing[0])}): it was trained on another CSV. Validations still work, but the "
                f"daily retraining learns from {reference}: set feedback.reference_csv in config/settings.yaml "
                "to the CSV the model was trained on.")
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


def _established_notice(proposal) -> None:
    """Established classification of an uploaded CSV, checked against the proposal."""
    if not proposal.established:
        return
    established = md_escape(proposal.established)
    if proposal.check == "agreement":
        st.success(f"Established classification **{established}**: agreement with the model.")
    elif proposal.check == "disagreement":
        st.warning(f"Established classification **{established}**: disagreement. The model proposes "
                   f"**{md_escape(proposal.category)}** and gives {established} a score of "
                   f"{proposal.established_score:.2f}.")
    else:
        st.warning(f"Established classification **{established}**: unknown category for the model "
                   "(new or misspelt). Check it, or create it when validating.")


def result_panel(proposal, settings: dict) -> None:
    """Proposed category, established classification, scores, closest ETFs and near-duplicate alert."""
    st.subheader("Proposed category")
    st.markdown(f"**{md_escape(proposal.category)}**")
    best = proposal.scores[0][1] if proposal.scores else 0.0
    if proposal.uncertain:
        threshold = settings["model"]["uncertain_threshold"]
        st.warning(f"Uncertain: the best score ({best:.2f}) is below {threshold:.2f}. "
                   "Check the category carefully, a new category may be needed.")
    _established_notice(proposal)
    if proposal.near_duplicate is not None:
        dup = proposal.near_duplicate
        st.info(f"Near-duplicate of {md_escape(dup.ticker)} · {md_escape(dup.name)} ({dup.overlap_pct:.0f} % of the "
                f"weight in common), categorised {md_escape(', '.join(dup.categories))}.")

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
        shared = md_escape(", ".join(ticker for ticker, _ in neighbor.shared_holdings))
        detail = f"{neighbor.overlap_pct:.0f} % of the weight in common" + (f" ({shared})" if shared else "")
        lines.append(f"- **{md_escape(neighbor.ticker)} · {md_escape(neighbor.name)}**, "
                     f"{md_escape(', '.join(neighbor.categories))}. Score {neighbor.score:.2f}, {detail}")
    st.markdown("\n".join(lines) if lines else "No similar ETF found.")


def _pick(prefix: str, category: str, options: list[str]) -> None:
    """Callback of the one-click category buttons: a known category is selected, an unknown one is created."""
    state = st.session_state
    if category in options:
        state[f"{prefix}_category"] = category
        state[f"{prefix}_create"] = False
    else:
        state[f"{prefix}_create"] = True
        state[f"{prefix}_new_name"] = category


def _do_validate(prefix: str, proposal, settings: dict, validator: str) -> None:
    """Callback of the Validate button: runs before the page is redrawn."""
    state = st.session_state
    create = bool(state.get(f"{prefix}_create"))
    category = (str(state.get(f"{prefix}_new_name", "")).strip() if create
                else state.get(f"{prefix}_category", proposal.category))
    llm = state.get(f"{prefix}_llm")
    try:
        record = services.validate(proposal, category, validator, settings,
                                   comment=state.get(f"{prefix}_comment", ""), create_category=create,
                                   category_description=str(state.get(f"{prefix}_new_description", "")).strip(),
                                   justification_id=llm.justification_id if llm else None)
        state[f"{prefix}_validated"] = record
        state.pop(f"{prefix}_validation_error", None)
    except services.AppError as exc:
        state[f"{prefix}_validation_error"] = str(exc)


N_ALTERNATIVES = 3  # next best categories offered as one-click choices


def validation_form(prefix: str, proposal, settings: dict, validator: str) -> dict | None:
    """Category choice (pre-selected with the proposal) and Validate button. Returns the record once validated."""
    st.subheader("Validate")
    record = st.session_state.get(f"{prefix}_validated")
    done = record is not None
    options = sorted(set(services.known_categories(settings)) | {proposal.category})
    # initial value set through the session state (not `index=`), so that the buttons below can change it
    st.session_state.setdefault(f"{prefix}_category", proposal.category)
    create = st.session_state.get(f"{prefix}_create", False)
    st.selectbox("Category", options, key=f"{prefix}_category", disabled=done or create)

    alternatives = [(c, s) for c, s in proposal.scores[1:N_ALTERNATIVES + 1]]
    if alternatives and not done:
        st.caption("Other likely categories:")
        for i, (category, score) in enumerate(alternatives):  # one per line: long names stay readable
            st.button(f"{category} ({score:.2f})", key=f"{prefix}_alt_{i}", on_click=_pick,
                      args=(prefix, category, options))
    if proposal.established and proposal.established != proposal.category and not done:
        st.button(f"Use the established classification: {proposal.established}", key=f"{prefix}_use_established",
                  on_click=_pick, args=(prefix, proposal.established, options))

    create = st.checkbox("Create a new category", key=f"{prefix}_create", disabled=done)
    if create:
        st.text_input("New category", key=f"{prefix}_new_name")
        st.text_input("Description of the new category", key=f"{prefix}_new_description")
        st.caption("A new category is only proposed by the model after the next retraining, "
                   "once it has enough examples.")
    st.text_input("Comment (optional)", key=f"{prefix}_comment", disabled=done)
    if not validator:
        st.warning("Choose a validator in the sidebar to validate.")
    st.button("Validate", key=f"{prefix}_validate", type="primary", disabled=not validator or done,
              on_click=_do_validate, args=(prefix, proposal, settings, validator))
    if error := st.session_state.get(f"{prefix}_validation_error"):
        st.error(error)
    if done:
        st.success(f"Validated as {record['Category']}, {record['Decision']} (id {record['ValidationId']}).")
        if record.get("warning"):
            st.warning(record["warning"])
    return record


def reset(prefix: str, *keys: str) -> None:
    """Forgets the proposal and validation state of a screen."""
    for key in (*keys, "proposal", "validated", "validation_error", "cost_confirmed", "request", "warnings",
                "category", "create", "new_name", "new_description", "comment", "llm", "llm_estimate", "llm_error"):
        st.session_state.pop(f"{prefix}_{key}", None)


# --- LLM justification ---------------------------------------------------------------------------

VERDICT_LABELS = {"fits": "Fits", "doubtful": "Doubtful", "does_not_fit": "Does not fit"}


@st.cache_resource(show_spinner=False)
def _cached_justifier(config_path: str | None):
    return services.llm_justifier(load_settings(config_path))


def _apply(prefix: str) -> None:
    """Callback of "Apply the suggestion": runs before the widgets are drawn again."""
    state = st.session_state
    justification = state[f"{prefix}_llm"].justification
    state[f"{prefix}_category"] = services.suggested_category(state.get(f"{prefix}_category", ""), justification)
    if justification.new_category:
        state[f"{prefix}_create"] = True
        state[f"{prefix}_new_name"] = justification.new_category["name"]
        state[f"{prefix}_new_description"] = justification.new_category["description"]


def _request_explanation(prefix: str, proposal, settings: dict, justifier) -> None:
    state = st.session_state
    state.pop(f"{prefix}_llm_error", None)
    try:
        estimate = services.llm_estimate(justifier, proposal, settings)
        if estimate.cached:  # already answered: free, no confirmation needed
            state[f"{prefix}_llm"] = services.llm_justify(justifier, proposal, settings)
        else:
            state[f"{prefix}_llm_estimate"] = estimate
    except services.AppError as exc:
        state[f"{prefix}_llm_error"] = str(exc)


def _confirm_explanation(prefix: str, proposal, settings: dict, justifier) -> None:
    state = st.session_state
    state.pop(f"{prefix}_llm_estimate", None)
    try:
        state[f"{prefix}_llm"] = services.llm_justify(justifier, proposal, settings, confirmed=True)
    except services.AppError as exc:
        state[f"{prefix}_llm_error"] = str(exc)


def justification_section(prefix: str, proposal, settings: dict) -> None:
    """"Explain with GPT-6.1 Sol" button, cost confirmation and answer panel."""
    state = st.session_state
    st.subheader("Second opinion")
    justifier = _cached_justifier(os.environ.get(CONFIG_ENV))
    result = state.get(f"{prefix}_llm")
    if result is None:
        st.button("Explain with GPT-6.1 Sol", key=f"{prefix}_explain", disabled=not justifier.available,
                  on_click=_request_explanation, args=(prefix, proposal, settings, justifier))
        if not justifier.available:
            st.caption(f"Unavailable: {justifier.unavailable_reason}")
        estimate = state.get(f"{prefix}_llm_estimate")
        if estimate is not None:
            cost = estimate.cost
            st.info(f"GPT-6.1 Sol reads the ETF, its 10 largest holdings, the proposal, the closest ETFs and the "
                    f"known categories: estimated cost ${cost.cost_usd:.4f} (approximate).")
            st.button("Confirm and explain", key=f"{prefix}_explain_confirm", type="primary",
                      on_click=_confirm_explanation, args=(prefix, proposal, settings, justifier))
        if error := state.get(f"{prefix}_llm_error"):
            st.error(error)
        return

    j = result.justification
    show_justification(j)
    origin = "from the cache, no cost" if result.cached else f"cost ${result.cost_usd:.4f}"
    notes = " " + md_escape(" ".join(j.notes)) if j.notes else ""
    st.caption(f"Confidence: {j.confidence}. Answer {origin}.{notes}")
    if (j.alternative or j.new_category) and state.get(f"{prefix}_validated") is None:
        st.button("Apply the suggestion", key=f"{prefix}_apply", on_click=_apply, args=(prefix,))


def show_justification(j) -> None:
    """Summary, inconsistency, verdict, alternative and new category of a justification (LLM text escaped)."""
    st.markdown(md_escape(j.summary))
    if j.inconsistency:
        st.warning(f"Inconsistency: {md_escape(j.inconsistency)}")
    st.markdown(f"Proposed category: **{VERDICT_LABELS.get(j.verdict, md_escape(j.verdict))}**. "
                f"{md_escape(j.reason)}")
    if j.alternative:
        st.markdown(f"Suggested category: **{md_escape(j.alternative)}**")
    if j.new_category:
        new = j.new_category
        st.markdown(f"Possible new category: **{md_escape(new['name'])}**. "
                    + md_escape(f"{new['description']} {new['reason']}".strip()))
