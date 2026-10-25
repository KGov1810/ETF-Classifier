"""Batch: upload a CSV of new ETFs, then validate them one by one in a queue."""

import hashlib

import pandas as pd
import streamlit as st

from etf_classifier.app import components, services

PREFIX = "batch"
FORM_KEYS = ("categories", "create", "new_name", "new_description", "comment", "validated", "validation_error",
             "llm", "llm_estimate", "llm_error")

st.set_page_config(page_title="Batch · ETF Classifier", layout="wide")
settings = components.app_settings()
bundle, validator, error = components.sidebar(settings)
state = st.session_state

st.title("Validate a batch of ETFs")
if error:
    st.error(error)
    st.stop()

uploaded = st.file_uploader("CSV of new ETFs (Name;Description;Composition, UTF-8, ';' separator)",
                            type=["csv"], key=f"{PREFIX}_file")
if uploaded is None:
    st.info("Upload a CSV to start. The proposals are computed for the whole batch, then you validate "
            "the ETFs one by one; each validation is saved immediately.")
    st.stop()

content = uploaded.getvalue()
fingerprint = hashlib.sha256(content).hexdigest()
if state.get(f"{PREFIX}_fingerprint") != fingerprint:  # new file: forget the previous queue
    for key in [k for k in state if str(k).startswith(f"{PREFIX}_") and k != f"{PREFIX}_file"]:
        del state[key]
    state[f"{PREFIX}_fingerprint"] = fingerprint

try:
    batch = services.load_batch(content, settings)
except services.AppError as exc:
    st.error(str(exc))
    st.stop()

if batch.exclusions:
    st.warning(f"{len(batch.exclusions)} row(s) refused; fix them in the file if needed.")
    st.dataframe(pd.DataFrame(batch.exclusions), hide_index=True, width="stretch")
if batch.data.empty:
    st.error("No usable ETF in this file.")
    st.stop()


def _clear_form() -> None:
    for key in FORM_KEYS:
        state.pop(f"{PREFIX}_{key}", None)


def _go_to(step: int) -> None:
    _clear_form()
    state[f"{PREFIX}_step"] = step


def _skip(step: int) -> None:
    state.setdefault(f"{PREFIX}_skipped", set()).add(state[f"{PREFIX}_order"][step])
    _go_to(step + 1)


proposals = state.get(f"{PREFIX}_proposals")
if proposals is None:
    plan = services.estimate(batch.data, bundle, settings)
    label = "Propose categories"
    if plan is not None and plan.n_to_compute:
        components.cost_notice(plan)
        label = "Confirm the cost and propose categories"
    st.write(f"{len(batch.data)} ETF(s) ready.")
    uncertain_first = st.checkbox("Show uncertain ETFs first", value=True, key=f"{PREFIX}_uncertain_first")
    if not st.button(label, key=f"{PREFIX}_propose", type="primary"):
        st.stop()
    with st.spinner("Computing the proposals…"):
        proposals = services.propose(batch.data, bundle, settings, confirmed=True)
    state[f"{PREFIX}_proposals"] = proposals
    state[f"{PREFIX}_order"] = services.queue_order(proposals, uncertain_first)
    state[f"{PREFIX}_step"] = 0
    state[f"{PREFIX}_done"] = {}
    st.rerun()  # redraw without the upload controls

order, step, done = state[f"{PREFIX}_order"], state[f"{PREFIX}_step"], state[f"{PREFIX}_done"]
skipped = state.get(f"{PREFIX}_skipped", set())
n_uncertain = sum(p.uncertain for p in proposals)

if step >= len(order):
    st.success(f"Batch finished: {len(done)} validated, {len(skipped)} skipped.")
    if skipped:
        st.caption("Skipped ETFs: " + components.md_escape(", ".join(proposals[i].name for i in sorted(skipped))))
    st.stop()

index = order[step]
proposal = proposals[index]
st.progress(step / len(order))
st.caption(f"ETF {step + 1} / {len(order)}. {n_uncertain} uncertain in the batch; "
           f"{len(done)} validated, {len(skipped)} skipped so far.")
st.header(components.md_escape(proposal.name))
if proposal.description:
    st.markdown(components.md_escape(proposal.description))

with st.expander("Correct the composition before validating"):
    current = pd.DataFrame(list(proposal.composition.items()), columns=[services.TICKER_COLUMN, services.WEIGHT_COLUMN])
    table = components.composition_editor(f"{PREFIX}_{index}", current)
    if st.button("Update the proposal", key=f"{PREFIX}_update"):
        checked = services.etf_from_inputs(proposal.name, proposal.description, table, settings)
        if checked.errors:
            for message in checked.errors:
                st.error(message)
        else:
            proposals[index] = services.propose(checked.data, bundle, settings, confirmed=True)[0]
            _clear_form()
            st.rerun()

left, right = st.columns([3, 2], gap="large")
with left:
    components.result_panel(proposal, settings)
    components.justification_section(PREFIX, proposal, settings)
with right:
    record = components.validation_form(PREFIX, proposal, settings, validator)
    if record is not None:
        done[index] = record["ValidationId"]
    is_last = step + 1 >= len(order)
    nav_left, nav_right = st.columns(2)
    nav_left.button("Finish" if is_last else "Next ETF", key=f"{PREFIX}_next", disabled=record is None,
                    on_click=_go_to, args=(step + 1,))
    nav_right.button("Skip", key=f"{PREFIX}_skip", disabled=record is not None, on_click=_skip, args=(step,))
