"""History: recent validations, filters, and cancellation of a mistake (nothing is ever deleted)."""

from datetime import date, timedelta

import streamlit as st

from etf_classifier.app import components, services

PREFIX = "history"

st.set_page_config(page_title="History · ETF Classifier", layout="wide")
settings = components.app_settings()
_, validator, _ = components.sidebar(settings)  # the history does not need the model
state = st.session_state

st.title("Validation history")

if message := state.pop(f"{PREFIX}_message", None):
    st.success(message)

days = settings.get("app", {}).get("history_days", 7)
filters = st.columns(4)
since = filters[0].date_input("Since", value=date.today() - timedelta(days=days), key=f"{PREFIX}_since")
who = filters[1].selectbox("Validator", ["All"] + services.validators(settings), key=f"{PREFIX}_who")
decision = filters[2].selectbox("Decision", ["All", "accepted", "corrected", "no_proposal"], key=f"{PREFIX}_decision")
name = filters[3].text_input("Name contains", key=f"{PREFIX}_name")

try:
    history = services.history(settings, since=since, validator=None if who == "All" else who,
                               decision=None if decision == "All" else decision, name=name or None)
except services.AppError as exc:
    st.error(str(exc))
    st.stop()

if history.empty:
    st.info("No validation for these filters. Validations appear here as soon as they are saved.")
    st.stop()

st.caption(f"{len(history)} validation(s), newest first; {int(history['Cancelled'].sum())} cancelled.")
st.dataframe(history, hide_index=True, width="stretch",
             column_config={"Cancelled": st.column_config.CheckboxColumn("Cancelled"),
                            "CancellationComment": "Cancellation comment"})

st.subheader("LLM justification")
ids = list(history["ValidationId"])
titles = {row.ValidationId: f"{row.Name} ({row.ValidatedBy}, {row.ValidatedAt})" for row in history.itertuples()}
shown = st.selectbox("Validation to inspect", ids, format_func=titles.get, key=f"{PREFIX}_justification_id")
stored = services.justification_for_validation(settings, shown)
if stored is None:
    st.caption("No LLM justification was requested before this validation.")
else:
    components.show_justification(services.stored_justification(stored))
    st.caption(f"GPT-6.1 Sol ({stored['deployment']}), prompt {stored['prompt_version']}, {stored['created_at']}, "
               f"confidence {stored['answer']['confidence']}.")

st.subheader("Cancel a validation")
st.write("The validation stays in the journal with a cancellation line; the ETF leaves the reference "
         "dataset at the next retraining.")
active = history[~history["Cancelled"]]
if active.empty:
    st.info("Every listed validation is already cancelled.")
    st.stop()
labels = {row.ValidationId: f"{row.Name} ({row.ValidatedBy}, {row.ValidatedAt})" for row in active.itertuples()}
choice = st.selectbox("Validation", list(labels), format_func=labels.get, key=f"{PREFIX}_cancel_id")
comment = st.text_input("Why cancel it? (required)", key=f"{PREFIX}_cancel_comment")
confirm = st.checkbox("I confirm the cancellation", key=f"{PREFIX}_cancel_confirm")
if not validator:
    st.warning("Choose a validator in the sidebar to cancel a validation.")
if st.button("Cancel this validation", key=f"{PREFIX}_cancel", disabled=not (validator and confirm)):
    try:
        services.cancel(choice, validator, comment, settings)
    except services.AppError as exc:
        st.error(str(exc))
    else:
        state[f"{PREFIX}_message"] = f"Cancelled: {labels[choice]}."
        for key in ("cancel_comment", "cancel_confirm"):
            state.pop(f"{PREFIX}_{key}", None)
        st.rerun()
