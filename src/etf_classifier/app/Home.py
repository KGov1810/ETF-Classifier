"""ETF Classifier: entry point of the interface.

    streamlit run src/etf_classifier/app/Home.py
"""

import streamlit as st

from etf_classifier.app import components

st.set_page_config(page_title="ETF Classifier", layout="wide")

settings = components.app_settings()
bundle, validator, error = components.sidebar(settings)

st.title("ETF Classifier")
st.write("The model proposes a category for each new ETF; you validate or correct it. "
         "Every validation enriches the reference dataset used by the next daily retraining.")

if error:
    st.error(error)
else:
    metrics = bundle.metrics
    left, middle, right = st.columns(3)
    left.metric("Current model", bundle.version)
    middle.metric("Macro F1 on the test set", f"{metrics.get('f1_macro', 0):.3f}",
                  help="Every category counts equally")
    right.metric("Trivial baseline", f"{metrics.get('baseline', {}).get('f1_macro', 0):.3f}",
                 help="Always predicting the most frequent category")

st.subheader("Screens")
st.markdown(
    "- **Single ETF**: enter one ETF, review the proposal and the closest ETFs, validate.\n"
    "- **Batch**: upload a CSV of ETFs (and check their established classification), validate them one by one.\n"
    "- **History**: recent validations; cancel a mistake.\n"
    "- **Monitoring**: acceptance rates, bias alerts and Azure costs."
)
if not validator:
    st.info("Choose your name in the sidebar before validating.")
