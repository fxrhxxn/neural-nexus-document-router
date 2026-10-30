"""Streamlit demo V2 (Uddish).

Changes: Added a review queue on the right hand side of the screen.

Run from the project root:
    streamlit run src/demo/app_v2.py
"""
import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

from src.demo.predictor import USING_PLACEHOLDER, predict_document
from src.evaluation.thresholding import route_single_document

# Frozen on the validation set. Update when the team agrees the final value.
TAU = 0.95

st.set_page_config(page_title="Neural Nexus Document Router", layout="wide")
st.title("Neural Nexus Document Router")
st.caption("Upload a scanned document and see where it would be routed.")

if USING_PLACEHOLDER:
    st.warning("Demo mode: the predictor is a placeholder, not the trained "
               "ResNet-18. These results are NOT real.")

# Text-only record of the last documents (kept in memory, resets on refresh).
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_key = None

REASON_TEXT = {"LOW_CONFIDENCE": "low confidence", "HANDWRITTEN": "handwritten"}


def load_scan(uploaded):
    """Return a PIL image from an uploaded PNG/JPG, or from page 1 of a PDF."""
    if uploaded.name.lower().endswith(".pdf"):
        pdf = fitz.open(stream=uploaded.read(), filetype="pdf")
        pix = pdf[0].get_pixmap(dpi=200)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.open(uploaded)


left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader(
        "Drag and drop a scan here (PNG, JPG or PDF - first page only)",
        type=["png", "jpg", "jpeg", "pdf"])

    if uploaded is not None:
        image = load_scan(uploaded)

        # Future work: show the OCR text next to the prediction.

        probs = predict_document(image)
        result = route_single_document(probs, TAU)

        # Scan and result side by side, so everything fits on one screen.
        scan_col, result_col = st.columns([3, 2])

        with scan_col:
            st.image(image, caption="Uploaded scan", width="stretch")

        with result_col:
            st.subheader("Result")
            st.write(f"**Predicted type:** {result['predicted_class']}")
            st.write(f"**Department:** {result['department']}")
            st.write(f"**Confidence:** {result['route_confidence']:.1%}")

            if result["decision"] == "AUTO_ROUTE":
                st.success("Decision: Automatically routed")
            else:
                st.error(f"Decision: Manual review "
                         f"(reason: {result['review_reason']})")

            with st.expander("Class probabilities"):
                st.bar_chart(pd.Series(probs))

        # Add to the history once per new upload (Streamlit re-runs this script
        # on every click, so we remember which file we already added).
        key = (uploaded.name, uploaded.size)
        if st.session_state.last_key != key:
            st.session_state.last_key = key
            st.session_state.history.insert(0, {
                "name": uploaded.name,
                "type": result["predicted_class"],
                "department": result["department"],
                "confidence": result["route_confidence"],
                "decision": result["decision"],
                "reason": result["review_reason"],
            })
            st.session_state.history = st.session_state.history[:5]

with right:
    st.subheader("Review Queue (Last 5 documents)")
    if not st.session_state.history:
        st.caption("Uploaded documents will appear here.")
    for item in st.session_state.history:
        with st.container(border=True):
            st.markdown(f"**{item['name']}**")
            st.write(f"{item['type']} → {item['department']} "
                     f"({item['confidence']:.1%})")
            if item["decision"] == "AUTO_ROUTE":
                st.success("Auto-routed")
            else:
                why = REASON_TEXT.get(item["reason"], item["reason"])
                st.error(f"Manual review: {why}")