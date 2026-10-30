"""Streamlit demo (Uddish).

Run from the project root:
    streamlit run src/demo/app.py
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

st.set_page_config(page_title="Neural Nexus Document Router")
st.title("Neural Nexus Document Router")
st.caption("Upload a scanned document and see where it would be routed.")

if USING_PLACEHOLDER:
    st.warning("Demo mode: the predictor is a placeholder, not the trained "
               "ResNet-18. These results are NOT real.")

def load_scan(uploaded):
    """Return a PIL image from an uploaded PNG/JPG, or from page 1 of a PDF."""
    if uploaded.name.lower().endswith(".pdf"):
        pdf = fitz.open(stream=uploaded.read(), filetype="pdf")
        pix = pdf[0].get_pixmap(dpi=200)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.open(uploaded)

uploaded = st.file_uploader("Drag and drop a scan here (PNG, JPG or PDF - first page only)",
                            type=["png", "jpg", "jpeg", "pdf"])

if uploaded is not None:
    image = load_scan(uploaded)
    st.image(image, caption="Uploaded scan", use_container_width=True)

    # Future work: show the OCR text next to the prediction.

    probs = predict_document(image)
    result = route_single_document(probs, TAU)

    st.subheader("Result")
    st.write(f"**Predicted type:** {result['predicted_class']}")
    st.write(f"**Department:** {result['department']}")
    st.write(f"**Confidence:** {result['route_confidence']:.1%}")

    if result["decision"] == "AUTO_ROUTE":
        st.success("Decision: Automatically routed")
    else:
        st.error(f"Decision: Manual review (reason: {result['review_reason']})")

    with st.expander("Class probabilities"):
        st.bar_chart(pd.Series(probs))