import numpy as np
import streamlit as st
from PIL import Image

from ocr_utils import extract_structured_info_with_llm, extract_text

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="🪪 Vietnamese ID Card OCR", layout="centered")
st.title("🪪 Vietnamese ID Card OCR")

st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an ID card image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    with st.expander("📸 Uploaded Image Preview", expanded=True):
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # -------------------------------
    # OCR Text Extraction
    # -------------------------------
    with st.spinner("🔍 Extracting text..."):
        text = extract_text(image_np)

    with st.expander("📝 Raw OCR Output", expanded=False):
        st.text_area("OCR Detected Text", value=text, height=200)

    # -------------------------------
    # LLM-Based Parsing
    # -------------------------------
    st.subheader("🤖 Smart Info Extraction (LLM-Powered)")

    with st.spinner("📡 Calling LLM to extract structured fields..."):
        llm_info = extract_structured_info_with_llm(text)

    if "error" in llm_info:
        st.error(f"LLM Error: {llm_info['error']}")
    else:
        with st.expander("📄 Extracted Fields", expanded=True):
            for key, value in llm_info.items():
                col1, col2 = st.columns([1, 3])
                col1.markdown(f"**{key}**")
                col2.markdown(
                    f": {', '.join(value) if isinstance(value, list) else value or 'Not Found'}"
                )
