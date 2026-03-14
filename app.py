import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from typing import List
from src.pdf_to_images import PDFConverter

# Import your custom modules
# Depending on your exact function signatures, you might need to adjust these imports and calls
from src.superglue_aligner import align_images
from src.gap_analysis import Comparator
from src.field_identification import YOLODetector


st.set_page_config(page_title="Document Analysis Pipeline", layout="wide")

st.title(" Multi-Model Document Analysis")
st.markdown("Upload a reference document and a test document to align, identify fields, and perform gap analysis.")

# --- SIDEBAR FOR SETTINGS ---
st.sidebar.header("Pipeline Settings")

# Remove model selection since we enforce an explicit model dynamically
# Model selection (dynamic reading from weights folder if possible)
selected_model = "armvectores/yolov8n_handwritten_text_detection"


conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.sidebar.markdown("---")
st.sidebar.info("Adjust settings before running the analysis.")

# --- FILE UPLOADERS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Document")
    ref_file = st.file_uploader("Upload Reference Document", type=["jpg", "jpeg", "png", "pdf"], key="ref")

with col2:
    st.subheader("Test Document")
    test_file = st.file_uploader("Upload Test Document", type=["jpg", "jpeg", "png", "pdf"], key="test")

def load_document_pages(uploaded_file) -> List[np.ndarray]:
    # Handle PDF
    if uploaded_file.name.lower().endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        converter = PDFConverter()
        images = converter.convert_to_images(tmp_path)
        os.remove(tmp_path)
        
        if images:
            return images
        else:
            st.error("Failed to extract images from PDF.")
            return []
    # Handle Image
    else:
        image = Image.open(uploaded_file)
        return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)]


def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    vis_img = image.copy()
    for item in detections:
        x1, y1, x2, y2 = map(int, item.get("box", [0, 0, 0, 0]))
        conf = item.get("confidence", 0)
        label = item.get("label", "detected")
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"{label} {conf:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return vis_img

# --- EXECUTION PIPELINE ---
if st.button("Run Analysis", type="primary"):
    if ref_file is None or test_file is None:
        st.error("Please upload both Reference and Test documents.")
    else:
        with st.spinner("Processing..."):
            # 1. Load Images
            ref_pages = load_document_pages(ref_file)
            test_pages = load_document_pages(test_file)

            if not ref_pages or not test_pages:
                st.stop()

            pages_to_process = min(len(ref_pages), len(test_pages))
            if len(ref_pages) != len(test_pages):
                st.warning(
                    f"Page count mismatch: reference has {len(ref_pages)} page(s), "
                    f"test has {len(test_pages)} page(s). Processing first {pages_to_process} page(s)."
                )
            
            # Create tabs for structured output
            tab1, tab2, tab3 = st.tabs(["1. Alignment", "2. Field Identification", "3. Gap Analysis"])

            aligned_pages = []
            ref_fields_by_page = []
            test_fields_by_page = []
            
            # --- STEP 1: ALIGNMENT ---
            with tab1:
                st.markdown("### Document Alignment")
                for idx in range(pages_to_process):
                    ref_img = ref_pages[idx]
                    test_img = test_pages[idx]
                    st.markdown(f"#### Page {idx + 1}")
                    try:
                        aligned_test_img = align_images(ref_img, test_img)
                    except Exception as e:
                        st.error(f"Alignment Error on page {idx + 1}: {e}")
                        aligned_test_img = test_img

                    aligned_pages.append(aligned_test_img)
                    col_a, col_b = st.columns(2)
                    col_a.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption=f"Reference Page {idx + 1}")
                    col_b.image(cv2.cvtColor(aligned_test_img, cv2.COLOR_BGR2RGB), caption=f"Aligned Test Page {idx + 1}")
            
            # --- STEP 2: FIELD IDENTIFICATION ---
            with tab2:
                st.markdown("### YOLO Field Identification")
                try:
                    # Uses local best.pt through YOLODetector implementation
                    detector = YOLODetector(conf_thres=conf_threshold)

                    for idx in range(pages_to_process):
                        ref_img = ref_pages[idx]
                        aligned_test_img = aligned_pages[idx] if idx < len(aligned_pages) else test_pages[idx]

                        ref_fields = detector.detect(ref_img)
                        test_fields = detector.detect(aligned_test_img)

                        ref_fields_by_page.append(ref_fields)
                        test_fields_by_page.append(test_fields)

                        ref_drawn = draw_detections(ref_img, ref_fields)
                        test_drawn = draw_detections(aligned_test_img, test_fields)

                        st.markdown(f"#### Page {idx + 1}")
                        st.caption(
                            f"Reference detections: {len(ref_fields)} | "
                            f"Test detections: {len(test_fields)}"
                        )
                        col_c, col_d = st.columns(2)
                        col_c.image(cv2.cvtColor(ref_drawn, cv2.COLOR_BGR2RGB), caption=f"Reference Fields Page {idx + 1}")
                        col_d.image(cv2.cvtColor(test_drawn, cv2.COLOR_BGR2RGB), caption=f"Test Fields Page {idx + 1}")
                except Exception as e:
                    st.error(f"Field Identification Error: {e}")
                    ref_fields_by_page, test_fields_by_page = [], []
            
            # --- STEP 3: GAP ANALYSIS ---
            with tab3:
                st.markdown("### Gap Analysis Results")
                try:
                    # Using existing Comparator to find gaps
                    comparator = Comparator()
                    all_gaps = []
                    for idx in range(min(len(ref_fields_by_page), len(test_fields_by_page))):
                        page_gaps = comparator.find_missing_data(ref_fields_by_page[idx], test_fields_by_page[idx])
                        for gap in page_gaps:
                            item = gap.copy()
                            item["page"] = idx + 1
                            all_gaps.append(item)

                    if not all_gaps:
                        st.success("No gaps found! Documents match perfectly.")
                    else:
                        st.warning(f"Found {len(all_gaps)} potential gaps/anomalies across {pages_to_process} page(s).")
                        # Display gaps as a dataframe or list
                        st.dataframe(all_gaps)
                except Exception as e:
                    st.error(f"Gap Analysis Error: {e}")
