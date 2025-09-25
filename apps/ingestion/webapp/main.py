import os
from pathlib import Path
from typing import Optional

import streamlit as st

# Import controller with fallback for script execution
try:
    from .controller import run_stage1, run_stage1_5
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from controller import run_stage1, run_stage1_5


st.set_page_config(page_title="Thera‑RAG Ingestion (HITL)", layout="wide")
st.title("Thera‑RAG Ingestion — Human‑in‑the‑Loop")


def _save_uploaded_file(uploaded_file) -> Optional[Path]:
    if not uploaded_file:
        return None
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = uploaded_file.name if uploaded_file.name.lower().endswith(".pdf") else f"uploaded_{uploaded_file.name}.pdf"
    target = data_dir / safe_name
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def sidebar_controls():
    st.sidebar.header("Input & Options")
    uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    # Or select from data/
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    pdfs = sorted([p.name for p in data_dir.glob("*.pdf")])
    existing = st.sidebar.selectbox("Or choose from data/", ["(none)"] + pdfs)

    text_only = st.sidebar.checkbox("Text‑only extraction (faster)", value=True)
    use_medical_filter = st.sidebar.checkbox("Enable Medical Filtering (Stage 1.5)", value=True)

    options = {
        "text_only": text_only,
        "use_medical_filter": use_medical_filter,
    }

    start_btn = st.sidebar.button("Start / Reset Session", type="primary")
    return uploaded, existing, options, start_btn


def init_session():
    for key in [
        "pdf_path",
        "stage1",
        "stage1_approved",
        "stage15",
        "stage15_approved",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None


def main():
    init_session()
    uploaded, existing, options, start_btn = sidebar_controls()

    # Handle start/reset
    if start_btn:
        st.session_state["stage1"] = None
        st.session_state["stage1_approved"] = None
        st.session_state["stage15"] = None
        st.session_state["stage15_approved"] = None
        st.session_state["pdf_path"] = None

        if uploaded is not None:
            saved = _save_uploaded_file(uploaded)
            st.session_state["pdf_path"] = str(saved)
        elif existing and existing != "(none)":
            st.session_state["pdf_path"] = str(Path("data") / existing)
        else:
            st.warning("Upload or choose a PDF to begin.")

    pdf_path = st.session_state.get("pdf_path")
    if not pdf_path:
        st.info("Use the sidebar to upload or select a PDF, then click Start / Reset Session.")
        return

    st.markdown(f"**Selected PDF:** `{pdf_path}`")
    st.divider()

    # Stage 1 — Docling extraction
    st.header("Stage 1 — Extraction (Docling)")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.get("stage1") is None and st.button("Run Stage 1"):
            with st.spinner("Extracting text with Docling..."):
                st.session_state["stage1"] = run_stage1(pdf_path, text_only=options["text_only"])
                st.session_state["stage1_approved"] = None

        if st.session_state.get("stage1") is not None and st.session_state.get("stage1_approved") is None:
            if st.button("Approve Stage 1", type="primary"):
                st.session_state["stage1_approved"] = True

    if st.session_state.get("stage1"):
        res = st.session_state["stage1"]
        stats = res["stats"]
        st.caption(f"Characters: {stats['characters']:,} • Words: {stats['words']:,} • Lines: {stats['lines']:,} • Paragraphs: {stats['paragraphs']}")
        # Preview first 3 paragraphs
        paragraphs = [p.strip() for p in (res["full_text"] or "").split("\n\n") if p.strip()][:3]
        for i, para in enumerate(paragraphs, 1):
            with st.expander(f"Paragraph {i}", expanded=(i == 1)):
                st.write(para)

    st.divider()

    # Stage 1.5 — Medical Filtering
    st.header("Stage 1.5 — Medical Filtering (Optional)")
    if not st.session_state.get("stage1_approved"):
        st.info("Approve Stage 1 to proceed.")
        return

    if not options["use_medical_filter"]:
        st.warning("Medical Filtering is disabled in the sidebar. Enable it to run Stage 1.5.")
        return

    col3, col4 = st.columns([1, 1])
    with col3:
        if st.session_state.get("stage15") is None and st.button("Run Stage 1.5"):
            with st.spinner("Filtering medical content..."):
                stage1 = st.session_state["stage1"]
                st.session_state["stage15"] = run_stage1_5(stage1["full_text"], stage1["stage_dir"]) 
                st.session_state["stage15_approved"] = None

        if st.session_state.get("stage15") is not None and st.session_state.get("stage15_approved") is None:
            if st.button("Approve Stage 1.5", type="primary"):
                st.session_state["stage15_approved"] = True

    if st.session_state.get("stage15"):
        res15 = st.session_state["stage15"]
        stats15 = res15["stats"]
        st.caption(
            f"Sections: {stats15['sections_detected']} • Reduction: {stats15['reduction_percent']:.1f}% • Filtered length: {stats15['filtered_length']:,} chars"
        )

        # Section names preview
        sections = list(res15["sections"].keys())
        if sections:
            st.markdown("**Detected Sections (first 10):** " + ", ".join(sections[:10]) + (" …" if len(sections) > 10 else ""))

        # Filtered content preview
        lines = (res15["filtered_text"] or "").split("\n")
        preview = [ln for ln in lines if ln.strip()][:5]
        with st.expander("Filtered Content Preview", expanded=True):
            st.write("\n".join(preview))

    st.info("Next stages (Chunking, Metadata, Citations, Embeddings, Upload) can be added similarly.")


if __name__ == "__main__":
    main()
