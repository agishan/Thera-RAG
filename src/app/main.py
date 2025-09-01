import streamlit as st
st.set_page_config(page_title="Viscoelastic Testing Guideline Chatbot", layout="wide")

import uuid
from datetime import datetime
import re
from typing import List, Dict, Optional

from config import get_config, validate_config
from rag_service import RAGService
from sheets_service import SheetsService
from content_utils import (
    render_enhanced_content,
    get_matched_references_for_text,
    format_reference_line,
    extract_citation_titles_from_chunks,
)

from dotenv import load_dotenv
load_dotenv()

# ---------- UI Constants ----------
MAX_SENTENCES_IN_CONCISE = 3

# ---------- Global CSS ----------
st.markdown("""
<style>
/* Centered title - improved visibility */
.app-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    margin: 2rem 0 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb; /* thin underline */
    color: #1f2937;
}
/* General layout polish */
.block-container{ padding-top:1rem }
.section-card{
  border:1px solid #E5E7EB; border-radius:12px; padding:16px; background:#FFFFFF;
  box-shadow:0 2px 10px rgba(0,0,0,0.03); margin-bottom:16px;
}
.section-title{ font-size:18px; font-weight:600; margin:0 0 8px 0 }
.code-like{ background:#F8FAFC; border:1px solid #E5E7EB; border-radius:8px; padding:10px; white-space:pre-wrap }

/* Sidebar typography - prevent text cutoff */
[data-testid="stSidebar"] .stMarkdown{ font-size:0.95rem }
[data-testid="stSidebar"] .stButton button{ font-size:0.9rem; padding:0.5rem 1rem }
[data-testid="stSidebar"] .stTextInput input{ font-size:0.9rem }
[data-testid="stSidebar"] .stNumberInput input{ font-size:0.9rem }

/* Ensure sidebar content doesn't get cut off */
[data-testid="stSidebar"] { overflow-y: auto; max-height: 100vh; }

/* Chat-style input with inline send button */
.input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: .25rem;
}
.input-row .stTextInput {
    flex: 1;
}
/* Compact round icon button */
.send-btn button {
    width: 44px;
    height: 44px;
    border-radius: 9999px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #111827;
    font-weight: 700;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.send-btn button:hover {
    background: #eef2f7;
    border-color: #d1d5db;
}
/* Make the input look like a chat bar */
.input-row input {
    height: 44px;
    border-radius: 9999px;
    padding-left: 14px;
    padding-right: 14px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}
.input-help {
    color: #6b7280;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

/* Hide default form styling (not used now, safe to keep) */
.stForm { border: none !important; padding: 0 !important; }
.stForm > div { border: none !important; padding: 0 !important; }

/* Chat-style text input */
.input-pill input {
    height: 44px;
    border-radius: 9999px;
    padding: 0 14px;
    border: 1px solid #e5e7eb;
    background: #fff;
}

/* Round send button, same height as input */
.send-btn button {
    width: 44px;
    height: 44px;
    border-radius: 9999px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #111827;
    font-weight: 700;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.send-btn button:hover { background: #eef2f7; border-color: #d1d5db; }

/* Vertically center the button within its column */
.send-btn-wrap { display: flex; align-items: center; justify-content: flex-start; }

/* Subtext under the row */
.input-help { color: #6b7280; font-size: 0.85rem; margin: 0.25rem 0 0.75rem; }
</style>
""", unsafe_allow_html=True)

def render_title():
    st.markdown('<div class="app-title">Viscoelastic Testing Guideline Chatbot</div>', unsafe_allow_html=True)

def to_concise(text: str, max_sentences: int = MAX_SENTENCES_IN_CONCISE) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return text.strip() if len(sentences) <= max_sentences else " ".join(sentences[:max_sentences]).strip()

# ---------- Config & services ----------
config = get_config()
validate_config(config)

@st.cache_resource
def init_services():
    rag = RAGService(config)
    sheets = SheetsService(config) if config.get('google_sheets_enabled') else None
    if sheets:
        sheets.setup_sheet()
    return rag, sheets

rag_service, sheets_service = init_services()

# ---------- Session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 30
if "history" not in st.session_state:
    # list of dicts: {q, a, elapsed, k}
    st.session_state.history = []
# A flag to submit when Enter is pressed in the input
if "do_submit" not in st.session_state:
    st.session_state.do_submit = False



# ---------- Sources & Chunks ----------
def render_sources_summary(source_docs):
    if not source_docs:
        st.info("No sources for this query.")
        return
    counts = {}
    for d in source_docs:
        src = (getattr(d, "metadata", {}) or {}).get("source", "Unknown Source")
        counts[src] = counts.get(src, 0) + 1
    st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
    for src, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        st.markdown(f"- {src} — {cnt} chunk(s)")

def render_source_documents(source_docs):
    if not source_docs:
        st.info("No chunks retrieved.")
        return
    st.markdown('<div class="section-title">Retrieved Chunks</div>', unsafe_allow_html=True)
    for i, doc in enumerate(source_docs, 1):
        snippet = (doc.page_content or "")[:120].replace("\n", " ")
        with st.expander(f"Chunk {i}: {snippet}...", expanded=False):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**Chunk {i}**")
            with right:
                if hasattr(doc, "score") and doc.score is not None:
                    st.metric("Relevance", f"{doc.score:.1%}", help="Vector similarity to the query")
            meta = getattr(doc, "metadata", {}) or {}
            vector_id = next((meta[k] for k in ["id", "_id", "chunk_id", "vector_id"] if k in meta), None)
            if vector_id:
                st.caption(f"Vector ID: `{vector_id}`")
            if "source" in meta:
                st.caption(f"Source: {meta['source']}")
            text = (doc.page_content or "").strip()
            st.caption(f"Length: {len(text)} characters")
            render_enhanced_content(text)

# ---------- Query handling ----------
def answer_query(user_query: str):
    start = datetime.now()
    result = rag_service.get_response(
        user_query,
        [],  # one-shot Q&A
        retrieval_k=st.session_state.retrieval_k,
    )
    elapsed = (datetime.now() - start).total_seconds()

    answer = result["answer"]
    source_docs = result.get("source_documents", [])
    concise = to_concise(answer)
    return concise, answer, source_docs, elapsed

# Helper to mark submit when Enter is pressed in text_input
def _mark_submit():
    st.session_state.do_submit = True

# ---------- Main ----------
def main():
    render_title()

    # --- Input row: columns keep widgets on the same line ---
    col_input, col_btn = st.columns([12, 1], gap="small")

    with col_input:
        # Wrap with a class so CSS can style the input as a pill
        with st.container():
            st.markdown('<div class="input-pill">', unsafe_allow_html=True)
            q = st.text_input(
                "Ask a question about the documents",
                value="",
                placeholder="e.g., How should I interpret R-time prolongation on TEG?",
                key="user_query",
                label_visibility="collapsed",
                on_change=_mark_submit  # Enter submits
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col_btn:
        # Vertically center the round button
        st.markdown('<div class="send-btn-wrap">', unsafe_allow_html=True)
        submitted = st.button("➤", key="send_btn", help="Send", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-help">Press Enter to send, or click ➤</div>', unsafe_allow_html=True)

    user_submitted = submitted or st.session_state.do_submit

    # First-visit hint
    if not user_submitted and not st.session_state.history:
        st.info("Enter a question above to get a concise answer with references, plus detailed chunks and sources below.")
        return

    if user_submitted:
        st.session_state.do_submit = False
        if not q.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Fetching answer…"):
            concise_ans, full_ans, source_docs, elapsed = answer_query(q.strip())

        # Answer card
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown(full_ans)

            # Citation titles from chunks, shown in the same block
            citation_titles = extract_citation_titles_from_chunks(source_docs)
            if citation_titles:
                st.markdown('<div class="section-title" style="margin-top:12px;">Citations from Sources</div>', unsafe_allow_html=True)
                for i, citation in enumerate(citation_titles, 1):
                    st.markdown(f"**{i}.** {citation['title']}")
                    st.caption(f"{citation['authors']} ({citation['year']})")
            else:
                # Fallback to answer-based references if no citations found in chunks
                matched_refs = get_matched_references_for_text(full_ans)
                if matched_refs:
                    st.markdown('<div class="section-title" style="margin-top:12px;">References</div>', unsafe_allow_html=True)
                    for m in matched_refs:
                        st.markdown(f"Reference {m['number']}: {format_reference_line(m['ref'])}")

            # Footer meta for the answer card
            st.caption(f"Elapsed: {elapsed:.2f}s • k={st.session_state.retrieval_k}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Sources card
        with st.container():
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            render_sources_summary(source_docs)
            st.markdown('</div>', unsafe_allow_html=True)

        # Log + save to history
        st.session_state.history.append({
            "q": q.strip(),
            "a": concise_ans,
            "elapsed": elapsed,
            "k": st.session_state.retrieval_k
        })
        if sheets_service:
            ok = sheets_service.log_interaction(
                st.session_state.session_id, q.strip(), concise_ans, elapsed, st.session_state.retrieval_k, flagged=False
            )
            if ok:
                st.success("Logged to Google Sheets")

if __name__ == "__main__":
    main()
