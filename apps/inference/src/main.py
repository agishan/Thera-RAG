import streamlit as st
st.set_page_config(page_title="Viscoelastic Testing Guideline Chatbot", layout="wide")

import uuid
from datetime import datetime
import re
from typing import List, Dict, Optional

# Handle both direct run and module import
try:
    from .config import get_config, validate_config
    from .rag_service import RAGService
    from .sheets_service import SheetsService
    from .content_utils import (
        render_enhanced_content,
        get_matched_references_for_text,
        format_reference_line,
        extract_citation_titles_from_chunks,
        load_references,
        extract_citations,
        find_reference_match,
    )
except ImportError:
    # Direct run - use absolute imports
    from config import get_config, validate_config
    from rag_service import RAGService
    from sheets_service import SheetsService
    from content_utils import (
        render_enhanced_content,
        get_matched_references_for_text,
        format_reference_line,
        extract_citation_titles_from_chunks,
        load_references,
        extract_citations,
        find_reference_match,
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

/* Enhanced chunk display styles */
.citation-chunk {
    border-left: 4px solid #3b82f6 !important;
    background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%) !important;
}
.citation-chunk .streamlit-expanderHeader {
    background: #eff6ff !important;
}
.context-chunk {
    border-left: 4px solid #6b7280 !important;
    background: linear-gradient(90deg, #f9fafb 0%, #ffffff 100%) !important;
}
.context-chunk .streamlit-expanderHeader {
    background: #f9fafb !important;
}
.chunk-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.citation-badge {
    background: #dbeafe;
    color: #1e40af;
}
.context-badge {
    background: #f3f4f6;
    color: #374151;
}
.chunk-citations {
    margin-top: 8px;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 6px;
    border-left: 3px solid #e2e8f0;
}
.citation-item {
    font-size: 0.85rem;
    color: #475569;
    margin: 2px 0;
}
</style>
""", unsafe_allow_html=True)

def render_title():
    st.markdown('<div class="app-title">Viscoelastic Testing Guideline Chatbot</div>', unsafe_allow_html=True)

def to_concise(text: str, max_sentences: int = MAX_SENTENCES_IN_CONCISE) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return text.strip() if len(sentences) <= max_sentences else " ".join(sentences[:max_sentences]).strip()

def render_enhanced_content_with_citation_info(content: str):
    """Display content with citation highlighting and extraction results"""
    if not content:
        st.text("No content available")
        return

    # Use the existing enhanced content renderer
    render_enhanced_content(content)

    # Extract and display citation information
    try:
        references = load_references()
        citations = extract_citations(content)

        if citations:
            st.markdown('<div class="chunk-citations">', unsafe_allow_html=True)
            st.markdown("**📖 Citations Found in This Chunk:**")

            citation_list = []
            for citation in citations:
                ref_match = find_reference_match(citation, references)
                if ref_match:
                    citation_text = f"• \"{citation['full_match']}\" → {ref_match.get('title', 'Unknown Title')}"
                    citation_list.append(citation_text)
                else:
                    citation_text = f"• \"{citation['full_match']}\" → [Unmatched]"
                    citation_list.append(citation_text)

            for citation_text in citation_list:
                st.markdown(f'<div class="citation-item">{citation_text}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chunk-citations">📝 No citations found in this chunk</div>', unsafe_allow_html=True)

    except Exception as e:
        st.caption(f"Error processing citations: {e}")


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
    st.session_state.retrieval_k = 15
if "citation_source_k" not in st.session_state:
    st.session_state.citation_source_k = 10
if "citation_display_k" not in st.session_state:
    st.session_state.citation_display_k = 15
if "history" not in st.session_state:
    # list of dicts: {q, a, elapsed, k}
    st.session_state.history = []
# Conversation mode settings
if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prompt_type" not in st.session_state:
    st.session_state.prompt_type = "medical_rag"
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

def render_enhanced_source_chunks(source_docs, citation_source_k, retrieval_k):
    """Enhanced chunk display with citation vs context distinction"""
    if not source_docs:
        st.info("No chunks retrieved.")
        return

    # Section header with transparency metrics
    citation_chunks = min(citation_source_k, len(source_docs))
    context_chunks = max(0, len(source_docs) - citation_chunks)

    st.markdown('<div class="section-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Source Chunks Used for This Answer</div>', unsafe_allow_html=True)
    st.caption(f"Showing all {len(source_docs)} chunks retrieved • {citation_chunks} used for citations • {context_chunks} additional context")

    for i, doc in enumerate(source_docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        snippet = (doc.page_content or "")[:120].replace("\n", " ")

        # Determine if this is a citation source chunk or context-only chunk
        is_citation_chunk = i <= citation_source_k

        # Enhanced title with ranking info and type badge
        rank_info = f"#{meta.get('final_rank', i)}" if 'final_rank' in meta else f"#{i}"
        doc_type = meta.get('doc_type', '')
        title_suffix = f" ({doc_type})" if doc_type else ""

        # Add visual distinction
        if is_citation_chunk:
            chunk_icon = "📚"
            badge_html = '<span class="chunk-badge citation-badge">Citation Source</span>'
            chunk_class = "citation-chunk"
        else:
            chunk_icon = "📄"
            badge_html = '<span class="chunk-badge context-badge">Context Only</span>'
            chunk_class = "context-chunk"

        title_html = f"{chunk_icon} Chunk {rank_info}: {snippet}...{title_suffix}{badge_html}"

        # Wrap the expander in styled container
        st.markdown(f'<div class="{chunk_class}">', unsafe_allow_html=True)
        with st.expander(title_html, expanded=False):
            # Metrics row
            metric_cols = st.columns(4)

            with metric_cols[0]:
                # Enhanced relevance display
                if 'relevance_percent' in meta:
                    st.metric("Relevance", f"{meta['relevance_percent']:.1f}%",
                             help="Combined relevance score")
                elif hasattr(doc, "score") and doc.score is not None:
                    st.metric("Relevance", f"{doc.score:.1%}", help="Vector similarity")

            with metric_cols[1]:
                if 'rerank_score' in meta:
                    st.metric("Re-rank", f"{meta['rerank_score']:.3f}",
                             help="Cross-encoder re-ranking score")

            with metric_cols[2]:
                if 'pinecone_score' in meta:
                    st.metric("Vector", f"{meta['pinecone_score']:.3f}",
                             help="Original Pinecone similarity")

            with metric_cols[3]:
                text_len = len(doc.page_content or "")
                st.metric("Length", f"{text_len}", help="Characters in chunk")

            # Reference information
            ref_info = []

            # Vector/Document ID
            vector_id = next((meta[k] for k in ["vector_id", "id", "_id", "chunk_id"] if k in meta), None)
            if vector_id:
                ref_info.append(f"**ID:** `{vector_id}`")

            # Enhanced source information
            if 'formatted_citation' in meta:
                ref_info.append(f"**Citation:** {meta['formatted_citation']}")
            elif "source" in meta:
                source_text = f"**Source:** {meta['source']}"
                if 'reference_location' in meta:
                    source_text += f" | {meta['reference_location']}"
                ref_info.append(source_text)

            # Add clickable reference if available
            if 'clickable_reference' in meta:
                ref_info.append(f"**Link:** [View Source]({meta['clickable_reference']})")

            # Short citation for copying
            if 'short_citation' in meta:
                ref_info.append(f"**Quick Cite:** {meta['short_citation']}")

            # Display reference info
            if ref_info:
                for info in ref_info:
                    st.caption(info)

            # Content with citation extraction results
            if is_citation_chunk:
                render_enhanced_content_with_citation_info(doc.page_content or "")
            else:
                st.markdown("**Content:**")
                st.text(doc.page_content or "")
                st.markdown('<div class="chunk-citations">ℹ️ No citations extracted (context only)</div>', unsafe_allow_html=True)

            # Debug info (if available)
            debug_info = []
            if 'combined_score' in meta:
                debug_info.append(f"Combined: {meta['combined_score']:.4f}")
            if 'query_used' in meta:
                debug_info.append(f"Query: {meta['query_used'][:50]}...")

            if debug_info and st.checkbox("Show debug info", key=f"debug_enhanced_{vector_id or i}"):
                st.caption(" | ".join(debug_info))

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_source_documents(source_docs):
    if not source_docs:
        st.info("No chunks retrieved.")
        return
    st.markdown('<div class="section-title">Retrieved Chunks</div>', unsafe_allow_html=True)

    for i, doc in enumerate(source_docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        snippet = (doc.page_content or "")[:120].replace("\n", " ")

        # Enhanced title with ranking info
        rank_info = f"#{meta.get('final_rank', i)}" if 'final_rank' in meta else f"#{i}"
        doc_type = meta.get('doc_type', '')
        title_suffix = f" ({doc_type})" if doc_type else ""

        with st.expander(f"Chunk {rank_info}: {snippet}...{title_suffix}", expanded=False):
            # Metrics row
            metric_cols = st.columns(4)

            with metric_cols[0]:
                # Enhanced relevance display
                if 'relevance_percent' in meta:
                    st.metric("Relevance", f"{meta['relevance_percent']:.1f}%",
                             help="Combined relevance score")
                elif hasattr(doc, "score") and doc.score is not None:
                    st.metric("Relevance", f"{doc.score:.1%}", help="Vector similarity")

            with metric_cols[1]:
                if 'rerank_score' in meta:
                    st.metric("Re-rank", f"{meta['rerank_score']:.3f}",
                             help="Cross-encoder re-ranking score")

            with metric_cols[2]:
                if 'pinecone_score' in meta:
                    st.metric("Vector", f"{meta['pinecone_score']:.3f}",
                             help="Original Pinecone similarity")

            with metric_cols[3]:
                text_len = len(doc.page_content or "")
                st.metric("Length", f"{text_len}", help="Characters in chunk")

            # Reference information
            ref_info = []

            # Vector/Document ID
            vector_id = next((meta[k] for k in ["vector_id", "id", "_id", "chunk_id"] if k in meta), None)
            if vector_id:
                ref_info.append(f"**ID:** `{vector_id}`")

            # Enhanced source information
            if 'formatted_citation' in meta:
                ref_info.append(f"**Citation:** {meta['formatted_citation']}")
            elif "source" in meta:
                source_text = f"**Source:** {meta['source']}"
                if 'reference_location' in meta:
                    source_text += f" | {meta['reference_location']}"
                ref_info.append(source_text)

            # Add clickable reference if available
            if 'clickable_reference' in meta:
                ref_info.append(f"**Link:** [View Source]({meta['clickable_reference']})")

            # Short citation for copying
            if 'short_citation' in meta:
                ref_info.append(f"**Quick Cite:** {meta['short_citation']}")

            # Display reference info
            if ref_info:
                for info in ref_info:
                    st.caption(info)

            # Debug info (if available)
            debug_info = []
            if 'combined_score' in meta:
                debug_info.append(f"Combined: {meta['combined_score']:.4f}")
            if 'query_used' in meta:
                debug_info.append(f"Query: {meta['query_used'][:50]}...")

            if debug_info and st.checkbox("Show debug info", key=f"debug_{vector_id or i}"):
                st.caption(" | ".join(debug_info))

            # Content
            render_enhanced_content(doc.page_content or "")

# ---------- Query handling ----------
def answer_query(user_query: str):
    start = datetime.now()

    # Use conversation history if conversation mode is enabled
    chat_history = st.session_state.chat_history if st.session_state.conversation_mode else []

    result = rag_service.get_response(
        user_query,
        chat_history,
        retrieval_k=st.session_state.retrieval_k,
        prompt_type=st.session_state.prompt_type
    )
    elapsed = (datetime.now() - start).total_seconds()

    answer = result["answer"]
    source_docs = result.get("source_documents", [])
    concise = to_concise(answer)

    # Update chat history if in conversation mode
    if st.session_state.conversation_mode:
        st.session_state.chat_history.append((user_query, answer))
        # Keep only last 10 exchanges to avoid context overflow
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]

    return concise, answer, source_docs, elapsed

# Helper to mark submit when Enter is pressed in text_input
def _mark_submit():
    st.session_state.do_submit = True

# ---------- Main ----------
def main():
    render_title()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("🎛️ RAG Controls")

        # Retrieval K slider
        st.session_state.retrieval_k = st.slider(
            "Retrieval K (chunks)",
            min_value=0,
            max_value=30,
            value=st.session_state.retrieval_k,
            help="Number of chunks to retrieve from vector database"
        )

        # Citation source K slider - which chunks to extract citations from
        st.session_state.citation_source_k = st.slider(
            "Citation Source K",
            min_value=1,
            max_value=30,
            value=st.session_state.citation_source_k,
            help="Extract citations only from top K most relevant chunks"
        )

        # Citation display K slider - how many citations to show in UI
        st.session_state.citation_display_k = st.slider(
            "Citation Display K",
            min_value=1,
            max_value=100,
            value=st.session_state.citation_display_k,
            help="Maximum number of citations to display in UI"
        )

        st.divider()

        # Performance metrics (if we have recent query data)
        if st.session_state.history:
            last_query = st.session_state.history[-1]
            st.metric("Last Response Time", f"{last_query.get('elapsed', 0):.2f}s")
            st.metric("Last Retrieval K", last_query.get('k', 'N/A'))

        # Current session info
        st.caption(f"Session ID: {st.session_state.session_id}")
        st.caption(f"Total queries: {len(st.session_state.history)}")

    # Note: Conversation mode infrastructure is enabled but hidden from UI
    # To enable conversation mode later, just set st.session_state.conversation_mode = True
    # To change prompt type, modify st.session_state.prompt_type

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
            # Only extract citations from top citation_source_k most relevant chunks
            citation_titles = extract_citation_titles_from_chunks(source_docs[:st.session_state.citation_source_k])
            if citation_titles:
                # Step 2: Limit display to citation_display_k citations (UI management)
                limited_citations = citation_titles[:st.session_state.citation_display_k]
                total_citations = len(citation_titles)

                st.markdown('<div class="section-title" style="margin-top:12px;">Citations from Sources</div>', unsafe_allow_html=True)
                if total_citations > st.session_state.citation_display_k:
                    st.caption(f"Showing {len(limited_citations)} of {total_citations} citations (from top {st.session_state.citation_source_k} chunks)")

                for i, citation in enumerate(limited_citations, 1):
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

        # Enhanced chunk display with citation vs context distinction
        render_enhanced_source_chunks(source_docs, st.session_state.citation_source_k, st.session_state.retrieval_k)

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
                print("Logged to Google Sheets")
                # st.success("Logged to Google Sheets")

if __name__ == "__main__":
    main()
