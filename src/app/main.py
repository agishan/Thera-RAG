import streamlit as st
st.set_page_config(page_title="Thera-RAG Chat", layout="wide")

import uuid
from datetime import datetime

from config import get_config, validate_config
from rag_service import RAGService
from sheets_service import SheetsService
from content_utils import render_enhanced_content

from dotenv import load_dotenv
load_dotenv()

# Initialize configuration
config = get_config()
validate_config(config)

# Initialize services
@st.cache_resource
def init_services():
    rag = RAGService(config)
    sheets = SheetsService(config) if config.get('google_sheets_enabled') else None
    if sheets:
        sheets.setup_sheet()
    return rag, sheets

rag_service, sheets_service = init_services()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

def render_sidebar():
    """Render sidebar with session info and configuration status"""
    with st.sidebar:
        st.markdown("### 🤖 Model Information")
        st.info("**Current Model:** Gemini 1.5 Pro\n\n**Rate Limits:** 1,000 req/min, no daily cap")

        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        st.markdown(f"**Conversations:** {len(st.session_state.chat_history)}")
        
        # Configuration status
        st.markdown("### 🔐 Configuration Status")
        st.success("✅ Pinecone API Key" if config.get('pinecone_api_key') else "❌ Pinecone API Key")
        st.success("✅ Google API Key" if config.get('google_api_key') else "❌ Google API Key")
        st.success("✅ Google Sheets" if sheets_service else "⚠️ Google Sheets (optional)")

        # Conversation history
        if st.checkbox("Show conversation history"):
            st.markdown("### 📝 Past Q & A")
            if st.session_state.chat_history:
                for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
                    with st.expander(f"{i}. {q[:50]}…"):
                        st.markdown(f"**Question:** {q}")
                        st.markdown(f"**Answer:** {a}")
            else:
                st.markdown("*No conversation history yet.*")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

def render_source_documents(source_docs):
    """Render source documents with enhanced formatting"""
    if not source_docs:
        return
        
    with st.expander(f"📚 Source Documents ({len(source_docs)} chunks)", expanded=False):
        for i, doc in enumerate(source_docs, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### 📄 Source {i}")
                with col2:
                    if hasattr(doc, "score"):
                        st.metric("Relevance", f"{doc.score:.1%}", help="Similarity to your query")

                # Show metadata
                if hasattr(doc, "metadata") and doc.metadata:
                    meta = doc.metadata
                    
                    # Try to find vector ID
                    vector_id = None
                    for key in ["id", "_id", "chunk_id", "vector_id"]:
                        if key in meta:
                            vector_id = meta[key]
                            break
                    
                    if vector_id:
                        st.info(f"🔑 **Vector ID:** `{vector_id}`")
                    
                    if "source" in meta:
                        doc_name = meta["source"].replace(".pdf", "").replace("_", " ").title()
                        st.markdown(f"**Document:** {doc_name}")

                # Show content
                st.markdown("**Content:**")
                raw_content = doc.page_content.strip()
                st.caption(f"📏 Chunk size: {len(raw_content)} characters")
                
                # Use the content renderer for tables and structured content
                render_enhanced_content(raw_content)

                if i < len(source_docs):
                    st.divider()

def handle_user_input(user_input):
    """Handle user input and generate response"""
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Fetching answer…"):
            try:
                start = datetime.now()
                result = rag_service.get_response(user_input, st.session_state.chat_history)
                elapsed = (datetime.now() - start).total_seconds()

                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                st.markdown(answer)
                render_source_documents(source_docs)

                # Update chat history
                st.session_state.chat_history.append((user_input, answer))

                # Log to Google Sheets
                if sheets_service:
                    success = sheets_service.log_interaction(
                        st.session_state.session_id, user_input, answer, elapsed
                    )
                    if success:
                        st.success("✅ Logged to Google Sheets", icon="📊")

            except Exception as err:
                st.error(f"Error: {err}")
                st.info("Please check your API keys, dependencies and internet connection.")

# Main app
def main():
    st.title("💬 Therapy Assistant (Gemini 1.5 Pro + Pinecone)")
    
    render_sidebar()
    
    # Show chat history
    for q, a in st.session_state.chat_history:
        st.chat_message("user").markdown(q)
        st.chat_message("assistant").markdown(a)
    
    # Handle new input
    user_input = st.chat_input("Ask a question about the documents…")
    if user_input:
        handle_user_input(user_input)

if __name__ == "__main__":
    main()