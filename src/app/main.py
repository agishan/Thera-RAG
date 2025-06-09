import streamlit as st
st.set_page_config(page_title="Thera-RAG Chat", layout="wide")

import os
import json
import pandas as pd
import re
import io
import textwrap
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore

# Google Sheets imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ---------- Google Sheets Setup ---------- #
@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets API connection"""
    # Using service account credentials from environment variable
    google_creds_json = os.getenv("GOOGLE_SHEETS_CREDS_JSON")
    if not google_creds_json:
        st.error("Google Sheets credentials not found. Please set GOOGLE_SHEETS_CREDS_JSON environment variable.")
        return None, None
    
    try:
        creds_dict = json.loads(google_creds_json)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        
        service = build('sheets', 'v4', credentials=creds)
        return service, creds
    except Exception as e:
        st.error(f"Failed to initialize Google Sheets: {e}")
        return None, None


def create_or_get_sheet(service, spreadsheet_id, sheet_name="Chat_Logs"):
    """Create a new sheet if it doesn't exist"""
    try:
        # Get all sheets
        sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheets = sheet_metadata.get('sheets', [])
        
        # Check if our sheet exists
        sheet_exists = any(sheet['properties']['title'] == sheet_name for sheet in sheets)
        
        if not sheet_exists:
            # Create new sheet
            request_body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': sheet_name
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=request_body
            ).execute()
            
            # Add headers
            headers = [['Timestamp', 'Session ID', 'User Query', 'Assistant Response', 'Response Time (s)', 'Answer Length']]
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"{sheet_name}!A1:F1",
                valueInputOption='RAW',
                body={'values': headers}
            ).execute()
            
            # Format headers
            requests = [{
                'repeatCell': {
                    'range': {
                        'sheetId': get_sheet_id(service, spreadsheet_id, sheet_name),
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {'red': 0.2, 'green': 0.5, 'blue': 0.8},
                            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
                            'horizontalAlignment': 'CENTER'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
                }
            }]
            
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': requests}
            ).execute()
            
        return True
    except Exception as e:
        st.error(f"Error creating/accessing sheet: {e}")
        return False


def get_sheet_id(service, spreadsheet_id, sheet_name):
    """Get the sheet ID for a given sheet name"""
    sheet_metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in sheet_metadata.get('sheets', []):
        if sheet['properties']['title'] == sheet_name:
            return sheet['properties']['sheetId']
    return None


def log_to_google_sheets(service, spreadsheet_id, session_id, query, answer, elapsed, sheet_name="Chat_Logs"):
    """Log chat interaction to Google Sheets"""
    if not service:
        return False
    
    try:
        # Prepare the row data
        row_data = [[
            str(datetime.now()),
            session_id,
            query,
            answer[:1000] if len(answer) > 1000 else answer,  # Truncate very long answers
            round(elapsed, 2),
            len(answer)
        ]]
        
        # Append to sheet
        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=f"{sheet_name}!A:F",
            valueInputOption='RAW',
            insertDataOption='INSERT_ROWS',
            body={'values': row_data}
        ).execute()
        
        return True
    except HttpError as error:
        st.error(f"An error occurred logging to Google Sheets: {error}")
        return False


# ---------- helpers ---------- #
def parse_markdown_table(table_text: str) -> pd.DataFrame | None:
    """
    Robust markdown-table to DataFrame converter.
    Handles complex tables with merged cells, multi-line content, and irregular formatting.
    """
    # Normalize indentation and line endings
    table_text = textwrap.dedent(table_text).strip()
    lines = [ln.rstrip() for ln in table_text.splitlines() if ln.strip()]
    
    if len(lines) < 1:  # Changed from 2 to 1 to handle tables without headers
        return None
    
    # Find and remove separator rows (lines with only dashes, colons, spaces, and pipes)
    clean_lines = []
    for line in lines:
        # Skip separator rows like |---|---|---| or |:---|:---:|---:|
        if re.match(r'^\s*\|?\s*[:|\-\s|]+\s*\|?\s*$', line):
            continue
        clean_lines.append(line)
    
    if len(clean_lines) < 1:  # Changed from 2 to 1
        return None
    
    # Process each line to extract cells
    processed_rows = []
    for line in clean_lines:
        # Remove leading/trailing whitespace and pipes
        line = line.strip().strip('|')
        # Split by | and clean each cell
        cells = [cell.strip() for cell in line.split('|')]
        # Don't remove empty cells - they might be intentional (merged cells)
        processed_rows.append(cells)
    
    if not processed_rows:
        return None
    
    # Determine the maximum number of columns
    max_cols = max(len(row) for row in processed_rows)
    
    # Pad all rows to have the same number of columns
    for row in processed_rows:
        while len(row) < max_cols:
            row.append('')
    
    # For tables without clear headers (like your example), use generic headers
    # Check if first row looks like headers vs data
    first_row = processed_rows[0] if processed_rows else []
    
    # If first row has mostly empty cells or looks like data, create generic headers
    empty_cells_in_first = sum(1 for cell in first_row if not cell.strip())
    if empty_cells_in_first > len(first_row) / 2:  # More than half empty
        headers = [f"Column {i+1}" for i in range(max_cols)]
        data_rows = processed_rows  # Use all rows as data
    else:
        # Try to use first row as headers
        headers = first_row
        data_rows = processed_rows[1:] if len(processed_rows) > 1 else []
        
        # If no data rows, still create the table with headers only
        if not data_rows:
            headers = [f"Column {i+1}" for i in range(max_cols)]
            data_rows = processed_rows
    
    # Clean up headers (replace empty headers with generic names)
    clean_headers = []
    for i, header in enumerate(headers[:max_cols]):  # Ensure we don't exceed max_cols
        if header and header.strip():
            clean_headers.append(header.strip())
        else:
            clean_headers.append(f"Column {i+1}")
    
    try:
        df = pd.DataFrame(data_rows, columns=clean_headers)
        return df if not df.empty else None
        
    except Exception as e:
        print(f"DataFrame creation failed: {e}")
        return None


def render_enhanced_content(content: str) -> None:
    """
    Show markdown. Detect and render tables as DataFrames when possible.
    Enhanced detection for various table formats.
    """
    # More comprehensive table detection
    # Look for patterns with multiple lines containing pipes
    table_pattern = r'(\|[^\n]*\|(?:\n\|[^\n]*\|)*)'
    
    # Find all potential table matches
    matches = list(re.finditer(table_pattern, content))
    
    # If no pipe-based tables found, check if the entire content looks like a table
    if not matches and content.count("-") > 20:
        # This might be a dash-based table from PDF extraction
        st.markdown("**Structured Content:**")
        st.code(content, language=None)
        return
    
    if not matches:
        # No tables found, render as regular markdown
        st.markdown(content)
        return
    
    # Process content with tables
    last_end = 0
    
    for match in matches:
        # Render content before the table
        before_table = content[last_end:match.start()].strip()
        if before_table:
            st.markdown(before_table)
        
        # Extract and process the table
        table_content = match.group(1)
        
        # Check if this is actually a table (has multiple lines and sufficient pipes)
        table_lines = [line for line in table_content.split('\n') if line.strip()]
        total_pipes = sum(line.count('|') for line in table_lines)
        
        if len(table_lines) >= 1 and total_pipes >= 6:  # Lowered threshold
            df = parse_markdown_table(table_content)
            if df is not None and not df.empty:
                st.markdown("**Table:**")
                # Enhanced styling for better readability
                styled_df = df.style.set_properties(**{
                    'background-color': 'rgba(0,0,0,0.02)',
                    'border': '1px solid #ddd',
                    'padding': '8px',
                    'text-align': 'left',
                    'white-space': 'pre-wrap',
                    'word-wrap': 'break-word',
                    'vertical-align': 'top'  # Top-align for multi-line content
                }).set_table_styles([
                    {
                        'selector': 'th',
                        'props': [
                            ('background-color', '#f0f2f6'),
                            ('font-weight', 'bold'),
                            ('border', '1px solid #ddd'),
                            ('padding', '10px'),
                            ('white-space', 'pre-wrap'),
                            ('vertical-align', 'top')
                        ]
                    },
                    {
                        'selector': 'td',
                        'props': [
                            ('border', '1px solid #ddd'),
                            ('padding', '8px'),
                            ('white-space', 'pre-wrap'),
                            ('word-wrap', 'break-word'),
                            ('max-width', '400px'),  # Increased for longer content
                            ('vertical-align', 'top')
                        ]
                    }
                ])
                st.dataframe(styled_df, use_container_width=True)
            else:
                # Fallback: show as formatted code
                st.markdown("**Table (raw format):**")
                st.code(table_content, language=None)
        else:
            # Not actually a table, render as markdown
            st.markdown(table_content)
        
        last_end = match.end()
    
    # Render any remaining content after the last table
    remaining_content = content[last_end:].strip()
    if remaining_content:
        st.markdown(remaining_content)


# ---------- environment ---------- #
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

assert PINECONE_API_KEY and GOOGLE_API_KEY, "Missing API keys"

# Initialize Google Sheets
sheets_service, sheets_creds = init_google_sheets()
if sheets_service and GOOGLE_SHEETS_SPREADSHEET_ID:
    create_or_get_sheet(sheets_service, GOOGLE_SHEETS_SPREADSHEET_ID)


# ---------- page ---------- #
st.title("💬 Therapy Assistant (Gemini 1.5 Pro + Pinecone)")


# ---------- chain setup ---------- #
@st.cache_resource
def setup_chain():
    pc = PineconeClient(api_key=PINECONE_API_KEY)
    index_name = "medical-rag-index"
    namespace = "thera-rag"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        max_tokens=8192,
        top_p=0.9,
        top_k=40,
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )


qa_chain = setup_chain()

# ---------- session state ---------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ---------- show past messages ---------- #
for q, a in st.session_state.chat_history:
    st.chat_message("user").markdown(q)
    st.chat_message("assistant").markdown(a)

# ---------- user input ---------- #
user_input = st.chat_input("Ask a question about the documents…")
if user_input:
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Fetching answer…"):
            try:
                start = datetime.now()
                result = qa_chain.invoke(
                    {
                        "question": user_input,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                elapsed = (datetime.now() - start).total_seconds()

                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                st.markdown(answer)

                # ---------- show sources ---------- #
                if source_docs:
                    with st.expander(
                        f"📚 Source Documents ({len(source_docs)} chunks)",
                        expanded=False,
                    ):
                        for i, doc in enumerate(source_docs, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### 📄 Source {i}")
                                with col2:
                                    if hasattr(doc, "score"):
                                        st.metric(
                                            "Relevance",
                                            f"{doc.score:.1%}",
                                            help="Similarity to your query",
                                        )

                                # metadata
                                if hasattr(doc, "metadata") and doc.metadata:
                                    meta = doc.metadata
                                    
                                    # Try to find vector ID in various places
                                    vector_id = None
                                    if "id" in meta:
                                        vector_id = meta['id']
                                    elif "_id" in meta:
                                        vector_id = meta['_id']
                                    elif "chunk_id" in meta:
                                        vector_id = meta['chunk_id']
                                    elif "vector_id" in meta:
                                        vector_id = meta['vector_id']
                                    
                                    # Check if ID might be stored as an attribute of the document
                                    if not vector_id and hasattr(doc, 'id'):
                                        vector_id = doc.id
                                    elif not vector_id and hasattr(doc, '_id'):
                                        vector_id = doc._id
                                    
                                    if vector_id:
                                        st.info(f"🔑 **Vector ID:** `{vector_id}`")
                                    else:
                                        st.warning("⚠️ **Vector ID not found in metadata**")
                                        # Show all available attributes of the document object
                                        st.caption("Available document attributes:")
                                        doc_attrs = [attr for attr in dir(doc) if not attr.startswith('_')]
                                        st.code(", ".join(doc_attrs))
                                    
                                    # Debug: show metadata as JSON
                                    st.caption("📋 **Metadata:**")
                                    st.json(meta)
                                    
                                    if "source" in meta:
                                        doc_name = (
                                            meta["source"]
                                            .replace(".pdf", "")
                                            .replace("_", " ")
                                            .title()
                                        )
                                        st.markdown(f"**Document:** {doc_name}")

                                st.markdown("**Content:**")
                                
                                # Debug info
                                raw_content = doc.page_content.strip()
                                content_length = len(raw_content)
                                st.caption(f"📏 Chunk size: {content_length} characters")
                                
                                # Enhanced table detection
                                # Check for pipe characters and multiple lines
                                has_pipes = "|" in raw_content
                                pipe_count = raw_content.count("|")
                                has_newlines = "\n" in raw_content
                                
                                # Check for dash-based tables (common in PDF extractions)
                                has_many_dashes = raw_content.count("-") > 20
                                has_table_like_structure = (
                                    has_many_dashes and has_newlines and 
                                    len(raw_content.split("\n")) >= 3
                                )
                                
                                # More lenient detection for tables
                                looks_like_table = (
                                    (has_pipes and pipe_count >= 4 and (has_newlines or pipe_count >= 6)) or
                                    has_table_like_structure
                                )

                                if looks_like_table:
                                    st.markdown(
                                        "*📊 This chunk contains tabulated data:*"
                                    )
                                    render_enhanced_content(raw_content)
                                else:
                                    # For content that might still be table-like but doesn't meet criteria
                                    # show it in a code block if it has a structured appearance
                                    if has_many_dashes or (has_newlines and len(raw_content.split("\n")) > 5):
                                        st.markdown("**Structured Content:**")
                                        # Show in code block to preserve formatting
                                        st.code(raw_content, language=None)
                                        
                                        # If it looks incomplete (just dashes/formatting), show warning
                                        if raw_content.replace("-", "").replace("|", "").replace("\n", "").replace(" ", "").strip() == "":
                                            st.warning("⚠️ This chunk appears to contain only formatting characters. The actual content may not have been extracted properly from the source document.")
                                    else:
                                        # Show full content without ANY truncation
                                        # For very long content, use an expander to keep UI clean
                                        if len(raw_content) > 3000:
                                            with st.container():
                                                st.markdown("**Content:** (Large chunk - showing in full)")
                                                # Create a scrollable text area for very long content
                                                st.text_area(
                                                    "Full content:",
                                                    value=raw_content,
                                                    height=400,
                                                    disabled=True,
                                                    label_visibility="collapsed"
                                                )
                                        else:
                                            # For normal-sized content, show directly
                                            st.markdown("**Full Content:**")
                                            st.markdown(raw_content)

                                if i < len(source_docs):
                                    st.divider()

                # update chat history
                st.session_state.chat_history.append((user_input, answer))

                # Log to Google Sheets
                if sheets_service and GOOGLE_SHEETS_SPREADSHEET_ID:
                    success = log_to_google_sheets(
                        sheets_service,
                        GOOGLE_SHEETS_SPREADSHEET_ID,
                        st.session_state.session_id,
                        user_input,
                        answer,
                        elapsed
                    )
                    if success:
                        st.success("✅ Logged to Google Sheets", icon="📊")
                else:
                    st.warning("⚠️ Google Sheets logging not configured")

            except Exception as err:
                st.error(f"Error: {err}")
                st.info(
                    "Please check your API keys, dependencies and internet connection."
                )

# ---------- sidebar ---------- #
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    st.info(
        "**Current Model:** Gemini 1.5 Pro\n\n"
        "**Rate Limits:** 1,000 req/min, no daily cap"
    )

    st.markdown("### 📊 Session Info")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Conversations:** {len(st.session_state.chat_history)}")

    if st.checkbox("Show conversation history"):
        st.markdown("### 📝 Past Q & A")
        if st.session_state.chat_history:
            for i, (q, a) in enumerate(
                reversed(st.session_state.chat_history[-10:]), 1
            ):
                with st.expander(f"{i}. {q[:50]}…"):
                    st.markdown(f"**Question:** {q}")
                    st.markdown(f"**Answer:** {a}")
        else:
            st.markdown("*No conversation history yet.*")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("### 📋 Logging")
    if GOOGLE_SHEETS_SPREADSHEET_ID:
        st.markdown(f"Logs saved to [Google Sheet](https://docs.google.com/spreadsheets/d/{GOOGLE_SHEETS_SPREADSHEET_ID})")
    else:
        st.markdown("⚠️ Google Sheets not configured")
    
    # Show logging status
    if sheets_service:
        st.success("✅ Connected to Google Sheets")
    else:
        st.error("❌ Not connected to Google Sheets")