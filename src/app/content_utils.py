import re
import textwrap
import pandas as pd
import streamlit as st

def parse_markdown_table(table_text):
    """Convert markdown table to DataFrame"""
    table_text = textwrap.dedent(table_text).strip()
    lines = [ln.rstrip() for ln in table_text.splitlines() if ln.strip()]
    
    if len(lines) < 1:
        return None
    
    # Remove separator rows
    clean_lines = []
    for line in lines:
        if re.match(r'^\s*\|?\s*[:|\-\s|]+\s*\|?\s*$', line):
            continue
        clean_lines.append(line)
    
    if len(clean_lines) < 1:
        return None
    
    # Process rows
    processed_rows = []
    for line in clean_lines:
        line = line.strip().strip('|')
        cells = [cell.strip() for cell in line.split('|')]
        processed_rows.append(cells)
    
    if not processed_rows:
        return None
    
    # Make all rows same length
    max_cols = max(len(row) for row in processed_rows)
    for row in processed_rows:
        while len(row) < max_cols:
            row.append('')
    
    # Create headers
    first_row = processed_rows[0]
    empty_cells = sum(1 for cell in first_row if not cell.strip())
    
    if empty_cells > len(first_row) / 2:
        headers = [f"Column {i+1}" for i in range(max_cols)]
        data_rows = processed_rows
    else:
        headers = first_row
        data_rows = processed_rows[1:] if len(processed_rows) > 1 else []
        if not data_rows:
            headers = [f"Column {i+1}" for i in range(max_cols)]
            data_rows = processed_rows
    
    # Clean headers
    clean_headers = []
    for i, header in enumerate(headers[:max_cols]):
        if header and header.strip():
            clean_headers.append(header.strip())
        else:
            clean_headers.append(f"Column {i+1}")
    
    try:
        df = pd.DataFrame(data_rows, columns=clean_headers)
        return df if not df.empty else None
    except Exception:
        return None

def render_enhanced_content(content):
    """Render content with table detection"""
    # Look for table patterns
    table_pattern = r'(\|[^\n]*\|(?:\n\|[^\n]*\|)*)'
    matches = list(re.finditer(table_pattern, content))
    
    # Handle dash-based tables
    if not matches and content.count("-") > 20:
        st.markdown("**Structured Content:**")
        st.code(content, language=None)
        return
    
    if not matches:
        st.markdown(content)
        return
    
    # Process content with tables
    last_end = 0
    
    for match in matches:
        # Content before table
        before_table = content[last_end:match.start()].strip()
        if before_table:
            st.markdown(before_table)
        
        # Process table
        table_content = match.group(1)
        table_lines = [line for line in table_content.split('\n') if line.strip()]
        total_pipes = sum(line.count('|') for line in table_lines)
        
        if len(table_lines) >= 1 and total_pipes >= 6:
            df = parse_markdown_table(table_content)
            if df is not None and not df.empty:
                st.markdown("**Table:**")
                st.dataframe(df, use_container_width=True)
            else:
                st.markdown("**Table (raw format):**")
                st.code(table_content, language=None)
        else:
            st.markdown(table_content)
        
        last_end = match.end()
    
    # Remaining content
    remaining_content = content[last_end:].strip()
    if remaining_content:
        st.markdown(remaining_content)