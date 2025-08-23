import re
import textwrap
import pandas as pd
import streamlit as st
import json
import os

def load_references():
    """Load references from references.json file"""
    try:
        ref_path = os.path.join(os.path.dirname(__file__), 'references.json')
        with open(ref_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load references.json: {e}")
        return []

def extract_citations(text):
    """Extract citations from text using regex patterns"""
    citations = []
    
    # Debug: Print the text being searched
    st.write("🔍 Debug: Searching text for citations...")
    st.write(f"Text length: {len(text)} characters")
    
    # Pattern 1: (Author et al, year) format - handles PDF format with parentheses around entire citation
    pattern1 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4})\)'
    matches1 = re.finditer(pattern1, text)
    for match in matches1:
        authors = match.group(1).strip().rstrip(',').strip()  # Remove trailing comma
        year = match.group(2)
        st.write(f"Pattern 1 found: {match.group(0)}")
        citations.append({
            'authors': authors,
            'year': year,
            'full_match': match.group(0),
            'start': match.start(),
            'end': match.end()
        })
    
    # Pattern 2: (Author, Author & Author, year) format - handles ampersand citations
    pattern2 = r'\(([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*(?:\s*&\s*[A-Z][a-z]+)?)\s*,?\s*(\d{4})\)'
    matches2 = re.finditer(pattern2, text)
    for match in matches2:
        authors = match.group(1).strip()
        year = match.group(2)
        st.write(f"Pattern 2 found: {match.group(0)}")
        # Avoid duplicates
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 3: Handle "et al" without period and with extra spaces - PDF format
    pattern3 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4})\)'
    matches3 = re.finditer(pattern3, text)
    for match in matches3:
        authors = match.group(1).strip().rstrip(',').strip()  # Remove trailing comma
        year = match.group(2)
        st.write(f"Pattern 3 found: {match.group(0)}")
        # Avoid duplicates
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 4: Handle single author with "et al" variations - PDF format
    pattern4 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?)\s*,?\s*(\d{4})\)'
    matches4 = re.finditer(pattern4, text)
    for match in matches4:
        authors = match.group(1).strip().rstrip(',').strip()  # Remove trailing comma
        year = match.group(2)
        st.write(f"Pattern 4 found: {match.group(0)}")
        # Avoid duplicates
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 5: More specific for Kozek-Langenecker style citations - PDF format
    pattern5 = r'\(([A-Z][a-z]+-[A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?)\s*,?\s*(\d{4})\)'
    matches5 = re.finditer(pattern5, text)
    for match in matches5:
        authors = match.group(1).strip().rstrip(',').strip()  # Remove trailing comma
        year = match.group(2)
        st.write(f"Pattern 5 found: {match.group(0)}")
        # Avoid duplicates
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 6: Handle citations with letters in year like "2015a" - PDF format
    pattern6 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4}[a-z]?)\)'
    matches6 = re.finditer(pattern6, text)
    for match in matches6:
        authors = match.group(1).strip().rstrip(',').strip()  # Remove trailing comma
        year = match.group(2)
        st.write(f"Pattern 6 found: {match.group(0)}")
        # Avoid duplicates
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    st.write(f"Total citations found: {len(citations)}")
    return citations

def find_reference_match(citation, references):
    """Find matching reference in the references list"""
    citation_authors = citation['authors'].lower()
    citation_year = citation['year']
    
    # Clean up citation authors (remove extra spaces, normalize)
    citation_authors_clean = re.sub(r'\s+', ' ', citation_authors).strip()
    
    for ref in references:
        ref_authors = ref['authors'].lower()
        ref_year = ref['year']
        
        # Check if year matches
        if ref_year == citation_year:
            # Clean up reference authors
            ref_authors_clean = re.sub(r'\s+', ' ', ref_authors).strip()
            
            # Check if authors match (allowing for variations)
            if citation_authors_clean in ref_authors_clean or ref_authors_clean in citation_authors_clean:
                return ref
            
            # Check for et al. variations
            if 'et al' in citation_authors_clean:
                # Extract first author before "et al"
                first_author = citation_authors_clean.split('et al')[0].strip().rstrip(',').strip()
                if first_author in ref_authors_clean:
                    return ref
            
            # Check for partial matches (first author)
            citation_first = citation_authors_clean.split(',')[0].strip()
            ref_first = ref_authors_clean.split(',')[0].strip()
            if citation_first == ref_first:
                return ref
    
    return None

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
    """Display raw markdown content as a string with citation highlighting"""
    try:
        # Load references
        references = load_references()
        
        # Extract citations
        citations = extract_citations(content)
        
        if not citations:
            # No citations found, display as plain text
            st.markdown("**Content:**")
            st.text(content)
            return
        
        # Display content with citation highlighting
        st.markdown("**Content with detected citations:**")
        
        # Create highlighted content
        highlighted_content = content
        citation_info = []
        unmatched_citations = []
        
        # Sort citations by position (reverse order to avoid index shifting)
        sorted_citations = sorted(citations, key=lambda x: x['start'], reverse=True)
        
        for i, citation in enumerate(sorted_citations):
            # Find matching reference
            ref_match = find_reference_match(citation, references)
            
            if ref_match:
                # Highlight the citation with reference number
                start = citation['start']
                end = citation['end']
                original_text = content[start:end]
                highlighted_text = f"**{original_text}** [📚 Reference {i+1}]"
                
                # Replace in highlighted content
                highlighted_content = highlighted_content[:start] + highlighted_text + highlighted_content[end:]
                
                # Store citation info
                citation_info.append({
                    'citation': citation,
                    'reference': ref_match,
                    'number': i+1
                })
            else:
                # Highlight unmatched citations too
                start = citation['start']
                end = citation['end']
                original_text = content[start:end]
                highlighted_text = f"**{original_text}** [❓ Unmatched]"
                
                # Replace in highlighted content
                highlighted_content = highlighted_content[:start] + highlighted_text + highlighted_content[end:]
                
                # Store unmatched citation info
                unmatched_citations.append({
                    'citation': citation,
                    'number': i+1
                })
        
        # Display highlighted content
        st.markdown(highlighted_content)
        
        # Display matched citation details
        if citation_info:
            st.markdown("**📚 Matched References:**")
            for info in citation_info:
                ref = info['reference']
                st.markdown(f"**Reference {info['number']}: {ref['authors']} ({ref['year']})**")
                st.markdown(f"**Title:** {ref.get('title', 'N/A')}")
                
                # Handle journal information - try different possible field names
                journal_info = ref.get('journal_info') or ref.get('journal') or 'N/A'
                if journal_info and journal_info != 'N/A':
                    st.markdown(f"**Journal:** {journal_info}")
                
                st.markdown(f"**Full Citation:** {ref.get('raw', 'N/A')}")
                st.markdown(f"**Detected as:** {info['citation']['full_match']}")
                st.divider()
        
        # Display unmatched citations
        if unmatched_citations:
            st.markdown("**❓ Unmatched Citations:**")
            for info in unmatched_citations:
                citation = info['citation']
                st.markdown(f"**Unmatched {info['number']}: {citation['authors']} ({citation['year']})**")
                st.markdown(f"**Detected Citation:** {citation['full_match']}")
                st.markdown(f"**Authors:** {citation['authors']}")
                st.markdown(f"**Year:** {citation['year']}")
                st.info("⚠️ This citation was not found in the reference database. Consider adding it to references.json if needed.")
                st.divider()
        
        # Also show raw content
        st.markdown("**Raw Content:**")
        st.text(content)
        
    except Exception as e:
        # If citation processing fails, just display the raw content
        st.error(f"Error processing citations: {str(e)}")
        st.text(content)