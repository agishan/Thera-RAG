import re
import textwrap
import pandas as pd
import streamlit as st
import json
import os
from typing import List, Dict, Optional

# -------------------------
# Reference loading
# -------------------------
def load_references() -> List[Dict]:
    """Load references from references.json file."""
    try:
        ref_path = os.path.join(os.path.dirname(__file__), 'references.json')
        with open(ref_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load references.json: {e}")
        return []

# -------------------------
# Citation extraction
# -------------------------
def extract_citations(text: str) -> List[Dict]:
    """Extract citations from text using regex patterns."""
    citations: List[Dict] = []
    if not text:
        return citations

    # Pattern 1: (Author et al, year) with optional comma/space noise
    pattern1 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4})\)'
    for match in re.finditer(pattern1, text):
        authors = match.group(1).strip().rstrip(',').strip()
        year = match.group(2)
        citations.append({
            'authors': authors,
            'year': year,
            'full_match': match.group(0),
            'start': match.start(),
            'end': match.end()
        })

    # Pattern 2: (Author, Author & Author, year) ampersand style
    pattern2 = r'\(([A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+)*(?:\s*&\s*[A-Z][a-z]+)?)\s*,?\s*(\d{4})\)'
    for match in re.finditer(pattern2, text):
        authors = match.group(1).strip()
        year = match.group(2)
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    # Pattern 3: handles "et al" variants
    pattern3 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4})\)'
    for match in re.finditer(pattern3, text):
        authors = match.group(1).strip().rstrip(',').strip()
        year = match.group(2)
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    # Pattern 4: single author + et al variants
    pattern4 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?)\s*,?\s*(\d{4})\)'
    for match in re.finditer(pattern4, text):
        authors = match.group(1).strip().rstrip(',').strip()
        year = match.group(2)
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    # Pattern 5: hyphenated surname (e.g., Kozek-Langenecker)
    pattern5 = r'\(([A-Z][a-z]+-[A-Z][a-z]+(?:\s+et\s+al\s*\.?\s*,?\s*)?)\s*,?\s*(\d{4})\)'
    for match in re.finditer(pattern5, text):
        authors = match.group(1).strip().rstrip(',').strip()
        year = match.group(2)
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    # Pattern 6: year with letter suffix (2015a)
    pattern6 = r'\(([A-Z][a-z]+(?:\s+et\s+al\s*,?\s*)?(?:\s*,\s*[A-Z][a-z]+)*)\s*,?\s*(\d{4}[a-z]?)\)'
    for match in re.finditer(pattern6, text):
        authors = match.group(1).strip().rstrip(',').strip()
        year = match.group(2)
        if not any(c['authors'] == authors and c['year'] == year for c in citations):
            citations.append({
                'authors': authors,
                'year': year,
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    # Deduplicate by (authors, year, start)
    unique = {(c['authors'], c['year'], c['start']): c for c in citations}
    citations = list(unique.values())
    citations.sort(key=lambda c: c['start'])
    return citations

def find_reference_match(citation: Dict, references: List[Dict]) -> Optional[Dict]:
    """Find matching reference in the references list."""
    citation_authors = citation['authors'].lower()
    citation_year = citation['year']
    citation_authors_clean = re.sub(r'\s+', ' ', citation_authors).strip()
    
    for ref in references:
        ref_authors = (ref.get('authors') or '').lower()
        ref_year = ref.get('year') or ''
        if ref_year != citation_year:
            continue
        
        ref_authors_clean = re.sub(r'\s+', ' ', ref_authors).strip()
        if citation_authors_clean in ref_authors_clean or ref_authors_clean in citation_authors_clean:
            return ref
        
        if 'et al' in citation_authors_clean:
            first_author = citation_authors_clean.split('et al')[0].strip().rstrip(',').strip()
            if first_author and first_author in ref_authors_clean:
                return ref
        
        citation_first = citation_authors_clean.split(',')[0].strip()
        ref_first = ref_authors_clean.split(',')[0].strip()
        if citation_first and citation_first == ref_first:
            return ref
    return None

# -------------------------
# Markdown table helper
# -------------------------
def parse_markdown_table(table_text: str):
    """Convert markdown table to DataFrame."""
    table_text = textwrap.dedent(table_text).strip()
    lines = [ln.rstrip() for ln in table_text.splitlines() if ln.strip()]
    if len(lines) < 1:
        return None
    
    clean_lines = []
    for line in lines:
        if re.match(r'^\s*\|?\s*[:|\-\s|]+\s*\|?\s*$', line):
            continue
        clean_lines.append(line)
    if len(clean_lines) < 1:
        return None
    
    processed_rows = []
    for line in clean_lines:
        line = line.strip().strip('|')
        cells = [cell.strip() for cell in line.split('|')]
        processed_rows.append(cells)
    if not processed_rows:
        return None
    
    max_cols = max(len(row) for row in processed_rows)
    for row in processed_rows:
        while len(row) < max_cols:
            row.append('')
    
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
    
    clean_headers = []
    for i, header in enumerate(headers[:max_cols]):
        clean_headers.append(header.strip() if header and header.strip() else f"Column {i+1}")
    
    try:
        df = pd.DataFrame(data_rows, columns=clean_headers)
        return df if not df.empty else None
    except Exception:
        return None

# -------------------------
# Chunk-based citation title extraction
# -------------------------
def extract_citation_titles_from_chunks(source_docs: List) -> List[Dict]:
    """
    Extract citation titles from source documents/chunks.
    Returns list of unique citation titles found in the chunks.
    """
    references_db = load_references()
    all_citations = []
    
    for doc in source_docs:
        if not hasattr(doc, 'page_content') or not doc.page_content:
            continue
            
        content = doc.page_content
        citations = extract_citations(content)
        
        for citation in citations:
            ref_match = find_reference_match(citation, references_db)
            if ref_match and ref_match.get('title'):
                # Create a unique key for deduplication
                title = ref_match.get('title', '').strip()
                authors = ref_match.get('authors', '').strip()
                year = ref_match.get('year', '').strip()
                
                citation_info = {
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'citation_text': citation['full_match'],
                    'source_chunk': content[:100] + "..." if len(content) > 100 else content
                }
                
                # Check if this citation is already in our list
                is_duplicate = False
                for existing in all_citations:
                    if (existing['title'] == title and 
                        existing['authors'] == authors and 
                        existing['year'] == year):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_citations.append(citation_info)
    
    # Sort by year (newest first), then by authors
    all_citations.sort(key=lambda x: (x['year'], x['authors']), reverse=True)
    return all_citations

# -------------------------
# Answer-block reference helpers (for main Answer card)
# -------------------------
def format_reference_line(ref: Dict) -> str:
    """
    Render one reference exactly as requested:
    'Anesthesia and Analgesia, 106, 32 –44. Chan, K.L., Summerhayes, R.G., Ignjatovic, V., Horton, S.B. & Monagle, P.T (2007)'
    Falls back to 'raw' if journal_info is missing.
    """
    authors = (ref.get("authors") or "").strip()
    year = (ref.get("year") or "").strip()
    journal_info = (ref.get("journal_info") or ref.get("journal") or "").strip()
    if journal_info:
        return f"{journal_info}. {authors} ({year})"
    raw = ref.get("raw")
    return raw.strip() if isinstance(raw, str) and raw.strip() else f"{authors} ({year})"

def get_matched_references_for_text(content: str) -> List[Dict]:
    """
    Use extract_citations + find_reference_match to return unique, ordered matches.
    Returns: list of { "number": int, "ref": <reference_dict>, "citation": <citation_dict> }
    """
    references_db = load_references()
    citations = extract_citations(content) or []
    if not citations:
        return []

    citations = sorted(citations, key=lambda c: c["start"])
    matched: List[Dict] = []
    seen = set()
    ref_index = 1
    for cit in citations:
        ref = find_reference_match(cit, references_db)
        if not ref:
            continue
        key = (ref.get("authors",""), ref.get("year",""), ref.get("title",""), ref.get("raw",""))
        if key in seen:
            continue
        seen.add(key)
        matched.append({"number": ref_index, "ref": ref, "citation": cit})
        ref_index += 1
    return matched

# -------------------------
# Chunk content rendering with citation highlighting (emoji-free)
# -------------------------
def render_enhanced_content(content: str):
    """
    Display content with citation highlighting for the Chunks view.
    Adds [Reference i] or [Unmatched] after detected in-text citations.
    """
    try:
        references = load_references()
        citations = extract_citations(content)
        
        if not citations:
            st.markdown("**Content:**")
            st.text(content)
            return
        
        st.markdown("**Content with detected citations:**")
        
        highlighted_content = content
        citation_info = []
        unmatched_citations = []
        
        sorted_citations = sorted(citations, key=lambda x: x['start'], reverse=True)
        ref_counter = 1
        ref_map = {}

        for citation in sorted_citations:
            ref_match = find_reference_match(citation, references)
            start, end = citation['start'], citation['end']
            original_text = content[start:end]
            if ref_match:
                if (start, end) not in ref_map:
                    ref_map[(start, end)] = ref_counter
                    ref_counter += 1
                number = ref_map[(start, end)]
                highlighted_text = f"**{original_text}** [Reference {number}]"
                highlighted_content = highlighted_content[:start] + highlighted_text + highlighted_content[end:]
                citation_info.append({
                    'citation': citation,
                    'reference': ref_match,
                    'number': number
                })
            else:
                highlighted_text = f"**{original_text}** [Unmatched]"
                highlighted_content = highlighted_content[:start] + highlighted_text + highlighted_content[end:]
                unmatched_citations.append({'citation': citation})
        
        st.markdown(highlighted_content)
        
        if citation_info:
            st.markdown("**Matched References:**")
            for info in sorted(citation_info, key=lambda x: x['number']):
                ref = info['reference']
                st.markdown(f"**Reference {info['number']}: {ref.get('authors','N/A')} ({ref.get('year','N/A')})**")
                st.markdown(f"**Title:** {ref.get('title', 'N/A')}")
                journal_info = ref.get('journal_info') or ref.get('journal') or 'N/A'
                if journal_info and journal_info != 'N/A':
                    st.markdown(f"**Journal:** {journal_info}")
                st.markdown(f"**Full Citation:** {ref.get('raw', 'N/A')}")
                st.markdown(f"**Detected as:** {info['citation']['full_match']}")
                st.divider()
        
        if unmatched_citations:
            st.markdown("**Unmatched Citations:**")
            for info in unmatched_citations:
                citation = info['citation']
                st.markdown(f"**Detected Citation:** {citation['full_match']}")
                st.markdown(f"**Authors:** {citation['authors']}")
                st.markdown(f"**Year:** {citation['year']}")
                st.info("This citation was not found in the reference database. Consider adding it to references.json.")
                st.divider()
        
        st.markdown("**Raw Content:**")
        st.text(content)
        
    except Exception as e:
        st.error(f"Error processing citations: {str(e)}")
        st.text(content)
