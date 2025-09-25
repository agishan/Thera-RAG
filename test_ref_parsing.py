#!/usr/bin/env python3
"""
Test reference parsing patterns
"""

import re

# Test reference from extracted_citations.json
test_ref = "Andreasen, J.B., Hvas, A.-M., Christiansen, K. & Ravn, H.B. (2011) Can ROTEM analysis be applied for haemostatic monitoring in paediatric congenital heart surgery? Cardiology in the Young , 21 , 684 -691."

# Reference patterns from CitationExtractor
reference_patterns = [
    # Journal articles
    r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^,]+?),?\s*(\d+(?:\(\d+\))?),?\s*([^.]+)',

    # Books
    r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^:]+):\s*([^.]+)',

    # Simple format - FIXED to capture full author list
    r'([A-Z][^(]+?)\s*\((\d{4})\)\s*([^.]+?)(?:\.\s*(.+?))?'
]

print("FIXED PATTERNS:")
fixed_patterns = [
    # MEDICAL JOURNAL: Author list up to year in parentheses
    r'([A-Z][^(]+?)\s+\((\d{4}[a-z]*)\)\s+([^.]+?)(?:\.\s*(.+?))?',

    # ALTERNATIVE: More specific medical format
    r'([A-Z][a-zA-Z\s,.-&]+?)\s+\((\d{4}[a-z]*)\)\s+([^.]+?)(?:\.\s*(.+?))?'
]

print("Testing both original and fixed patterns:")
print()

all_patterns = reference_patterns + fixed_patterns

def parse_authors(author_string: str):
    """Parse author string into list of individual authors"""
    # Remove common suffixes and prefixes
    cleaned = re.sub(r'\s*et\s+al\.?', '', author_string)

    # Split by common separators
    authors = re.split(r',\s*(?=[A-Z])|;\s*|&\s*|\sand\s+', cleaned)

    # Clean each author name
    cleaned_authors = []
    for author in authors:
        author = author.strip()
        if len(author) > 2 and re.search(r'[A-Z]', author):
            cleaned_authors.append(author)

    return cleaned_authors

print("Testing reference parsing:")
print("=" * 60)
print(f"Reference: {test_ref}")
print()

for i, pattern in enumerate(all_patterns, 1):
    print(f"Pattern {i}: {pattern}")
    match = re.search(pattern, test_ref)
    if match:
        groups = match.groups()
        print(f"MATCH! Groups: {groups}")
        if groups:
            print(f"Raw author string (group 0): '{groups[0]}'")
            parsed_authors = parse_authors(groups[0])
            print(f"Parsed authors: {parsed_authors}")
        print()
    else:
        print("NO MATCH")
        print()