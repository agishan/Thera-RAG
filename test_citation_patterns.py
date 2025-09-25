#!/usr/bin/env python3
"""
Test citation patterns against actual chunk content
"""

import re

# Test content from chunk 5
test_content = """## Sample type and pre-analytical issues

Native whole blood samples need immediate analysis. Anticoagulated samples should be analysed within four hours of venesection. No specific rest period is recommended by manufacturers prior to analysing citrated blood samples. Users of ROTEM devices have variously recommended resting samples between 30 (Andreasen et al , 2011; Armstrong et al , 2011) and 120 (Theusinger et al , 2010) minutes, whilst others report immediate testing (Oswald et al , 2010; Ogawa et al , 2012a,b).

Pragmatically, it is reasonable to perform immediate testing for all VHA tests."""

# Enhanced patterns from CitationExtractor
inline_patterns = [
    # MEDICAL FORMAT: Multiple citations with semicolons (PRIMARY PATTERN)
    r'\(([^)]+(?:et\s+al\s*[.,]?|[A-Z][a-zA-Z]+)\s*,\s*\d{4}[a-z]*(?:,[a-z])*(?:\s*;\s*[^)]+)*?)\)',

    # MEDICAL FORMAT: Single citation in parentheses with flexible spacing
    r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\s*[.,]?)?\s*,\s*\d{4}[a-z]*(?:,[a-z]*)?)\)',

    # MEDICAL FORMAT: Multiple authors with &
    r'\(([A-Z][a-zA-Z]+(?:\s+&\s+[A-Z][a-zA-Z]+)+)\s*,\s*(\d{4}[a-z]*)\)',

    # ACADEMIC FORMAT: Author outside parentheses
    r'([A-Z][a-zA-Z]+(?:\s+et\s+al\s*[.,]?)?\s*)\((\d{4}[a-z]*)\)',
]

print("Testing citation patterns against chunk 5 content:")
print("=" * 60)

for i, pattern in enumerate(inline_patterns, 1):
    print(f"\nPattern {i}: {pattern}")
    matches = re.findall(pattern, test_content)
    if matches:
        print(f"FOUND {len(matches)} matches:")
        for match in matches:
            print(f"   - {match}")
    else:
        print("NO matches found")

print("\n" + "=" * 60)
print("ALL CITATIONS FOUND WITH finditer():")

for i, pattern in enumerate(inline_patterns, 1):
    matches = list(re.finditer(pattern, test_content))
    for match in matches:
        print(f"Pattern {i}: '{match.group(0)}' at position {match.start()}-{match.end()}")