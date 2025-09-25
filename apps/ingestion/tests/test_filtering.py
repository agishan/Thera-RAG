#!/usr/bin/env python3
"""
Test medical document filtering
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "ingestion"))

from medical_document_processor import MedicalDocumentProcessor

def test_filtering():
    """Test filtering with existing processed text"""

    # Read the existing processed text
    text_file = "enhanced_output/vha-guideline/1_raw_text/full_text.txt"
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    print(f"Original text length: {len(full_text):,} characters")

    # Test medical processor
    processor = MedicalDocumentProcessor(exclude_references=True)
    sections = processor.detect_sections(full_text)
    filtered_content = processor.filter_for_rag(sections)

    print(f"Sections detected: {len(sections)}")
    print(f"Filtered content length: {len(filtered_content):,} characters")
    print(f"Reduction: {(1 - len(filtered_content)/len(full_text))*100:.1f}%")

    # Check if references were filtered out
    has_references = 'references' in sections
    references_size = len(sections.get('references', type('', (), {'content': ''})()).content)

    print(f"References section found: {has_references}")
    print(f"References section size: {references_size:,} chars")

    # Show which sections were included
    included_sections = []
    for name, section in sections.items():
        if name in processor.RAG_RELEVANT_SECTIONS:
            included_sections.append(name)

    print(f"RAG-relevant sections included: {len(included_sections)}")
    print("Sections:", included_sections[:10])  # First 10

    # Show sample of filtered content
    print(f"\nFILTERED CONTENT SAMPLE (first 500 chars):")
    print(filtered_content[:500])

    return {
        'success': True,
        'original_length': len(full_text),
        'filtered_length': len(filtered_content),
        'sections_detected': len(sections),
        'references_filtered': has_references and references_size > 1000
    }

if __name__ == "__main__":
    try:
        results = test_filtering()
        print(f"\nTEST RESULTS:")
        print(f"Success: {results['success']}")
        print(f"Sections detected: {results['sections_detected']}")
        print(f"References filtered: {results['references_filtered']}")
        print(f"Size reduction: {(1-results['filtered_length']/results['original_length'])*100:.1f}%")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()