#!/usr/bin/env python3
"""
Test script for medical document processor
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "ingestion"))

from medical_document_processor import MedicalDocumentProcessor, analyze_medical_document
from document_processor import DoclingBookLoader

def test_with_vha_guideline():
    """Test medical processor with VHA guideline"""

    # Extract text using existing pipeline
    print("Extracting text from VHA guideline...")
    doc_processor = DoclingBookLoader("data/vha-guideline.pdf", text_only=True)
    structured_content = doc_processor.extract_structured_content()
    full_text = structured_content["full_text"]

    print(f"Original text length: {len(full_text):,} characters")
    print("=" * 60)

    # Analyze with medical processor
    print("Analyzing medical document structure...")
    analysis = analyze_medical_document(full_text)

    print(f"\n=== DOCUMENT ANALYSIS ===")
    print(f"Total sections detected: {analysis['total_sections']}")
    print(f"Detected sections: {analysis['detected_sections']}")
    print(f"Has standard medical structure: {analysis['has_standard_structure']}")
    print(f"Quality score: {analysis['quality_score']:.2f}")
    print(f"RAG content ratio: {analysis['rag_content_ratio']:.2f}")
    print(f"References section size: {analysis['reference_section_size']:,} chars")

    # Show section details
    print(f"\n=== SECTION BREAKDOWN ===")
    sections = analysis['sections']
    for name, section in sections.items():
        print(f"{name.upper()}: {len(section.content):,} chars (confidence: {section.confidence:.2f})")
        # Show first 100 chars of content
        preview = section.content.replace('\n', ' ').strip()[:100]
        print(f"  Preview: {preview}...")
        print()

    # Test filtering
    processor = MedicalDocumentProcessor(exclude_references=True)
    filtered_content = processor.filter_for_rag(sections)

    print(f"=== RAG FILTERING RESULTS ===")
    print(f"Original length: {len(full_text):,} characters")
    print(f"Filtered length: {len(filtered_content):,} characters")
    print(f"Reduction: {(1 - len(filtered_content)/len(full_text))*100:.1f}%")

    # Show filtered content preview
    print(f"\n=== FILTERED CONTENT PREVIEW ===")
    print(filtered_content[:1000] + "..." if len(filtered_content) > 1000 else filtered_content)

    return {
        'original_length': len(full_text),
        'filtered_length': len(filtered_content),
        'sections': list(sections.keys()),
        'quality_score': analysis['quality_score']
    }

if __name__ == "__main__":
    try:
        results = test_with_vha_guideline()
        print(f"\n=== TEST SUMMARY ===")
        print(f"✅ Section detection: {len(results['sections'])} sections found")
        print(f"✅ Content filtering: {results['filtered_length']:,} chars (RAG-ready)")
        print(f"✅ Quality score: {results['quality_score']:.2f}")
        print(f"✅ Size reduction: {(1-results['filtered_length']/results['original_length'])*100:.1f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()