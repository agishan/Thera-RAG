#!/usr/bin/env python3
"""
Simple test of medical filtering without interactive parts
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "ingestion"))

def test_medical_filtering():
    """Test just the medical filtering component"""

    # Test if we can import the medical processor
    try:
        from medical_document_processor import MedicalDocumentProcessor
        print("[+] Medical processor import successful")
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

    # Test if we have a processed text file to work with
    text_file = project_root / "enhanced_output" / "vha-guideline" / "1_raw_text" / "full_text.txt"
    if not text_file.exists():
        print(f"[ERROR] No processed text file found at: {text_file}")
        print("Run the pipeline first to generate this file")
        return False

    # Read the text
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    print(f"[+] Loaded text: {len(full_text):,} characters")

    # Test medical processor
    processor = MedicalDocumentProcessor(exclude_references=True)

    # Test section detection
    sections = processor.detect_sections(full_text)
    print(f"[+] Sections detected: {len(sections)}")

    # Test filtering
    filtered_content = processor.filter_for_rag(sections)
    reduction_percent = (1 - len(filtered_content)/len(full_text)) * 100

    print(f"[+] Original: {len(full_text):,} chars")
    print(f"[+] Filtered: {len(filtered_content):,} chars")
    print(f"[+] Reduction: {reduction_percent:.1f}%")

    # Show a few sections that were kept
    kept_sections = []
    for name, section in sections.items():
        if name in processor.RAG_RELEVANT_SECTIONS:
            kept_sections.append(f"{name} ({len(section.content)} chars)")

    print(f"[+] Kept sections: {len(kept_sections)}")
    for section in kept_sections[:5]:
        print(f"    - {section}")

    # Show sample of filtered content
    print(f"\n[SAMPLE] First 300 chars of filtered content:")
    print(filtered_content[:300].replace('\n', ' '))

    return True

if __name__ == "__main__":
    success = test_medical_filtering()
    if success:
        print(f"\n[SUCCESS] Medical filtering test completed")
    else:
        print(f"\n[ERROR] Medical filtering test failed")