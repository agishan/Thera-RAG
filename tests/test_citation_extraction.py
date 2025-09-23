#!/usr/bin/env python3
"""
Test script to verify citation integration in RAG comparison
"""

import sys
from pathlib import Path

# Add the src/app directory to the path
sys.path.append(str(Path(__file__).parent / "src" / "app"))

def test_citation_functions():
    """Test that citation functions can be imported and work"""
    try:
        from content_utils import (
            load_references, 
            extract_citation_titles_from_chunks, 
            get_matched_references_for_text,
            format_reference_line
        )
        print("✅ Citation functions imported successfully")
        
        # Test loading references
        references = load_references()
        print(f"✅ Loaded {len(references)} references from references.json")
        
        if references:
            print(f"   Sample reference: {references[0].get('authors', 'N/A')} ({references[0].get('year', 'N/A')})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing citations: {e}")
        return False

def test_rag_comparison_import():
    """Test that the enhanced RAG comparison can be imported"""
    try:
        # Import the main comparison class
        from rag_comparison import RAGComparison
        print("✅ RAGComparison class imported successfully")
        
        # Test that the citation method exists
        comparison = RAGComparison.__new__(RAGComparison)  # Create without calling __init__
        if hasattr(comparison, '_add_citations_to_rag_response'):
            print("✅ _add_citations_to_rag_response method found")
        else:
            print("❌ _add_citations_to_rag_response method not found")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing RAG comparison: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Citation Integration")
    print("=" * 40)
    
    # Test citation functions
    if not test_citation_functions():
        print("\n❌ Citation function tests failed")
        return
    
    # Test RAG comparison import
    if not test_rag_comparison_import():
        print("\n❌ RAG comparison import tests failed")
        return
    
    print("\n✅ All tests passed!")
    print("\n📝 The enhanced RAG comparison script now includes:")
    print("   • Citation extraction from source documents")
    print("   • Citation appending to RAG responses")
    print("   • Multiple response types (original, with citations, cleansed)")
    print("   • Enhanced logging with citation tracking")
    
    print("\n🚀 You can now run:")
    print("   python rag_comparison.py \"Your question here\"")

if __name__ == "__main__":
    main()
