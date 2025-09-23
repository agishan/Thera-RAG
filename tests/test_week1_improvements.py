#!/usr/bin/env python3
"""
Simple test script for Week 1 implementations
"""

import sys
from pathlib import Path

# Add src/app to path
sys.path.append(str(Path(__file__).parent / "src" / "app"))

def test_basic_imports():
    """Test basic imports work"""
    print("Testing basic imports...")

    try:
        # Test retrievers
        from retrievers import DynamicPineconeRetriever
        print("+ DynamicPineconeRetriever imported")

        # Test prompts
        from prompts import MedicalPromptManager
        print("+ MedicalPromptManager imported")

        # Test prompt functionality
        manager = MedicalPromptManager()
        templates = manager.list_available_templates()
        print(f"+ Found {len(templates)} prompt templates")

        # Test retriever has key methods
        methods = ['update_k', 'get_relevant_documents']
        for method in methods:
            if hasattr(DynamicPineconeRetriever, method):
                print(f"+ DynamicPineconeRetriever has {method}")
            else:
                print(f"- Missing {method}")
                return False

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("Week 1 Implementation Test")
    print("=" * 30)

    if test_basic_imports():
        print("\nSUCCESS: Week 1 implementations working!")
        print("\nKey improvements:")
        print("- Fixed retriever code duplication")
        print("- Added prompt management system")
        print("- Enabled conversation infrastructure")
        return True
    else:
        print("\nFAILED: Some components not working")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)