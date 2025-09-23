#!/usr/bin/env python3
"""
Test script for RAG comparison functionality
"""

import sys
from pathlib import Path

# Add the src/app directory to the path
sys.path.append(str(Path(__file__).parent / "src" / "app"))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from config import get_config, validate_config
        from rag_service import RAGService
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import get_config, validate_config
        config = get_config()
        
        required_keys = ['pinecone_api_key', 'google_api_key', 'llm_model']
        missing = [key for key in required_keys if not config.get(key)]
        
        if missing:
            print(f"Missing config keys: {missing}")
            print("   Make sure to set up your .env file or environment variables")
            return False
        else:
            print("Configuration loaded successfully")
            print(f"   Model: {config['llm_model']}")
            print(f"   Retrieval k: {config['retrieval_k']}")
            return True
            
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def main():
    """Run basic tests"""
    print("Testing RAG Comparison Setup")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check your dependencies.")
        return
    
    # Test config
    if not test_config():
        print("\n❌ Config test failed. Check your environment setup.")
        return
    
    print("\n✅ All tests passed! You can now run:")
    print("   python rag_comparison.py \"Your question here\"")
    
    # Example usage
    print("\n📝 Example questions to try:")
    examples = [
        "Which ROTEM parameter best predicts the need for a platelet transfusion if abnormal?"]
    
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")

if __name__ == "__main__":
    main()
