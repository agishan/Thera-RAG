"""
Test script to verify source documents are being returned
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'app'))

from config import get_config
from rag_service import RAGService

def test_source_documents():
    """Test if source documents are being returned"""
    
    print("🧪 Testing Source Document Retrieval")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_config()
        print("✅ Configuration loaded")
        
        # Initialize RAG service
        rag_service = RAGService(config)
        print("✅ RAG service initialized")
        
        # Test query
        test_question = "Tell me about viscoelastic testing in liver disease?"
        chat_history = []
        
        print(f"🔍 Testing query: {test_question}")
        
        # Get response
        result = rag_service.get_response(test_question, chat_history, retrieval_k=5)
        
        print(f"📊 Response keys: {list(result.keys())}")
        
        # Check for source documents
        source_docs = result.get('source_documents', [])
        print(f"📚 Source documents found: {len(source_docs)}")
        
        if source_docs:
            print("✅ Source documents are being returned!")
            for i, doc in enumerate(source_docs, 1):
                print(f"\n📄 Document {i}:")
                print(f"   Content preview: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"   Metadata: {doc.metadata}")
                if hasattr(doc, 'score'):
                    print(f"   Score: {doc.score}")
        else:
            print("❌ No source documents returned")
            
        # Check answer
        answer = result.get('answer', 'No answer found')
        print(f"\n🤖 Answer: {answer[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_source_documents()
