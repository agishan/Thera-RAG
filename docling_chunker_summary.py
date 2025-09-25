#!/usr/bin/env python3
"""
Simple Docling chunker comparison for medical documents.
"""

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

def main():
    print("DOCLING CHUNKING ALGORITHMS FOR MEDICAL DOCUMENTS")
    print("=" * 55)
    print()

    # Test basic initialization
    print("1. CHUNKER INITIALIZATION:")
    print("-" * 30)

    # HierarchicalChunker
    try:
        hierarchical = HierarchicalChunker(delim="\n")
        print("[SUCCESS] HierarchicalChunker initialized")
        print(f"   - Delimiter: {repr(hierarchical.delim)}")
    except Exception as e:
        print(f"[ERROR] HierarchicalChunker failed: {e}")

    # HybridChunker
    try:
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=512
        )
        hybrid = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            delim="\n"
        )
        print("[SUCCESS] HybridChunker initialized")
        print(f"   - Max tokens: {hybrid.max_tokens}")
        print(f"   - Merge peers: {hybrid.merge_peers}")
        print(f"   - Delimiter: {repr(hybrid.delim)}")
    except Exception as e:
        print(f"[ERROR] HybridChunker failed: {e}")

    print()
    print("2. KEY DIFFERENCES:")
    print("-" * 20)

    print("""
HIERARCHICAL CHUNKER:
- Structure-focused: Uses document layout (headers, sections, lists)
- No token limits: Chunks can vary significantly in size
- Perfect for preserving semantic sections
- Best for: Clinical guidelines, structured medical documents
- Citation handling: Keeps citations with their context
- Parameters: delim (delimiter for text merging)

HYBRID CHUNKER:
- Structure + Token awareness: Combines layout with size constraints
- Token-limited: Ensures chunks fit embedding model limits (default 512 tokens)
- Smart splitting: Uses semchunk for oversized content
- Peer merging: Combines small chunks with similar metadata
- Best for: Varied medical content, production RAG systems
- Citation handling: Prevents citation-context separation
- Parameters: tokenizer, max_tokens, merge_peers, delim

FOR MEDICAL DOCUMENTS WITH CITATIONS:

Choose HierarchicalChunker when:
- Document has clear, well-defined sections
- Section sizes are appropriate for your embedding model
- Structure preservation is paramount
- Processing clinical guidelines or protocols

Choose HybridChunker when:
- Need consistent chunk sizes for embedding models
- Working with varied medical content types
- Building production systems with token constraints
- Processing research papers with dense citations
- Want optimal balance of structure and size control

CITATION PRESERVATION (Both chunkers):
- Inline citations stay with supporting text
- Reference sections preserved as distinct chunks
- Citation formatting maintained (parenthetical, bracketed, author-year)
- Contextual relationships preserved

STRUCTURE HANDLING:
- Both preserve document hierarchy
- Both maintain heading relationships
- Both keep lists and structured content intact
- HybridChunker additionally enforces size limits

RECOMMENDATION FOR MEDICAL RAG:
HybridChunker is generally preferred for production medical RAG systems
due to its balance of structure preservation and practical constraints.
It ensures compatibility with embedding models while maintaining the
semantic integrity crucial for medical accuracy.
""")

if __name__ == "__main__":
    main()