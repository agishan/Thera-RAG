#!/usr/bin/env python3
"""
Practical examples of using Docling chunkers with medical documents.
"""

from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

def example_chunker_configurations():
    """Show different chunker configurations for medical documents."""

    print("DOCLING CHUNKER CONFIGURATION EXAMPLES")
    print("=" * 45)
    print()

    print("1. BASIC HIERARCHICAL CHUNKER (Structure-focused)")
    print("-" * 50)
    basic_hierarchical = HierarchicalChunker()
    print(f"Default delimiter: {repr(basic_hierarchical.delim)}")

    # Custom configuration for medical documents
    medical_hierarchical = HierarchicalChunker(
        delim="\n"  # Preserve line structure important for medical formatting
    )
    print("Medical-optimized configuration:")
    print(f"  - Delimiter: {repr(medical_hierarchical.delim)}")
    print("  - Preserves: Section structure, heading hierarchy, list formatting")
    print()

    print("2. BASIC HYBRID CHUNKER (Structure + Token limits)")
    print("-" * 50)

    # Default configuration uses sentence-transformers/all-MiniLM-L6-v2
    default_hybrid = HybridChunker()
    print(f"Default max tokens: {default_hybrid.max_tokens}")
    print(f"Default merge peers: {default_hybrid.merge_peers}")

    # Medical-optimized configuration
    medical_tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Good for medical content
        max_tokens=384  # Optimal for medical complexity while fitting most embedding models
    )

    medical_hybrid = HybridChunker(
        tokenizer=medical_tokenizer,
        merge_peers=True,  # Combine small related chunks (e.g., short diagnostic criteria)
        delim="\n"  # Preserve medical document formatting
    )

    print("Medical-optimized configuration:")
    print(f"  - Max tokens: {medical_hybrid.max_tokens}")
    print(f"  - Merge peers: {medical_hybrid.merge_peers}")
    print(f"  - Delimiter: {repr(medical_hybrid.delim)}")
    print("  - Tokenizer: all-MiniLM-L6-v2 (good for medical content)")
    print()

    print("3. SPECIALIZED CONFIGURATIONS")
    print("-" * 30)

    # For research papers with many citations
    research_tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=512  # Larger chunks for dense citation contexts
    )

    research_hybrid = HybridChunker(
        tokenizer=research_tokenizer,
        merge_peers=True,  # Important for consolidating related findings
        delim="\n"
    )

    print("Research paper configuration (HybridChunker):")
    print(f"  - Max tokens: {research_hybrid.max_tokens} (larger for dense citations)")
    print(f"  - Merge peers: {research_hybrid.merge_peers}")
    print("  - Best for: Medical research papers, systematic reviews")
    print()

    # For clinical guidelines (very structured)
    guidelines_hierarchical = HierarchicalChunker(
        delim="\n"  # Preserve structured formatting
    )

    print("Clinical guidelines configuration (HierarchicalChunker):")
    print(f"  - Delimiter: {repr(guidelines_hierarchical.delim)}")
    print("  - No token limits: Preserves complete protocols")
    print("  - Best for: Treatment guidelines, diagnostic protocols")
    print()

    print("4. PARAMETER EXPLANATIONS")
    print("-" * 25)
    print("""
HierarchicalChunker Parameters:
  delim: Text delimiter for merging content
    - Default: '\\n' (newline)
    - Recommended for medical: '\\n' (preserves formatting)

HybridChunker Parameters:
  tokenizer: Token counting mechanism
    - Can be: tokenizer object, model name string, or HuggingFace tokenizer
    - Recommended: HuggingFaceTokenizer with medical-friendly model

  max_tokens: Maximum tokens per chunk
    - Auto-detected from tokenizer if not specified
    - Recommended for medical: 384-512 tokens

  merge_peers: Whether to merge small chunks with same metadata
    - Default: True
    - Recommended: True (consolidates related medical content)

  delim: Text delimiter for merging
    - Default: '\\n'
    - Recommended for medical: '\\n' (preserves structure)

CITATION PRESERVATION TIPS:
- Both chunkers keep inline citations with their context
- Reference sections become separate chunks
- Citation formatting is preserved (Smith et al., 2023) or [1-3]
- Context-citation relationships maintained for medical accuracy
""")

def example_chunker_workflow():
    """Show a complete workflow for chunking medical documents."""

    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW: MEDICAL DOCUMENT CHUNKING")
    print("=" * 60)

    print("""
STEP 1: Choose appropriate chunker based on document type

For Structured Medical Guidelines:
  - Use: HierarchicalChunker
  - Reason: Preserves complete treatment protocols and diagnostic criteria
  - Example: Clinical practice guidelines, treatment algorithms

For Research Papers/Reviews:
  - Use: HybridChunker with 512 tokens
  - Reason: Handles dense citations while maintaining readability
  - Example: Systematic reviews, meta-analyses, case studies

For Mixed Medical Content:
  - Use: HybridChunker with 384 tokens
  - Reason: Balances structure with consistent sizing
  - Example: Medical textbooks, clinical summaries

STEP 2: Configure chunker for medical optimization

HierarchicalChunker setup:
  chunker = HierarchicalChunker(delim="\\n")

HybridChunker setup:
  tokenizer = HuggingFaceTokenizer.from_pretrained(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      max_tokens=384  # or 512 for research papers
  )
  chunker = HybridChunker(
      tokenizer=tokenizer,
      merge_peers=True,
      delim="\\n"
  )

STEP 3: Process document with Docling

from docling.document_converter import DocumentConverter

converter = DocumentConverter()
doc_result = converter.convert("medical_paper.pdf")
docling_doc = doc_result.document

STEP 4: Apply chunking

chunks = list(chunker.chunk(docling_doc))

STEP 5: Extract chunk information

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Text preview: {chunk.text[:100]}...")
    print(f"  Metadata: {chunk.meta}")
    print(f"  Headings: {chunk.meta.headings}")

STEP 6: Quality validation

- Check citation preservation: Inline citations stay with context
- Verify structure: Headings and sections properly maintained
- Confirm sizing: Chunks fit embedding model constraints (if using HybridChunker)
- Validate references: Reference sections captured as separate chunks

This workflow ensures optimal chunking for medical RAG systems while
preserving the critical semantic relationships needed for accurate
medical information retrieval.
""")

if __name__ == "__main__":
    example_chunker_configurations()
    example_chunker_workflow()