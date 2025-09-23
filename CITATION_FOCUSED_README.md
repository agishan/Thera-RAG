# Medical RAG with LLM Citation Extraction

A focused system for processing medical documents with **LLM-powered citation extraction** as the centerpiece feature.

## The Core Problem We Solve

Medical and academic documents contain complex citation patterns that standard RAG systems fail to capture properly. Inline citations like `(Smith et al., 2023)` get lost, and full references in bibliographies aren't linked to the content chunks that reference them.

## Our Solution: LLM + Regex Hybrid Citation Extraction

**This is what makes our system unique:** We use Google Gemini LLM combined with regex patterns to:

1. **Extract inline citations** from document chunks with context
2. **Parse bibliography sections** into structured citation data
3. **Link inline citations to full references** with high accuracy
4. **Preserve full content** in Pinecone (no 500-character limits)
5. **Enable citation-aware retrieval** for better source attribution

## Quick Start

### 1. Setup Environment

```bash
# Required for LLM citation extraction (the key feature)
export GOOGLE_API_KEY="your-google-api-key"

# Optional for vector storage
export PINECONE_API_KEY="your-pinecone-key"
```

### 2. Add Documents

```bash
mkdir data
# Add your PDF files to data/
```

### 3. Run Citation-Focused Ingestion

```bash
python run_citation_focused_ingestion.py
```

### 4. Verify Citation Quality

```bash
python verify_simple_quality.py
```

## What You Get

### Enhanced Chunks with Citation Metadata

Each chunk includes:

```json
{
  "content": "Medical treatment approaches have evolved (Smith et al., 2023)...",
  "metadata": {
    "citations": [
      {
        "inline_text": "(Smith et al., 2023)",
        "full_reference": "Smith, J., Jones, M. (2023). Advanced Medical Treatments. Nature Medicine, 45(2), 123-145.",
        "authors": ["Smith, J.", "Jones, M."],
        "year": "2023",
        "title": "Advanced Medical Treatments",
        "journal": "Nature Medicine",
        "doi": "10.1038/example"
      }
    ],
    "text": "FULL CONTENT PRESERVED - NO 500 CHAR LIMIT",
    "chunk_size": 2847,
    "word_count": 421
  }
}
```

### Citation-Aware RAG Retrieval

The enhanced retriever now includes full citation context:

```python
from src.app.rag_service import RAGService

# Retrieval now includes rich citation metadata
response = rag_service.get_response(
    question="What are the latest treatment approaches?",
    chat_history=[],
    retrieval_k=15  # Simplified: no complex re-ranking
)

# Source documents include full citations
for doc in response['source_documents']:
    citations = doc.metadata.get('citations', [])
    print(f"References: {len(citations)} citations found")
```

## System Architecture (Simplified)

### Core Components

1. **DocumentProcessor** - Docling-based PDF text extraction
2. **ChunkProcessor** - Smart section-aware chunking
3. **CitationExtractor** - LLM + regex hybrid (THE KEY COMPONENT)
4. **MetadataEnhancer** - Rich metadata without char limits
5. **VectorUploader** - Direct Pinecone integration

### What We Removed (Simplified)

- ❌ Cross-encoder re-ranking (disabled by default, available as option)
- ❌ Complex abstraction layers
- ❌ Over-engineered factory patterns
- ❌ Unnecessary intermediate classes
- ❌ 500-character content limits

### What We Kept (Essential)

- ✅ **LLM Citation Extraction** (the whole point!)
- ✅ Full content preservation in vectors
- ✅ Smart chunking with overlap
- ✅ Rich metadata for references
- ✅ Simple, clean interfaces

## Citation Extraction Details

### Supported Citation Formats

**Inline Citations:**
- Author-year: `(Smith et al., 2023)`
- Multiple authors: `(Smith & Jones, 2023)`
- Numbered: `[1,2,3]`
- Superscript style: `¹²³`

**Bibliography Formats:**
- Journal articles with DOI
- Book chapters and references
- Conference proceedings
- Electronic sources

### LLM Enhancement

When `GOOGLE_API_KEY` is provided, the system:

1. Uses **Google Gemini** to parse complex bibliography sections
2. Extracts structured metadata (authors, year, title, journal, DOI)
3. Matches inline citations to full references with confidence scoring
4. Falls back to regex patterns for robustness

### Quality Metrics

The system tracks:
- Citation coverage percentage
- Average citations per chunk
- Match confidence scores
- Processing success rates

## Configuration

### Simple Pipeline Usage

```python
from src.ingestion_core import SimplePipeline

pipeline = SimplePipeline(
    pinecone_api_key="your-key",     # Optional
    google_api_key="your-key",       # Required for LLM citations
    index_name="medical-rag-index",
    namespace="thera-rag"
)

# Process single document
result = pipeline.process_document("paper.pdf", "output/")

# Focus on citation stats
print(f"Citations extracted: {result['citations_extracted']}")
print(f"Coverage: {result['citation_coverage']['citation_coverage_percent']:.1f}%")
```

### RAG Service (Re-ranking Disabled)

```python
from src.app.rag_service import RAGService

# Enhanced retriever with citations, no complex re-ranking
rag_service = RAGService(config)

# Simple, fast retrieval with rich citation metadata
response = rag_service.get_response(
    question="Your question",
    chat_history=[],
    retrieval_k=15  # Clean, simple retrieval
)
```

## Why This Approach Works

### Before: Standard RAG Issues
- Lost citation context
- Truncated content (500 char limits)
- Poor source attribution
- Complex, slow re-ranking

### After: Citation-Focused RAG
- Full citation preservation with LLM accuracy
- Complete content in vectors
- Rich source metadata
- Fast, clean retrieval
- Focused on the core value: **linking references properly**

## File Structure

```
src/
├── ingestion_core/           # Simplified ingestion
│   ├── simple_pipeline.py    # Clean pipeline focusing on citations
│   ├── citation_extractor.py # LLM + regex hybrid (CORE)
│   └── ...                   # Support components
├── ingestion_package/        # Legacy (for reference)
├── app/
│   ├── rag_service.py        # Re-ranking disabled
│   └── enhanced_retriever.py # Citation-aware retrieval
└── ...

enhanced_output/              # Rich citation metadata chunks
data/                        # Input PDFs
```

## Next Steps

1. **Add your medical PDFs** to `data/` directory
2. **Set GOOGLE_API_KEY** for full LLM citation extraction
3. **Run the citation-focused ingestion**
4. **Verify citation quality** with the included tools
5. **Use enhanced RAG** with proper source attribution

The system is now focused on what matters: **extracting and linking citations properly** while keeping everything else simple and fast.