# API Reference

Complete API documentation for the Medical Document Ingestion Package.

## Table of Contents
- [Core Classes](#core-classes)
- [Configuration Classes](#configuration-classes)
- [Data Classes](#data-classes)
- [Utility Functions](#utility-functions)
- [Error Handling](#error-handling)
- [Type Definitions](#type-definitions)

---

## Core Classes

### IngestionPipeline

Main orchestration class for the complete document processing workflow.

```python
class IngestionPipeline:
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        pinecone_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        index_name: str = "medical-rag-index",
        namespace: str = "thera-rag-enhanced",
        enable_citations: bool = True,
        enable_upload: bool = True
    )
```

**Parameters**:
- `chunking_config`: Configuration for chunking strategy
- `pinecone_api_key`: Pinecone API key for vector database upload
- `google_api_key`: Google API key for LLM-based citation extraction
- `index_name`: Name of the Pinecone index
- `namespace`: Namespace within the Pinecone index
- `enable_citations`: Whether to extract and link citations
- `enable_upload`: Whether to upload processed chunks to vector database

#### Methods

##### `process_single_file(file_path, output_dir=None, save_intermediate=True)`

Process a single document file through the complete pipeline.

**Parameters**:
- `file_path` (str): Absolute path to the document file
- `output_dir` (str, optional): Directory to save intermediate outputs
- `save_intermediate` (bool): Whether to save intermediate processing files

**Returns**:
```python
{
    "success": bool,
    "file_path": str,
    "document_metadata": dict,
    "chunks_created": int,
    "citations_found": int,
    "citation_matches": int,
    "upload_stats": dict,
    "chunk_analysis": dict,
    "processing_time": str
}
```

**Example**:
```python
pipeline = IngestionPipeline()
result = pipeline.process_single_file(
    "medical_paper.pdf",
    output_dir="processed/",
    save_intermediate=True
)
print(f"Created {result['chunks_created']} chunks")
```

##### `process_directory(input_dir, output_dir=None, file_pattern="*.pdf", save_intermediate=True)`

Process all matching files in a directory.

**Parameters**:
- `input_dir` (str): Directory containing documents to process
- `output_dir` (str, optional): Directory to save outputs
- `file_pattern` (str): Glob pattern for file selection
- `save_intermediate` (bool): Whether to save intermediate files

**Returns**: `List[dict]` - List of processing results for each file

**Example**:
```python
results = pipeline.process_directory(
    "medical_docs/",
    "processed/",
    "*.pdf"
)
print(f"Processed {len(results)} files")
```

##### `get_processing_stats()`

Get detailed processing statistics.

**Returns**:
```python
{
    "files_processed": int,
    "total_chunks": int,
    "citations_extracted": int,
    "vectors_uploaded": int,
    "errors": List[str]
}
```

##### `verify_upload(sample_query="medical treatment")`

Verify vector database upload success.

**Parameters**:
- `sample_query` (str): Query to test retrieval

**Returns**: Verification results dictionary

---

### DoclingBookLoader

Advanced PDF text extraction using Docling.

```python
class DoclingBookLoader:
    def __init__(
        self,
        file_path: str,
        num_threads: int = 8,
        do_ocr: bool = True,
        do_table_structure: bool = True,
        accelerator_device: str = "auto"
    )
```

**Parameters**:
- `file_path`: Path to PDF file
- `num_threads`: Number of processing threads
- `do_ocr`: Enable OCR for scanned documents
- `do_table_structure`: Extract table structures
- `accelerator_device`: Processing device ("auto", "cpu", "gpu")

#### Methods

##### `extract_text()`

Extract text content as markdown.

**Returns**: `str` - Markdown-formatted text

##### `extract_structured_content()`

Extract comprehensive structured content.

**Returns**:
```python
{
    "text": str,                    # Markdown text
    "raw_document": object,       # Raw Docling document
    "metadata": dict,             # File metadata
    "tables": List[dict],         # Extracted tables
    "images": List[dict]          # Image information
}
```

---

### SmartChunker

Intelligent document chunking with structure preservation.

```python
class SmartChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None)
```

#### Methods

##### `create_chunks(text, base_metadata)`

Create intelligent chunks from document text.

**Parameters**:
- `text` (str): Full document text
- `base_metadata` (dict): Base metadata for all chunks

**Returns**: `List[Document]` - List of LangChain Document objects with enhanced metadata

**Example**:
```python
chunker = SmartChunker()
chunks = chunker.create_chunks(
    text="Document content...",
    base_metadata={"source": "paper.pdf"}
)
```

##### `analyze_chunks(chunks)`

Analyze chunk statistics for optimization.

**Parameters**:
- `chunks` (List[Document]): List of document chunks

**Returns**: Dictionary with chunk analysis statistics

---

### DocumentMetadataExtractor

Extract document-level metadata for proper referencing.

```python
class DocumentMetadataExtractor:
    def __init__(self)
```

#### Methods

##### `extract_document_metadata(text, filename, structured_content=None)`

Extract comprehensive document metadata.

**Parameters**:
- `text` (str): Full document text
- `filename` (str): Original filename
- `structured_content` (dict, optional): Additional structured content

**Returns**: `DocumentReference` object with extracted metadata

---

### ChunkMetadataEnhancer

Enhance chunk-level metadata with contextual information.

```python
class ChunkMetadataEnhancer:
    def __init__(self, doc_ref: DocumentReference)
```

#### Methods

##### `enhance_chunk_metadata(chunk, full_text, chunk_index)`

Enhance a chunk with comprehensive metadata.

**Parameters**:
- `chunk` (Document): Document chunk to enhance
- `full_text` (str): Full document text for context
- `chunk_index` (int): Index of chunk in document

**Returns**: Enhanced `Document` with rich metadata

---

### CitationExtractor

Advanced citation detection and linking using LLM + regex hybrid approach.

```python
class CitationExtractor:
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        use_llm: bool = True
    )
```

#### Methods

##### `extract_citations_from_document(full_text, chunks)`

Extract and link citations from document.

**Parameters**:
- `full_text` (str): Complete document text
- `chunks` (List[Document]): Document chunks

**Returns**: `Tuple[List[Citation], List[CitationMatch]]`

##### `enhance_chunks_with_citations(chunks, citation_matches)`

Add citation information to chunk metadata.

**Parameters**:
- `chunks` (List[Document]): Document chunks
- `citation_matches` (List[CitationMatch]): Citation matches

**Returns**: Enhanced chunks with citation metadata

---

### EnhancedVectorUploader

Upload processed chunks to Pinecone with full metadata preservation.

```python
class EnhancedVectorUploader:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        embedding_model: str = "intfloat/e5-base"
    )
```

#### Methods

##### `upload_chunks(chunks, batch_size=50, show_progress=True)`

Upload chunks to vector database.

**Parameters**:
- `chunks` (List[Document]): Chunks to upload
- `batch_size` (int): Vectors per batch
- `show_progress` (bool): Show progress bar

**Returns**: Upload statistics dictionary

##### `verify_upload(sample_query, expected_chunks=0)`

Verify upload success and metadata quality.

**Parameters**:
- `sample_query` (str): Test query
- `expected_chunks` (int): Expected number of uploaded chunks

**Returns**: Verification results

---

## Configuration Classes

### ChunkingConfig

Configuration for document chunking strategy.

```python
@dataclass
class ChunkingConfig:
    target_chunk_size: int = 3000      # Target size in characters
    overlap_size: int = 300            # Overlap between chunks
    max_chunk_size: int = 5000         # Maximum allowed size
    min_chunk_size: int = 500          # Minimum chunk size
    preserve_sentences: bool = True     # Don't break sentences
    preserve_paragraphs: bool = True    # Preserve paragraph boundaries
    section_aware: bool = True          # Use section-based chunking
```

**Example**:
```python
config = ChunkingConfig(
    target_chunk_size=4000,
    overlap_size=400,
    section_aware=True
)
```

---

## Data Classes

### DocumentReference

Structured reference information for documents.

```python
@dataclass
class DocumentReference:
    source_title: str
    authors: Optional[str] = None
    publication_year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_hierarchy: Optional[List[str]] = None
    citation_format: Optional[str] = None
    document_type: str = "document"
```

### Citation

Represents an extracted citation.

```python
@dataclass
class Citation:
    inline_text: str              # Inline citation text
    full_reference: str           # Full bibliography reference
    authors: List[str]            # Author names
    year: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pages: Optional[str] = None
    citation_type: str = "unknown"
    confidence: float = 0.0
```

### CitationMatch

Links inline citations to full references.

```python
@dataclass
class CitationMatch:
    inline_citation: str          # Inline citation text
    chunk_location: str           # Where citation appears
    full_reference: Citation      # Full citation object
    context_before: str           # Text before citation
    context_after: str            # Text after citation
```

---

## Utility Functions

### `process_documents(input_path, output_dir, **kwargs)`

Convenience function for quick document processing.

**Parameters**:
- `input_path` (str): Path to file or directory
- `output_dir` (str): Output directory
- `pinecone_api_key` (str, optional): Pinecone API key
- `google_api_key` (str, optional): Google API key
- `**kwargs`: Additional pipeline configuration

**Returns**: List of processing results

**Example**:
```python
from ingestion_package import process_documents

results = process_documents(
    "medical_papers/",
    "processed/",
    pinecone_api_key="your_key",
    enable_citations=True
)
```

### `LoaderFactory.create_loader(file_path, **kwargs)`

Factory method for creating document loaders.

**Parameters**:
- `file_path` (str): Path to document
- `**kwargs`: Additional loader configuration

**Returns**: Appropriate loader instance

---

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when input files or directories don't exist.

#### `ValueError`
Raised for invalid configuration or parameters.

#### `RuntimeError`
Raised for processing failures (PDF extraction, API errors, etc.).

### Error Recovery

The pipeline includes robust error handling:

```python
try:
    result = pipeline.process_single_file("document.pdf")
except FileNotFoundError:
    print("File not found")
except RuntimeError as e:
    print(f"Processing failed: {e}")
```

Error information is also captured in processing results:

```python
result = pipeline.process_single_file("document.pdf")
if not result["success"]:
    print(f"Error: {result['error']}")
```

---

## Type Definitions

### Common Types

```python
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document

# Processing result type
ProcessingResult = Dict[str, Any]

# Chunk list type
ChunkList = List[Document]

# Metadata dictionary type
MetadataDict = Dict[str, Any]

# Citation types
CitationList = List[Citation]
CitationMatchList = List[CitationMatch]
```

### Return Type Examples

```python
# Single file processing
result: ProcessingResult = pipeline.process_single_file("file.pdf")

# Directory processing
results: List[ProcessingResult] = pipeline.process_directory("docs/")

# Citation extraction
citations: CitationList
matches: CitationMatchList
citations, matches = extractor.extract_citations_from_document(text, chunks)

# Upload statistics
stats: Dict[str, int] = uploader.upload_chunks(chunks)
```

---

## Configuration Examples

### Minimal Configuration
```python
from ingestion_package import IngestionPipeline

pipeline = IngestionPipeline(
    enable_upload=False,
    enable_citations=False
)
```

### Production Configuration
```python
from ingestion_package import IngestionPipeline, ChunkingConfig

config = ChunkingConfig(
    target_chunk_size=4000,
    overlap_size=400,
    section_aware=True
)

pipeline = IngestionPipeline(
    chunking_config=config,
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    index_name="production-index",
    namespace="medical-docs-v2",
    enable_citations=True,
    enable_upload=True
)
```

### Research-Focused Configuration
```python
# Focus on citation extraction and analysis
pipeline = IngestionPipeline(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    enable_citations=True,
    enable_upload=False  # Local processing only
)
```

This API reference provides complete documentation for all classes, methods, and configuration options in the Medical Document Ingestion Package.