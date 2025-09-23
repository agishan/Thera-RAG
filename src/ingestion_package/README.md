# Medical Document Ingestion Package

A comprehensive Python package for processing medical documents, extracting enhanced metadata, and preparing them for RAG (Retrieval-Augmented Generation) systems with advanced citation tracking.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Components](#components)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This package transforms medical PDFs into semantically rich, searchable chunks with comprehensive metadata and citation tracking. It's designed specifically for medical/academic documents that contain complex citations, hierarchical structures, and require precise referencing.

### Key Problems Solved

1. **Poor Reference Management**: Links inline citations to full bibliographic references
2. **Limited Metadata**: Extracts comprehensive document and chunk-level metadata
3. **Inefficient Chunking**: Creates semantically aware chunks that preserve context
4. **Citation Tracking**: Maintains traceable citations for academic integrity
5. **Vector Database Integration**: Seamless upload to Pinecone with full metadata preservation

## ✨ Features

### Document Processing
- **Advanced PDF Processing**: Uses Docling for high-quality text extraction
- **Smart Chunking**: Section-aware chunking that preserves document structure
- **Full Metadata Extraction**: Comprehensive bibliographic and contextual metadata
- **Citation Extraction**: LLM + regex hybrid system for citation detection and linking

### Enhanced Metadata
- **Document-level**: Title, authors, year, journal, DOI, document type
- **Chunk-level**: Page numbers, section hierarchy, content statistics
- **Citation-level**: Inline citations linked to full references
- **Location-level**: Page ranges, section breadcrumbs, hierarchical context

### Vector Database Integration
- **Full Content Preservation**: No character limits on stored text
- **Rich Metadata**: All extracted metadata stored in vector database
- **Search Enhancement**: Multiple search fields and tags for improved retrieval
- **Quality Verification**: Automated upload verification and quality assessment

## 🚀 Installation

### Prerequisites
```bash
# Required dependencies
pip install docling
pip install langchain
pip install langchain-google-genai
pip install sentence-transformers
pip install pinecone-client
pip install python-dotenv
pip install tqdm
```

### Environment Setup
Create a `.env` file with your API keys:
```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Package Installation
```bash
# Add the package to your Python path or install locally
cd src/
pip install -e ingestion_package/
```

## 🚀 Quick Start

### Basic Usage
```python
from ingestion_package import process_documents

# Process all PDFs in a directory
results = process_documents(
    input_path="medical_pdfs/",
    output_dir="processed_output/",
    pinecone_api_key="your_key",
    google_api_key="your_key"
)

print(f"Processed {len(results)} documents")
```

### Advanced Usage
```python
from ingestion_package import IngestionPipeline, ChunkingConfig

# Custom configuration
config = ChunkingConfig(
    target_chunk_size=4000,
    overlap_size=400,
    section_aware=True
)

# Initialize pipeline
pipeline = IngestionPipeline(
    chunking_config=config,
    pinecone_api_key="your_key",
    google_api_key="your_key",
    namespace="medical-docs-v2",
    enable_citations=True
)

# Process documents
results = pipeline.process_directory("pdfs/", "output/")
```

## 🏗️ Architecture

The package follows a modular pipeline architecture:

```
Input PDF → Document Loader → Chunking → Metadata Enhancement → Citation Extraction → Vector Upload
```

### Data Flow
1. **PDF Input**: Raw medical/academic PDF files
2. **Text Extraction**: Docling converts PDF to structured markdown
3. **Document Analysis**: Extract document-level metadata (title, authors, etc.)
4. **Intelligent Chunking**: Create semantically aware chunks
5. **Metadata Enhancement**: Add comprehensive chunk-level metadata
6. **Citation Processing**: Extract and link citations
7. **Vector Storage**: Upload to Pinecone with full metadata

## 📦 Components

### 1. Document Loader (`document_loader.py`)
**Purpose**: High-quality PDF text extraction using Docling

**Key Classes**:
- `DoclingBookLoader`: Main PDF processing class
- `LoaderFactory`: Factory for creating appropriate loaders

**Features**:
- OCR support for scanned documents
- Table structure extraction
- Configurable processing options
- Multi-threaded processing

### 2. Chunking Strategy (`chunking_strategy.py`)
**Purpose**: Intelligent document chunking that preserves structure

**Key Classes**:
- `SmartChunker`: Main chunking engine
- `ChunkingConfig`: Configuration options

**Strategies**:
- **Section-based**: Splits by document headers/sections
- **Recursive**: Falls back to recursive character splitting
- **Hybrid**: Combines both approaches for optimal results

### 3. Metadata Extractor (`metadata_extractor.py`)
**Purpose**: Extract comprehensive metadata for better referencing

**Key Classes**:
- `DocumentMetadataExtractor`: Document-level metadata extraction
- `ChunkMetadataEnhancer`: Chunk-level metadata enhancement
- `DocumentReference`: Structured reference information

**Extracted Data**:
- Bibliographic information (title, authors, year, journal, DOI)
- Document classification (research paper, clinical guide, etc.)
- Location information (pages, sections, hierarchy)
- Content statistics (word count, sentence count, etc.)

### 4. Citation Extractor (`citation_extractor.py`)
**Purpose**: Advanced citation detection and linking

**Key Classes**:
- `CitationExtractor`: Main citation processing engine
- `Citation`: Structured citation data
- `CitationMatch`: Links inline citations to full references

**Capabilities**:
- **LLM-based extraction**: Uses Google's Gemini for high-quality parsing
- **Regex backup**: Fallback patterns for various citation formats
- **Smart matching**: Links inline citations to bibliography entries
- **Context preservation**: Maintains citation context within chunks

### 5. Vector Uploader (`vector_uploader.py`)
**Purpose**: Upload processed chunks to Pinecone with full metadata

**Key Classes**:
- `EnhancedVectorUploader`: Main upload engine

**Features**:
- **Full content preservation**: No character limits
- **Rich metadata**: All extracted metadata included
- **Quality verification**: Automated upload validation
- **Batch processing**: Efficient batch uploads

### 6. Ingestion Pipeline (`ingestion_pipeline.py`)
**Purpose**: Orchestrates the entire processing workflow

**Key Classes**:
- `IngestionPipeline`: Main orchestration class

**Features**:
- **Modular processing**: Enable/disable individual components
- **Error handling**: Robust error management and reporting
- **Progress tracking**: Detailed processing statistics
- **Intermediate saves**: Optional intermediate file outputs

## ⚙️ Configuration

### Chunking Configuration
```python
from ingestion_package import ChunkingConfig

config = ChunkingConfig(
    target_chunk_size=3000,      # Target size in characters
    overlap_size=300,            # Overlap between chunks
    max_chunk_size=5000,         # Maximum allowed chunk size
    min_chunk_size=500,          # Minimum chunk size
    preserve_sentences=True,     # Don't break sentences
    preserve_paragraphs=True,    # Preserve paragraph boundaries
    section_aware=True           # Use section-based chunking
)
```

### Pipeline Configuration
```python
pipeline = IngestionPipeline(
    chunking_config=config,
    pinecone_api_key="your_key",
    google_api_key="your_key",
    index_name="medical-rag-index",
    namespace="enhanced-docs",
    enable_citations=True,       # Extract citations
    enable_upload=True          # Upload to Pinecone
)
```

## 📖 API Reference

### Core Functions

#### `process_documents(input_path, output_dir, **kwargs)`
Convenience function for quick document processing.

**Parameters**:
- `input_path` (str): Path to file or directory
- `output_dir` (str): Output directory for intermediate files
- `pinecone_api_key` (str, optional): Pinecone API key
- `google_api_key` (str, optional): Google API key
- `**kwargs`: Additional pipeline configuration

**Returns**: List of processing results

#### `IngestionPipeline.process_single_file(file_path, output_dir, save_intermediate)`
Process a single document file.

**Parameters**:
- `file_path` (str): Path to document file
- `output_dir` (str, optional): Output directory
- `save_intermediate` (bool): Save intermediate files

**Returns**: Processing result dictionary

#### `IngestionPipeline.process_directory(input_dir, output_dir, file_pattern, save_intermediate)`
Process all files in a directory.

**Parameters**:
- `input_dir` (str): Input directory path
- `output_dir` (str, optional): Output directory
- `file_pattern` (str): File pattern (default: "*.pdf")
- `save_intermediate` (bool): Save intermediate files

**Returns**: List of processing results

### Metadata Structure

#### Document-Level Metadata
```python
{
    "source_title": "Paper Title",
    "source_authors": "Author1, Author2",
    "source_year": "2023",
    "source_journal": "Journal Name",
    "source_doi": "10.1234/example",
    "document_type": "research_paper",
    "citation_format": "Full formatted citation"
}
```

#### Chunk-Level Metadata
```python
{
    "chunk_id": "unique_chunk_identifier",
    "chunk_index": 0,
    "page_start": 15,
    "page_end": 16,
    "section_hierarchy": ["Methods", "Statistical Analysis"],
    "hierarchy_breadcrumb": "Methods > Statistical Analysis",
    "text": "Full chunk content (no limits)",
    "chunk_size": 2847,
    "word_count": 425,
    "citations": [
        {
            "inline_text": "(Smith et al., 2023)",
            "full_reference": "Complete citation",
            "authors": ["Smith, J.", "Jones, A."],
            "year": "2023"
        }
    ]
}
```

## 📋 Examples

### Example 1: Basic Processing
```python
from ingestion_package import process_documents

# Process a single file
results = process_documents(
    input_path="research_paper.pdf",
    output_dir="output/",
    enable_upload=False  # Just process locally
)

print(f"Created {results[0]['chunks_created']} chunks")
```

### Example 2: Citation-Focused Processing
```python
from ingestion_package import IngestionPipeline

pipeline = IngestionPipeline(
    google_api_key="your_key",
    enable_citations=True,
    enable_upload=False
)

result = pipeline.process_single_file(
    "medical_review.pdf",
    "citation_analysis/"
)

print(f"Found {result['citations_found']} citations")
print(f"Linked {result['citation_matches']} inline citations")
```

### Example 3: Custom Chunking
```python
from ingestion_package import IngestionPipeline, ChunkingConfig

# Large chunks for comprehensive context
config = ChunkingConfig(
    target_chunk_size=5000,
    overlap_size=500,
    section_aware=True
)

pipeline = IngestionPipeline(chunking_config=config)
results = pipeline.process_directory("large_documents/", "output/")
```

### Example 4: Production Pipeline
```python
from ingestion_package import IngestionPipeline
import os

# Production configuration
pipeline = IngestionPipeline(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    index_name="production-medical-index",
    namespace="clinical-guidelines-v1",
    enable_citations=True,
    enable_upload=True
)

# Process all clinical guidelines
results = pipeline.process_directory(
    input_dir="clinical_guidelines/",
    output_dir="processed_guidelines/",
    file_pattern="*.pdf"
)

# Verify upload
verification = pipeline.verify_upload("treatment guidelines")
print(f"Upload successful: {verification['query_test']['test_passed']}")
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Docling Installation Issues
```bash
# If Docling fails to install
pip install --upgrade pip
pip install docling --no-cache-dir
```

#### 2. Memory Issues with Large Documents
```python
# Use smaller chunk sizes for large documents
config = ChunkingConfig(target_chunk_size=2000)
```

#### 3. Citation Extraction Failures
```python
# Disable LLM-based extraction if API issues
pipeline = IngestionPipeline(
    enable_citations=True,
    google_api_key=None  # Will use regex-only extraction
)
```

#### 4. Pinecone Upload Errors
```python
# Verify index exists and dimensions match
from pinecone import Pinecone
pc = Pinecone(api_key="your_key")
print(pc.list_indexes())
```

### Error Messages

- **"Missing Pinecone env vars"**: Set `PINECONE_API_KEY` in environment
- **"Failed to extract text"**: Check if PDF is readable/not corrupted
- **"Chunk size exceeds limit"**: Reduce `target_chunk_size` in configuration
- **"Citation extraction failed"**: Check Google API key or disable LLM features

### Performance Optimization

1. **Disable features you don't need**:
   ```python
   pipeline = IngestionPipeline(
       enable_citations=False,  # Skip if not needed
       enable_upload=False      # Process locally only
   )
   ```

2. **Use appropriate chunk sizes**:
   - Small documents: 2000-3000 characters
   - Large documents: 4000-5000 characters
   - Very large documents: Consider splitting first

3. **Batch processing**:
   ```python
   # Process in smaller batches for memory management
   results = pipeline.process_directory(
       "large_corpus/",
       "output/",
       file_pattern="batch_*.pdf"
   )
   ```

## 📝 Output Files

When `save_intermediate=True`, the following files are created:

- `{filename}.md`: Extracted markdown text
- `{filename}_chunks.json`: Processed chunks with metadata
- `{filename}_metadata.json`: Document-level metadata
- `{filename}_citations.json`: Extracted citations and matches
- `{filename}_analysis.json`: Chunk analysis statistics

## 🤝 Contributing

1. Follow the modular architecture
2. Add comprehensive docstrings
3. Include type hints
4. Add error handling
5. Update documentation

## 📄 License

This package is part of the Thera-RAG project for medical document processing.