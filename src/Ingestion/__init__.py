"""
Clean Medical Document Ingestion with LLM Citation Extraction

The centerpiece system for processing medical PDFs with LLM-powered citation extraction.

Core Components:
- CitationExtractor: LLM + regex hybrid for parsing citations (THE KEY INNOVATION)
- SimplePipeline: Clean ingestion pipeline focused on citation quality
- DocumentProcessor: PDF text extraction with Docling
- ChunkProcessor: Smart section-aware chunking
- MetadataEnhancer: Rich metadata without character limits
- VectorUploader: Direct Pinecone integration

The citation extraction system solves the critical source attribution problem
in medical document RAG systems.
"""

from .citation_extractor import CitationExtractor, Citation, CitationMatch
from .pipeline import SimplePipeline, ingest_documents
from .document_processor import DoclingBookLoader as DocumentProcessor
from .chunk_processor import SmartChunker as ChunkProcessor, ChunkingConfig as ChunkConfig
from .metadata_enhancer import ChunkMetadataEnhancer as MetadataEnhancer
from .vector_uploader import EnhancedVectorUploader as VectorUploader

__version__ = "3.0.0"
__all__ = [
    # The star of the show
    "CitationExtractor",
    "Citation",
    "CitationMatch",

    # Clean pipeline
    "SimplePipeline",
    "ingest_documents",

    # Supporting components
    "DocumentProcessor",
    "ChunkProcessor",
    "ChunkConfig",
    "MetadataEnhancer",
    "VectorUploader"
]