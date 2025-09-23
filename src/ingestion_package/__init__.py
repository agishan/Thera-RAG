"""
Medical Document Ingestion Package

A comprehensive system for processing medical documents, extracting metadata,
and preparing them for RAG systems with enhanced citation tracking.

This package provides:
- Advanced PDF processing with Docling
- Intelligent chunking that preserves document structure
- Comprehensive metadata extraction (bibliographic, contextual, citation)
- LLM + regex hybrid citation extraction and linking
- Full-content vector database upload with rich metadata

Key Components:
- DoclingBookLoader: PDF text extraction
- SmartChunker: Intelligent document chunking
- DocumentMetadataExtractor: Document-level metadata extraction
- ChunkMetadataEnhancer: Chunk-level metadata enhancement
- CitationExtractor: Citation detection and linking
- EnhancedVectorUploader: Vector database upload
- IngestionPipeline: Complete processing orchestration

Quick Start:
    from ingestion_package import process_documents

    results = process_documents(
        input_path="medical_pdfs/",
        output_dir="processed/",
        pinecone_api_key="your_key"
    )
"""

from .document_loader import DoclingBookLoader, LoaderFactory
from .chunking_strategy import SmartChunker, ChunkingConfig
from .metadata_extractor import (
    DocumentMetadataExtractor,
    ChunkMetadataEnhancer,
    DocumentReference
)
from .citation_extractor import CitationExtractor, Citation, CitationMatch
from .vector_uploader import EnhancedVectorUploader
from .ingestion_pipeline import IngestionPipeline, process_documents

__version__ = "1.0.0"
__author__ = "Thera-RAG Team"
__email__ = "support@thera-rag.com"

__all__ = [
    # Main classes
    "IngestionPipeline",
    "process_documents",

    # Core components
    "DoclingBookLoader",
    "LoaderFactory",
    "SmartChunker",
    "ChunkingConfig",
    "DocumentMetadataExtractor",
    "ChunkMetadataEnhancer",
    "DocumentReference",
    "CitationExtractor",
    "Citation",
    "CitationMatch",
    "EnhancedVectorUploader"
]

# Package metadata
__package_info__ = {
    "name": "ingestion_package",
    "version": __version__,
    "description": "Medical document ingestion with enhanced citation tracking",
    "features": [
        "Advanced PDF processing",
        "Smart document chunking",
        "Comprehensive metadata extraction",
        "Citation detection and linking",
        "Vector database integration"
    ],
    "supported_formats": [".pdf"],
    "dependencies": [
        "docling",
        "langchain",
        "langchain-google-genai",
        "sentence-transformers",
        "pinecone-client"
    ]
}