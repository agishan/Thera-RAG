"""
Simplified Medical Document Ingestion Core

A clean, focused system for processing medical PDFs with LLM-powered citation extraction.

Core workflow:
1. PDF Processing - Extract text with Docling
2. Smart Chunking - Section-aware chunking with overlap
3. Metadata Enhancement - Rich metadata for references (NO 500 char limit)
4. LLM Citation Extraction - The centerpiece feature using Google Gemini
5. Vector Upload - Direct Pinecone integration

KEY FEATURE: LLM + regex hybrid citation extraction that links inline citations
to full references, addressing the core problem of reference management in RAG.
"""

from .document_processor import DocumentProcessor
from .chunk_processor import ChunkProcessor, ChunkConfig
from .metadata_enhancer import MetadataEnhancer
from .vector_uploader import VectorUploader
from .simple_pipeline import SimplePipeline, ingest_documents

# Import the star of the show
from ..ingestion_package.citation_extractor import CitationExtractor, Citation, CitationMatch

__version__ = "2.1.0"
__all__ = [
    "DocumentProcessor",
    "ChunkProcessor",
    "ChunkConfig",
    "MetadataEnhancer",
    "VectorUploader",
    "SimplePipeline",
    "ingest_documents",
    "CitationExtractor",
    "Citation",
    "CitationMatch"
]