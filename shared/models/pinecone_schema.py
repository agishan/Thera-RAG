#!/usr/bin/env python3
"""
Simple, single-tier Pinecone metadata schema
You control what gets included by commenting out fields you don't want
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

@dataclass
class Citation:
    """Citation data structure"""
    citation_id: str
    authors: List[str]
    title: str
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    inline_mention: Optional[str] = None  # "(Smith et al., 2023)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Pinecone"""
        return {
            "id": self.citation_id,
            "authors": self.authors,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "mention": self.inline_mention
        }

@dataclass
class ChunkMetadata:
    """Single comprehensive metadata schema - enable/disable fields as needed"""

    # === ESSENTIAL FIELDS (always keep these) ===
    document_id: str
    chunk_id: str
    chunk_index: int
    section: str                        # methods, results, discussion, etc.
    content_hash: str
    processed_date: str

    # === DOCUMENT INFO ===
    document_title: Optional[str] = None
    source_path: Optional[str] = None

    # === CHUNK DETAILS ===
    subsection: Optional[str] = None    # more granular than section
    chunk_length: Optional[int] = None
    overlap_with_next: Optional[int] = None

    # === HITL & QUALITY ===
    quality_score: Optional[float] = None
    human_approved: Optional[bool] = None
    processing_notes: Optional[str] = None
    is_complete_concept: Optional[bool] = None

    # === PROCESSING INFO ===
    processing_version: Optional[str] = None
    confidence_score: Optional[float] = None    # LLM confidence
    contains_references: Optional[bool] = None

    # === MEDICAL METADATA ===
    medical_specialty: Optional[str] = None     # hematology, cardiology, etc.
    evidence_level: Optional[str] = None        # guideline, rct, case_study
    clinical_relevance: Optional[str] = None    # high, medium, low
    content_type: Optional[str] = None          # methodology, findings, recommendation
    patient_population: Optional[str] = None   # major_bleeding, trauma, etc.

    # === CITATIONS ===
    citations: Optional[List[Citation]] = None
    citation_count: Optional[int] = None

    # === RETRIEVAL OPTIMIZATION ===
    medical_terms: Optional[List[str]] = None   # key terms for better retrieval
    retrieval_priority: Optional[str] = None    # high, medium, low

    def to_pinecone_metadata(self) -> Dict[str, Any]:
        """Convert to Pinecone metadata format"""
        metadata = {}

        # Always include essential fields
        metadata["document_id"] = self.document_id
        metadata["chunk_id"] = self.chunk_id
        metadata["chunk_index"] = self.chunk_index
        metadata["section"] = self.section
        metadata["content_hash"] = self.content_hash
        metadata["processed_date"] = self.processed_date

        # Add optional fields only if they have values
        optional_fields = [
            "document_title", "source_path", "subsection", "chunk_length",
            "overlap_with_next", "quality_score", "human_approved",
            "processing_notes", "is_complete_concept", "processing_version",
            "confidence_score", "contains_references", "medical_specialty",
            "evidence_level", "clinical_relevance", "content_type",
            "patient_population", "citation_count", "retrieval_priority"
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                metadata[field] = value

        # Handle citations separately (convert to dict format)
        if self.citations:
            metadata["citations"] = [citation.to_dict() for citation in self.citations]

        # Handle medical terms (limit to reasonable number)
        if self.medical_terms:
            metadata["medical_terms"] = self.medical_terms[:10]  # Limit to 10 terms

        return metadata

    def estimate_size(self) -> int:
        """Estimate metadata size in bytes"""
        metadata = self.to_pinecone_metadata()
        return len(json.dumps(metadata).encode('utf-8'))

# Simple usage - just create the metadata and upload
def create_chunk_metadata(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simple function to create Pinecone metadata from chunk data"""
    metadata = ChunkMetadata(**chunk_data)
    return metadata.to_pinecone_metadata()

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_chunk = {
        "document_id": "vha-guideline-2024",
        "chunk_id": "chunk_015",
        "chunk_index": 14,
        "section": "methods",
        "content_hash": "f4d3b2a1c567",
        "processed_date": "2024-09-23T15:30:00Z",

        # Only include fields you actually want
        "document_title": "TEG Guidelines",
        "quality_score": 0.89,
        "human_approved": True,
        "chunk_length": 487,
        "medical_specialty": "hematology",
        "citations": [
            Citation("ref_042", ["Smith, J."], "TEG in bleeding", 2023, "Blood", None, "(Smith, 2023)")
        ],
        "citation_count": 1
    }

    # Create metadata
    metadata = create_chunk_metadata(sample_chunk)

    # Show results
    print("SIMPLE PINECONE METADATA")
    print("=" * 40)
    print(json.dumps(metadata, indent=2, default=str))

    # Show size
    chunk_metadata = ChunkMetadata(**sample_chunk)
    size = chunk_metadata.estimate_size()
    print(f"\nMetadata size: {size} bytes")
    print(f"Fields included: {len(metadata)}")

    print(f"\nTo customize:")
    print("1. Comment out fields you don't want in the ChunkMetadata class")
    print("2. Or just don't pass them in your chunk_data")
    print("3. The system will only include fields that have values")