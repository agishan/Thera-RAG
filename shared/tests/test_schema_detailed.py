#!/usr/bin/env python3
"""
Schema sanity test using the current simple Pinecone schema.
"""

import sys
from pathlib import Path
import json

# Add shared models path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from simple_pinecone_schema import ChunkMetadata, Citation, create_chunk_metadata


def test_schema_details():
    sample_chunk = {
        "document_id": "vha-guideline-2024",
        "chunk_id": "chunk_015",
        "chunk_index": 14,
        "section": "methods",
        "content_hash": "f4d3b2a1c567",
        "processed_date": "2024-09-23T15:30:00Z",

        # Optional fields
        "document_title": "Use of viscoelastic haemostatic assays in management of major bleeding",
        "chunk_length": 487,
        "medical_specialty": "hematology",
        "citations": [
            Citation(
                "ref_042",
                ["Smith, J.A.", "Brown, K.L."],
                "Thromboelastography in bleeding disorders: systematic review",
                2023,
                "Blood",
                "10.1182/blood.2023.142156",
                "(Smith & Brown, 2023)",
            )
        ],
        "citation_count": 1,
    }

    metadata = create_chunk_metadata(sample_chunk)
    encoded = json.dumps(metadata, default=str)

    # Basic assertions
    assert metadata["document_id"] == "vha-guideline-2024"
    assert metadata["chunk_id"] == "chunk_015"
    assert metadata["section"] == "methods"
    assert "citations" in metadata and isinstance(metadata["citations"], list)
    # Size check
    size = len(encoded.encode("utf-8"))
    assert size > 50


if __name__ == "__main__":
    test_schema_details()

