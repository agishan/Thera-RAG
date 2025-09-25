"""
Citation Resolver - Resolves lightweight citation references to full citations at inference time

This module provides utilities to resolve citation ref_ids back to full citation
objects using the citation index, enabling complete citation information during
RAG response generation.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class CitationResolver:
    """Resolve lightweight citation references using citation index"""

    def __init__(self, citation_index_path: Optional[str] = None):
        """
        Initialize citation resolver

        Args:
            citation_index_path: Path to citation_index.json file
                                If None, will try to load from common locations
        """
        self.citation_index: Dict[str, Dict[str, Any]] = {}
        self.index_loaded = False

        if citation_index_path:
            self.load_citation_index(citation_index_path)
        else:
            self._try_auto_load_index()

    def load_citation_index(self, citation_index_path: str) -> bool:
        """
        Load citation index from JSON file

        Args:
            citation_index_path: Path to citation index file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = Path(citation_index_path)
            if not index_path.exists():
                print(f"Warning: Citation index not found at {index_path}")
                return False

            with open(index_path, 'r', encoding='utf-8') as f:
                self.citation_index = json.load(f)

            self.index_loaded = True
            print(f"Loaded citation index with {len(self.citation_index)} citations from {index_path}")
            return True

        except Exception as e:
            print(f"Error loading citation index: {e}")
            return False

    def _try_auto_load_index(self):
        """Try to auto-load citation index from common locations"""
        common_paths = [
            "enhanced_output/vha-guideline/5_citation_index/citation_index.json",
            "enhanced_output/vha-guideline/citation_index.json",
            "citation_index.json"
        ]

        for path in common_paths:
            if self.load_citation_index(path):
                return

        print("No citation index found in common locations. Load manually with load_citation_index()")

    def resolve_chunk_citations(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve citations in a single chunk

        Args:
            chunk_data: Chunk data with potentially lightweight citations

        Returns:
            Chunk data with resolved citations
        """
        if not self.index_loaded:
            print("Warning: Citation index not loaded - cannot resolve citations")
            return chunk_data

        # Create a copy to avoid modifying original
        resolved_chunk = json.loads(json.dumps(chunk_data))

        # Check for citations in metadata
        metadata = resolved_chunk.get("metadata", {})
        citations = metadata.get("citations", [])

        if not citations:
            return resolved_chunk

        resolved_citations = []

        for citation in citations:
            if isinstance(citation, dict):
                # Check if this is a lightweight citation with ref_id
                if "ref_id" in citation:
                    resolved_citation = self._resolve_lightweight_citation(citation)
                    if resolved_citation:
                        resolved_citations.append(resolved_citation)
                else:
                    # Already a full citation
                    resolved_citations.append(citation)

        # Update with resolved citations
        if resolved_citations:
            resolved_chunk["metadata"]["citations"] = resolved_citations
            resolved_chunk["metadata"]["resolved_citation_count"] = len(resolved_citations)

        return resolved_chunk

    def resolve_chunks_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve citations in multiple chunks

        Args:
            chunks: List of chunk data

        Returns:
            List of chunks with resolved citations
        """
        if not self.index_loaded:
            print("Warning: Citation index not loaded - returning chunks unchanged")
            return chunks

        resolved_chunks = []

        for chunk in chunks:
            resolved_chunk = self.resolve_chunk_citations(chunk)
            resolved_chunks.append(resolved_chunk)

        total_resolved = sum(
            len(chunk.get("metadata", {}).get("citations", []))
            for chunk in resolved_chunks
        )

        if total_resolved > 0:
            print(f"Resolved {total_resolved} citations across {len(chunks)} chunks")

        return resolved_chunks

    def _resolve_lightweight_citation(self, lightweight_citation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve a single lightweight citation to full citation

        Args:
            lightweight_citation: {"inline": "...", "ref_id": "..."}

        Returns:
            Full citation object or None if not found
        """
        ref_id = lightweight_citation.get("ref_id")
        if not ref_id:
            return None

        # Look up in citation index
        full_citation_data = self.citation_index.get(ref_id)
        if not full_citation_data:
            print(f"Warning: Citation ref_id '{ref_id}' not found in index")
            return None

        # Build full citation object
        resolved_citation = {
            "inline_text": lightweight_citation.get("inline", ""),
            "full_reference": full_citation_data.get("full_reference", ""),
            "authors": full_citation_data.get("authors", []),
            "year": full_citation_data.get("year", ""),
            "title": full_citation_data.get("title", ""),
            "journal": full_citation_data.get("journal"),
            "doi": full_citation_data.get("doi"),
            "ref_id": ref_id  # Keep ref_id for tracking
        }

        return resolved_citation

    def format_citations_for_response(
        self,
        chunks: List[Dict[str, Any]],
        format_style: str = "inline"
    ) -> Dict[str, Any]:
        """
        Format citations from chunks for inclusion in RAG response

        Args:
            chunks: List of chunks with resolved citations
            format_style: "inline", "references", or "both"

        Returns:
            Formatted citation information
        """
        if not chunks:
            return {"citations": [], "reference_list": []}

        all_citations = []
        unique_citations = {}

        # Collect all citations
        for chunk in chunks:
            citations = chunk.get("metadata", {}).get("citations", [])
            for citation in citations:
                if isinstance(citation, dict) and citation.get("full_reference"):
                    ref_id = citation.get("ref_id", citation.get("inline_text", "unknown"))
                    unique_citations[ref_id] = citation
                    all_citations.append(citation)

        # Format based on style
        if format_style == "inline":
            inline_citations = [cit.get("inline_text", "") for cit in all_citations if cit.get("inline_text")]
            return {"inline_citations": inline_citations}

        elif format_style == "references":
            reference_list = [
                cit.get("full_reference", "") for cit in unique_citations.values()
                if cit.get("full_reference")
            ]
            return {"reference_list": reference_list}

        elif format_style == "both":
            inline_citations = [cit.get("inline_text", "") for cit in all_citations if cit.get("inline_text")]
            reference_list = [
                cit.get("full_reference", "") for cit in unique_citations.values()
                if cit.get("full_reference")
            ]
            return {
                "inline_citations": inline_citations,
                "reference_list": reference_list,
                "citation_count": len(unique_citations)
            }

        return {"citations": all_citations}

    def get_citation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded citation index"""
        if not self.index_loaded:
            return {"loaded": False}

        stats = {
            "loaded": True,
            "total_citations": len(self.citation_index),
            "citations_by_decade": {},
            "top_authors": {}
        }

        # Analyze by decade
        decade_counts = {}
        author_counts = {}

        for ref_id, citation_data in self.citation_index.items():
            year = citation_data.get("year", "")
            if year:
                try:
                    year_int = int(str(year))
                    decade = f"{(year_int // 10) * 10}s"
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
                except (ValueError, TypeError):
                    pass

            # Count authors
            authors = citation_data.get("authors", [])
            if isinstance(authors, list):
                for author in authors[:1]:  # Just first author
                    if author:
                        author_counts[str(author)] = author_counts.get(str(author), 0) + 1

        # Top 10 decades and authors
        stats["citations_by_decade"] = dict(
            sorted(decade_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        stats["top_authors"] = dict(
            sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return stats


def resolve_citations_in_chunks(chunks: List[Dict], citation_index_path: str) -> List[Dict]:
    """
    Convenience function to resolve citations in chunks

    Args:
        chunks: List of chunk data
        citation_index_path: Path to citation index file

    Returns:
        Chunks with resolved citations
    """
    resolver = CitationResolver(citation_index_path)
    return resolver.resolve_chunks_citations(chunks)


if __name__ == "__main__":
    # Test citation resolver
    resolver = CitationResolver()

    if resolver.index_loaded:
        stats = resolver.get_citation_stats()
        print("\nCitation Index Stats:")
        print(json.dumps(stats, indent=2))

        # Test with sample lightweight citation
        sample_chunk = {
            "content": "Test content with citations",
            "metadata": {
                "chunk_id": "test_001",
                "citations": [
                    {"inline": "(Andreasen et al, 2011)", "ref_id": "andreasen_2011"},
                    {"inline": "(Armstrong et al, 2011)", "ref_id": "armstrong_2011"}
                ]
            }
        }

        resolved_chunk = resolver.resolve_chunk_citations(sample_chunk)
        print("\nSample Resolution:")
        print(json.dumps(resolved_chunk.get("metadata", {}).get("citations", [])[:1], indent=2))
    else:
        print("No citation index found for testing")