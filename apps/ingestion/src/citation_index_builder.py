"""
Citation Index Builder - Creates deduplicated citation index and lightweight chunk references

This module extracts all citations from enhanced chunks, deduplicates them,
and creates a separate citation index file while replacing verbose citation
objects in chunks with lightweight references.
"""

import json
import re
import hashlib
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict


class CitationIndexBuilder:
    """Build deduplicated citation index and convert chunks to use lightweight references"""

    def __init__(self):
        self.citation_index: Dict[str, Dict[str, Any]] = {}
        self.ref_id_map: Dict[str, str] = {}  # Maps citation hash to ref_id

    def process_enhanced_chunks(self, enhanced_chunks_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Process enhanced chunks file to create citation index and lightweight chunk references

        Args:
            enhanced_chunks_file: Path to enhanced chunks JSON file
            output_dir: Directory to save citation index and updated chunks

        Returns:
            Processing statistics
        """
        print(f"Building citation index from {enhanced_chunks_file}")

        # Load enhanced chunks
        with open(enhanced_chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        original_chunks = len(chunks)
        print(f"   Processing {original_chunks} chunks...")

        # Phase 1: Extract and deduplicate all citations
        all_citations = self._extract_all_citations(chunks)
        self.citation_index = self._build_citation_index(all_citations)

        # Phase 2: Replace verbose citations in chunks with lightweight references
        lightweight_chunks = self._create_lightweight_chunks(chunks)

        # Phase 3: Save outputs
        self._save_citation_index(output_dir)
        lightweight_file = self._save_lightweight_chunks(lightweight_chunks, output_dir, enhanced_chunks_file.stem)

        # Calculate size savings
        original_size = enhanced_chunks_file.stat().st_size
        new_size = lightweight_file.stat().st_size
        index_size = (output_dir / "citation_index.json").stat().st_size

        return {
            "original_chunks": original_chunks,
            "unique_citations": len(self.citation_index),
            "original_file_size_kb": round(original_size / 1024, 1),
            "lightweight_file_size_kb": round(new_size / 1024, 1),
            "citation_index_size_kb": round(index_size / 1024, 1),
            "total_new_size_kb": round((new_size + index_size) / 1024, 1),
            "size_reduction_percent": round((1 - (new_size + index_size) / original_size) * 100, 1),
            "artifacts": [
                str(output_dir / "citation_index.json"),
                str(lightweight_file)
            ]
        }

    def _extract_all_citations(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Extract all citation objects from all chunks"""
        all_citations = []

        for chunk in chunks:
            citations = chunk.get("metadata", {}).get("citations", [])
            if citations:
                all_citations.extend(citations)

        print(f"   Found {len(all_citations)} total citation instances")
        return all_citations

    def _build_citation_index(self, all_citations: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Build deduplicated citation index with stable ref_ids"""
        citation_groups = defaultdict(list)

        # Group citations by content hash for deduplication
        for citation in all_citations:
            citation_hash = self._hash_citation(citation)
            citation_groups[citation_hash].append(citation)

        print(f"   Deduplicated to {len(citation_groups)} unique citations")

        citation_index = {}

        for citation_hash, citation_group in citation_groups.items():
            # Use the most complete citation from the group
            best_citation = max(citation_group, key=self._citation_completeness_score)

            # Generate stable ref_id
            ref_id = self._generate_ref_id(best_citation)
            self.ref_id_map[citation_hash] = ref_id

            # Store in index
            citation_index[ref_id] = {
                "full_reference": best_citation.get("full_reference", ""),
                "authors": best_citation.get("authors", []),
                "year": best_citation.get("year", ""),
                "title": best_citation.get("title", ""),
                "journal": best_citation.get("journal"),
                "doi": best_citation.get("doi"),
                "inline_variants": list(set(c.get("inline_text", "") for c in citation_group if c.get("inline_text")))
            }

        return citation_index

    def _hash_citation(self, citation: Dict) -> str:
        """Create stable hash for citation deduplication"""
        # Use authors + year as primary key for grouping
        authors = citation.get("authors", [])
        year = citation.get("year", "")

        if isinstance(authors, list):
            author_key = "|".join(sorted([str(a).strip() for a in authors if a]))
        else:
            author_key = str(authors).strip()

        key = f"{author_key}_{year}".lower()
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _citation_completeness_score(self, citation: Dict) -> int:
        """Score citation completeness to pick the best version"""
        score = 0
        if citation.get("full_reference"): score += 5
        if citation.get("authors"): score += 3
        if citation.get("title"): score += 2
        if citation.get("journal"): score += 1
        if citation.get("doi"): score += 1
        return score

    def _generate_ref_id(self, citation: Dict) -> str:
        """Generate stable, readable ref_id"""
        authors = citation.get("authors", [])
        year = citation.get("year", "")

        # Get first author surname
        if isinstance(authors, list) and authors:
            first_author = str(authors[0]).strip()
        else:
            first_author = str(authors).strip() if authors else "unknown"

        # Clean first author (take last word as surname)
        surname_parts = first_author.split()
        surname = surname_parts[-1] if surname_parts else first_author
        surname = re.sub(r'[^\w]', '', surname.lower())

        # Clean year
        year_clean = re.sub(r'[^\d]', '', str(year))[:4] if year else "0000"

        base_ref_id = f"{surname}_{year_clean}"

        # Handle duplicates with suffix
        ref_id = base_ref_id
        counter = 1
        while ref_id in [v for v in self.ref_id_map.values()] or ref_id in self.citation_index:
            counter += 1
            ref_id = f"{base_ref_id}_{counter}"

        return ref_id

    def _create_lightweight_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Create truly lightweight chunks with only essential metadata"""
        lightweight_chunks = []

        # Define essential fields for vector storage
        essential_fields = [
            # Core identification
            'chunk_id', 'chunk_index',
            # Source information
            'source_title', 'source_authors', 'source_year', 'source_doi',
            # Location information
            'page_start', 'page_end', 'hierarchy_breadcrumb',
            # Document classification
            'document_type',
            # Reference information (keep for citation context)
            'citation_text', 'reference_url'
        ]

        for chunk in chunks:
            original_metadata = chunk.get("metadata", {})

            # Create new chunk with only essential metadata
            lightweight_metadata = {}

            # Copy essential fields
            for field in essential_fields:
                if field in original_metadata and original_metadata[field] is not None:
                    lightweight_metadata[field] = original_metadata[field]

            # Handle citations - convert to lightweight references
            if "citations" in original_metadata and original_metadata["citations"]:
                original_citations = original_metadata["citations"]
                lightweight_citations = []

                for citation in original_citations:
                    citation_hash = self._hash_citation(citation)
                    ref_id = self.ref_id_map.get(citation_hash)

                    if ref_id:
                        lightweight_citations.append({
                            "inline": citation.get("inline_text", ""),
                            "ref_id": ref_id
                        })

                lightweight_metadata["citations"] = lightweight_citations
                lightweight_metadata["citation_count"] = len(lightweight_citations)

                # Add author summary for search
                cited_authors = []
                for citation in original_citations:
                    authors = citation.get("authors", [])
                    if isinstance(authors, list) and authors:
                        cited_authors.append(str(authors[0]).split()[0])  # First name of first author

                if cited_authors:
                    lightweight_metadata["cited_authors"] = " | ".join(cited_authors[:5])  # First 5

            # Create lightweight chunk
            lightweight_chunk = {
                "content": chunk.get("content", ""),
                "metadata": lightweight_metadata
            }

            lightweight_chunks.append(lightweight_chunk)

        return lightweight_chunks

    def _save_citation_index(self, output_dir: Path):
        """Save citation index to JSON file"""
        index_file = output_dir / "citation_index.json"

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(self.citation_index, f, indent=2, ensure_ascii=False, default=str)

        print(f"   Saved citation index: {index_file}")
        print(f"   Index contains {len(self.citation_index)} unique citations")

    def _save_lightweight_chunks(self, chunks: List[Dict], output_dir: Path, original_name: str) -> Path:
        """Save lightweight chunks to JSON file"""
        lightweight_file = output_dir / f"{original_name}_lightweight.json"

        with open(lightweight_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)

        print(f"   Saved lightweight chunks: {lightweight_file}")
        return lightweight_file


def build_citation_index(enhanced_chunks_file: str, output_dir: str) -> Dict[str, Any]:
    """
    Convenience function to build citation index

    Args:
        enhanced_chunks_file: Path to enhanced chunks JSON file
        output_dir: Directory to save outputs

    Returns:
        Processing statistics
    """
    builder = CitationIndexBuilder()
    return builder.process_enhanced_chunks(Path(enhanced_chunks_file), Path(output_dir))


if __name__ == "__main__":
    # Test with vha-guideline
    enhanced_file = "enhanced_output/vha-guideline/enhanced_vha-guideline.json"
    output_dir = "enhanced_output/vha-guideline/5_citation_index"

    if Path(enhanced_file).exists():
        stats = build_citation_index(enhanced_file, output_dir)
        print("\nCitation Index Build Results:")
        print(json.dumps(stats, indent=2))
    else:
        print(f"Enhanced chunks file not found: {enhanced_file}")