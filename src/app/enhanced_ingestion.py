"""
Enhanced Ingestion with Rich Metadata for Better References
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class DocumentReference:
    """Structured reference information for chunks"""
    source_title: str
    authors: Optional[str] = None
    publication_year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_hierarchy: Optional[List[str]] = None
    citation_format: Optional[str] = None


class EnhancedMetadataExtractor:
    """Extract rich metadata for better reference linking"""

    def __init__(self):
        self.author_patterns = [
            r"Authors?:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)",
            r"By:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)",
            r"([A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*(?:,\s*[A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*)*)"
        ]

        self.year_patterns = [
            r"\b(19|20)\d{2}\b",
            r"\((\d{4})\)",
            r"Published:?\s*(\d{4})"
        ]

        self.doi_patterns = [
            r"doi:?\s*(10\.\d+\/[^\s]+)",
            r"DOI:?\s*(10\.\d+\/[^\s]+)",
            r"https?://doi\.org/(10\.\d+\/[^\s]+)"
        ]

    def extract_document_metadata(self, text: str, filename: str) -> DocumentReference:
        """Extract document-level metadata from full text"""

        # Extract title (usually first substantial line or header)
        title = self._extract_title(text, filename)
        authors = self._extract_authors(text)
        year = self._extract_year(text)
        journal = self._extract_journal(text)
        doi = self._extract_doi(text)
        citation = self._generate_citation(title, authors, year, journal)

        return DocumentReference(
            source_title=title,
            authors=authors,
            publication_year=year,
            journal=journal,
            doi=doi,
            citation_format=citation
        )

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract document title"""
        lines = text.split('\n')[:20]  # Check first 20 lines

        for line in lines:
            line = line.strip()
            # Skip empty lines and common headers
            if not line or line.lower().startswith(('abstract', 'introduction', 'table of contents')):
                continue

            # Look for substantial titles (not single words)
            if len(line) > 10 and not line.startswith('#'):
                # Clean up the title
                title = re.sub(r'^#+\s*', '', line)  # Remove markdown headers
                title = re.sub(r'\s+', ' ', title)   # Normalize whitespace
                if len(title.split()) >= 3:  # At least 3 words
                    return title

        # Fallback to filename
        return Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

    def _extract_authors(self, text: str) -> Optional[str]:
        """Extract author information"""
        first_page = text[:2000]  # Check first 2000 chars

        for pattern in self.author_patterns:
            match = re.search(pattern, first_page, re.IGNORECASE)
            if match:
                authors = match.group(1).strip()
                # Clean up author list
                authors = re.sub(r'\s+', ' ', authors)
                return authors

        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """Extract publication year"""
        first_page = text[:1500]

        for pattern in self.year_patterns:
            matches = re.findall(pattern, first_page)
            if matches:
                years = [year for year in matches if isinstance(year, str) and year.isdigit()]
                if not years:
                    years = [match if isinstance(match, str) else match[0] for match in matches]

                # Return most reasonable year (1980-2030)
                valid_years = [y for y in years if 1980 <= int(y) <= 2030]
                if valid_years:
                    return valid_years[0]

        return None

    def _extract_journal(self, text: str) -> Optional[str]:
        """Extract journal name"""
        first_page = text[:1000]

        # Common journal patterns
        journal_patterns = [
            r"Journal of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Journal",
            r"Published in:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"Source:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]

        for pattern in journal_patterns:
            match = re.search(pattern, first_page, re.IGNORECASE)
            if match:
                journal = match.group(1).strip()
                if len(journal.split()) >= 2:  # At least 2 words
                    return journal

        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI"""
        for pattern in self.doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _generate_citation(self, title: str, authors: Optional[str],
                          year: Optional[str], journal: Optional[str]) -> str:
        """Generate a formatted citation"""
        parts = []

        if authors:
            parts.append(authors)
        if year:
            parts.append(f"({year})")
        if title:
            parts.append(f'"{title}"')
        if journal:
            parts.append(f"*{journal}*")

        return ". ".join(parts) if parts else title


class ChunkMetadataEnhancer:
    """Enhance chunk-level metadata with contextual information"""

    def __init__(self, doc_ref: DocumentReference):
        self.doc_ref = doc_ref

    def extract_page_info(self, content: str, full_text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract page information from content"""
        # Look for page markers in content
        page_patterns = [
            r"Page\s+(\d+)",
            r"- (\d+) -",
            r"\[(\d+)\]",
            r"^(\d+)$"  # Standalone numbers
        ]

        pages = []
        for pattern in page_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            pages.extend([int(m) for m in matches if m.isdigit()])

        if pages:
            return min(pages), max(pages)

        # Estimate based on position in full text
        position_ratio = full_text.find(content[:50]) / len(full_text) if content[:50] in full_text else 0.5
        estimated_page = max(1, int(position_ratio * 50))  # Assume ~50 pages max

        return estimated_page, estimated_page

    def extract_section_hierarchy(self, content: str, full_text: str) -> List[str]:
        """Extract section hierarchy for this chunk"""
        hierarchy = []

        # Find headers before this content in the full text
        content_start = full_text.find(content[:100])
        if content_start == -1:
            content_start = 0

        preceding_text = full_text[:content_start + 500]  # Include some overlap

        # Extract headers
        header_patterns = [
            (r'^# (.+)$', 1),      # H1
            (r'^## (.+)$', 2),     # H2
            (r'^### (.+)$', 3),    # H3
            (r'^#### (.+)$', 4),   # H4
        ]

        headers = []
        for pattern, level in header_patterns:
            matches = re.finditer(pattern, preceding_text, re.MULTILINE)
            for match in matches:
                headers.append((match.start(), level, match.group(1).strip()))

        # Sort by position and build hierarchy
        headers.sort()
        current_hierarchy = {}

        for pos, level, title in headers:
            # Update hierarchy at this level and clear deeper levels
            current_hierarchy[level] = title
            for deeper_level in list(current_hierarchy.keys()):
                if deeper_level > level:
                    del current_hierarchy[deeper_level]

        # Convert to ordered list
        hierarchy = [current_hierarchy[level] for level in sorted(current_hierarchy.keys())]

        return hierarchy

    def create_enhanced_metadata(self, content: str, full_text: str,
                                chunk_index: int, base_metadata: Dict) -> Dict:
        """Create enhanced metadata for a chunk"""

        # Extract positional info
        page_start, page_end = self.extract_page_info(content, full_text)
        hierarchy = self.extract_section_hierarchy(content, full_text)

        # Create content hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Generate unique identifier
        chunk_id = f"{Path(self.doc_ref.source_title).stem}_{chunk_index:03d}_{content_hash[:8]}"

        # Extract first sentence for preview
        sentences = re.split(r'[.!?]+', content.strip())
        first_sentence = sentences[0].strip() if sentences else ""

        # Calculate reading level (simple proxy)
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        enhanced_metadata = {
            # Core identification
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "chunk_index": chunk_index,

            # Document reference
            "source_title": self.doc_ref.source_title,
            "source_authors": self.doc_ref.authors,
            "source_year": self.doc_ref.publication_year,
            "source_journal": self.doc_ref.journal,
            "source_doi": self.doc_ref.doi,
            "source_citation": self.doc_ref.citation_format,

            # Location within document
            "page_start": page_start,
            "page_end": page_end,
            "section_hierarchy": hierarchy,
            "hierarchy_breadcrumb": " > ".join(hierarchy) if hierarchy else "",

            # Content characteristics
            "text": content,  # Full content for retrieval
            "preview": first_sentence[:150] + "..." if len(first_sentence) > 150 else first_sentence,
            "chunk_size": len(content),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": round(avg_word_length, 1),

            # Citeable reference
            "citation_text": self._generate_chunk_citation(chunk_index, page_start, page_end),
            "reference_url": self._generate_reference_url(),

            # Processing metadata
            "processed_at": datetime.now().isoformat(),
            "processing_version": "enhanced_v1.0",

            # Preserve original metadata
            **base_metadata
        }

        return enhanced_metadata

    def _generate_chunk_citation(self, chunk_index: int,
                               page_start: Optional[int], page_end: Optional[int]) -> str:
        """Generate a citeable reference for this specific chunk"""
        base_citation = self.doc_ref.citation_format or self.doc_ref.source_title

        location_info = []
        if page_start and page_end:
            if page_start == page_end:
                location_info.append(f"p. {page_start}")
            else:
                location_info.append(f"pp. {page_start}-{page_end}")

        location_info.append(f"chunk {chunk_index + 1}")

        return f"{base_citation}, {', '.join(location_info)}"

    def _generate_reference_url(self) -> Optional[str]:
        """Generate a reference URL if DOI available"""
        if self.doc_ref.doi:
            return f"https://doi.org/{self.doc_ref.doi}"
        return None


def enhance_existing_chunks(input_dir: str, output_dir: str):
    """Enhance existing chunk files with better metadata"""

    os.makedirs(output_dir, exist_ok=True)
    metadata_extractor = EnhancedMetadataExtractor()

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json') and not f.endswith('_stats.json')]

    for json_file in json_files:
        print(f"\n📄 Enhancing {json_file}...")

        # Load existing chunks
        json_path = os.path.join(input_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # Load original markdown if available
        md_file = json_file.replace('.json', '.md')
        md_path = os.path.join(input_dir, md_file)
        full_text = ""
        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

        # Extract document-level metadata
        filename = json_file.replace('.json', '.pdf')
        doc_ref = metadata_extractor.extract_document_metadata(full_text or "", filename)

        # Enhance each chunk
        enhancer = ChunkMetadataEnhancer(doc_ref)
        enhanced_chunks = []

        for i, chunk_data in enumerate(chunks):
            content = chunk_data.get('content', chunk_data.get('page_content', ''))
            base_metadata = chunk_data.get('metadata', {})

            enhanced_metadata = enhancer.create_enhanced_metadata(
                content, full_text, i, base_metadata
            )

            enhanced_chunks.append({
                'content': content,
                'metadata': enhanced_metadata
            })

        # Save enhanced chunks
        output_path = os.path.join(output_dir, f"enhanced_{json_file}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_chunks, f, indent=2, ensure_ascii=False)

        print(f"✅ Enhanced {len(enhanced_chunks)} chunks with rich metadata")
        print(f"   Document: {doc_ref.source_title}")
        print(f"   Authors: {doc_ref.authors or 'Unknown'}")
        print(f"   Year: {doc_ref.publication_year or 'Unknown'}")


if __name__ == "__main__":
    # Enhance existing chunks
    input_directory = "src/Ingestion/outputs"
    output_directory = "src/Ingestion/enhanced_outputs"

    enhance_existing_chunks(input_directory, output_directory)