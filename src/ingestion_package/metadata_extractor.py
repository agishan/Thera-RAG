"""
Metadata extraction for enhanced document referencing
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document


@dataclass
class DocumentReference:
    """Structured reference information for documents"""
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


class DocumentMetadataExtractor:
    """Extract document-level metadata for proper referencing"""

    def __init__(self):
        """Initialize with common academic patterns"""
        self.author_patterns = [
            # Various author formats
            r"Authors?:?\s*([A-Z][a-zA-Z\s,.-]+(?:,\s*[A-Z][a-zA-Z\s,.-]+)*)",
            r"By:?\s*([A-Z][a-zA-Z\s,.-]+(?:,\s*[A-Z][a-zA-Z\s,.-]+)*)",
            r"([A-Z][a-zA-Z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*(?:,\s*[A-Z][a-zA-Z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*)*)",
            r"([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+(?:,\s*[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)*)"
        ]

        self.year_patterns = [
            r"\b(19|20)\d{2}\b",
            r"\((\d{4})\)",
            r"Published:?\s*(\d{4})",
            r"Date:?\s*(\d{4})",
            r"Year:?\s*(\d{4})"
        ]

        self.doi_patterns = [
            r"doi:?\s*(10\.\d+\/[^\s]+)",
            r"DOI:?\s*(10\.\d+\/[^\s]+)",
            r"https?://doi\.org/(10\.\d+\/[^\s]+)",
            r"Digital Object Identifier:?\s*(10\.\d+\/[^\s]+)"
        ]

        self.journal_patterns = [
            r"Journal of ([A-Z][a-zA-Z\s]+)",
            r"([A-Z][a-zA-Z\s]+)\s+Journal",
            r"Published in:?\s*([A-Z][a-zA-Z\s]+)",
            r"Source:?\s*([A-Z][a-zA-Z\s]+)",
            r"In:?\s*([A-Z][a-zA-Z\s]+)\s*\(",
            r"Proceedings of ([A-Z][a-zA-Z\s]+)"
        ]

    def extract_document_metadata(
        self,
        text: str,
        filename: str,
        structured_content: Optional[Dict] = None
    ) -> DocumentReference:
        """
        Extract comprehensive document metadata

        Args:
            text: Full document text
            filename: Original filename
            structured_content: Additional structured content from loader

        Returns:
            DocumentReference with extracted metadata
        """
        # Extract basic bibliographic information
        title = self._extract_title(text, filename)
        authors = self._extract_authors(text)
        year = self._extract_year(text)
        journal = self._extract_journal(text)
        doi = self._extract_doi(text)
        doc_type = self._classify_document_type(text, filename)

        # Generate citation
        citation = self._generate_citation(title, authors, year, journal)

        return DocumentReference(
            source_title=title,
            authors=authors,
            publication_year=year,
            journal=journal,
            doi=doi,
            citation_format=citation,
            document_type=doc_type
        )

    def _extract_title(self, text: str, filename: str) -> str:
        """Extract document title using multiple strategies"""
        lines = text.split('\n')[:30]  # Check first 30 lines

        # Strategy 1: Look for large headers or emphasized text
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Skip common non-title content
            skip_patterns = [
                r'^(abstract|introduction|table of contents|contents)',
                r'^(page \d+|chapter \d+|\d+\s*$)',
                r'^(author|date|published|doi)',
                r'^\w{1,3}$'  # Single short words
            ]

            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue

            # Clean up potential title
            clean_line = re.sub(r'^#+\s*', '', line)  # Remove markdown headers
            clean_line = re.sub(r'\s+', ' ', clean_line)  # Normalize whitespace
            clean_line = clean_line.strip('*_`')  # Remove markdown formatting

            # Check if it looks like a title
            if (len(clean_line.split()) >= 3 and
                not clean_line.endswith(':') and
                len(clean_line) <= 200):  # Reasonable title length
                return clean_line

        # Strategy 2: Look for largest text block in first few lines
        candidates = []
        for line in lines[:10]:
            line = line.strip()
            if 10 <= len(line) <= 200 and len(line.split()) >= 3:
                candidates.append(line)

        if candidates:
            # Return the longest reasonable candidate
            return max(candidates, key=len)

        # Fallback: Use filename
        return Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

    def _extract_authors(self, text: str) -> Optional[str]:
        """Extract author information with improved patterns"""
        # Focus on first 3000 characters where authors typically appear
        search_text = text[:3000]

        for pattern in self.author_patterns:
            matches = re.finditer(pattern, search_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                authors = match.group(1).strip()

                # Clean up author string
                authors = re.sub(r'\s+', ' ', authors)  # Normalize whitespace
                authors = re.sub(r'[,\s]+$', '', authors)  # Remove trailing commas/spaces

                # Validate author format (should have at least one capital letter pattern)
                if (len(authors) > 5 and
                    re.search(r'[A-Z][a-z]+', authors) and
                    len(authors.split()) >= 2):
                    return authors

        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """Extract publication year with validation"""
        search_text = text[:2000]  # Check first 2000 characters

        year_candidates = []
        for pattern in self.year_patterns:
            matches = re.findall(pattern, search_text)
            for match in matches:
                if isinstance(match, tuple):
                    year = match[0] if match[0] else match[1]
                else:
                    year = match

                if year and year.isdigit():
                    year_int = int(year)
                    if 1950 <= year_int <= 2030:  # Reasonable range
                        year_candidates.append((year, search_text.find(year)))

        if year_candidates:
            # Return the year that appears earliest in the text
            return min(year_candidates, key=lambda x: x[1])[0]

        return None

    def _extract_journal(self, text: str) -> Optional[str]:
        """Extract journal name with improved patterns"""
        search_text = text[:1500]

        for pattern in self.journal_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                journal = match.group(1).strip()
                # Clean up journal name
                journal = re.sub(r'\s+', ' ', journal)
                if (len(journal.split()) >= 2 and
                    len(journal) > 5 and
                    not journal.lower().startswith(('the', 'a ', 'an '))):
                    return journal

        return None

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI with comprehensive patterns"""
        for pattern in self.doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doi = match.group(1)
                # Validate DOI format
                if re.match(r'10\.\d+\/.+', doi):
                    return doi
        return None

    def _classify_document_type(self, text: str, filename: str) -> str:
        """Classify document type for better categorization"""
        text_lower = text[:2000].lower()
        filename_lower = filename.lower()

        # Research paper indicators
        research_indicators = [
            'abstract', 'methodology', 'results', 'conclusion',
            'references', 'bibliography', 'study', 'research'
        ]

        # Clinical guide indicators
        guide_indicators = [
            'guideline', 'protocol', 'recommendation', 'standard',
            'procedure', 'treatment', 'therapy', 'care'
        ]

        # Review indicators
        review_indicators = [
            'systematic review', 'meta-analysis', 'literature review',
            'review article', 'survey'
        ]

        if any(indicator in text_lower for indicator in review_indicators):
            return 'review_paper'
        elif any(indicator in text_lower for indicator in research_indicators):
            return 'research_paper'
        elif any(indicator in text_lower for indicator in guide_indicators):
            return 'clinical_guide'
        elif 'manual' in filename_lower or 'handbook' in filename_lower:
            return 'manual'
        else:
            return 'document'

    def _generate_citation(
        self,
        title: str,
        authors: Optional[str],
        year: Optional[str],
        journal: Optional[str]
    ) -> str:
        """Generate a formatted citation"""
        parts = []

        if authors:
            # Format authors (limit to first few if many)
            author_list = [a.strip() for a in authors.split(',')]
            if len(author_list) > 3:
                formatted_authors = f"{author_list[0]} et al."
            else:
                formatted_authors = authors
            parts.append(formatted_authors)

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
        """
        Initialize with document reference

        Args:
            doc_ref: DocumentReference containing document-level metadata
        """
        self.doc_ref = doc_ref

    def enhance_chunk_metadata(
        self,
        chunk: Document,
        full_text: str,
        chunk_index: int
    ) -> Document:
        """
        Enhance a chunk with comprehensive metadata

        Args:
            chunk: Document chunk to enhance
            full_text: Full document text for context
            chunk_index: Index of this chunk in the document

        Returns:
            Enhanced Document with rich metadata
        """
        content = chunk.page_content
        base_metadata = chunk.metadata.copy()

        # Extract positional information
        page_start, page_end = self._estimate_page_numbers(content, full_text)
        hierarchy = self._extract_section_hierarchy(content, full_text)

        # Generate enhanced metadata
        enhanced_metadata = self._create_enhanced_metadata(
            content, chunk_index, base_metadata, page_start, page_end, hierarchy
        )

        # Create new document with enhanced metadata
        return Document(page_content=content, metadata=enhanced_metadata)

    def _estimate_page_numbers(self, content: str, full_text: str) -> Tuple[Optional[int], Optional[int]]:
        """Estimate page numbers for chunk content"""
        # Look for explicit page markers in content
        page_patterns = [
            r"Page\s+(\d+)",
            r"p\.\s*(\d+)",
            r"- (\d+) -",
            r"\[(\d+)\]"
        ]

        found_pages = []
        for pattern in page_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_pages.extend([int(m) for m in matches if m.isdigit()])

        if found_pages:
            return min(found_pages), max(found_pages)

        # Estimate based on position in full document
        if content[:100] in full_text:
            position_ratio = full_text.find(content[:100]) / len(full_text)
            estimated_page = max(1, int(position_ratio * 50))  # Assume ~50 pages max
            return estimated_page, estimated_page

        return None, None

    def _extract_section_hierarchy(self, content: str, full_text: str) -> List[str]:
        """Extract section hierarchy for contextual placement"""
        hierarchy = []

        # Find where this content appears in the full text
        content_start = full_text.find(content[:200]) if len(content) >= 200 else -1
        if content_start == -1:
            return hierarchy

        # Look for headers before this content
        preceding_text = full_text[:content_start + 1000]  # Include some overlap

        # Extract headers with their levels
        header_patterns = [
            (r'^# (.+)$', 1),      # H1
            (r'^## (.+)$', 2),     # H2
            (r'^### (.+)$', 3),    # H3
            (r'^#### (.+)$', 4),   # H4
        ]

        found_headers = []
        for pattern, level in header_patterns:
            matches = re.finditer(pattern, preceding_text, re.MULTILINE)
            for match in matches:
                header_text = match.group(1).strip()
                found_headers.append((match.start(), level, header_text))

        # Sort by position and build hierarchy
        found_headers.sort()
        current_hierarchy = {}

        for pos, level, title in found_headers:
            # Update hierarchy at this level and clear deeper levels
            current_hierarchy[level] = title
            # Remove deeper levels
            for deeper_level in list(current_hierarchy.keys()):
                if deeper_level > level:
                    del current_hierarchy[deeper_level]

        # Convert to ordered list
        for level in sorted(current_hierarchy.keys()):
            hierarchy.append(current_hierarchy[level])

        return hierarchy

    def _create_enhanced_metadata(
        self,
        content: str,
        chunk_index: int,
        base_metadata: Dict,
        page_start: Optional[int],
        page_end: Optional[int],
        hierarchy: List[str]
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for chunk"""

        # Generate content hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Generate unique chunk identifier
        chunk_id = f"{Path(self.doc_ref.source_title).stem}_{chunk_index:03d}_{content_hash[:8]}"

        # Extract preview text
        sentences = re.split(r'[.!?]+', content.strip())
        preview = sentences[0].strip() if sentences else ""
        if len(preview) > 150:
            preview = preview[:150] + "..."

        # Calculate content statistics
        words = content.split()
        sentences_clean = [s.strip() for s in sentences if s.strip()]
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Build enhanced metadata
        enhanced_metadata = {
            # Core identification
            "chunk_id": chunk_id,
            "content_hash": content_hash,
            "chunk_index": chunk_index,

            # Document reference information
            "source_title": self.doc_ref.source_title,
            "source_authors": self.doc_ref.authors,
            "source_year": self.doc_ref.publication_year,
            "source_journal": self.doc_ref.journal,
            "source_doi": self.doc_ref.doi,
            "source_citation": self.doc_ref.citation_format,
            "document_type": self.doc_ref.document_type,

            # Location within document
            "page_start": page_start,
            "page_end": page_end,
            "section_hierarchy": hierarchy,
            "hierarchy_breadcrumb": " > ".join(hierarchy) if hierarchy else "",

            # Content characteristics (NO 500 CHAR LIMIT)
            "text": content,  # FULL CONTENT
            "preview": preview,
            "chunk_size": len(content),
            "word_count": len(words),
            "sentence_count": len(sentences_clean),
            "avg_word_length": round(avg_word_length, 1),

            # Citation information
            "citation_text": self._generate_chunk_citation(chunk_index, page_start, page_end),
            "reference_url": self._generate_reference_url(),

            # Processing metadata
            "processed_at": datetime.now().isoformat(),
            "processing_version": "enhanced_v2.0",

            # Preserve original metadata
            **base_metadata
        }

        return enhanced_metadata

    def _generate_chunk_citation(
        self,
        chunk_index: int,
        page_start: Optional[int],
        page_end: Optional[int]
    ) -> str:
        """Generate citeable reference for this chunk"""
        base_citation = self.doc_ref.citation_format or self.doc_ref.source_title

        location_parts = []
        if page_start and page_end:
            if page_start == page_end:
                location_parts.append(f"p. {page_start}")
            else:
                location_parts.append(f"pp. {page_start}-{page_end}")

        location_parts.append(f"chunk {chunk_index + 1}")

        if location_parts:
            return f"{base_citation}, {', '.join(location_parts)}"
        else:
            return base_citation

    def _generate_reference_url(self) -> Optional[str]:
        """Generate reference URL if DOI available"""
        if self.doc_ref.doi:
            return f"https://doi.org/{self.doc_ref.doi}"
        return None