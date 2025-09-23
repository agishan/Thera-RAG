"""
Advanced citation extraction using LLM + regex hybrid approach
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class Citation:
    """Represents an extracted citation"""
    inline_text: str  # The inline citation as it appears (e.g., "(Smith et al., 2023)")
    full_reference: str  # The full reference from bibliography
    authors: List[str]  # Extracted author names
    year: Optional[str] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pages: Optional[str] = None
    citation_type: str = "unknown"  # journal, book, conference, etc.
    confidence: float = 0.0  # Confidence in the extraction


@dataclass
class CitationMatch:
    """Links inline citations to their full references"""
    inline_citation: str
    chunk_location: str  # Where the inline citation appears
    full_reference: Citation
    context_before: str  # Text before the citation
    context_after: str   # Text after the citation


class CitationExtractor:
    """
    Hybrid LLM + regex system for extracting and linking citations
    """

    def __init__(self, llm_api_key: Optional[str] = None, use_llm: bool = True):
        """
        Initialize citation extractor

        Args:
            llm_api_key: Google API key for LLM-based extraction
            use_llm: Whether to use LLM for extraction (falls back to regex only)
        """
        self.use_llm = use_llm and llm_api_key is not None

        if self.use_llm:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=llm_api_key
            )
        else:
            self.llm = None

        # Regex patterns for various citation formats
        self.inline_patterns = [
            # Author-year formats
            r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?(?:,\s*[A-Z][a-zA-Z]+)*),?\s+(\d{4}[a-z]?)\)',
            r'\(([A-Z][a-zA-Z]+(?:\s+&\s+[A-Z][a-zA-Z]+)?),?\s+(\d{4}[a-z]?)\)',
            r'([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?)\s+\((\d{4}[a-z]?)\)',

            # Numbered citations
            r'\[(\d+(?:-\d+)?(?:,\s*\d+)*)\]',
            r'\((\d+(?:-\d+)?(?:,\s*\d+)*)\)',

            # Superscript style
            r'(\d+(?:-\d+)?(?:,\d+)*)'
        ]

        self.reference_patterns = [
            # Journal articles
            r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^,]+?),?\s*(\d+(?:\(\d+\))?),?\s*([^.]+)',

            # Books
            r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^:]+):\s*([^.]+)',

            # Simple format
            r'([A-Z][a-zA-Z\s,.-]+?)\s*\((\d{4})\)\s*([^.]+?)(?:\.\s*(.+?))?'
        ]

    def extract_citations_from_document(
        self,
        full_text: str,
        chunks: List[Document]
    ) -> Tuple[List[Citation], List[CitationMatch]]:
        """
        Extract citations from full document and link them to chunks

        Args:
            full_text: Complete document text
            chunks: List of document chunks

        Returns:
            Tuple of (extracted citations, citation matches to chunks)
        """
        # Step 1: Extract full references from bibliography/references section
        bibliography_citations = self._extract_bibliography_citations(full_text)

        # Step 2: Extract inline citations from chunks
        inline_citations = self._extract_inline_citations_from_chunks(chunks)

        # Step 3: Match inline citations to full references
        citation_matches = self._match_inline_to_full_citations(
            inline_citations, bibliography_citations, chunks
        )

        return bibliography_citations, citation_matches

    def _extract_bibliography_citations(self, full_text: str) -> List[Citation]:
        """Extract full citations from bibliography section"""

        # Find the references/bibliography section
        ref_section = self._find_references_section(full_text)
        if not ref_section:
            return []

        citations = []

        if self.use_llm:
            # Use LLM for high-quality extraction
            llm_citations = self._llm_extract_bibliography(ref_section)
            citations.extend(llm_citations)

        # Always run regex as backup/supplement
        regex_citations = self._regex_extract_bibliography(ref_section)
        citations.extend(regex_citations)

        # Deduplicate and return
        return self._deduplicate_citations(citations)

    def _find_references_section(self, text: str) -> Optional[str]:
        """Find and extract the references/bibliography section"""

        # Common section headers
        ref_headers = [
            r'(?i)^#+\s*references?\s*$',
            r'(?i)^#+\s*bibliography\s*$',
            r'(?i)^#+\s*works?\s+cited\s*$',
            r'(?i)^#+\s*literature\s+cited\s*$'
        ]

        for pattern in ref_headers:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                # Extract from this point to end or next major section
                start_pos = match.end()

                # Look for next major section
                next_section = re.search(r'\n#+\s*\w+', text[start_pos:])
                end_pos = start_pos + next_section.start() if next_section else len(text)

                return text[start_pos:end_pos].strip()

        return None

    def _llm_extract_bibliography(self, ref_section: str) -> List[Citation]:
        """Use LLM to extract structured citations from bibliography"""

        prompt = f"""
Extract all citations from this bibliography section and return them as a JSON list.
For each citation, extract:
- authors (list of author names)
- year
- title
- journal (if applicable)
- doi (if present)
- pages (if present)
- citation_type (journal, book, conference, etc.)

Bibliography section:
{ref_section[:3000]}  # Limit to avoid token limits

Return only valid JSON in this format:
[
  {{
    "authors": ["Author1", "Author2"],
    "year": "2023",
    "title": "Paper Title",
    "journal": "Journal Name",
    "doi": "10.1234/example",
    "pages": "123-145",
    "citation_type": "journal"
  }}
]
"""

        try:
            response = self.llm.invoke(prompt)
            citations_data = json.loads(response.content)

            citations = []
            for data in citations_data:
                citation = Citation(
                    inline_text="",  # Will be filled during matching
                    full_reference=self._format_full_reference(data),
                    authors=data.get('authors', []),
                    year=data.get('year'),
                    title=data.get('title'),
                    journal=data.get('journal'),
                    doi=data.get('doi'),
                    pages=data.get('pages'),
                    citation_type=data.get('citation_type', 'unknown'),
                    confidence=0.9  # High confidence for LLM extraction
                )
                citations.append(citation)

            return citations

        except Exception as e:
            print(f"LLM citation extraction failed: {e}")
            return []

    def _regex_extract_bibliography(self, ref_section: str) -> List[Citation]:
        """Extract citations using regex patterns"""
        citations = []

        # Split into individual references (usually by newlines or numbers)
        ref_lines = re.split(r'\n\s*(?=\d+\.|\[|\w)', ref_section)

        for ref_line in ref_lines:
            ref_line = ref_line.strip()
            if len(ref_line) < 50:  # Skip short lines
                continue

            for pattern in self.reference_patterns:
                match = re.search(pattern, ref_line)
                if match:
                    groups = match.groups()

                    citation = Citation(
                        inline_text="",
                        full_reference=ref_line,
                        authors=self._parse_authors(groups[0]) if len(groups) > 0 else [],
                        year=groups[1] if len(groups) > 1 else None,
                        title=groups[2] if len(groups) > 2 else None,
                        journal=groups[3] if len(groups) > 3 else None,
                        citation_type=self._classify_citation_type(ref_line),
                        confidence=0.7  # Medium confidence for regex
                    )
                    citations.append(citation)
                    break

        return citations

    def _extract_inline_citations_from_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """Extract inline citations from document chunks"""
        inline_citations = []

        for chunk_idx, chunk in enumerate(chunks):
            content = chunk.page_content

            for pattern in self.inline_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    # Extract context around citation
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]

                    inline_citations.append({
                        'citation_text': match.group(0),
                        'chunk_index': chunk_idx,
                        'chunk_id': chunk.metadata.get('chunk_id', f'chunk_{chunk_idx}'),
                        'position': match.start(),
                        'context': context,
                        'context_before': content[start:match.start()],
                        'context_after': content[match.end():end]
                    })

        return inline_citations

    def _match_inline_to_full_citations(
        self,
        inline_citations: List[Dict],
        full_citations: List[Citation],
        chunks: List[Document]
    ) -> List[CitationMatch]:
        """Match inline citations to their full references"""

        matches = []

        for inline in inline_citations:
            citation_text = inline['citation_text']

            # Extract identifiers from inline citation
            identifiers = self._extract_citation_identifiers(citation_text)

            # Find matching full citation
            best_match = None
            best_score = 0

            for full_citation in full_citations:
                score = self._calculate_match_score(identifiers, full_citation)
                if score > best_score and score > 0.5:  # Minimum confidence threshold
                    best_score = score
                    best_match = full_citation

            if best_match:
                # Update the full citation's inline text
                best_match.inline_text = citation_text

                match = CitationMatch(
                    inline_citation=citation_text,
                    chunk_location=f"Chunk {inline['chunk_index'] + 1}",
                    full_reference=best_match,
                    context_before=inline['context_before'],
                    context_after=inline['context_after']
                )
                matches.append(match)

        return matches

    def _extract_citation_identifiers(self, citation_text: str) -> Dict[str, Any]:
        """Extract identifiers from inline citation for matching"""
        identifiers = {}

        # Author-year pattern
        author_year_match = re.search(r'([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?),?\s+(\d{4})', citation_text)
        if author_year_match:
            identifiers['first_author'] = author_year_match.group(1).replace(' et al', '').strip()
            identifiers['year'] = author_year_match.group(2)

        # Number pattern
        number_match = re.search(r'(\d+)', citation_text)
        if number_match:
            identifiers['number'] = number_match.group(1)

        return identifiers

    def _calculate_match_score(self, identifiers: Dict, citation: Citation) -> float:
        """Calculate matching score between inline citation and full reference"""
        score = 0.0

        # Check first author match
        if 'first_author' in identifiers and citation.authors:
            first_author = citation.authors[0] if citation.authors else ""
            if identifiers['first_author'].lower() in first_author.lower():
                score += 0.6

        # Check year match
        if 'year' in identifiers and citation.year:
            if identifiers['year'] == citation.year:
                score += 0.4

        return score

    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of individual authors"""
        # Remove common suffixes and prefixes
        cleaned = re.sub(r'\s*et\s+al\.?', '', author_string)

        # Split by common separators
        authors = re.split(r',\s*(?=[A-Z])|;\s*|&\s*|\sand\s+', cleaned)

        # Clean each author name
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if len(author) > 2 and re.search(r'[A-Z]', author):
                cleaned_authors.append(author)

        return cleaned_authors

    def _classify_citation_type(self, reference_text: str) -> str:
        """Classify citation type based on content"""
        text_lower = reference_text.lower()

        if any(keyword in text_lower for keyword in ['journal', 'vol.', 'volume', 'issue']):
            return 'journal'
        elif any(keyword in text_lower for keyword in ['proceedings', 'conference', 'symposium']):
            return 'conference'
        elif any(keyword in text_lower for keyword in ['book', 'edition', 'publisher']):
            return 'book'
        elif 'doi:' in text_lower or 'http' in text_lower:
            return 'electronic'
        else:
            return 'unknown'

    def _format_full_reference(self, data: Dict) -> str:
        """Format structured data into full reference string"""
        parts = []

        if data.get('authors'):
            authors = ', '.join(data['authors'])
            parts.append(authors)

        if data.get('year'):
            parts.append(f"({data['year']})")

        if data.get('title'):
            parts.append(f'"{data["title"]}"')

        if data.get('journal'):
            parts.append(f"*{data['journal']}*")

        if data.get('pages'):
            parts.append(f"pp. {data['pages']}")

        return '. '.join(parts)

    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations"""
        seen = set()
        unique_citations = []

        for citation in citations:
            # Create a signature for deduplication
            signature = (
                tuple(citation.authors) if citation.authors else (),
                citation.year,
                citation.title
            )

            if signature not in seen:
                seen.add(signature)
                unique_citations.append(citation)

        return unique_citations

    def enhance_chunks_with_citations(
        self,
        chunks: List[Document],
        citation_matches: List[CitationMatch]
    ) -> List[Document]:
        """Enhance chunks with citation information"""

        enhanced_chunks = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get('chunk_id', '')

            # Find citations in this chunk
            chunk_citations = [
                match for match in citation_matches
                if chunk_id in match.chunk_location or
                any(str(i) in match.chunk_location for i, c in enumerate(chunks) if c == chunk)
            ]

            # Add citation metadata
            chunk.metadata['citations'] = [
                {
                    'inline_text': match.inline_citation,
                    'full_reference': match.full_reference.full_reference,
                    'authors': match.full_reference.authors,
                    'year': match.full_reference.year,
                    'title': match.full_reference.title,
                    'journal': match.full_reference.journal,
                    'doi': match.full_reference.doi
                }
                for match in chunk_citations
            ]

            chunk.metadata['citation_count'] = len(chunk_citations)

            enhanced_chunks.append(chunk)

        return enhanced_chunks