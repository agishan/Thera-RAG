"""
Advanced citation extraction using LLM + regex hybrid approach
"""

import re
import json
import string
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


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
    raw_line: Optional[str] = None  # Original reference line from document
    doi_url: Optional[str] = None  # Normalized DOI or URL
    match_score: float = 0.0  # Calculated match confidence


@dataclass
class SpanCandidate:
    """Represents a candidate references section span"""
    start: int
    end: int
    confidence: float
    header_text: str
    density_score: float = 0.0


@dataclass
class ExtractorConfig:
    """Configuration for citation extraction"""
    use_llm_for_refs: bool = True
    sentence_aware: bool = True
    context_before_chars: int = 100
    context_after_chars: int = 100
    match_threshold: float = 0.6
    title_weight: float = 0.3
    author_weight: float = 0.3
    year_weight: float = 0.2
    doi_weight: float = 0.2


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

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        use_llm: bool = True,
        config: Optional[ExtractorConfig] = None
    ):
        """
        Initialize citation extractor

        Args:
            llm_api_key: Google API key for LLM-based extraction
            use_llm: Whether to use LLM for extraction (falls back to regex only)
            config: Configuration for extraction behavior
        """
        self.config = config or ExtractorConfig()
        self.use_llm = use_llm and llm_api_key is not None

        if self.use_llm:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=llm_api_key
            )
        else:
            self.llm = None

        # JSON schema for LLM output validation
        self.citation_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw_line": {"type": "string"},
                    "parsed": {
                        "type": "object",
                        "properties": {
                            "authors": {"type": "array", "items": {"type": "string"}},
                            "year": {"type": "string"},
                            "title": {"type": "string"},
                            "journal": {"type": "string"},
                            "doi": {"type": "string"},
                            "pages": {"type": "string"},
                            "citation_type": {"type": "string"}
                        },
                        "required": ["authors", "year", "title"]
                    }
                },
                "required": ["raw_line", "parsed"]
            }
        }

        # ULTRATHINK ENHANCED: Comprehensive medical citation patterns
        self.inline_patterns = [
            # MEDICAL FORMAT: Multiple citations with semicolons (PRIMARY PATTERN)
            # Matches: (Andreasen et al , 2011; Armstrong et al , 2011)
            r'\(([^)]+(?:et\s+al\s*[.,]?|[A-Z][a-zA-Z]+)\s*,\s*\d{4}[a-z]*(?:,[a-z])*(?:\s*;\s*[^)]+)*?)\)',

            # MEDICAL FORMAT: Single citation in parentheses with flexible spacing
            # Matches: (Theusinger et al , 2010)
            r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\s*[.,]?)?)\s*,\s*(\d{4}[a-z]*(?:,[a-z]*)?)\)',

            # MEDICAL FORMAT: Multiple authors with &
            # Matches: (Smith & Jones, 2020)
            r'\(([A-Z][a-zA-Z]+(?:\s+&\s+[A-Z][a-zA-Z]+)+)\s*,\s*(\d{4}[a-z]*)\)',

            # ACADEMIC FORMAT: Author outside parentheses
            # Matches: Smith et al (2020)
            r'([A-Z][a-zA-Z]+(?:\s+et\s+al\s*[.,]?)?)\s*\((\d{4}[a-z]*)\)',

            # NUMBERED FORMATS
            r'\[(\d+(?:-\d+)?(?:,\s*\d+)*)\]'

            # SKIP superscript pattern as it creates noise with random numbers
        ]

        self.reference_patterns = [
            # Journal articles
            r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^,]+?),?\s*(\d+(?:\(\d+\))?),?\s*([^.]+)',

            # Books
            r'([A-Z][a-zA-Z\s,.-]+?)\.\s*\((\d{4})\)\.\s*([^.]+?)\.\s*([^:]+):\s*([^.]+)',

            # ULTRATHINK FIXED: Medical journals - capture full author list up to year in parentheses
            r'([A-Z][^(]+?)\s*\((\d{4}[a-z]*)\)\s*([^.]+?)(?:\.\s*(.+?))?'
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

    def find_references_candidates(self, text: str) -> List[SpanCandidate]:
        """
        Find candidate references sections using LLM + heuristics

        Args:
            text: Full document text

        Returns:
            List of SpanCandidate objects ordered by confidence
        """
        candidates = []

        # Stage A: LLM-based discovery (if enabled)
        if self.use_llm and self.config.use_llm_for_refs:
            llm_candidates = self._llm_find_references(text)
            candidates.extend(llm_candidates)

        # Stage B: Heuristic discovery (always run)
        heuristic_candidates = self._heuristic_find_references(text)
        candidates.extend(heuristic_candidates)

        # Deduplicate and sort by confidence
        return self._deduplicate_candidates(candidates)

    def _llm_find_references(self, text: str) -> List[SpanCandidate]:
        """Use LLM to find candidate references sections"""
        # Search in last 30% of document for efficiency
        search_start = int(len(text) * 0.7)
        search_text = text[search_start:]

        prompt = f"""
Find the references/bibliography section in this document excerpt.
Return JSON with candidate spans:

{{
  "candidates": [
    {{
      "start_char": <relative_position_in_excerpt>,
      "end_char": <relative_position_in_excerpt>,
      "confidence": <0.0_to_1.0>,
      "header_text": "<detected_header>"
    }}
  ]
}}

Look for headers like: REFERENCES, Bibliography, Works Cited, Literature Cited
May or may not have markdown # symbols.

Document excerpt (last 30%):
{search_text[:4000]}
"""

        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)

            candidates = []
            for candidate in data.get("candidates", []):
                # Adjust positions to full document
                abs_start = search_start + candidate["start_char"]
                abs_end = search_start + candidate["end_char"]

                candidates.append(SpanCandidate(
                    start=abs_start,
                    end=abs_end,
                    confidence=candidate["confidence"],
                    header_text=candidate["header_text"]
                ))

            return candidates

        except Exception as e:
            print(f"LLM references discovery failed: {e}")
            return []

    def _heuristic_find_references(self, text: str) -> List[SpanCandidate]:
        """Find references using heuristic patterns and density analysis"""
        candidates = []

        # Enhanced header patterns (with and without markdown)
        ref_patterns = [
            r'(?i)^#+\s*(references?)\s*$',
            r'(?i)^#+\s*(bibliography)\s*$',
            r'(?i)^#+\s*(works?\s+cited)\s*$',
            r'(?i)^#+\s*(literature\s+cited)\s*$',
            # Without markdown
            r'(?i)^\s*(references?)\s*$',
            r'(?i)^\s*(bibliography)\s*$',
            r'(?i)^\s*(works?\s+cited)\s*$',
            r'(?i)^\s*(literature\s+cited)\s*$'
        ]

        for pattern in ref_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                start_pos = match.end()

                # Find end using next header or document end
                end_pos = self._find_section_end(text, start_pos)

                if end_pos > start_pos:
                    section_text = text[start_pos:end_pos]
                    density = self._calculate_reference_density(section_text)

                    # Higher confidence for markdown headers
                    base_confidence = 0.8 if '#' in match.group(0) else 0.6

                    candidates.append(SpanCandidate(
                        start=start_pos,
                        end=end_pos,
                        confidence=base_confidence * density,
                        header_text=match.group(1),
                        density_score=density
                    ))

        # Also scan for high-density reference-like sections without headers
        text_lines = text.split('\n')
        for i in range(len(text_lines) - 10):  # Need at least 10 lines
            window = '\n'.join(text_lines[i:i+20])  # 20-line window
            density = self._calculate_reference_density(window)

            if density > 0.7:  # High density threshold
                start_char = len('\n'.join(text_lines[:i]))
                end_char = start_char + len(window)

                candidates.append(SpanCandidate(
                    start=start_char,
                    end=end_char,
                    confidence=density * 0.5,  # Lower confidence without header
                    header_text="(no header detected)",
                    density_score=density
                ))

        return candidates

    def _calculate_reference_density(self, text_block: str) -> float:
        """
        Calculate how reference-like a text block is

        Returns score 0.0-1.0 based on reference indicators
        """
        if not text_block.strip():
            return 0.0

        lines = [line.strip() for line in text_block.split('\n') if line.strip()]
        if len(lines) < 3:
            return 0.0

        score = 0.0
        indicators = 0

        for line in lines:
            line_score = 0.0

            # Check for numbered prefixes
            if re.match(r'^\d+[\.\)]\s+', line):
                line_score += 0.3
            elif re.match(r'^\[\d+\]\s+', line):
                line_score += 0.3

            # Check for year patterns
            if re.search(r'\(\d{4}[a-z]?\)', line):
                line_score += 0.2
            elif re.search(r'\b\d{4}[a-z]?\b', line):
                line_score += 0.1

            # Check for journal/volume indicators
            if re.search(r'\bvol\.?\s*\d+|\bvolume\s*\d+|\bissue\s*\d+', line, re.I):
                line_score += 0.15

            # Check for DOI/URL presence
            if re.search(r'doi:|https?://|www\.', line, re.I):
                line_score += 0.15

            # Check for page numbers
            if re.search(r'\bpp?\.\s*\d+|\b\d+[-–]\d+\b', line):
                line_score += 0.1

            # Check for typical citation structure (Author, Title, Journal)
            if len(line) > 50 and line.count(',') >= 2:
                line_score += 0.1

            if line_score > 0.1:
                indicators += 1
                score += line_score

        # Normalize by number of lines and boost if high indicator density
        density = indicators / len(lines)
        normalized_score = (score / len(lines)) * (1 + density)

        return min(1.0, normalized_score)

    def _find_section_end(self, text: str, start_pos: int) -> int:
        """Find where a section ends based on next header or document end"""
        remaining_text = text[start_pos:]

        # Look for next major section header
        next_header = re.search(r'\n\s*#+\s*[A-Z]|\n\s*[A-Z][A-Z\s]+\s*\n', remaining_text)
        if next_header:
            return start_pos + next_header.start()

        # Look for significant density drop
        lines = remaining_text.split('\n')
        best_end = len(remaining_text)

        # Check density in sliding windows
        for i in range(10, len(lines), 5):
            window = '\n'.join(lines[max(0, i-10):i])
            if self._calculate_reference_density(window) < 0.3:
                char_pos = len('\n'.join(lines[:i]))
                best_end = min(best_end, char_pos)
                break

        return start_pos + best_end

    def select_best_references_span(self, candidates: List[SpanCandidate]) -> Optional[SpanCandidate]:
        """Select the best references section from candidates"""
        if not candidates:
            return None

        # Sort by confidence descending
        sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)

        # Return highest confidence candidate above threshold
        best = sorted_candidates[0]
        if best.confidence > 0.4:  # Minimum confidence threshold
            return best

        return None

    def _deduplicate_candidates(self, candidates: List[SpanCandidate]) -> List[SpanCandidate]:
        """Remove overlapping candidates, keeping highest confidence"""
        if not candidates:
            return []

        # Sort by confidence descending
        sorted_candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        unique_candidates = []

        for candidate in sorted_candidates:
            # Check for significant overlap with existing candidates
            overlaps = False
            for existing in unique_candidates:
                overlap_start = max(candidate.start, existing.start)
                overlap_end = min(candidate.end, existing.end)
                overlap_length = max(0, overlap_end - overlap_start)

                # If more than 50% overlap, consider it duplicate
                min_length = min(candidate.end - candidate.start, existing.end - existing.start)
                if overlap_length > min_length * 0.5:
                    overlaps = True
                    break

            if not overlaps:
                unique_candidates.append(candidate)

        return unique_candidates

    def _find_references_section(self, text: str) -> Optional[str]:
        """Legacy method - now uses robust discovery"""
        candidates = self.find_references_candidates(text)
        best_candidate = self.select_best_references_span(candidates)

        if best_candidate:
            return text[best_candidate.start:best_candidate.end].strip()

        return None

    def _llm_extract_bibliography(self, ref_section: str) -> List[Citation]:
        """Use LLM to extract structured citations with guardrails"""
        return self._llm_parse_references_with_validation(ref_section)

    def _llm_parse_references_with_validation(self, ref_section: str) -> List[Citation]:
        """
        Parse references using LLM with strict JSON validation and fallbacks
        """
        prompt = f"""
Extract all citations from this bibliography section. Return strict JSON format.
For each citation, provide both the raw line and parsed data:

{{
  "citations": [
    {{
      "raw_line": "1. Smith, J. (2023). Title here. Journal Name, 45(2), 123-145.",
      "parsed": {{
        "authors": ["Smith, J."],
        "year": "2023",
        "title": "Title here",
        "journal": "Journal Name",
        "pages": "123-145",
        "citation_type": "journal"
      }}
    }}
  ]
}}

Required fields: authors (array), year (string), title (string)
Optional: journal, doi, pages, citation_type

Bibliography section:
{ref_section[:3000]}
"""

        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)

            # Validate schema if available
            if JSONSCHEMA_AVAILABLE:
                try:
                    jsonschema.validate(data.get("citations", []), self.citation_schema)
                except jsonschema.ValidationError as e:
                    print(f"LLM output validation failed: {e}")
                    # Fall back to regex parsing
                    return self._regex_extract_bibliography(ref_section)

            citations = []
            for item in data.get("citations", []):
                raw_line = item.get("raw_line", "")
                parsed = item.get("parsed", {})

                # Check required fields
                if not all(k in parsed for k in ["authors", "year", "title"]):
                    print(f"Missing required fields in citation: {parsed}")
                    # Attempt regex fallback on raw line
                    fallback_citation = self._fallback_regex_parse(raw_line)
                    if fallback_citation:
                        citations.append(fallback_citation)
                    continue

                citation = Citation(
                    inline_text="",
                    full_reference=raw_line,
                    raw_line=raw_line,
                    authors=parsed.get('authors', []),
                    year=parsed.get('year'),
                    title=parsed.get('title'),
                    journal=parsed.get('journal'),
                    doi=parsed.get('doi'),
                    pages=parsed.get('pages'),
                    citation_type=parsed.get('citation_type', 'unknown'),
                    confidence=0.9
                )

                # Post-process for DOI/URL extraction
                citation = self.fill_missing_doi_and_url(citation)
                citations.append(citation)

            return citations

        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM JSON parsing failed: {e}. Falling back to regex.")
            return self._regex_extract_bibliography(ref_section)
        except Exception as e:
            print(f"LLM citation extraction failed: {e}")
            return []

    def _validate_citation_json(self, json_data: dict) -> bool:
        """Validate citation JSON against schema"""
        if not JSONSCHEMA_AVAILABLE:
            # Basic validation without jsonschema
            required_fields = ["authors", "year", "title"]
            return all(field in json_data for field in required_fields)

        try:
            jsonschema.validate(json_data, self.citation_schema["items"]["properties"]["parsed"])
            return True
        except jsonschema.ValidationError:
            return False

    def _fallback_regex_parse(self, raw_line: str) -> Optional[Citation]:
        """Fallback regex parsing for failed LLM extractions"""
        if not raw_line or len(raw_line) < 20:
            return None

        for pattern in self.reference_patterns:
            match = re.search(pattern, raw_line)
            if match:
                groups = match.groups()
                return Citation(
                    inline_text="",
                    full_reference=raw_line,
                    raw_line=raw_line,
                    authors=self._parse_authors(groups[0]) if len(groups) > 0 else [],
                    year=groups[1] if len(groups) > 1 else None,
                    title=groups[2] if len(groups) > 2 else None,
                    journal=groups[3] if len(groups) > 3 else None,
                    citation_type=self._classify_citation_type(raw_line),
                    confidence=0.5  # Lower confidence for fallback
                )

        return None

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
        """ULTRATHINK ENHANCED: Extract and parse inline citations from document chunks"""
        inline_citations = []
        print(f"DEBUG: Processing {len(chunks)} chunks for inline citations")

        for chunk_idx, chunk in enumerate(chunks):
            content = chunk.page_content
            chunk_id = chunk.metadata.get('chunk_id', f'chunk_{chunk_idx}')

            # DEBUG: Check chunk 5 specifically
            if chunk_id == 'chunk_005':
                print(f"DEBUG: Processing chunk_005, content length: {len(content)}")
                print(f"DEBUG: First 200 chars: {content[:200]}")

            found_in_chunk = 0
            for pattern_idx, pattern in enumerate(self.inline_patterns):
                matches = list(re.finditer(pattern, content))
                if matches and chunk_id == 'chunk_005':
                    print(f"DEBUG: Pattern {pattern_idx+1} found {len(matches)} matches in chunk_005")

                for match in matches:
                    found_in_chunk += 1
                    # Extract context around citation
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]

                    full_citation_text = match.group(0)

                    if chunk_id == 'chunk_005':
                        print(f"DEBUG: Found citation in chunk_005: '{full_citation_text}'")

                    # ULTRATHINK: Parse complex citations with semicolons
                    individual_citations = self._parse_complex_citation(full_citation_text)

                    if chunk_id == 'chunk_005':
                        print(f"DEBUG: Parsed into {len(individual_citations)} individual citations")

                    # Create entry for each individual citation
                    for individual_citation in individual_citations:
                        inline_citations.append({
                            'citation_text': individual_citation['citation_text'],
                            'authors': individual_citation['authors'],
                            'year': individual_citation['year'],
                            'chunk_index': chunk_idx,
                            'chunk_id': chunk_id,
                            'position': match.start(),
                            'context': context,
                            'context_before': content[start:match.start()],
                            'context_after': content[match.end():end],
                            'full_parenthetical': full_citation_text  # Keep original for reference
                        })

            if chunk_id == 'chunk_005':
                print(f"DEBUG: Total citations found in chunk_005: {found_in_chunk}")

        print(f"DEBUG: Total inline citations extracted: {len(inline_citations)}")
        return inline_citations

    def _parse_complex_citation(self, citation_text: str) -> List[Dict[str, str]]:
        """Parse complex citation strings into individual author-year pairs"""
        citations = []

        # Remove outer parentheses if present
        inner_text = citation_text.strip('()')

        # Split on semicolons for multiple citations
        citation_parts = [part.strip() for part in inner_text.split(';')]

        for part in citation_parts:
            # Pattern to extract author(s) and year from each part
            # Handles: "Andreasen et al , 2011", "Smith & Jones, 2020", "Ogawa et al , 2012a,b"
            author_year_pattern = r'([^,]+?)\s*,\s*(\d{4}[a-z]*(?:,[a-z]*)*)'
            match = re.search(author_year_pattern, part)

            if match:
                authors = match.group(1).strip()
                year = match.group(2).strip()

                # Handle multiple year suffixes (2012a,b -> 2012a and 2012b)
                if ',' in year and not year.startswith(('19', '20')):
                    base_year = re.search(r'(\d{4})', year).group(1)
                    suffixes = re.findall(r'[a-z]', year)

                    for suffix in suffixes:
                        citations.append({
                            'citation_text': f"({authors}, {base_year}{suffix})",
                            'authors': authors,
                            'year': f"{base_year}{suffix}"
                        })
                else:
                    citations.append({
                        'citation_text': f"({authors}, {year})",
                        'authors': authors,
                        'year': year
                    })
            else:
                # Fallback: treat as single citation if parsing fails
                citations.append({
                    'citation_text': f"({part})",
                    'authors': part,
                    'year': ""
                })

        return citations if citations else [{'citation_text': citation_text, 'authors': citation_text, 'year': ""}]

    def _match_inline_to_full_citations(
        self,
        inline_citations: List[Dict],
        full_citations: List[Citation],
        chunks: List[Document]
    ) -> List[CitationMatch]:
        """Match inline citations to their full references"""

        matches = []
        print(f"DEBUG: Matching {len(inline_citations)} inline citations to {len(full_citations)} full citations")

        # DEBUG: Show first few inline citations
        for i, inline in enumerate(inline_citations[:5]):
            print(f"DEBUG: Inline {i+1}: '{inline['citation_text']}' (authors: '{inline['authors']}', year: '{inline['year']}')")

        # DEBUG: Show first few full citations
        for i, full in enumerate(full_citations[:3]):
            print(f"DEBUG: Full {i+1}: authors={full.authors}, year={full.year}")

        for inline in inline_citations:
            citation_text = inline['citation_text']

            # Extract identifiers from inline citation
            identifiers = self._extract_citation_identifiers(citation_text)

            if inline.get('chunk_id') == 'chunk_005':
                print(f"DEBUG: Processing inline from chunk_005: '{citation_text}'")
                print(f"DEBUG: Extracted identifiers: {identifiers}")

            # Find matching full citation
            best_match = None
            best_score = 0

            for full_citation in full_citations:
                score = self._calculate_match_score(identifiers, full_citation)

                if inline.get('chunk_id') == 'chunk_005' and score > 0:
                    print(f"DEBUG: Match score {score:.3f} for '{citation_text}' vs full citation authors={full_citation.authors}")

                if score > best_score and score >= 0.5:  # ULTRATHINK: Include exact 0.5 matches
                    best_score = score
                    best_match = full_citation

            if best_match:
                if inline.get('chunk_id') == 'chunk_005':
                    print(f"DEBUG: MATCH FOUND for '{citation_text}' with score {best_score:.3f}")

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

    def _calculate_match_score(self, identifiers: Dict, citation: Citation, context: str = "") -> float:
        """Enhanced match scoring with configurable weights"""
        score = 0.0

        # DOI exact match (highest priority)
        if 'doi' in identifiers and citation.doi:
            if identifiers['doi'].lower() == citation.doi.lower():
                score += self.config.doi_weight * 2  # Double weight for exact DOI match

        # Author matching
        if 'authors' in identifiers and citation.authors:
            author_score = self._author_similarity(identifiers['authors'], citation.authors)
            score += author_score * self.config.author_weight

        # Year matching
        if 'year' in identifiers and citation.year:
            if identifiers['year'] == citation.year:
                score += self.config.year_weight

        # Title similarity (if available)
        if 'title' in identifiers and citation.title:
            title_score = self.title_similarity(identifiers['title'], citation.title)
            score += title_score * self.config.title_weight

        # Fallback scoring for existing logic
        if 'first_author' in identifiers and citation.authors:
            first_author = citation.authors[0] if citation.authors else ""
            if identifiers['first_author'].lower() in first_author.lower():
                score += 0.3  # Legacy compatibility

        return min(1.0, score)

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

    # ===== NEW ENHANCED METHODS =====

    def parse_numbered_references(self, ref_text: str) -> Tuple[Dict[int, Citation], List[Citation]]:
        """
        Parse numbered references section into ordered map and citation list

        Args:
            ref_text: References section text

        Returns:
            Tuple of (number -> citation map, all citations list)
        """
        index_map = {}
        citations = []

        # Split into lines and look for numbered entries
        lines = ref_text.split('\n')
        current_citation = ""
        current_number = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for number prefix
            number = self._extract_reference_number(line)

            if number is not None:
                # Save previous citation if exists
                if current_citation and current_number is not None:
                    citation = self._parse_single_reference(current_citation, current_number)
                    if citation:
                        citations.append(citation)
                        index_map[current_number] = citation

                # Start new citation
                current_number = number
                current_citation = line
            else:
                # Continue previous citation
                if current_citation:
                    current_citation += " " + line

        # Handle last citation
        if current_citation and current_number is not None:
            citation = self._parse_single_reference(current_citation, current_number)
            if citation:
                citations.append(citation)
                index_map[current_number] = citation

        return index_map, citations

    def _extract_reference_number(self, line: str) -> Optional[int]:
        """Extract reference number from line start"""
        # Pattern 1: "1. Author..."
        match = re.match(r'^(\d+)\.\s+', line)
        if match:
            return int(match.group(1))

        # Pattern 2: "1) Author..."
        match = re.match(r'^(\d+)\)\s+', line)
        if match:
            return int(match.group(1))

        # Pattern 3: "[1] Author..."
        match = re.match(r'^\[(\d+)\]\s+', line)
        if match:
            return int(match.group(1))

        return None

    def _parse_single_reference(self, ref_text: str, ref_number: int) -> Optional[Citation]:
        """Parse a single numbered reference"""
        # Remove number prefix
        clean_text = re.sub(r'^\d+[\.\)]\s+|\^\[\d+\]\s+', '', ref_text)

        # Try LLM parsing first
        if self.use_llm:
            llm_citation = self._llm_parse_single_reference(clean_text)
            if llm_citation:
                llm_citation.raw_line = ref_text
                return llm_citation

        # Fallback to regex
        fallback = self._fallback_regex_parse(clean_text)
        if fallback:
            fallback.raw_line = ref_text

        return fallback

    def _llm_parse_single_reference(self, ref_text: str) -> Optional[Citation]:
        """Parse single reference using LLM"""
        prompt = f"""
Parse this single reference and return JSON:
{{
  "authors": ["Author1", "Author2"],
  "year": "2023",
  "title": "Title here",
  "journal": "Journal Name",
  "doi": "10.xxxx/yyyy",
  "pages": "123-145",
  "citation_type": "journal"
}}

Reference: {ref_text[:500]}
"""
        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)

            if self._validate_citation_json(data):
                return Citation(
                    inline_text="",
                    full_reference=ref_text,
                    raw_line=ref_text,
                    authors=data.get('authors', []),
                    year=data.get('year'),
                    title=data.get('title'),
                    journal=data.get('journal'),
                    doi=data.get('doi'),
                    pages=data.get('pages'),
                    citation_type=data.get('citation_type', 'unknown'),
                    confidence=0.8
                )
        except Exception:
            pass

        return None

    def _resolve_numeric_citations(self, inline_text: str, index_map: Dict[int, Citation]) -> List[Citation]:
        """Resolve numeric inline citations to full citations"""
        citations = []

        # Extract numbers from inline citation
        numbers = self._parse_citation_ranges(inline_text)

        for num in numbers:
            if num in index_map:
                citations.append(index_map[num])
            else:
                print(f"Warning: Numeric citation {num} not found in references")

        return citations

    def _parse_citation_ranges(self, range_text: str) -> List[int]:
        """Parse citation ranges and lists into individual numbers"""
        numbers = []

        # Remove brackets and parentheses
        clean_text = re.sub(r'[\[\]()]', '', range_text)

        # Split by commas
        parts = [p.strip() for p in clean_text.split(',')]

        for part in parts:
            # Check for ranges like "3-5" or "3–5"
            range_match = re.match(r'^(\d+)[-–](\d+)$', part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                numbers.extend(range(start, end + 1))
            else:
                # Single number
                num_match = re.match(r'^(\d+)$', part)
                if num_match:
                    numbers.append(int(num_match.group(1)))

        return numbers

    def fill_missing_doi_and_url(self, citation: Citation) -> Citation:
        """Extract and normalize DOI/URL from citation text"""
        # Check both full_reference and raw_line
        texts_to_check = [citation.full_reference]
        if citation.raw_line:
            texts_to_check.append(citation.raw_line)

        for text in texts_to_check:
            if not text:
                continue

            # Extract DOI first (prefer over URL)
            doi = self._extract_doi_from_text(text)
            if doi:
                citation.doi = doi
                citation.doi_url = self._normalize_doi(doi)
                return citation

            # Extract URL if no DOI
            url = self._extract_url_from_text(text)
            if url:
                citation.doi_url = url

        return citation

    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract DOI from text"""
        # Pattern 1: doi:10.xxxx/yyyy
        match = re.search(r'doi:\s*(10\.\d+/.+?)(?:\s|$|[,\.\)])', text, re.I)
        if match:
            return match.group(1).rstrip('.,)')

        # Pattern 2: https://doi.org/10.xxxx/yyyy
        match = re.search(r'https?://doi\.org/(10\.\d+/.+?)(?:\s|$|[,\.\)])', text, re.I)
        if match:
            return match.group(1).rstrip('.,)')

        return None

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """Extract URL from text"""
        # Look for http/https URLs
        match = re.search(r'(https?://[^\s,\.\)]+)', text, re.I)
        if match:
            return match.group(1).rstrip('.,)')

        return None

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI to standard URL format"""
        # Remove doi: prefix if present
        clean_doi = re.sub(r'^doi:\s*', '', doi, flags=re.I)
        return f"https://doi.org/{clean_doi}"

    def title_similarity(self, title_a: str, title_b: str) -> float:
        """Calculate title similarity using rapidfuzz if available, else basic comparison"""
        if not title_a or not title_b:
            return 0.0

        # Normalize titles
        norm_a = self.normalize_title(title_a)
        norm_b = self.normalize_title(title_b)

        if RAPIDFUZZ_AVAILABLE:
            # Use rapidfuzz for better similarity
            return fuzz.ratio(norm_a, norm_b) / 100.0
        else:
            # Basic token-based similarity
            tokens_a = set(norm_a.split())
            tokens_b = set(norm_b.split())

            if not tokens_a or not tokens_b:
                return 0.0

            intersection = tokens_a.intersection(tokens_b)
            union = tokens_a.union(tokens_b)

            return len(intersection) / len(union) if union else 0.0

    def normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ""

        # Convert to lowercase
        normalized = title.lower()

        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', normalized)

        # Remove punctuation except hyphens
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)

        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in normalized.split() if w not in stopwords and len(w) > 1]

        return ' '.join(words)

    def _author_similarity(self, inline_authors: List[str], ref_authors: List[str]) -> float:
        """Calculate author similarity score"""
        if not inline_authors or not ref_authors:
            return 0.0

        # Normalize author names
        norm_inline = [self._normalize_author_name(a) for a in inline_authors]
        norm_ref = [self._normalize_author_name(a) for a in ref_authors]

        matches = 0
        for inline_author in norm_inline:
            for ref_author in norm_ref:
                if self._author_names_match(inline_author, ref_author):
                    matches += 1
                    break

        return matches / max(len(norm_inline), len(norm_ref))

    def _normalize_author_name(self, name: str) -> str:
        """Normalize author name for matching"""
        # Remove common suffixes
        name = re.sub(r'\s+(Jr\.?|Sr\.?|III?|PhD|MD)$', '', name, flags=re.I)

        # Extract last name (typically the first part before comma, or last word)
        if ',' in name:
            # "Smith, John" format
            return name.split(',')[0].strip().lower()
        else:
            # "John Smith" format - take last word
            return name.split()[-1].lower() if name.split() else name.lower()

    def _author_names_match(self, name1: str, name2: str) -> bool:
        """Check if two normalized author names match"""
        # Exact match
        if name1 == name2:
            return True

        # One is substring of other (for "Smith" vs "Smithson")
        if len(name1) >= 3 and len(name2) >= 3:
            return name1 in name2 or name2 in name1

        return False

    def _get_sentence_context(self, text: str, match_pos: int) -> Tuple[str, str]:
        """Get sentence-aware context around citation"""
        if not self.config.sentence_aware:
            return self._get_char_context(text, match_pos)

        try:
            sentences = self._simple_sentence_split(text)

            # Find which sentence contains the match
            char_count = 0
            target_sentence = -1

            for i, sentence in enumerate(sentences):
                if char_count <= match_pos < char_count + len(sentence):
                    target_sentence = i
                    break
                char_count += len(sentence) + 1  # +1 for space/newline

            if target_sentence == -1:
                return self._get_char_context(text, match_pos)

            # Get context sentences
            start_sentence = max(0, target_sentence - 1)
            end_sentence = min(len(sentences), target_sentence + 2)

            before_sentences = sentences[start_sentence:target_sentence]
            after_sentences = sentences[target_sentence + 1:end_sentence]

            context_before = ' '.join(before_sentences)
            context_after = ' '.join(after_sentences)

            return context_before, context_after

        except Exception:
            # Fallback to character-based context
            return self._get_char_context(text, match_pos)

    def _get_char_context(self, text: str, match_pos: int) -> Tuple[str, str]:
        """Get character-based context around citation"""
        start = max(0, match_pos - self.config.context_before_chars)
        end = min(len(text), match_pos + self.config.context_after_chars)

        context_before = text[start:match_pos]
        context_after = text[match_pos:end]

        return context_before, context_after

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple rule-based sentence splitting"""
        # Split on common sentence endings
        sentences = re.split(r'[.!?]+\s+', text)

        # Clean up empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences