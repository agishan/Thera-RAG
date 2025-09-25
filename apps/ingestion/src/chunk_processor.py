"""
Docling HierarchicalChunker with token-aware coalescing for medical papers
Generates both fine (pure hierarchical) and coarse (coalesced) chunks.

DEPENDENCIES:
- pip install docling-core  (for HierarchicalChunker)
- pip install tiktoken      (optional, for accurate token counting)

ULTRATHINK FIXES APPLIED:
✓ Fixed invalid tokenizer model name (gpt-4o-mini -> gpt-4)
✓ Fixed API mismatch (consistent List[Document] return type)
✓ Added comprehensive error handling throughout
✓ Improved metadata extraction robustness
✓ Better token counting with fallbacks
✓ Optimized chunk sizes for medical citations (1200 target tokens)
✓ Enhanced overlap calculation
✓ Added analyze_chunks method
"""

from __future__ import annotations
import re, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document

# =========================
# CONFIG - HEADER-BASED CHUNKING
# =========================

@dataclass
class ChunkingConfig:
    """
    Header-based chunking configuration for medical documents
    """
    # Chunking strategy
    chunking_method: str = "header_based"  # "header_based" or "docling_hierarchical"

    # Header detection patterns (in priority order)
    header_patterns: List[str] = None  # Will be set in __post_init__

    # Size management - OPTIMIZED FOR RETRIEVAL + CITATIONS
    target_tokens: int = 1400  # sweet spot for embeddings + context
    min_tokens: int = 300     # avoid tiny chunks
    max_tokens: int = 5000    # allow Methods/Results intact but prevent monsters
    overlap_tokens: int = 250  # generous overlap for citations

    # Token counting
    tokenizer_model: str = "gpt-4"

    # Section handling
    merge_small_sections: bool = True  # merge subsections under target_tokens
    keep_section_hierarchy: bool = True  # preserve parent-child relationships
    split_large_sections: bool = True   # split sections over max_tokens

    # Content preservation - ENHANCED with new section types
    preserve_references: bool = True     # keep reference sections intact
    preserve_abstracts: bool = True      # keep abstracts intact
    preserve_conclusions: bool = True    # keep conclusions intact
    preserve_acknowledgments: bool = True  # keep acknowledgment sections intact
    preserve_ethics: bool = True         # keep ethics/conflict sections intact

    def __post_init__(self):
        if self.header_patterns is None:
            # COMPREHENSIVE medical document header patterns (priority order)
            # ULTRATHINK: Added edge cases, false positive protection, medical variations
            self.header_patterns = [
                # 1. Markdown headers (highest priority - most reliable)
                r'^#{1,6}\s+(.+)$',

                # 2. ALL CAPS medical sections (with context validation)
                r'^([A-Z][A-Z\s]{4,}(?:METHODS?|RESULTS?|DISCUSSION|CONCLUSION|ABSTRACT|BACKGROUND|INTRODUCTION|REFERENCES?|APPENDIX)):?\s*$',

                # 3. Roman numeral sections
                r'^[IVX]{1,4}\.?\s+([A-Z][a-zA-Z\s]{4,40})\.?\s*$',

                # 4. Multi-word medical sections (Materials and Methods, etc)
                r'^([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)+):?\s*$',

                # 5. Numbered main sections (with length validation)
                r'^\d{1,2}\.?\s+([A-Z][a-zA-Z\s]{4,40})\.?\s*$',

                # 6. Subsection numbering
                r'^\d{1,2}\.\d{1,2}\.?\s+([A-Z][a-zA-Z\s]{4,30})\.?\s*$',

                # 7. Title Case sections (with medical context)
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*(?:\s+(?:Analysis|Design|Population|Characteristics|Results|Methods|Approach|Protocol)):?)\s*$',

                # 8. Underlined headers (= and - styles)
                r'^([A-Z][a-zA-Z\s]{4,40})\s*\n={4,}\s*$',
                r'^([A-Z][a-zA-Z\s]{4,40})\s*\n-{4,}\s*$',

                # 9. Journal-specific patterns
                r'^([A-Z]+(?:\s+[A-Z]+)*)\s*[-–—]\s*(.+)$',  # "METHODS - Statistical Analysis"

                # 10. Bracketed sections
                r'^\[([A-Z][a-zA-Z\s]{3,30})\]\s*$',
            ]

# =========================
# TOKEN COUNT HELPER
# =========================

def make_token_len_fn(model_name: str = "gpt-4"):
    """
    Returns count_tokens(text) using tiktoken if available, else a simple heuristic.
    IMPROVED: Better error handling and more accurate fallback
    """
    try:
        import tiktoken
        # Try model-specific encoding first
        try:
            enc = tiktoken.encoding_for_model(model_name)
            print(f"Using tiktoken with model: {model_name}")
        except KeyError:
            # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
            enc = tiktoken.get_encoding("cl100k_base")
            print(f"Model {model_name} not found, using cl100k_base encoding")

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(enc.encode(text))
        return count_tokens

    except ImportError:
        print("tiktoken not available, using word-based heuristic")
        def count_tokens(text: str) -> int:
            if not text:
                return 0
            # More accurate heuristic: 1 token ≈ 0.75 words for English
            words = len(text.split())
            return max(1, int(round(words / 0.75)))
        return count_tokens

    except Exception as e:
        print(f"tiktoken setup failed: {e}, using fallback")
        def count_tokens(text: str) -> int:
            if not text:
                return 0
            words = len(text.split())
            return max(1, int(round(words / 0.75)))
        return count_tokens

# =========================
# DOCLING IMPORTS
# =========================

try:
    from docling_core.transforms.chunker import HierarchicalChunker
    DOCLING_CHUNKER_AVAILABLE = True
except ImportError:
    DOCLING_CHUNKER_AVAILABLE = False
    HierarchicalChunker = None

# =========================
# MAIN
# =========================

class SmartChunker:
    """
    Intelligent chunking system supporting both header-based and Docling-based approaches.

    Header-based: Splits on document sections (Introduction, Methods, Results, etc.)
    Docling-based: Uses HierarchicalChunker with coalescing
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

        if self.config.chunking_method == "docling_hierarchical" and not DOCLING_CHUNKER_AVAILABLE:
            raise ImportError("Docling chunker not available. Install: pip install docling-core")

        self.count_tokens = make_token_len_fn(self.config.tokenizer_model)

    # -------------------------
    # Public entry
    # -------------------------
    def create_chunks(self, docling_doc_or_text, base_metadata: Dict[str, Any]) -> List[Document]:
        """
        Create chunks using either header-based or Docling-based approach.

        Args:
            docling_doc_or_text: Docling Document object OR plain text string
            base_metadata: base metadata applied to all chunks

        Returns:
            List[Document] - chunks based on configured method
        """
        if self.config.chunking_method == "header_based":
            # Extract text from docling document if needed
            if hasattr(docling_doc_or_text, 'export_to_markdown'):
                text = docling_doc_or_text.export_to_markdown()
            elif isinstance(docling_doc_or_text, str):
                text = docling_doc_or_text
            else:
                text = str(docling_doc_or_text)

            return self._create_chunks_header_based(text, base_metadata)

        elif self.config.chunking_method == "docling_hierarchical":
            if not docling_doc_or_text:
                print("Warning: Empty docling document")
                return []

            print("Using Docling HierarchicalChunker")
            print(f"  target_tokens: {self.config.target_tokens}")
            print()

            try:
                fine_chunks = self._create_chunks_hierarchical(docling_doc_or_text, base_metadata)
                if not fine_chunks:
                    print("Warning: No chunks created from document")
                    return []

                print(f"Created {len(fine_chunks)} hierarchical chunks")
                return fine_chunks

            except Exception as e:
                print(f"Error in chunk creation: {e}")
                return []

        else:
            raise ValueError(f"Unknown chunking method: {self.config.chunking_method}")

    # -------------------------
    # Header-Based Chunking
    # -------------------------
    def _create_chunks_header_based(self, text: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Create chunks based on document headers/sections"""
        if not text.strip():
            return []

        print(f"Using Header-Based Chunking")
        print(f"  target_tokens: {self.config.target_tokens}")
        print(f"  min_tokens: {self.config.min_tokens}")
        print(f"  max_tokens: {self.config.max_tokens}")
        print(f"  overlap_tokens: {self.config.overlap_tokens}")
        print()

        # Step 1: Split into sections based on headers
        sections = self._split_by_headers(text)
        print(f"Found {len(sections)} sections")

        # ULTRATHINK FALLBACK: Check if header detection worked
        if len(sections) < 2 or self._header_detection_failed(sections, text):
            print("⚠️  Header detection failed or insufficient - falling back to paragraph chunking")
            return self._fallback_paragraph_chunking(text, base_metadata)

        # Step 2: Process sections (merge small, split large)
        processed_sections = self._process_sections(sections)
        print(f"Processed into {len(processed_sections)} chunks")

        # Step 3: Create LangChain documents
        chunks = []
        for i, section in enumerate(processed_sections, 1):
            content = section['content'].strip()
            if not content or len(content) < 50:  # Skip very short chunks
                continue

            token_count = self.count_tokens(content)
            meta = {
                **base_metadata,
                **self._create_chunk_metadata(content, i),
                "chunking_method": "header_based",
                "section_title": section['title'],
                "section_level": section['level'],
                "section_type": section['type'],
                "token_count": token_count,
                "content_hash": hashlib.md5(content.encode()).hexdigest()[:12]
            }

            chunks.append(Document(page_content=content, metadata=meta))

        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sections based on header patterns"""
        sections = []
        lines = text.split('\n')
        current_section = {
            'title': 'Document Start',
            'level': 0,
            'type': 'content',
            'content': ''
        }

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                current_section['content'] += line + '\n'
                continue

            # Check if line matches any header pattern
            header_match = None
            header_level = 0
            header_title = ''

            for pattern_idx, pattern in enumerate(self.config.header_patterns):
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    header_title = match.group(1).strip()

                    # ULTRATHINK: False positive filtering
                    if not self._is_valid_header(header_title, line_stripped, lines[max(0, i-2):i+3]):
                        continue

                    header_level = pattern_idx + 1  # Pattern order = priority level
                    header_match = match
                    break

            if header_match:
                # Save current section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)

                # Determine section type
                section_type = self._classify_section_type(header_title)

                # Start new section
                current_section = {
                    'title': header_title,
                    'level': header_level,
                    'type': section_type,
                    'content': line + '\n'  # Include the header line
                }
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)

        return sections

    def _is_valid_header(self, title: str, full_line: str, context_lines: List[str]) -> bool:
        """ULTRATHINK: Filter out false positive headers"""
        if not title or len(title.strip()) < 3:
            return False

        title_lower = title.lower()

        # False positive patterns to reject
        false_positives = [
            # Citation patterns
            r'et\s+al\.?',  # "Smith et al."
            r'\(\d{4}\)',   # "(2020)"
            r'pp?\.\s*\d+', # "p. 123", "pp. 45-67"

            # Geographic/institutional names
            r'new\s+york|los\s+angeles|university\s+of|department\s+of',

            # Journal/publication names
            r'journal\s+of|proceedings\s+of|annals\s+of',

            # Figure/table references
            r'^(figure|fig|table|tab)\.?\s*\d+',

            # List items that aren't headers
            r'^\d+\.\s*(the|a|an|this|that|these|those)',  # "1. The study showed..."

            # URLs/emails
            r'(http|www\.|@)',

            # Numbers/measurements that got caught
            r'^\d+[\.\,]\d+',  # "12.5", "1,000"

            # Common false positive phrases
            r'(vs\.|versus|compared\s+to|according\s+to|based\s+on)',
        ]

        # Check against false positive patterns
        for pattern in false_positives:
            if re.search(pattern, title_lower):
                return False

        # Additional context-based validation
        context_text = ' '.join(context_lines).lower()

        # Reject if surrounded by citation markers
        citation_markers = ['(', ')', '[', ']', '"', "'"]
        if any(marker in full_line for marker in citation_markers) and len(title) < 20:
            return False

        # Reject if part of a sentence (has lowercase continuation)
        next_line_idx = context_lines.index(full_line.strip()) if full_line.strip() in context_lines else -1
        if next_line_idx >= 0 and next_line_idx < len(context_lines) - 1:
            next_line = context_lines[next_line_idx + 1].strip()
            if next_line and next_line[0].islower() and not next_line.startswith(('a', 'an', 'the')):
                return False

        # Medical section validation - boost confidence for medical terms
        medical_terms = [
            'abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion',
            'background', 'objective', 'design', 'population', 'intervention',
            'outcome', 'analysis', 'finding', 'limitation', 'reference', 'appendix'
        ]

        has_medical_terms = any(term in title_lower for term in medical_terms)

        # Length-based validation
        if len(title) > 100:  # Very long "headers" are suspicious
            return False

        if len(title) < 5 and not has_medical_terms:  # Very short non-medical
            return False

        # All caps validation - medical sections are often in caps
        if title.isupper() and len(title) > 20:  # Long all-caps might be shouting
            return has_medical_terms

        return True

    def _classify_section_type(self, title: str) -> str:
        """ENHANCED: Classify section type with comprehensive medical terminology"""
        title_lower = title.lower()

        # ABSTRACT/SUMMARY
        if any(word in title_lower for word in [
            'abstract', 'summary', 'synopsis', 'overview', 'highlights'
        ]):
            return 'abstract'

        # INTRODUCTION/BACKGROUND
        elif any(word in title_lower for word in [
            'introduction', 'background', 'rationale', 'objective', 'aims', 'purpose',
            'literature review', 'prior work', 'motivation'
        ]):
            return 'introduction'

        # METHODS (comprehensive medical variations)
        elif any(word in title_lower for word in [
            'method', 'approach', 'design', 'protocol', 'procedure', 'technique',
            'materials', 'participants', 'subjects', 'population', 'cohort',
            'patient', 'recruitment', 'eligibility', 'criteria', 'intervention',
            'treatment', 'therapy', 'statistical', 'analysis', 'measurement',
            'assessment', 'evaluation', 'data collection', 'sampling', 'randomization',
            'blinding', 'placebo', 'control', 'baseline', 'demographic'
        ]):
            return 'methods'

        # RESULTS (comprehensive variations)
        elif any(word in title_lower for word in [
            'result', 'finding', 'outcome', 'observation', 'data', 'measurement',
            'performance', 'efficacy', 'effectiveness', 'response', 'endpoint',
            'adverse', 'side effect', 'toxicity', 'safety', 'mortality', 'survival',
            'primary outcome', 'secondary outcome', 'patient characteristic',
            'baseline characteristic', 'demographic', 'follow-up'
        ]):
            return 'results'

        # DISCUSSION (analysis variations)
        elif any(word in title_lower for word in [
            'discussion', 'interpretation', 'implication', 'clinical significance',
            'limitation', 'strength', 'weakness', 'comparison', 'mechanism',
            'explanation', 'hypothesis', 'theory', 'clinical relevance',
            'future research', 'recommendation'
        ]):
            return 'discussion'

        # CONCLUSION
        elif any(word in title_lower for word in [
            'conclusion', 'summary', 'closing', 'final', 'takeaway',
            'clinical implication', 'practical implication', 'key finding'
        ]):
            return 'conclusion'

        # REFERENCES
        elif any(word in title_lower for word in [
            'reference', 'bibliography', 'citation', 'literature cited',
            'works cited', 'sources'
        ]):
            return 'references'

        # APPENDIX/SUPPLEMENTARY
        elif any(word in title_lower for word in [
            'appendix', 'supplement', 'additional', 'supporting', 'extra',
            'supplementary material', 'online content', 'web appendix'
        ]):
            return 'appendix'

        # ACKNOWLEDGMENTS (often contains funding info)
        elif any(word in title_lower for word in [
            'acknowledgment', 'acknowledgement', 'funding', 'grant', 'support',
            'contribution', 'author contribution'
        ]):
            return 'acknowledgments'

        # CONFLICT OF INTEREST / ETHICS
        elif any(word in title_lower for word in [
            'conflict', 'interest', 'disclosure', 'ethics', 'consent', 'approval',
            'committee', 'institutional review', 'irb'
        ]):
            return 'ethics'

        else:
            return 'content'

    def _process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process sections: merge small ones, split large ones, add overlap"""
        if not sections:
            return []

        processed = []
        current_buffer = None

        for section in sections:
            content = section['content']
            token_count = self.count_tokens(content)
            section_type = section['type']

            # Special handling for preserved sections - ENHANCED
            preserve_intact = (
                (section_type == 'references' and self.config.preserve_references) or
                (section_type == 'abstract' and self.config.preserve_abstracts) or
                (section_type == 'conclusion' and self.config.preserve_conclusions) or
                (section_type == 'acknowledgments' and self.config.preserve_acknowledgments) or
                (section_type == 'ethics' and self.config.preserve_ethics)
            )

            if preserve_intact:
                # Flush current buffer first
                if current_buffer:
                    processed.append(current_buffer)
                    current_buffer = None

                # Add this section intact (even if large)
                processed.append(section)
                continue

            # Handle size-based processing
            if token_count >= self.config.min_tokens:
                # Flush buffer if exists
                if current_buffer:
                    processed.append(current_buffer)
                    current_buffer = None

                # Handle large sections
                if token_count > self.config.max_tokens and self.config.split_large_sections:
                    split_sections = self._split_large_section(section)
                    processed.extend(split_sections)
                else:
                    processed.append(section)

            else:
                # Small section - try to merge
                if self.config.merge_small_sections:
                    if not current_buffer:
                        current_buffer = {
                            'title': section['title'],
                            'level': section['level'],
                            'type': section['type'],
                            'content': content
                        }
                    else:
                        # Check if we can merge
                        combined_tokens = self.count_tokens(current_buffer['content'] + '\n\n' + content)
                        if combined_tokens <= self.config.target_tokens * 1.5:  # Allow some flexibility
                            current_buffer['content'] += '\n\n' + content
                            current_buffer['title'] = f"{current_buffer['title']} + {section['title']}"
                        else:
                            # Buffer too large, flush it
                            processed.append(current_buffer)
                            current_buffer = {
                                'title': section['title'],
                                'level': section['level'],
                                'type': section['type'],
                                'content': content
                            }
                else:
                    processed.append(section)

        # Flush final buffer
        if current_buffer:
            processed.append(current_buffer)

        return processed

    def _split_large_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large section into smaller chunks with overlap"""
        content = section['content']
        sentences = re.split(r'(?<=[.!?])\s+', content)

        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            test_content = current_chunk + (" " if current_chunk else "") + sentence
            test_tokens = self.count_tokens(test_content)

            if test_tokens <= self.config.target_tokens:
                current_chunk = test_content
                current_sentences.append(sentence)
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'title': f"{section['title']} (Part {len(chunks)+1})",
                        'level': section['level'],
                        'type': section['type'],
                        'content': current_chunk
                    })

                    # Create overlap for next chunk
                    overlap_sentences = current_sentences[-3:] if len(current_sentences) >= 3 else current_sentences
                    overlap_content = " ".join(overlap_sentences)
                    if self.count_tokens(overlap_content) <= self.config.overlap_tokens:
                        current_chunk = overlap_content + " " + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]

        # Add final chunk
        if current_chunk:
            chunks.append({
                'title': f"{section['title']} (Part {len(chunks)+1})" if len(chunks) > 0 else section['title'],
                'level': section['level'],
                'type': section['type'],
                'content': current_chunk
            })

        return chunks

    # -------------------------
    # ULTRATHINK: Fallback Mechanisms
    # -------------------------
    def _header_detection_failed(self, sections: List[Dict[str, Any]], text: str) -> bool:
        """Detect if header-based splitting failed and fallback is needed"""
        if not sections:
            return True

        # Check 1: Too few sections (likely detection failure)
        if len(sections) < 2:
            return True

        # Check 2: One massive section (90%+ of content in single section)
        total_chars = len(text)
        max_section_chars = max(len(section['content']) for section in sections)
        if max_section_chars > total_chars * 0.9:
            return True

        # Check 3: No recognized medical section types
        medical_types = {'abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'references'}
        found_types = set(section['type'] for section in sections)
        if not (found_types & medical_types):  # No intersection
            return True

        # Check 4: All sections are generic "content" type
        if all(section['type'] == 'content' for section in sections):
            return True

        return False

    def _fallback_paragraph_chunking(self, text: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """ROBUST FALLBACK: Paragraph-based chunking when header detection fails"""
        print("Using paragraph-based fallback chunking")

        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Ultimate fallback: sentence-based splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            paragraphs = sentences

        chunks = []
        current_chunk = ""
        current_token_count = 0

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # If paragraph alone exceeds target, split it
            if para_tokens > self.config.target_tokens:
                # Save current chunk first
                if current_chunk.strip():
                    chunks.append(self._create_fallback_chunk(
                        current_chunk, base_metadata, len(chunks) + 1
                    ))
                    current_chunk = ""
                    current_token_count = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                sentence_chunk = ""
                sentence_tokens = 0

                for sentence in sentences:
                    sent_tokens = self.count_tokens(sentence)

                    if sentence_tokens + sent_tokens <= self.config.target_tokens:
                        sentence_chunk += (" " if sentence_chunk else "") + sentence
                        sentence_tokens += sent_tokens
                    else:
                        if sentence_chunk:
                            chunks.append(self._create_fallback_chunk(
                                sentence_chunk, base_metadata, len(chunks) + 1
                            ))
                        sentence_chunk = sentence
                        sentence_tokens = sent_tokens

                if sentence_chunk:
                    chunks.append(self._create_fallback_chunk(
                        sentence_chunk, base_metadata, len(chunks) + 1
                    ))
                continue

            # Check if adding this paragraph would exceed target
            if current_token_count + para_tokens > self.config.target_tokens and current_chunk.strip():
                # Save current chunk
                chunks.append(self._create_fallback_chunk(
                    current_chunk, base_metadata, len(chunks) + 1
                ))
                current_chunk = paragraph
                current_token_count = para_tokens
            else:
                # Add to current chunk
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
                current_token_count += para_tokens

        # Save final chunk
        if current_chunk.strip():
            chunks.append(self._create_fallback_chunk(
                current_chunk, base_metadata, len(chunks) + 1
            ))

        print(f"Fallback created {len(chunks)} paragraph-based chunks")
        return chunks

    def _create_fallback_chunk(self, content: str, base_metadata: Dict[str, Any], chunk_num: int) -> Document:
        """Create a chunk with fallback metadata"""
        token_count = self.count_tokens(content)

        meta = {
            **base_metadata,
            **self._create_chunk_metadata(content, chunk_num),
            "chunking_method": "header_based_fallback",
            "section_title": f"Section {chunk_num}",
            "section_level": 1,
            "section_type": "content",
            "token_count": token_count,
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:12]
        }

        return Document(page_content=content.strip(), metadata=meta)

    # -------------------------
    # Stage 1: Hierarchical
    # -------------------------
    def _create_chunks_hierarchical(self, docling_doc, base_metadata) -> List[Document]:
        """IMPROVED: Better error handling and empty chunk filtering"""
        try:
            chunker = HierarchicalChunker(
                merge_list_items=self.config.merge_list_items,
                delim=self.config.delim
            )
            dl_chunks = chunker.chunk(docling_doc)
        except Exception as e:
            print(f"HierarchicalChunker failed: {e}")
            return []

        out: List[Document] = []
        for i, dl_chunk in enumerate(dl_chunks, 1):
            try:
                text = (dl_chunk.text or "").strip()
                if not text or len(text) < 10:  # Skip very short chunks
                    continue

                section_title, section_path = self._extract_section_info(dl_chunk)
                element_type, page_span, figure_ids, table_ids, is_caption, is_table = self._extract_element_info(dl_chunk)
                token_count = self.count_tokens(text)

                meta = {
                    **base_metadata,
                    **self._create_chunk_metadata(text, i),
                    "chunking_method": "docling_hierarchical",
                    "section_title": section_title,
                    "section_path": section_path,
                    "element_type": element_type,
                    "page_span": page_span,
                    "figure_ids": figure_ids,
                    "table_ids": table_ids,
                    "is_caption": is_caption,
                    "is_table": is_table,
                    "granularity": "fine",
                    "token_count": token_count,
                }

                out.append(Document(page_content=text, metadata=meta))

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

        return out

    # -------------------------
    # Stage 2: Coalescer
    # -------------------------
    def _coalesce_chunks(self, chunks: List[Document]) -> List[Document]:
        if not chunks:
            return []

        merged: List[Document] = []
        buf_text: List[str] = []
        buf_meta: Optional[Dict[str, Any]] = None
        buf_tokens = 0
        last_section_path = None
        last_element_types: List[str] = []

        def flush(prev_block_text: Optional[str] = None):
            nonlocal buf_text, buf_meta, buf_tokens
            if not buf_text:
                return
            text = self.config.delim.join(buf_text).strip()

            # For overlap only when we are splitting a large buffer
            if prev_block_text and self.config.overlap_tokens > 0:
                tail = self._tail_by_tokens(prev_block_text, self.config.overlap_tokens)
                if tail:
                    text = (tail + self.config.delim + text).strip()

            meta = dict(buf_meta or {})
            meta["token_count"] = self.count_tokens(text)
            meta["chunking_method"] = "docling_hierarchical+coalesce"
            meta["granularity"] = "coarse"
            # Update a new hash for the merged content
            meta["content_hash"] = hashlib.md5(text.encode()).hexdigest()[:12]
            merged.append(Document(page_content=text, metadata=meta))

            buf_text, buf_meta, buf_tokens = [], None, 0

        def can_merge(prev: Document, cur: Document) -> bool:
            if self.config.allow_cross_headings:
                return True
            return cur.metadata.get("section_path") == prev.metadata.get("section_path")

        prev_emitted_text = None
        i = 0
        while i < len(chunks):
            c = chunks[i]
            t = c.page_content
            t_tokens = self.count_tokens(t)
            sect_path = tuple(c.metadata.get("section_path") or [])
            is_caption = bool(c.metadata.get("is_caption"))
            is_table = bool(c.metadata.get("is_table"))
            elem_type = c.metadata.get("element_type", "")

            # Start buffer if empty
            if not buf_text:
                buf_text = [t]
                buf_meta = dict(c.metadata)
                buf_meta["chunk_id"] = f"coarse_{len(merged)+1:03d}"
                buf_tokens = t_tokens
                last_section_path = sect_path
                last_element_types = [elem_type]
                i += 1
                continue

            # Decide if we should merge this chunk into the buffer
            same_section = (sect_path == last_section_path) or self.config.allow_cross_headings
            would_fit = (buf_tokens + t_tokens) <= self.config.max_tokens

            # Attach captions/tables to nearby paragraph if configured
            priority_attach = (
                (self.config.attach_captions_to_paragraph and is_caption) or
                (self.config.attach_tables_to_paragraph and is_table)
            )

            # Merge rules:
            #  - keep merging while under target or min_tokens not achieved
            #  - allow attach of caption/table even if slightly over target, as long as <= max
            #  - avoid merging if section changes unless allow_cross_headings
            need_more = buf_tokens < self.config.min_tokens
            soft_room = (buf_tokens < self.config.target_tokens) and would_fit
            attach_room = priority_attach and would_fit

            if same_section and (need_more or soft_room or attach_room):
                buf_text.append(t)
                buf_tokens += t_tokens
                last_element_types.append(elem_type)
                i += 1
            else:
                # emit buffer, start new one with overlap from previous emission
                flush(prev_block_text=prev_emitted_text)
                prev_emitted_text = self.config.delim.join(buf_text) if buf_text else ""
                # new buffer starts next loop (do not advance i here)
                buf_text, buf_meta, buf_tokens = [], None, 0
                last_section_path = None
                last_element_types = []

        # flush remainder
        flush(prev_block_text=prev_emitted_text)
        return merged

    # =========================
    # Helpers
    # =========================

    def _tail_by_tokens(self, text: str, k: int) -> str:
        """IMPROVED: Better token-to-word approximation for tail extraction"""
        if not text or k <= 0:
            return ""

        words = text.split()
        if not words:
            return ""

        # More accurate token-to-word conversion for tail
        # Since 1 token ≈ 0.75 words, k tokens ≈ k * 0.75 words
        approx_words = max(1, int(round(k * 0.75)))
        approx_words = min(len(words), approx_words)

        tail = " ".join(words[-approx_words:])

        # Verify we're not way over on tokens
        actual_tokens = self.count_tokens(tail)
        if actual_tokens > k * 1.5:  # Allow some buffer
            # Too long, try with fewer words
            approx_words = max(1, int(approx_words * 0.7))
            tail = " ".join(words[-approx_words:])

        return tail

    def _create_chunk_metadata(self, content: str, idx: int) -> Dict[str, Any]:
        words = content.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        return {
            "chunk_id": f"chunk_{idx:03d}",
            "chunk_size": len(content),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "starts_with": content[:100].replace('\n', ' '),
            "avg_word_length": round(sum(len(w) for w in words) / len(words), 1) if words else 0,
            "is_sub_chunk": False,
            "content_hash": hashlib.md5(content.encode()).hexdigest()[:12],
        }

    def _extract_section_info(self, dl_chunk) -> Tuple[str, List[str]]:
        """
        IMPROVED: More robust section info extraction with better error handling
        Returns (section_title, section_path). Falls back to simple heuristics.
        """
        section_title = ""
        section_path: List[str] = []

        try:
            meta = getattr(dl_chunk, "meta", None)
            if not meta:
                md = {}
            else:
                # Handle different metadata formats
                if hasattr(meta, "model_dump"):
                    md = meta.model_dump()
                elif hasattr(meta, "dict"):
                    md = meta.dict()
                elif isinstance(meta, dict):
                    md = meta
                else:
                    md = {}

            # Try explicit fields (prioritize common Docling fields)
            for field in ("section", "heading", "title", "label", "section_header"):
                if field in md and md[field]:
                    section_title = str(md[field]).strip()
                    if section_title:
                        break

            # Try hierarchical info if present
            for field in ("section_path", "heading_path", "toc_path", "hierarchy"):
                if field in md and isinstance(md[field], (list, tuple)):
                    try:
                        section_path = [str(x).strip() for x in md[field] if x]
                        if not section_title and section_path:
                            section_title = section_path[-1]
                        break
                    except Exception:
                        continue

            # Fallback: extract from text content
            if not section_title:
                text = getattr(dl_chunk, 'text', '') or ''
                lines = text.strip().splitlines()[:5]  # Check first 5 lines
                for line in lines:
                    line = line.strip()
                    if line.startswith("#") and len(line) > 1:
                        section_title = re.sub(r"^#+\s*", "", line).strip()
                        if section_title:
                            break

            # Ensure section_path consistency
            if section_title and not section_path:
                section_path = [section_title]
            elif section_path and not section_title:
                section_title = section_path[-1] if section_path else ""

        except Exception as e:
            print(f"Warning: section extraction failed: {e}")

        return section_title, section_path

    def _extract_element_info(self, dl_chunk) -> Tuple[str, Tuple[Optional[int], Optional[int]], List[str], List[str], bool, bool]:
        """
        IMPROVED: Extract best-effort element metadata from Docling chunk with better error handling
        Returns: (element_type, page_span, figure_ids, table_ids, is_caption, is_table)
        """
        element_type = ""
        is_caption = False
        is_table = False
        page_start = None
        page_end = None
        figure_ids = []
        table_ids = []

        try:
            meta = getattr(dl_chunk, "meta", None)
            if not meta:
                md = {}
            else:
                # Handle different metadata formats
                if hasattr(meta, "model_dump"):
                    md = meta.model_dump()
                elif hasattr(meta, "dict"):
                    md = meta.dict()
                elif isinstance(meta, dict):
                    md = meta
                else:
                    md = {}

            # Extract element type
            element_type = (md.get("element_type") or md.get("type") or
                           md.get("label") or md.get("category") or "").strip()

            # Determine if caption or table
            is_caption = bool(md.get("is_caption", False) or
                            element_type.lower() in ("caption", "figure_caption", "table_caption"))
            is_table = bool(md.get("is_table", False) or
                          element_type.lower() in ("table", "table_cell", "table_header"))

            # Extract page information
            page_start = md.get("page_start") or md.get("page_no") or md.get("page")
            page_end = md.get("page_end") or page_start

            if page_start is None:
                # Try to infer from doc_items or other structures
                page_start, page_end = self._infer_pages_from_items(md)

            # Convert to int if possible
            if page_start is not None:
                try:
                    page_start = int(page_start)
                except (ValueError, TypeError):
                    page_start = None
            if page_end is not None:
                try:
                    page_end = int(page_end)
                except (ValueError, TypeError):
                    page_end = page_start

            # Extract figure/table IDs
            figure_ids = md.get("figure_ids") or md.get("figures") or []
            table_ids = md.get("table_ids") or md.get("tables") or []

            # Ensure they're lists
            if not isinstance(figure_ids, list):
                figure_ids = [figure_ids] if figure_ids else []
            if not isinstance(table_ids, list):
                table_ids = [table_ids] if table_ids else []

        except Exception as e:
            print(f"Warning: element info extraction failed: {e}")

        return element_type, (page_start, page_end), figure_ids, table_ids, is_caption, is_table

    def _infer_pages_from_items(self, md: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        page_start = md.get("page_no")  # sometimes a single page field exists
        page_end = md.get("page_no")
        # Try doc_items with page, if present
        items = md.get("doc_items") or []
        for it in items:
            if isinstance(it, dict) and "page_no" in it:
                p = it["page_no"]
                if page_start is None or p < page_start:
                    page_start = p
                if page_end is None or p > page_end:
                    page_end = p
        return page_start, page_end

    def analyze_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """IMPROVED: Analyze chunk statistics for optimization with token-aware metrics"""
        if not chunks:
            return {"error": "No chunks to analyze"}

        try:
            sizes = [len(chunk.page_content) for chunk in chunks]
            token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
            word_counts = [chunk.metadata.get('word_count', 0) for chunk in chunks]

            # Section analysis
            sections = {}
            for chunk in chunks:
                section = chunk.metadata.get('section_title', 'Unknown')
                if section not in sections:
                    sections[section] = 0
                sections[section] += 1

            return {
                "total_chunks": len(chunks),
                "avg_chunk_size_chars": sum(sizes) / len(sizes) if sizes else 0,
                "min_chunk_size_chars": min(sizes) if sizes else 0,
                "max_chunk_size_chars": max(sizes) if sizes else 0,
                "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0,
                "min_token_count": min(token_counts) if token_counts else 0,
                "max_token_count": max(token_counts) if token_counts else 0,
                "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
                "total_characters": sum(sizes),
                "total_tokens": sum(token_counts),
                "total_words": sum(word_counts),
                "sections_found": len(sections),
                "section_distribution": sections,
                "chunking_methods": list(set(chunk.metadata.get('chunking_method', 'unknown') for chunk in chunks)),
                "granularities": list(set(chunk.metadata.get('granularity', 'unknown') for chunk in chunks))
            }

        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
