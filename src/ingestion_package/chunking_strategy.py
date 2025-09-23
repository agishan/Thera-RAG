"""
Advanced chunking strategies for medical documents
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""
    target_chunk_size: int = 3000
    overlap_size: int = 300
    max_chunk_size: int = 5000
    min_chunk_size: int = 500
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    section_aware: bool = True


class SmartChunker:
    """
    Intelligent chunking system that preserves document structure and context
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker with configuration

        Args:
            config: ChunkingConfig instance, defaults to standard medical document settings
        """
        self.config = config or ChunkingConfig()

        # Validate overlap ratio
        overlap_ratio = self.config.overlap_size / self.config.target_chunk_size
        if overlap_ratio > 0.2:
            self.config.overlap_size = int(self.config.target_chunk_size * 0.2)
            print(f"⚠️  Capped overlap to 20% of chunk size: {self.config.overlap_size} chars")

    def create_chunks(self, text: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """
        Create intelligently structured chunks from text

        Args:
            text: Full document text
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of Document objects with enhanced metadata
        """
        if not text.strip():
            return []

        # Determine chunking strategy based on document structure
        if self.config.section_aware and self._has_clear_sections(text):
            chunks = self._section_based_chunking(text, base_metadata)
        else:
            chunks = self._recursive_chunking(text, base_metadata)

        # Post-process chunks
        return self._post_process_chunks(chunks)

    def _has_clear_sections(self, text: str) -> bool:
        """Determine if document has clear section structure"""
        # Count different types of headers
        h1_count = len(re.findall(r'^# ', text, re.MULTILINE))
        h2_count = len(re.findall(r'^## ', text, re.MULTILINE))
        h3_count = len(re.findall(r'^### ', text, re.MULTILINE))

        # If we have multiple headers, use section-based chunking
        return (h1_count + h2_count + h3_count) >= 3

    def _section_based_chunking(self, text: str, base_metadata: Dict) -> List[Document]:
        """Create chunks based on document sections"""
        chunks = []

        # Try different header levels
        sections = self._split_by_headers(text)
        if len(sections) <= 1:
            # Fall back to recursive chunking
            return self._recursive_chunking(text, base_metadata)

        current_chunk = ""
        previous_chunk_end = ""
        chunk_id = 1

        for section in sections:
            # If adding this section would exceed target size, save current chunk
            if (len(current_chunk) + len(section['content']) > self.config.target_chunk_size
                and current_chunk.strip()):

                # Create chunk with overlap
                chunk_content = previous_chunk_end + current_chunk
                chunks.append(self._create_chunk_document(
                    chunk_content, base_metadata, chunk_id, section_info=section.get('header')
                ))

                # Prepare overlap for next chunk
                previous_chunk_end = self._get_smart_overlap(current_chunk)
                current_chunk = section['content']
                chunk_id += 1
            else:
                current_chunk += ("\n" if current_chunk else "") + section['content']

        # Add final chunk
        if current_chunk.strip():
            chunk_content = previous_chunk_end + current_chunk
            chunks.append(self._create_chunk_document(
                chunk_content, base_metadata, chunk_id
            ))

        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, str]]:
        """Split text by markdown headers"""
        # Try H1 first, then H2, then H3
        for header_pattern in [r'^# (.+)$', r'^## (.+)$', r'^### (.+)$']:
            sections = []
            parts = re.split(f'({header_pattern})', text, flags=re.MULTILINE)

            if len(parts) > 1:
                current_header = None
                current_content = parts[0] if parts[0].strip() else ""

                i = 1
                while i < len(parts):
                    if re.match(header_pattern, parts[i], re.MULTILINE):
                        # Save previous section
                        if current_content.strip():
                            sections.append({
                                'header': current_header,
                                'content': current_content.strip()
                            })

                        # Start new section
                        current_header = parts[i]
                        current_content = parts[i + 1] if i + 1 < len(parts) else ""
                        i += 2
                    else:
                        current_content += parts[i]
                        i += 1

                # Add final section
                if current_content.strip():
                    sections.append({
                        'header': current_header,
                        'content': current_content.strip()
                    })

                if len(sections) > 1:
                    return sections

        # If no good sections found, return whole text
        return [{'header': None, 'content': text}]

    def _recursive_chunking(self, text: str, base_metadata: Dict) -> List[Document]:
        """Use recursive character text splitter as fallback"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=self.config.overlap_size,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True
        )

        doc = Document(page_content=text, metadata=base_metadata)
        split_chunks = splitter.split_documents([doc])

        # Enhance metadata for each chunk
        for i, chunk in enumerate(split_chunks, 1):
            chunk.metadata.update(self._create_chunk_metadata(chunk.page_content, i))
            chunk.metadata['chunking_method'] = 'recursive_with_overlap'

        return split_chunks

    def _get_smart_overlap(self, text: str) -> str:
        """Extract intelligent overlap that ends at natural boundaries"""
        if len(text) <= self.config.overlap_size:
            return text

        overlap_text = text[-self.config.overlap_size:]

        # Try to end at sentence boundary
        for separator in ['. ', '! ', '? ', '\n\n']:
            last_sep = overlap_text.rfind(separator)
            if last_sep > self.config.overlap_size * 0.5:
                return overlap_text[last_sep + len(separator):]

        # Try paragraph boundary
        last_para = overlap_text.rfind('\n')
        if last_para > self.config.overlap_size * 0.3:
            return overlap_text[last_para + 1:]

        # Fall back to character boundary
        return overlap_text

    def _create_chunk_document(
        self,
        content: str,
        base_metadata: Dict,
        chunk_id: int,
        section_info: Optional[str] = None
    ) -> Document:
        """Create a document chunk with enhanced metadata"""

        # Generate unique hash for content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]

        # Extract section title if available
        section_title = self._extract_section_title(content, section_info)

        # Create enhanced metadata
        enhanced_metadata = {
            **base_metadata,
            **self._create_chunk_metadata(content, chunk_id),
            "content_hash": content_hash,
            "section_title": section_title,
            "chunking_method": "section_based_with_overlap"
        }

        return Document(page_content=content.strip(), metadata=enhanced_metadata)

    def _create_chunk_metadata(self, content: str, chunk_id: int) -> Dict[str, Any]:
        """Create standard chunk metadata"""
        words = content.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]

        return {
            "chunk_id": f"chunk_{chunk_id:03d}",
            "chunk_size": len(content),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "starts_with": content[:100].replace('\n', ' '),
            "avg_word_length": round(sum(len(word) for word in words) / len(words), 1) if words else 0,
            "is_sub_chunk": False
        }

    def _extract_section_title(self, content: str, section_info: Optional[str]) -> str:
        """Extract section title from content or section info"""
        if section_info:
            return re.sub(r'^#+\s*', '', section_info).strip()

        # Look for title in first few lines
        lines = content.strip().split('\n')
        for line in lines[:3]:
            if line.startswith('#'):
                return re.sub(r'^#+\s*', '', line).strip()

        return ""

    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks to handle size constraints and quality"""
        processed_chunks = []

        for chunk in chunks:
            content_length = len(chunk.page_content)

            # Handle oversized chunks
            if content_length > self.config.max_chunk_size:
                sub_chunks = self._split_oversized_chunk(chunk)
                processed_chunks.extend(sub_chunks)

            # Handle undersized chunks (merge with next if possible)
            elif content_length < self.config.min_chunk_size:
                # For now, keep small chunks but mark them
                chunk.metadata['is_small_chunk'] = True
                processed_chunks.append(chunk)

            else:
                processed_chunks.append(chunk)

        return processed_chunks

    def _split_oversized_chunk(self, chunk: Document) -> List[Document]:
        """Split oversized chunks into smaller pieces"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=self.config.overlap_size,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        sub_chunks = splitter.split_documents([chunk])

        # Update metadata for sub-chunks
        parent_chunk_id = chunk.metadata.get('chunk_id', 'unknown')
        for j, sub_chunk in enumerate(sub_chunks):
            sub_chunk.metadata.update({
                "chunk_id": f"{parent_chunk_id}.{j+1}",
                "parent_chunk": parent_chunk_id,
                "is_sub_chunk": True,
                "chunking_method": "hybrid_section_recursive"
            })

        return sub_chunks

    def analyze_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze chunk statistics for optimization"""
        if not chunks:
            return {}

        sizes = [len(chunk.page_content) for chunk in chunks]
        word_counts = [chunk.metadata.get('word_count', 0) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "sections_with_titles": sum(1 for chunk in chunks if chunk.metadata.get('section_title')),
            "total_characters": sum(sizes),
            "total_words": sum(word_counts),
            "chunking_methods": list(set(chunk.metadata.get('chunking_method', 'unknown') for chunk in chunks))
        }