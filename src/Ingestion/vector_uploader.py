"""
Enhanced vector uploader with full metadata preservation
"""

import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
from langchain_core.documents import Document


class EnhancedVectorUploader:
    """Upload document chunks with comprehensive metadata to Pinecone"""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str,
        embedding_model: str = "intfloat/e5-base"
    ):
        """
        Initialize the uploader

        Args:
            api_key: Pinecone API key
            index_name: Name of Pinecone index
            namespace: Namespace for vectors
            embedding_model: SentenceTransformer model for embeddings
        """
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.embedder = SentenceTransformer(embedding_model)

    def upload_chunks(
        self,
        chunks: List[Document],
        batch_size: int = 50,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Upload chunks with full metadata preservation

        Args:
            chunks: List of Document chunks to upload
            batch_size: Number of vectors per batch
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with upload statistics
        """
        if not chunks:
            return {"uploaded": 0, "skipped": 0, "errors": 0}

        print(f"🧠 Embedding and uploading {len(chunks)} chunks...")

        vectors = []
        stats = {"uploaded": 0, "skipped": 0, "errors": 0}

        iterator = tqdm(chunks, desc="Processing chunks") if show_progress else chunks

        for i, chunk in enumerate(iterator):
            try:
                # Skip empty chunks
                if not chunk.page_content.strip():
                    stats["skipped"] += 1
                    continue

                # Generate embedding
                input_text = f"passage: {chunk.page_content.strip()}"
                embedding = self.embedder.encode(input_text, normalize_embeddings=True)

                # Prepare metadata for Pinecone (NO TRUNCATION)
                pinecone_metadata = self._prepare_metadata_for_pinecone(chunk.metadata)

                # Create vector
                vector_id = chunk.metadata.get('chunk_id', f"chunk_{i:04d}")
                vectors.append({
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": pinecone_metadata
                })

                # Upload in batches
                if len(vectors) >= batch_size:
                    self._upload_batch(vectors)
                    stats["uploaded"] += len(vectors)
                    vectors = []

            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
                stats["errors"] += 1
                continue

        # Upload final batch
        if vectors:
            self._upload_batch(vectors)
            stats["uploaded"] += len(vectors)

        return stats

    def _prepare_metadata_for_pinecone(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for Pinecone with full content preservation

        NO 500 CHARACTER LIMIT - stores full text content
        """
        pinecone_metadata = {}

        # Essential fields that should always be preserved
        essential_fields = [
            'chunk_id', 'content_hash', 'chunk_index',
            'source_title', 'source_authors', 'source_year', 'source_journal', 'source_doi',
            'page_start', 'page_end', 'hierarchy_breadcrumb',
            'chunk_size', 'word_count', 'sentence_count',
            'citation_text', 'reference_url', 'document_type'
        ]

        # Store FULL text content (removed 500 char limit)
        text_content = metadata.get('text', '')
        pinecone_metadata['text'] = text_content  # FULL CONTENT PRESERVED

        # Add essential metadata with type conversion
        for field in essential_fields:
            if field in metadata and metadata[field] is not None:
                value = metadata[field]
                if isinstance(value, (str, int, float, bool)):
                    pinecone_metadata[field] = value
                else:
                    pinecone_metadata[field] = str(value)

        # Handle complex fields
        self._add_complex_fields(metadata, pinecone_metadata)

        # Add derived/computed fields for search
        self._add_search_fields(metadata, pinecone_metadata)

        return pinecone_metadata

    def _add_complex_fields(self, metadata: Dict, pinecone_metadata: Dict):
        """Add complex fields with appropriate formatting"""

        # Section hierarchy as searchable string
        if 'section_hierarchy' in metadata:
            hierarchy = metadata['section_hierarchy']
            if isinstance(hierarchy, list) and hierarchy:
                pinecone_metadata['sections'] = " | ".join(str(h) for h in hierarchy)
            elif hierarchy:
                pinecone_metadata['sections'] = str(hierarchy)

        # Citations information
        if 'citations' in metadata:
            citations = metadata['citations']
            if isinstance(citations, list) and citations:
                # Store citation count
                pinecone_metadata['citation_count'] = len(citations)

                # Store first few citation authors for search
                authors = []
                for citation in citations[:3]:  # First 3 citations
                    if isinstance(citation, dict) and citation.get('authors'):
                        if isinstance(citation['authors'], list):
                            authors.extend(citation['authors'][:2])  # First 2 authors per citation

                if authors:
                    pinecone_metadata['cited_authors'] = " | ".join(authors)

                # Store citation titles for search
                titles = []
                for citation in citations[:3]:
                    if isinstance(citation, dict) and citation.get('title'):
                        titles.append(str(citation['title'])[:100])  # Truncate long titles

                if titles:
                    pinecone_metadata['cited_titles'] = " | ".join(titles)

        # Processing metadata
        processing_fields = ['processed_at', 'processing_version', 'chunking_method']
        for field in processing_fields:
            if field in metadata:
                pinecone_metadata[field] = str(metadata[field])

    def _add_search_fields(self, metadata: Dict, pinecone_metadata: Dict):
        """Add derived fields for better search capabilities"""

        # Document classification
        pinecone_metadata['doc_type'] = self._classify_document_type(metadata)

        # Time-based tags
        year = metadata.get('source_year')
        if year:
            try:
                year_int = int(str(year))
                # Add decade classification
                decade = (year_int // 10) * 10
                pinecone_metadata['decade'] = f"{decade}s"

                # Add recency classification
                if year_int >= 2020:
                    pinecone_metadata['recency'] = 'recent'
                elif year_int >= 2010:
                    pinecone_metadata['recency'] = 'modern'
                else:
                    pinecone_metadata['recency'] = 'older'
            except (ValueError, TypeError):
                pass

        # Content characteristics
        chunk_size = metadata.get('chunk_size', 0)
        if chunk_size:
            if chunk_size < 1000:
                pinecone_metadata['chunk_length_category'] = 'short'
            elif chunk_size < 3000:
                pinecone_metadata['chunk_length_category'] = 'medium'
            else:
                pinecone_metadata['chunk_length_category'] = 'long'

        # Generate search tags
        search_tags = self._generate_search_tags(metadata)
        if search_tags:
            pinecone_metadata['search_tags'] = search_tags

    def _classify_document_type(self, metadata: Dict[str, Any]) -> str:
        """Enhanced document type classification"""
        doc_type = metadata.get('document_type', 'unknown')

        # Use existing classification if available
        if doc_type != 'unknown':
            return doc_type

        # Fallback classification based on other metadata
        source_title = metadata.get('source_title', '').lower()
        source_journal = metadata.get('source_journal', '').lower()

        if source_journal:
            return 'journal_article'
        elif any(term in source_title for term in ['guideline', 'protocol', 'standard']):
            return 'clinical_guide'
        elif any(term in source_title for term in ['review', 'systematic', 'meta-analysis']):
            return 'review_paper'
        elif any(term in source_title for term in ['manual', 'handbook', 'guide']):
            return 'manual'
        else:
            return 'document'

    def _generate_search_tags(self, metadata: Dict[str, Any]) -> str:
        """Generate searchable tags from metadata"""
        tags = []

        # Add year tag
        year = metadata.get('source_year')
        if year:
            tags.append(f"year_{year}")

        # Add author surnames (simplified extraction)
        authors = metadata.get('source_authors', '')
        if authors:
            # Simple extraction of potential surnames
            author_parts = str(authors).replace(',', ' ').split()
            potential_surnames = [
                part.strip('.,') for part in author_parts
                if len(part) > 2 and part[0].isupper() and part.isalpha()
            ]
            tags.extend(potential_surnames[:3])  # First 3 surnames

        # Add section-based tags
        hierarchy = metadata.get('section_hierarchy', [])
        if isinstance(hierarchy, list):
            for section in hierarchy:
                section_lower = str(section).lower()
                # Extract key section types
                if any(term in section_lower for term in [
                    'method', 'result', 'discussion', 'conclusion',
                    'introduction', 'background', 'analysis'
                ]):
                    section_word = section_lower.split()[0] if section_lower.split() else ''
                    if len(section_word) > 3:
                        tags.append(section_word)

        # Add document type tag
        doc_type = metadata.get('document_type', '')
        if doc_type:
            tags.append(doc_type)

        return " ".join(tags[:10])  # Limit to 10 tags

    def _upload_batch(self, vectors: List[Dict]) -> None:
        """Upload a batch of vectors to Pinecone"""
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
        except Exception as e:
            print(f"❌ Error uploading batch: {e}")
            raise

    def verify_upload(
        self,
        sample_query: str = "medical treatment",
        expected_chunks: int = 0
    ) -> Dict[str, Any]:
        """
        Verify upload success and metadata quality

        Args:
            sample_query: Query to test with
            expected_chunks: Expected number of chunks uploaded

        Returns:
            Verification results
        """
        print(f"🧪 Verifying upload with query: '{sample_query}'")

        # Get index stats
        stats = self.index.describe_index_stats()
        namespace_count = stats.namespaces.get(self.namespace, {}).vector_count if stats.namespaces else 0

        # Test query
        query_embedding = self.embedder.encode(f"query: {sample_query}", normalize_embeddings=True)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            namespace=self.namespace,
            include_metadata=True
        )

        # Analyze results
        verification = {
            "namespace": self.namespace,
            "total_vectors": namespace_count,
            "expected_chunks": expected_chunks,
            "upload_success": namespace_count > 0,
            "query_test": {
                "query": sample_query,
                "results_found": len(results.matches),
                "test_passed": len(results.matches) > 0
            },
            "metadata_quality": self._assess_metadata_quality(results.matches),
            "sample_metadata": {}
        }

        if results.matches:
            sample_metadata = results.matches[0].metadata
            verification["sample_metadata"] = {
                "has_full_text": 'text' in sample_metadata and len(sample_metadata.get('text', '')) > 500,
                "has_citations": 'citation_text' in sample_metadata,
                "has_location": 'page_start' in sample_metadata,
                "metadata_fields": list(sample_metadata.keys())[:15]  # Show first 15 fields
            }

        # Print verification summary
        print(f"✅ Vectors in namespace '{self.namespace}': {namespace_count}")
        print(f"📝 Full text preservation: {verification['sample_metadata'].get('has_full_text', False)}")
        print(f"📄 Citation information: {verification['sample_metadata'].get('has_citations', False)}")
        print(f"📍 Location metadata: {verification['sample_metadata'].get('has_location', False)}")

        return verification

    def _assess_metadata_quality(self, matches: List) -> Dict[str, Any]:
        """Assess quality of uploaded metadata"""
        if not matches:
            return {"quality_score": 0.0, "issues": ["No results found"]}

        quality_checks = {
            "has_full_text": 0,
            "has_source_info": 0,
            "has_location": 0,
            "has_citations": 0,
            "has_search_tags": 0
        }

        total_matches = len(matches)

        for match in matches:
            metadata = match.metadata

            # Check for full text preservation
            if metadata.get('text') and len(metadata.get('text', '')) > 500:
                quality_checks["has_full_text"] += 1

            # Check for source information
            if metadata.get('source_title') and metadata.get('source_authors'):
                quality_checks["has_source_info"] += 1

            # Check for location information
            if metadata.get('page_start') or metadata.get('hierarchy_breadcrumb'):
                quality_checks["has_location"] += 1

            # Check for citation information
            if metadata.get('citation_text') or metadata.get('citation_count'):
                quality_checks["has_citations"] += 1

            # Check for search tags
            if metadata.get('search_tags') or metadata.get('doc_type'):
                quality_checks["has_search_tags"] += 1

        # Calculate quality percentages
        quality_percentages = {
            check: (count / total_matches) * 100
            for check, count in quality_checks.items()
        }

        # Overall quality score
        quality_score = sum(quality_percentages.values()) / len(quality_percentages)

        return {
            "quality_score": round(quality_score, 1),
            "details": quality_percentages,
            "total_samples": total_matches
        }