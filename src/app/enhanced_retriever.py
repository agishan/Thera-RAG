"""
Enhanced Retriever with Re-ranking and Better Reference Management
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from collections import defaultdict


class EnhancedPineconeRetriever(BaseRetriever):
    """
    Advanced Pinecone retriever with:
    - Hybrid retrieval (more initial chunks, then re-rank)
    - Cross-encoder re-ranking
    - Source diversity optimization
    - Better reference management
    """

    def __init__(
        self,
        index,
        embedder,
        namespace,
        target_k=15,
        initial_k_multiplier=1.0,  # No extra retrieval for re-ranking
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranking=False,  # DISABLED BY DEFAULT
        source_diversity_weight=0.2
    ):
        super().__init__()
        self._index = index
        self._embedder = embedder
        self._namespace = namespace
        self._target_k = target_k
        self._initial_k = int(target_k * initial_k_multiplier)  # Retrieve more initially

        # Re-ranking setup
        self._enable_reranking = enable_reranking
        self._reranker = None
        if enable_reranking:
            try:
                self._reranker = CrossEncoder(rerank_model)
            except Exception as e:
                print(f"Warning: Could not load re-ranker {rerank_model}: {e}")
                self._enable_reranking = False

        self._source_diversity_weight = source_diversity_weight

    def update_k(self, new_k: int):
        """Update target retrieval count"""
        self._target_k = new_k
        self._initial_k = int(new_k * 2.5)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Enhanced retrieval with re-ranking and source diversity
        """
        # Step 1: Initial retrieval (get more than needed)
        initial_docs = self._initial_retrieval(query)

        if not initial_docs:
            return []

        # Step 2: Re-rank if enabled
        if self._enable_reranking and self._reranker and len(initial_docs) > self._target_k:
            reranked_docs = self._rerank_documents(query, initial_docs)
        else:
            reranked_docs = initial_docs

        # Step 3: Apply source diversity and final selection
        final_docs = self._apply_source_diversity(reranked_docs)

        # Step 4: Enhance metadata for better references
        enhanced_docs = self._enhance_metadata(final_docs, query)

        return enhanced_docs[:self._target_k]

    def _initial_retrieval(self, query: str) -> List[Document]:
        """Initial Pinecone retrieval"""
        formatted_query = f"query: {query}"
        embedding = self._embedder.encode(formatted_query, normalize_embeddings=True)

        results = self._index.query(
            vector=embedding.tolist(),
            top_k=self._initial_k,
            namespace=self._namespace,
            include_metadata=True
        )

        documents = []
        for match in results.matches:
            if match.metadata:
                content = match.metadata.get('text', '')
                metadata = {
                    'source': match.metadata.get('source', 'Unknown'),
                    'pinecone_score': match.score,
                    'vector_id': match.id,
                    **match.metadata
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        return documents

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents using cross-encoder"""
        if not self._reranker or len(documents) <= 1:
            return documents

        # Prepare query-document pairs for re-ranking
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            # Get cross-encoder scores
            rerank_scores = self._reranker.predict(pairs)

            # Combine with original scores (weighted)
            for i, doc in enumerate(documents):
                original_score = doc.metadata.get('pinecone_score', 0)
                rerank_score = float(rerank_scores[i])

                # Weighted combination (70% rerank, 30% original)
                combined_score = 0.7 * rerank_score + 0.3 * original_score

                doc.metadata['rerank_score'] = rerank_score
                doc.metadata['combined_score'] = combined_score
                doc.metadata['score'] = combined_score  # For UI display

            # Sort by combined score
            documents.sort(key=lambda x: x.metadata['combined_score'], reverse=True)

        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fall back to original ranking
            pass

        return documents

    def _apply_source_diversity(self, documents: List[Document]) -> List[Document]:
        """
        Apply source diversity to avoid over-representation from single sources
        """
        if not documents:
            return documents

        # Group by source
        source_groups = defaultdict(list)
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            source_groups[source].append(doc)

        # If we have diversity across sources, balance them
        if len(source_groups) > 1:
            balanced_docs = []
            max_per_source = max(2, self._target_k // len(source_groups))

            # Round-robin selection from different sources
            source_iterators = {source: iter(docs) for source, docs in source_groups.items()}
            remaining_slots = self._target_k

            while remaining_slots > 0 and source_iterators:
                for source in list(source_iterators.keys()):
                    if remaining_slots <= 0:
                        break
                    try:
                        doc = next(source_iterators[source])
                        balanced_docs.append(doc)
                        remaining_slots -= 1

                        # Remove source if we've taken max per source
                        source_count = sum(1 for d in balanced_docs if d.metadata.get('source') == source)
                        if source_count >= max_per_source:
                            del source_iterators[source]
                    except StopIteration:
                        del source_iterators[source]

            return balanced_docs

        return documents

    def _enhance_metadata(self, documents: List[Document], query: str) -> List[Document]:
        """Enhance metadata for better reference management"""
        for i, doc in enumerate(documents):
            # Add ranking information
            doc.metadata['final_rank'] = i + 1
            doc.metadata['query_used'] = query

            # Extract additional reference info if available
            self._extract_reference_info(doc)

            # Calculate relevance percentage for UI
            if 'score' in doc.metadata:
                # Normalize score to percentage (assuming scores are 0-1)
                relevance_pct = min(100, max(0, doc.metadata['score'] * 100))
                doc.metadata['relevance_percent'] = relevance_pct

            # Enhanced reference formatting for display
            self._format_display_reference(doc)

        return documents

    def _extract_reference_info(self, doc: Document):
        """Extract better reference information from metadata"""
        metadata = doc.metadata

        # Enhanced page/section information from new metadata structure
        location_parts = []

        # Page information
        page_start = metadata.get('page_start')
        page_end = metadata.get('page_end')
        if page_start and page_end:
            if page_start == page_end:
                location_parts.append(f"p. {page_start}")
            else:
                location_parts.append(f"pp. {page_start}-{page_end}")

        # Section hierarchy information
        hierarchy_breadcrumb = metadata.get('hierarchy_breadcrumb', '')
        if hierarchy_breadcrumb:
            location_parts.append(f"Section: {hierarchy_breadcrumb}")

        # Combine location info
        if location_parts:
            metadata['reference_location'] = " | ".join(location_parts)

        # Use enhanced document type if available, fallback to simple detection
        if 'doc_type' not in metadata:
            source = metadata.get('source', metadata.get('source_title', ''))
            if source.endswith('.pdf'):
                metadata['doc_type'] = 'PDF'
            elif source.endswith(('.txt', '.md')):
                metadata['doc_type'] = 'Text'
            elif 'url' in source.lower() or 'http' in source.lower():
                metadata['doc_type'] = 'Web'
            else:
                metadata['doc_type'] = 'Document'

        # Add chunk size info for debugging
        metadata['chunk_length'] = len(doc.page_content)

    def _format_display_reference(self, doc: Document):
        """Format references for better UI display"""
        metadata = doc.metadata

        # Create a formatted citation if available
        if 'citation_text' in metadata:
            metadata['formatted_citation'] = metadata['citation_text']
        else:
            # Build citation from available parts
            citation_parts = []

            # Author and year
            authors = metadata.get('source_authors')
            year = metadata.get('source_year')
            if authors and year:
                citation_parts.append(f"{authors} ({year})")
            elif authors:
                citation_parts.append(authors)
            elif year:
                citation_parts.append(f"({year})")

            # Title
            title = metadata.get('source_title')
            if title:
                citation_parts.append(f'"{title}"')

            # Journal
            journal = metadata.get('source_journal')
            if journal:
                citation_parts.append(f"*{journal}*")

            # Page/location info
            location = metadata.get('reference_location')
            if location:
                citation_parts.append(location)

            metadata['formatted_citation'] = ". ".join(citation_parts) if citation_parts else "Unknown source"

        # Create a clickable reference URL if available
        reference_url = metadata.get('reference_url')
        doi = metadata.get('source_doi')
        if reference_url:
            metadata['clickable_reference'] = reference_url
        elif doi:
            metadata['clickable_reference'] = f"https://doi.org/{doi}"

        # Create a short reference for inline citations
        authors = metadata.get('source_authors', '').split(',')[0] if metadata.get('source_authors') else ''
        year = metadata.get('source_year', '')
        chunk_index = metadata.get('chunk_index', 0)

        if authors and year:
            metadata['short_citation'] = f"({authors.strip()}, {year}, chunk {chunk_index + 1})"
        elif metadata.get('source_title'):
            title_short = metadata['source_title'][:30] + "..." if len(metadata['source_title']) > 30 else metadata['source_title']
            metadata['short_citation'] = f"({title_short}, chunk {chunk_index + 1})"
        else:
            metadata['short_citation'] = f"(Chunk {chunk_index + 1})"

    @property
    def k(self) -> int:
        """Get current target k value"""
        return self._target_k

    @property
    def namespace(self) -> str:
        """Get current namespace"""
        return self._namespace

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get current retrieval configuration stats"""
        return {
            "target_k": self._target_k,
            "initial_k": self._initial_k,
            "reranking_enabled": self._enable_reranking,
            "reranker_model": self._reranker.__class__.__name__ if self._reranker else None,
            "source_diversity_weight": self._source_diversity_weight,
            "namespace": self._namespace
        }