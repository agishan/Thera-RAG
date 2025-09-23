"""
Enhanced Pinecone Retriever with dynamic configuration support
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class DynamicPineconeRetriever(BaseRetriever):
    """
    Pinecone retriever with dynamic k value updates to avoid code duplication
    """

    def __init__(self, index, embedder, namespace, initial_k=15):
        super().__init__()
        self._index = index
        self._embedder = embedder
        self._namespace = namespace
        self._k = initial_k

    def update_k(self, new_k: int):
        """Update retrieval count without recreating retriever"""
        self._k = new_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents from Pinecone

        Args:
            query: The search query

        Returns:
            List of relevant documents with metadata
        """
        # Format query for E5 model
        formatted_query = f"query: {query}"
        embedding = self._embedder.encode(formatted_query, normalize_embeddings=True)

        # Query Pinecone directly
        results = self._index.query(
            vector=embedding.tolist(),
            top_k=self._k,
            namespace=self._namespace,
            include_metadata=True
        )

        # Convert to LangChain documents
        documents = []
        for match in results.matches:
            if match.metadata:
                content = match.metadata.get('text', '')
                metadata = {
                    'source': match.metadata.get('source', 'Unknown'),
                    'score': match.score,
                    **match.metadata
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        return documents

    @property
    def k(self) -> int:
        """Get current k value"""
        return self._k

    @property
    def namespace(self) -> str:
        """Get current namespace"""
        return self._namespace