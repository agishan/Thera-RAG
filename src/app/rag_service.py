from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

class RAGService:
    """Simple RAG service for handling question-answering"""
    
    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the conversational retrieval chain"""
        # Initialize Pinecone
        pc = Pinecone(api_key=self.config['pinecone_api_key'])
        
        # Initialize Pinecone and embeddings
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.embedder = SentenceTransformer("intfloat/e5-base")
        
        # Create custom retriever
        class PineconeRetriever(BaseRetriever):
            def __init__(self, index, embedder, namespace, k):
                super().__init__()
                self._index = index
                self._embedder = embedder
                self._namespace = namespace
                self._k = k
            
            def get_relevant_documents(self, query: str) -> List[Document]:
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
        
        # Create retriever
        self.retriever = PineconeRetriever(
            index=self.index,
            embedder=self.embedder,
            namespace=self.config['pinecone_namespace'],
            k=self.config['retrieval_k']
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config['llm_model'],
            temperature=self.config['llm_temperature'],
            max_tokens=self.config['llm_max_tokens'],
            google_api_key=self.config['google_api_key']
        )
        
        # Create chain
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
        )
    
    def get_response(self, question, chat_history, retrieval_k=None):
        """Get response from the RAG chain with optional dynamic retrieval_k"""
        # Update retriever k value if provided
        if retrieval_k is not None and retrieval_k != self.config['retrieval_k']:
            # Create new retriever with updated k value
            class PineconeRetriever(BaseRetriever):
                def __init__(self, index, embedder, namespace, k):
                    super().__init__()
                    self._index = index
                    self._embedder = embedder
                    self._namespace = namespace
                    self._k = k
                
                def get_relevant_documents(self, query: str) -> List[Document]:
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
            
            self.retriever = PineconeRetriever(
                index=self.index,
                embedder=self.embedder,
                namespace=self.config['pinecone_namespace'],
                k=retrieval_k
            )
            # Recreate chain with updated retriever
            self.chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                verbose=False,
            )
        
        # Get response from chain
        response = self.chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        
        return response