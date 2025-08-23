from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore

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
        pc = PineconeClient(api_key=self.config['pinecone_api_key'])
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config['embedding_model'],
            google_api_key=self.config['google_api_key']
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.config['pinecone_index_name'],
            embedding=embeddings,
            namespace=self.config['pinecone_namespace'],
        )
        
        # Create retriever with default k value
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config['retrieval_k']}
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
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": retrieval_k}
            )
            # Recreate chain with updated retriever
            self.chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                verbose=False,
            )
        
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })