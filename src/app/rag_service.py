from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional

# Import new components
from .retrievers import DynamicPineconeRetriever
from .enhanced_retriever import EnhancedPineconeRetriever
from .prompts import MedicalPromptManager

class RAGService:
    """Enhanced RAG service with dynamic prompts and efficient retrieval"""

    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None

        # Initialize prompt manager
        self.prompt_manager = MedicalPromptManager()
        self.current_prompt_type = "medical_rag"

        # Get RAG prompt
        self.custom_prompt = self.prompt_manager.get_rag_prompt()

        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the conversational retrieval chain with enhanced components"""
        # Initialize Pinecone (single instance)
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.embedder = SentenceTransformer("intfloat/e5-base")

        # Create enhanced retriever with better reference management (re-ranking disabled)
        self.retriever = EnhancedPineconeRetriever(
            index=self.index,
            embedder=self.embedder,
            namespace=self.config['pinecone_namespace'],
            target_k=self.config['retrieval_k'],
            enable_reranking=False,  # Disabled for performance
            source_diversity_weight=0.2
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config['llm_model'],
            temperature=self.config['llm_temperature'],
            max_tokens=self.config['llm_max_tokens'],
            google_api_key=self.config['google_api_key']
        )

        # Create chain with custom prompt
        self._create_chain()

    def _create_chain(self, prompt_template=None):
        """Create or recreate the chain with specified prompt"""
        if prompt_template is None:
            prompt_template = self.custom_prompt

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )
    
    def get_response(self, question, chat_history, retrieval_k=None, prompt_type=None):
        """
        Get response from the RAG chain with dynamic configuration

        Args:
            question: The user's question
            chat_history: List of (question, answer) tuples for conversation context
            retrieval_k: Number of documents to retrieve (optional)
            prompt_type: Type of prompt to use (optional)

        Returns:
            Response dictionary with answer and source documents
        """
        # Update retriever k value if provided (much more efficient now)
        if retrieval_k is not None and retrieval_k != self.retriever.k:
            self.retriever.update_k(retrieval_k)

        # Update prompt if different type requested
        if prompt_type is not None and prompt_type != self.current_prompt_type:
            self.set_prompt_type(prompt_type)

        # Get response from chain
        response = self.chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        return response

    def set_prompt_type(self, prompt_type: str):
        """
        Change the prompt type dynamically

        Args:
            prompt_type: Currently only 'medical_rag' is supported
        """
        try:
            if prompt_type in ["clinical_qa", "medical_rag"]:
                new_prompt = self.prompt_manager.get_rag_prompt()
            else:
                # Try to get it directly from prompt manager
                new_prompt = self.prompt_manager.get_prompt(prompt_type)

            self.custom_prompt = new_prompt
            self.current_prompt_type = prompt_type
            self._create_chain(new_prompt)

        except Exception as e:
            print(f"Warning: Could not set prompt type '{prompt_type}': {e}")
            print(f"Available prompts: {self.prompt_manager.list_available_templates()}")

    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types"""
        return self.prompt_manager.list_available_templates()

    def get_current_config(self) -> dict:
        """Get current configuration for debugging"""
        config = {
            "retrieval_k": self.retriever.k,
            "prompt_type": self.current_prompt_type,
            "namespace": self.retriever.namespace,
            "available_prompts": self.get_available_prompt_types()
        }

        # Add enhanced retriever stats if available
        if hasattr(self.retriever, 'get_retrieval_stats'):
            config.update(self.retriever.get_retrieval_stats())

        return config